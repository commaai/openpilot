# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-DHAVE_ENDIAN_H', '-DHAVE_STRUCT_TIMESPEC', '-DHAVE_PTHREAD', '-I/tmp/mesa-mesa-25.2.4/src', '-I/tmp/mesa-mesa-25.2.4/include', '-I/tmp/mesa-mesa-25.2.4/gen', '-I/tmp/mesa-mesa-25.2.4/src/compiler/nir', '-I/tmp/mesa-mesa-25.2.4/src/gallium/auxiliary', '-I/tmp/mesa-mesa-25.2.4/src/gallium/include', '-I/usr/lib/llvm-20/include']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util, os, gzip, base64, subprocess, tinygrad.helpers as helpers
PATHS_TO_TRY = [
  (BASE:=os.getenv('MESA_PATH', f"/usr{'/local/' if helpers.OSX else '/'}lib"))+'/libtinymesa_cpu'+(EXT:='.dylib' if helpers.OSX else '.so'),
  f'{BASE}/libtinymesa{EXT}',
  '/opt/homebrew/lib/libtinymesa_cpu.dylib',
  '/opt/homebrew/lib/libtinymesa.dylib',
]
def _try_dlopen_tinymesa_cpu():
  library = ctypes.util.find_library("tinymesa_cpu")
  if library:
    try: return ctypes.CDLL(library)
    except OSError: pass
  for candidate in PATHS_TO_TRY:
    try: return ctypes.CDLL(candidate)
    except OSError: pass
  return None


class AsDictMixin:
    import sys
    if sys.version_info >= (3, 14): _layout_ = 'ms'
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



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

_libraries = {}
_libraries['libtinymesa_cpu.so'] = (dll := _try_dlopen_tinymesa_cpu())
class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  (dll := _try_dlopen_tinymesa_cpu())


class struct_blob(Structure):
    pass

struct_blob._pack_ = 1 # source:False
struct_blob._fields_ = [
    ('data', ctypes.POINTER(ctypes.c_ubyte)),
    ('allocated', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('fixed_allocation', ctypes.c_bool),
    ('out_of_memory', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 6),
]

class struct_blob_reader(Structure):
    pass

struct_blob_reader._pack_ = 1 # source:False
struct_blob_reader._fields_ = [
    ('data', ctypes.POINTER(ctypes.c_ubyte)),
    ('end', ctypes.POINTER(ctypes.c_ubyte)),
    ('current', ctypes.POINTER(ctypes.c_ubyte)),
    ('overrun', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

try:
    blob_init = _libraries['libtinymesa_cpu.so'].blob_init
    blob_init.restype = None
    blob_init.argtypes = [ctypes.POINTER(struct_blob)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    blob_init_fixed = _libraries['libtinymesa_cpu.so'].blob_init_fixed
    blob_init_fixed.restype = None
    blob_init_fixed.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    blob_finish = _libraries['FIXME_STUB'].blob_finish
    blob_finish.restype = None
    blob_finish.argtypes = [ctypes.POINTER(struct_blob)]
except AttributeError:
    pass
try:
    blob_finish_get_buffer = _libraries['libtinymesa_cpu.so'].blob_finish_get_buffer
    blob_finish_get_buffer.restype = None
    blob_finish_get_buffer.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    blob_align = _libraries['libtinymesa_cpu.so'].blob_align
    blob_align.restype = ctypes.c_bool
    blob_align.argtypes = [ctypes.POINTER(struct_blob), size_t]
except AttributeError:
    pass
try:
    blob_write_bytes = _libraries['libtinymesa_cpu.so'].blob_write_bytes
    blob_write_bytes.restype = ctypes.c_bool
    blob_write_bytes.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
intptr_t = ctypes.c_int64
try:
    blob_reserve_bytes = _libraries['libtinymesa_cpu.so'].blob_reserve_bytes
    blob_reserve_bytes.restype = intptr_t
    blob_reserve_bytes.argtypes = [ctypes.POINTER(struct_blob), size_t]
except AttributeError:
    pass
try:
    blob_reserve_uint32 = _libraries['libtinymesa_cpu.so'].blob_reserve_uint32
    blob_reserve_uint32.restype = intptr_t
    blob_reserve_uint32.argtypes = [ctypes.POINTER(struct_blob)]
except AttributeError:
    pass
try:
    blob_reserve_intptr = _libraries['libtinymesa_cpu.so'].blob_reserve_intptr
    blob_reserve_intptr.restype = intptr_t
    blob_reserve_intptr.argtypes = [ctypes.POINTER(struct_blob)]
except AttributeError:
    pass
try:
    blob_overwrite_bytes = _libraries['libtinymesa_cpu.so'].blob_overwrite_bytes
    blob_overwrite_bytes.restype = ctypes.c_bool
    blob_overwrite_bytes.argtypes = [ctypes.POINTER(struct_blob), size_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
uint8_t = ctypes.c_uint8
try:
    blob_write_uint8 = _libraries['libtinymesa_cpu.so'].blob_write_uint8
    blob_write_uint8.restype = ctypes.c_bool
    blob_write_uint8.argtypes = [ctypes.POINTER(struct_blob), uint8_t]
except AttributeError:
    pass
try:
    blob_overwrite_uint8 = _libraries['libtinymesa_cpu.so'].blob_overwrite_uint8
    blob_overwrite_uint8.restype = ctypes.c_bool
    blob_overwrite_uint8.argtypes = [ctypes.POINTER(struct_blob), size_t, uint8_t]
except AttributeError:
    pass
uint16_t = ctypes.c_uint16
try:
    blob_write_uint16 = _libraries['libtinymesa_cpu.so'].blob_write_uint16
    blob_write_uint16.restype = ctypes.c_bool
    blob_write_uint16.argtypes = [ctypes.POINTER(struct_blob), uint16_t]
except AttributeError:
    pass
uint32_t = ctypes.c_uint32
try:
    blob_write_uint32 = _libraries['libtinymesa_cpu.so'].blob_write_uint32
    blob_write_uint32.restype = ctypes.c_bool
    blob_write_uint32.argtypes = [ctypes.POINTER(struct_blob), uint32_t]
except AttributeError:
    pass
try:
    blob_overwrite_uint32 = _libraries['libtinymesa_cpu.so'].blob_overwrite_uint32
    blob_overwrite_uint32.restype = ctypes.c_bool
    blob_overwrite_uint32.argtypes = [ctypes.POINTER(struct_blob), size_t, uint32_t]
except AttributeError:
    pass
uint64_t = ctypes.c_uint64
try:
    blob_write_uint64 = _libraries['libtinymesa_cpu.so'].blob_write_uint64
    blob_write_uint64.restype = ctypes.c_bool
    blob_write_uint64.argtypes = [ctypes.POINTER(struct_blob), uint64_t]
except AttributeError:
    pass
try:
    blob_write_intptr = _libraries['libtinymesa_cpu.so'].blob_write_intptr
    blob_write_intptr.restype = ctypes.c_bool
    blob_write_intptr.argtypes = [ctypes.POINTER(struct_blob), intptr_t]
except AttributeError:
    pass
try:
    blob_overwrite_intptr = _libraries['libtinymesa_cpu.so'].blob_overwrite_intptr
    blob_overwrite_intptr.restype = ctypes.c_bool
    blob_overwrite_intptr.argtypes = [ctypes.POINTER(struct_blob), size_t, intptr_t]
except AttributeError:
    pass
try:
    blob_write_string = _libraries['libtinymesa_cpu.so'].blob_write_string
    blob_write_string.restype = ctypes.c_bool
    blob_write_string.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    blob_reader_init = _libraries['libtinymesa_cpu.so'].blob_reader_init
    blob_reader_init.restype = None
    blob_reader_init.argtypes = [ctypes.POINTER(struct_blob_reader), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    blob_reader_align = _libraries['libtinymesa_cpu.so'].blob_reader_align
    blob_reader_align.restype = None
    blob_reader_align.argtypes = [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError:
    pass
try:
    blob_read_bytes = _libraries['libtinymesa_cpu.so'].blob_read_bytes
    blob_read_bytes.restype = ctypes.POINTER(None)
    blob_read_bytes.argtypes = [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError:
    pass
try:
    blob_copy_bytes = _libraries['libtinymesa_cpu.so'].blob_copy_bytes
    blob_copy_bytes.restype = None
    blob_copy_bytes.argtypes = [ctypes.POINTER(struct_blob_reader), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    blob_skip_bytes = _libraries['libtinymesa_cpu.so'].blob_skip_bytes
    blob_skip_bytes.restype = None
    blob_skip_bytes.argtypes = [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError:
    pass
try:
    blob_read_uint8 = _libraries['libtinymesa_cpu.so'].blob_read_uint8
    blob_read_uint8.restype = uint8_t
    blob_read_uint8.argtypes = [ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass
try:
    blob_read_uint16 = _libraries['libtinymesa_cpu.so'].blob_read_uint16
    blob_read_uint16.restype = uint16_t
    blob_read_uint16.argtypes = [ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass
try:
    blob_read_uint32 = _libraries['libtinymesa_cpu.so'].blob_read_uint32
    blob_read_uint32.restype = uint32_t
    blob_read_uint32.argtypes = [ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass
try:
    blob_read_uint64 = _libraries['libtinymesa_cpu.so'].blob_read_uint64
    blob_read_uint64.restype = uint64_t
    blob_read_uint64.argtypes = [ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass
try:
    blob_read_intptr = _libraries['libtinymesa_cpu.so'].blob_read_intptr
    blob_read_intptr.restype = intptr_t
    blob_read_intptr.argtypes = [ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass
try:
    blob_read_string = _libraries['libtinymesa_cpu.so'].blob_read_string
    blob_read_string.restype = ctypes.POINTER(ctypes.c_char)
    blob_read_string.argtypes = [ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass
class struct_glsl_type(Structure):
    pass


# values for enumeration 'glsl_base_type'
glsl_base_type__enumvalues = {
    0: 'GLSL_TYPE_UINT',
    1: 'GLSL_TYPE_INT',
    2: 'GLSL_TYPE_FLOAT',
    3: 'GLSL_TYPE_FLOAT16',
    4: 'GLSL_TYPE_BFLOAT16',
    5: 'GLSL_TYPE_FLOAT_E4M3FN',
    6: 'GLSL_TYPE_FLOAT_E5M2',
    7: 'GLSL_TYPE_DOUBLE',
    8: 'GLSL_TYPE_UINT8',
    9: 'GLSL_TYPE_INT8',
    10: 'GLSL_TYPE_UINT16',
    11: 'GLSL_TYPE_INT16',
    12: 'GLSL_TYPE_UINT64',
    13: 'GLSL_TYPE_INT64',
    14: 'GLSL_TYPE_BOOL',
    15: 'GLSL_TYPE_COOPERATIVE_MATRIX',
    16: 'GLSL_TYPE_SAMPLER',
    17: 'GLSL_TYPE_TEXTURE',
    18: 'GLSL_TYPE_IMAGE',
    19: 'GLSL_TYPE_ATOMIC_UINT',
    20: 'GLSL_TYPE_STRUCT',
    21: 'GLSL_TYPE_INTERFACE',
    22: 'GLSL_TYPE_ARRAY',
    23: 'GLSL_TYPE_VOID',
    24: 'GLSL_TYPE_SUBROUTINE',
    25: 'GLSL_TYPE_ERROR',
}
GLSL_TYPE_UINT = 0
GLSL_TYPE_INT = 1
GLSL_TYPE_FLOAT = 2
GLSL_TYPE_FLOAT16 = 3
GLSL_TYPE_BFLOAT16 = 4
GLSL_TYPE_FLOAT_E4M3FN = 5
GLSL_TYPE_FLOAT_E5M2 = 6
GLSL_TYPE_DOUBLE = 7
GLSL_TYPE_UINT8 = 8
GLSL_TYPE_INT8 = 9
GLSL_TYPE_UINT16 = 10
GLSL_TYPE_INT16 = 11
GLSL_TYPE_UINT64 = 12
GLSL_TYPE_INT64 = 13
GLSL_TYPE_BOOL = 14
GLSL_TYPE_COOPERATIVE_MATRIX = 15
GLSL_TYPE_SAMPLER = 16
GLSL_TYPE_TEXTURE = 17
GLSL_TYPE_IMAGE = 18
GLSL_TYPE_ATOMIC_UINT = 19
GLSL_TYPE_STRUCT = 20
GLSL_TYPE_INTERFACE = 21
GLSL_TYPE_ARRAY = 22
GLSL_TYPE_VOID = 23
GLSL_TYPE_SUBROUTINE = 24
GLSL_TYPE_ERROR = 25
glsl_base_type = ctypes.c_uint32 # enum
class struct_glsl_cmat_description(Structure):
    pass

struct_glsl_cmat_description._pack_ = 1 # source:False
struct_glsl_cmat_description._fields_ = [
    ('element_type', ctypes.c_ubyte, 5),
    ('scope', ctypes.c_ubyte, 3),
    ('rows', ctypes.c_ubyte, 8),
    ('cols', ctypes.c_ubyte),
    ('use', ctypes.c_ubyte),
]

class union_glsl_type_fields(Union):
    pass

class struct_glsl_struct_field(Structure):
    pass

union_glsl_type_fields._pack_ = 1 # source:False
union_glsl_type_fields._fields_ = [
    ('array', ctypes.POINTER(struct_glsl_type)),
    ('structure', ctypes.POINTER(struct_glsl_struct_field)),
]

struct_glsl_type._pack_ = 1 # source:False
struct_glsl_type._fields_ = [
    ('gl_type', ctypes.c_uint32),
    ('base_type', glsl_base_type, 8),
    ('sampled_type', glsl_base_type, 8),
    ('sampler_dimensionality', glsl_base_type, 4),
    ('sampler_shadow', glsl_base_type, 1),
    ('sampler_array', glsl_base_type, 1),
    ('interface_packing', glsl_base_type, 2),
    ('interface_row_major', glsl_base_type, 1),
    ('PADDING_0', ctypes.c_uint8, 7),
    ('cmat_desc', struct_glsl_cmat_description),
    ('packed', ctypes.c_uint32, 1),
    ('has_builtin_name', ctypes.c_uint32, 1),
    ('PADDING_1', ctypes.c_uint8, 6),
    ('vector_elements', ctypes.c_uint32, 8),
    ('matrix_columns', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte),
    ('length', ctypes.c_uint32),
    ('PADDING_3', ctypes.c_ubyte * 4),
    ('name_id', ctypes.c_uint64),
    ('explicit_stride', ctypes.c_uint32),
    ('explicit_alignment', ctypes.c_uint32),
    ('fields', union_glsl_type_fields),
]

glsl_type = struct_glsl_type

# values for enumeration 'pipe_format'
pipe_format__enumvalues = {
    0: 'PIPE_FORMAT_NONE',
    1: 'PIPE_FORMAT_R64_UINT',
    2: 'PIPE_FORMAT_R64G64_UINT',
    3: 'PIPE_FORMAT_R64G64B64_UINT',
    4: 'PIPE_FORMAT_R64G64B64A64_UINT',
    5: 'PIPE_FORMAT_R64_SINT',
    6: 'PIPE_FORMAT_R64G64_SINT',
    7: 'PIPE_FORMAT_R64G64B64_SINT',
    8: 'PIPE_FORMAT_R64G64B64A64_SINT',
    9: 'PIPE_FORMAT_R64_FLOAT',
    10: 'PIPE_FORMAT_R64G64_FLOAT',
    11: 'PIPE_FORMAT_R64G64B64_FLOAT',
    12: 'PIPE_FORMAT_R64G64B64A64_FLOAT',
    13: 'PIPE_FORMAT_R32_FLOAT',
    14: 'PIPE_FORMAT_R32G32_FLOAT',
    15: 'PIPE_FORMAT_R32G32B32_FLOAT',
    16: 'PIPE_FORMAT_R32G32B32A32_FLOAT',
    17: 'PIPE_FORMAT_R32_UNORM',
    18: 'PIPE_FORMAT_R32G32_UNORM',
    19: 'PIPE_FORMAT_R32G32B32_UNORM',
    20: 'PIPE_FORMAT_R32G32B32A32_UNORM',
    21: 'PIPE_FORMAT_R32_USCALED',
    22: 'PIPE_FORMAT_R32G32_USCALED',
    23: 'PIPE_FORMAT_R32G32B32_USCALED',
    24: 'PIPE_FORMAT_R32G32B32A32_USCALED',
    25: 'PIPE_FORMAT_R32_SNORM',
    26: 'PIPE_FORMAT_R32G32_SNORM',
    27: 'PIPE_FORMAT_R32G32B32_SNORM',
    28: 'PIPE_FORMAT_R32G32B32A32_SNORM',
    29: 'PIPE_FORMAT_R32_SSCALED',
    30: 'PIPE_FORMAT_R32G32_SSCALED',
    31: 'PIPE_FORMAT_R32G32B32_SSCALED',
    32: 'PIPE_FORMAT_R32G32B32A32_SSCALED',
    33: 'PIPE_FORMAT_R16_UNORM',
    34: 'PIPE_FORMAT_R16G16_UNORM',
    35: 'PIPE_FORMAT_R16G16B16_UNORM',
    36: 'PIPE_FORMAT_R16G16B16A16_UNORM',
    37: 'PIPE_FORMAT_R16_USCALED',
    38: 'PIPE_FORMAT_R16G16_USCALED',
    39: 'PIPE_FORMAT_R16G16B16_USCALED',
    40: 'PIPE_FORMAT_R16G16B16A16_USCALED',
    41: 'PIPE_FORMAT_R16_SNORM',
    42: 'PIPE_FORMAT_R16G16_SNORM',
    43: 'PIPE_FORMAT_R16G16B16_SNORM',
    44: 'PIPE_FORMAT_R16G16B16A16_SNORM',
    45: 'PIPE_FORMAT_R16_SSCALED',
    46: 'PIPE_FORMAT_R16G16_SSCALED',
    47: 'PIPE_FORMAT_R16G16B16_SSCALED',
    48: 'PIPE_FORMAT_R16G16B16A16_SSCALED',
    49: 'PIPE_FORMAT_R8_UNORM',
    50: 'PIPE_FORMAT_R8G8_UNORM',
    51: 'PIPE_FORMAT_R8G8B8_UNORM',
    52: 'PIPE_FORMAT_B8G8R8_UNORM',
    53: 'PIPE_FORMAT_R8G8B8A8_UNORM',
    54: 'PIPE_FORMAT_B8G8R8A8_UNORM',
    55: 'PIPE_FORMAT_R8_USCALED',
    56: 'PIPE_FORMAT_R8G8_USCALED',
    57: 'PIPE_FORMAT_R8G8B8_USCALED',
    58: 'PIPE_FORMAT_B8G8R8_USCALED',
    59: 'PIPE_FORMAT_R8G8B8A8_USCALED',
    60: 'PIPE_FORMAT_B8G8R8A8_USCALED',
    61: 'PIPE_FORMAT_A8B8G8R8_USCALED',
    62: 'PIPE_FORMAT_R8_SNORM',
    63: 'PIPE_FORMAT_R8G8_SNORM',
    64: 'PIPE_FORMAT_R8G8B8_SNORM',
    65: 'PIPE_FORMAT_B8G8R8_SNORM',
    66: 'PIPE_FORMAT_R8G8B8A8_SNORM',
    67: 'PIPE_FORMAT_B8G8R8A8_SNORM',
    68: 'PIPE_FORMAT_R8_SSCALED',
    69: 'PIPE_FORMAT_R8G8_SSCALED',
    70: 'PIPE_FORMAT_R8G8B8_SSCALED',
    71: 'PIPE_FORMAT_B8G8R8_SSCALED',
    72: 'PIPE_FORMAT_R8G8B8A8_SSCALED',
    73: 'PIPE_FORMAT_B8G8R8A8_SSCALED',
    74: 'PIPE_FORMAT_A8B8G8R8_SSCALED',
    75: 'PIPE_FORMAT_A8R8G8B8_UNORM',
    76: 'PIPE_FORMAT_R32_FIXED',
    77: 'PIPE_FORMAT_R32G32_FIXED',
    78: 'PIPE_FORMAT_R32G32B32_FIXED',
    79: 'PIPE_FORMAT_R32G32B32A32_FIXED',
    80: 'PIPE_FORMAT_R16_FLOAT',
    81: 'PIPE_FORMAT_R16G16_FLOAT',
    82: 'PIPE_FORMAT_R16G16B16_FLOAT',
    83: 'PIPE_FORMAT_R16G16B16A16_FLOAT',
    84: 'PIPE_FORMAT_R8_UINT',
    85: 'PIPE_FORMAT_R8G8_UINT',
    86: 'PIPE_FORMAT_R8G8B8_UINT',
    87: 'PIPE_FORMAT_B8G8R8_UINT',
    88: 'PIPE_FORMAT_R8G8B8A8_UINT',
    89: 'PIPE_FORMAT_B8G8R8A8_UINT',
    90: 'PIPE_FORMAT_R8_SINT',
    91: 'PIPE_FORMAT_R8G8_SINT',
    92: 'PIPE_FORMAT_R8G8B8_SINT',
    93: 'PIPE_FORMAT_B8G8R8_SINT',
    94: 'PIPE_FORMAT_R8G8B8A8_SINT',
    95: 'PIPE_FORMAT_B8G8R8A8_SINT',
    96: 'PIPE_FORMAT_R16_UINT',
    97: 'PIPE_FORMAT_R16G16_UINT',
    98: 'PIPE_FORMAT_R16G16B16_UINT',
    99: 'PIPE_FORMAT_R16G16B16A16_UINT',
    100: 'PIPE_FORMAT_R16_SINT',
    101: 'PIPE_FORMAT_R16G16_SINT',
    102: 'PIPE_FORMAT_R16G16B16_SINT',
    103: 'PIPE_FORMAT_R16G16B16A16_SINT',
    104: 'PIPE_FORMAT_R32_UINT',
    105: 'PIPE_FORMAT_R32G32_UINT',
    106: 'PIPE_FORMAT_R32G32B32_UINT',
    107: 'PIPE_FORMAT_R32G32B32A32_UINT',
    108: 'PIPE_FORMAT_R32_SINT',
    109: 'PIPE_FORMAT_R32G32_SINT',
    110: 'PIPE_FORMAT_R32G32B32_SINT',
    111: 'PIPE_FORMAT_R32G32B32A32_SINT',
    112: 'PIPE_FORMAT_R10G10B10A2_UNORM',
    113: 'PIPE_FORMAT_R10G10B10A2_SNORM',
    114: 'PIPE_FORMAT_R10G10B10A2_USCALED',
    115: 'PIPE_FORMAT_R10G10B10A2_SSCALED',
    116: 'PIPE_FORMAT_B10G10R10A2_UNORM',
    117: 'PIPE_FORMAT_B10G10R10A2_SNORM',
    118: 'PIPE_FORMAT_B10G10R10A2_USCALED',
    119: 'PIPE_FORMAT_B10G10R10A2_SSCALED',
    120: 'PIPE_FORMAT_R11G11B10_FLOAT',
    121: 'PIPE_FORMAT_R10G10B10A2_UINT',
    122: 'PIPE_FORMAT_R10G10B10A2_SINT',
    123: 'PIPE_FORMAT_B10G10R10A2_UINT',
    124: 'PIPE_FORMAT_B10G10R10A2_SINT',
    125: 'PIPE_FORMAT_B8G8R8X8_UNORM',
    126: 'PIPE_FORMAT_X8B8G8R8_UNORM',
    127: 'PIPE_FORMAT_X8R8G8B8_UNORM',
    128: 'PIPE_FORMAT_B5G5R5A1_UNORM',
    129: 'PIPE_FORMAT_R4G4B4A4_UNORM',
    130: 'PIPE_FORMAT_B4G4R4A4_UNORM',
    131: 'PIPE_FORMAT_R5G6B5_UNORM',
    132: 'PIPE_FORMAT_B5G6R5_UNORM',
    133: 'PIPE_FORMAT_L8_UNORM',
    134: 'PIPE_FORMAT_A8_UNORM',
    135: 'PIPE_FORMAT_I8_UNORM',
    136: 'PIPE_FORMAT_L8A8_UNORM',
    137: 'PIPE_FORMAT_L16_UNORM',
    138: 'PIPE_FORMAT_UYVY',
    139: 'PIPE_FORMAT_VYUY',
    140: 'PIPE_FORMAT_YUYV',
    141: 'PIPE_FORMAT_YVYU',
    142: 'PIPE_FORMAT_Z16_UNORM',
    143: 'PIPE_FORMAT_Z16_UNORM_S8_UINT',
    144: 'PIPE_FORMAT_Z32_UNORM',
    145: 'PIPE_FORMAT_Z32_FLOAT',
    146: 'PIPE_FORMAT_Z24_UNORM_S8_UINT',
    147: 'PIPE_FORMAT_S8_UINT_Z24_UNORM',
    148: 'PIPE_FORMAT_Z24X8_UNORM',
    149: 'PIPE_FORMAT_X8Z24_UNORM',
    150: 'PIPE_FORMAT_S8_UINT',
    151: 'PIPE_FORMAT_L8_SRGB',
    152: 'PIPE_FORMAT_R8_SRGB',
    153: 'PIPE_FORMAT_L8A8_SRGB',
    154: 'PIPE_FORMAT_R8G8_SRGB',
    155: 'PIPE_FORMAT_R8G8B8_SRGB',
    156: 'PIPE_FORMAT_B8G8R8_SRGB',
    157: 'PIPE_FORMAT_A8B8G8R8_SRGB',
    158: 'PIPE_FORMAT_X8B8G8R8_SRGB',
    159: 'PIPE_FORMAT_B8G8R8A8_SRGB',
    160: 'PIPE_FORMAT_B8G8R8X8_SRGB',
    161: 'PIPE_FORMAT_A8R8G8B8_SRGB',
    162: 'PIPE_FORMAT_X8R8G8B8_SRGB',
    163: 'PIPE_FORMAT_R8G8B8A8_SRGB',
    164: 'PIPE_FORMAT_DXT1_RGB',
    165: 'PIPE_FORMAT_DXT1_RGBA',
    166: 'PIPE_FORMAT_DXT3_RGBA',
    167: 'PIPE_FORMAT_DXT5_RGBA',
    168: 'PIPE_FORMAT_DXT1_SRGB',
    169: 'PIPE_FORMAT_DXT1_SRGBA',
    170: 'PIPE_FORMAT_DXT3_SRGBA',
    171: 'PIPE_FORMAT_DXT5_SRGBA',
    172: 'PIPE_FORMAT_RGTC1_UNORM',
    173: 'PIPE_FORMAT_RGTC1_SNORM',
    174: 'PIPE_FORMAT_RGTC2_UNORM',
    175: 'PIPE_FORMAT_RGTC2_SNORM',
    176: 'PIPE_FORMAT_R8G8_B8G8_UNORM',
    177: 'PIPE_FORMAT_G8R8_G8B8_UNORM',
    178: 'PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM',
    179: 'PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM',
    180: 'PIPE_FORMAT_X6R10_UNORM',
    181: 'PIPE_FORMAT_X6R10X6G10_UNORM',
    182: 'PIPE_FORMAT_X4R12_UNORM',
    183: 'PIPE_FORMAT_X4R12X4G12_UNORM',
    184: 'PIPE_FORMAT_R8SG8SB8UX8U_NORM',
    185: 'PIPE_FORMAT_R5SG5SB6U_NORM',
    186: 'PIPE_FORMAT_A8B8G8R8_UNORM',
    187: 'PIPE_FORMAT_B5G5R5X1_UNORM',
    188: 'PIPE_FORMAT_R9G9B9E5_FLOAT',
    189: 'PIPE_FORMAT_Z32_FLOAT_S8X24_UINT',
    190: 'PIPE_FORMAT_R1_UNORM',
    191: 'PIPE_FORMAT_R10G10B10X2_USCALED',
    192: 'PIPE_FORMAT_R10G10B10X2_SNORM',
    193: 'PIPE_FORMAT_L4A4_UNORM',
    194: 'PIPE_FORMAT_A2R10G10B10_UNORM',
    195: 'PIPE_FORMAT_A2B10G10R10_UNORM',
    196: 'PIPE_FORMAT_R10SG10SB10SA2U_NORM',
    197: 'PIPE_FORMAT_R8G8Bx_SNORM',
    198: 'PIPE_FORMAT_R8G8B8X8_UNORM',
    199: 'PIPE_FORMAT_B4G4R4X4_UNORM',
    200: 'PIPE_FORMAT_X24S8_UINT',
    201: 'PIPE_FORMAT_S8X24_UINT',
    202: 'PIPE_FORMAT_X32_S8X24_UINT',
    203: 'PIPE_FORMAT_R3G3B2_UNORM',
    204: 'PIPE_FORMAT_B2G3R3_UNORM',
    205: 'PIPE_FORMAT_L16A16_UNORM',
    206: 'PIPE_FORMAT_A16_UNORM',
    207: 'PIPE_FORMAT_I16_UNORM',
    208: 'PIPE_FORMAT_LATC1_UNORM',
    209: 'PIPE_FORMAT_LATC1_SNORM',
    210: 'PIPE_FORMAT_LATC2_UNORM',
    211: 'PIPE_FORMAT_LATC2_SNORM',
    212: 'PIPE_FORMAT_A8_SNORM',
    213: 'PIPE_FORMAT_L8_SNORM',
    214: 'PIPE_FORMAT_L8A8_SNORM',
    215: 'PIPE_FORMAT_I8_SNORM',
    216: 'PIPE_FORMAT_A16_SNORM',
    217: 'PIPE_FORMAT_L16_SNORM',
    218: 'PIPE_FORMAT_L16A16_SNORM',
    219: 'PIPE_FORMAT_I16_SNORM',
    220: 'PIPE_FORMAT_A16_FLOAT',
    221: 'PIPE_FORMAT_L16_FLOAT',
    222: 'PIPE_FORMAT_L16A16_FLOAT',
    223: 'PIPE_FORMAT_I16_FLOAT',
    224: 'PIPE_FORMAT_A32_FLOAT',
    225: 'PIPE_FORMAT_L32_FLOAT',
    226: 'PIPE_FORMAT_L32A32_FLOAT',
    227: 'PIPE_FORMAT_I32_FLOAT',
    228: 'PIPE_FORMAT_YV12',
    229: 'PIPE_FORMAT_YV16',
    230: 'PIPE_FORMAT_IYUV',
    231: 'PIPE_FORMAT_NV12',
    232: 'PIPE_FORMAT_NV21',
    233: 'PIPE_FORMAT_NV16',
    234: 'PIPE_FORMAT_NV15',
    235: 'PIPE_FORMAT_NV20',
    236: 'PIPE_FORMAT_Y8_400_UNORM',
    237: 'PIPE_FORMAT_Y8_U8_V8_422_UNORM',
    238: 'PIPE_FORMAT_Y8_U8_V8_444_UNORM',
    239: 'PIPE_FORMAT_Y8_U8_V8_440_UNORM',
    240: 'PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM',
    241: 'PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM',
    242: 'PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM',
    243: 'PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM',
    244: 'PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM',
    245: 'PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM',
    246: 'PIPE_FORMAT_Y16_U16_V16_420_UNORM',
    247: 'PIPE_FORMAT_Y16_U16_V16_422_UNORM',
    248: 'PIPE_FORMAT_Y16_U16V16_422_UNORM',
    249: 'PIPE_FORMAT_Y16_U16_V16_444_UNORM',
    250: 'PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED',
    251: 'PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED',
    252: 'PIPE_FORMAT_A4R4_UNORM',
    253: 'PIPE_FORMAT_R4A4_UNORM',
    254: 'PIPE_FORMAT_R8A8_UNORM',
    255: 'PIPE_FORMAT_A8R8_UNORM',
    256: 'PIPE_FORMAT_A8_UINT',
    257: 'PIPE_FORMAT_I8_UINT',
    258: 'PIPE_FORMAT_L8_UINT',
    259: 'PIPE_FORMAT_L8A8_UINT',
    260: 'PIPE_FORMAT_A8_SINT',
    261: 'PIPE_FORMAT_I8_SINT',
    262: 'PIPE_FORMAT_L8_SINT',
    263: 'PIPE_FORMAT_L8A8_SINT',
    264: 'PIPE_FORMAT_A16_UINT',
    265: 'PIPE_FORMAT_I16_UINT',
    266: 'PIPE_FORMAT_L16_UINT',
    267: 'PIPE_FORMAT_L16A16_UINT',
    268: 'PIPE_FORMAT_A16_SINT',
    269: 'PIPE_FORMAT_I16_SINT',
    270: 'PIPE_FORMAT_L16_SINT',
    271: 'PIPE_FORMAT_L16A16_SINT',
    272: 'PIPE_FORMAT_A32_UINT',
    273: 'PIPE_FORMAT_I32_UINT',
    274: 'PIPE_FORMAT_L32_UINT',
    275: 'PIPE_FORMAT_L32A32_UINT',
    276: 'PIPE_FORMAT_A32_SINT',
    277: 'PIPE_FORMAT_I32_SINT',
    278: 'PIPE_FORMAT_L32_SINT',
    279: 'PIPE_FORMAT_L32A32_SINT',
    280: 'PIPE_FORMAT_A8R8G8B8_UINT',
    281: 'PIPE_FORMAT_A8B8G8R8_UINT',
    282: 'PIPE_FORMAT_A2R10G10B10_UINT',
    283: 'PIPE_FORMAT_A2B10G10R10_UINT',
    284: 'PIPE_FORMAT_R5G6B5_UINT',
    285: 'PIPE_FORMAT_B5G6R5_UINT',
    286: 'PIPE_FORMAT_R5G5B5A1_UINT',
    287: 'PIPE_FORMAT_B5G5R5A1_UINT',
    288: 'PIPE_FORMAT_A1R5G5B5_UINT',
    289: 'PIPE_FORMAT_A1B5G5R5_UINT',
    290: 'PIPE_FORMAT_R4G4B4A4_UINT',
    291: 'PIPE_FORMAT_B4G4R4A4_UINT',
    292: 'PIPE_FORMAT_A4R4G4B4_UINT',
    293: 'PIPE_FORMAT_A4B4G4R4_UINT',
    294: 'PIPE_FORMAT_R3G3B2_UINT',
    295: 'PIPE_FORMAT_B2G3R3_UINT',
    296: 'PIPE_FORMAT_ETC1_RGB8',
    297: 'PIPE_FORMAT_R8G8_R8B8_UNORM',
    298: 'PIPE_FORMAT_R8B8_R8G8_UNORM',
    299: 'PIPE_FORMAT_G8R8_B8R8_UNORM',
    300: 'PIPE_FORMAT_B8R8_G8R8_UNORM',
    301: 'PIPE_FORMAT_G8B8_G8R8_UNORM',
    302: 'PIPE_FORMAT_B8G8_R8G8_UNORM',
    303: 'PIPE_FORMAT_R8G8B8X8_SNORM',
    304: 'PIPE_FORMAT_R8G8B8X8_SRGB',
    305: 'PIPE_FORMAT_R8G8B8X8_UINT',
    306: 'PIPE_FORMAT_R8G8B8X8_SINT',
    307: 'PIPE_FORMAT_B10G10R10X2_UNORM',
    308: 'PIPE_FORMAT_R16G16B16X16_UNORM',
    309: 'PIPE_FORMAT_R16G16B16X16_SNORM',
    310: 'PIPE_FORMAT_R16G16B16X16_FLOAT',
    311: 'PIPE_FORMAT_R16G16B16X16_UINT',
    312: 'PIPE_FORMAT_R16G16B16X16_SINT',
    313: 'PIPE_FORMAT_R32G32B32X32_FLOAT',
    314: 'PIPE_FORMAT_R32G32B32X32_UINT',
    315: 'PIPE_FORMAT_R32G32B32X32_SINT',
    316: 'PIPE_FORMAT_R8A8_SNORM',
    317: 'PIPE_FORMAT_R16A16_UNORM',
    318: 'PIPE_FORMAT_R16A16_SNORM',
    319: 'PIPE_FORMAT_R16A16_FLOAT',
    320: 'PIPE_FORMAT_R32A32_FLOAT',
    321: 'PIPE_FORMAT_R8A8_UINT',
    322: 'PIPE_FORMAT_R8A8_SINT',
    323: 'PIPE_FORMAT_R16A16_UINT',
    324: 'PIPE_FORMAT_R16A16_SINT',
    325: 'PIPE_FORMAT_R32A32_UINT',
    326: 'PIPE_FORMAT_R32A32_SINT',
    327: 'PIPE_FORMAT_B5G6R5_SRGB',
    328: 'PIPE_FORMAT_BPTC_RGBA_UNORM',
    329: 'PIPE_FORMAT_BPTC_SRGBA',
    330: 'PIPE_FORMAT_BPTC_RGB_FLOAT',
    331: 'PIPE_FORMAT_BPTC_RGB_UFLOAT',
    332: 'PIPE_FORMAT_G8R8_UNORM',
    333: 'PIPE_FORMAT_G8R8_SNORM',
    334: 'PIPE_FORMAT_G16R16_UNORM',
    335: 'PIPE_FORMAT_G16R16_SNORM',
    336: 'PIPE_FORMAT_A8B8G8R8_SNORM',
    337: 'PIPE_FORMAT_X8B8G8R8_SNORM',
    338: 'PIPE_FORMAT_ETC2_RGB8',
    339: 'PIPE_FORMAT_ETC2_SRGB8',
    340: 'PIPE_FORMAT_ETC2_RGB8A1',
    341: 'PIPE_FORMAT_ETC2_SRGB8A1',
    342: 'PIPE_FORMAT_ETC2_RGBA8',
    343: 'PIPE_FORMAT_ETC2_SRGBA8',
    344: 'PIPE_FORMAT_ETC2_R11_UNORM',
    345: 'PIPE_FORMAT_ETC2_R11_SNORM',
    346: 'PIPE_FORMAT_ETC2_RG11_UNORM',
    347: 'PIPE_FORMAT_ETC2_RG11_SNORM',
    348: 'PIPE_FORMAT_ASTC_4x4',
    349: 'PIPE_FORMAT_ASTC_5x4',
    350: 'PIPE_FORMAT_ASTC_5x5',
    351: 'PIPE_FORMAT_ASTC_6x5',
    352: 'PIPE_FORMAT_ASTC_6x6',
    353: 'PIPE_FORMAT_ASTC_8x5',
    354: 'PIPE_FORMAT_ASTC_8x6',
    355: 'PIPE_FORMAT_ASTC_8x8',
    356: 'PIPE_FORMAT_ASTC_10x5',
    357: 'PIPE_FORMAT_ASTC_10x6',
    358: 'PIPE_FORMAT_ASTC_10x8',
    359: 'PIPE_FORMAT_ASTC_10x10',
    360: 'PIPE_FORMAT_ASTC_12x10',
    361: 'PIPE_FORMAT_ASTC_12x12',
    362: 'PIPE_FORMAT_ASTC_4x4_SRGB',
    363: 'PIPE_FORMAT_ASTC_5x4_SRGB',
    364: 'PIPE_FORMAT_ASTC_5x5_SRGB',
    365: 'PIPE_FORMAT_ASTC_6x5_SRGB',
    366: 'PIPE_FORMAT_ASTC_6x6_SRGB',
    367: 'PIPE_FORMAT_ASTC_8x5_SRGB',
    368: 'PIPE_FORMAT_ASTC_8x6_SRGB',
    369: 'PIPE_FORMAT_ASTC_8x8_SRGB',
    370: 'PIPE_FORMAT_ASTC_10x5_SRGB',
    371: 'PIPE_FORMAT_ASTC_10x6_SRGB',
    372: 'PIPE_FORMAT_ASTC_10x8_SRGB',
    373: 'PIPE_FORMAT_ASTC_10x10_SRGB',
    374: 'PIPE_FORMAT_ASTC_12x10_SRGB',
    375: 'PIPE_FORMAT_ASTC_12x12_SRGB',
    376: 'PIPE_FORMAT_ASTC_3x3x3',
    377: 'PIPE_FORMAT_ASTC_4x3x3',
    378: 'PIPE_FORMAT_ASTC_4x4x3',
    379: 'PIPE_FORMAT_ASTC_4x4x4',
    380: 'PIPE_FORMAT_ASTC_5x4x4',
    381: 'PIPE_FORMAT_ASTC_5x5x4',
    382: 'PIPE_FORMAT_ASTC_5x5x5',
    383: 'PIPE_FORMAT_ASTC_6x5x5',
    384: 'PIPE_FORMAT_ASTC_6x6x5',
    385: 'PIPE_FORMAT_ASTC_6x6x6',
    386: 'PIPE_FORMAT_ASTC_3x3x3_SRGB',
    387: 'PIPE_FORMAT_ASTC_4x3x3_SRGB',
    388: 'PIPE_FORMAT_ASTC_4x4x3_SRGB',
    389: 'PIPE_FORMAT_ASTC_4x4x4_SRGB',
    390: 'PIPE_FORMAT_ASTC_5x4x4_SRGB',
    391: 'PIPE_FORMAT_ASTC_5x5x4_SRGB',
    392: 'PIPE_FORMAT_ASTC_5x5x5_SRGB',
    393: 'PIPE_FORMAT_ASTC_6x5x5_SRGB',
    394: 'PIPE_FORMAT_ASTC_6x6x5_SRGB',
    395: 'PIPE_FORMAT_ASTC_6x6x6_SRGB',
    396: 'PIPE_FORMAT_ASTC_4x4_FLOAT',
    397: 'PIPE_FORMAT_ASTC_5x4_FLOAT',
    398: 'PIPE_FORMAT_ASTC_5x5_FLOAT',
    399: 'PIPE_FORMAT_ASTC_6x5_FLOAT',
    400: 'PIPE_FORMAT_ASTC_6x6_FLOAT',
    401: 'PIPE_FORMAT_ASTC_8x5_FLOAT',
    402: 'PIPE_FORMAT_ASTC_8x6_FLOAT',
    403: 'PIPE_FORMAT_ASTC_8x8_FLOAT',
    404: 'PIPE_FORMAT_ASTC_10x5_FLOAT',
    405: 'PIPE_FORMAT_ASTC_10x6_FLOAT',
    406: 'PIPE_FORMAT_ASTC_10x8_FLOAT',
    407: 'PIPE_FORMAT_ASTC_10x10_FLOAT',
    408: 'PIPE_FORMAT_ASTC_12x10_FLOAT',
    409: 'PIPE_FORMAT_ASTC_12x12_FLOAT',
    410: 'PIPE_FORMAT_FXT1_RGB',
    411: 'PIPE_FORMAT_FXT1_RGBA',
    412: 'PIPE_FORMAT_P010',
    413: 'PIPE_FORMAT_P012',
    414: 'PIPE_FORMAT_P016',
    415: 'PIPE_FORMAT_P030',
    416: 'PIPE_FORMAT_Y210',
    417: 'PIPE_FORMAT_Y212',
    418: 'PIPE_FORMAT_Y216',
    419: 'PIPE_FORMAT_Y410',
    420: 'PIPE_FORMAT_Y412',
    421: 'PIPE_FORMAT_Y416',
    422: 'PIPE_FORMAT_R10G10B10X2_UNORM',
    423: 'PIPE_FORMAT_A1R5G5B5_UNORM',
    424: 'PIPE_FORMAT_A1B5G5R5_UNORM',
    425: 'PIPE_FORMAT_X1B5G5R5_UNORM',
    426: 'PIPE_FORMAT_R5G5B5A1_UNORM',
    427: 'PIPE_FORMAT_A4R4G4B4_UNORM',
    428: 'PIPE_FORMAT_A4B4G4R4_UNORM',
    429: 'PIPE_FORMAT_G8R8_SINT',
    430: 'PIPE_FORMAT_A8B8G8R8_SINT',
    431: 'PIPE_FORMAT_X8B8G8R8_SINT',
    432: 'PIPE_FORMAT_ATC_RGB',
    433: 'PIPE_FORMAT_ATC_RGBA_EXPLICIT',
    434: 'PIPE_FORMAT_ATC_RGBA_INTERPOLATED',
    435: 'PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8',
    436: 'PIPE_FORMAT_AYUV',
    437: 'PIPE_FORMAT_XYUV',
    438: 'PIPE_FORMAT_R8G8B8_420_UNORM_PACKED',
    439: 'PIPE_FORMAT_R8_G8B8_420_UNORM',
    440: 'PIPE_FORMAT_R8_B8G8_420_UNORM',
    441: 'PIPE_FORMAT_G8_B8R8_420_UNORM',
    442: 'PIPE_FORMAT_R10G10B10_420_UNORM_PACKED',
    443: 'PIPE_FORMAT_R10_G10B10_420_UNORM',
    444: 'PIPE_FORMAT_R10_G10B10_422_UNORM',
    445: 'PIPE_FORMAT_R8_G8_B8_420_UNORM',
    446: 'PIPE_FORMAT_R8_B8_G8_420_UNORM',
    447: 'PIPE_FORMAT_G8_B8_R8_420_UNORM',
    448: 'PIPE_FORMAT_R8_G8B8_422_UNORM',
    449: 'PIPE_FORMAT_R8_B8G8_422_UNORM',
    450: 'PIPE_FORMAT_G8_B8R8_422_UNORM',
    451: 'PIPE_FORMAT_R8_G8_B8_UNORM',
    452: 'PIPE_FORMAT_Y8_UNORM',
    453: 'PIPE_FORMAT_B8G8R8X8_SNORM',
    454: 'PIPE_FORMAT_B8G8R8X8_UINT',
    455: 'PIPE_FORMAT_B8G8R8X8_SINT',
    456: 'PIPE_FORMAT_A8R8G8B8_SNORM',
    457: 'PIPE_FORMAT_A8R8G8B8_SINT',
    458: 'PIPE_FORMAT_X8R8G8B8_SNORM',
    459: 'PIPE_FORMAT_X8R8G8B8_SINT',
    460: 'PIPE_FORMAT_R5G5B5X1_UNORM',
    461: 'PIPE_FORMAT_X1R5G5B5_UNORM',
    462: 'PIPE_FORMAT_R4G4B4X4_UNORM',
    463: 'PIPE_FORMAT_B10G10R10X2_SNORM',
    464: 'PIPE_FORMAT_R5G6B5_SRGB',
    465: 'PIPE_FORMAT_R10G10B10X2_SINT',
    466: 'PIPE_FORMAT_B10G10R10X2_SINT',
    467: 'PIPE_FORMAT_G16R16_SINT',
    468: 'PIPE_FORMAT_COUNT',
}
PIPE_FORMAT_NONE = 0
PIPE_FORMAT_R64_UINT = 1
PIPE_FORMAT_R64G64_UINT = 2
PIPE_FORMAT_R64G64B64_UINT = 3
PIPE_FORMAT_R64G64B64A64_UINT = 4
PIPE_FORMAT_R64_SINT = 5
PIPE_FORMAT_R64G64_SINT = 6
PIPE_FORMAT_R64G64B64_SINT = 7
PIPE_FORMAT_R64G64B64A64_SINT = 8
PIPE_FORMAT_R64_FLOAT = 9
PIPE_FORMAT_R64G64_FLOAT = 10
PIPE_FORMAT_R64G64B64_FLOAT = 11
PIPE_FORMAT_R64G64B64A64_FLOAT = 12
PIPE_FORMAT_R32_FLOAT = 13
PIPE_FORMAT_R32G32_FLOAT = 14
PIPE_FORMAT_R32G32B32_FLOAT = 15
PIPE_FORMAT_R32G32B32A32_FLOAT = 16
PIPE_FORMAT_R32_UNORM = 17
PIPE_FORMAT_R32G32_UNORM = 18
PIPE_FORMAT_R32G32B32_UNORM = 19
PIPE_FORMAT_R32G32B32A32_UNORM = 20
PIPE_FORMAT_R32_USCALED = 21
PIPE_FORMAT_R32G32_USCALED = 22
PIPE_FORMAT_R32G32B32_USCALED = 23
PIPE_FORMAT_R32G32B32A32_USCALED = 24
PIPE_FORMAT_R32_SNORM = 25
PIPE_FORMAT_R32G32_SNORM = 26
PIPE_FORMAT_R32G32B32_SNORM = 27
PIPE_FORMAT_R32G32B32A32_SNORM = 28
PIPE_FORMAT_R32_SSCALED = 29
PIPE_FORMAT_R32G32_SSCALED = 30
PIPE_FORMAT_R32G32B32_SSCALED = 31
PIPE_FORMAT_R32G32B32A32_SSCALED = 32
PIPE_FORMAT_R16_UNORM = 33
PIPE_FORMAT_R16G16_UNORM = 34
PIPE_FORMAT_R16G16B16_UNORM = 35
PIPE_FORMAT_R16G16B16A16_UNORM = 36
PIPE_FORMAT_R16_USCALED = 37
PIPE_FORMAT_R16G16_USCALED = 38
PIPE_FORMAT_R16G16B16_USCALED = 39
PIPE_FORMAT_R16G16B16A16_USCALED = 40
PIPE_FORMAT_R16_SNORM = 41
PIPE_FORMAT_R16G16_SNORM = 42
PIPE_FORMAT_R16G16B16_SNORM = 43
PIPE_FORMAT_R16G16B16A16_SNORM = 44
PIPE_FORMAT_R16_SSCALED = 45
PIPE_FORMAT_R16G16_SSCALED = 46
PIPE_FORMAT_R16G16B16_SSCALED = 47
PIPE_FORMAT_R16G16B16A16_SSCALED = 48
PIPE_FORMAT_R8_UNORM = 49
PIPE_FORMAT_R8G8_UNORM = 50
PIPE_FORMAT_R8G8B8_UNORM = 51
PIPE_FORMAT_B8G8R8_UNORM = 52
PIPE_FORMAT_R8G8B8A8_UNORM = 53
PIPE_FORMAT_B8G8R8A8_UNORM = 54
PIPE_FORMAT_R8_USCALED = 55
PIPE_FORMAT_R8G8_USCALED = 56
PIPE_FORMAT_R8G8B8_USCALED = 57
PIPE_FORMAT_B8G8R8_USCALED = 58
PIPE_FORMAT_R8G8B8A8_USCALED = 59
PIPE_FORMAT_B8G8R8A8_USCALED = 60
PIPE_FORMAT_A8B8G8R8_USCALED = 61
PIPE_FORMAT_R8_SNORM = 62
PIPE_FORMAT_R8G8_SNORM = 63
PIPE_FORMAT_R8G8B8_SNORM = 64
PIPE_FORMAT_B8G8R8_SNORM = 65
PIPE_FORMAT_R8G8B8A8_SNORM = 66
PIPE_FORMAT_B8G8R8A8_SNORM = 67
PIPE_FORMAT_R8_SSCALED = 68
PIPE_FORMAT_R8G8_SSCALED = 69
PIPE_FORMAT_R8G8B8_SSCALED = 70
PIPE_FORMAT_B8G8R8_SSCALED = 71
PIPE_FORMAT_R8G8B8A8_SSCALED = 72
PIPE_FORMAT_B8G8R8A8_SSCALED = 73
PIPE_FORMAT_A8B8G8R8_SSCALED = 74
PIPE_FORMAT_A8R8G8B8_UNORM = 75
PIPE_FORMAT_R32_FIXED = 76
PIPE_FORMAT_R32G32_FIXED = 77
PIPE_FORMAT_R32G32B32_FIXED = 78
PIPE_FORMAT_R32G32B32A32_FIXED = 79
PIPE_FORMAT_R16_FLOAT = 80
PIPE_FORMAT_R16G16_FLOAT = 81
PIPE_FORMAT_R16G16B16_FLOAT = 82
PIPE_FORMAT_R16G16B16A16_FLOAT = 83
PIPE_FORMAT_R8_UINT = 84
PIPE_FORMAT_R8G8_UINT = 85
PIPE_FORMAT_R8G8B8_UINT = 86
PIPE_FORMAT_B8G8R8_UINT = 87
PIPE_FORMAT_R8G8B8A8_UINT = 88
PIPE_FORMAT_B8G8R8A8_UINT = 89
PIPE_FORMAT_R8_SINT = 90
PIPE_FORMAT_R8G8_SINT = 91
PIPE_FORMAT_R8G8B8_SINT = 92
PIPE_FORMAT_B8G8R8_SINT = 93
PIPE_FORMAT_R8G8B8A8_SINT = 94
PIPE_FORMAT_B8G8R8A8_SINT = 95
PIPE_FORMAT_R16_UINT = 96
PIPE_FORMAT_R16G16_UINT = 97
PIPE_FORMAT_R16G16B16_UINT = 98
PIPE_FORMAT_R16G16B16A16_UINT = 99
PIPE_FORMAT_R16_SINT = 100
PIPE_FORMAT_R16G16_SINT = 101
PIPE_FORMAT_R16G16B16_SINT = 102
PIPE_FORMAT_R16G16B16A16_SINT = 103
PIPE_FORMAT_R32_UINT = 104
PIPE_FORMAT_R32G32_UINT = 105
PIPE_FORMAT_R32G32B32_UINT = 106
PIPE_FORMAT_R32G32B32A32_UINT = 107
PIPE_FORMAT_R32_SINT = 108
PIPE_FORMAT_R32G32_SINT = 109
PIPE_FORMAT_R32G32B32_SINT = 110
PIPE_FORMAT_R32G32B32A32_SINT = 111
PIPE_FORMAT_R10G10B10A2_UNORM = 112
PIPE_FORMAT_R10G10B10A2_SNORM = 113
PIPE_FORMAT_R10G10B10A2_USCALED = 114
PIPE_FORMAT_R10G10B10A2_SSCALED = 115
PIPE_FORMAT_B10G10R10A2_UNORM = 116
PIPE_FORMAT_B10G10R10A2_SNORM = 117
PIPE_FORMAT_B10G10R10A2_USCALED = 118
PIPE_FORMAT_B10G10R10A2_SSCALED = 119
PIPE_FORMAT_R11G11B10_FLOAT = 120
PIPE_FORMAT_R10G10B10A2_UINT = 121
PIPE_FORMAT_R10G10B10A2_SINT = 122
PIPE_FORMAT_B10G10R10A2_UINT = 123
PIPE_FORMAT_B10G10R10A2_SINT = 124
PIPE_FORMAT_B8G8R8X8_UNORM = 125
PIPE_FORMAT_X8B8G8R8_UNORM = 126
PIPE_FORMAT_X8R8G8B8_UNORM = 127
PIPE_FORMAT_B5G5R5A1_UNORM = 128
PIPE_FORMAT_R4G4B4A4_UNORM = 129
PIPE_FORMAT_B4G4R4A4_UNORM = 130
PIPE_FORMAT_R5G6B5_UNORM = 131
PIPE_FORMAT_B5G6R5_UNORM = 132
PIPE_FORMAT_L8_UNORM = 133
PIPE_FORMAT_A8_UNORM = 134
PIPE_FORMAT_I8_UNORM = 135
PIPE_FORMAT_L8A8_UNORM = 136
PIPE_FORMAT_L16_UNORM = 137
PIPE_FORMAT_UYVY = 138
PIPE_FORMAT_VYUY = 139
PIPE_FORMAT_YUYV = 140
PIPE_FORMAT_YVYU = 141
PIPE_FORMAT_Z16_UNORM = 142
PIPE_FORMAT_Z16_UNORM_S8_UINT = 143
PIPE_FORMAT_Z32_UNORM = 144
PIPE_FORMAT_Z32_FLOAT = 145
PIPE_FORMAT_Z24_UNORM_S8_UINT = 146
PIPE_FORMAT_S8_UINT_Z24_UNORM = 147
PIPE_FORMAT_Z24X8_UNORM = 148
PIPE_FORMAT_X8Z24_UNORM = 149
PIPE_FORMAT_S8_UINT = 150
PIPE_FORMAT_L8_SRGB = 151
PIPE_FORMAT_R8_SRGB = 152
PIPE_FORMAT_L8A8_SRGB = 153
PIPE_FORMAT_R8G8_SRGB = 154
PIPE_FORMAT_R8G8B8_SRGB = 155
PIPE_FORMAT_B8G8R8_SRGB = 156
PIPE_FORMAT_A8B8G8R8_SRGB = 157
PIPE_FORMAT_X8B8G8R8_SRGB = 158
PIPE_FORMAT_B8G8R8A8_SRGB = 159
PIPE_FORMAT_B8G8R8X8_SRGB = 160
PIPE_FORMAT_A8R8G8B8_SRGB = 161
PIPE_FORMAT_X8R8G8B8_SRGB = 162
PIPE_FORMAT_R8G8B8A8_SRGB = 163
PIPE_FORMAT_DXT1_RGB = 164
PIPE_FORMAT_DXT1_RGBA = 165
PIPE_FORMAT_DXT3_RGBA = 166
PIPE_FORMAT_DXT5_RGBA = 167
PIPE_FORMAT_DXT1_SRGB = 168
PIPE_FORMAT_DXT1_SRGBA = 169
PIPE_FORMAT_DXT3_SRGBA = 170
PIPE_FORMAT_DXT5_SRGBA = 171
PIPE_FORMAT_RGTC1_UNORM = 172
PIPE_FORMAT_RGTC1_SNORM = 173
PIPE_FORMAT_RGTC2_UNORM = 174
PIPE_FORMAT_RGTC2_SNORM = 175
PIPE_FORMAT_R8G8_B8G8_UNORM = 176
PIPE_FORMAT_G8R8_G8B8_UNORM = 177
PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM = 178
PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM = 179
PIPE_FORMAT_X6R10_UNORM = 180
PIPE_FORMAT_X6R10X6G10_UNORM = 181
PIPE_FORMAT_X4R12_UNORM = 182
PIPE_FORMAT_X4R12X4G12_UNORM = 183
PIPE_FORMAT_R8SG8SB8UX8U_NORM = 184
PIPE_FORMAT_R5SG5SB6U_NORM = 185
PIPE_FORMAT_A8B8G8R8_UNORM = 186
PIPE_FORMAT_B5G5R5X1_UNORM = 187
PIPE_FORMAT_R9G9B9E5_FLOAT = 188
PIPE_FORMAT_Z32_FLOAT_S8X24_UINT = 189
PIPE_FORMAT_R1_UNORM = 190
PIPE_FORMAT_R10G10B10X2_USCALED = 191
PIPE_FORMAT_R10G10B10X2_SNORM = 192
PIPE_FORMAT_L4A4_UNORM = 193
PIPE_FORMAT_A2R10G10B10_UNORM = 194
PIPE_FORMAT_A2B10G10R10_UNORM = 195
PIPE_FORMAT_R10SG10SB10SA2U_NORM = 196
PIPE_FORMAT_R8G8Bx_SNORM = 197
PIPE_FORMAT_R8G8B8X8_UNORM = 198
PIPE_FORMAT_B4G4R4X4_UNORM = 199
PIPE_FORMAT_X24S8_UINT = 200
PIPE_FORMAT_S8X24_UINT = 201
PIPE_FORMAT_X32_S8X24_UINT = 202
PIPE_FORMAT_R3G3B2_UNORM = 203
PIPE_FORMAT_B2G3R3_UNORM = 204
PIPE_FORMAT_L16A16_UNORM = 205
PIPE_FORMAT_A16_UNORM = 206
PIPE_FORMAT_I16_UNORM = 207
PIPE_FORMAT_LATC1_UNORM = 208
PIPE_FORMAT_LATC1_SNORM = 209
PIPE_FORMAT_LATC2_UNORM = 210
PIPE_FORMAT_LATC2_SNORM = 211
PIPE_FORMAT_A8_SNORM = 212
PIPE_FORMAT_L8_SNORM = 213
PIPE_FORMAT_L8A8_SNORM = 214
PIPE_FORMAT_I8_SNORM = 215
PIPE_FORMAT_A16_SNORM = 216
PIPE_FORMAT_L16_SNORM = 217
PIPE_FORMAT_L16A16_SNORM = 218
PIPE_FORMAT_I16_SNORM = 219
PIPE_FORMAT_A16_FLOAT = 220
PIPE_FORMAT_L16_FLOAT = 221
PIPE_FORMAT_L16A16_FLOAT = 222
PIPE_FORMAT_I16_FLOAT = 223
PIPE_FORMAT_A32_FLOAT = 224
PIPE_FORMAT_L32_FLOAT = 225
PIPE_FORMAT_L32A32_FLOAT = 226
PIPE_FORMAT_I32_FLOAT = 227
PIPE_FORMAT_YV12 = 228
PIPE_FORMAT_YV16 = 229
PIPE_FORMAT_IYUV = 230
PIPE_FORMAT_NV12 = 231
PIPE_FORMAT_NV21 = 232
PIPE_FORMAT_NV16 = 233
PIPE_FORMAT_NV15 = 234
PIPE_FORMAT_NV20 = 235
PIPE_FORMAT_Y8_400_UNORM = 236
PIPE_FORMAT_Y8_U8_V8_422_UNORM = 237
PIPE_FORMAT_Y8_U8_V8_444_UNORM = 238
PIPE_FORMAT_Y8_U8_V8_440_UNORM = 239
PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM = 240
PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM = 241
PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM = 242
PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM = 243
PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM = 244
PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM = 245
PIPE_FORMAT_Y16_U16_V16_420_UNORM = 246
PIPE_FORMAT_Y16_U16_V16_422_UNORM = 247
PIPE_FORMAT_Y16_U16V16_422_UNORM = 248
PIPE_FORMAT_Y16_U16_V16_444_UNORM = 249
PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED = 250
PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED = 251
PIPE_FORMAT_A4R4_UNORM = 252
PIPE_FORMAT_R4A4_UNORM = 253
PIPE_FORMAT_R8A8_UNORM = 254
PIPE_FORMAT_A8R8_UNORM = 255
PIPE_FORMAT_A8_UINT = 256
PIPE_FORMAT_I8_UINT = 257
PIPE_FORMAT_L8_UINT = 258
PIPE_FORMAT_L8A8_UINT = 259
PIPE_FORMAT_A8_SINT = 260
PIPE_FORMAT_I8_SINT = 261
PIPE_FORMAT_L8_SINT = 262
PIPE_FORMAT_L8A8_SINT = 263
PIPE_FORMAT_A16_UINT = 264
PIPE_FORMAT_I16_UINT = 265
PIPE_FORMAT_L16_UINT = 266
PIPE_FORMAT_L16A16_UINT = 267
PIPE_FORMAT_A16_SINT = 268
PIPE_FORMAT_I16_SINT = 269
PIPE_FORMAT_L16_SINT = 270
PIPE_FORMAT_L16A16_SINT = 271
PIPE_FORMAT_A32_UINT = 272
PIPE_FORMAT_I32_UINT = 273
PIPE_FORMAT_L32_UINT = 274
PIPE_FORMAT_L32A32_UINT = 275
PIPE_FORMAT_A32_SINT = 276
PIPE_FORMAT_I32_SINT = 277
PIPE_FORMAT_L32_SINT = 278
PIPE_FORMAT_L32A32_SINT = 279
PIPE_FORMAT_A8R8G8B8_UINT = 280
PIPE_FORMAT_A8B8G8R8_UINT = 281
PIPE_FORMAT_A2R10G10B10_UINT = 282
PIPE_FORMAT_A2B10G10R10_UINT = 283
PIPE_FORMAT_R5G6B5_UINT = 284
PIPE_FORMAT_B5G6R5_UINT = 285
PIPE_FORMAT_R5G5B5A1_UINT = 286
PIPE_FORMAT_B5G5R5A1_UINT = 287
PIPE_FORMAT_A1R5G5B5_UINT = 288
PIPE_FORMAT_A1B5G5R5_UINT = 289
PIPE_FORMAT_R4G4B4A4_UINT = 290
PIPE_FORMAT_B4G4R4A4_UINT = 291
PIPE_FORMAT_A4R4G4B4_UINT = 292
PIPE_FORMAT_A4B4G4R4_UINT = 293
PIPE_FORMAT_R3G3B2_UINT = 294
PIPE_FORMAT_B2G3R3_UINT = 295
PIPE_FORMAT_ETC1_RGB8 = 296
PIPE_FORMAT_R8G8_R8B8_UNORM = 297
PIPE_FORMAT_R8B8_R8G8_UNORM = 298
PIPE_FORMAT_G8R8_B8R8_UNORM = 299
PIPE_FORMAT_B8R8_G8R8_UNORM = 300
PIPE_FORMAT_G8B8_G8R8_UNORM = 301
PIPE_FORMAT_B8G8_R8G8_UNORM = 302
PIPE_FORMAT_R8G8B8X8_SNORM = 303
PIPE_FORMAT_R8G8B8X8_SRGB = 304
PIPE_FORMAT_R8G8B8X8_UINT = 305
PIPE_FORMAT_R8G8B8X8_SINT = 306
PIPE_FORMAT_B10G10R10X2_UNORM = 307
PIPE_FORMAT_R16G16B16X16_UNORM = 308
PIPE_FORMAT_R16G16B16X16_SNORM = 309
PIPE_FORMAT_R16G16B16X16_FLOAT = 310
PIPE_FORMAT_R16G16B16X16_UINT = 311
PIPE_FORMAT_R16G16B16X16_SINT = 312
PIPE_FORMAT_R32G32B32X32_FLOAT = 313
PIPE_FORMAT_R32G32B32X32_UINT = 314
PIPE_FORMAT_R32G32B32X32_SINT = 315
PIPE_FORMAT_R8A8_SNORM = 316
PIPE_FORMAT_R16A16_UNORM = 317
PIPE_FORMAT_R16A16_SNORM = 318
PIPE_FORMAT_R16A16_FLOAT = 319
PIPE_FORMAT_R32A32_FLOAT = 320
PIPE_FORMAT_R8A8_UINT = 321
PIPE_FORMAT_R8A8_SINT = 322
PIPE_FORMAT_R16A16_UINT = 323
PIPE_FORMAT_R16A16_SINT = 324
PIPE_FORMAT_R32A32_UINT = 325
PIPE_FORMAT_R32A32_SINT = 326
PIPE_FORMAT_B5G6R5_SRGB = 327
PIPE_FORMAT_BPTC_RGBA_UNORM = 328
PIPE_FORMAT_BPTC_SRGBA = 329
PIPE_FORMAT_BPTC_RGB_FLOAT = 330
PIPE_FORMAT_BPTC_RGB_UFLOAT = 331
PIPE_FORMAT_G8R8_UNORM = 332
PIPE_FORMAT_G8R8_SNORM = 333
PIPE_FORMAT_G16R16_UNORM = 334
PIPE_FORMAT_G16R16_SNORM = 335
PIPE_FORMAT_A8B8G8R8_SNORM = 336
PIPE_FORMAT_X8B8G8R8_SNORM = 337
PIPE_FORMAT_ETC2_RGB8 = 338
PIPE_FORMAT_ETC2_SRGB8 = 339
PIPE_FORMAT_ETC2_RGB8A1 = 340
PIPE_FORMAT_ETC2_SRGB8A1 = 341
PIPE_FORMAT_ETC2_RGBA8 = 342
PIPE_FORMAT_ETC2_SRGBA8 = 343
PIPE_FORMAT_ETC2_R11_UNORM = 344
PIPE_FORMAT_ETC2_R11_SNORM = 345
PIPE_FORMAT_ETC2_RG11_UNORM = 346
PIPE_FORMAT_ETC2_RG11_SNORM = 347
PIPE_FORMAT_ASTC_4x4 = 348
PIPE_FORMAT_ASTC_5x4 = 349
PIPE_FORMAT_ASTC_5x5 = 350
PIPE_FORMAT_ASTC_6x5 = 351
PIPE_FORMAT_ASTC_6x6 = 352
PIPE_FORMAT_ASTC_8x5 = 353
PIPE_FORMAT_ASTC_8x6 = 354
PIPE_FORMAT_ASTC_8x8 = 355
PIPE_FORMAT_ASTC_10x5 = 356
PIPE_FORMAT_ASTC_10x6 = 357
PIPE_FORMAT_ASTC_10x8 = 358
PIPE_FORMAT_ASTC_10x10 = 359
PIPE_FORMAT_ASTC_12x10 = 360
PIPE_FORMAT_ASTC_12x12 = 361
PIPE_FORMAT_ASTC_4x4_SRGB = 362
PIPE_FORMAT_ASTC_5x4_SRGB = 363
PIPE_FORMAT_ASTC_5x5_SRGB = 364
PIPE_FORMAT_ASTC_6x5_SRGB = 365
PIPE_FORMAT_ASTC_6x6_SRGB = 366
PIPE_FORMAT_ASTC_8x5_SRGB = 367
PIPE_FORMAT_ASTC_8x6_SRGB = 368
PIPE_FORMAT_ASTC_8x8_SRGB = 369
PIPE_FORMAT_ASTC_10x5_SRGB = 370
PIPE_FORMAT_ASTC_10x6_SRGB = 371
PIPE_FORMAT_ASTC_10x8_SRGB = 372
PIPE_FORMAT_ASTC_10x10_SRGB = 373
PIPE_FORMAT_ASTC_12x10_SRGB = 374
PIPE_FORMAT_ASTC_12x12_SRGB = 375
PIPE_FORMAT_ASTC_3x3x3 = 376
PIPE_FORMAT_ASTC_4x3x3 = 377
PIPE_FORMAT_ASTC_4x4x3 = 378
PIPE_FORMAT_ASTC_4x4x4 = 379
PIPE_FORMAT_ASTC_5x4x4 = 380
PIPE_FORMAT_ASTC_5x5x4 = 381
PIPE_FORMAT_ASTC_5x5x5 = 382
PIPE_FORMAT_ASTC_6x5x5 = 383
PIPE_FORMAT_ASTC_6x6x5 = 384
PIPE_FORMAT_ASTC_6x6x6 = 385
PIPE_FORMAT_ASTC_3x3x3_SRGB = 386
PIPE_FORMAT_ASTC_4x3x3_SRGB = 387
PIPE_FORMAT_ASTC_4x4x3_SRGB = 388
PIPE_FORMAT_ASTC_4x4x4_SRGB = 389
PIPE_FORMAT_ASTC_5x4x4_SRGB = 390
PIPE_FORMAT_ASTC_5x5x4_SRGB = 391
PIPE_FORMAT_ASTC_5x5x5_SRGB = 392
PIPE_FORMAT_ASTC_6x5x5_SRGB = 393
PIPE_FORMAT_ASTC_6x6x5_SRGB = 394
PIPE_FORMAT_ASTC_6x6x6_SRGB = 395
PIPE_FORMAT_ASTC_4x4_FLOAT = 396
PIPE_FORMAT_ASTC_5x4_FLOAT = 397
PIPE_FORMAT_ASTC_5x5_FLOAT = 398
PIPE_FORMAT_ASTC_6x5_FLOAT = 399
PIPE_FORMAT_ASTC_6x6_FLOAT = 400
PIPE_FORMAT_ASTC_8x5_FLOAT = 401
PIPE_FORMAT_ASTC_8x6_FLOAT = 402
PIPE_FORMAT_ASTC_8x8_FLOAT = 403
PIPE_FORMAT_ASTC_10x5_FLOAT = 404
PIPE_FORMAT_ASTC_10x6_FLOAT = 405
PIPE_FORMAT_ASTC_10x8_FLOAT = 406
PIPE_FORMAT_ASTC_10x10_FLOAT = 407
PIPE_FORMAT_ASTC_12x10_FLOAT = 408
PIPE_FORMAT_ASTC_12x12_FLOAT = 409
PIPE_FORMAT_FXT1_RGB = 410
PIPE_FORMAT_FXT1_RGBA = 411
PIPE_FORMAT_P010 = 412
PIPE_FORMAT_P012 = 413
PIPE_FORMAT_P016 = 414
PIPE_FORMAT_P030 = 415
PIPE_FORMAT_Y210 = 416
PIPE_FORMAT_Y212 = 417
PIPE_FORMAT_Y216 = 418
PIPE_FORMAT_Y410 = 419
PIPE_FORMAT_Y412 = 420
PIPE_FORMAT_Y416 = 421
PIPE_FORMAT_R10G10B10X2_UNORM = 422
PIPE_FORMAT_A1R5G5B5_UNORM = 423
PIPE_FORMAT_A1B5G5R5_UNORM = 424
PIPE_FORMAT_X1B5G5R5_UNORM = 425
PIPE_FORMAT_R5G5B5A1_UNORM = 426
PIPE_FORMAT_A4R4G4B4_UNORM = 427
PIPE_FORMAT_A4B4G4R4_UNORM = 428
PIPE_FORMAT_G8R8_SINT = 429
PIPE_FORMAT_A8B8G8R8_SINT = 430
PIPE_FORMAT_X8B8G8R8_SINT = 431
PIPE_FORMAT_ATC_RGB = 432
PIPE_FORMAT_ATC_RGBA_EXPLICIT = 433
PIPE_FORMAT_ATC_RGBA_INTERPOLATED = 434
PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8 = 435
PIPE_FORMAT_AYUV = 436
PIPE_FORMAT_XYUV = 437
PIPE_FORMAT_R8G8B8_420_UNORM_PACKED = 438
PIPE_FORMAT_R8_G8B8_420_UNORM = 439
PIPE_FORMAT_R8_B8G8_420_UNORM = 440
PIPE_FORMAT_G8_B8R8_420_UNORM = 441
PIPE_FORMAT_R10G10B10_420_UNORM_PACKED = 442
PIPE_FORMAT_R10_G10B10_420_UNORM = 443
PIPE_FORMAT_R10_G10B10_422_UNORM = 444
PIPE_FORMAT_R8_G8_B8_420_UNORM = 445
PIPE_FORMAT_R8_B8_G8_420_UNORM = 446
PIPE_FORMAT_G8_B8_R8_420_UNORM = 447
PIPE_FORMAT_R8_G8B8_422_UNORM = 448
PIPE_FORMAT_R8_B8G8_422_UNORM = 449
PIPE_FORMAT_G8_B8R8_422_UNORM = 450
PIPE_FORMAT_R8_G8_B8_UNORM = 451
PIPE_FORMAT_Y8_UNORM = 452
PIPE_FORMAT_B8G8R8X8_SNORM = 453
PIPE_FORMAT_B8G8R8X8_UINT = 454
PIPE_FORMAT_B8G8R8X8_SINT = 455
PIPE_FORMAT_A8R8G8B8_SNORM = 456
PIPE_FORMAT_A8R8G8B8_SINT = 457
PIPE_FORMAT_X8R8G8B8_SNORM = 458
PIPE_FORMAT_X8R8G8B8_SINT = 459
PIPE_FORMAT_R5G5B5X1_UNORM = 460
PIPE_FORMAT_X1R5G5B5_UNORM = 461
PIPE_FORMAT_R4G4B4X4_UNORM = 462
PIPE_FORMAT_B10G10R10X2_SNORM = 463
PIPE_FORMAT_R5G6B5_SRGB = 464
PIPE_FORMAT_R10G10B10X2_SINT = 465
PIPE_FORMAT_B10G10R10X2_SINT = 466
PIPE_FORMAT_G16R16_SINT = 467
PIPE_FORMAT_COUNT = 468
pipe_format = ctypes.c_uint32 # enum
class union_glsl_struct_field_0(Union):
    pass

class struct_glsl_struct_field_0_0(Structure):
    pass

struct_glsl_struct_field_0_0._pack_ = 1 # source:False
struct_glsl_struct_field_0_0._fields_ = [
    ('interpolation', ctypes.c_uint32, 3),
    ('centroid', ctypes.c_uint32, 1),
    ('sample', ctypes.c_uint32, 1),
    ('matrix_layout', ctypes.c_uint32, 2),
    ('patch', ctypes.c_uint32, 1),
    ('precision', ctypes.c_uint32, 2),
    ('memory_read_only', ctypes.c_uint32, 1),
    ('memory_write_only', ctypes.c_uint32, 1),
    ('memory_coherent', ctypes.c_uint32, 1),
    ('memory_volatile', ctypes.c_uint32, 1),
    ('memory_restrict', ctypes.c_uint32, 1),
    ('explicit_xfb_buffer', ctypes.c_uint32, 1),
    ('implicit_sized_array', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint16, 15),
]

union_glsl_struct_field_0._pack_ = 1 # source:False
union_glsl_struct_field_0._anonymous_ = ('_0',)
union_glsl_struct_field_0._fields_ = [
    ('_0', struct_glsl_struct_field_0_0),
    ('flags', ctypes.c_uint32),
]

struct_glsl_struct_field._pack_ = 1 # source:False
struct_glsl_struct_field._anonymous_ = ('_0',)
struct_glsl_struct_field._fields_ = [
    ('type', ctypes.POINTER(struct_glsl_type)),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('location', ctypes.c_int32),
    ('component', ctypes.c_int32),
    ('offset', ctypes.c_int32),
    ('xfb_buffer', ctypes.c_int32),
    ('xfb_stride', ctypes.c_int32),
    ('image_format', pipe_format),
    ('_0', union_glsl_struct_field_0),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

glsl_struct_field = struct_glsl_struct_field
try:
    glsl_type_singleton_init_or_ref = _libraries['libtinymesa_cpu.so'].glsl_type_singleton_init_or_ref
    glsl_type_singleton_init_or_ref.restype = None
    glsl_type_singleton_init_or_ref.argtypes = []
except AttributeError:
    pass
try:
    glsl_type_singleton_decref = _libraries['libtinymesa_cpu.so'].glsl_type_singleton_decref
    glsl_type_singleton_decref.restype = None
    glsl_type_singleton_decref.argtypes = []
except AttributeError:
    pass
try:
    encode_type_to_blob = _libraries['libtinymesa_cpu.so'].encode_type_to_blob
    encode_type_to_blob.restype = None
    encode_type_to_blob.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    decode_type_from_blob = _libraries['libtinymesa_cpu.so'].decode_type_from_blob
    decode_type_from_blob.restype = ctypes.POINTER(struct_glsl_type)
    decode_type_from_blob.argtypes = [ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass
glsl_type_size_align_func = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32))
try:
    glsl_base_type_bit_size = _libraries['FIXME_STUB'].glsl_base_type_bit_size
    glsl_base_type_bit_size.restype = ctypes.c_uint32
    glsl_base_type_bit_size.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_base_type_is_16bit = _libraries['FIXME_STUB'].glsl_base_type_is_16bit
    glsl_base_type_is_16bit.restype = ctypes.c_bool
    glsl_base_type_is_16bit.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_base_type_is_64bit = _libraries['FIXME_STUB'].glsl_base_type_is_64bit
    glsl_base_type_is_64bit.restype = ctypes.c_bool
    glsl_base_type_is_64bit.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_base_type_is_integer = _libraries['FIXME_STUB'].glsl_base_type_is_integer
    glsl_base_type_is_integer.restype = ctypes.c_bool
    glsl_base_type_is_integer.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_base_type_is_float = _libraries['FIXME_STUB'].glsl_base_type_is_float
    glsl_base_type_is_float.restype = ctypes.c_bool
    glsl_base_type_is_float.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_base_type_get_bit_size = _libraries['FIXME_STUB'].glsl_base_type_get_bit_size
    glsl_base_type_get_bit_size.restype = ctypes.c_uint32
    glsl_base_type_get_bit_size.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_unsigned_base_type_of = _libraries['FIXME_STUB'].glsl_unsigned_base_type_of
    glsl_unsigned_base_type_of.restype = glsl_base_type
    glsl_unsigned_base_type_of.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_signed_base_type_of = _libraries['FIXME_STUB'].glsl_signed_base_type_of
    glsl_signed_base_type_of.restype = glsl_base_type
    glsl_signed_base_type_of.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_apply_signedness_to_base_type = _libraries['libtinymesa_cpu.so'].glsl_apply_signedness_to_base_type
    glsl_apply_signedness_to_base_type.restype = glsl_base_type
    glsl_apply_signedness_to_base_type.argtypes = [glsl_base_type, ctypes.c_bool]
except AttributeError:
    pass

# values for enumeration 'glsl_sampler_dim'
glsl_sampler_dim__enumvalues = {
    0: 'GLSL_SAMPLER_DIM_1D',
    1: 'GLSL_SAMPLER_DIM_2D',
    2: 'GLSL_SAMPLER_DIM_3D',
    3: 'GLSL_SAMPLER_DIM_CUBE',
    4: 'GLSL_SAMPLER_DIM_RECT',
    5: 'GLSL_SAMPLER_DIM_BUF',
    6: 'GLSL_SAMPLER_DIM_EXTERNAL',
    7: 'GLSL_SAMPLER_DIM_MS',
    8: 'GLSL_SAMPLER_DIM_SUBPASS',
    9: 'GLSL_SAMPLER_DIM_SUBPASS_MS',
}
GLSL_SAMPLER_DIM_1D = 0
GLSL_SAMPLER_DIM_2D = 1
GLSL_SAMPLER_DIM_3D = 2
GLSL_SAMPLER_DIM_CUBE = 3
GLSL_SAMPLER_DIM_RECT = 4
GLSL_SAMPLER_DIM_BUF = 5
GLSL_SAMPLER_DIM_EXTERNAL = 6
GLSL_SAMPLER_DIM_MS = 7
GLSL_SAMPLER_DIM_SUBPASS = 8
GLSL_SAMPLER_DIM_SUBPASS_MS = 9
glsl_sampler_dim = ctypes.c_uint32 # enum
try:
    glsl_get_sampler_dim_coordinate_components = _libraries['libtinymesa_cpu.so'].glsl_get_sampler_dim_coordinate_components
    glsl_get_sampler_dim_coordinate_components.restype = ctypes.c_int32
    glsl_get_sampler_dim_coordinate_components.argtypes = [glsl_sampler_dim]
except AttributeError:
    pass

# values for enumeration 'glsl_matrix_layout'
glsl_matrix_layout__enumvalues = {
    0: 'GLSL_MATRIX_LAYOUT_INHERITED',
    1: 'GLSL_MATRIX_LAYOUT_COLUMN_MAJOR',
    2: 'GLSL_MATRIX_LAYOUT_ROW_MAJOR',
}
GLSL_MATRIX_LAYOUT_INHERITED = 0
GLSL_MATRIX_LAYOUT_COLUMN_MAJOR = 1
GLSL_MATRIX_LAYOUT_ROW_MAJOR = 2
glsl_matrix_layout = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_GLSL_PRECISION_NONE'
c__Ea_GLSL_PRECISION_NONE__enumvalues = {
    0: 'GLSL_PRECISION_NONE',
    1: 'GLSL_PRECISION_HIGH',
    2: 'GLSL_PRECISION_MEDIUM',
    3: 'GLSL_PRECISION_LOW',
}
GLSL_PRECISION_NONE = 0
GLSL_PRECISION_HIGH = 1
GLSL_PRECISION_MEDIUM = 2
GLSL_PRECISION_LOW = 3
c__Ea_GLSL_PRECISION_NONE = ctypes.c_uint32 # enum

# values for enumeration 'glsl_cmat_use'
glsl_cmat_use__enumvalues = {
    0: 'GLSL_CMAT_USE_NONE',
    1: 'GLSL_CMAT_USE_A',
    2: 'GLSL_CMAT_USE_B',
    3: 'GLSL_CMAT_USE_ACCUMULATOR',
}
GLSL_CMAT_USE_NONE = 0
GLSL_CMAT_USE_A = 1
GLSL_CMAT_USE_B = 2
GLSL_CMAT_USE_ACCUMULATOR = 3
glsl_cmat_use = ctypes.c_uint32 # enum
try:
    glsl_get_type_name = _libraries['libtinymesa_cpu.so'].glsl_get_type_name
    glsl_get_type_name.restype = ctypes.POINTER(ctypes.c_char)
    glsl_get_type_name.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_base_type = _libraries['FIXME_STUB'].glsl_get_base_type
    glsl_get_base_type.restype = glsl_base_type
    glsl_get_base_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_bit_size = _libraries['FIXME_STUB'].glsl_get_bit_size
    glsl_get_bit_size.restype = ctypes.c_uint32
    glsl_get_bit_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_boolean = _libraries['FIXME_STUB'].glsl_type_is_boolean
    glsl_type_is_boolean.restype = ctypes.c_bool
    glsl_type_is_boolean.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_sampler = _libraries['FIXME_STUB'].glsl_type_is_sampler
    glsl_type_is_sampler.restype = ctypes.c_bool
    glsl_type_is_sampler.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_texture = _libraries['FIXME_STUB'].glsl_type_is_texture
    glsl_type_is_texture.restype = ctypes.c_bool
    glsl_type_is_texture.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_image = _libraries['FIXME_STUB'].glsl_type_is_image
    glsl_type_is_image.restype = ctypes.c_bool
    glsl_type_is_image.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_atomic_uint = _libraries['FIXME_STUB'].glsl_type_is_atomic_uint
    glsl_type_is_atomic_uint.restype = ctypes.c_bool
    glsl_type_is_atomic_uint.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_struct = _libraries['FIXME_STUB'].glsl_type_is_struct
    glsl_type_is_struct.restype = ctypes.c_bool
    glsl_type_is_struct.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_interface = _libraries['FIXME_STUB'].glsl_type_is_interface
    glsl_type_is_interface.restype = ctypes.c_bool
    glsl_type_is_interface.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_array = _libraries['FIXME_STUB'].glsl_type_is_array
    glsl_type_is_array.restype = ctypes.c_bool
    glsl_type_is_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_cmat = _libraries['FIXME_STUB'].glsl_type_is_cmat
    glsl_type_is_cmat.restype = ctypes.c_bool
    glsl_type_is_cmat.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_void = _libraries['FIXME_STUB'].glsl_type_is_void
    glsl_type_is_void.restype = ctypes.c_bool
    glsl_type_is_void.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_subroutine = _libraries['FIXME_STUB'].glsl_type_is_subroutine
    glsl_type_is_subroutine.restype = ctypes.c_bool
    glsl_type_is_subroutine.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_error = _libraries['FIXME_STUB'].glsl_type_is_error
    glsl_type_is_error.restype = ctypes.c_bool
    glsl_type_is_error.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_double = _libraries['FIXME_STUB'].glsl_type_is_double
    glsl_type_is_double.restype = ctypes.c_bool
    glsl_type_is_double.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_float = _libraries['FIXME_STUB'].glsl_type_is_float
    glsl_type_is_float.restype = ctypes.c_bool
    glsl_type_is_float.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_numeric = _libraries['FIXME_STUB'].glsl_type_is_numeric
    glsl_type_is_numeric.restype = ctypes.c_bool
    glsl_type_is_numeric.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_integer = _libraries['FIXME_STUB'].glsl_type_is_integer
    glsl_type_is_integer.restype = ctypes.c_bool
    glsl_type_is_integer.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_struct_or_ifc = _libraries['FIXME_STUB'].glsl_type_is_struct_or_ifc
    glsl_type_is_struct_or_ifc.restype = ctypes.c_bool
    glsl_type_is_struct_or_ifc.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_packed = _libraries['FIXME_STUB'].glsl_type_is_packed
    glsl_type_is_packed.restype = ctypes.c_bool
    glsl_type_is_packed.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_16bit = _libraries['FIXME_STUB'].glsl_type_is_16bit
    glsl_type_is_16bit.restype = ctypes.c_bool
    glsl_type_is_16bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_32bit = _libraries['FIXME_STUB'].glsl_type_is_32bit
    glsl_type_is_32bit.restype = ctypes.c_bool
    glsl_type_is_32bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_64bit = _libraries['FIXME_STUB'].glsl_type_is_64bit
    glsl_type_is_64bit.restype = ctypes.c_bool
    glsl_type_is_64bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_integer_16 = _libraries['FIXME_STUB'].glsl_type_is_integer_16
    glsl_type_is_integer_16.restype = ctypes.c_bool
    glsl_type_is_integer_16.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_integer_32 = _libraries['FIXME_STUB'].glsl_type_is_integer_32
    glsl_type_is_integer_32.restype = ctypes.c_bool
    glsl_type_is_integer_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_integer_64 = _libraries['FIXME_STUB'].glsl_type_is_integer_64
    glsl_type_is_integer_64.restype = ctypes.c_bool
    glsl_type_is_integer_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_integer_32_64 = _libraries['FIXME_STUB'].glsl_type_is_integer_32_64
    glsl_type_is_integer_32_64.restype = ctypes.c_bool
    glsl_type_is_integer_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_integer_16_32 = _libraries['FIXME_STUB'].glsl_type_is_integer_16_32
    glsl_type_is_integer_16_32.restype = ctypes.c_bool
    glsl_type_is_integer_16_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_integer_16_32_64 = _libraries['FIXME_STUB'].glsl_type_is_integer_16_32_64
    glsl_type_is_integer_16_32_64.restype = ctypes.c_bool
    glsl_type_is_integer_16_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_float_16 = _libraries['FIXME_STUB'].glsl_type_is_float_16
    glsl_type_is_float_16.restype = ctypes.c_bool
    glsl_type_is_float_16.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_float_16_32 = _libraries['FIXME_STUB'].glsl_type_is_float_16_32
    glsl_type_is_float_16_32.restype = ctypes.c_bool
    glsl_type_is_float_16_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_float_16_32_64 = _libraries['FIXME_STUB'].glsl_type_is_float_16_32_64
    glsl_type_is_float_16_32_64.restype = ctypes.c_bool
    glsl_type_is_float_16_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_bfloat_16 = _libraries['FIXME_STUB'].glsl_type_is_bfloat_16
    glsl_type_is_bfloat_16.restype = ctypes.c_bool
    glsl_type_is_bfloat_16.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_e4m3fn = _libraries['FIXME_STUB'].glsl_type_is_e4m3fn
    glsl_type_is_e4m3fn.restype = ctypes.c_bool
    glsl_type_is_e4m3fn.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_e5m2 = _libraries['FIXME_STUB'].glsl_type_is_e5m2
    glsl_type_is_e5m2.restype = ctypes.c_bool
    glsl_type_is_e5m2.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_int_16_32_64 = _libraries['FIXME_STUB'].glsl_type_is_int_16_32_64
    glsl_type_is_int_16_32_64.restype = ctypes.c_bool
    glsl_type_is_int_16_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_uint_16_32_64 = _libraries['FIXME_STUB'].glsl_type_is_uint_16_32_64
    glsl_type_is_uint_16_32_64.restype = ctypes.c_bool
    glsl_type_is_uint_16_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_int_16_32 = _libraries['FIXME_STUB'].glsl_type_is_int_16_32
    glsl_type_is_int_16_32.restype = ctypes.c_bool
    glsl_type_is_int_16_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_uint_16_32 = _libraries['FIXME_STUB'].glsl_type_is_uint_16_32
    glsl_type_is_uint_16_32.restype = ctypes.c_bool
    glsl_type_is_uint_16_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_unsized_array = _libraries['FIXME_STUB'].glsl_type_is_unsized_array
    glsl_type_is_unsized_array.restype = ctypes.c_bool
    glsl_type_is_unsized_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_array_of_arrays = _libraries['FIXME_STUB'].glsl_type_is_array_of_arrays
    glsl_type_is_array_of_arrays.restype = ctypes.c_bool
    glsl_type_is_array_of_arrays.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_bare_sampler = _libraries['FIXME_STUB'].glsl_type_is_bare_sampler
    glsl_type_is_bare_sampler.restype = ctypes.c_bool
    glsl_type_is_bare_sampler.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_vector = _libraries['libtinymesa_cpu.so'].glsl_type_is_vector
    glsl_type_is_vector.restype = ctypes.c_bool
    glsl_type_is_vector.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_scalar = _libraries['libtinymesa_cpu.so'].glsl_type_is_scalar
    glsl_type_is_scalar.restype = ctypes.c_bool
    glsl_type_is_scalar.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_vector_or_scalar = _libraries['libtinymesa_cpu.so'].glsl_type_is_vector_or_scalar
    glsl_type_is_vector_or_scalar.restype = ctypes.c_bool
    glsl_type_is_vector_or_scalar.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_matrix = _libraries['libtinymesa_cpu.so'].glsl_type_is_matrix
    glsl_type_is_matrix.restype = ctypes.c_bool
    glsl_type_is_matrix.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_array_or_matrix = _libraries['libtinymesa_cpu.so'].glsl_type_is_array_or_matrix
    glsl_type_is_array_or_matrix.restype = ctypes.c_bool
    glsl_type_is_array_or_matrix.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_dual_slot = _libraries['libtinymesa_cpu.so'].glsl_type_is_dual_slot
    glsl_type_is_dual_slot.restype = ctypes.c_bool
    glsl_type_is_dual_slot.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_is_leaf = _libraries['libtinymesa_cpu.so'].glsl_type_is_leaf
    glsl_type_is_leaf.restype = ctypes.c_bool
    glsl_type_is_leaf.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_matrix_type_is_row_major = _libraries['FIXME_STUB'].glsl_matrix_type_is_row_major
    glsl_matrix_type_is_row_major.restype = ctypes.c_bool
    glsl_matrix_type_is_row_major.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_sampler_type_is_shadow = _libraries['FIXME_STUB'].glsl_sampler_type_is_shadow
    glsl_sampler_type_is_shadow.restype = ctypes.c_bool
    glsl_sampler_type_is_shadow.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_sampler_type_is_array = _libraries['FIXME_STUB'].glsl_sampler_type_is_array
    glsl_sampler_type_is_array.restype = ctypes.c_bool
    glsl_sampler_type_is_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_struct_type_is_packed = _libraries['FIXME_STUB'].glsl_struct_type_is_packed
    glsl_struct_type_is_packed.restype = ctypes.c_bool
    glsl_struct_type_is_packed.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_bare_type = _libraries['libtinymesa_cpu.so'].glsl_get_bare_type
    glsl_get_bare_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_bare_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_scalar_type = _libraries['libtinymesa_cpu.so'].glsl_get_scalar_type
    glsl_get_scalar_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_scalar_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_base_glsl_type = _libraries['libtinymesa_cpu.so'].glsl_get_base_glsl_type
    glsl_get_base_glsl_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_base_glsl_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_length = _libraries['libtinymesa_cpu.so'].glsl_get_length
    glsl_get_length.restype = ctypes.c_uint32
    glsl_get_length.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_vector_elements = _libraries['FIXME_STUB'].glsl_get_vector_elements
    glsl_get_vector_elements.restype = ctypes.c_uint32
    glsl_get_vector_elements.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_components = _libraries['FIXME_STUB'].glsl_get_components
    glsl_get_components.restype = ctypes.c_uint32
    glsl_get_components.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_matrix_columns = _libraries['FIXME_STUB'].glsl_get_matrix_columns
    glsl_get_matrix_columns.restype = ctypes.c_uint32
    glsl_get_matrix_columns.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_wrap_in_arrays = _libraries['libtinymesa_cpu.so'].glsl_type_wrap_in_arrays
    glsl_type_wrap_in_arrays.restype = ctypes.POINTER(struct_glsl_type)
    glsl_type_wrap_in_arrays.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_array_size = _libraries['FIXME_STUB'].glsl_array_size
    glsl_array_size.restype = ctypes.c_int32
    glsl_array_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_aoa_size = _libraries['libtinymesa_cpu.so'].glsl_get_aoa_size
    glsl_get_aoa_size.restype = ctypes.c_uint32
    glsl_get_aoa_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_array_element = _libraries['libtinymesa_cpu.so'].glsl_get_array_element
    glsl_get_array_element.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_array_element.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_without_array = _libraries['libtinymesa_cpu.so'].glsl_without_array
    glsl_without_array.restype = ctypes.POINTER(struct_glsl_type)
    glsl_without_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_without_array_or_matrix = _libraries['libtinymesa_cpu.so'].glsl_without_array_or_matrix
    glsl_without_array_or_matrix.restype = ctypes.POINTER(struct_glsl_type)
    glsl_without_array_or_matrix.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_cmat_element = _libraries['libtinymesa_cpu.so'].glsl_get_cmat_element
    glsl_get_cmat_element.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_cmat_element.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_cmat_description = _libraries['libtinymesa_cpu.so'].glsl_get_cmat_description
    glsl_get_cmat_description.restype = ctypes.POINTER(struct_glsl_cmat_description)
    glsl_get_cmat_description.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_atomic_size = _libraries['libtinymesa_cpu.so'].glsl_atomic_size
    glsl_atomic_size.restype = ctypes.c_uint32
    glsl_atomic_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_contains_32bit = _libraries['libtinymesa_cpu.so'].glsl_type_contains_32bit
    glsl_type_contains_32bit.restype = ctypes.c_bool
    glsl_type_contains_32bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_contains_64bit = _libraries['libtinymesa_cpu.so'].glsl_type_contains_64bit
    glsl_type_contains_64bit.restype = ctypes.c_bool
    glsl_type_contains_64bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_contains_image = _libraries['libtinymesa_cpu.so'].glsl_type_contains_image
    glsl_type_contains_image.restype = ctypes.c_bool
    glsl_type_contains_image.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_contains_atomic = _libraries['libtinymesa_cpu.so'].glsl_contains_atomic
    glsl_contains_atomic.restype = ctypes.c_bool
    glsl_contains_atomic.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_contains_double = _libraries['libtinymesa_cpu.so'].glsl_contains_double
    glsl_contains_double.restype = ctypes.c_bool
    glsl_contains_double.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_contains_integer = _libraries['libtinymesa_cpu.so'].glsl_contains_integer
    glsl_contains_integer.restype = ctypes.c_bool
    glsl_contains_integer.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_contains_opaque = _libraries['libtinymesa_cpu.so'].glsl_contains_opaque
    glsl_contains_opaque.restype = ctypes.c_bool
    glsl_contains_opaque.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_contains_sampler = _libraries['libtinymesa_cpu.so'].glsl_contains_sampler
    glsl_contains_sampler.restype = ctypes.c_bool
    glsl_contains_sampler.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_contains_array = _libraries['libtinymesa_cpu.so'].glsl_contains_array
    glsl_contains_array.restype = ctypes.c_bool
    glsl_contains_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_contains_subroutine = _libraries['libtinymesa_cpu.so'].glsl_contains_subroutine
    glsl_contains_subroutine.restype = ctypes.c_bool
    glsl_contains_subroutine.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_sampler_dim = _libraries['FIXME_STUB'].glsl_get_sampler_dim
    glsl_get_sampler_dim.restype = glsl_sampler_dim
    glsl_get_sampler_dim.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_sampler_result_type = _libraries['FIXME_STUB'].glsl_get_sampler_result_type
    glsl_get_sampler_result_type.restype = glsl_base_type
    glsl_get_sampler_result_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_sampler_coordinate_components = _libraries['libtinymesa_cpu.so'].glsl_get_sampler_coordinate_components
    glsl_get_sampler_coordinate_components.restype = ctypes.c_int32
    glsl_get_sampler_coordinate_components.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_compare_no_precision = _libraries['libtinymesa_cpu.so'].glsl_type_compare_no_precision
    glsl_type_compare_no_precision.restype = ctypes.c_bool
    glsl_type_compare_no_precision.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_record_compare = _libraries['libtinymesa_cpu.so'].glsl_record_compare
    glsl_record_compare.restype = ctypes.c_bool
    glsl_record_compare.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(struct_glsl_type), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_struct_field = _libraries['libtinymesa_cpu.so'].glsl_get_struct_field
    glsl_get_struct_field.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_struct_field.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_get_struct_field_data = _libraries['libtinymesa_cpu.so'].glsl_get_struct_field_data
    glsl_get_struct_field_data.restype = ctypes.POINTER(struct_glsl_struct_field)
    glsl_get_struct_field_data.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_get_struct_location_offset = _libraries['libtinymesa_cpu.so'].glsl_get_struct_location_offset
    glsl_get_struct_location_offset.restype = ctypes.c_uint32
    glsl_get_struct_location_offset.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_get_field_index = _libraries['libtinymesa_cpu.so'].glsl_get_field_index
    glsl_get_field_index.restype = ctypes.c_int32
    glsl_get_field_index.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    glsl_get_field_type = _libraries['libtinymesa_cpu.so'].glsl_get_field_type
    glsl_get_field_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_field_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    glsl_get_struct_field_offset = _libraries['FIXME_STUB'].glsl_get_struct_field_offset
    glsl_get_struct_field_offset.restype = ctypes.c_int32
    glsl_get_struct_field_offset.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_get_struct_elem_name = _libraries['FIXME_STUB'].glsl_get_struct_elem_name
    glsl_get_struct_elem_name.restype = ctypes.POINTER(ctypes.c_char)
    glsl_get_struct_elem_name.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_void_type = _libraries['FIXME_STUB'].glsl_void_type
    glsl_void_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_void_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_float_type = _libraries['FIXME_STUB'].glsl_float_type
    glsl_float_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_float_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_float16_t_type = _libraries['FIXME_STUB'].glsl_float16_t_type
    glsl_float16_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_float16_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_double_type = _libraries['FIXME_STUB'].glsl_double_type
    glsl_double_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_double_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_vec2_type = _libraries['FIXME_STUB'].glsl_vec2_type
    glsl_vec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_vec2_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_dvec2_type = _libraries['FIXME_STUB'].glsl_dvec2_type
    glsl_dvec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_dvec2_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_uvec2_type = _libraries['FIXME_STUB'].glsl_uvec2_type
    glsl_uvec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uvec2_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_ivec2_type = _libraries['FIXME_STUB'].glsl_ivec2_type
    glsl_ivec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_ivec2_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_bvec2_type = _libraries['FIXME_STUB'].glsl_bvec2_type
    glsl_bvec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bvec2_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_vec4_type = _libraries['FIXME_STUB'].glsl_vec4_type
    glsl_vec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_vec4_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_dvec4_type = _libraries['FIXME_STUB'].glsl_dvec4_type
    glsl_dvec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_dvec4_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_uvec4_type = _libraries['FIXME_STUB'].glsl_uvec4_type
    glsl_uvec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uvec4_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_ivec4_type = _libraries['FIXME_STUB'].glsl_ivec4_type
    glsl_ivec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_ivec4_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_bvec4_type = _libraries['FIXME_STUB'].glsl_bvec4_type
    glsl_bvec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bvec4_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_int_type = _libraries['FIXME_STUB'].glsl_int_type
    glsl_int_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_uint_type = _libraries['FIXME_STUB'].glsl_uint_type
    glsl_uint_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_int64_t_type = _libraries['FIXME_STUB'].glsl_int64_t_type
    glsl_int64_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int64_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_uint64_t_type = _libraries['FIXME_STUB'].glsl_uint64_t_type
    glsl_uint64_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint64_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_int16_t_type = _libraries['FIXME_STUB'].glsl_int16_t_type
    glsl_int16_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int16_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_uint16_t_type = _libraries['FIXME_STUB'].glsl_uint16_t_type
    glsl_uint16_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint16_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_int8_t_type = _libraries['FIXME_STUB'].glsl_int8_t_type
    glsl_int8_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int8_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_uint8_t_type = _libraries['FIXME_STUB'].glsl_uint8_t_type
    glsl_uint8_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint8_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_bool_type = _libraries['FIXME_STUB'].glsl_bool_type
    glsl_bool_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bool_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_atomic_uint_type = _libraries['FIXME_STUB'].glsl_atomic_uint_type
    glsl_atomic_uint_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_atomic_uint_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_bfloat16_t_type = _libraries['FIXME_STUB'].glsl_bfloat16_t_type
    glsl_bfloat16_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bfloat16_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_e4m3fn_t_type = _libraries['FIXME_STUB'].glsl_e4m3fn_t_type
    glsl_e4m3fn_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_e4m3fn_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_e5m2_t_type = _libraries['FIXME_STUB'].glsl_e5m2_t_type
    glsl_e5m2_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_e5m2_t_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_floatN_t_type = _libraries['FIXME_STUB'].glsl_floatN_t_type
    glsl_floatN_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_floatN_t_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_bfloatN_t_type = _libraries['FIXME_STUB'].glsl_bfloatN_t_type
    glsl_bfloatN_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bfloatN_t_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_intN_t_type = _libraries['FIXME_STUB'].glsl_intN_t_type
    glsl_intN_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_intN_t_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_uintN_t_type = _libraries['FIXME_STUB'].glsl_uintN_t_type
    glsl_uintN_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uintN_t_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_vec_type = _libraries['libtinymesa_cpu.so'].glsl_vec_type
    glsl_vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_f16vec_type = _libraries['libtinymesa_cpu.so'].glsl_f16vec_type
    glsl_f16vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_f16vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_bf16vec_type = _libraries['libtinymesa_cpu.so'].glsl_bf16vec_type
    glsl_bf16vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bf16vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_e4m3fnvec_type = _libraries['libtinymesa_cpu.so'].glsl_e4m3fnvec_type
    glsl_e4m3fnvec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_e4m3fnvec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_e5m2vec_type = _libraries['libtinymesa_cpu.so'].glsl_e5m2vec_type
    glsl_e5m2vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_e5m2vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_dvec_type = _libraries['libtinymesa_cpu.so'].glsl_dvec_type
    glsl_dvec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_dvec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_ivec_type = _libraries['libtinymesa_cpu.so'].glsl_ivec_type
    glsl_ivec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_ivec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_uvec_type = _libraries['libtinymesa_cpu.so'].glsl_uvec_type
    glsl_uvec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uvec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_bvec_type = _libraries['libtinymesa_cpu.so'].glsl_bvec_type
    glsl_bvec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bvec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_i64vec_type = _libraries['libtinymesa_cpu.so'].glsl_i64vec_type
    glsl_i64vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_i64vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_u64vec_type = _libraries['libtinymesa_cpu.so'].glsl_u64vec_type
    glsl_u64vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_u64vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_i16vec_type = _libraries['libtinymesa_cpu.so'].glsl_i16vec_type
    glsl_i16vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_i16vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_u16vec_type = _libraries['libtinymesa_cpu.so'].glsl_u16vec_type
    glsl_u16vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_u16vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_i8vec_type = _libraries['libtinymesa_cpu.so'].glsl_i8vec_type
    glsl_i8vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_i8vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_u8vec_type = _libraries['libtinymesa_cpu.so'].glsl_u8vec_type
    glsl_u8vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_u8vec_type.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_simple_explicit_type = _libraries['libtinymesa_cpu.so'].glsl_simple_explicit_type
    glsl_simple_explicit_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_simple_explicit_type.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_simple_type = _libraries['FIXME_STUB'].glsl_simple_type
    glsl_simple_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_simple_type.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_sampler_type = _libraries['libtinymesa_cpu.so'].glsl_sampler_type
    glsl_sampler_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_sampler_type.argtypes = [glsl_sampler_dim, ctypes.c_bool, ctypes.c_bool, glsl_base_type]
except AttributeError:
    pass
try:
    glsl_bare_sampler_type = _libraries['libtinymesa_cpu.so'].glsl_bare_sampler_type
    glsl_bare_sampler_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bare_sampler_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_bare_shadow_sampler_type = _libraries['libtinymesa_cpu.so'].glsl_bare_shadow_sampler_type
    glsl_bare_shadow_sampler_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bare_shadow_sampler_type.argtypes = []
except AttributeError:
    pass
try:
    glsl_texture_type = _libraries['libtinymesa_cpu.so'].glsl_texture_type
    glsl_texture_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_texture_type.argtypes = [glsl_sampler_dim, ctypes.c_bool, glsl_base_type]
except AttributeError:
    pass
try:
    glsl_image_type = _libraries['libtinymesa_cpu.so'].glsl_image_type
    glsl_image_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_image_type.argtypes = [glsl_sampler_dim, ctypes.c_bool, glsl_base_type]
except AttributeError:
    pass
try:
    glsl_array_type = _libraries['libtinymesa_cpu.so'].glsl_array_type
    glsl_array_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_array_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_cmat_type = _libraries['libtinymesa_cpu.so'].glsl_cmat_type
    glsl_cmat_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_cmat_type.argtypes = [ctypes.POINTER(struct_glsl_cmat_description)]
except AttributeError:
    pass
try:
    glsl_struct_type_with_explicit_alignment = _libraries['libtinymesa_cpu.so'].glsl_struct_type_with_explicit_alignment
    glsl_struct_type_with_explicit_alignment.restype = ctypes.POINTER(struct_glsl_type)
    glsl_struct_type_with_explicit_alignment.argtypes = [ctypes.POINTER(struct_glsl_struct_field), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_bool, ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_struct_type = _libraries['FIXME_STUB'].glsl_struct_type
    glsl_struct_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_struct_type.argtypes = [ctypes.POINTER(struct_glsl_struct_field), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_bool]
except AttributeError:
    pass

# values for enumeration 'glsl_interface_packing'
glsl_interface_packing__enumvalues = {
    0: 'GLSL_INTERFACE_PACKING_STD140',
    1: 'GLSL_INTERFACE_PACKING_SHARED',
    2: 'GLSL_INTERFACE_PACKING_PACKED',
    3: 'GLSL_INTERFACE_PACKING_STD430',
}
GLSL_INTERFACE_PACKING_STD140 = 0
GLSL_INTERFACE_PACKING_SHARED = 1
GLSL_INTERFACE_PACKING_PACKED = 2
GLSL_INTERFACE_PACKING_STD430 = 3
glsl_interface_packing = ctypes.c_uint32 # enum
try:
    glsl_interface_type = _libraries['libtinymesa_cpu.so'].glsl_interface_type
    glsl_interface_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_interface_type.argtypes = [ctypes.POINTER(struct_glsl_struct_field), ctypes.c_uint32, glsl_interface_packing, ctypes.c_bool, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    glsl_subroutine_type = _libraries['libtinymesa_cpu.so'].glsl_subroutine_type
    glsl_subroutine_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_subroutine_type.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    glsl_get_row_type = _libraries['libtinymesa_cpu.so'].glsl_get_row_type
    glsl_get_row_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_row_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_column_type = _libraries['libtinymesa_cpu.so'].glsl_get_column_type
    glsl_get_column_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_column_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_explicit_type_for_size_align = _libraries['libtinymesa_cpu.so'].glsl_get_explicit_type_for_size_align
    glsl_get_explicit_type_for_size_align.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_explicit_type_for_size_align.argtypes = [ctypes.POINTER(struct_glsl_type), glsl_type_size_align_func, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    glsl_type_replace_vec3_with_vec4 = _libraries['libtinymesa_cpu.so'].glsl_type_replace_vec3_with_vec4
    glsl_type_replace_vec3_with_vec4.restype = ctypes.POINTER(struct_glsl_type)
    glsl_type_replace_vec3_with_vec4.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_float16_type = _libraries['libtinymesa_cpu.so'].glsl_float16_type
    glsl_float16_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_float16_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_int16_type = _libraries['libtinymesa_cpu.so'].glsl_int16_type
    glsl_int16_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int16_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_uint16_type = _libraries['libtinymesa_cpu.so'].glsl_uint16_type
    glsl_uint16_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint16_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_to_16bit = _libraries['libtinymesa_cpu.so'].glsl_type_to_16bit
    glsl_type_to_16bit.restype = ctypes.POINTER(struct_glsl_type)
    glsl_type_to_16bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_scalar_type = _libraries['FIXME_STUB'].glsl_scalar_type
    glsl_scalar_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_scalar_type.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    glsl_vector_type = _libraries['FIXME_STUB'].glsl_vector_type
    glsl_vector_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_vector_type.argtypes = [glsl_base_type, ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_matrix_type = _libraries['FIXME_STUB'].glsl_matrix_type
    glsl_matrix_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_matrix_type.argtypes = [glsl_base_type, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_explicit_matrix_type = _libraries['FIXME_STUB'].glsl_explicit_matrix_type
    glsl_explicit_matrix_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_explicit_matrix_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32, ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_transposed_type = _libraries['FIXME_STUB'].glsl_transposed_type
    glsl_transposed_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_transposed_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_texture_type_to_sampler = _libraries['FIXME_STUB'].glsl_texture_type_to_sampler
    glsl_texture_type_to_sampler.restype = ctypes.POINTER(struct_glsl_type)
    glsl_texture_type_to_sampler.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_sampler_type_to_texture = _libraries['FIXME_STUB'].glsl_sampler_type_to_texture
    glsl_sampler_type_to_texture.restype = ctypes.POINTER(struct_glsl_type)
    glsl_sampler_type_to_texture.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_replace_vector_type = _libraries['libtinymesa_cpu.so'].glsl_replace_vector_type
    glsl_replace_vector_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_replace_vector_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_channel_type = _libraries['libtinymesa_cpu.so'].glsl_channel_type
    glsl_channel_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_channel_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_mul_type = _libraries['libtinymesa_cpu.so'].glsl_get_mul_type
    glsl_get_mul_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_mul_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_get_sampler_count = _libraries['libtinymesa_cpu.so'].glsl_type_get_sampler_count
    glsl_type_get_sampler_count.restype = ctypes.c_uint32
    glsl_type_get_sampler_count.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_get_texture_count = _libraries['libtinymesa_cpu.so'].glsl_type_get_texture_count
    glsl_type_get_texture_count.restype = ctypes.c_uint32
    glsl_type_get_texture_count.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_get_image_count = _libraries['libtinymesa_cpu.so'].glsl_type_get_image_count
    glsl_type_get_image_count.restype = ctypes.c_uint32
    glsl_type_get_image_count.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_count_vec4_slots = _libraries['libtinymesa_cpu.so'].glsl_count_vec4_slots
    glsl_count_vec4_slots.restype = ctypes.c_uint32
    glsl_count_vec4_slots.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_count_dword_slots = _libraries['libtinymesa_cpu.so'].glsl_count_dword_slots
    glsl_count_dword_slots.restype = ctypes.c_uint32
    glsl_count_dword_slots.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_component_slots = _libraries['libtinymesa_cpu.so'].glsl_get_component_slots
    glsl_get_component_slots.restype = ctypes.c_uint32
    glsl_get_component_slots.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_component_slots_aligned = _libraries['libtinymesa_cpu.so'].glsl_get_component_slots_aligned
    glsl_get_component_slots_aligned.restype = ctypes.c_uint32
    glsl_get_component_slots_aligned.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    glsl_varying_count = _libraries['libtinymesa_cpu.so'].glsl_varying_count
    glsl_varying_count.restype = ctypes.c_uint32
    glsl_varying_count.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_type_uniform_locations = _libraries['libtinymesa_cpu.so'].glsl_type_uniform_locations
    glsl_type_uniform_locations.restype = ctypes.c_uint32
    glsl_type_uniform_locations.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_count_attribute_slots = _libraries['FIXME_STUB'].glsl_count_attribute_slots
    glsl_count_attribute_slots.restype = ctypes.c_uint32
    glsl_count_attribute_slots.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_cl_size = _libraries['libtinymesa_cpu.so'].glsl_get_cl_size
    glsl_get_cl_size.restype = ctypes.c_uint32
    glsl_get_cl_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_cl_alignment = _libraries['libtinymesa_cpu.so'].glsl_get_cl_alignment
    glsl_get_cl_alignment.restype = ctypes.c_uint32
    glsl_get_cl_alignment.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_cl_type_size_align = _libraries['libtinymesa_cpu.so'].glsl_get_cl_type_size_align
    glsl_get_cl_type_size_align.restype = None
    glsl_get_cl_type_size_align.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    glsl_get_internal_ifc_packing = _libraries['libtinymesa_cpu.so'].glsl_get_internal_ifc_packing
    glsl_get_internal_ifc_packing.restype = glsl_interface_packing
    glsl_get_internal_ifc_packing.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_ifc_packing = _libraries['FIXME_STUB'].glsl_get_ifc_packing
    glsl_get_ifc_packing.restype = glsl_interface_packing
    glsl_get_ifc_packing.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_std140_base_alignment = _libraries['libtinymesa_cpu.so'].glsl_get_std140_base_alignment
    glsl_get_std140_base_alignment.restype = ctypes.c_uint32
    glsl_get_std140_base_alignment.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_std140_size = _libraries['libtinymesa_cpu.so'].glsl_get_std140_size
    glsl_get_std140_size.restype = ctypes.c_uint32
    glsl_get_std140_size.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_std430_array_stride = _libraries['libtinymesa_cpu.so'].glsl_get_std430_array_stride
    glsl_get_std430_array_stride.restype = ctypes.c_uint32
    glsl_get_std430_array_stride.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_std430_base_alignment = _libraries['libtinymesa_cpu.so'].glsl_get_std430_base_alignment
    glsl_get_std430_base_alignment.restype = ctypes.c_uint32
    glsl_get_std430_base_alignment.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_std430_size = _libraries['libtinymesa_cpu.so'].glsl_get_std430_size
    glsl_get_std430_size.restype = ctypes.c_uint32
    glsl_get_std430_size.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_explicit_size = _libraries['libtinymesa_cpu.so'].glsl_get_explicit_size
    glsl_get_explicit_size.restype = ctypes.c_uint32
    glsl_get_explicit_size.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_explicit_stride = _libraries['FIXME_STUB'].glsl_get_explicit_stride
    glsl_get_explicit_stride.restype = ctypes.c_uint32
    glsl_get_explicit_stride.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_explicit_alignment = _libraries['FIXME_STUB'].glsl_get_explicit_alignment
    glsl_get_explicit_alignment.restype = ctypes.c_uint32
    glsl_get_explicit_alignment.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    glsl_get_explicit_std140_type = _libraries['libtinymesa_cpu.so'].glsl_get_explicit_std140_type
    glsl_get_explicit_std140_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_explicit_std140_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_explicit_std430_type = _libraries['libtinymesa_cpu.so'].glsl_get_explicit_std430_type
    glsl_get_explicit_std430_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_explicit_std430_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_get_explicit_interface_type = _libraries['FIXME_STUB'].glsl_get_explicit_interface_type
    glsl_get_explicit_interface_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_explicit_interface_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except AttributeError:
    pass
try:
    glsl_size_align_handle_array_and_structs = _libraries['libtinymesa_cpu.so'].glsl_size_align_handle_array_and_structs
    glsl_size_align_handle_array_and_structs.restype = None
    glsl_size_align_handle_array_and_structs.argtypes = [ctypes.POINTER(struct_glsl_type), glsl_type_size_align_func, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    glsl_get_natural_size_align_bytes = _libraries['libtinymesa_cpu.so'].glsl_get_natural_size_align_bytes
    glsl_get_natural_size_align_bytes.restype = None
    glsl_get_natural_size_align_bytes.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    glsl_get_word_size_align_bytes = _libraries['libtinymesa_cpu.so'].glsl_get_word_size_align_bytes
    glsl_get_word_size_align_bytes.restype = None
    glsl_get_word_size_align_bytes.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    glsl_get_vec4_size_align_bytes = _libraries['libtinymesa_cpu.so'].glsl_get_vec4_size_align_bytes
    glsl_get_vec4_size_align_bytes.restype = None
    glsl_get_vec4_size_align_bytes.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    ralloc_context = _libraries['libtinymesa_cpu.so'].ralloc_context
    ralloc_context.restype = ctypes.POINTER(None)
    ralloc_context.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    ralloc_size = _libraries['libtinymesa_cpu.so'].ralloc_size
    ralloc_size.restype = ctypes.POINTER(None)
    ralloc_size.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    rzalloc_size = _libraries['libtinymesa_cpu.so'].rzalloc_size
    rzalloc_size.restype = ctypes.POINTER(None)
    rzalloc_size.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    reralloc_size = _libraries['libtinymesa_cpu.so'].reralloc_size
    reralloc_size.restype = ctypes.POINTER(None)
    reralloc_size.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    rerzalloc_size = _libraries['libtinymesa_cpu.so'].rerzalloc_size
    rerzalloc_size.restype = ctypes.POINTER(None)
    rerzalloc_size.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t]
except AttributeError:
    pass
try:
    ralloc_array_size = _libraries['libtinymesa_cpu.so'].ralloc_array_size
    ralloc_array_size.restype = ctypes.POINTER(None)
    ralloc_array_size.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    rzalloc_array_size = _libraries['libtinymesa_cpu.so'].rzalloc_array_size
    rzalloc_array_size.restype = ctypes.POINTER(None)
    rzalloc_array_size.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    reralloc_array_size = _libraries['libtinymesa_cpu.so'].reralloc_array_size
    reralloc_array_size.restype = ctypes.POINTER(None)
    reralloc_array_size.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    rerzalloc_array_size = _libraries['libtinymesa_cpu.so'].rerzalloc_array_size
    rerzalloc_array_size.restype = ctypes.POINTER(None)
    rerzalloc_array_size.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    ralloc_free = _libraries['libtinymesa_cpu.so'].ralloc_free
    ralloc_free.restype = None
    ralloc_free.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    ralloc_steal = _libraries['libtinymesa_cpu.so'].ralloc_steal
    ralloc_steal.restype = None
    ralloc_steal.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    ralloc_adopt = _libraries['libtinymesa_cpu.so'].ralloc_adopt
    ralloc_adopt.restype = None
    ralloc_adopt.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    ralloc_parent = _libraries['libtinymesa_cpu.so'].ralloc_parent
    ralloc_parent.restype = ctypes.POINTER(None)
    ralloc_parent.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    ralloc_set_destructor = _libraries['libtinymesa_cpu.so'].ralloc_set_destructor
    ralloc_set_destructor.restype = None
    ralloc_set_destructor.argtypes = [ctypes.POINTER(None), ctypes.CFUNCTYPE(None, ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    ralloc_memdup = _libraries['libtinymesa_cpu.so'].ralloc_memdup
    ralloc_memdup.restype = ctypes.POINTER(None)
    ralloc_memdup.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    ralloc_strdup = _libraries['libtinymesa_cpu.so'].ralloc_strdup
    ralloc_strdup.restype = ctypes.POINTER(ctypes.c_char)
    ralloc_strdup.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    ralloc_strndup = _libraries['libtinymesa_cpu.so'].ralloc_strndup
    ralloc_strndup.restype = ctypes.POINTER(ctypes.c_char)
    ralloc_strndup.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    ralloc_strcat = _libraries['libtinymesa_cpu.so'].ralloc_strcat
    ralloc_strcat.restype = ctypes.c_bool
    ralloc_strcat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    ralloc_strncat = _libraries['libtinymesa_cpu.so'].ralloc_strncat
    ralloc_strncat.restype = ctypes.c_bool
    ralloc_strncat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    ralloc_str_append = _libraries['libtinymesa_cpu.so'].ralloc_str_append
    ralloc_str_append.restype = ctypes.c_bool
    ralloc_str_append.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), size_t, size_t]
except AttributeError:
    pass
try:
    ralloc_asprintf = _libraries['libtinymesa_cpu.so'].ralloc_asprintf
    ralloc_asprintf.restype = ctypes.POINTER(ctypes.c_char)
    ralloc_asprintf.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
class struct___va_list_tag(Structure):
    pass

struct___va_list_tag._pack_ = 1 # source:False
struct___va_list_tag._fields_ = [
    ('gp_offset', ctypes.c_uint32),
    ('fp_offset', ctypes.c_uint32),
    ('overflow_arg_area', ctypes.POINTER(None)),
    ('reg_save_area', ctypes.POINTER(None)),
]

va_list = struct___va_list_tag * 1
try:
    ralloc_vasprintf = _libraries['libtinymesa_cpu.so'].ralloc_vasprintf
    ralloc_vasprintf.restype = ctypes.POINTER(ctypes.c_char)
    ralloc_vasprintf.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError:
    pass
try:
    ralloc_asprintf_rewrite_tail = _libraries['libtinymesa_cpu.so'].ralloc_asprintf_rewrite_tail
    ralloc_asprintf_rewrite_tail.restype = ctypes.c_bool
    ralloc_asprintf_rewrite_tail.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    ralloc_vasprintf_rewrite_tail = _libraries['libtinymesa_cpu.so'].ralloc_vasprintf_rewrite_tail
    ralloc_vasprintf_rewrite_tail.restype = ctypes.c_bool
    ralloc_vasprintf_rewrite_tail.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError:
    pass
try:
    ralloc_asprintf_append = _libraries['libtinymesa_cpu.so'].ralloc_asprintf_append
    ralloc_asprintf_append.restype = ctypes.c_bool
    ralloc_asprintf_append.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    ralloc_vasprintf_append = _libraries['libtinymesa_cpu.so'].ralloc_vasprintf_append
    ralloc_vasprintf_append.restype = ctypes.c_bool
    ralloc_vasprintf_append.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError:
    pass
try:
    ralloc_total_size = _libraries['libtinymesa_cpu.so'].ralloc_total_size
    ralloc_total_size.restype = size_t
    ralloc_total_size.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_gc_ctx(Structure):
    pass

gc_ctx = struct_gc_ctx
try:
    gc_context = _libraries['libtinymesa_cpu.so'].gc_context
    gc_context.restype = ctypes.POINTER(struct_gc_ctx)
    gc_context.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    gc_alloc_size = _libraries['libtinymesa_cpu.so'].gc_alloc_size
    gc_alloc_size.restype = ctypes.POINTER(None)
    gc_alloc_size.argtypes = [ctypes.POINTER(struct_gc_ctx), size_t, size_t]
except AttributeError:
    pass
try:
    gc_zalloc_size = _libraries['libtinymesa_cpu.so'].gc_zalloc_size
    gc_zalloc_size.restype = ctypes.POINTER(None)
    gc_zalloc_size.argtypes = [ctypes.POINTER(struct_gc_ctx), size_t, size_t]
except AttributeError:
    pass
try:
    gc_free = _libraries['libtinymesa_cpu.so'].gc_free
    gc_free.restype = None
    gc_free.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    gc_get_context = _libraries['libtinymesa_cpu.so'].gc_get_context
    gc_get_context.restype = ctypes.POINTER(struct_gc_ctx)
    gc_get_context.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    gc_sweep_start = _libraries['libtinymesa_cpu.so'].gc_sweep_start
    gc_sweep_start.restype = None
    gc_sweep_start.argtypes = [ctypes.POINTER(struct_gc_ctx)]
except AttributeError:
    pass
try:
    gc_mark_live = _libraries['libtinymesa_cpu.so'].gc_mark_live
    gc_mark_live.restype = None
    gc_mark_live.argtypes = [ctypes.POINTER(struct_gc_ctx), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    gc_sweep_end = _libraries['libtinymesa_cpu.so'].gc_sweep_end
    gc_sweep_end.restype = None
    gc_sweep_end.argtypes = [ctypes.POINTER(struct_gc_ctx)]
except AttributeError:
    pass
class struct_linear_ctx(Structure):
    pass

linear_ctx = struct_linear_ctx
try:
    linear_alloc_child = _libraries['libtinymesa_cpu.so'].linear_alloc_child
    linear_alloc_child.restype = ctypes.POINTER(None)
    linear_alloc_child.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.c_uint32]
except AttributeError:
    pass
class struct_c__SA_linear_opts(Structure):
    pass

struct_c__SA_linear_opts._pack_ = 1 # source:False
struct_c__SA_linear_opts._fields_ = [
    ('min_buffer_size', ctypes.c_uint32),
]

linear_opts = struct_c__SA_linear_opts
try:
    linear_context = _libraries['libtinymesa_cpu.so'].linear_context
    linear_context.restype = ctypes.POINTER(struct_linear_ctx)
    linear_context.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    linear_context_with_opts = _libraries['libtinymesa_cpu.so'].linear_context_with_opts
    linear_context_with_opts.restype = ctypes.POINTER(struct_linear_ctx)
    linear_context_with_opts.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_c__SA_linear_opts)]
except AttributeError:
    pass
try:
    linear_zalloc_child = _libraries['libtinymesa_cpu.so'].linear_zalloc_child
    linear_zalloc_child.restype = ctypes.POINTER(None)
    linear_zalloc_child.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.c_uint32]
except AttributeError:
    pass
try:
    linear_free_context = _libraries['libtinymesa_cpu.so'].linear_free_context
    linear_free_context.restype = None
    linear_free_context.argtypes = [ctypes.POINTER(struct_linear_ctx)]
except AttributeError:
    pass
try:
    ralloc_steal_linear_context = _libraries['libtinymesa_cpu.so'].ralloc_steal_linear_context
    ralloc_steal_linear_context.restype = None
    ralloc_steal_linear_context.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_linear_ctx)]
except AttributeError:
    pass
try:
    ralloc_parent_of_linear_context = _libraries['libtinymesa_cpu.so'].ralloc_parent_of_linear_context
    ralloc_parent_of_linear_context.restype = ctypes.POINTER(None)
    ralloc_parent_of_linear_context.argtypes = [ctypes.POINTER(struct_linear_ctx)]
except AttributeError:
    pass
try:
    linear_alloc_child_array = _libraries['libtinymesa_cpu.so'].linear_alloc_child_array
    linear_alloc_child_array.restype = ctypes.POINTER(None)
    linear_alloc_child_array.argtypes = [ctypes.POINTER(struct_linear_ctx), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    linear_zalloc_child_array = _libraries['libtinymesa_cpu.so'].linear_zalloc_child_array
    linear_zalloc_child_array.restype = ctypes.POINTER(None)
    linear_zalloc_child_array.argtypes = [ctypes.POINTER(struct_linear_ctx), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    linear_strdup = _libraries['libtinymesa_cpu.so'].linear_strdup
    linear_strdup.restype = ctypes.POINTER(ctypes.c_char)
    linear_strdup.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    linear_asprintf = _libraries['libtinymesa_cpu.so'].linear_asprintf
    linear_asprintf.restype = ctypes.POINTER(ctypes.c_char)
    linear_asprintf.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    linear_vasprintf = _libraries['libtinymesa_cpu.so'].linear_vasprintf
    linear_vasprintf.restype = ctypes.POINTER(ctypes.c_char)
    linear_vasprintf.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError:
    pass
try:
    linear_asprintf_append = _libraries['libtinymesa_cpu.so'].linear_asprintf_append
    linear_asprintf_append.restype = ctypes.c_bool
    linear_asprintf_append.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    linear_vasprintf_append = _libraries['libtinymesa_cpu.so'].linear_vasprintf_append
    linear_vasprintf_append.restype = ctypes.c_bool
    linear_vasprintf_append.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError:
    pass
try:
    linear_asprintf_rewrite_tail = _libraries['libtinymesa_cpu.so'].linear_asprintf_rewrite_tail
    linear_asprintf_rewrite_tail.restype = ctypes.c_bool
    linear_asprintf_rewrite_tail.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    linear_vasprintf_rewrite_tail = _libraries['libtinymesa_cpu.so'].linear_vasprintf_rewrite_tail
    linear_vasprintf_rewrite_tail.restype = ctypes.c_bool
    linear_vasprintf_rewrite_tail.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError:
    pass
try:
    linear_strcat = _libraries['libtinymesa_cpu.so'].linear_strcat
    linear_strcat.restype = ctypes.c_bool
    linear_strcat.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass

# values for enumeration 'c__Ea_RALLOC_PRINT_INFO_SUMMARY_ONLY'
c__Ea_RALLOC_PRINT_INFO_SUMMARY_ONLY__enumvalues = {
    1: 'RALLOC_PRINT_INFO_SUMMARY_ONLY',
}
RALLOC_PRINT_INFO_SUMMARY_ONLY = 1
c__Ea_RALLOC_PRINT_INFO_SUMMARY_ONLY = ctypes.c_uint32 # enum
class struct__IO_FILE(Structure):
    pass

class struct__IO_marker(Structure):
    pass

class struct__IO_codecvt(Structure):
    pass

class struct__IO_wide_data(Structure):
    pass

struct__IO_FILE._pack_ = 1 # source:False
struct__IO_FILE._fields_ = [
    ('_flags', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_IO_read_ptr', ctypes.POINTER(ctypes.c_char)),
    ('_IO_read_end', ctypes.POINTER(ctypes.c_char)),
    ('_IO_read_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_write_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_write_ptr', ctypes.POINTER(ctypes.c_char)),
    ('_IO_write_end', ctypes.POINTER(ctypes.c_char)),
    ('_IO_buf_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_buf_end', ctypes.POINTER(ctypes.c_char)),
    ('_IO_save_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_backup_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_save_end', ctypes.POINTER(ctypes.c_char)),
    ('_markers', ctypes.POINTER(struct__IO_marker)),
    ('_chain', ctypes.POINTER(struct__IO_FILE)),
    ('_fileno', ctypes.c_int32),
    ('_flags2', ctypes.c_int32),
    ('_old_offset', ctypes.c_int64),
    ('_cur_column', ctypes.c_uint16),
    ('_vtable_offset', ctypes.c_byte),
    ('_shortbuf', ctypes.c_char * 1),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('_lock', ctypes.POINTER(None)),
    ('_offset', ctypes.c_int64),
    ('_codecvt', ctypes.POINTER(struct__IO_codecvt)),
    ('_wide_data', ctypes.POINTER(struct__IO_wide_data)),
    ('_freeres_list', ctypes.POINTER(struct__IO_FILE)),
    ('_freeres_buf', ctypes.POINTER(None)),
    ('__pad5', ctypes.c_uint64),
    ('_mode', ctypes.c_int32),
    ('_unused2', ctypes.c_char * 20),
]

try:
    ralloc_print_info = _libraries['libtinymesa_cpu.so'].ralloc_print_info
    ralloc_print_info.restype = None
    ralloc_print_info.argtypes = [ctypes.POINTER(struct__IO_FILE), ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_int64_options'
c__EA_nir_lower_int64_options__enumvalues = {
    1: 'nir_lower_imul64',
    2: 'nir_lower_isign64',
    4: 'nir_lower_divmod64',
    8: 'nir_lower_imul_high64',
    16: 'nir_lower_bcsel64',
    32: 'nir_lower_icmp64',
    64: 'nir_lower_iadd64',
    128: 'nir_lower_iabs64',
    256: 'nir_lower_ineg64',
    512: 'nir_lower_logic64',
    1024: 'nir_lower_minmax64',
    2048: 'nir_lower_shift64',
    4096: 'nir_lower_imul_2x32_64',
    8192: 'nir_lower_extract64',
    16384: 'nir_lower_ufind_msb64',
    32768: 'nir_lower_bit_count64',
    65536: 'nir_lower_subgroup_shuffle64',
    131072: 'nir_lower_scan_reduce_bitwise64',
    262144: 'nir_lower_scan_reduce_iadd64',
    524288: 'nir_lower_vote_ieq64',
    1048576: 'nir_lower_usub_sat64',
    2097152: 'nir_lower_iadd_sat64',
    4194304: 'nir_lower_find_lsb64',
    8388608: 'nir_lower_conv64',
    16777216: 'nir_lower_uadd_sat64',
    33554432: 'nir_lower_iadd3_64',
    67108864: 'nir_lower_bitfield_reverse64',
    134217728: 'nir_lower_bitfield_extract64',
}
nir_lower_imul64 = 1
nir_lower_isign64 = 2
nir_lower_divmod64 = 4
nir_lower_imul_high64 = 8
nir_lower_bcsel64 = 16
nir_lower_icmp64 = 32
nir_lower_iadd64 = 64
nir_lower_iabs64 = 128
nir_lower_ineg64 = 256
nir_lower_logic64 = 512
nir_lower_minmax64 = 1024
nir_lower_shift64 = 2048
nir_lower_imul_2x32_64 = 4096
nir_lower_extract64 = 8192
nir_lower_ufind_msb64 = 16384
nir_lower_bit_count64 = 32768
nir_lower_subgroup_shuffle64 = 65536
nir_lower_scan_reduce_bitwise64 = 131072
nir_lower_scan_reduce_iadd64 = 262144
nir_lower_vote_ieq64 = 524288
nir_lower_usub_sat64 = 1048576
nir_lower_iadd_sat64 = 2097152
nir_lower_find_lsb64 = 4194304
nir_lower_conv64 = 8388608
nir_lower_uadd_sat64 = 16777216
nir_lower_iadd3_64 = 33554432
nir_lower_bitfield_reverse64 = 67108864
nir_lower_bitfield_extract64 = 134217728
c__EA_nir_lower_int64_options = ctypes.c_uint32 # enum
nir_lower_int64_options = c__EA_nir_lower_int64_options
nir_lower_int64_options__enumvalues = c__EA_nir_lower_int64_options__enumvalues

# values for enumeration 'c__EA_nir_lower_doubles_options'
c__EA_nir_lower_doubles_options__enumvalues = {
    1: 'nir_lower_drcp',
    2: 'nir_lower_dsqrt',
    4: 'nir_lower_drsq',
    8: 'nir_lower_dtrunc',
    16: 'nir_lower_dfloor',
    32: 'nir_lower_dceil',
    64: 'nir_lower_dfract',
    128: 'nir_lower_dround_even',
    256: 'nir_lower_dmod',
    512: 'nir_lower_dsub',
    1024: 'nir_lower_ddiv',
    2048: 'nir_lower_dsign',
    4096: 'nir_lower_dminmax',
    8192: 'nir_lower_dsat',
    16384: 'nir_lower_fp64_full_software',
}
nir_lower_drcp = 1
nir_lower_dsqrt = 2
nir_lower_drsq = 4
nir_lower_dtrunc = 8
nir_lower_dfloor = 16
nir_lower_dceil = 32
nir_lower_dfract = 64
nir_lower_dround_even = 128
nir_lower_dmod = 256
nir_lower_dsub = 512
nir_lower_ddiv = 1024
nir_lower_dsign = 2048
nir_lower_dminmax = 4096
nir_lower_dsat = 8192
nir_lower_fp64_full_software = 16384
c__EA_nir_lower_doubles_options = ctypes.c_uint32 # enum
nir_lower_doubles_options = c__EA_nir_lower_doubles_options
nir_lower_doubles_options__enumvalues = c__EA_nir_lower_doubles_options__enumvalues

# values for enumeration 'c__EA_nir_divergence_options'
c__EA_nir_divergence_options__enumvalues = {
    1: 'nir_divergence_single_prim_per_subgroup',
    2: 'nir_divergence_single_patch_per_tcs_subgroup',
    4: 'nir_divergence_single_patch_per_tes_subgroup',
    8: 'nir_divergence_view_index_uniform',
    16: 'nir_divergence_single_frag_shading_rate_per_subgroup',
    32: 'nir_divergence_multiple_workgroup_per_compute_subgroup',
    64: 'nir_divergence_shader_record_ptr_uniform',
    128: 'nir_divergence_uniform_load_tears',
    256: 'nir_divergence_ignore_undef_if_phi_srcs',
}
nir_divergence_single_prim_per_subgroup = 1
nir_divergence_single_patch_per_tcs_subgroup = 2
nir_divergence_single_patch_per_tes_subgroup = 4
nir_divergence_view_index_uniform = 8
nir_divergence_single_frag_shading_rate_per_subgroup = 16
nir_divergence_multiple_workgroup_per_compute_subgroup = 32
nir_divergence_shader_record_ptr_uniform = 64
nir_divergence_uniform_load_tears = 128
nir_divergence_ignore_undef_if_phi_srcs = 256
c__EA_nir_divergence_options = ctypes.c_uint32 # enum
nir_divergence_options = c__EA_nir_divergence_options
nir_divergence_options__enumvalues = c__EA_nir_divergence_options__enumvalues
class struct_nir_instr(Structure):
    pass

class struct_nir_block(Structure):
    pass

class struct_exec_node(Structure):
    pass

struct_exec_node._pack_ = 1 # source:False
struct_exec_node._fields_ = [
    ('next', ctypes.POINTER(struct_exec_node)),
    ('prev', ctypes.POINTER(struct_exec_node)),
]

struct_nir_instr._pack_ = 1 # source:False
struct_nir_instr._fields_ = [
    ('node', struct_exec_node),
    ('block', ctypes.POINTER(struct_nir_block)),
    ('type', ctypes.c_ubyte),
    ('pass_flags', ctypes.c_ubyte),
    ('has_debug_info', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('index', ctypes.c_uint32),
]

class struct_set(Structure):
    pass

class struct_nir_cf_node(Structure):
    pass


# values for enumeration 'c__EA_nir_cf_node_type'
c__EA_nir_cf_node_type__enumvalues = {
    0: 'nir_cf_node_block',
    1: 'nir_cf_node_if',
    2: 'nir_cf_node_loop',
    3: 'nir_cf_node_function',
}
nir_cf_node_block = 0
nir_cf_node_if = 1
nir_cf_node_loop = 2
nir_cf_node_function = 3
c__EA_nir_cf_node_type = ctypes.c_uint32 # enum
struct_nir_cf_node._pack_ = 1 # source:False
struct_nir_cf_node._fields_ = [
    ('node', struct_exec_node),
    ('type', c__EA_nir_cf_node_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('parent', ctypes.POINTER(struct_nir_cf_node)),
]

class struct_exec_list(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('head_sentinel', struct_exec_node),
    ('tail_sentinel', struct_exec_node),
     ]

struct_nir_block._pack_ = 1 # source:False
struct_nir_block._fields_ = [
    ('cf_node', struct_nir_cf_node),
    ('instr_list', struct_exec_list),
    ('index', ctypes.c_uint32),
    ('divergent', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('successors', ctypes.POINTER(struct_nir_block) * 2),
    ('predecessors', ctypes.POINTER(struct_set)),
    ('imm_dom', ctypes.POINTER(struct_nir_block)),
    ('num_dom_children', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dom_children', ctypes.POINTER(ctypes.POINTER(struct_nir_block))),
    ('dom_frontier', ctypes.POINTER(struct_set)),
    ('dom_pre_index', ctypes.c_uint32),
    ('dom_post_index', ctypes.c_uint32),
    ('start_ip', ctypes.c_uint32),
    ('end_ip', ctypes.c_uint32),
    ('live_in', ctypes.POINTER(ctypes.c_uint32)),
    ('live_out', ctypes.POINTER(ctypes.c_uint32)),
]

class struct_set_entry(Structure):
    pass

struct_set._pack_ = 1 # source:False
struct_set._fields_ = [
    ('mem_ctx', ctypes.POINTER(None)),
    ('table', ctypes.POINTER(struct_set_entry)),
    ('key_hash_function', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(None))),
    ('key_equals_function', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(None), ctypes.POINTER(None))),
    ('size', ctypes.c_uint32),
    ('rehash', ctypes.c_uint32),
    ('size_magic', ctypes.c_uint64),
    ('rehash_magic', ctypes.c_uint64),
    ('max_entries', ctypes.c_uint32),
    ('size_index', ctypes.c_uint32),
    ('entries', ctypes.c_uint32),
    ('deleted_entries', ctypes.c_uint32),
]

struct_set_entry._pack_ = 1 # source:False
struct_set_entry._fields_ = [
    ('hash', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('key', ctypes.POINTER(None)),
]

nir_instr_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))

# values for enumeration 'c__EA_nir_io_options'
c__EA_nir_io_options__enumvalues = {
    1: 'nir_io_has_flexible_input_interpolation_except_flat',
    2: 'nir_io_dont_use_pos_for_non_fs_varyings',
    4: 'nir_io_16bit_input_output_support',
    8: 'nir_io_mediump_is_32bit',
    16: 'nir_io_prefer_scalar_fs_inputs',
    32: 'nir_io_mix_convergent_flat_with_interpolated',
    64: 'nir_io_vectorizer_ignores_types',
    128: 'nir_io_always_interpolate_convergent_fs_inputs',
    256: 'nir_io_compaction_rotates_color_channels',
    512: 'nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups',
    1024: 'nir_io_radv_intrinsic_component_workaround',
    65536: 'nir_io_has_intrinsics',
    131072: 'nir_io_separate_clip_cull_distance_arrays',
}
nir_io_has_flexible_input_interpolation_except_flat = 1
nir_io_dont_use_pos_for_non_fs_varyings = 2
nir_io_16bit_input_output_support = 4
nir_io_mediump_is_32bit = 8
nir_io_prefer_scalar_fs_inputs = 16
nir_io_mix_convergent_flat_with_interpolated = 32
nir_io_vectorizer_ignores_types = 64
nir_io_always_interpolate_convergent_fs_inputs = 128
nir_io_compaction_rotates_color_channels = 256
nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups = 512
nir_io_radv_intrinsic_component_workaround = 1024
nir_io_has_intrinsics = 65536
nir_io_separate_clip_cull_distance_arrays = 131072
c__EA_nir_io_options = ctypes.c_uint32 # enum
nir_io_options = c__EA_nir_io_options
nir_io_options__enumvalues = c__EA_nir_io_options__enumvalues

# values for enumeration 'c__EA_nir_lower_packing_op'
c__EA_nir_lower_packing_op__enumvalues = {
    0: 'nir_lower_packing_op_pack_64_2x32',
    1: 'nir_lower_packing_op_unpack_64_2x32',
    2: 'nir_lower_packing_op_pack_64_4x16',
    3: 'nir_lower_packing_op_unpack_64_4x16',
    4: 'nir_lower_packing_op_pack_32_2x16',
    5: 'nir_lower_packing_op_unpack_32_2x16',
    6: 'nir_lower_packing_op_pack_32_4x8',
    7: 'nir_lower_packing_op_unpack_32_4x8',
    8: 'nir_lower_packing_num_ops',
}
nir_lower_packing_op_pack_64_2x32 = 0
nir_lower_packing_op_unpack_64_2x32 = 1
nir_lower_packing_op_pack_64_4x16 = 2
nir_lower_packing_op_unpack_64_4x16 = 3
nir_lower_packing_op_pack_32_2x16 = 4
nir_lower_packing_op_unpack_32_2x16 = 5
nir_lower_packing_op_pack_32_4x8 = 6
nir_lower_packing_op_unpack_32_4x8 = 7
nir_lower_packing_num_ops = 8
c__EA_nir_lower_packing_op = ctypes.c_uint32 # enum
nir_lower_packing_op = c__EA_nir_lower_packing_op
nir_lower_packing_op__enumvalues = c__EA_nir_lower_packing_op__enumvalues
class struct_nir_shader_compiler_options(Structure):
    pass


# values for enumeration 'c__EA_nir_variable_mode'
c__EA_nir_variable_mode__enumvalues = {
    1: 'nir_var_system_value',
    2: 'nir_var_uniform',
    4: 'nir_var_shader_in',
    8: 'nir_var_shader_out',
    16: 'nir_var_image',
    32: 'nir_var_shader_call_data',
    64: 'nir_var_ray_hit_attrib',
    128: 'nir_var_mem_ubo',
    256: 'nir_var_mem_push_const',
    512: 'nir_var_mem_ssbo',
    1024: 'nir_var_mem_constant',
    2048: 'nir_var_mem_task_payload',
    4096: 'nir_var_mem_node_payload',
    8192: 'nir_var_mem_node_payload_in',
    16384: 'nir_var_function_in',
    32768: 'nir_var_function_out',
    65536: 'nir_var_function_inout',
    131072: 'nir_var_shader_temp',
    262144: 'nir_var_function_temp',
    524288: 'nir_var_mem_shared',
    1048576: 'nir_var_mem_global',
    1966080: 'nir_var_mem_generic',
    1159: 'nir_var_read_only_modes',
    1969033: 'nir_var_vec_indexable_modes',
    21: 'nir_num_variable_modes',
    2097151: 'nir_var_all',
}
nir_var_system_value = 1
nir_var_uniform = 2
nir_var_shader_in = 4
nir_var_shader_out = 8
nir_var_image = 16
nir_var_shader_call_data = 32
nir_var_ray_hit_attrib = 64
nir_var_mem_ubo = 128
nir_var_mem_push_const = 256
nir_var_mem_ssbo = 512
nir_var_mem_constant = 1024
nir_var_mem_task_payload = 2048
nir_var_mem_node_payload = 4096
nir_var_mem_node_payload_in = 8192
nir_var_function_in = 16384
nir_var_function_out = 32768
nir_var_function_inout = 65536
nir_var_shader_temp = 131072
nir_var_function_temp = 262144
nir_var_mem_shared = 524288
nir_var_mem_global = 1048576
nir_var_mem_generic = 1966080
nir_var_read_only_modes = 1159
nir_var_vec_indexable_modes = 1969033
nir_num_variable_modes = 21
nir_var_all = 2097151
c__EA_nir_variable_mode = ctypes.c_uint32 # enum
class struct_nir_shader(Structure):
    pass

struct_nir_shader_compiler_options._pack_ = 1 # source:False
struct_nir_shader_compiler_options._fields_ = [
    ('lower_fdiv', ctypes.c_bool),
    ('lower_ffma16', ctypes.c_bool),
    ('lower_ffma32', ctypes.c_bool),
    ('lower_ffma64', ctypes.c_bool),
    ('fuse_ffma16', ctypes.c_bool),
    ('fuse_ffma32', ctypes.c_bool),
    ('fuse_ffma64', ctypes.c_bool),
    ('lower_flrp16', ctypes.c_bool),
    ('lower_flrp32', ctypes.c_bool),
    ('lower_flrp64', ctypes.c_bool),
    ('lower_fpow', ctypes.c_bool),
    ('lower_fsat', ctypes.c_bool),
    ('lower_fsqrt', ctypes.c_bool),
    ('lower_sincos', ctypes.c_bool),
    ('lower_fmod', ctypes.c_bool),
    ('lower_bitfield_extract8', ctypes.c_bool),
    ('lower_bitfield_extract16', ctypes.c_bool),
    ('lower_bitfield_extract', ctypes.c_bool),
    ('lower_bitfield_insert', ctypes.c_bool),
    ('lower_bitfield_reverse', ctypes.c_bool),
    ('lower_bit_count', ctypes.c_bool),
    ('lower_ifind_msb', ctypes.c_bool),
    ('lower_ufind_msb', ctypes.c_bool),
    ('lower_find_lsb', ctypes.c_bool),
    ('lower_uadd_carry', ctypes.c_bool),
    ('lower_usub_borrow', ctypes.c_bool),
    ('lower_mul_high', ctypes.c_bool),
    ('lower_mul_high16', ctypes.c_bool),
    ('lower_fneg', ctypes.c_bool),
    ('lower_ineg', ctypes.c_bool),
    ('lower_fisnormal', ctypes.c_bool),
    ('lower_scmp', ctypes.c_bool),
    ('lower_vector_cmp', ctypes.c_bool),
    ('lower_bitops', ctypes.c_bool),
    ('lower_isign', ctypes.c_bool),
    ('lower_fsign', ctypes.c_bool),
    ('lower_iabs', ctypes.c_bool),
    ('lower_umax', ctypes.c_bool),
    ('lower_umin', ctypes.c_bool),
    ('lower_fminmax_signed_zero', ctypes.c_bool),
    ('lower_fdph', ctypes.c_bool),
    ('fdot_replicates', ctypes.c_bool),
    ('lower_ffloor', ctypes.c_bool),
    ('lower_ffract', ctypes.c_bool),
    ('lower_fceil', ctypes.c_bool),
    ('lower_ftrunc', ctypes.c_bool),
    ('lower_fround_even', ctypes.c_bool),
    ('lower_ldexp', ctypes.c_bool),
    ('lower_pack_half_2x16', ctypes.c_bool),
    ('lower_pack_unorm_2x16', ctypes.c_bool),
    ('lower_pack_snorm_2x16', ctypes.c_bool),
    ('lower_pack_unorm_4x8', ctypes.c_bool),
    ('lower_pack_snorm_4x8', ctypes.c_bool),
    ('lower_pack_64_2x32', ctypes.c_bool),
    ('lower_pack_64_4x16', ctypes.c_bool),
    ('lower_pack_32_2x16', ctypes.c_bool),
    ('lower_pack_64_2x32_split', ctypes.c_bool),
    ('lower_pack_32_2x16_split', ctypes.c_bool),
    ('lower_unpack_half_2x16', ctypes.c_bool),
    ('lower_unpack_unorm_2x16', ctypes.c_bool),
    ('lower_unpack_snorm_2x16', ctypes.c_bool),
    ('lower_unpack_unorm_4x8', ctypes.c_bool),
    ('lower_unpack_snorm_4x8', ctypes.c_bool),
    ('lower_unpack_64_2x32_split', ctypes.c_bool),
    ('lower_unpack_32_2x16_split', ctypes.c_bool),
    ('lower_pack_split', ctypes.c_bool),
    ('lower_extract_byte', ctypes.c_bool),
    ('lower_extract_word', ctypes.c_bool),
    ('lower_insert_byte', ctypes.c_bool),
    ('lower_insert_word', ctypes.c_bool),
    ('vertex_id_zero_based', ctypes.c_bool),
    ('lower_base_vertex', ctypes.c_bool),
    ('instance_id_includes_base_index', ctypes.c_bool),
    ('lower_helper_invocation', ctypes.c_bool),
    ('optimize_sample_mask_in', ctypes.c_bool),
    ('optimize_load_front_face_fsign', ctypes.c_bool),
    ('optimize_quad_vote_to_reduce', ctypes.c_bool),
    ('lower_cs_local_index_to_id', ctypes.c_bool),
    ('lower_cs_local_id_to_index', ctypes.c_bool),
    ('has_cs_global_id', ctypes.c_bool),
    ('lower_device_index_to_zero', ctypes.c_bool),
    ('lower_wpos_pntc', ctypes.c_bool),
    ('lower_hadd', ctypes.c_bool),
    ('lower_hadd64', ctypes.c_bool),
    ('lower_uadd_sat', ctypes.c_bool),
    ('lower_usub_sat', ctypes.c_bool),
    ('lower_iadd_sat', ctypes.c_bool),
    ('lower_mul_32x16', ctypes.c_bool),
    ('lower_bfloat16_conversions', ctypes.c_bool),
    ('vectorize_tess_levels', ctypes.c_bool),
    ('lower_to_scalar', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 5),
    ('lower_to_scalar_filter', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('vectorize_vec2_16bit', ctypes.c_bool),
    ('unify_interfaces', ctypes.c_bool),
    ('lower_interpolate_at', ctypes.c_bool),
    ('lower_mul_2x32_64', ctypes.c_bool),
    ('has_rotate8', ctypes.c_bool),
    ('has_rotate16', ctypes.c_bool),
    ('has_rotate32', ctypes.c_bool),
    ('has_shfr32', ctypes.c_bool),
    ('has_iadd3', ctypes.c_bool),
    ('has_amul', ctypes.c_bool),
    ('has_imul24', ctypes.c_bool),
    ('has_umul24', ctypes.c_bool),
    ('has_mul24_relaxed', ctypes.c_bool),
    ('has_imad32', ctypes.c_bool),
    ('has_umad24', ctypes.c_bool),
    ('has_fused_comp_and_csel', ctypes.c_bool),
    ('has_icsel_eqz64', ctypes.c_bool),
    ('has_icsel_eqz32', ctypes.c_bool),
    ('has_icsel_eqz16', ctypes.c_bool),
    ('has_fneo_fcmpu', ctypes.c_bool),
    ('has_ford_funord', ctypes.c_bool),
    ('has_fsub', ctypes.c_bool),
    ('has_isub', ctypes.c_bool),
    ('has_pack_32_4x8', ctypes.c_bool),
    ('has_texture_scaling', ctypes.c_bool),
    ('has_sdot_4x8', ctypes.c_bool),
    ('has_udot_4x8', ctypes.c_bool),
    ('has_sudot_4x8', ctypes.c_bool),
    ('has_sdot_4x8_sat', ctypes.c_bool),
    ('has_udot_4x8_sat', ctypes.c_bool),
    ('has_sudot_4x8_sat', ctypes.c_bool),
    ('has_dot_2x16', ctypes.c_bool),
    ('has_bfdot2_bfadd', ctypes.c_bool),
    ('has_fmulz', ctypes.c_bool),
    ('has_fmulz_no_denorms', ctypes.c_bool),
    ('has_find_msb_rev', ctypes.c_bool),
    ('has_pack_half_2x16_rtz', ctypes.c_bool),
    ('has_bit_test', ctypes.c_bool),
    ('has_bfe', ctypes.c_bool),
    ('has_bfm', ctypes.c_bool),
    ('has_bfi', ctypes.c_bool),
    ('has_bitfield_select', ctypes.c_bool),
    ('has_uclz', ctypes.c_bool),
    ('has_msad', ctypes.c_bool),
    ('has_f2e4m3fn_satfn', ctypes.c_bool),
    ('has_load_global_bounded', ctypes.c_bool),
    ('intel_vec4', ctypes.c_bool),
    ('avoid_ternary_with_two_constants', ctypes.c_bool),
    ('support_8bit_alu', ctypes.c_bool),
    ('support_16bit_alu', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('max_unroll_iterations', ctypes.c_uint32),
    ('max_unroll_iterations_aggressive', ctypes.c_uint32),
    ('max_unroll_iterations_fp64', ctypes.c_uint32),
    ('lower_uniforms_to_ubo', ctypes.c_bool),
    ('force_indirect_unrolling_sampler', ctypes.c_bool),
    ('no_integers', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte),
    ('force_indirect_unrolling', c__EA_nir_variable_mode),
    ('driver_functions', ctypes.c_bool),
    ('late_lower_int64', ctypes.c_bool),
    ('PADDING_3', ctypes.c_ubyte * 2),
    ('lower_int64_options', nir_lower_int64_options),
    ('lower_doubles_options', nir_lower_doubles_options),
    ('divergence_analysis_options', nir_divergence_options),
    ('support_indirect_inputs', ctypes.c_ubyte),
    ('support_indirect_outputs', ctypes.c_ubyte),
    ('lower_image_offset_to_range_base', ctypes.c_bool),
    ('lower_atomic_offset_to_range_base', ctypes.c_bool),
    ('preserve_mediump', ctypes.c_bool),
    ('lower_fquantize2f16', ctypes.c_bool),
    ('force_f2f16_rtz', ctypes.c_bool),
    ('lower_layer_fs_input_to_sysval', ctypes.c_bool),
    ('compact_arrays', ctypes.c_bool),
    ('discard_is_demote', ctypes.c_bool),
    ('has_ddx_intrinsics', ctypes.c_bool),
    ('scalarize_ddx', ctypes.c_bool),
    ('per_view_unique_driver_locations', ctypes.c_bool),
    ('compact_view_index', ctypes.c_bool),
    ('PADDING_4', ctypes.c_ubyte * 2),
    ('io_options', nir_io_options),
    ('skip_lower_packing_ops', ctypes.c_uint32),
    ('lower_mediump_io', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_nir_shader))),
    ('varying_expression_max_cost', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader))),
    ('varying_estimate_instr_cost', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr))),
    ('max_varying_expression_cost', ctypes.c_uint32),
    ('PADDING_5', ctypes.c_ubyte * 4),
]

class struct_nir_xfb_info(Structure):
    pass

class struct_u_printf_info(Structure):
    pass

class struct_shader_info(Structure):
    pass


# values for enumeration 'pipe_shader_type'
pipe_shader_type__enumvalues = {
    -1: 'MESA_SHADER_NONE',
    0: 'MESA_SHADER_VERTEX',
    0: 'PIPE_SHADER_VERTEX',
    1: 'MESA_SHADER_TESS_CTRL',
    1: 'PIPE_SHADER_TESS_CTRL',
    2: 'MESA_SHADER_TESS_EVAL',
    2: 'PIPE_SHADER_TESS_EVAL',
    3: 'MESA_SHADER_GEOMETRY',
    3: 'PIPE_SHADER_GEOMETRY',
    4: 'MESA_SHADER_FRAGMENT',
    4: 'PIPE_SHADER_FRAGMENT',
    5: 'MESA_SHADER_COMPUTE',
    5: 'PIPE_SHADER_COMPUTE',
    6: 'PIPE_SHADER_TYPES',
    6: 'MESA_SHADER_TASK',
    6: 'PIPE_SHADER_TASK',
    7: 'MESA_SHADER_MESH',
    7: 'PIPE_SHADER_MESH',
    8: 'PIPE_SHADER_MESH_TYPES',
    8: 'MESA_SHADER_RAYGEN',
    9: 'MESA_SHADER_ANY_HIT',
    10: 'MESA_SHADER_CLOSEST_HIT',
    11: 'MESA_SHADER_MISS',
    12: 'MESA_SHADER_INTERSECTION',
    13: 'MESA_SHADER_CALLABLE',
    14: 'MESA_SHADER_KERNEL',
}
MESA_SHADER_NONE = -1
MESA_SHADER_VERTEX = 0
PIPE_SHADER_VERTEX = 0
MESA_SHADER_TESS_CTRL = 1
PIPE_SHADER_TESS_CTRL = 1
MESA_SHADER_TESS_EVAL = 2
PIPE_SHADER_TESS_EVAL = 2
MESA_SHADER_GEOMETRY = 3
PIPE_SHADER_GEOMETRY = 3
MESA_SHADER_FRAGMENT = 4
PIPE_SHADER_FRAGMENT = 4
MESA_SHADER_COMPUTE = 5
PIPE_SHADER_COMPUTE = 5
PIPE_SHADER_TYPES = 6
MESA_SHADER_TASK = 6
PIPE_SHADER_TASK = 6
MESA_SHADER_MESH = 7
PIPE_SHADER_MESH = 7
PIPE_SHADER_MESH_TYPES = 8
MESA_SHADER_RAYGEN = 8
MESA_SHADER_ANY_HIT = 9
MESA_SHADER_CLOSEST_HIT = 10
MESA_SHADER_MISS = 11
MESA_SHADER_INTERSECTION = 12
MESA_SHADER_CALLABLE = 13
MESA_SHADER_KERNEL = 14
pipe_shader_type = ctypes.c_int32 # enum

# values for enumeration 'gl_subgroup_size'
gl_subgroup_size__enumvalues = {
    0: 'SUBGROUP_SIZE_VARYING',
    1: 'SUBGROUP_SIZE_UNIFORM',
    2: 'SUBGROUP_SIZE_API_CONSTANT',
    3: 'SUBGROUP_SIZE_FULL_SUBGROUPS',
    4: 'SUBGROUP_SIZE_REQUIRE_4',
    8: 'SUBGROUP_SIZE_REQUIRE_8',
    16: 'SUBGROUP_SIZE_REQUIRE_16',
    32: 'SUBGROUP_SIZE_REQUIRE_32',
    64: 'SUBGROUP_SIZE_REQUIRE_64',
    128: 'SUBGROUP_SIZE_REQUIRE_128',
}
SUBGROUP_SIZE_VARYING = 0
SUBGROUP_SIZE_UNIFORM = 1
SUBGROUP_SIZE_API_CONSTANT = 2
SUBGROUP_SIZE_FULL_SUBGROUPS = 3
SUBGROUP_SIZE_REQUIRE_4 = 4
SUBGROUP_SIZE_REQUIRE_8 = 8
SUBGROUP_SIZE_REQUIRE_16 = 16
SUBGROUP_SIZE_REQUIRE_32 = 32
SUBGROUP_SIZE_REQUIRE_64 = 64
SUBGROUP_SIZE_REQUIRE_128 = 128
gl_subgroup_size = ctypes.c_uint32 # enum

# values for enumeration 'gl_derivative_group'
gl_derivative_group__enumvalues = {
    0: 'DERIVATIVE_GROUP_NONE',
    1: 'DERIVATIVE_GROUP_QUADS',
    2: 'DERIVATIVE_GROUP_LINEAR',
}
DERIVATIVE_GROUP_NONE = 0
DERIVATIVE_GROUP_QUADS = 1
DERIVATIVE_GROUP_LINEAR = 2
gl_derivative_group = ctypes.c_uint32 # enum
class union_shader_info_0(Union):
    pass

class struct_shader_info_0_vs(Structure):
    pass

struct_shader_info_0_vs._pack_ = 1 # source:False
struct_shader_info_0_vs._fields_ = [
    ('double_inputs', ctypes.c_uint64),
    ('blit_sgprs_amd', ctypes.c_ubyte, 4),
    ('tes_agx', ctypes.c_ubyte, 1),
    ('window_space_position', ctypes.c_ubyte, 1),
    ('needs_edge_flag', ctypes.c_ubyte, 1),
    ('PADDING_0', ctypes.c_uint64, 57),
]

class struct_shader_info_0_gs(Structure):
    pass


# values for enumeration 'mesa_prim'
mesa_prim__enumvalues = {
    0: 'MESA_PRIM_POINTS',
    1: 'MESA_PRIM_LINES',
    2: 'MESA_PRIM_LINE_LOOP',
    3: 'MESA_PRIM_LINE_STRIP',
    4: 'MESA_PRIM_TRIANGLES',
    5: 'MESA_PRIM_TRIANGLE_STRIP',
    6: 'MESA_PRIM_TRIANGLE_FAN',
    7: 'MESA_PRIM_QUADS',
    8: 'MESA_PRIM_QUAD_STRIP',
    9: 'MESA_PRIM_POLYGON',
    10: 'MESA_PRIM_LINES_ADJACENCY',
    11: 'MESA_PRIM_LINE_STRIP_ADJACENCY',
    12: 'MESA_PRIM_TRIANGLES_ADJACENCY',
    13: 'MESA_PRIM_TRIANGLE_STRIP_ADJACENCY',
    14: 'MESA_PRIM_PATCHES',
    14: 'MESA_PRIM_MAX',
    15: 'MESA_PRIM_COUNT',
    28: 'MESA_PRIM_UNKNOWN',
}
MESA_PRIM_POINTS = 0
MESA_PRIM_LINES = 1
MESA_PRIM_LINE_LOOP = 2
MESA_PRIM_LINE_STRIP = 3
MESA_PRIM_TRIANGLES = 4
MESA_PRIM_TRIANGLE_STRIP = 5
MESA_PRIM_TRIANGLE_FAN = 6
MESA_PRIM_QUADS = 7
MESA_PRIM_QUAD_STRIP = 8
MESA_PRIM_POLYGON = 9
MESA_PRIM_LINES_ADJACENCY = 10
MESA_PRIM_LINE_STRIP_ADJACENCY = 11
MESA_PRIM_TRIANGLES_ADJACENCY = 12
MESA_PRIM_TRIANGLE_STRIP_ADJACENCY = 13
MESA_PRIM_PATCHES = 14
MESA_PRIM_MAX = 14
MESA_PRIM_COUNT = 15
MESA_PRIM_UNKNOWN = 28
mesa_prim = ctypes.c_uint32 # enum
struct_shader_info_0_gs._pack_ = 1 # source:False
struct_shader_info_0_gs._fields_ = [
    ('output_primitive', mesa_prim),
    ('input_primitive', mesa_prim),
    ('vertices_out', ctypes.c_uint16),
    ('invocations', ctypes.c_ubyte),
    ('vertices_in', ctypes.c_ubyte, 3),
    ('uses_end_primitive', ctypes.c_ubyte, 1),
    ('active_stream_mask', ctypes.c_ubyte, 4),
]

class struct_shader_info_0_fs(Structure):
    pass


# values for enumeration 'c_uint64'
c_uint64__enumvalues = {
    0: 'FRAG_DEPTH_LAYOUT_NONE',
    1: 'FRAG_DEPTH_LAYOUT_ANY',
    2: 'FRAG_DEPTH_LAYOUT_GREATER',
    3: 'FRAG_DEPTH_LAYOUT_LESS',
    4: 'FRAG_DEPTH_LAYOUT_UNCHANGED',
}
FRAG_DEPTH_LAYOUT_NONE = 0
FRAG_DEPTH_LAYOUT_ANY = 1
FRAG_DEPTH_LAYOUT_GREATER = 2
FRAG_DEPTH_LAYOUT_LESS = 3
FRAG_DEPTH_LAYOUT_UNCHANGED = 4
c_uint64 = ctypes.c_uint32 # enum

# values for enumeration 'c_bool'
c_bool__enumvalues = {
    0: 'FRAG_STENCIL_LAYOUT_NONE',
    1: 'FRAG_STENCIL_LAYOUT_ANY',
    2: 'FRAG_STENCIL_LAYOUT_GREATER',
    3: 'FRAG_STENCIL_LAYOUT_LESS',
    4: 'FRAG_STENCIL_LAYOUT_UNCHANGED',
}
FRAG_STENCIL_LAYOUT_NONE = 0
FRAG_STENCIL_LAYOUT_ANY = 1
FRAG_STENCIL_LAYOUT_GREATER = 2
FRAG_STENCIL_LAYOUT_LESS = 3
FRAG_STENCIL_LAYOUT_UNCHANGED = 4
c_bool = ctypes.c_uint32 # enum
struct_shader_info_0_fs._pack_ = 1 # source:False
struct_shader_info_0_fs._fields_ = [
    ('uses_discard', ctypes.c_uint64, 1),
    ('uses_fbfetch_output', ctypes.c_uint64, 1),
    ('fbfetch_coherent', ctypes.c_uint64, 1),
    ('color_is_dual_source', ctypes.c_uint64, 1),
    ('require_full_quads', ctypes.c_uint64, 1),
    ('quad_derivatives', ctypes.c_uint64, 1),
    ('needs_coarse_quad_helper_invocations', ctypes.c_uint64, 1),
    ('needs_full_quad_helper_invocations', ctypes.c_uint64, 1),
    ('uses_sample_qualifier', ctypes.c_uint64, 1),
    ('uses_sample_shading', ctypes.c_uint64, 1),
    ('early_fragment_tests', ctypes.c_uint64, 1),
    ('inner_coverage', ctypes.c_uint64, 1),
    ('post_depth_coverage', ctypes.c_uint64, 1),
    ('pixel_center_integer', ctypes.c_uint64, 1),
    ('origin_upper_left', ctypes.c_uint64, 1),
    ('pixel_interlock_ordered', ctypes.c_uint64, 1),
    ('pixel_interlock_unordered', ctypes.c_uint64, 1),
    ('sample_interlock_ordered', ctypes.c_uint64, 1),
    ('sample_interlock_unordered', ctypes.c_uint64, 1),
    ('untyped_color_outputs', ctypes.c_uint64, 1),
    ('depth_layout', c_uint64, 3),
    ('color0_interp', ctypes.c_uint64, 3),
    ('color0_sample', ctypes.c_uint64, 1),
    ('color0_centroid', ctypes.c_uint64, 1),
    ('color1_interp', ctypes.c_uint64, 3),
    ('color1_sample', ctypes.c_uint64, 1),
    ('color1_centroid', ctypes.c_uint64, 1),
    ('PADDING_0', ctypes.c_uint32, 31),
    ('advanced_blend_modes', ctypes.c_uint32),
    ('early_and_late_fragment_tests', ctypes.c_bool, 1),
    ('stencil_front_layout', c_bool, 3),
    ('stencil_back_layout', c_bool, 3),
    ('PADDING_1', ctypes.c_uint32, 25),
]

class struct_shader_info_0_cs(Structure):
    pass

struct_shader_info_0_cs._pack_ = 1 # source:False
struct_shader_info_0_cs._fields_ = [
    ('workgroup_size_hint', ctypes.c_uint16 * 3),
    ('user_data_components_amd', ctypes.c_ubyte, 4),
    ('has_variable_shared_mem', ctypes.c_ubyte, 1),
    ('has_cooperative_matrix', ctypes.c_ubyte, 1),
    ('PADDING_0', ctypes.c_uint8, 2),
    ('image_block_size_per_thread_agx', ctypes.c_ubyte, 8),
    ('ptr_size', ctypes.c_uint32),
    ('shader_index', ctypes.c_uint32),
    ('node_payloads_size', ctypes.c_uint32),
    ('workgroup_count', ctypes.c_uint32 * 3),
]

class struct_shader_info_0_tess(Structure):
    pass


# values for enumeration 'tess_primitive_mode'
tess_primitive_mode__enumvalues = {
    0: 'TESS_PRIMITIVE_UNSPECIFIED',
    1: 'TESS_PRIMITIVE_TRIANGLES',
    2: 'TESS_PRIMITIVE_QUADS',
    3: 'TESS_PRIMITIVE_ISOLINES',
}
TESS_PRIMITIVE_UNSPECIFIED = 0
TESS_PRIMITIVE_TRIANGLES = 1
TESS_PRIMITIVE_QUADS = 2
TESS_PRIMITIVE_ISOLINES = 3
tess_primitive_mode = ctypes.c_uint32 # enum
struct_shader_info_0_tess._pack_ = 1 # source:False
struct_shader_info_0_tess._fields_ = [
    ('_primitive_mode', tess_primitive_mode),
    ('tcs_vertices_out', ctypes.c_ubyte),
    ('spacing', ctypes.c_uint32, 2),
    ('ccw', ctypes.c_uint32, 1),
    ('point_mode', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint32, 20),
    ('tcs_same_invocation_inputs_read', ctypes.c_uint64),
    ('tcs_cross_invocation_inputs_read', ctypes.c_uint64),
    ('tcs_cross_invocation_outputs_read', ctypes.c_uint64),
    ('tcs_cross_invocation_outputs_written', ctypes.c_uint64),
    ('tcs_outputs_read_by_tes', ctypes.c_uint64),
    ('tcs_patch_outputs_read_by_tes', ctypes.c_uint32),
    ('tcs_outputs_read_by_tes_16bit', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 2),
]

class struct_shader_info_0_mesh(Structure):
    pass

struct_shader_info_0_mesh._pack_ = 1 # source:False
struct_shader_info_0_mesh._fields_ = [
    ('ms_cross_invocation_output_access', ctypes.c_uint64),
    ('ts_mesh_dispatch_dimensions', ctypes.c_uint32 * 3),
    ('max_vertices_out', ctypes.c_uint16),
    ('max_primitives_out', ctypes.c_uint16),
    ('primitive_type', mesa_prim),
    ('nv', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

union_shader_info_0._pack_ = 1 # source:False
union_shader_info_0._fields_ = [
    ('vs', struct_shader_info_0_vs),
    ('gs', struct_shader_info_0_gs),
    ('fs', struct_shader_info_0_fs),
    ('cs', struct_shader_info_0_cs),
    ('tess', struct_shader_info_0_tess),
    ('mesh', struct_shader_info_0_mesh),
    ('PADDING_0', ctypes.c_ubyte * 24),
]

struct_shader_info._pack_ = 1 # source:False
struct_shader_info._anonymous_ = ('_0',)
struct_shader_info._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('label', ctypes.POINTER(ctypes.c_char)),
    ('internal', ctypes.c_bool),
    ('source_blake3', ctypes.c_ubyte * 32),
    ('stage', ctypes.c_ubyte),
    ('prev_stage', ctypes.c_ubyte),
    ('next_stage', ctypes.c_ubyte),
    ('prev_stage_has_xfb', ctypes.c_ubyte),
    ('num_textures', ctypes.c_ubyte),
    ('num_ubos', ctypes.c_ubyte),
    ('num_abos', ctypes.c_ubyte),
    ('num_ssbos', ctypes.c_ubyte),
    ('num_images', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('inputs_read', ctypes.c_uint64),
    ('dual_slot_inputs', ctypes.c_uint64),
    ('outputs_written', ctypes.c_uint64),
    ('outputs_read', ctypes.c_uint64),
    ('system_values_read', ctypes.c_uint32 * 4),
    ('per_primitive_inputs', ctypes.c_uint64),
    ('per_primitive_outputs', ctypes.c_uint64),
    ('per_view_outputs', ctypes.c_uint64),
    ('view_mask', ctypes.c_uint32),
    ('inputs_read_16bit', ctypes.c_uint16),
    ('outputs_written_16bit', ctypes.c_uint16),
    ('outputs_read_16bit', ctypes.c_uint16),
    ('inputs_read_indirectly_16bit', ctypes.c_uint16),
    ('outputs_read_indirectly_16bit', ctypes.c_uint16),
    ('outputs_written_indirectly_16bit', ctypes.c_uint16),
    ('patch_inputs_read', ctypes.c_uint32),
    ('patch_outputs_written', ctypes.c_uint32),
    ('patch_outputs_read', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('inputs_read_indirectly', ctypes.c_uint64),
    ('outputs_read_indirectly', ctypes.c_uint64),
    ('outputs_written_indirectly', ctypes.c_uint64),
    ('patch_inputs_read_indirectly', ctypes.c_uint32),
    ('patch_outputs_read_indirectly', ctypes.c_uint32),
    ('patch_outputs_written_indirectly', ctypes.c_uint32),
    ('textures_used', ctypes.c_uint32 * 4),
    ('textures_used_by_txf', ctypes.c_uint32 * 4),
    ('samplers_used', ctypes.c_uint32 * 1),
    ('images_used', ctypes.c_uint32 * 2),
    ('image_buffers', ctypes.c_uint32 * 2),
    ('msaa_images', ctypes.c_uint32 * 2),
    ('float_controls_execution_mode', ctypes.c_uint32),
    ('shared_size', ctypes.c_uint32),
    ('task_payload_size', ctypes.c_uint32),
    ('ray_queries', ctypes.c_uint32),
    ('workgroup_size', ctypes.c_uint16 * 3),
    ('PADDING_2', ctypes.c_ubyte * 2),
    ('subgroup_size', gl_subgroup_size),
    ('num_subgroups', ctypes.c_ubyte),
    ('uses_wide_subgroup_intrinsics', ctypes.c_bool),
    ('xfb_stride', ctypes.c_ubyte * 4),
    ('inlinable_uniform_dw_offsets', ctypes.c_uint16 * 4),
    ('num_inlinable_uniforms', ctypes.c_ubyte, 4),
    ('clip_distance_array_size', ctypes.c_ubyte, 4),
    ('cull_distance_array_size', ctypes.c_ubyte, 4),
    ('uses_texture_gather', ctypes.c_ubyte, 1),
    ('uses_resource_info_query', ctypes.c_ubyte, 1),
    ('PADDING_3', ctypes.c_uint8, 2),
    ('bit_sizes_float', ctypes.c_ubyte, 8),
    ('bit_sizes_int', ctypes.c_ubyte),
    ('first_ubo_is_default_ubo', ctypes.c_bool, 1),
    ('separate_shader', ctypes.c_bool, 1),
    ('has_transform_feedback_varyings', ctypes.c_bool, 1),
    ('flrp_lowered', ctypes.c_bool, 1),
    ('io_lowered', ctypes.c_bool, 1),
    ('var_copies_lowered', ctypes.c_bool, 1),
    ('writes_memory', ctypes.c_bool, 1),
    ('layer_viewport_relative', ctypes.c_bool, 1),
    ('uses_control_barrier', ctypes.c_bool, 1),
    ('uses_memory_barrier', ctypes.c_bool, 1),
    ('uses_bindless', ctypes.c_bool, 1),
    ('shared_memory_explicit_layout', ctypes.c_bool, 1),
    ('zero_initialize_shared_memory', ctypes.c_bool, 1),
    ('workgroup_size_variable', ctypes.c_bool, 1),
    ('uses_printf', ctypes.c_bool, 1),
    ('maximally_reconverges', ctypes.c_bool, 1),
    ('use_aco_amd', ctypes.c_bool, 1),
    ('use_lowered_image_to_global', ctypes.c_bool, 1),
    ('PADDING_4', ctypes.c_uint8, 6),
    ('use_legacy_math_rules', ctypes.c_bool, 8),
    ('derivative_group', gl_derivative_group, 2),
    ('PADDING_5', ctypes.c_uint64, 46),
    ('_0', union_shader_info_0),
]

struct_nir_shader._pack_ = 1 # source:False
struct_nir_shader._fields_ = [
    ('gctx', ctypes.POINTER(struct_gc_ctx)),
    ('variables', struct_exec_list),
    ('options', ctypes.POINTER(struct_nir_shader_compiler_options)),
    ('info', struct_shader_info),
    ('functions', struct_exec_list),
    ('num_inputs', ctypes.c_uint32),
    ('num_uniforms', ctypes.c_uint32),
    ('num_outputs', ctypes.c_uint32),
    ('global_mem_size', ctypes.c_uint32),
    ('scratch_size', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('constant_data', ctypes.POINTER(None)),
    ('constant_data_size', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('xfb_info', ctypes.POINTER(struct_nir_xfb_info)),
    ('printf_info_count', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('printf_info', ctypes.POINTER(struct_u_printf_info)),
    ('has_debug_info', ctypes.c_bool),
    ('PADDING_3', ctypes.c_ubyte * 7),
]

nir_shader_compiler_options = struct_nir_shader_compiler_options
u_printf_info = struct_u_printf_info
nir_debug = 0 # Variable ctypes.c_uint32
nir_debug_print_shader = [] # Variable ctypes.c_bool * 15
nir_component_mask_t = ctypes.c_uint16
try:
    nir_round_up_components = _libraries['FIXME_STUB'].nir_round_up_components
    nir_round_up_components.restype = ctypes.c_uint32
    nir_round_up_components.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_round_down_components = _libraries['FIXME_STUB'].nir_round_down_components
    nir_round_down_components.restype = ctypes.c_uint32
    nir_round_down_components.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_component_mask = _libraries['FIXME_STUB'].nir_component_mask
    nir_component_mask.restype = nir_component_mask_t
    nir_component_mask.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_process_debug_variable = _libraries['libtinymesa_cpu.so'].nir_process_debug_variable
    nir_process_debug_variable.restype = None
    nir_process_debug_variable.argtypes = []
except AttributeError:
    pass
try:
    nir_component_mask_can_reinterpret = _libraries['libtinymesa_cpu.so'].nir_component_mask_can_reinterpret
    nir_component_mask_can_reinterpret.restype = ctypes.c_bool
    nir_component_mask_can_reinterpret.argtypes = [nir_component_mask_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_component_mask_reinterpret = _libraries['libtinymesa_cpu.so'].nir_component_mask_reinterpret
    nir_component_mask_reinterpret.restype = nir_component_mask_t
    nir_component_mask_reinterpret.argtypes = [nir_component_mask_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
class struct_nir_state_slot(Structure):
    pass

struct_nir_state_slot._pack_ = 1 # source:False
struct_nir_state_slot._fields_ = [
    ('tokens', ctypes.c_int16 * 4),
]

nir_state_slot = struct_nir_state_slot

# values for enumeration 'c__EA_nir_rounding_mode'
c__EA_nir_rounding_mode__enumvalues = {
    0: 'nir_rounding_mode_undef',
    1: 'nir_rounding_mode_rtne',
    2: 'nir_rounding_mode_ru',
    3: 'nir_rounding_mode_rd',
    4: 'nir_rounding_mode_rtz',
}
nir_rounding_mode_undef = 0
nir_rounding_mode_rtne = 1
nir_rounding_mode_ru = 2
nir_rounding_mode_rd = 3
nir_rounding_mode_rtz = 4
c__EA_nir_rounding_mode = ctypes.c_uint32 # enum
nir_rounding_mode = c__EA_nir_rounding_mode
nir_rounding_mode__enumvalues = c__EA_nir_rounding_mode__enumvalues

# values for enumeration 'c__EA_nir_ray_query_value'
c__EA_nir_ray_query_value__enumvalues = {
    0: 'nir_ray_query_value_intersection_type',
    1: 'nir_ray_query_value_intersection_t',
    2: 'nir_ray_query_value_intersection_instance_custom_index',
    3: 'nir_ray_query_value_intersection_instance_id',
    4: 'nir_ray_query_value_intersection_instance_sbt_index',
    5: 'nir_ray_query_value_intersection_geometry_index',
    6: 'nir_ray_query_value_intersection_primitive_index',
    7: 'nir_ray_query_value_intersection_barycentrics',
    8: 'nir_ray_query_value_intersection_front_face',
    9: 'nir_ray_query_value_intersection_object_ray_direction',
    10: 'nir_ray_query_value_intersection_object_ray_origin',
    11: 'nir_ray_query_value_intersection_object_to_world',
    12: 'nir_ray_query_value_intersection_world_to_object',
    13: 'nir_ray_query_value_intersection_candidate_aabb_opaque',
    14: 'nir_ray_query_value_tmin',
    15: 'nir_ray_query_value_flags',
    16: 'nir_ray_query_value_world_ray_direction',
    17: 'nir_ray_query_value_world_ray_origin',
    18: 'nir_ray_query_value_intersection_triangle_vertex_positions',
}
nir_ray_query_value_intersection_type = 0
nir_ray_query_value_intersection_t = 1
nir_ray_query_value_intersection_instance_custom_index = 2
nir_ray_query_value_intersection_instance_id = 3
nir_ray_query_value_intersection_instance_sbt_index = 4
nir_ray_query_value_intersection_geometry_index = 5
nir_ray_query_value_intersection_primitive_index = 6
nir_ray_query_value_intersection_barycentrics = 7
nir_ray_query_value_intersection_front_face = 8
nir_ray_query_value_intersection_object_ray_direction = 9
nir_ray_query_value_intersection_object_ray_origin = 10
nir_ray_query_value_intersection_object_to_world = 11
nir_ray_query_value_intersection_world_to_object = 12
nir_ray_query_value_intersection_candidate_aabb_opaque = 13
nir_ray_query_value_tmin = 14
nir_ray_query_value_flags = 15
nir_ray_query_value_world_ray_direction = 16
nir_ray_query_value_world_ray_origin = 17
nir_ray_query_value_intersection_triangle_vertex_positions = 18
c__EA_nir_ray_query_value = ctypes.c_uint32 # enum
nir_ray_query_value = c__EA_nir_ray_query_value
nir_ray_query_value__enumvalues = c__EA_nir_ray_query_value__enumvalues

# values for enumeration 'c__EA_nir_resource_data_intel'
c__EA_nir_resource_data_intel__enumvalues = {
    1: 'nir_resource_intel_bindless',
    2: 'nir_resource_intel_pushable',
    4: 'nir_resource_intel_sampler',
    8: 'nir_resource_intel_non_uniform',
    16: 'nir_resource_intel_sampler_embedded',
}
nir_resource_intel_bindless = 1
nir_resource_intel_pushable = 2
nir_resource_intel_sampler = 4
nir_resource_intel_non_uniform = 8
nir_resource_intel_sampler_embedded = 16
c__EA_nir_resource_data_intel = ctypes.c_uint32 # enum
nir_resource_data_intel = c__EA_nir_resource_data_intel
nir_resource_data_intel__enumvalues = c__EA_nir_resource_data_intel__enumvalues

# values for enumeration 'c__EA_nir_preamble_class'
c__EA_nir_preamble_class__enumvalues = {
    0: 'nir_preamble_class_general',
    1: 'nir_preamble_class_image',
    2: 'nir_preamble_num_classes',
}
nir_preamble_class_general = 0
nir_preamble_class_image = 1
nir_preamble_num_classes = 2
c__EA_nir_preamble_class = ctypes.c_uint32 # enum
nir_preamble_class = c__EA_nir_preamble_class
nir_preamble_class__enumvalues = c__EA_nir_preamble_class__enumvalues

# values for enumeration 'c__EA_nir_cmat_signed'
c__EA_nir_cmat_signed__enumvalues = {
    1: 'NIR_CMAT_A_SIGNED',
    2: 'NIR_CMAT_B_SIGNED',
    4: 'NIR_CMAT_C_SIGNED',
    8: 'NIR_CMAT_RESULT_SIGNED',
}
NIR_CMAT_A_SIGNED = 1
NIR_CMAT_B_SIGNED = 2
NIR_CMAT_C_SIGNED = 4
NIR_CMAT_RESULT_SIGNED = 8
c__EA_nir_cmat_signed = ctypes.c_uint32 # enum
nir_cmat_signed = c__EA_nir_cmat_signed
nir_cmat_signed__enumvalues = c__EA_nir_cmat_signed__enumvalues
class union_c__UA_nir_const_value(Union):
    pass

union_c__UA_nir_const_value._pack_ = 1 # source:False
union_c__UA_nir_const_value._fields_ = [
    ('b', ctypes.c_bool),
    ('f32', ctypes.c_float),
    ('f64', ctypes.c_double),
    ('i8', ctypes.c_byte),
    ('u8', ctypes.c_ubyte),
    ('i16', ctypes.c_int16),
    ('u16', ctypes.c_uint16),
    ('i32', ctypes.c_int32),
    ('u32', ctypes.c_uint32),
    ('i64', ctypes.c_int64),
    ('u64', ctypes.c_uint64),
]

nir_const_value = union_c__UA_nir_const_value
try:
    nir_const_value_for_raw_uint = _libraries['FIXME_STUB'].nir_const_value_for_raw_uint
    nir_const_value_for_raw_uint.restype = nir_const_value
    nir_const_value_for_raw_uint.argtypes = [uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
int64_t = ctypes.c_int64
try:
    nir_const_value_for_int = _libraries['FIXME_STUB'].nir_const_value_for_int
    nir_const_value_for_int.restype = nir_const_value
    nir_const_value_for_int.argtypes = [int64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_for_uint = _libraries['FIXME_STUB'].nir_const_value_for_uint
    nir_const_value_for_uint.restype = nir_const_value
    nir_const_value_for_uint.argtypes = [uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_for_bool = _libraries['FIXME_STUB'].nir_const_value_for_bool
    nir_const_value_for_bool.restype = nir_const_value
    nir_const_value_for_bool.argtypes = [ctypes.c_bool, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_for_float = _libraries['libtinymesa_cpu.so'].nir_const_value_for_float
    nir_const_value_for_float.restype = nir_const_value
    nir_const_value_for_float.argtypes = [ctypes.c_double, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_as_int = _libraries['FIXME_STUB'].nir_const_value_as_int
    nir_const_value_as_int.restype = int64_t
    nir_const_value_as_int.argtypes = [nir_const_value, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_as_uint = _libraries['FIXME_STUB'].nir_const_value_as_uint
    nir_const_value_as_uint.restype = uint64_t
    nir_const_value_as_uint.argtypes = [nir_const_value, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_as_bool = _libraries['FIXME_STUB'].nir_const_value_as_bool
    nir_const_value_as_bool.restype = ctypes.c_bool
    nir_const_value_as_bool.argtypes = [nir_const_value, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_as_float = _libraries['libtinymesa_cpu.so'].nir_const_value_as_float
    nir_const_value_as_float.restype = ctypes.c_double
    nir_const_value_as_float.argtypes = [nir_const_value, ctypes.c_uint32]
except AttributeError:
    pass
class struct_nir_constant(Structure):
    pass

struct_nir_constant._pack_ = 1 # source:False
struct_nir_constant._fields_ = [
    ('values', union_c__UA_nir_const_value * 16),
    ('is_null_constant', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('num_elements', ctypes.c_uint32),
    ('elements', ctypes.POINTER(ctypes.POINTER(struct_nir_constant))),
]

nir_constant = struct_nir_constant

# values for enumeration 'c__EA_nir_depth_layout'
c__EA_nir_depth_layout__enumvalues = {
    0: 'nir_depth_layout_none',
    1: 'nir_depth_layout_any',
    2: 'nir_depth_layout_greater',
    3: 'nir_depth_layout_less',
    4: 'nir_depth_layout_unchanged',
}
nir_depth_layout_none = 0
nir_depth_layout_any = 1
nir_depth_layout_greater = 2
nir_depth_layout_less = 3
nir_depth_layout_unchanged = 4
c__EA_nir_depth_layout = ctypes.c_uint32 # enum
nir_depth_layout = c__EA_nir_depth_layout
nir_depth_layout__enumvalues = c__EA_nir_depth_layout__enumvalues

# values for enumeration 'c__EA_nir_var_declaration_type'
c__EA_nir_var_declaration_type__enumvalues = {
    0: 'nir_var_declared_normally',
    1: 'nir_var_declared_implicitly',
    2: 'nir_var_hidden',
}
nir_var_declared_normally = 0
nir_var_declared_implicitly = 1
nir_var_hidden = 2
c__EA_nir_var_declaration_type = ctypes.c_uint32 # enum
nir_var_declaration_type = c__EA_nir_var_declaration_type
nir_var_declaration_type__enumvalues = c__EA_nir_var_declaration_type__enumvalues
class struct_nir_variable_data(Structure):
    pass

class union_nir_variable_data_0(Union):
    pass

class struct_nir_variable_data_0_image(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('format', pipe_format),
     ]

class struct_nir_variable_data_0_sampler(Structure):
    pass

struct_nir_variable_data_0_sampler._pack_ = 1 # source:False
struct_nir_variable_data_0_sampler._fields_ = [
    ('is_inline_sampler', ctypes.c_uint32, 1),
    ('addressing_mode', ctypes.c_uint32, 3),
    ('normalized_coordinates', ctypes.c_uint32, 1),
    ('filter_mode', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint32, 26),
]

class struct_nir_variable_data_0_xfb(Structure):
    pass

struct_nir_variable_data_0_xfb._pack_ = 1 # source:False
struct_nir_variable_data_0_xfb._fields_ = [
    ('buffer', ctypes.c_uint16, 2),
    ('PADDING_0', ctypes.c_uint16, 14),
    ('stride', ctypes.c_uint16),
]

union_nir_variable_data_0._pack_ = 1 # source:False
union_nir_variable_data_0._fields_ = [
    ('image', struct_nir_variable_data_0_image),
    ('sampler', struct_nir_variable_data_0_sampler),
    ('xfb', struct_nir_variable_data_0_xfb),
]

struct_nir_variable_data._pack_ = 1 # source:False
struct_nir_variable_data._anonymous_ = ('_0',)
struct_nir_variable_data._fields_ = [
    ('mode', ctypes.c_uint64, 21),
    ('read_only', ctypes.c_uint64, 1),
    ('centroid', ctypes.c_uint64, 1),
    ('sample', ctypes.c_uint64, 1),
    ('patch', ctypes.c_uint64, 1),
    ('invariant', ctypes.c_uint64, 1),
    ('explicit_invariant', ctypes.c_uint64, 1),
    ('ray_query', ctypes.c_uint64, 1),
    ('precision', ctypes.c_uint64, 2),
    ('assigned', ctypes.c_uint64, 1),
    ('cannot_coalesce', ctypes.c_uint64, 1),
    ('always_active_io', ctypes.c_uint64, 1),
    ('interpolation', ctypes.c_uint64, 3),
    ('location_frac', ctypes.c_uint64, 2),
    ('compact', ctypes.c_uint64, 1),
    ('fb_fetch_output', ctypes.c_uint64, 1),
    ('bindless', ctypes.c_uint64, 1),
    ('explicit_binding', ctypes.c_uint64, 1),
    ('explicit_location', ctypes.c_uint64, 1),
    ('implicit_sized_array', ctypes.c_uint64, 1),
    ('PADDING_0', ctypes.c_uint32, 20),
    ('max_array_access', ctypes.c_int32),
    ('has_initializer', ctypes.c_uint64, 1),
    ('is_implicit_initializer', ctypes.c_uint64, 1),
    ('is_xfb', ctypes.c_uint64, 1),
    ('is_xfb_only', ctypes.c_uint64, 1),
    ('explicit_xfb_buffer', ctypes.c_uint64, 1),
    ('explicit_xfb_stride', ctypes.c_uint64, 1),
    ('explicit_offset', ctypes.c_uint64, 1),
    ('matrix_layout', ctypes.c_uint64, 2),
    ('from_named_ifc_block', ctypes.c_uint64, 1),
    ('from_ssbo_unsized_array', ctypes.c_uint64, 1),
    ('must_be_shader_input', ctypes.c_uint64, 1),
    ('used', ctypes.c_uint64, 1),
    ('how_declared', ctypes.c_uint64, 2),
    ('per_view', ctypes.c_uint64, 1),
    ('per_primitive', ctypes.c_uint64, 1),
    ('per_vertex', ctypes.c_uint64, 1),
    ('aliased_shared_memory', ctypes.c_uint64, 1),
    ('depth_layout', ctypes.c_uint64, 3),
    ('stream', ctypes.c_uint64, 9),
    ('PADDING_1', ctypes.c_uint8, 1),
    ('access', ctypes.c_uint64, 9),
    ('descriptor_set', ctypes.c_uint64, 5),
    ('PADDING_2', ctypes.c_uint32, 18),
    ('index', ctypes.c_uint32),
    ('binding', ctypes.c_uint32),
    ('location', ctypes.c_int32),
    ('alignment', ctypes.c_uint32),
    ('driver_location', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('_0', union_nir_variable_data_0),
    ('node_name', ctypes.POINTER(ctypes.c_char)),
]

nir_variable_data = struct_nir_variable_data
class struct_nir_variable(Structure):
    pass

struct_nir_variable._pack_ = 1 # source:False
struct_nir_variable._fields_ = [
    ('node', struct_exec_node),
    ('type', ctypes.POINTER(struct_glsl_type)),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('data', struct_nir_variable_data),
    ('index', ctypes.c_uint32),
    ('num_members', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('max_ifc_array_access', ctypes.POINTER(ctypes.c_int32)),
    ('num_state_slots', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 6),
    ('state_slots', ctypes.POINTER(struct_nir_state_slot)),
    ('constant_initializer', ctypes.POINTER(struct_nir_constant)),
    ('pointer_initializer', ctypes.POINTER(struct_nir_variable)),
    ('interface_type', ctypes.POINTER(struct_glsl_type)),
    ('members', ctypes.POINTER(struct_nir_variable_data)),
]

nir_variable = struct_nir_variable
try:
    _nir_shader_variable_has_mode = _libraries['FIXME_STUB']._nir_shader_variable_has_mode
    _nir_shader_variable_has_mode.restype = ctypes.c_bool
    _nir_shader_variable_has_mode.argtypes = [ctypes.POINTER(struct_nir_variable), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_variable_is_global = _libraries['FIXME_STUB'].nir_variable_is_global
    nir_variable_is_global.restype = ctypes.c_bool
    nir_variable_is_global.argtypes = [ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_instr_type'
c__EA_nir_instr_type__enumvalues = {
    0: 'nir_instr_type_alu',
    1: 'nir_instr_type_deref',
    2: 'nir_instr_type_call',
    3: 'nir_instr_type_tex',
    4: 'nir_instr_type_intrinsic',
    5: 'nir_instr_type_load_const',
    6: 'nir_instr_type_jump',
    7: 'nir_instr_type_undef',
    8: 'nir_instr_type_phi',
    9: 'nir_instr_type_parallel_copy',
}
nir_instr_type_alu = 0
nir_instr_type_deref = 1
nir_instr_type_call = 2
nir_instr_type_tex = 3
nir_instr_type_intrinsic = 4
nir_instr_type_load_const = 5
nir_instr_type_jump = 6
nir_instr_type_undef = 7
nir_instr_type_phi = 8
nir_instr_type_parallel_copy = 9
c__EA_nir_instr_type = ctypes.c_uint32 # enum
nir_instr_type = c__EA_nir_instr_type
nir_instr_type__enumvalues = c__EA_nir_instr_type__enumvalues
nir_instr = struct_nir_instr
try:
    nir_instr_next = _libraries['FIXME_STUB'].nir_instr_next
    nir_instr_next.restype = ctypes.POINTER(struct_nir_instr)
    nir_instr_next.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_prev = _libraries['FIXME_STUB'].nir_instr_prev
    nir_instr_prev.restype = ctypes.POINTER(struct_nir_instr)
    nir_instr_prev.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_is_first = _libraries['FIXME_STUB'].nir_instr_is_first
    nir_instr_is_first.restype = ctypes.c_bool
    nir_instr_is_first.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_is_last = _libraries['FIXME_STUB'].nir_instr_is_last
    nir_instr_is_last.restype = ctypes.c_bool
    nir_instr_is_last.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
class struct_nir_def(Structure):
    pass

class struct_list_head(Structure):
    pass

struct_list_head._pack_ = 1 # source:False
struct_list_head._fields_ = [
    ('prev', ctypes.POINTER(struct_list_head)),
    ('next', ctypes.POINTER(struct_list_head)),
]

struct_nir_def._pack_ = 1 # source:False
struct_nir_def._fields_ = [
    ('parent_instr', ctypes.POINTER(struct_nir_instr)),
    ('uses', struct_list_head),
    ('index', ctypes.c_uint32),
    ('num_components', ctypes.c_ubyte),
    ('bit_size', ctypes.c_ubyte),
    ('divergent', ctypes.c_bool),
    ('loop_invariant', ctypes.c_bool),
]

nir_def = struct_nir_def
class struct_nir_src(Structure):
    pass

struct_nir_src._pack_ = 1 # source:False
struct_nir_src._fields_ = [
    ('_parent', ctypes.c_uint64),
    ('use_link', struct_list_head),
    ('ssa', ctypes.POINTER(struct_nir_def)),
]

nir_src = struct_nir_src
try:
    nir_src_is_if = _libraries['FIXME_STUB'].nir_src_is_if
    nir_src_is_if.restype = ctypes.c_bool
    nir_src_is_if.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_src_parent_instr = _libraries['FIXME_STUB'].nir_src_parent_instr
    nir_src_parent_instr.restype = ctypes.POINTER(struct_nir_instr)
    nir_src_parent_instr.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
class struct_nir_if(Structure):
    pass


# values for enumeration 'c__EA_nir_selection_control'
c__EA_nir_selection_control__enumvalues = {
    0: 'nir_selection_control_none',
    1: 'nir_selection_control_flatten',
    2: 'nir_selection_control_dont_flatten',
    3: 'nir_selection_control_divergent_always_taken',
}
nir_selection_control_none = 0
nir_selection_control_flatten = 1
nir_selection_control_dont_flatten = 2
nir_selection_control_divergent_always_taken = 3
c__EA_nir_selection_control = ctypes.c_uint32 # enum
struct_nir_if._pack_ = 1 # source:False
struct_nir_if._fields_ = [
    ('cf_node', struct_nir_cf_node),
    ('condition', nir_src),
    ('control', c__EA_nir_selection_control),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('then_list', struct_exec_list),
    ('else_list', struct_exec_list),
]

try:
    nir_src_parent_if = _libraries['FIXME_STUB'].nir_src_parent_if
    nir_src_parent_if.restype = ctypes.POINTER(struct_nir_if)
    nir_src_parent_if.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    _nir_src_set_parent = _libraries['FIXME_STUB']._nir_src_set_parent
    _nir_src_set_parent.restype = None
    _nir_src_set_parent.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(None), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_src_set_parent_instr = _libraries['FIXME_STUB'].nir_src_set_parent_instr
    nir_src_set_parent_instr.restype = None
    nir_src_set_parent_instr.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_src_set_parent_if = _libraries['FIXME_STUB'].nir_src_set_parent_if
    nir_src_set_parent_if.restype = None
    nir_src_set_parent_if.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_src_init = _libraries['FIXME_STUB'].nir_src_init
    nir_src_init.restype = nir_src
    nir_src_init.argtypes = []
except AttributeError:
    pass
try:
    nir_def_used_by_if = _libraries['FIXME_STUB'].nir_def_used_by_if
    nir_def_used_by_if.restype = ctypes.c_bool
    nir_def_used_by_if.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_only_used_by_if = _libraries['FIXME_STUB'].nir_def_only_used_by_if
    nir_def_only_used_by_if.restype = ctypes.c_bool
    nir_def_only_used_by_if.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_src_for_ssa = _libraries['FIXME_STUB'].nir_src_for_ssa
    nir_src_for_ssa.restype = nir_src
    nir_src_for_ssa.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_src_bit_size = _libraries['FIXME_STUB'].nir_src_bit_size
    nir_src_bit_size.restype = ctypes.c_uint32
    nir_src_bit_size.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_num_components = _libraries['FIXME_STUB'].nir_src_num_components
    nir_src_num_components.restype = ctypes.c_uint32
    nir_src_num_components.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_is_const = _libraries['FIXME_STUB'].nir_src_is_const
    nir_src_is_const.restype = ctypes.c_bool
    nir_src_is_const.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_is_undef = _libraries['FIXME_STUB'].nir_src_is_undef
    nir_src_is_undef.restype = ctypes.c_bool
    nir_src_is_undef.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_is_divergent = _libraries['libtinymesa_cpu.so'].nir_src_is_divergent
    nir_src_is_divergent.restype = ctypes.c_bool
    nir_src_is_divergent.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_is_same_comp_swizzle = _libraries['FIXME_STUB'].nir_is_same_comp_swizzle
    nir_is_same_comp_swizzle.restype = ctypes.c_bool
    nir_is_same_comp_swizzle.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_sequential_comp_swizzle = _libraries['FIXME_STUB'].nir_is_sequential_comp_swizzle
    nir_is_sequential_comp_swizzle.restype = ctypes.c_bool
    nir_is_sequential_comp_swizzle.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32]
except AttributeError:
    pass
class struct_nir_alu_src(Structure):
    pass

struct_nir_alu_src._pack_ = 1 # source:False
struct_nir_alu_src._fields_ = [
    ('src', nir_src),
    ('swizzle', ctypes.c_ubyte * 16),
]

nir_alu_src = struct_nir_alu_src

# values for enumeration 'c__EA_nir_alu_type'
c__EA_nir_alu_type__enumvalues = {
    0: 'nir_type_invalid',
    2: 'nir_type_int',
    4: 'nir_type_uint',
    6: 'nir_type_bool',
    128: 'nir_type_float',
    7: 'nir_type_bool1',
    14: 'nir_type_bool8',
    22: 'nir_type_bool16',
    38: 'nir_type_bool32',
    3: 'nir_type_int1',
    10: 'nir_type_int8',
    18: 'nir_type_int16',
    34: 'nir_type_int32',
    66: 'nir_type_int64',
    5: 'nir_type_uint1',
    12: 'nir_type_uint8',
    20: 'nir_type_uint16',
    36: 'nir_type_uint32',
    68: 'nir_type_uint64',
    144: 'nir_type_float16',
    160: 'nir_type_float32',
    192: 'nir_type_float64',
}
nir_type_invalid = 0
nir_type_int = 2
nir_type_uint = 4
nir_type_bool = 6
nir_type_float = 128
nir_type_bool1 = 7
nir_type_bool8 = 14
nir_type_bool16 = 22
nir_type_bool32 = 38
nir_type_int1 = 3
nir_type_int8 = 10
nir_type_int16 = 18
nir_type_int32 = 34
nir_type_int64 = 66
nir_type_uint1 = 5
nir_type_uint8 = 12
nir_type_uint16 = 20
nir_type_uint32 = 36
nir_type_uint64 = 68
nir_type_float16 = 144
nir_type_float32 = 160
nir_type_float64 = 192
c__EA_nir_alu_type = ctypes.c_uint32 # enum
nir_alu_type = c__EA_nir_alu_type
nir_alu_type__enumvalues = c__EA_nir_alu_type__enumvalues
try:
    nir_get_nir_type_for_glsl_base_type = _libraries['libtinymesa_cpu.so'].nir_get_nir_type_for_glsl_base_type
    nir_get_nir_type_for_glsl_base_type.restype = nir_alu_type
    nir_get_nir_type_for_glsl_base_type.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    nir_get_nir_type_for_glsl_type = _libraries['FIXME_STUB'].nir_get_nir_type_for_glsl_type
    nir_get_nir_type_for_glsl_type.restype = nir_alu_type
    nir_get_nir_type_for_glsl_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_get_glsl_base_type_for_nir_type = _libraries['libtinymesa_cpu.so'].nir_get_glsl_base_type_for_nir_type
    nir_get_glsl_base_type_for_nir_type.restype = glsl_base_type
    nir_get_glsl_base_type_for_nir_type.argtypes = [nir_alu_type]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_op'
c__EA_nir_op__enumvalues = {
    0: 'nir_op_alignbyte_amd',
    1: 'nir_op_amul',
    2: 'nir_op_andg_ir3',
    3: 'nir_op_b16all_fequal16',
    4: 'nir_op_b16all_fequal2',
    5: 'nir_op_b16all_fequal3',
    6: 'nir_op_b16all_fequal4',
    7: 'nir_op_b16all_fequal5',
    8: 'nir_op_b16all_fequal8',
    9: 'nir_op_b16all_iequal16',
    10: 'nir_op_b16all_iequal2',
    11: 'nir_op_b16all_iequal3',
    12: 'nir_op_b16all_iequal4',
    13: 'nir_op_b16all_iequal5',
    14: 'nir_op_b16all_iequal8',
    15: 'nir_op_b16any_fnequal16',
    16: 'nir_op_b16any_fnequal2',
    17: 'nir_op_b16any_fnequal3',
    18: 'nir_op_b16any_fnequal4',
    19: 'nir_op_b16any_fnequal5',
    20: 'nir_op_b16any_fnequal8',
    21: 'nir_op_b16any_inequal16',
    22: 'nir_op_b16any_inequal2',
    23: 'nir_op_b16any_inequal3',
    24: 'nir_op_b16any_inequal4',
    25: 'nir_op_b16any_inequal5',
    26: 'nir_op_b16any_inequal8',
    27: 'nir_op_b16csel',
    28: 'nir_op_b2b1',
    29: 'nir_op_b2b16',
    30: 'nir_op_b2b32',
    31: 'nir_op_b2b8',
    32: 'nir_op_b2f16',
    33: 'nir_op_b2f32',
    34: 'nir_op_b2f64',
    35: 'nir_op_b2i1',
    36: 'nir_op_b2i16',
    37: 'nir_op_b2i32',
    38: 'nir_op_b2i64',
    39: 'nir_op_b2i8',
    40: 'nir_op_b32all_fequal16',
    41: 'nir_op_b32all_fequal2',
    42: 'nir_op_b32all_fequal3',
    43: 'nir_op_b32all_fequal4',
    44: 'nir_op_b32all_fequal5',
    45: 'nir_op_b32all_fequal8',
    46: 'nir_op_b32all_iequal16',
    47: 'nir_op_b32all_iequal2',
    48: 'nir_op_b32all_iequal3',
    49: 'nir_op_b32all_iequal4',
    50: 'nir_op_b32all_iequal5',
    51: 'nir_op_b32all_iequal8',
    52: 'nir_op_b32any_fnequal16',
    53: 'nir_op_b32any_fnequal2',
    54: 'nir_op_b32any_fnequal3',
    55: 'nir_op_b32any_fnequal4',
    56: 'nir_op_b32any_fnequal5',
    57: 'nir_op_b32any_fnequal8',
    58: 'nir_op_b32any_inequal16',
    59: 'nir_op_b32any_inequal2',
    60: 'nir_op_b32any_inequal3',
    61: 'nir_op_b32any_inequal4',
    62: 'nir_op_b32any_inequal5',
    63: 'nir_op_b32any_inequal8',
    64: 'nir_op_b32csel',
    65: 'nir_op_b32fcsel_mdg',
    66: 'nir_op_b8all_fequal16',
    67: 'nir_op_b8all_fequal2',
    68: 'nir_op_b8all_fequal3',
    69: 'nir_op_b8all_fequal4',
    70: 'nir_op_b8all_fequal5',
    71: 'nir_op_b8all_fequal8',
    72: 'nir_op_b8all_iequal16',
    73: 'nir_op_b8all_iequal2',
    74: 'nir_op_b8all_iequal3',
    75: 'nir_op_b8all_iequal4',
    76: 'nir_op_b8all_iequal5',
    77: 'nir_op_b8all_iequal8',
    78: 'nir_op_b8any_fnequal16',
    79: 'nir_op_b8any_fnequal2',
    80: 'nir_op_b8any_fnequal3',
    81: 'nir_op_b8any_fnequal4',
    82: 'nir_op_b8any_fnequal5',
    83: 'nir_op_b8any_fnequal8',
    84: 'nir_op_b8any_inequal16',
    85: 'nir_op_b8any_inequal2',
    86: 'nir_op_b8any_inequal3',
    87: 'nir_op_b8any_inequal4',
    88: 'nir_op_b8any_inequal5',
    89: 'nir_op_b8any_inequal8',
    90: 'nir_op_b8csel',
    91: 'nir_op_ball_fequal16',
    92: 'nir_op_ball_fequal2',
    93: 'nir_op_ball_fequal3',
    94: 'nir_op_ball_fequal4',
    95: 'nir_op_ball_fequal5',
    96: 'nir_op_ball_fequal8',
    97: 'nir_op_ball_iequal16',
    98: 'nir_op_ball_iequal2',
    99: 'nir_op_ball_iequal3',
    100: 'nir_op_ball_iequal4',
    101: 'nir_op_ball_iequal5',
    102: 'nir_op_ball_iequal8',
    103: 'nir_op_bany_fnequal16',
    104: 'nir_op_bany_fnequal2',
    105: 'nir_op_bany_fnequal3',
    106: 'nir_op_bany_fnequal4',
    107: 'nir_op_bany_fnequal5',
    108: 'nir_op_bany_fnequal8',
    109: 'nir_op_bany_inequal16',
    110: 'nir_op_bany_inequal2',
    111: 'nir_op_bany_inequal3',
    112: 'nir_op_bany_inequal4',
    113: 'nir_op_bany_inequal5',
    114: 'nir_op_bany_inequal8',
    115: 'nir_op_bcsel',
    116: 'nir_op_bf2f',
    117: 'nir_op_bfdot16',
    118: 'nir_op_bfdot2',
    119: 'nir_op_bfdot2_bfadd',
    120: 'nir_op_bfdot3',
    121: 'nir_op_bfdot4',
    122: 'nir_op_bfdot5',
    123: 'nir_op_bfdot8',
    124: 'nir_op_bffma',
    125: 'nir_op_bfi',
    126: 'nir_op_bfm',
    127: 'nir_op_bfmul',
    128: 'nir_op_bit_count',
    129: 'nir_op_bitfield_insert',
    130: 'nir_op_bitfield_reverse',
    131: 'nir_op_bitfield_select',
    132: 'nir_op_bitnz',
    133: 'nir_op_bitnz16',
    134: 'nir_op_bitnz32',
    135: 'nir_op_bitnz8',
    136: 'nir_op_bitz',
    137: 'nir_op_bitz16',
    138: 'nir_op_bitz32',
    139: 'nir_op_bitz8',
    140: 'nir_op_bounds_agx',
    141: 'nir_op_byte_perm_amd',
    142: 'nir_op_cube_amd',
    143: 'nir_op_e4m3fn2f',
    144: 'nir_op_e5m22f',
    145: 'nir_op_extr_agx',
    146: 'nir_op_extract_i16',
    147: 'nir_op_extract_i8',
    148: 'nir_op_extract_u16',
    149: 'nir_op_extract_u8',
    150: 'nir_op_f2bf',
    151: 'nir_op_f2e4m3fn',
    152: 'nir_op_f2e4m3fn_sat',
    153: 'nir_op_f2e4m3fn_satfn',
    154: 'nir_op_f2e5m2',
    155: 'nir_op_f2e5m2_sat',
    156: 'nir_op_f2f16',
    157: 'nir_op_f2f16_rtne',
    158: 'nir_op_f2f16_rtz',
    159: 'nir_op_f2f32',
    160: 'nir_op_f2f64',
    161: 'nir_op_f2fmp',
    162: 'nir_op_f2i1',
    163: 'nir_op_f2i16',
    164: 'nir_op_f2i32',
    165: 'nir_op_f2i64',
    166: 'nir_op_f2i8',
    167: 'nir_op_f2imp',
    168: 'nir_op_f2snorm_16_v3d',
    169: 'nir_op_f2u1',
    170: 'nir_op_f2u16',
    171: 'nir_op_f2u32',
    172: 'nir_op_f2u64',
    173: 'nir_op_f2u8',
    174: 'nir_op_f2ump',
    175: 'nir_op_f2unorm_16_v3d',
    176: 'nir_op_fabs',
    177: 'nir_op_fadd',
    178: 'nir_op_fall_equal16',
    179: 'nir_op_fall_equal2',
    180: 'nir_op_fall_equal3',
    181: 'nir_op_fall_equal4',
    182: 'nir_op_fall_equal5',
    183: 'nir_op_fall_equal8',
    184: 'nir_op_fany_nequal16',
    185: 'nir_op_fany_nequal2',
    186: 'nir_op_fany_nequal3',
    187: 'nir_op_fany_nequal4',
    188: 'nir_op_fany_nequal5',
    189: 'nir_op_fany_nequal8',
    190: 'nir_op_fceil',
    191: 'nir_op_fclamp_pos',
    192: 'nir_op_fcos',
    193: 'nir_op_fcos_amd',
    194: 'nir_op_fcos_mdg',
    195: 'nir_op_fcsel',
    196: 'nir_op_fcsel_ge',
    197: 'nir_op_fcsel_gt',
    198: 'nir_op_fdiv',
    199: 'nir_op_fdot16',
    200: 'nir_op_fdot16_replicated',
    201: 'nir_op_fdot2',
    202: 'nir_op_fdot2_replicated',
    203: 'nir_op_fdot3',
    204: 'nir_op_fdot3_replicated',
    205: 'nir_op_fdot4',
    206: 'nir_op_fdot4_replicated',
    207: 'nir_op_fdot5',
    208: 'nir_op_fdot5_replicated',
    209: 'nir_op_fdot8',
    210: 'nir_op_fdot8_replicated',
    211: 'nir_op_fdph',
    212: 'nir_op_fdph_replicated',
    213: 'nir_op_feq',
    214: 'nir_op_feq16',
    215: 'nir_op_feq32',
    216: 'nir_op_feq8',
    217: 'nir_op_fequ',
    218: 'nir_op_fequ16',
    219: 'nir_op_fequ32',
    220: 'nir_op_fequ8',
    221: 'nir_op_fexp2',
    222: 'nir_op_ffloor',
    223: 'nir_op_ffma',
    224: 'nir_op_ffmaz',
    225: 'nir_op_ffract',
    226: 'nir_op_fge',
    227: 'nir_op_fge16',
    228: 'nir_op_fge32',
    229: 'nir_op_fge8',
    230: 'nir_op_fgeu',
    231: 'nir_op_fgeu16',
    232: 'nir_op_fgeu32',
    233: 'nir_op_fgeu8',
    234: 'nir_op_find_lsb',
    235: 'nir_op_fisfinite',
    236: 'nir_op_fisfinite32',
    237: 'nir_op_fisnormal',
    238: 'nir_op_flog2',
    239: 'nir_op_flrp',
    240: 'nir_op_flt',
    241: 'nir_op_flt16',
    242: 'nir_op_flt32',
    243: 'nir_op_flt8',
    244: 'nir_op_fltu',
    245: 'nir_op_fltu16',
    246: 'nir_op_fltu32',
    247: 'nir_op_fltu8',
    248: 'nir_op_fmax',
    249: 'nir_op_fmax_agx',
    250: 'nir_op_fmin',
    251: 'nir_op_fmin_agx',
    252: 'nir_op_fmod',
    253: 'nir_op_fmul',
    254: 'nir_op_fmulz',
    255: 'nir_op_fneg',
    256: 'nir_op_fneo',
    257: 'nir_op_fneo16',
    258: 'nir_op_fneo32',
    259: 'nir_op_fneo8',
    260: 'nir_op_fneu',
    261: 'nir_op_fneu16',
    262: 'nir_op_fneu32',
    263: 'nir_op_fneu8',
    264: 'nir_op_ford',
    265: 'nir_op_ford16',
    266: 'nir_op_ford32',
    267: 'nir_op_ford8',
    268: 'nir_op_fpow',
    269: 'nir_op_fquantize2f16',
    270: 'nir_op_frcp',
    271: 'nir_op_frem',
    272: 'nir_op_frexp_exp',
    273: 'nir_op_frexp_sig',
    274: 'nir_op_fround_even',
    275: 'nir_op_frsq',
    276: 'nir_op_fsat',
    277: 'nir_op_fsat_signed',
    278: 'nir_op_fsign',
    279: 'nir_op_fsin',
    280: 'nir_op_fsin_agx',
    281: 'nir_op_fsin_amd',
    282: 'nir_op_fsin_mdg',
    283: 'nir_op_fsqrt',
    284: 'nir_op_fsub',
    285: 'nir_op_fsum2',
    286: 'nir_op_fsum3',
    287: 'nir_op_fsum4',
    288: 'nir_op_ftrunc',
    289: 'nir_op_funord',
    290: 'nir_op_funord16',
    291: 'nir_op_funord32',
    292: 'nir_op_funord8',
    293: 'nir_op_i2f16',
    294: 'nir_op_i2f32',
    295: 'nir_op_i2f64',
    296: 'nir_op_i2fmp',
    297: 'nir_op_i2i1',
    298: 'nir_op_i2i16',
    299: 'nir_op_i2i32',
    300: 'nir_op_i2i64',
    301: 'nir_op_i2i8',
    302: 'nir_op_i2imp',
    303: 'nir_op_i32csel_ge',
    304: 'nir_op_i32csel_gt',
    305: 'nir_op_iabs',
    306: 'nir_op_iadd',
    307: 'nir_op_iadd3',
    308: 'nir_op_iadd_sat',
    309: 'nir_op_iand',
    310: 'nir_op_ibfe',
    311: 'nir_op_ibitfield_extract',
    312: 'nir_op_icsel_eqz',
    313: 'nir_op_idiv',
    314: 'nir_op_ieq',
    315: 'nir_op_ieq16',
    316: 'nir_op_ieq32',
    317: 'nir_op_ieq8',
    318: 'nir_op_ifind_msb',
    319: 'nir_op_ifind_msb_rev',
    320: 'nir_op_ige',
    321: 'nir_op_ige16',
    322: 'nir_op_ige32',
    323: 'nir_op_ige8',
    324: 'nir_op_ihadd',
    325: 'nir_op_ilea_agx',
    326: 'nir_op_ilt',
    327: 'nir_op_ilt16',
    328: 'nir_op_ilt32',
    329: 'nir_op_ilt8',
    330: 'nir_op_imad',
    331: 'nir_op_imad24_ir3',
    332: 'nir_op_imadsh_mix16',
    333: 'nir_op_imadshl_agx',
    334: 'nir_op_imax',
    335: 'nir_op_imin',
    336: 'nir_op_imod',
    337: 'nir_op_imsubshl_agx',
    338: 'nir_op_imul',
    339: 'nir_op_imul24',
    340: 'nir_op_imul24_relaxed',
    341: 'nir_op_imul_2x32_64',
    342: 'nir_op_imul_32x16',
    343: 'nir_op_imul_high',
    344: 'nir_op_ine',
    345: 'nir_op_ine16',
    346: 'nir_op_ine32',
    347: 'nir_op_ine8',
    348: 'nir_op_ineg',
    349: 'nir_op_inot',
    350: 'nir_op_insert_u16',
    351: 'nir_op_insert_u8',
    352: 'nir_op_interleave_agx',
    353: 'nir_op_ior',
    354: 'nir_op_irem',
    355: 'nir_op_irhadd',
    356: 'nir_op_ishl',
    357: 'nir_op_ishr',
    358: 'nir_op_isign',
    359: 'nir_op_isub',
    360: 'nir_op_isub_sat',
    361: 'nir_op_ixor',
    362: 'nir_op_ldexp',
    363: 'nir_op_ldexp16_pan',
    364: 'nir_op_lea_nv',
    365: 'nir_op_mov',
    366: 'nir_op_mqsad_4x8',
    367: 'nir_op_msad_4x8',
    368: 'nir_op_pack_2x16_to_snorm_2x8_v3d',
    369: 'nir_op_pack_2x16_to_unorm_10_2_v3d',
    370: 'nir_op_pack_2x16_to_unorm_2x10_v3d',
    371: 'nir_op_pack_2x16_to_unorm_2x8_v3d',
    372: 'nir_op_pack_2x32_to_2x16_v3d',
    373: 'nir_op_pack_32_2x16',
    374: 'nir_op_pack_32_2x16_split',
    375: 'nir_op_pack_32_4x8',
    376: 'nir_op_pack_32_4x8_split',
    377: 'nir_op_pack_32_to_r11g11b10_v3d',
    378: 'nir_op_pack_4x16_to_4x8_v3d',
    379: 'nir_op_pack_64_2x32',
    380: 'nir_op_pack_64_2x32_split',
    381: 'nir_op_pack_64_4x16',
    382: 'nir_op_pack_double_2x32_dxil',
    383: 'nir_op_pack_half_2x16',
    384: 'nir_op_pack_half_2x16_rtz_split',
    385: 'nir_op_pack_half_2x16_split',
    386: 'nir_op_pack_sint_2x16',
    387: 'nir_op_pack_snorm_2x16',
    388: 'nir_op_pack_snorm_4x8',
    389: 'nir_op_pack_uint_2x16',
    390: 'nir_op_pack_uint_32_to_r10g10b10a2_v3d',
    391: 'nir_op_pack_unorm_2x16',
    392: 'nir_op_pack_unorm_4x8',
    393: 'nir_op_pack_uvec2_to_uint',
    394: 'nir_op_pack_uvec4_to_uint',
    395: 'nir_op_prmt_nv',
    396: 'nir_op_sdot_2x16_iadd',
    397: 'nir_op_sdot_2x16_iadd_sat',
    398: 'nir_op_sdot_4x8_iadd',
    399: 'nir_op_sdot_4x8_iadd_sat',
    400: 'nir_op_seq',
    401: 'nir_op_sge',
    402: 'nir_op_shfr',
    403: 'nir_op_shlg_ir3',
    404: 'nir_op_shlm_ir3',
    405: 'nir_op_shrg_ir3',
    406: 'nir_op_shrm_ir3',
    407: 'nir_op_slt',
    408: 'nir_op_sne',
    409: 'nir_op_sudot_4x8_iadd',
    410: 'nir_op_sudot_4x8_iadd_sat',
    411: 'nir_op_u2f16',
    412: 'nir_op_u2f32',
    413: 'nir_op_u2f64',
    414: 'nir_op_u2fmp',
    415: 'nir_op_u2u1',
    416: 'nir_op_u2u16',
    417: 'nir_op_u2u32',
    418: 'nir_op_u2u64',
    419: 'nir_op_u2u8',
    420: 'nir_op_uabs_isub',
    421: 'nir_op_uabs_usub',
    422: 'nir_op_uadd_carry',
    423: 'nir_op_uadd_sat',
    424: 'nir_op_ubfe',
    425: 'nir_op_ubitfield_extract',
    426: 'nir_op_uclz',
    427: 'nir_op_udiv',
    428: 'nir_op_udiv_aligned_4',
    429: 'nir_op_udot_2x16_uadd',
    430: 'nir_op_udot_2x16_uadd_sat',
    431: 'nir_op_udot_4x8_uadd',
    432: 'nir_op_udot_4x8_uadd_sat',
    433: 'nir_op_ufind_msb',
    434: 'nir_op_ufind_msb_rev',
    435: 'nir_op_uge',
    436: 'nir_op_uge16',
    437: 'nir_op_uge32',
    438: 'nir_op_uge8',
    439: 'nir_op_uhadd',
    440: 'nir_op_ulea_agx',
    441: 'nir_op_ult',
    442: 'nir_op_ult16',
    443: 'nir_op_ult32',
    444: 'nir_op_ult8',
    445: 'nir_op_umad24',
    446: 'nir_op_umad24_relaxed',
    447: 'nir_op_umax',
    448: 'nir_op_umax_4x8_vc4',
    449: 'nir_op_umin',
    450: 'nir_op_umin_4x8_vc4',
    451: 'nir_op_umod',
    452: 'nir_op_umul24',
    453: 'nir_op_umul24_relaxed',
    454: 'nir_op_umul_2x32_64',
    455: 'nir_op_umul_32x16',
    456: 'nir_op_umul_high',
    457: 'nir_op_umul_low',
    458: 'nir_op_umul_unorm_4x8_vc4',
    459: 'nir_op_unpack_32_2x16',
    460: 'nir_op_unpack_32_2x16_split_x',
    461: 'nir_op_unpack_32_2x16_split_y',
    462: 'nir_op_unpack_32_4x8',
    463: 'nir_op_unpack_64_2x32',
    464: 'nir_op_unpack_64_2x32_split_x',
    465: 'nir_op_unpack_64_2x32_split_y',
    466: 'nir_op_unpack_64_4x16',
    467: 'nir_op_unpack_double_2x32_dxil',
    468: 'nir_op_unpack_half_2x16',
    469: 'nir_op_unpack_half_2x16_split_x',
    470: 'nir_op_unpack_half_2x16_split_y',
    471: 'nir_op_unpack_snorm_2x16',
    472: 'nir_op_unpack_snorm_4x8',
    473: 'nir_op_unpack_unorm_2x16',
    474: 'nir_op_unpack_unorm_4x8',
    475: 'nir_op_urhadd',
    476: 'nir_op_urol',
    477: 'nir_op_uror',
    478: 'nir_op_usadd_4x8_vc4',
    479: 'nir_op_ushr',
    480: 'nir_op_ussub_4x8_vc4',
    481: 'nir_op_usub_borrow',
    482: 'nir_op_usub_sat',
    483: 'nir_op_vec16',
    484: 'nir_op_vec2',
    485: 'nir_op_vec3',
    486: 'nir_op_vec4',
    487: 'nir_op_vec5',
    488: 'nir_op_vec8',
    488: 'nir_last_opcode',
    489: 'nir_num_opcodes',
}
nir_op_alignbyte_amd = 0
nir_op_amul = 1
nir_op_andg_ir3 = 2
nir_op_b16all_fequal16 = 3
nir_op_b16all_fequal2 = 4
nir_op_b16all_fequal3 = 5
nir_op_b16all_fequal4 = 6
nir_op_b16all_fequal5 = 7
nir_op_b16all_fequal8 = 8
nir_op_b16all_iequal16 = 9
nir_op_b16all_iequal2 = 10
nir_op_b16all_iequal3 = 11
nir_op_b16all_iequal4 = 12
nir_op_b16all_iequal5 = 13
nir_op_b16all_iequal8 = 14
nir_op_b16any_fnequal16 = 15
nir_op_b16any_fnequal2 = 16
nir_op_b16any_fnequal3 = 17
nir_op_b16any_fnequal4 = 18
nir_op_b16any_fnequal5 = 19
nir_op_b16any_fnequal8 = 20
nir_op_b16any_inequal16 = 21
nir_op_b16any_inequal2 = 22
nir_op_b16any_inequal3 = 23
nir_op_b16any_inequal4 = 24
nir_op_b16any_inequal5 = 25
nir_op_b16any_inequal8 = 26
nir_op_b16csel = 27
nir_op_b2b1 = 28
nir_op_b2b16 = 29
nir_op_b2b32 = 30
nir_op_b2b8 = 31
nir_op_b2f16 = 32
nir_op_b2f32 = 33
nir_op_b2f64 = 34
nir_op_b2i1 = 35
nir_op_b2i16 = 36
nir_op_b2i32 = 37
nir_op_b2i64 = 38
nir_op_b2i8 = 39
nir_op_b32all_fequal16 = 40
nir_op_b32all_fequal2 = 41
nir_op_b32all_fequal3 = 42
nir_op_b32all_fequal4 = 43
nir_op_b32all_fequal5 = 44
nir_op_b32all_fequal8 = 45
nir_op_b32all_iequal16 = 46
nir_op_b32all_iequal2 = 47
nir_op_b32all_iequal3 = 48
nir_op_b32all_iequal4 = 49
nir_op_b32all_iequal5 = 50
nir_op_b32all_iequal8 = 51
nir_op_b32any_fnequal16 = 52
nir_op_b32any_fnequal2 = 53
nir_op_b32any_fnequal3 = 54
nir_op_b32any_fnequal4 = 55
nir_op_b32any_fnequal5 = 56
nir_op_b32any_fnequal8 = 57
nir_op_b32any_inequal16 = 58
nir_op_b32any_inequal2 = 59
nir_op_b32any_inequal3 = 60
nir_op_b32any_inequal4 = 61
nir_op_b32any_inequal5 = 62
nir_op_b32any_inequal8 = 63
nir_op_b32csel = 64
nir_op_b32fcsel_mdg = 65
nir_op_b8all_fequal16 = 66
nir_op_b8all_fequal2 = 67
nir_op_b8all_fequal3 = 68
nir_op_b8all_fequal4 = 69
nir_op_b8all_fequal5 = 70
nir_op_b8all_fequal8 = 71
nir_op_b8all_iequal16 = 72
nir_op_b8all_iequal2 = 73
nir_op_b8all_iequal3 = 74
nir_op_b8all_iequal4 = 75
nir_op_b8all_iequal5 = 76
nir_op_b8all_iequal8 = 77
nir_op_b8any_fnequal16 = 78
nir_op_b8any_fnequal2 = 79
nir_op_b8any_fnequal3 = 80
nir_op_b8any_fnequal4 = 81
nir_op_b8any_fnequal5 = 82
nir_op_b8any_fnequal8 = 83
nir_op_b8any_inequal16 = 84
nir_op_b8any_inequal2 = 85
nir_op_b8any_inequal3 = 86
nir_op_b8any_inequal4 = 87
nir_op_b8any_inequal5 = 88
nir_op_b8any_inequal8 = 89
nir_op_b8csel = 90
nir_op_ball_fequal16 = 91
nir_op_ball_fequal2 = 92
nir_op_ball_fequal3 = 93
nir_op_ball_fequal4 = 94
nir_op_ball_fequal5 = 95
nir_op_ball_fequal8 = 96
nir_op_ball_iequal16 = 97
nir_op_ball_iequal2 = 98
nir_op_ball_iequal3 = 99
nir_op_ball_iequal4 = 100
nir_op_ball_iequal5 = 101
nir_op_ball_iequal8 = 102
nir_op_bany_fnequal16 = 103
nir_op_bany_fnequal2 = 104
nir_op_bany_fnequal3 = 105
nir_op_bany_fnequal4 = 106
nir_op_bany_fnequal5 = 107
nir_op_bany_fnequal8 = 108
nir_op_bany_inequal16 = 109
nir_op_bany_inequal2 = 110
nir_op_bany_inequal3 = 111
nir_op_bany_inequal4 = 112
nir_op_bany_inequal5 = 113
nir_op_bany_inequal8 = 114
nir_op_bcsel = 115
nir_op_bf2f = 116
nir_op_bfdot16 = 117
nir_op_bfdot2 = 118
nir_op_bfdot2_bfadd = 119
nir_op_bfdot3 = 120
nir_op_bfdot4 = 121
nir_op_bfdot5 = 122
nir_op_bfdot8 = 123
nir_op_bffma = 124
nir_op_bfi = 125
nir_op_bfm = 126
nir_op_bfmul = 127
nir_op_bit_count = 128
nir_op_bitfield_insert = 129
nir_op_bitfield_reverse = 130
nir_op_bitfield_select = 131
nir_op_bitnz = 132
nir_op_bitnz16 = 133
nir_op_bitnz32 = 134
nir_op_bitnz8 = 135
nir_op_bitz = 136
nir_op_bitz16 = 137
nir_op_bitz32 = 138
nir_op_bitz8 = 139
nir_op_bounds_agx = 140
nir_op_byte_perm_amd = 141
nir_op_cube_amd = 142
nir_op_e4m3fn2f = 143
nir_op_e5m22f = 144
nir_op_extr_agx = 145
nir_op_extract_i16 = 146
nir_op_extract_i8 = 147
nir_op_extract_u16 = 148
nir_op_extract_u8 = 149
nir_op_f2bf = 150
nir_op_f2e4m3fn = 151
nir_op_f2e4m3fn_sat = 152
nir_op_f2e4m3fn_satfn = 153
nir_op_f2e5m2 = 154
nir_op_f2e5m2_sat = 155
nir_op_f2f16 = 156
nir_op_f2f16_rtne = 157
nir_op_f2f16_rtz = 158
nir_op_f2f32 = 159
nir_op_f2f64 = 160
nir_op_f2fmp = 161
nir_op_f2i1 = 162
nir_op_f2i16 = 163
nir_op_f2i32 = 164
nir_op_f2i64 = 165
nir_op_f2i8 = 166
nir_op_f2imp = 167
nir_op_f2snorm_16_v3d = 168
nir_op_f2u1 = 169
nir_op_f2u16 = 170
nir_op_f2u32 = 171
nir_op_f2u64 = 172
nir_op_f2u8 = 173
nir_op_f2ump = 174
nir_op_f2unorm_16_v3d = 175
nir_op_fabs = 176
nir_op_fadd = 177
nir_op_fall_equal16 = 178
nir_op_fall_equal2 = 179
nir_op_fall_equal3 = 180
nir_op_fall_equal4 = 181
nir_op_fall_equal5 = 182
nir_op_fall_equal8 = 183
nir_op_fany_nequal16 = 184
nir_op_fany_nequal2 = 185
nir_op_fany_nequal3 = 186
nir_op_fany_nequal4 = 187
nir_op_fany_nequal5 = 188
nir_op_fany_nequal8 = 189
nir_op_fceil = 190
nir_op_fclamp_pos = 191
nir_op_fcos = 192
nir_op_fcos_amd = 193
nir_op_fcos_mdg = 194
nir_op_fcsel = 195
nir_op_fcsel_ge = 196
nir_op_fcsel_gt = 197
nir_op_fdiv = 198
nir_op_fdot16 = 199
nir_op_fdot16_replicated = 200
nir_op_fdot2 = 201
nir_op_fdot2_replicated = 202
nir_op_fdot3 = 203
nir_op_fdot3_replicated = 204
nir_op_fdot4 = 205
nir_op_fdot4_replicated = 206
nir_op_fdot5 = 207
nir_op_fdot5_replicated = 208
nir_op_fdot8 = 209
nir_op_fdot8_replicated = 210
nir_op_fdph = 211
nir_op_fdph_replicated = 212
nir_op_feq = 213
nir_op_feq16 = 214
nir_op_feq32 = 215
nir_op_feq8 = 216
nir_op_fequ = 217
nir_op_fequ16 = 218
nir_op_fequ32 = 219
nir_op_fequ8 = 220
nir_op_fexp2 = 221
nir_op_ffloor = 222
nir_op_ffma = 223
nir_op_ffmaz = 224
nir_op_ffract = 225
nir_op_fge = 226
nir_op_fge16 = 227
nir_op_fge32 = 228
nir_op_fge8 = 229
nir_op_fgeu = 230
nir_op_fgeu16 = 231
nir_op_fgeu32 = 232
nir_op_fgeu8 = 233
nir_op_find_lsb = 234
nir_op_fisfinite = 235
nir_op_fisfinite32 = 236
nir_op_fisnormal = 237
nir_op_flog2 = 238
nir_op_flrp = 239
nir_op_flt = 240
nir_op_flt16 = 241
nir_op_flt32 = 242
nir_op_flt8 = 243
nir_op_fltu = 244
nir_op_fltu16 = 245
nir_op_fltu32 = 246
nir_op_fltu8 = 247
nir_op_fmax = 248
nir_op_fmax_agx = 249
nir_op_fmin = 250
nir_op_fmin_agx = 251
nir_op_fmod = 252
nir_op_fmul = 253
nir_op_fmulz = 254
nir_op_fneg = 255
nir_op_fneo = 256
nir_op_fneo16 = 257
nir_op_fneo32 = 258
nir_op_fneo8 = 259
nir_op_fneu = 260
nir_op_fneu16 = 261
nir_op_fneu32 = 262
nir_op_fneu8 = 263
nir_op_ford = 264
nir_op_ford16 = 265
nir_op_ford32 = 266
nir_op_ford8 = 267
nir_op_fpow = 268
nir_op_fquantize2f16 = 269
nir_op_frcp = 270
nir_op_frem = 271
nir_op_frexp_exp = 272
nir_op_frexp_sig = 273
nir_op_fround_even = 274
nir_op_frsq = 275
nir_op_fsat = 276
nir_op_fsat_signed = 277
nir_op_fsign = 278
nir_op_fsin = 279
nir_op_fsin_agx = 280
nir_op_fsin_amd = 281
nir_op_fsin_mdg = 282
nir_op_fsqrt = 283
nir_op_fsub = 284
nir_op_fsum2 = 285
nir_op_fsum3 = 286
nir_op_fsum4 = 287
nir_op_ftrunc = 288
nir_op_funord = 289
nir_op_funord16 = 290
nir_op_funord32 = 291
nir_op_funord8 = 292
nir_op_i2f16 = 293
nir_op_i2f32 = 294
nir_op_i2f64 = 295
nir_op_i2fmp = 296
nir_op_i2i1 = 297
nir_op_i2i16 = 298
nir_op_i2i32 = 299
nir_op_i2i64 = 300
nir_op_i2i8 = 301
nir_op_i2imp = 302
nir_op_i32csel_ge = 303
nir_op_i32csel_gt = 304
nir_op_iabs = 305
nir_op_iadd = 306
nir_op_iadd3 = 307
nir_op_iadd_sat = 308
nir_op_iand = 309
nir_op_ibfe = 310
nir_op_ibitfield_extract = 311
nir_op_icsel_eqz = 312
nir_op_idiv = 313
nir_op_ieq = 314
nir_op_ieq16 = 315
nir_op_ieq32 = 316
nir_op_ieq8 = 317
nir_op_ifind_msb = 318
nir_op_ifind_msb_rev = 319
nir_op_ige = 320
nir_op_ige16 = 321
nir_op_ige32 = 322
nir_op_ige8 = 323
nir_op_ihadd = 324
nir_op_ilea_agx = 325
nir_op_ilt = 326
nir_op_ilt16 = 327
nir_op_ilt32 = 328
nir_op_ilt8 = 329
nir_op_imad = 330
nir_op_imad24_ir3 = 331
nir_op_imadsh_mix16 = 332
nir_op_imadshl_agx = 333
nir_op_imax = 334
nir_op_imin = 335
nir_op_imod = 336
nir_op_imsubshl_agx = 337
nir_op_imul = 338
nir_op_imul24 = 339
nir_op_imul24_relaxed = 340
nir_op_imul_2x32_64 = 341
nir_op_imul_32x16 = 342
nir_op_imul_high = 343
nir_op_ine = 344
nir_op_ine16 = 345
nir_op_ine32 = 346
nir_op_ine8 = 347
nir_op_ineg = 348
nir_op_inot = 349
nir_op_insert_u16 = 350
nir_op_insert_u8 = 351
nir_op_interleave_agx = 352
nir_op_ior = 353
nir_op_irem = 354
nir_op_irhadd = 355
nir_op_ishl = 356
nir_op_ishr = 357
nir_op_isign = 358
nir_op_isub = 359
nir_op_isub_sat = 360
nir_op_ixor = 361
nir_op_ldexp = 362
nir_op_ldexp16_pan = 363
nir_op_lea_nv = 364
nir_op_mov = 365
nir_op_mqsad_4x8 = 366
nir_op_msad_4x8 = 367
nir_op_pack_2x16_to_snorm_2x8_v3d = 368
nir_op_pack_2x16_to_unorm_10_2_v3d = 369
nir_op_pack_2x16_to_unorm_2x10_v3d = 370
nir_op_pack_2x16_to_unorm_2x8_v3d = 371
nir_op_pack_2x32_to_2x16_v3d = 372
nir_op_pack_32_2x16 = 373
nir_op_pack_32_2x16_split = 374
nir_op_pack_32_4x8 = 375
nir_op_pack_32_4x8_split = 376
nir_op_pack_32_to_r11g11b10_v3d = 377
nir_op_pack_4x16_to_4x8_v3d = 378
nir_op_pack_64_2x32 = 379
nir_op_pack_64_2x32_split = 380
nir_op_pack_64_4x16 = 381
nir_op_pack_double_2x32_dxil = 382
nir_op_pack_half_2x16 = 383
nir_op_pack_half_2x16_rtz_split = 384
nir_op_pack_half_2x16_split = 385
nir_op_pack_sint_2x16 = 386
nir_op_pack_snorm_2x16 = 387
nir_op_pack_snorm_4x8 = 388
nir_op_pack_uint_2x16 = 389
nir_op_pack_uint_32_to_r10g10b10a2_v3d = 390
nir_op_pack_unorm_2x16 = 391
nir_op_pack_unorm_4x8 = 392
nir_op_pack_uvec2_to_uint = 393
nir_op_pack_uvec4_to_uint = 394
nir_op_prmt_nv = 395
nir_op_sdot_2x16_iadd = 396
nir_op_sdot_2x16_iadd_sat = 397
nir_op_sdot_4x8_iadd = 398
nir_op_sdot_4x8_iadd_sat = 399
nir_op_seq = 400
nir_op_sge = 401
nir_op_shfr = 402
nir_op_shlg_ir3 = 403
nir_op_shlm_ir3 = 404
nir_op_shrg_ir3 = 405
nir_op_shrm_ir3 = 406
nir_op_slt = 407
nir_op_sne = 408
nir_op_sudot_4x8_iadd = 409
nir_op_sudot_4x8_iadd_sat = 410
nir_op_u2f16 = 411
nir_op_u2f32 = 412
nir_op_u2f64 = 413
nir_op_u2fmp = 414
nir_op_u2u1 = 415
nir_op_u2u16 = 416
nir_op_u2u32 = 417
nir_op_u2u64 = 418
nir_op_u2u8 = 419
nir_op_uabs_isub = 420
nir_op_uabs_usub = 421
nir_op_uadd_carry = 422
nir_op_uadd_sat = 423
nir_op_ubfe = 424
nir_op_ubitfield_extract = 425
nir_op_uclz = 426
nir_op_udiv = 427
nir_op_udiv_aligned_4 = 428
nir_op_udot_2x16_uadd = 429
nir_op_udot_2x16_uadd_sat = 430
nir_op_udot_4x8_uadd = 431
nir_op_udot_4x8_uadd_sat = 432
nir_op_ufind_msb = 433
nir_op_ufind_msb_rev = 434
nir_op_uge = 435
nir_op_uge16 = 436
nir_op_uge32 = 437
nir_op_uge8 = 438
nir_op_uhadd = 439
nir_op_ulea_agx = 440
nir_op_ult = 441
nir_op_ult16 = 442
nir_op_ult32 = 443
nir_op_ult8 = 444
nir_op_umad24 = 445
nir_op_umad24_relaxed = 446
nir_op_umax = 447
nir_op_umax_4x8_vc4 = 448
nir_op_umin = 449
nir_op_umin_4x8_vc4 = 450
nir_op_umod = 451
nir_op_umul24 = 452
nir_op_umul24_relaxed = 453
nir_op_umul_2x32_64 = 454
nir_op_umul_32x16 = 455
nir_op_umul_high = 456
nir_op_umul_low = 457
nir_op_umul_unorm_4x8_vc4 = 458
nir_op_unpack_32_2x16 = 459
nir_op_unpack_32_2x16_split_x = 460
nir_op_unpack_32_2x16_split_y = 461
nir_op_unpack_32_4x8 = 462
nir_op_unpack_64_2x32 = 463
nir_op_unpack_64_2x32_split_x = 464
nir_op_unpack_64_2x32_split_y = 465
nir_op_unpack_64_4x16 = 466
nir_op_unpack_double_2x32_dxil = 467
nir_op_unpack_half_2x16 = 468
nir_op_unpack_half_2x16_split_x = 469
nir_op_unpack_half_2x16_split_y = 470
nir_op_unpack_snorm_2x16 = 471
nir_op_unpack_snorm_4x8 = 472
nir_op_unpack_unorm_2x16 = 473
nir_op_unpack_unorm_4x8 = 474
nir_op_urhadd = 475
nir_op_urol = 476
nir_op_uror = 477
nir_op_usadd_4x8_vc4 = 478
nir_op_ushr = 479
nir_op_ussub_4x8_vc4 = 480
nir_op_usub_borrow = 481
nir_op_usub_sat = 482
nir_op_vec16 = 483
nir_op_vec2 = 484
nir_op_vec3 = 485
nir_op_vec4 = 486
nir_op_vec5 = 487
nir_op_vec8 = 488
nir_last_opcode = 488
nir_num_opcodes = 489
c__EA_nir_op = ctypes.c_uint32 # enum
nir_op = c__EA_nir_op
nir_op__enumvalues = c__EA_nir_op__enumvalues
try:
    nir_type_conversion_op = _libraries['libtinymesa_cpu.so'].nir_type_conversion_op
    nir_type_conversion_op.restype = nir_op
    nir_type_conversion_op.argtypes = [nir_alu_type, nir_alu_type, nir_rounding_mode]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_atomic_op'
c__EA_nir_atomic_op__enumvalues = {
    0: 'nir_atomic_op_iadd',
    1: 'nir_atomic_op_imin',
    2: 'nir_atomic_op_umin',
    3: 'nir_atomic_op_imax',
    4: 'nir_atomic_op_umax',
    5: 'nir_atomic_op_iand',
    6: 'nir_atomic_op_ior',
    7: 'nir_atomic_op_ixor',
    8: 'nir_atomic_op_xchg',
    9: 'nir_atomic_op_fadd',
    10: 'nir_atomic_op_fmin',
    11: 'nir_atomic_op_fmax',
    12: 'nir_atomic_op_cmpxchg',
    13: 'nir_atomic_op_fcmpxchg',
    14: 'nir_atomic_op_inc_wrap',
    15: 'nir_atomic_op_dec_wrap',
    16: 'nir_atomic_op_ordered_add_gfx12_amd',
}
nir_atomic_op_iadd = 0
nir_atomic_op_imin = 1
nir_atomic_op_umin = 2
nir_atomic_op_imax = 3
nir_atomic_op_umax = 4
nir_atomic_op_iand = 5
nir_atomic_op_ior = 6
nir_atomic_op_ixor = 7
nir_atomic_op_xchg = 8
nir_atomic_op_fadd = 9
nir_atomic_op_fmin = 10
nir_atomic_op_fmax = 11
nir_atomic_op_cmpxchg = 12
nir_atomic_op_fcmpxchg = 13
nir_atomic_op_inc_wrap = 14
nir_atomic_op_dec_wrap = 15
nir_atomic_op_ordered_add_gfx12_amd = 16
c__EA_nir_atomic_op = ctypes.c_uint32 # enum
nir_atomic_op = c__EA_nir_atomic_op
nir_atomic_op__enumvalues = c__EA_nir_atomic_op__enumvalues
try:
    nir_atomic_op_type = _libraries['FIXME_STUB'].nir_atomic_op_type
    nir_atomic_op_type.restype = nir_alu_type
    nir_atomic_op_type.argtypes = [nir_atomic_op]
except AttributeError:
    pass
try:
    nir_atomic_op_to_alu = _libraries['libtinymesa_cpu.so'].nir_atomic_op_to_alu
    nir_atomic_op_to_alu.restype = nir_op
    nir_atomic_op_to_alu.argtypes = [nir_atomic_op]
except AttributeError:
    pass
try:
    nir_op_vec = _libraries['libtinymesa_cpu.so'].nir_op_vec
    nir_op_vec.restype = nir_op
    nir_op_vec.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_op_is_vec = _libraries['libtinymesa_cpu.so'].nir_op_is_vec
    nir_op_is_vec.restype = ctypes.c_bool
    nir_op_is_vec.argtypes = [nir_op]
except AttributeError:
    pass
try:
    nir_op_is_vec_or_mov = _libraries['FIXME_STUB'].nir_op_is_vec_or_mov
    nir_op_is_vec_or_mov.restype = ctypes.c_bool
    nir_op_is_vec_or_mov.argtypes = [nir_op]
except AttributeError:
    pass
try:
    nir_is_float_control_signed_zero_preserve = _libraries['FIXME_STUB'].nir_is_float_control_signed_zero_preserve
    nir_is_float_control_signed_zero_preserve.restype = ctypes.c_bool
    nir_is_float_control_signed_zero_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_float_control_inf_preserve = _libraries['FIXME_STUB'].nir_is_float_control_inf_preserve
    nir_is_float_control_inf_preserve.restype = ctypes.c_bool
    nir_is_float_control_inf_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_float_control_nan_preserve = _libraries['FIXME_STUB'].nir_is_float_control_nan_preserve
    nir_is_float_control_nan_preserve.restype = ctypes.c_bool
    nir_is_float_control_nan_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_float_control_signed_zero_inf_nan_preserve = _libraries['FIXME_STUB'].nir_is_float_control_signed_zero_inf_nan_preserve
    nir_is_float_control_signed_zero_inf_nan_preserve.restype = ctypes.c_bool
    nir_is_float_control_signed_zero_inf_nan_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_denorm_flush_to_zero = _libraries['FIXME_STUB'].nir_is_denorm_flush_to_zero
    nir_is_denorm_flush_to_zero.restype = ctypes.c_bool
    nir_is_denorm_flush_to_zero.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_denorm_preserve = _libraries['FIXME_STUB'].nir_is_denorm_preserve
    nir_is_denorm_preserve.restype = ctypes.c_bool
    nir_is_denorm_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_rounding_mode_rtne = _libraries['FIXME_STUB'].nir_is_rounding_mode_rtne
    nir_is_rounding_mode_rtne.restype = ctypes.c_bool
    nir_is_rounding_mode_rtne.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_rounding_mode_rtz = _libraries['FIXME_STUB'].nir_is_rounding_mode_rtz
    nir_is_rounding_mode_rtz.restype = ctypes.c_bool
    nir_is_rounding_mode_rtz.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_has_any_rounding_mode_rtz = _libraries['FIXME_STUB'].nir_has_any_rounding_mode_rtz
    nir_has_any_rounding_mode_rtz.restype = ctypes.c_bool
    nir_has_any_rounding_mode_rtz.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_has_any_rounding_mode_rtne = _libraries['FIXME_STUB'].nir_has_any_rounding_mode_rtne
    nir_has_any_rounding_mode_rtne.restype = ctypes.c_bool
    nir_has_any_rounding_mode_rtne.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_get_rounding_mode_from_float_controls = _libraries['FIXME_STUB'].nir_get_rounding_mode_from_float_controls
    nir_get_rounding_mode_from_float_controls.restype = nir_rounding_mode
    nir_get_rounding_mode_from_float_controls.argtypes = [ctypes.c_uint32, nir_alu_type]
except AttributeError:
    pass
try:
    nir_has_any_rounding_mode_enabled = _libraries['FIXME_STUB'].nir_has_any_rounding_mode_enabled
    nir_has_any_rounding_mode_enabled.restype = ctypes.c_bool
    nir_has_any_rounding_mode_enabled.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_op_algebraic_property'
c__EA_nir_op_algebraic_property__enumvalues = {
    1: 'NIR_OP_IS_2SRC_COMMUTATIVE',
    2: 'NIR_OP_IS_ASSOCIATIVE',
    4: 'NIR_OP_IS_SELECTION',
}
NIR_OP_IS_2SRC_COMMUTATIVE = 1
NIR_OP_IS_ASSOCIATIVE = 2
NIR_OP_IS_SELECTION = 4
c__EA_nir_op_algebraic_property = ctypes.c_uint32 # enum
nir_op_algebraic_property = c__EA_nir_op_algebraic_property
nir_op_algebraic_property__enumvalues = c__EA_nir_op_algebraic_property__enumvalues
class struct_nir_op_info(Structure):
    pass

struct_nir_op_info._pack_ = 1 # source:False
struct_nir_op_info._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('num_inputs', ctypes.c_ubyte),
    ('output_size', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('output_type', nir_alu_type),
    ('input_sizes', ctypes.c_ubyte * 16),
    ('input_types', c__EA_nir_alu_type * 16),
    ('algebraic_properties', nir_op_algebraic_property),
    ('is_conversion', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
]

nir_op_info = struct_nir_op_info
try: nir_op_infos = (struct_nir_op_info * 489).in_dll(_libraries['libtinymesa_cpu.so'], 'nir_op_infos')
except (AttributeError, ValueError): pass
try:
    nir_op_is_selection = _libraries['FIXME_STUB'].nir_op_is_selection
    nir_op_is_selection.restype = ctypes.c_bool
    nir_op_is_selection.argtypes = [nir_op]
except AttributeError:
    pass
class struct_nir_alu_instr(Structure):
    pass

struct_nir_alu_instr._pack_ = 1 # source:False
struct_nir_alu_instr._fields_ = [
    ('instr', nir_instr),
    ('op', nir_op),
    ('exact', ctypes.c_bool, 1),
    ('no_signed_wrap', ctypes.c_bool, 1),
    ('no_unsigned_wrap', ctypes.c_bool, 1),
    ('fp_fast_math', ctypes.c_uint32, 9),
    ('PADDING_0', ctypes.c_uint32, 20),
    ('def', nir_def),
    ('src', struct_nir_alu_src * 0),
]

nir_alu_instr = struct_nir_alu_instr
try:
    nir_alu_instr_is_signed_zero_preserve = _libraries['FIXME_STUB'].nir_alu_instr_is_signed_zero_preserve
    nir_alu_instr_is_signed_zero_preserve.restype = ctypes.c_bool
    nir_alu_instr_is_signed_zero_preserve.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_alu_instr_is_inf_preserve = _libraries['FIXME_STUB'].nir_alu_instr_is_inf_preserve
    nir_alu_instr_is_inf_preserve.restype = ctypes.c_bool
    nir_alu_instr_is_inf_preserve.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_alu_instr_is_nan_preserve = _libraries['FIXME_STUB'].nir_alu_instr_is_nan_preserve
    nir_alu_instr_is_nan_preserve.restype = ctypes.c_bool
    nir_alu_instr_is_nan_preserve.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_alu_instr_is_signed_zero_inf_nan_preserve = _libraries['FIXME_STUB'].nir_alu_instr_is_signed_zero_inf_nan_preserve
    nir_alu_instr_is_signed_zero_inf_nan_preserve.restype = ctypes.c_bool
    nir_alu_instr_is_signed_zero_inf_nan_preserve.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_alu_src_copy = _libraries['libtinymesa_cpu.so'].nir_alu_src_copy
    nir_alu_src_copy.restype = None
    nir_alu_src_copy.argtypes = [ctypes.POINTER(struct_nir_alu_src), ctypes.POINTER(struct_nir_alu_src)]
except AttributeError:
    pass
try:
    nir_alu_instr_src_read_mask = _libraries['libtinymesa_cpu.so'].nir_alu_instr_src_read_mask
    nir_alu_instr_src_read_mask.restype = nir_component_mask_t
    nir_alu_instr_src_read_mask.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_ssa_alu_instr_src_components = _libraries['libtinymesa_cpu.so'].nir_ssa_alu_instr_src_components
    nir_ssa_alu_instr_src_components.restype = ctypes.c_uint32
    nir_ssa_alu_instr_src_components.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_instr_channel_used = _libraries['FIXME_STUB'].nir_alu_instr_channel_used
    nir_alu_instr_channel_used.restype = ctypes.c_bool
    nir_alu_instr_channel_used.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_instr_is_comparison = _libraries['libtinymesa_cpu.so'].nir_alu_instr_is_comparison
    nir_alu_instr_is_comparison.restype = ctypes.c_bool
    nir_alu_instr_is_comparison.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_const_value_negative_equal = _libraries['libtinymesa_cpu.so'].nir_const_value_negative_equal
    nir_const_value_negative_equal.restype = ctypes.c_bool
    nir_const_value_negative_equal.argtypes = [nir_const_value, nir_const_value, nir_alu_type]
except AttributeError:
    pass
try:
    nir_alu_srcs_equal = _libraries['libtinymesa_cpu.so'].nir_alu_srcs_equal
    nir_alu_srcs_equal.restype = ctypes.c_bool
    nir_alu_srcs_equal.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_srcs_negative_equal_typed = _libraries['libtinymesa_cpu.so'].nir_alu_srcs_negative_equal_typed
    nir_alu_srcs_negative_equal_typed.restype = ctypes.c_bool
    nir_alu_srcs_negative_equal_typed.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32, nir_alu_type]
except AttributeError:
    pass
try:
    nir_alu_srcs_negative_equal = _libraries['libtinymesa_cpu.so'].nir_alu_srcs_negative_equal
    nir_alu_srcs_negative_equal.restype = ctypes.c_bool
    nir_alu_srcs_negative_equal.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_src_is_trivial_ssa = _libraries['libtinymesa_cpu.so'].nir_alu_src_is_trivial_ssa
    nir_alu_src_is_trivial_ssa.restype = ctypes.c_bool
    nir_alu_src_is_trivial_ssa.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_deref_type'
c__EA_nir_deref_type__enumvalues = {
    0: 'nir_deref_type_var',
    1: 'nir_deref_type_array',
    2: 'nir_deref_type_array_wildcard',
    3: 'nir_deref_type_ptr_as_array',
    4: 'nir_deref_type_struct',
    5: 'nir_deref_type_cast',
}
nir_deref_type_var = 0
nir_deref_type_array = 1
nir_deref_type_array_wildcard = 2
nir_deref_type_ptr_as_array = 3
nir_deref_type_struct = 4
nir_deref_type_cast = 5
c__EA_nir_deref_type = ctypes.c_uint32 # enum
nir_deref_type = c__EA_nir_deref_type
nir_deref_type__enumvalues = c__EA_nir_deref_type__enumvalues
class struct_nir_deref_instr(Structure):
    pass

class union_nir_deref_instr_0(Union):
    pass

union_nir_deref_instr_0._pack_ = 1 # source:False
union_nir_deref_instr_0._fields_ = [
    ('var', ctypes.POINTER(struct_nir_variable)),
    ('parent', nir_src),
]

class union_nir_deref_instr_1(Union):
    pass

class struct_nir_deref_instr_1_arr(Structure):
    pass

struct_nir_deref_instr_1_arr._pack_ = 1 # source:False
struct_nir_deref_instr_1_arr._fields_ = [
    ('index', nir_src),
    ('in_bounds', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

class struct_nir_deref_instr_1_strct(Structure):
    pass

struct_nir_deref_instr_1_strct._pack_ = 1 # source:False
struct_nir_deref_instr_1_strct._fields_ = [
    ('index', ctypes.c_uint32),
]

class struct_nir_deref_instr_1_cast(Structure):
    pass

struct_nir_deref_instr_1_cast._pack_ = 1 # source:False
struct_nir_deref_instr_1_cast._fields_ = [
    ('ptr_stride', ctypes.c_uint32),
    ('align_mul', ctypes.c_uint32),
    ('align_offset', ctypes.c_uint32),
]

union_nir_deref_instr_1._pack_ = 1 # source:False
union_nir_deref_instr_1._fields_ = [
    ('arr', struct_nir_deref_instr_1_arr),
    ('strct', struct_nir_deref_instr_1_strct),
    ('cast', struct_nir_deref_instr_1_cast),
    ('PADDING_0', ctypes.c_ubyte * 28),
]

struct_nir_deref_instr._pack_ = 1 # source:False
struct_nir_deref_instr._anonymous_ = ('_0', '_1',)
struct_nir_deref_instr._fields_ = [
    ('instr', nir_instr),
    ('deref_type', nir_deref_type),
    ('modes', c__EA_nir_variable_mode),
    ('type', ctypes.POINTER(struct_glsl_type)),
    ('_0', union_nir_deref_instr_0),
    ('_1', union_nir_deref_instr_1),
    ('def', nir_def),
]

nir_deref_instr = struct_nir_deref_instr
try:
    nir_deref_cast_is_trivial = _libraries['libtinymesa_cpu.so'].nir_deref_cast_is_trivial
    nir_deref_cast_is_trivial.restype = ctypes.c_bool
    nir_deref_cast_is_trivial.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
nir_variable_mode = c__EA_nir_variable_mode
nir_variable_mode__enumvalues = c__EA_nir_variable_mode__enumvalues
try:
    nir_deref_mode_may_be = _libraries['FIXME_STUB'].nir_deref_mode_may_be
    nir_deref_mode_may_be.restype = ctypes.c_bool
    nir_deref_mode_may_be.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_deref_mode_must_be = _libraries['FIXME_STUB'].nir_deref_mode_must_be
    nir_deref_mode_must_be.restype = ctypes.c_bool
    nir_deref_mode_must_be.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_deref_mode_is = _libraries['FIXME_STUB'].nir_deref_mode_is
    nir_deref_mode_is.restype = ctypes.c_bool
    nir_deref_mode_is.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_deref_mode_is_one_of = _libraries['FIXME_STUB'].nir_deref_mode_is_one_of
    nir_deref_mode_is_one_of.restype = ctypes.c_bool
    nir_deref_mode_is_one_of.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_deref_mode_is_in_set = _libraries['FIXME_STUB'].nir_deref_mode_is_in_set
    nir_deref_mode_is_in_set.restype = ctypes.c_bool
    nir_deref_mode_is_in_set.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_src_as_deref = _libraries['FIXME_STUB'].nir_src_as_deref
    nir_src_as_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_src_as_deref.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_deref_instr_parent = _libraries['FIXME_STUB'].nir_deref_instr_parent
    nir_deref_instr_parent.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_deref_instr_parent.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_deref_instr_get_variable = _libraries['FIXME_STUB'].nir_deref_instr_get_variable
    nir_deref_instr_get_variable.restype = ctypes.POINTER(struct_nir_variable)
    nir_deref_instr_get_variable.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_deref_instr_has_indirect = _libraries['libtinymesa_cpu.so'].nir_deref_instr_has_indirect
    nir_deref_instr_has_indirect.restype = ctypes.c_bool
    nir_deref_instr_has_indirect.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_deref_instr_is_known_out_of_bounds = _libraries['libtinymesa_cpu.so'].nir_deref_instr_is_known_out_of_bounds
    nir_deref_instr_is_known_out_of_bounds.restype = ctypes.c_bool
    nir_deref_instr_is_known_out_of_bounds.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_deref_instr_has_complex_use_options'
c__EA_nir_deref_instr_has_complex_use_options__enumvalues = {
    1: 'nir_deref_instr_has_complex_use_allow_memcpy_src',
    2: 'nir_deref_instr_has_complex_use_allow_memcpy_dst',
    4: 'nir_deref_instr_has_complex_use_allow_atomics',
}
nir_deref_instr_has_complex_use_allow_memcpy_src = 1
nir_deref_instr_has_complex_use_allow_memcpy_dst = 2
nir_deref_instr_has_complex_use_allow_atomics = 4
c__EA_nir_deref_instr_has_complex_use_options = ctypes.c_uint32 # enum
nir_deref_instr_has_complex_use_options = c__EA_nir_deref_instr_has_complex_use_options
nir_deref_instr_has_complex_use_options__enumvalues = c__EA_nir_deref_instr_has_complex_use_options__enumvalues
try:
    nir_deref_instr_has_complex_use = _libraries['libtinymesa_cpu.so'].nir_deref_instr_has_complex_use
    nir_deref_instr_has_complex_use.restype = ctypes.c_bool
    nir_deref_instr_has_complex_use.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_deref_instr_has_complex_use_options]
except AttributeError:
    pass
try:
    nir_deref_instr_remove_if_unused = _libraries['libtinymesa_cpu.so'].nir_deref_instr_remove_if_unused
    nir_deref_instr_remove_if_unused.restype = ctypes.c_bool
    nir_deref_instr_remove_if_unused.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_deref_instr_array_stride = _libraries['libtinymesa_cpu.so'].nir_deref_instr_array_stride
    nir_deref_instr_array_stride.restype = ctypes.c_uint32
    nir_deref_instr_array_stride.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
class struct_nir_call_instr(Structure):
    pass

class struct_nir_function(Structure):
    pass

struct_nir_call_instr._pack_ = 1 # source:False
struct_nir_call_instr._fields_ = [
    ('instr', nir_instr),
    ('callee', ctypes.POINTER(struct_nir_function)),
    ('indirect_callee', nir_src),
    ('num_params', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', struct_nir_src * 0),
]

class struct_nir_parameter(Structure):
    pass

class struct_nir_function_impl(Structure):
    pass

struct_nir_function._pack_ = 1 # source:False
struct_nir_function._fields_ = [
    ('node', struct_exec_node),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('shader', ctypes.POINTER(struct_nir_shader)),
    ('num_params', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', ctypes.POINTER(struct_nir_parameter)),
    ('impl', ctypes.POINTER(struct_nir_function_impl)),
    ('driver_attributes', ctypes.c_uint32),
    ('is_entrypoint', ctypes.c_bool),
    ('is_exported', ctypes.c_bool),
    ('is_preamble', ctypes.c_bool),
    ('should_inline', ctypes.c_bool),
    ('dont_inline', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('workgroup_size', ctypes.c_uint32 * 3),
    ('is_subroutine', ctypes.c_bool),
    ('is_tmp_globals_wrapper', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte * 2),
    ('num_subroutine_types', ctypes.c_int32),
    ('subroutine_types', ctypes.POINTER(ctypes.POINTER(struct_glsl_type))),
    ('subroutine_index', ctypes.c_int32),
    ('pass_flags', ctypes.c_uint32),
]

struct_nir_parameter._pack_ = 1 # source:False
struct_nir_parameter._fields_ = [
    ('num_components', ctypes.c_ubyte),
    ('bit_size', ctypes.c_ubyte),
    ('is_return', ctypes.c_bool),
    ('implicit_conversion_prohibited', ctypes.c_bool),
    ('is_uniform', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('mode', nir_variable_mode),
    ('driver_attributes', ctypes.c_uint32),
    ('type', ctypes.POINTER(struct_glsl_type)),
    ('name', ctypes.POINTER(ctypes.c_char)),
]


# values for enumeration 'c__EA_nir_metadata'
c__EA_nir_metadata__enumvalues = {
    0: 'nir_metadata_none',
    1: 'nir_metadata_block_index',
    2: 'nir_metadata_dominance',
    4: 'nir_metadata_live_defs',
    8: 'nir_metadata_not_properly_reset',
    16: 'nir_metadata_loop_analysis',
    32: 'nir_metadata_instr_index',
    64: 'nir_metadata_divergence',
    3: 'nir_metadata_control_flow',
    -9: 'nir_metadata_all',
}
nir_metadata_none = 0
nir_metadata_block_index = 1
nir_metadata_dominance = 2
nir_metadata_live_defs = 4
nir_metadata_not_properly_reset = 8
nir_metadata_loop_analysis = 16
nir_metadata_instr_index = 32
nir_metadata_divergence = 64
nir_metadata_control_flow = 3
nir_metadata_all = -9
c__EA_nir_metadata = ctypes.c_int32 # enum
struct_nir_function_impl._pack_ = 1 # source:False
struct_nir_function_impl._fields_ = [
    ('cf_node', struct_nir_cf_node),
    ('function', ctypes.POINTER(struct_nir_function)),
    ('preamble', ctypes.POINTER(struct_nir_function)),
    ('body', struct_exec_list),
    ('end_block', ctypes.POINTER(struct_nir_block)),
    ('locals', struct_exec_list),
    ('ssa_alloc', ctypes.c_uint32),
    ('num_blocks', ctypes.c_uint32),
    ('structured', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('valid_metadata', c__EA_nir_metadata),
    ('loop_analysis_indirect_mask', nir_variable_mode),
    ('loop_analysis_force_unroll_sampler_indirect', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
]

nir_call_instr = struct_nir_call_instr

# values for enumeration 'c__EA_nir_intrinsic_op'
c__EA_nir_intrinsic_op__enumvalues = {
    0: 'nir_intrinsic_accept_ray_intersection',
    1: 'nir_intrinsic_addr_mode_is',
    2: 'nir_intrinsic_al2p_nv',
    3: 'nir_intrinsic_ald_nv',
    4: 'nir_intrinsic_alpha_to_coverage',
    5: 'nir_intrinsic_as_uniform',
    6: 'nir_intrinsic_ast_nv',
    7: 'nir_intrinsic_atomic_add_gen_prim_count_amd',
    8: 'nir_intrinsic_atomic_add_gs_emit_prim_count_amd',
    9: 'nir_intrinsic_atomic_add_shader_invocation_count_amd',
    10: 'nir_intrinsic_atomic_add_xfb_prim_count_amd',
    11: 'nir_intrinsic_atomic_counter_add',
    12: 'nir_intrinsic_atomic_counter_add_deref',
    13: 'nir_intrinsic_atomic_counter_and',
    14: 'nir_intrinsic_atomic_counter_and_deref',
    15: 'nir_intrinsic_atomic_counter_comp_swap',
    16: 'nir_intrinsic_atomic_counter_comp_swap_deref',
    17: 'nir_intrinsic_atomic_counter_exchange',
    18: 'nir_intrinsic_atomic_counter_exchange_deref',
    19: 'nir_intrinsic_atomic_counter_inc',
    20: 'nir_intrinsic_atomic_counter_inc_deref',
    21: 'nir_intrinsic_atomic_counter_max',
    22: 'nir_intrinsic_atomic_counter_max_deref',
    23: 'nir_intrinsic_atomic_counter_min',
    24: 'nir_intrinsic_atomic_counter_min_deref',
    25: 'nir_intrinsic_atomic_counter_or',
    26: 'nir_intrinsic_atomic_counter_or_deref',
    27: 'nir_intrinsic_atomic_counter_post_dec',
    28: 'nir_intrinsic_atomic_counter_post_dec_deref',
    29: 'nir_intrinsic_atomic_counter_pre_dec',
    30: 'nir_intrinsic_atomic_counter_pre_dec_deref',
    31: 'nir_intrinsic_atomic_counter_read',
    32: 'nir_intrinsic_atomic_counter_read_deref',
    33: 'nir_intrinsic_atomic_counter_xor',
    34: 'nir_intrinsic_atomic_counter_xor_deref',
    35: 'nir_intrinsic_ballot',
    36: 'nir_intrinsic_ballot_bit_count_exclusive',
    37: 'nir_intrinsic_ballot_bit_count_inclusive',
    38: 'nir_intrinsic_ballot_bit_count_reduce',
    39: 'nir_intrinsic_ballot_bitfield_extract',
    40: 'nir_intrinsic_ballot_find_lsb',
    41: 'nir_intrinsic_ballot_find_msb',
    42: 'nir_intrinsic_ballot_relaxed',
    43: 'nir_intrinsic_bar_break_nv',
    44: 'nir_intrinsic_bar_set_nv',
    45: 'nir_intrinsic_bar_sync_nv',
    46: 'nir_intrinsic_barrier',
    47: 'nir_intrinsic_begin_invocation_interlock',
    48: 'nir_intrinsic_bindgen_return',
    49: 'nir_intrinsic_bindless_image_agx',
    50: 'nir_intrinsic_bindless_image_atomic',
    51: 'nir_intrinsic_bindless_image_atomic_swap',
    52: 'nir_intrinsic_bindless_image_descriptor_amd',
    53: 'nir_intrinsic_bindless_image_format',
    54: 'nir_intrinsic_bindless_image_fragment_mask_load_amd',
    55: 'nir_intrinsic_bindless_image_levels',
    56: 'nir_intrinsic_bindless_image_load',
    57: 'nir_intrinsic_bindless_image_load_raw_intel',
    58: 'nir_intrinsic_bindless_image_order',
    59: 'nir_intrinsic_bindless_image_samples',
    60: 'nir_intrinsic_bindless_image_samples_identical',
    61: 'nir_intrinsic_bindless_image_size',
    62: 'nir_intrinsic_bindless_image_sparse_load',
    63: 'nir_intrinsic_bindless_image_store',
    64: 'nir_intrinsic_bindless_image_store_block_agx',
    65: 'nir_intrinsic_bindless_image_store_raw_intel',
    66: 'nir_intrinsic_bindless_image_texel_address',
    67: 'nir_intrinsic_bindless_resource_ir3',
    68: 'nir_intrinsic_brcst_active_ir3',
    69: 'nir_intrinsic_btd_retire_intel',
    70: 'nir_intrinsic_btd_spawn_intel',
    71: 'nir_intrinsic_btd_stack_push_intel',
    72: 'nir_intrinsic_bvh64_intersect_ray_amd',
    73: 'nir_intrinsic_bvh8_intersect_ray_amd',
    74: 'nir_intrinsic_bvh_stack_rtn_amd',
    75: 'nir_intrinsic_cmat_binary_op',
    76: 'nir_intrinsic_cmat_bitcast',
    77: 'nir_intrinsic_cmat_construct',
    78: 'nir_intrinsic_cmat_convert',
    79: 'nir_intrinsic_cmat_copy',
    80: 'nir_intrinsic_cmat_extract',
    81: 'nir_intrinsic_cmat_insert',
    82: 'nir_intrinsic_cmat_length',
    83: 'nir_intrinsic_cmat_load',
    84: 'nir_intrinsic_cmat_muladd',
    85: 'nir_intrinsic_cmat_muladd_amd',
    86: 'nir_intrinsic_cmat_muladd_nv',
    87: 'nir_intrinsic_cmat_scalar_op',
    88: 'nir_intrinsic_cmat_store',
    89: 'nir_intrinsic_cmat_transpose',
    90: 'nir_intrinsic_cmat_unary_op',
    91: 'nir_intrinsic_convert_alu_types',
    92: 'nir_intrinsic_convert_cmat_intel',
    93: 'nir_intrinsic_copy_deref',
    94: 'nir_intrinsic_copy_fs_outputs_nv',
    95: 'nir_intrinsic_copy_global_to_uniform_ir3',
    96: 'nir_intrinsic_copy_push_const_to_uniform_ir3',
    97: 'nir_intrinsic_copy_ubo_to_uniform_ir3',
    98: 'nir_intrinsic_ddx',
    99: 'nir_intrinsic_ddx_coarse',
    100: 'nir_intrinsic_ddx_fine',
    101: 'nir_intrinsic_ddy',
    102: 'nir_intrinsic_ddy_coarse',
    103: 'nir_intrinsic_ddy_fine',
    104: 'nir_intrinsic_debug_break',
    105: 'nir_intrinsic_decl_reg',
    106: 'nir_intrinsic_demote',
    107: 'nir_intrinsic_demote_if',
    108: 'nir_intrinsic_demote_samples',
    109: 'nir_intrinsic_deref_atomic',
    110: 'nir_intrinsic_deref_atomic_swap',
    111: 'nir_intrinsic_deref_buffer_array_length',
    112: 'nir_intrinsic_deref_implicit_array_length',
    113: 'nir_intrinsic_deref_mode_is',
    114: 'nir_intrinsic_deref_texture_src',
    115: 'nir_intrinsic_doorbell_agx',
    116: 'nir_intrinsic_dpas_intel',
    117: 'nir_intrinsic_dpp16_shift_amd',
    118: 'nir_intrinsic_elect',
    119: 'nir_intrinsic_elect_any_ir3',
    120: 'nir_intrinsic_emit_primitive_poly',
    121: 'nir_intrinsic_emit_vertex',
    122: 'nir_intrinsic_emit_vertex_nv',
    123: 'nir_intrinsic_emit_vertex_with_counter',
    124: 'nir_intrinsic_end_invocation_interlock',
    125: 'nir_intrinsic_end_primitive',
    126: 'nir_intrinsic_end_primitive_nv',
    127: 'nir_intrinsic_end_primitive_with_counter',
    128: 'nir_intrinsic_enqueue_node_payloads',
    129: 'nir_intrinsic_exclusive_scan',
    130: 'nir_intrinsic_exclusive_scan_clusters_ir3',
    131: 'nir_intrinsic_execute_callable',
    132: 'nir_intrinsic_execute_closest_hit_amd',
    133: 'nir_intrinsic_execute_miss_amd',
    134: 'nir_intrinsic_export_agx',
    135: 'nir_intrinsic_export_amd',
    136: 'nir_intrinsic_export_dual_src_blend_amd',
    137: 'nir_intrinsic_export_row_amd',
    138: 'nir_intrinsic_fence_helper_exit_agx',
    139: 'nir_intrinsic_fence_mem_to_tex_agx',
    140: 'nir_intrinsic_fence_pbe_to_tex_agx',
    141: 'nir_intrinsic_fence_pbe_to_tex_pixel_agx',
    142: 'nir_intrinsic_final_primitive_nv',
    143: 'nir_intrinsic_finalize_incoming_node_payload',
    144: 'nir_intrinsic_first_invocation',
    145: 'nir_intrinsic_fs_out_nv',
    146: 'nir_intrinsic_gds_atomic_add_amd',
    147: 'nir_intrinsic_get_ssbo_size',
    148: 'nir_intrinsic_get_ubo_size',
    149: 'nir_intrinsic_global_atomic',
    150: 'nir_intrinsic_global_atomic_2x32',
    151: 'nir_intrinsic_global_atomic_agx',
    152: 'nir_intrinsic_global_atomic_amd',
    153: 'nir_intrinsic_global_atomic_swap',
    154: 'nir_intrinsic_global_atomic_swap_2x32',
    155: 'nir_intrinsic_global_atomic_swap_agx',
    156: 'nir_intrinsic_global_atomic_swap_amd',
    157: 'nir_intrinsic_ignore_ray_intersection',
    158: 'nir_intrinsic_imadsp_nv',
    159: 'nir_intrinsic_image_atomic',
    160: 'nir_intrinsic_image_atomic_swap',
    161: 'nir_intrinsic_image_deref_atomic',
    162: 'nir_intrinsic_image_deref_atomic_swap',
    163: 'nir_intrinsic_image_deref_descriptor_amd',
    164: 'nir_intrinsic_image_deref_format',
    165: 'nir_intrinsic_image_deref_fragment_mask_load_amd',
    166: 'nir_intrinsic_image_deref_levels',
    167: 'nir_intrinsic_image_deref_load',
    168: 'nir_intrinsic_image_deref_load_info_nv',
    169: 'nir_intrinsic_image_deref_load_param_intel',
    170: 'nir_intrinsic_image_deref_load_raw_intel',
    171: 'nir_intrinsic_image_deref_order',
    172: 'nir_intrinsic_image_deref_samples',
    173: 'nir_intrinsic_image_deref_samples_identical',
    174: 'nir_intrinsic_image_deref_size',
    175: 'nir_intrinsic_image_deref_sparse_load',
    176: 'nir_intrinsic_image_deref_store',
    177: 'nir_intrinsic_image_deref_store_block_agx',
    178: 'nir_intrinsic_image_deref_store_raw_intel',
    179: 'nir_intrinsic_image_deref_texel_address',
    180: 'nir_intrinsic_image_descriptor_amd',
    181: 'nir_intrinsic_image_format',
    182: 'nir_intrinsic_image_fragment_mask_load_amd',
    183: 'nir_intrinsic_image_levels',
    184: 'nir_intrinsic_image_load',
    185: 'nir_intrinsic_image_load_raw_intel',
    186: 'nir_intrinsic_image_order',
    187: 'nir_intrinsic_image_samples',
    188: 'nir_intrinsic_image_samples_identical',
    189: 'nir_intrinsic_image_size',
    190: 'nir_intrinsic_image_sparse_load',
    191: 'nir_intrinsic_image_store',
    192: 'nir_intrinsic_image_store_block_agx',
    193: 'nir_intrinsic_image_store_raw_intel',
    194: 'nir_intrinsic_image_texel_address',
    195: 'nir_intrinsic_inclusive_scan',
    196: 'nir_intrinsic_inclusive_scan_clusters_ir3',
    197: 'nir_intrinsic_initialize_node_payloads',
    198: 'nir_intrinsic_interp_deref_at_centroid',
    199: 'nir_intrinsic_interp_deref_at_offset',
    200: 'nir_intrinsic_interp_deref_at_sample',
    201: 'nir_intrinsic_interp_deref_at_vertex',
    202: 'nir_intrinsic_inverse_ballot',
    203: 'nir_intrinsic_ipa_nv',
    204: 'nir_intrinsic_is_helper_invocation',
    205: 'nir_intrinsic_is_sparse_resident_zink',
    206: 'nir_intrinsic_is_sparse_texels_resident',
    207: 'nir_intrinsic_is_subgroup_invocation_lt_amd',
    208: 'nir_intrinsic_isberd_nv',
    209: 'nir_intrinsic_lane_permute_16_amd',
    210: 'nir_intrinsic_last_invocation',
    211: 'nir_intrinsic_launch_mesh_workgroups',
    212: 'nir_intrinsic_launch_mesh_workgroups_with_payload_deref',
    213: 'nir_intrinsic_ldc_nv',
    214: 'nir_intrinsic_ldcx_nv',
    215: 'nir_intrinsic_ldtram_nv',
    216: 'nir_intrinsic_load_aa_line_width',
    217: 'nir_intrinsic_load_accel_struct_amd',
    218: 'nir_intrinsic_load_active_samples_agx',
    219: 'nir_intrinsic_load_active_subgroup_count_agx',
    220: 'nir_intrinsic_load_active_subgroup_invocation_agx',
    221: 'nir_intrinsic_load_agx',
    222: 'nir_intrinsic_load_alpha_reference_amd',
    223: 'nir_intrinsic_load_api_sample_mask_agx',
    224: 'nir_intrinsic_load_attrib_clamp_agx',
    225: 'nir_intrinsic_load_attribute_pan',
    226: 'nir_intrinsic_load_back_face_agx',
    227: 'nir_intrinsic_load_barycentric_at_offset',
    228: 'nir_intrinsic_load_barycentric_at_offset_nv',
    229: 'nir_intrinsic_load_barycentric_at_sample',
    230: 'nir_intrinsic_load_barycentric_centroid',
    231: 'nir_intrinsic_load_barycentric_coord_at_offset',
    232: 'nir_intrinsic_load_barycentric_coord_at_sample',
    233: 'nir_intrinsic_load_barycentric_coord_centroid',
    234: 'nir_intrinsic_load_barycentric_coord_pixel',
    235: 'nir_intrinsic_load_barycentric_coord_sample',
    236: 'nir_intrinsic_load_barycentric_model',
    237: 'nir_intrinsic_load_barycentric_optimize_amd',
    238: 'nir_intrinsic_load_barycentric_pixel',
    239: 'nir_intrinsic_load_barycentric_sample',
    240: 'nir_intrinsic_load_base_global_invocation_id',
    241: 'nir_intrinsic_load_base_instance',
    242: 'nir_intrinsic_load_base_vertex',
    243: 'nir_intrinsic_load_base_workgroup_id',
    244: 'nir_intrinsic_load_blend_const_color_a_float',
    245: 'nir_intrinsic_load_blend_const_color_aaaa8888_unorm',
    246: 'nir_intrinsic_load_blend_const_color_b_float',
    247: 'nir_intrinsic_load_blend_const_color_g_float',
    248: 'nir_intrinsic_load_blend_const_color_r_float',
    249: 'nir_intrinsic_load_blend_const_color_rgba',
    250: 'nir_intrinsic_load_blend_const_color_rgba8888_unorm',
    251: 'nir_intrinsic_load_btd_global_arg_addr_intel',
    252: 'nir_intrinsic_load_btd_local_arg_addr_intel',
    253: 'nir_intrinsic_load_btd_resume_sbt_addr_intel',
    254: 'nir_intrinsic_load_btd_shader_type_intel',
    255: 'nir_intrinsic_load_btd_stack_id_intel',
    256: 'nir_intrinsic_load_buffer_amd',
    257: 'nir_intrinsic_load_callable_sbt_addr_intel',
    258: 'nir_intrinsic_load_callable_sbt_stride_intel',
    259: 'nir_intrinsic_load_clamp_vertex_color_amd',
    260: 'nir_intrinsic_load_clip_half_line_width_amd',
    261: 'nir_intrinsic_load_clip_z_coeff_agx',
    262: 'nir_intrinsic_load_coalesced_input_count',
    263: 'nir_intrinsic_load_coefficients_agx',
    264: 'nir_intrinsic_load_color0',
    265: 'nir_intrinsic_load_color1',
    266: 'nir_intrinsic_load_const_buf_base_addr_lvp',
    267: 'nir_intrinsic_load_const_ir3',
    268: 'nir_intrinsic_load_constant',
    269: 'nir_intrinsic_load_constant_agx',
    270: 'nir_intrinsic_load_constant_base_ptr',
    271: 'nir_intrinsic_load_converted_output_pan',
    272: 'nir_intrinsic_load_core_id_agx',
    273: 'nir_intrinsic_load_cull_any_enabled_amd',
    274: 'nir_intrinsic_load_cull_back_face_enabled_amd',
    275: 'nir_intrinsic_load_cull_ccw_amd',
    276: 'nir_intrinsic_load_cull_front_face_enabled_amd',
    277: 'nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd',
    278: 'nir_intrinsic_load_cull_mask',
    279: 'nir_intrinsic_load_cull_mask_and_flags_amd',
    280: 'nir_intrinsic_load_cull_small_line_precision_amd',
    281: 'nir_intrinsic_load_cull_small_lines_enabled_amd',
    282: 'nir_intrinsic_load_cull_small_triangle_precision_amd',
    283: 'nir_intrinsic_load_cull_small_triangles_enabled_amd',
    284: 'nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd',
    285: 'nir_intrinsic_load_debug_log_desc_amd',
    286: 'nir_intrinsic_load_depth_never_agx',
    287: 'nir_intrinsic_load_deref',
    288: 'nir_intrinsic_load_deref_block_intel',
    289: 'nir_intrinsic_load_draw_id',
    290: 'nir_intrinsic_load_esgs_vertex_stride_amd',
    291: 'nir_intrinsic_load_exported_agx',
    292: 'nir_intrinsic_load_fb_layers_v3d',
    293: 'nir_intrinsic_load_fbfetch_image_desc_amd',
    294: 'nir_intrinsic_load_fbfetch_image_fmask_desc_amd',
    295: 'nir_intrinsic_load_fep_w_v3d',
    296: 'nir_intrinsic_load_first_vertex',
    297: 'nir_intrinsic_load_fixed_point_size_agx',
    298: 'nir_intrinsic_load_flat_mask',
    299: 'nir_intrinsic_load_force_vrs_rates_amd',
    300: 'nir_intrinsic_load_frag_coord',
    301: 'nir_intrinsic_load_frag_coord_unscaled_ir3',
    302: 'nir_intrinsic_load_frag_coord_w',
    303: 'nir_intrinsic_load_frag_coord_z',
    304: 'nir_intrinsic_load_frag_coord_zw_pan',
    305: 'nir_intrinsic_load_frag_invocation_count',
    306: 'nir_intrinsic_load_frag_offset_ir3',
    307: 'nir_intrinsic_load_frag_shading_rate',
    308: 'nir_intrinsic_load_frag_size',
    309: 'nir_intrinsic_load_frag_size_ir3',
    310: 'nir_intrinsic_load_from_texture_handle_agx',
    311: 'nir_intrinsic_load_front_face',
    312: 'nir_intrinsic_load_front_face_fsign',
    313: 'nir_intrinsic_load_fs_input_interp_deltas',
    314: 'nir_intrinsic_load_fs_msaa_intel',
    315: 'nir_intrinsic_load_fully_covered',
    316: 'nir_intrinsic_load_geometry_param_buffer_poly',
    317: 'nir_intrinsic_load_global',
    318: 'nir_intrinsic_load_global_2x32',
    319: 'nir_intrinsic_load_global_amd',
    320: 'nir_intrinsic_load_global_base_ptr',
    321: 'nir_intrinsic_load_global_block_intel',
    322: 'nir_intrinsic_load_global_bounded',
    323: 'nir_intrinsic_load_global_constant',
    324: 'nir_intrinsic_load_global_constant_bounded',
    325: 'nir_intrinsic_load_global_constant_offset',
    326: 'nir_intrinsic_load_global_constant_uniform_block_intel',
    327: 'nir_intrinsic_load_global_etna',
    328: 'nir_intrinsic_load_global_invocation_id',
    329: 'nir_intrinsic_load_global_invocation_index',
    330: 'nir_intrinsic_load_global_ir3',
    331: 'nir_intrinsic_load_global_size',
    332: 'nir_intrinsic_load_gs_header_ir3',
    333: 'nir_intrinsic_load_gs_vertex_offset_amd',
    334: 'nir_intrinsic_load_gs_wave_id_amd',
    335: 'nir_intrinsic_load_helper_arg_hi_agx',
    336: 'nir_intrinsic_load_helper_arg_lo_agx',
    337: 'nir_intrinsic_load_helper_invocation',
    338: 'nir_intrinsic_load_helper_op_id_agx',
    339: 'nir_intrinsic_load_hit_attrib_amd',
    340: 'nir_intrinsic_load_hs_out_patch_data_offset_amd',
    341: 'nir_intrinsic_load_hs_patch_stride_ir3',
    342: 'nir_intrinsic_load_initial_edgeflags_amd',
    343: 'nir_intrinsic_load_inline_data_intel',
    344: 'nir_intrinsic_load_input',
    345: 'nir_intrinsic_load_input_assembly_buffer_poly',
    346: 'nir_intrinsic_load_input_attachment_conv_pan',
    347: 'nir_intrinsic_load_input_attachment_coord',
    348: 'nir_intrinsic_load_input_attachment_target_pan',
    349: 'nir_intrinsic_load_input_topology_poly',
    350: 'nir_intrinsic_load_input_vertex',
    351: 'nir_intrinsic_load_instance_id',
    352: 'nir_intrinsic_load_interpolated_input',
    353: 'nir_intrinsic_load_intersection_opaque_amd',
    354: 'nir_intrinsic_load_invocation_id',
    355: 'nir_intrinsic_load_is_first_fan_agx',
    356: 'nir_intrinsic_load_is_indexed_draw',
    357: 'nir_intrinsic_load_kernel_input',
    358: 'nir_intrinsic_load_layer_id',
    359: 'nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd',
    360: 'nir_intrinsic_load_leaf_opaque_intel',
    361: 'nir_intrinsic_load_leaf_procedural_intel',
    362: 'nir_intrinsic_load_line_coord',
    363: 'nir_intrinsic_load_line_width',
    364: 'nir_intrinsic_load_local_invocation_id',
    365: 'nir_intrinsic_load_local_invocation_index',
    366: 'nir_intrinsic_load_local_pixel_agx',
    367: 'nir_intrinsic_load_local_shared_r600',
    368: 'nir_intrinsic_load_lshs_vertex_stride_amd',
    369: 'nir_intrinsic_load_max_polygon_intel',
    370: 'nir_intrinsic_load_merged_wave_info_amd',
    371: 'nir_intrinsic_load_mesh_view_count',
    372: 'nir_intrinsic_load_mesh_view_indices',
    373: 'nir_intrinsic_load_multisampled_pan',
    374: 'nir_intrinsic_load_noperspective_varyings_pan',
    375: 'nir_intrinsic_load_num_subgroups',
    376: 'nir_intrinsic_load_num_vertices',
    377: 'nir_intrinsic_load_num_vertices_per_primitive_amd',
    378: 'nir_intrinsic_load_num_workgroups',
    379: 'nir_intrinsic_load_ordered_id_amd',
    380: 'nir_intrinsic_load_output',
    381: 'nir_intrinsic_load_packed_passthrough_primitive_amd',
    382: 'nir_intrinsic_load_param',
    383: 'nir_intrinsic_load_patch_vertices_in',
    384: 'nir_intrinsic_load_per_primitive_input',
    385: 'nir_intrinsic_load_per_primitive_output',
    386: 'nir_intrinsic_load_per_primitive_remap_intel',
    387: 'nir_intrinsic_load_per_vertex_input',
    388: 'nir_intrinsic_load_per_vertex_output',
    389: 'nir_intrinsic_load_per_view_output',
    390: 'nir_intrinsic_load_persp_center_rhw_ir3',
    391: 'nir_intrinsic_load_pipeline_stat_query_enabled_amd',
    392: 'nir_intrinsic_load_pixel_coord',
    393: 'nir_intrinsic_load_point_coord',
    394: 'nir_intrinsic_load_point_coord_maybe_flipped',
    395: 'nir_intrinsic_load_poly_line_smooth_enabled',
    396: 'nir_intrinsic_load_polygon_stipple_agx',
    397: 'nir_intrinsic_load_polygon_stipple_buffer_amd',
    398: 'nir_intrinsic_load_preamble',
    399: 'nir_intrinsic_load_prim_gen_query_enabled_amd',
    400: 'nir_intrinsic_load_prim_xfb_query_enabled_amd',
    401: 'nir_intrinsic_load_primitive_id',
    402: 'nir_intrinsic_load_primitive_location_ir3',
    403: 'nir_intrinsic_load_printf_buffer_address',
    404: 'nir_intrinsic_load_printf_buffer_size',
    405: 'nir_intrinsic_load_provoking_last',
    406: 'nir_intrinsic_load_provoking_vtx_amd',
    407: 'nir_intrinsic_load_provoking_vtx_in_prim_amd',
    408: 'nir_intrinsic_load_push_constant',
    409: 'nir_intrinsic_load_push_constant_zink',
    410: 'nir_intrinsic_load_r600_indirect_per_vertex_input',
    411: 'nir_intrinsic_load_rasterization_primitive_amd',
    412: 'nir_intrinsic_load_rasterization_samples_amd',
    413: 'nir_intrinsic_load_rasterization_stream',
    414: 'nir_intrinsic_load_raw_output_pan',
    415: 'nir_intrinsic_load_raw_vertex_id_pan',
    416: 'nir_intrinsic_load_raw_vertex_offset_pan',
    417: 'nir_intrinsic_load_ray_base_mem_addr_intel',
    418: 'nir_intrinsic_load_ray_flags',
    419: 'nir_intrinsic_load_ray_geometry_index',
    420: 'nir_intrinsic_load_ray_hit_kind',
    421: 'nir_intrinsic_load_ray_hit_sbt_addr_intel',
    422: 'nir_intrinsic_load_ray_hit_sbt_stride_intel',
    423: 'nir_intrinsic_load_ray_hw_stack_size_intel',
    424: 'nir_intrinsic_load_ray_instance_custom_index',
    425: 'nir_intrinsic_load_ray_launch_id',
    426: 'nir_intrinsic_load_ray_launch_size',
    427: 'nir_intrinsic_load_ray_miss_sbt_addr_intel',
    428: 'nir_intrinsic_load_ray_miss_sbt_stride_intel',
    429: 'nir_intrinsic_load_ray_num_dss_rt_stacks_intel',
    430: 'nir_intrinsic_load_ray_object_direction',
    431: 'nir_intrinsic_load_ray_object_origin',
    432: 'nir_intrinsic_load_ray_object_to_world',
    433: 'nir_intrinsic_load_ray_query_global_intel',
    434: 'nir_intrinsic_load_ray_sw_stack_size_intel',
    435: 'nir_intrinsic_load_ray_t_max',
    436: 'nir_intrinsic_load_ray_t_min',
    437: 'nir_intrinsic_load_ray_tracing_stack_base_lvp',
    438: 'nir_intrinsic_load_ray_triangle_vertex_positions',
    439: 'nir_intrinsic_load_ray_world_direction',
    440: 'nir_intrinsic_load_ray_world_origin',
    441: 'nir_intrinsic_load_ray_world_to_object',
    442: 'nir_intrinsic_load_readonly_output_pan',
    443: 'nir_intrinsic_load_reg',
    444: 'nir_intrinsic_load_reg_indirect',
    445: 'nir_intrinsic_load_rel_patch_id_ir3',
    446: 'nir_intrinsic_load_reloc_const_intel',
    447: 'nir_intrinsic_load_resume_shader_address_amd',
    448: 'nir_intrinsic_load_ring_attr_amd',
    449: 'nir_intrinsic_load_ring_attr_offset_amd',
    450: 'nir_intrinsic_load_ring_es2gs_offset_amd',
    451: 'nir_intrinsic_load_ring_esgs_amd',
    452: 'nir_intrinsic_load_ring_gs2vs_offset_amd',
    453: 'nir_intrinsic_load_ring_gsvs_amd',
    454: 'nir_intrinsic_load_ring_mesh_scratch_amd',
    455: 'nir_intrinsic_load_ring_mesh_scratch_offset_amd',
    456: 'nir_intrinsic_load_ring_task_draw_amd',
    457: 'nir_intrinsic_load_ring_task_payload_amd',
    458: 'nir_intrinsic_load_ring_tess_factors_amd',
    459: 'nir_intrinsic_load_ring_tess_factors_offset_amd',
    460: 'nir_intrinsic_load_ring_tess_offchip_amd',
    461: 'nir_intrinsic_load_ring_tess_offchip_offset_amd',
    462: 'nir_intrinsic_load_root_agx',
    463: 'nir_intrinsic_load_rt_arg_scratch_offset_amd',
    464: 'nir_intrinsic_load_rt_conversion_pan',
    465: 'nir_intrinsic_load_sample_id',
    466: 'nir_intrinsic_load_sample_id_no_per_sample',
    467: 'nir_intrinsic_load_sample_mask',
    468: 'nir_intrinsic_load_sample_mask_in',
    469: 'nir_intrinsic_load_sample_pos',
    470: 'nir_intrinsic_load_sample_pos_from_id',
    471: 'nir_intrinsic_load_sample_pos_or_center',
    472: 'nir_intrinsic_load_sample_positions_agx',
    473: 'nir_intrinsic_load_sample_positions_amd',
    474: 'nir_intrinsic_load_sample_positions_pan',
    475: 'nir_intrinsic_load_sampler_handle_agx',
    476: 'nir_intrinsic_load_sampler_lod_parameters',
    477: 'nir_intrinsic_load_samples_log2_agx',
    478: 'nir_intrinsic_load_sbt_base_amd',
    479: 'nir_intrinsic_load_sbt_offset_amd',
    480: 'nir_intrinsic_load_sbt_stride_amd',
    481: 'nir_intrinsic_load_scalar_arg_amd',
    482: 'nir_intrinsic_load_scratch',
    483: 'nir_intrinsic_load_scratch_base_ptr',
    484: 'nir_intrinsic_load_shader_call_data_offset_lvp',
    485: 'nir_intrinsic_load_shader_index',
    486: 'nir_intrinsic_load_shader_output_pan',
    487: 'nir_intrinsic_load_shader_part_tests_zs_agx',
    488: 'nir_intrinsic_load_shader_record_ptr',
    489: 'nir_intrinsic_load_shared',
    490: 'nir_intrinsic_load_shared2_amd',
    491: 'nir_intrinsic_load_shared_base_ptr',
    492: 'nir_intrinsic_load_shared_block_intel',
    493: 'nir_intrinsic_load_shared_ir3',
    494: 'nir_intrinsic_load_shared_lock_nv',
    495: 'nir_intrinsic_load_shared_uniform_block_intel',
    496: 'nir_intrinsic_load_simd_width_intel',
    497: 'nir_intrinsic_load_sm_count_nv',
    498: 'nir_intrinsic_load_sm_id_nv',
    499: 'nir_intrinsic_load_smem_amd',
    500: 'nir_intrinsic_load_ssbo',
    501: 'nir_intrinsic_load_ssbo_address',
    502: 'nir_intrinsic_load_ssbo_block_intel',
    503: 'nir_intrinsic_load_ssbo_intel',
    504: 'nir_intrinsic_load_ssbo_ir3',
    505: 'nir_intrinsic_load_ssbo_uniform_block_intel',
    506: 'nir_intrinsic_load_stack',
    507: 'nir_intrinsic_load_stat_query_address_agx',
    508: 'nir_intrinsic_load_streamout_buffer_amd',
    509: 'nir_intrinsic_load_streamout_config_amd',
    510: 'nir_intrinsic_load_streamout_offset_amd',
    511: 'nir_intrinsic_load_streamout_write_index_amd',
    512: 'nir_intrinsic_load_subgroup_eq_mask',
    513: 'nir_intrinsic_load_subgroup_ge_mask',
    514: 'nir_intrinsic_load_subgroup_gt_mask',
    515: 'nir_intrinsic_load_subgroup_id',
    516: 'nir_intrinsic_load_subgroup_id_shift_ir3',
    517: 'nir_intrinsic_load_subgroup_invocation',
    518: 'nir_intrinsic_load_subgroup_le_mask',
    519: 'nir_intrinsic_load_subgroup_lt_mask',
    520: 'nir_intrinsic_load_subgroup_size',
    521: 'nir_intrinsic_load_sysval_agx',
    522: 'nir_intrinsic_load_sysval_nv',
    523: 'nir_intrinsic_load_task_payload',
    524: 'nir_intrinsic_load_task_ring_entry_amd',
    525: 'nir_intrinsic_load_tcs_header_ir3',
    526: 'nir_intrinsic_load_tcs_in_param_base_r600',
    527: 'nir_intrinsic_load_tcs_mem_attrib_stride',
    528: 'nir_intrinsic_load_tcs_num_patches_amd',
    529: 'nir_intrinsic_load_tcs_out_param_base_r600',
    530: 'nir_intrinsic_load_tcs_primitive_mode_amd',
    531: 'nir_intrinsic_load_tcs_rel_patch_id_r600',
    532: 'nir_intrinsic_load_tcs_tess_factor_base_r600',
    533: 'nir_intrinsic_load_tcs_tess_levels_to_tes_amd',
    534: 'nir_intrinsic_load_tess_coord',
    535: 'nir_intrinsic_load_tess_coord_xy',
    536: 'nir_intrinsic_load_tess_factor_base_ir3',
    537: 'nir_intrinsic_load_tess_level_inner',
    538: 'nir_intrinsic_load_tess_level_inner_default',
    539: 'nir_intrinsic_load_tess_level_outer',
    540: 'nir_intrinsic_load_tess_level_outer_default',
    541: 'nir_intrinsic_load_tess_param_base_ir3',
    542: 'nir_intrinsic_load_tess_param_buffer_poly',
    543: 'nir_intrinsic_load_tess_rel_patch_id_amd',
    544: 'nir_intrinsic_load_tex_sprite_mask_agx',
    545: 'nir_intrinsic_load_texture_handle_agx',
    546: 'nir_intrinsic_load_texture_scale',
    547: 'nir_intrinsic_load_texture_size_etna',
    548: 'nir_intrinsic_load_tlb_color_brcm',
    549: 'nir_intrinsic_load_topology_id_intel',
    550: 'nir_intrinsic_load_typed_buffer_amd',
    551: 'nir_intrinsic_load_uav_ir3',
    552: 'nir_intrinsic_load_ubo',
    553: 'nir_intrinsic_load_ubo_uniform_block_intel',
    554: 'nir_intrinsic_load_ubo_vec4',
    555: 'nir_intrinsic_load_uniform',
    556: 'nir_intrinsic_load_user_clip_plane',
    557: 'nir_intrinsic_load_user_data_amd',
    558: 'nir_intrinsic_load_uvs_index_agx',
    559: 'nir_intrinsic_load_vbo_base_agx',
    560: 'nir_intrinsic_load_vector_arg_amd',
    561: 'nir_intrinsic_load_vertex_id',
    562: 'nir_intrinsic_load_vertex_id_zero_base',
    563: 'nir_intrinsic_load_view_index',
    564: 'nir_intrinsic_load_viewport_offset',
    565: 'nir_intrinsic_load_viewport_scale',
    566: 'nir_intrinsic_load_viewport_x_offset',
    567: 'nir_intrinsic_load_viewport_x_scale',
    568: 'nir_intrinsic_load_viewport_y_offset',
    569: 'nir_intrinsic_load_viewport_y_scale',
    570: 'nir_intrinsic_load_viewport_z_offset',
    571: 'nir_intrinsic_load_viewport_z_scale',
    572: 'nir_intrinsic_load_vs_output_buffer_poly',
    573: 'nir_intrinsic_load_vs_outputs_poly',
    574: 'nir_intrinsic_load_vs_primitive_stride_ir3',
    575: 'nir_intrinsic_load_vs_vertex_stride_ir3',
    576: 'nir_intrinsic_load_vulkan_descriptor',
    577: 'nir_intrinsic_load_warp_id_nv',
    578: 'nir_intrinsic_load_warps_per_sm_nv',
    579: 'nir_intrinsic_load_work_dim',
    580: 'nir_intrinsic_load_workgroup_id',
    581: 'nir_intrinsic_load_workgroup_index',
    582: 'nir_intrinsic_load_workgroup_num_input_primitives_amd',
    583: 'nir_intrinsic_load_workgroup_num_input_vertices_amd',
    584: 'nir_intrinsic_load_workgroup_size',
    585: 'nir_intrinsic_load_xfb_address',
    586: 'nir_intrinsic_load_xfb_index_buffer',
    587: 'nir_intrinsic_load_xfb_size',
    588: 'nir_intrinsic_load_xfb_state_address_gfx12_amd',
    589: 'nir_intrinsic_masked_swizzle_amd',
    590: 'nir_intrinsic_mbcnt_amd',
    591: 'nir_intrinsic_memcpy_deref',
    592: 'nir_intrinsic_nop',
    593: 'nir_intrinsic_nop_amd',
    594: 'nir_intrinsic_optimization_barrier_sgpr_amd',
    595: 'nir_intrinsic_optimization_barrier_vgpr_amd',
    596: 'nir_intrinsic_ordered_add_loop_gfx12_amd',
    597: 'nir_intrinsic_ordered_xfb_counter_add_gfx11_amd',
    598: 'nir_intrinsic_overwrite_tes_arguments_amd',
    599: 'nir_intrinsic_overwrite_vs_arguments_amd',
    600: 'nir_intrinsic_pin_cx_handle_nv',
    601: 'nir_intrinsic_preamble_end_ir3',
    602: 'nir_intrinsic_preamble_start_ir3',
    603: 'nir_intrinsic_prefetch_sam_ir3',
    604: 'nir_intrinsic_prefetch_tex_ir3',
    605: 'nir_intrinsic_prefetch_ubo_ir3',
    606: 'nir_intrinsic_printf',
    607: 'nir_intrinsic_printf_abort',
    608: 'nir_intrinsic_quad_ballot_agx',
    609: 'nir_intrinsic_quad_broadcast',
    610: 'nir_intrinsic_quad_swap_diagonal',
    611: 'nir_intrinsic_quad_swap_horizontal',
    612: 'nir_intrinsic_quad_swap_vertical',
    613: 'nir_intrinsic_quad_swizzle_amd',
    614: 'nir_intrinsic_quad_vote_all',
    615: 'nir_intrinsic_quad_vote_any',
    616: 'nir_intrinsic_r600_indirect_vertex_at_index',
    617: 'nir_intrinsic_ray_intersection_ir3',
    618: 'nir_intrinsic_read_attribute_payload_intel',
    619: 'nir_intrinsic_read_first_invocation',
    620: 'nir_intrinsic_read_getlast_ir3',
    621: 'nir_intrinsic_read_invocation',
    622: 'nir_intrinsic_read_invocation_cond_ir3',
    623: 'nir_intrinsic_reduce',
    624: 'nir_intrinsic_reduce_clusters_ir3',
    625: 'nir_intrinsic_report_ray_intersection',
    626: 'nir_intrinsic_resource_intel',
    627: 'nir_intrinsic_rotate',
    628: 'nir_intrinsic_rq_confirm_intersection',
    629: 'nir_intrinsic_rq_generate_intersection',
    630: 'nir_intrinsic_rq_initialize',
    631: 'nir_intrinsic_rq_load',
    632: 'nir_intrinsic_rq_proceed',
    633: 'nir_intrinsic_rq_terminate',
    634: 'nir_intrinsic_rt_execute_callable',
    635: 'nir_intrinsic_rt_resume',
    636: 'nir_intrinsic_rt_return_amd',
    637: 'nir_intrinsic_rt_trace_ray',
    638: 'nir_intrinsic_sample_mask_agx',
    639: 'nir_intrinsic_select_vertex_poly',
    640: 'nir_intrinsic_sendmsg_amd',
    641: 'nir_intrinsic_set_vertex_and_primitive_count',
    642: 'nir_intrinsic_shader_clock',
    643: 'nir_intrinsic_shared_append_amd',
    644: 'nir_intrinsic_shared_atomic',
    645: 'nir_intrinsic_shared_atomic_swap',
    646: 'nir_intrinsic_shared_consume_amd',
    647: 'nir_intrinsic_shuffle',
    648: 'nir_intrinsic_shuffle_down',
    649: 'nir_intrinsic_shuffle_down_uniform_ir3',
    650: 'nir_intrinsic_shuffle_up',
    651: 'nir_intrinsic_shuffle_up_uniform_ir3',
    652: 'nir_intrinsic_shuffle_xor',
    653: 'nir_intrinsic_shuffle_xor_uniform_ir3',
    654: 'nir_intrinsic_sleep_amd',
    655: 'nir_intrinsic_sparse_residency_code_and',
    656: 'nir_intrinsic_ssa_bar_nv',
    657: 'nir_intrinsic_ssbo_atomic',
    658: 'nir_intrinsic_ssbo_atomic_ir3',
    659: 'nir_intrinsic_ssbo_atomic_swap',
    660: 'nir_intrinsic_ssbo_atomic_swap_ir3',
    661: 'nir_intrinsic_stack_map_agx',
    662: 'nir_intrinsic_stack_unmap_agx',
    663: 'nir_intrinsic_store_agx',
    664: 'nir_intrinsic_store_buffer_amd',
    665: 'nir_intrinsic_store_combined_output_pan',
    666: 'nir_intrinsic_store_const_ir3',
    667: 'nir_intrinsic_store_deref',
    668: 'nir_intrinsic_store_deref_block_intel',
    669: 'nir_intrinsic_store_global',
    670: 'nir_intrinsic_store_global_2x32',
    671: 'nir_intrinsic_store_global_amd',
    672: 'nir_intrinsic_store_global_block_intel',
    673: 'nir_intrinsic_store_global_etna',
    674: 'nir_intrinsic_store_global_ir3',
    675: 'nir_intrinsic_store_hit_attrib_amd',
    676: 'nir_intrinsic_store_local_pixel_agx',
    677: 'nir_intrinsic_store_local_shared_r600',
    678: 'nir_intrinsic_store_output',
    679: 'nir_intrinsic_store_per_primitive_output',
    680: 'nir_intrinsic_store_per_primitive_payload_intel',
    681: 'nir_intrinsic_store_per_vertex_output',
    682: 'nir_intrinsic_store_per_view_output',
    683: 'nir_intrinsic_store_preamble',
    684: 'nir_intrinsic_store_raw_output_pan',
    685: 'nir_intrinsic_store_reg',
    686: 'nir_intrinsic_store_reg_indirect',
    687: 'nir_intrinsic_store_scalar_arg_amd',
    688: 'nir_intrinsic_store_scratch',
    689: 'nir_intrinsic_store_shared',
    690: 'nir_intrinsic_store_shared2_amd',
    691: 'nir_intrinsic_store_shared_block_intel',
    692: 'nir_intrinsic_store_shared_ir3',
    693: 'nir_intrinsic_store_shared_unlock_nv',
    694: 'nir_intrinsic_store_ssbo',
    695: 'nir_intrinsic_store_ssbo_block_intel',
    696: 'nir_intrinsic_store_ssbo_intel',
    697: 'nir_intrinsic_store_ssbo_ir3',
    698: 'nir_intrinsic_store_stack',
    699: 'nir_intrinsic_store_task_payload',
    700: 'nir_intrinsic_store_tf_r600',
    701: 'nir_intrinsic_store_tlb_sample_color_v3d',
    702: 'nir_intrinsic_store_uvs_agx',
    703: 'nir_intrinsic_store_vector_arg_amd',
    704: 'nir_intrinsic_store_zs_agx',
    705: 'nir_intrinsic_strict_wqm_coord_amd',
    706: 'nir_intrinsic_subfm_nv',
    707: 'nir_intrinsic_suclamp_nv',
    708: 'nir_intrinsic_sueau_nv',
    709: 'nir_intrinsic_suldga_nv',
    710: 'nir_intrinsic_sustga_nv',
    711: 'nir_intrinsic_task_payload_atomic',
    712: 'nir_intrinsic_task_payload_atomic_swap',
    713: 'nir_intrinsic_terminate',
    714: 'nir_intrinsic_terminate_if',
    715: 'nir_intrinsic_terminate_ray',
    716: 'nir_intrinsic_trace_ray',
    717: 'nir_intrinsic_trace_ray_intel',
    718: 'nir_intrinsic_unit_test_amd',
    719: 'nir_intrinsic_unit_test_divergent_amd',
    720: 'nir_intrinsic_unit_test_uniform_amd',
    721: 'nir_intrinsic_unpin_cx_handle_nv',
    722: 'nir_intrinsic_use',
    723: 'nir_intrinsic_vild_nv',
    724: 'nir_intrinsic_vote_all',
    725: 'nir_intrinsic_vote_any',
    726: 'nir_intrinsic_vote_feq',
    727: 'nir_intrinsic_vote_ieq',
    728: 'nir_intrinsic_vulkan_resource_index',
    729: 'nir_intrinsic_vulkan_resource_reindex',
    730: 'nir_intrinsic_write_invocation_amd',
    731: 'nir_intrinsic_xfb_counter_sub_gfx11_amd',
    731: 'nir_last_intrinsic',
    732: 'nir_num_intrinsics',
}
nir_intrinsic_accept_ray_intersection = 0
nir_intrinsic_addr_mode_is = 1
nir_intrinsic_al2p_nv = 2
nir_intrinsic_ald_nv = 3
nir_intrinsic_alpha_to_coverage = 4
nir_intrinsic_as_uniform = 5
nir_intrinsic_ast_nv = 6
nir_intrinsic_atomic_add_gen_prim_count_amd = 7
nir_intrinsic_atomic_add_gs_emit_prim_count_amd = 8
nir_intrinsic_atomic_add_shader_invocation_count_amd = 9
nir_intrinsic_atomic_add_xfb_prim_count_amd = 10
nir_intrinsic_atomic_counter_add = 11
nir_intrinsic_atomic_counter_add_deref = 12
nir_intrinsic_atomic_counter_and = 13
nir_intrinsic_atomic_counter_and_deref = 14
nir_intrinsic_atomic_counter_comp_swap = 15
nir_intrinsic_atomic_counter_comp_swap_deref = 16
nir_intrinsic_atomic_counter_exchange = 17
nir_intrinsic_atomic_counter_exchange_deref = 18
nir_intrinsic_atomic_counter_inc = 19
nir_intrinsic_atomic_counter_inc_deref = 20
nir_intrinsic_atomic_counter_max = 21
nir_intrinsic_atomic_counter_max_deref = 22
nir_intrinsic_atomic_counter_min = 23
nir_intrinsic_atomic_counter_min_deref = 24
nir_intrinsic_atomic_counter_or = 25
nir_intrinsic_atomic_counter_or_deref = 26
nir_intrinsic_atomic_counter_post_dec = 27
nir_intrinsic_atomic_counter_post_dec_deref = 28
nir_intrinsic_atomic_counter_pre_dec = 29
nir_intrinsic_atomic_counter_pre_dec_deref = 30
nir_intrinsic_atomic_counter_read = 31
nir_intrinsic_atomic_counter_read_deref = 32
nir_intrinsic_atomic_counter_xor = 33
nir_intrinsic_atomic_counter_xor_deref = 34
nir_intrinsic_ballot = 35
nir_intrinsic_ballot_bit_count_exclusive = 36
nir_intrinsic_ballot_bit_count_inclusive = 37
nir_intrinsic_ballot_bit_count_reduce = 38
nir_intrinsic_ballot_bitfield_extract = 39
nir_intrinsic_ballot_find_lsb = 40
nir_intrinsic_ballot_find_msb = 41
nir_intrinsic_ballot_relaxed = 42
nir_intrinsic_bar_break_nv = 43
nir_intrinsic_bar_set_nv = 44
nir_intrinsic_bar_sync_nv = 45
nir_intrinsic_barrier = 46
nir_intrinsic_begin_invocation_interlock = 47
nir_intrinsic_bindgen_return = 48
nir_intrinsic_bindless_image_agx = 49
nir_intrinsic_bindless_image_atomic = 50
nir_intrinsic_bindless_image_atomic_swap = 51
nir_intrinsic_bindless_image_descriptor_amd = 52
nir_intrinsic_bindless_image_format = 53
nir_intrinsic_bindless_image_fragment_mask_load_amd = 54
nir_intrinsic_bindless_image_levels = 55
nir_intrinsic_bindless_image_load = 56
nir_intrinsic_bindless_image_load_raw_intel = 57
nir_intrinsic_bindless_image_order = 58
nir_intrinsic_bindless_image_samples = 59
nir_intrinsic_bindless_image_samples_identical = 60
nir_intrinsic_bindless_image_size = 61
nir_intrinsic_bindless_image_sparse_load = 62
nir_intrinsic_bindless_image_store = 63
nir_intrinsic_bindless_image_store_block_agx = 64
nir_intrinsic_bindless_image_store_raw_intel = 65
nir_intrinsic_bindless_image_texel_address = 66
nir_intrinsic_bindless_resource_ir3 = 67
nir_intrinsic_brcst_active_ir3 = 68
nir_intrinsic_btd_retire_intel = 69
nir_intrinsic_btd_spawn_intel = 70
nir_intrinsic_btd_stack_push_intel = 71
nir_intrinsic_bvh64_intersect_ray_amd = 72
nir_intrinsic_bvh8_intersect_ray_amd = 73
nir_intrinsic_bvh_stack_rtn_amd = 74
nir_intrinsic_cmat_binary_op = 75
nir_intrinsic_cmat_bitcast = 76
nir_intrinsic_cmat_construct = 77
nir_intrinsic_cmat_convert = 78
nir_intrinsic_cmat_copy = 79
nir_intrinsic_cmat_extract = 80
nir_intrinsic_cmat_insert = 81
nir_intrinsic_cmat_length = 82
nir_intrinsic_cmat_load = 83
nir_intrinsic_cmat_muladd = 84
nir_intrinsic_cmat_muladd_amd = 85
nir_intrinsic_cmat_muladd_nv = 86
nir_intrinsic_cmat_scalar_op = 87
nir_intrinsic_cmat_store = 88
nir_intrinsic_cmat_transpose = 89
nir_intrinsic_cmat_unary_op = 90
nir_intrinsic_convert_alu_types = 91
nir_intrinsic_convert_cmat_intel = 92
nir_intrinsic_copy_deref = 93
nir_intrinsic_copy_fs_outputs_nv = 94
nir_intrinsic_copy_global_to_uniform_ir3 = 95
nir_intrinsic_copy_push_const_to_uniform_ir3 = 96
nir_intrinsic_copy_ubo_to_uniform_ir3 = 97
nir_intrinsic_ddx = 98
nir_intrinsic_ddx_coarse = 99
nir_intrinsic_ddx_fine = 100
nir_intrinsic_ddy = 101
nir_intrinsic_ddy_coarse = 102
nir_intrinsic_ddy_fine = 103
nir_intrinsic_debug_break = 104
nir_intrinsic_decl_reg = 105
nir_intrinsic_demote = 106
nir_intrinsic_demote_if = 107
nir_intrinsic_demote_samples = 108
nir_intrinsic_deref_atomic = 109
nir_intrinsic_deref_atomic_swap = 110
nir_intrinsic_deref_buffer_array_length = 111
nir_intrinsic_deref_implicit_array_length = 112
nir_intrinsic_deref_mode_is = 113
nir_intrinsic_deref_texture_src = 114
nir_intrinsic_doorbell_agx = 115
nir_intrinsic_dpas_intel = 116
nir_intrinsic_dpp16_shift_amd = 117
nir_intrinsic_elect = 118
nir_intrinsic_elect_any_ir3 = 119
nir_intrinsic_emit_primitive_poly = 120
nir_intrinsic_emit_vertex = 121
nir_intrinsic_emit_vertex_nv = 122
nir_intrinsic_emit_vertex_with_counter = 123
nir_intrinsic_end_invocation_interlock = 124
nir_intrinsic_end_primitive = 125
nir_intrinsic_end_primitive_nv = 126
nir_intrinsic_end_primitive_with_counter = 127
nir_intrinsic_enqueue_node_payloads = 128
nir_intrinsic_exclusive_scan = 129
nir_intrinsic_exclusive_scan_clusters_ir3 = 130
nir_intrinsic_execute_callable = 131
nir_intrinsic_execute_closest_hit_amd = 132
nir_intrinsic_execute_miss_amd = 133
nir_intrinsic_export_agx = 134
nir_intrinsic_export_amd = 135
nir_intrinsic_export_dual_src_blend_amd = 136
nir_intrinsic_export_row_amd = 137
nir_intrinsic_fence_helper_exit_agx = 138
nir_intrinsic_fence_mem_to_tex_agx = 139
nir_intrinsic_fence_pbe_to_tex_agx = 140
nir_intrinsic_fence_pbe_to_tex_pixel_agx = 141
nir_intrinsic_final_primitive_nv = 142
nir_intrinsic_finalize_incoming_node_payload = 143
nir_intrinsic_first_invocation = 144
nir_intrinsic_fs_out_nv = 145
nir_intrinsic_gds_atomic_add_amd = 146
nir_intrinsic_get_ssbo_size = 147
nir_intrinsic_get_ubo_size = 148
nir_intrinsic_global_atomic = 149
nir_intrinsic_global_atomic_2x32 = 150
nir_intrinsic_global_atomic_agx = 151
nir_intrinsic_global_atomic_amd = 152
nir_intrinsic_global_atomic_swap = 153
nir_intrinsic_global_atomic_swap_2x32 = 154
nir_intrinsic_global_atomic_swap_agx = 155
nir_intrinsic_global_atomic_swap_amd = 156
nir_intrinsic_ignore_ray_intersection = 157
nir_intrinsic_imadsp_nv = 158
nir_intrinsic_image_atomic = 159
nir_intrinsic_image_atomic_swap = 160
nir_intrinsic_image_deref_atomic = 161
nir_intrinsic_image_deref_atomic_swap = 162
nir_intrinsic_image_deref_descriptor_amd = 163
nir_intrinsic_image_deref_format = 164
nir_intrinsic_image_deref_fragment_mask_load_amd = 165
nir_intrinsic_image_deref_levels = 166
nir_intrinsic_image_deref_load = 167
nir_intrinsic_image_deref_load_info_nv = 168
nir_intrinsic_image_deref_load_param_intel = 169
nir_intrinsic_image_deref_load_raw_intel = 170
nir_intrinsic_image_deref_order = 171
nir_intrinsic_image_deref_samples = 172
nir_intrinsic_image_deref_samples_identical = 173
nir_intrinsic_image_deref_size = 174
nir_intrinsic_image_deref_sparse_load = 175
nir_intrinsic_image_deref_store = 176
nir_intrinsic_image_deref_store_block_agx = 177
nir_intrinsic_image_deref_store_raw_intel = 178
nir_intrinsic_image_deref_texel_address = 179
nir_intrinsic_image_descriptor_amd = 180
nir_intrinsic_image_format = 181
nir_intrinsic_image_fragment_mask_load_amd = 182
nir_intrinsic_image_levels = 183
nir_intrinsic_image_load = 184
nir_intrinsic_image_load_raw_intel = 185
nir_intrinsic_image_order = 186
nir_intrinsic_image_samples = 187
nir_intrinsic_image_samples_identical = 188
nir_intrinsic_image_size = 189
nir_intrinsic_image_sparse_load = 190
nir_intrinsic_image_store = 191
nir_intrinsic_image_store_block_agx = 192
nir_intrinsic_image_store_raw_intel = 193
nir_intrinsic_image_texel_address = 194
nir_intrinsic_inclusive_scan = 195
nir_intrinsic_inclusive_scan_clusters_ir3 = 196
nir_intrinsic_initialize_node_payloads = 197
nir_intrinsic_interp_deref_at_centroid = 198
nir_intrinsic_interp_deref_at_offset = 199
nir_intrinsic_interp_deref_at_sample = 200
nir_intrinsic_interp_deref_at_vertex = 201
nir_intrinsic_inverse_ballot = 202
nir_intrinsic_ipa_nv = 203
nir_intrinsic_is_helper_invocation = 204
nir_intrinsic_is_sparse_resident_zink = 205
nir_intrinsic_is_sparse_texels_resident = 206
nir_intrinsic_is_subgroup_invocation_lt_amd = 207
nir_intrinsic_isberd_nv = 208
nir_intrinsic_lane_permute_16_amd = 209
nir_intrinsic_last_invocation = 210
nir_intrinsic_launch_mesh_workgroups = 211
nir_intrinsic_launch_mesh_workgroups_with_payload_deref = 212
nir_intrinsic_ldc_nv = 213
nir_intrinsic_ldcx_nv = 214
nir_intrinsic_ldtram_nv = 215
nir_intrinsic_load_aa_line_width = 216
nir_intrinsic_load_accel_struct_amd = 217
nir_intrinsic_load_active_samples_agx = 218
nir_intrinsic_load_active_subgroup_count_agx = 219
nir_intrinsic_load_active_subgroup_invocation_agx = 220
nir_intrinsic_load_agx = 221
nir_intrinsic_load_alpha_reference_amd = 222
nir_intrinsic_load_api_sample_mask_agx = 223
nir_intrinsic_load_attrib_clamp_agx = 224
nir_intrinsic_load_attribute_pan = 225
nir_intrinsic_load_back_face_agx = 226
nir_intrinsic_load_barycentric_at_offset = 227
nir_intrinsic_load_barycentric_at_offset_nv = 228
nir_intrinsic_load_barycentric_at_sample = 229
nir_intrinsic_load_barycentric_centroid = 230
nir_intrinsic_load_barycentric_coord_at_offset = 231
nir_intrinsic_load_barycentric_coord_at_sample = 232
nir_intrinsic_load_barycentric_coord_centroid = 233
nir_intrinsic_load_barycentric_coord_pixel = 234
nir_intrinsic_load_barycentric_coord_sample = 235
nir_intrinsic_load_barycentric_model = 236
nir_intrinsic_load_barycentric_optimize_amd = 237
nir_intrinsic_load_barycentric_pixel = 238
nir_intrinsic_load_barycentric_sample = 239
nir_intrinsic_load_base_global_invocation_id = 240
nir_intrinsic_load_base_instance = 241
nir_intrinsic_load_base_vertex = 242
nir_intrinsic_load_base_workgroup_id = 243
nir_intrinsic_load_blend_const_color_a_float = 244
nir_intrinsic_load_blend_const_color_aaaa8888_unorm = 245
nir_intrinsic_load_blend_const_color_b_float = 246
nir_intrinsic_load_blend_const_color_g_float = 247
nir_intrinsic_load_blend_const_color_r_float = 248
nir_intrinsic_load_blend_const_color_rgba = 249
nir_intrinsic_load_blend_const_color_rgba8888_unorm = 250
nir_intrinsic_load_btd_global_arg_addr_intel = 251
nir_intrinsic_load_btd_local_arg_addr_intel = 252
nir_intrinsic_load_btd_resume_sbt_addr_intel = 253
nir_intrinsic_load_btd_shader_type_intel = 254
nir_intrinsic_load_btd_stack_id_intel = 255
nir_intrinsic_load_buffer_amd = 256
nir_intrinsic_load_callable_sbt_addr_intel = 257
nir_intrinsic_load_callable_sbt_stride_intel = 258
nir_intrinsic_load_clamp_vertex_color_amd = 259
nir_intrinsic_load_clip_half_line_width_amd = 260
nir_intrinsic_load_clip_z_coeff_agx = 261
nir_intrinsic_load_coalesced_input_count = 262
nir_intrinsic_load_coefficients_agx = 263
nir_intrinsic_load_color0 = 264
nir_intrinsic_load_color1 = 265
nir_intrinsic_load_const_buf_base_addr_lvp = 266
nir_intrinsic_load_const_ir3 = 267
nir_intrinsic_load_constant = 268
nir_intrinsic_load_constant_agx = 269
nir_intrinsic_load_constant_base_ptr = 270
nir_intrinsic_load_converted_output_pan = 271
nir_intrinsic_load_core_id_agx = 272
nir_intrinsic_load_cull_any_enabled_amd = 273
nir_intrinsic_load_cull_back_face_enabled_amd = 274
nir_intrinsic_load_cull_ccw_amd = 275
nir_intrinsic_load_cull_front_face_enabled_amd = 276
nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd = 277
nir_intrinsic_load_cull_mask = 278
nir_intrinsic_load_cull_mask_and_flags_amd = 279
nir_intrinsic_load_cull_small_line_precision_amd = 280
nir_intrinsic_load_cull_small_lines_enabled_amd = 281
nir_intrinsic_load_cull_small_triangle_precision_amd = 282
nir_intrinsic_load_cull_small_triangles_enabled_amd = 283
nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd = 284
nir_intrinsic_load_debug_log_desc_amd = 285
nir_intrinsic_load_depth_never_agx = 286
nir_intrinsic_load_deref = 287
nir_intrinsic_load_deref_block_intel = 288
nir_intrinsic_load_draw_id = 289
nir_intrinsic_load_esgs_vertex_stride_amd = 290
nir_intrinsic_load_exported_agx = 291
nir_intrinsic_load_fb_layers_v3d = 292
nir_intrinsic_load_fbfetch_image_desc_amd = 293
nir_intrinsic_load_fbfetch_image_fmask_desc_amd = 294
nir_intrinsic_load_fep_w_v3d = 295
nir_intrinsic_load_first_vertex = 296
nir_intrinsic_load_fixed_point_size_agx = 297
nir_intrinsic_load_flat_mask = 298
nir_intrinsic_load_force_vrs_rates_amd = 299
nir_intrinsic_load_frag_coord = 300
nir_intrinsic_load_frag_coord_unscaled_ir3 = 301
nir_intrinsic_load_frag_coord_w = 302
nir_intrinsic_load_frag_coord_z = 303
nir_intrinsic_load_frag_coord_zw_pan = 304
nir_intrinsic_load_frag_invocation_count = 305
nir_intrinsic_load_frag_offset_ir3 = 306
nir_intrinsic_load_frag_shading_rate = 307
nir_intrinsic_load_frag_size = 308
nir_intrinsic_load_frag_size_ir3 = 309
nir_intrinsic_load_from_texture_handle_agx = 310
nir_intrinsic_load_front_face = 311
nir_intrinsic_load_front_face_fsign = 312
nir_intrinsic_load_fs_input_interp_deltas = 313
nir_intrinsic_load_fs_msaa_intel = 314
nir_intrinsic_load_fully_covered = 315
nir_intrinsic_load_geometry_param_buffer_poly = 316
nir_intrinsic_load_global = 317
nir_intrinsic_load_global_2x32 = 318
nir_intrinsic_load_global_amd = 319
nir_intrinsic_load_global_base_ptr = 320
nir_intrinsic_load_global_block_intel = 321
nir_intrinsic_load_global_bounded = 322
nir_intrinsic_load_global_constant = 323
nir_intrinsic_load_global_constant_bounded = 324
nir_intrinsic_load_global_constant_offset = 325
nir_intrinsic_load_global_constant_uniform_block_intel = 326
nir_intrinsic_load_global_etna = 327
nir_intrinsic_load_global_invocation_id = 328
nir_intrinsic_load_global_invocation_index = 329
nir_intrinsic_load_global_ir3 = 330
nir_intrinsic_load_global_size = 331
nir_intrinsic_load_gs_header_ir3 = 332
nir_intrinsic_load_gs_vertex_offset_amd = 333
nir_intrinsic_load_gs_wave_id_amd = 334
nir_intrinsic_load_helper_arg_hi_agx = 335
nir_intrinsic_load_helper_arg_lo_agx = 336
nir_intrinsic_load_helper_invocation = 337
nir_intrinsic_load_helper_op_id_agx = 338
nir_intrinsic_load_hit_attrib_amd = 339
nir_intrinsic_load_hs_out_patch_data_offset_amd = 340
nir_intrinsic_load_hs_patch_stride_ir3 = 341
nir_intrinsic_load_initial_edgeflags_amd = 342
nir_intrinsic_load_inline_data_intel = 343
nir_intrinsic_load_input = 344
nir_intrinsic_load_input_assembly_buffer_poly = 345
nir_intrinsic_load_input_attachment_conv_pan = 346
nir_intrinsic_load_input_attachment_coord = 347
nir_intrinsic_load_input_attachment_target_pan = 348
nir_intrinsic_load_input_topology_poly = 349
nir_intrinsic_load_input_vertex = 350
nir_intrinsic_load_instance_id = 351
nir_intrinsic_load_interpolated_input = 352
nir_intrinsic_load_intersection_opaque_amd = 353
nir_intrinsic_load_invocation_id = 354
nir_intrinsic_load_is_first_fan_agx = 355
nir_intrinsic_load_is_indexed_draw = 356
nir_intrinsic_load_kernel_input = 357
nir_intrinsic_load_layer_id = 358
nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd = 359
nir_intrinsic_load_leaf_opaque_intel = 360
nir_intrinsic_load_leaf_procedural_intel = 361
nir_intrinsic_load_line_coord = 362
nir_intrinsic_load_line_width = 363
nir_intrinsic_load_local_invocation_id = 364
nir_intrinsic_load_local_invocation_index = 365
nir_intrinsic_load_local_pixel_agx = 366
nir_intrinsic_load_local_shared_r600 = 367
nir_intrinsic_load_lshs_vertex_stride_amd = 368
nir_intrinsic_load_max_polygon_intel = 369
nir_intrinsic_load_merged_wave_info_amd = 370
nir_intrinsic_load_mesh_view_count = 371
nir_intrinsic_load_mesh_view_indices = 372
nir_intrinsic_load_multisampled_pan = 373
nir_intrinsic_load_noperspective_varyings_pan = 374
nir_intrinsic_load_num_subgroups = 375
nir_intrinsic_load_num_vertices = 376
nir_intrinsic_load_num_vertices_per_primitive_amd = 377
nir_intrinsic_load_num_workgroups = 378
nir_intrinsic_load_ordered_id_amd = 379
nir_intrinsic_load_output = 380
nir_intrinsic_load_packed_passthrough_primitive_amd = 381
nir_intrinsic_load_param = 382
nir_intrinsic_load_patch_vertices_in = 383
nir_intrinsic_load_per_primitive_input = 384
nir_intrinsic_load_per_primitive_output = 385
nir_intrinsic_load_per_primitive_remap_intel = 386
nir_intrinsic_load_per_vertex_input = 387
nir_intrinsic_load_per_vertex_output = 388
nir_intrinsic_load_per_view_output = 389
nir_intrinsic_load_persp_center_rhw_ir3 = 390
nir_intrinsic_load_pipeline_stat_query_enabled_amd = 391
nir_intrinsic_load_pixel_coord = 392
nir_intrinsic_load_point_coord = 393
nir_intrinsic_load_point_coord_maybe_flipped = 394
nir_intrinsic_load_poly_line_smooth_enabled = 395
nir_intrinsic_load_polygon_stipple_agx = 396
nir_intrinsic_load_polygon_stipple_buffer_amd = 397
nir_intrinsic_load_preamble = 398
nir_intrinsic_load_prim_gen_query_enabled_amd = 399
nir_intrinsic_load_prim_xfb_query_enabled_amd = 400
nir_intrinsic_load_primitive_id = 401
nir_intrinsic_load_primitive_location_ir3 = 402
nir_intrinsic_load_printf_buffer_address = 403
nir_intrinsic_load_printf_buffer_size = 404
nir_intrinsic_load_provoking_last = 405
nir_intrinsic_load_provoking_vtx_amd = 406
nir_intrinsic_load_provoking_vtx_in_prim_amd = 407
nir_intrinsic_load_push_constant = 408
nir_intrinsic_load_push_constant_zink = 409
nir_intrinsic_load_r600_indirect_per_vertex_input = 410
nir_intrinsic_load_rasterization_primitive_amd = 411
nir_intrinsic_load_rasterization_samples_amd = 412
nir_intrinsic_load_rasterization_stream = 413
nir_intrinsic_load_raw_output_pan = 414
nir_intrinsic_load_raw_vertex_id_pan = 415
nir_intrinsic_load_raw_vertex_offset_pan = 416
nir_intrinsic_load_ray_base_mem_addr_intel = 417
nir_intrinsic_load_ray_flags = 418
nir_intrinsic_load_ray_geometry_index = 419
nir_intrinsic_load_ray_hit_kind = 420
nir_intrinsic_load_ray_hit_sbt_addr_intel = 421
nir_intrinsic_load_ray_hit_sbt_stride_intel = 422
nir_intrinsic_load_ray_hw_stack_size_intel = 423
nir_intrinsic_load_ray_instance_custom_index = 424
nir_intrinsic_load_ray_launch_id = 425
nir_intrinsic_load_ray_launch_size = 426
nir_intrinsic_load_ray_miss_sbt_addr_intel = 427
nir_intrinsic_load_ray_miss_sbt_stride_intel = 428
nir_intrinsic_load_ray_num_dss_rt_stacks_intel = 429
nir_intrinsic_load_ray_object_direction = 430
nir_intrinsic_load_ray_object_origin = 431
nir_intrinsic_load_ray_object_to_world = 432
nir_intrinsic_load_ray_query_global_intel = 433
nir_intrinsic_load_ray_sw_stack_size_intel = 434
nir_intrinsic_load_ray_t_max = 435
nir_intrinsic_load_ray_t_min = 436
nir_intrinsic_load_ray_tracing_stack_base_lvp = 437
nir_intrinsic_load_ray_triangle_vertex_positions = 438
nir_intrinsic_load_ray_world_direction = 439
nir_intrinsic_load_ray_world_origin = 440
nir_intrinsic_load_ray_world_to_object = 441
nir_intrinsic_load_readonly_output_pan = 442
nir_intrinsic_load_reg = 443
nir_intrinsic_load_reg_indirect = 444
nir_intrinsic_load_rel_patch_id_ir3 = 445
nir_intrinsic_load_reloc_const_intel = 446
nir_intrinsic_load_resume_shader_address_amd = 447
nir_intrinsic_load_ring_attr_amd = 448
nir_intrinsic_load_ring_attr_offset_amd = 449
nir_intrinsic_load_ring_es2gs_offset_amd = 450
nir_intrinsic_load_ring_esgs_amd = 451
nir_intrinsic_load_ring_gs2vs_offset_amd = 452
nir_intrinsic_load_ring_gsvs_amd = 453
nir_intrinsic_load_ring_mesh_scratch_amd = 454
nir_intrinsic_load_ring_mesh_scratch_offset_amd = 455
nir_intrinsic_load_ring_task_draw_amd = 456
nir_intrinsic_load_ring_task_payload_amd = 457
nir_intrinsic_load_ring_tess_factors_amd = 458
nir_intrinsic_load_ring_tess_factors_offset_amd = 459
nir_intrinsic_load_ring_tess_offchip_amd = 460
nir_intrinsic_load_ring_tess_offchip_offset_amd = 461
nir_intrinsic_load_root_agx = 462
nir_intrinsic_load_rt_arg_scratch_offset_amd = 463
nir_intrinsic_load_rt_conversion_pan = 464
nir_intrinsic_load_sample_id = 465
nir_intrinsic_load_sample_id_no_per_sample = 466
nir_intrinsic_load_sample_mask = 467
nir_intrinsic_load_sample_mask_in = 468
nir_intrinsic_load_sample_pos = 469
nir_intrinsic_load_sample_pos_from_id = 470
nir_intrinsic_load_sample_pos_or_center = 471
nir_intrinsic_load_sample_positions_agx = 472
nir_intrinsic_load_sample_positions_amd = 473
nir_intrinsic_load_sample_positions_pan = 474
nir_intrinsic_load_sampler_handle_agx = 475
nir_intrinsic_load_sampler_lod_parameters = 476
nir_intrinsic_load_samples_log2_agx = 477
nir_intrinsic_load_sbt_base_amd = 478
nir_intrinsic_load_sbt_offset_amd = 479
nir_intrinsic_load_sbt_stride_amd = 480
nir_intrinsic_load_scalar_arg_amd = 481
nir_intrinsic_load_scratch = 482
nir_intrinsic_load_scratch_base_ptr = 483
nir_intrinsic_load_shader_call_data_offset_lvp = 484
nir_intrinsic_load_shader_index = 485
nir_intrinsic_load_shader_output_pan = 486
nir_intrinsic_load_shader_part_tests_zs_agx = 487
nir_intrinsic_load_shader_record_ptr = 488
nir_intrinsic_load_shared = 489
nir_intrinsic_load_shared2_amd = 490
nir_intrinsic_load_shared_base_ptr = 491
nir_intrinsic_load_shared_block_intel = 492
nir_intrinsic_load_shared_ir3 = 493
nir_intrinsic_load_shared_lock_nv = 494
nir_intrinsic_load_shared_uniform_block_intel = 495
nir_intrinsic_load_simd_width_intel = 496
nir_intrinsic_load_sm_count_nv = 497
nir_intrinsic_load_sm_id_nv = 498
nir_intrinsic_load_smem_amd = 499
nir_intrinsic_load_ssbo = 500
nir_intrinsic_load_ssbo_address = 501
nir_intrinsic_load_ssbo_block_intel = 502
nir_intrinsic_load_ssbo_intel = 503
nir_intrinsic_load_ssbo_ir3 = 504
nir_intrinsic_load_ssbo_uniform_block_intel = 505
nir_intrinsic_load_stack = 506
nir_intrinsic_load_stat_query_address_agx = 507
nir_intrinsic_load_streamout_buffer_amd = 508
nir_intrinsic_load_streamout_config_amd = 509
nir_intrinsic_load_streamout_offset_amd = 510
nir_intrinsic_load_streamout_write_index_amd = 511
nir_intrinsic_load_subgroup_eq_mask = 512
nir_intrinsic_load_subgroup_ge_mask = 513
nir_intrinsic_load_subgroup_gt_mask = 514
nir_intrinsic_load_subgroup_id = 515
nir_intrinsic_load_subgroup_id_shift_ir3 = 516
nir_intrinsic_load_subgroup_invocation = 517
nir_intrinsic_load_subgroup_le_mask = 518
nir_intrinsic_load_subgroup_lt_mask = 519
nir_intrinsic_load_subgroup_size = 520
nir_intrinsic_load_sysval_agx = 521
nir_intrinsic_load_sysval_nv = 522
nir_intrinsic_load_task_payload = 523
nir_intrinsic_load_task_ring_entry_amd = 524
nir_intrinsic_load_tcs_header_ir3 = 525
nir_intrinsic_load_tcs_in_param_base_r600 = 526
nir_intrinsic_load_tcs_mem_attrib_stride = 527
nir_intrinsic_load_tcs_num_patches_amd = 528
nir_intrinsic_load_tcs_out_param_base_r600 = 529
nir_intrinsic_load_tcs_primitive_mode_amd = 530
nir_intrinsic_load_tcs_rel_patch_id_r600 = 531
nir_intrinsic_load_tcs_tess_factor_base_r600 = 532
nir_intrinsic_load_tcs_tess_levels_to_tes_amd = 533
nir_intrinsic_load_tess_coord = 534
nir_intrinsic_load_tess_coord_xy = 535
nir_intrinsic_load_tess_factor_base_ir3 = 536
nir_intrinsic_load_tess_level_inner = 537
nir_intrinsic_load_tess_level_inner_default = 538
nir_intrinsic_load_tess_level_outer = 539
nir_intrinsic_load_tess_level_outer_default = 540
nir_intrinsic_load_tess_param_base_ir3 = 541
nir_intrinsic_load_tess_param_buffer_poly = 542
nir_intrinsic_load_tess_rel_patch_id_amd = 543
nir_intrinsic_load_tex_sprite_mask_agx = 544
nir_intrinsic_load_texture_handle_agx = 545
nir_intrinsic_load_texture_scale = 546
nir_intrinsic_load_texture_size_etna = 547
nir_intrinsic_load_tlb_color_brcm = 548
nir_intrinsic_load_topology_id_intel = 549
nir_intrinsic_load_typed_buffer_amd = 550
nir_intrinsic_load_uav_ir3 = 551
nir_intrinsic_load_ubo = 552
nir_intrinsic_load_ubo_uniform_block_intel = 553
nir_intrinsic_load_ubo_vec4 = 554
nir_intrinsic_load_uniform = 555
nir_intrinsic_load_user_clip_plane = 556
nir_intrinsic_load_user_data_amd = 557
nir_intrinsic_load_uvs_index_agx = 558
nir_intrinsic_load_vbo_base_agx = 559
nir_intrinsic_load_vector_arg_amd = 560
nir_intrinsic_load_vertex_id = 561
nir_intrinsic_load_vertex_id_zero_base = 562
nir_intrinsic_load_view_index = 563
nir_intrinsic_load_viewport_offset = 564
nir_intrinsic_load_viewport_scale = 565
nir_intrinsic_load_viewport_x_offset = 566
nir_intrinsic_load_viewport_x_scale = 567
nir_intrinsic_load_viewport_y_offset = 568
nir_intrinsic_load_viewport_y_scale = 569
nir_intrinsic_load_viewport_z_offset = 570
nir_intrinsic_load_viewport_z_scale = 571
nir_intrinsic_load_vs_output_buffer_poly = 572
nir_intrinsic_load_vs_outputs_poly = 573
nir_intrinsic_load_vs_primitive_stride_ir3 = 574
nir_intrinsic_load_vs_vertex_stride_ir3 = 575
nir_intrinsic_load_vulkan_descriptor = 576
nir_intrinsic_load_warp_id_nv = 577
nir_intrinsic_load_warps_per_sm_nv = 578
nir_intrinsic_load_work_dim = 579
nir_intrinsic_load_workgroup_id = 580
nir_intrinsic_load_workgroup_index = 581
nir_intrinsic_load_workgroup_num_input_primitives_amd = 582
nir_intrinsic_load_workgroup_num_input_vertices_amd = 583
nir_intrinsic_load_workgroup_size = 584
nir_intrinsic_load_xfb_address = 585
nir_intrinsic_load_xfb_index_buffer = 586
nir_intrinsic_load_xfb_size = 587
nir_intrinsic_load_xfb_state_address_gfx12_amd = 588
nir_intrinsic_masked_swizzle_amd = 589
nir_intrinsic_mbcnt_amd = 590
nir_intrinsic_memcpy_deref = 591
nir_intrinsic_nop = 592
nir_intrinsic_nop_amd = 593
nir_intrinsic_optimization_barrier_sgpr_amd = 594
nir_intrinsic_optimization_barrier_vgpr_amd = 595
nir_intrinsic_ordered_add_loop_gfx12_amd = 596
nir_intrinsic_ordered_xfb_counter_add_gfx11_amd = 597
nir_intrinsic_overwrite_tes_arguments_amd = 598
nir_intrinsic_overwrite_vs_arguments_amd = 599
nir_intrinsic_pin_cx_handle_nv = 600
nir_intrinsic_preamble_end_ir3 = 601
nir_intrinsic_preamble_start_ir3 = 602
nir_intrinsic_prefetch_sam_ir3 = 603
nir_intrinsic_prefetch_tex_ir3 = 604
nir_intrinsic_prefetch_ubo_ir3 = 605
nir_intrinsic_printf = 606
nir_intrinsic_printf_abort = 607
nir_intrinsic_quad_ballot_agx = 608
nir_intrinsic_quad_broadcast = 609
nir_intrinsic_quad_swap_diagonal = 610
nir_intrinsic_quad_swap_horizontal = 611
nir_intrinsic_quad_swap_vertical = 612
nir_intrinsic_quad_swizzle_amd = 613
nir_intrinsic_quad_vote_all = 614
nir_intrinsic_quad_vote_any = 615
nir_intrinsic_r600_indirect_vertex_at_index = 616
nir_intrinsic_ray_intersection_ir3 = 617
nir_intrinsic_read_attribute_payload_intel = 618
nir_intrinsic_read_first_invocation = 619
nir_intrinsic_read_getlast_ir3 = 620
nir_intrinsic_read_invocation = 621
nir_intrinsic_read_invocation_cond_ir3 = 622
nir_intrinsic_reduce = 623
nir_intrinsic_reduce_clusters_ir3 = 624
nir_intrinsic_report_ray_intersection = 625
nir_intrinsic_resource_intel = 626
nir_intrinsic_rotate = 627
nir_intrinsic_rq_confirm_intersection = 628
nir_intrinsic_rq_generate_intersection = 629
nir_intrinsic_rq_initialize = 630
nir_intrinsic_rq_load = 631
nir_intrinsic_rq_proceed = 632
nir_intrinsic_rq_terminate = 633
nir_intrinsic_rt_execute_callable = 634
nir_intrinsic_rt_resume = 635
nir_intrinsic_rt_return_amd = 636
nir_intrinsic_rt_trace_ray = 637
nir_intrinsic_sample_mask_agx = 638
nir_intrinsic_select_vertex_poly = 639
nir_intrinsic_sendmsg_amd = 640
nir_intrinsic_set_vertex_and_primitive_count = 641
nir_intrinsic_shader_clock = 642
nir_intrinsic_shared_append_amd = 643
nir_intrinsic_shared_atomic = 644
nir_intrinsic_shared_atomic_swap = 645
nir_intrinsic_shared_consume_amd = 646
nir_intrinsic_shuffle = 647
nir_intrinsic_shuffle_down = 648
nir_intrinsic_shuffle_down_uniform_ir3 = 649
nir_intrinsic_shuffle_up = 650
nir_intrinsic_shuffle_up_uniform_ir3 = 651
nir_intrinsic_shuffle_xor = 652
nir_intrinsic_shuffle_xor_uniform_ir3 = 653
nir_intrinsic_sleep_amd = 654
nir_intrinsic_sparse_residency_code_and = 655
nir_intrinsic_ssa_bar_nv = 656
nir_intrinsic_ssbo_atomic = 657
nir_intrinsic_ssbo_atomic_ir3 = 658
nir_intrinsic_ssbo_atomic_swap = 659
nir_intrinsic_ssbo_atomic_swap_ir3 = 660
nir_intrinsic_stack_map_agx = 661
nir_intrinsic_stack_unmap_agx = 662
nir_intrinsic_store_agx = 663
nir_intrinsic_store_buffer_amd = 664
nir_intrinsic_store_combined_output_pan = 665
nir_intrinsic_store_const_ir3 = 666
nir_intrinsic_store_deref = 667
nir_intrinsic_store_deref_block_intel = 668
nir_intrinsic_store_global = 669
nir_intrinsic_store_global_2x32 = 670
nir_intrinsic_store_global_amd = 671
nir_intrinsic_store_global_block_intel = 672
nir_intrinsic_store_global_etna = 673
nir_intrinsic_store_global_ir3 = 674
nir_intrinsic_store_hit_attrib_amd = 675
nir_intrinsic_store_local_pixel_agx = 676
nir_intrinsic_store_local_shared_r600 = 677
nir_intrinsic_store_output = 678
nir_intrinsic_store_per_primitive_output = 679
nir_intrinsic_store_per_primitive_payload_intel = 680
nir_intrinsic_store_per_vertex_output = 681
nir_intrinsic_store_per_view_output = 682
nir_intrinsic_store_preamble = 683
nir_intrinsic_store_raw_output_pan = 684
nir_intrinsic_store_reg = 685
nir_intrinsic_store_reg_indirect = 686
nir_intrinsic_store_scalar_arg_amd = 687
nir_intrinsic_store_scratch = 688
nir_intrinsic_store_shared = 689
nir_intrinsic_store_shared2_amd = 690
nir_intrinsic_store_shared_block_intel = 691
nir_intrinsic_store_shared_ir3 = 692
nir_intrinsic_store_shared_unlock_nv = 693
nir_intrinsic_store_ssbo = 694
nir_intrinsic_store_ssbo_block_intel = 695
nir_intrinsic_store_ssbo_intel = 696
nir_intrinsic_store_ssbo_ir3 = 697
nir_intrinsic_store_stack = 698
nir_intrinsic_store_task_payload = 699
nir_intrinsic_store_tf_r600 = 700
nir_intrinsic_store_tlb_sample_color_v3d = 701
nir_intrinsic_store_uvs_agx = 702
nir_intrinsic_store_vector_arg_amd = 703
nir_intrinsic_store_zs_agx = 704
nir_intrinsic_strict_wqm_coord_amd = 705
nir_intrinsic_subfm_nv = 706
nir_intrinsic_suclamp_nv = 707
nir_intrinsic_sueau_nv = 708
nir_intrinsic_suldga_nv = 709
nir_intrinsic_sustga_nv = 710
nir_intrinsic_task_payload_atomic = 711
nir_intrinsic_task_payload_atomic_swap = 712
nir_intrinsic_terminate = 713
nir_intrinsic_terminate_if = 714
nir_intrinsic_terminate_ray = 715
nir_intrinsic_trace_ray = 716
nir_intrinsic_trace_ray_intel = 717
nir_intrinsic_unit_test_amd = 718
nir_intrinsic_unit_test_divergent_amd = 719
nir_intrinsic_unit_test_uniform_amd = 720
nir_intrinsic_unpin_cx_handle_nv = 721
nir_intrinsic_use = 722
nir_intrinsic_vild_nv = 723
nir_intrinsic_vote_all = 724
nir_intrinsic_vote_any = 725
nir_intrinsic_vote_feq = 726
nir_intrinsic_vote_ieq = 727
nir_intrinsic_vulkan_resource_index = 728
nir_intrinsic_vulkan_resource_reindex = 729
nir_intrinsic_write_invocation_amd = 730
nir_intrinsic_xfb_counter_sub_gfx11_amd = 731
nir_last_intrinsic = 731
nir_num_intrinsics = 732
c__EA_nir_intrinsic_op = ctypes.c_uint32 # enum
nir_intrinsic_op = c__EA_nir_intrinsic_op
nir_intrinsic_op__enumvalues = c__EA_nir_intrinsic_op__enumvalues

# values for enumeration 'c__EA_nir_intrinsic_index_flag'
c__EA_nir_intrinsic_index_flag__enumvalues = {
    0: 'NIR_INTRINSIC_BASE',
    1: 'NIR_INTRINSIC_WRITE_MASK',
    2: 'NIR_INTRINSIC_STREAM_ID',
    3: 'NIR_INTRINSIC_UCP_ID',
    4: 'NIR_INTRINSIC_RANGE_BASE',
    5: 'NIR_INTRINSIC_RANGE',
    6: 'NIR_INTRINSIC_DESC_SET',
    7: 'NIR_INTRINSIC_BINDING',
    8: 'NIR_INTRINSIC_COMPONENT',
    9: 'NIR_INTRINSIC_COLUMN',
    10: 'NIR_INTRINSIC_INTERP_MODE',
    11: 'NIR_INTRINSIC_REDUCTION_OP',
    12: 'NIR_INTRINSIC_CLUSTER_SIZE',
    13: 'NIR_INTRINSIC_PARAM_IDX',
    14: 'NIR_INTRINSIC_IMAGE_DIM',
    15: 'NIR_INTRINSIC_IMAGE_ARRAY',
    16: 'NIR_INTRINSIC_FORMAT',
    17: 'NIR_INTRINSIC_ACCESS',
    18: 'NIR_INTRINSIC_CALL_IDX',
    19: 'NIR_INTRINSIC_STACK_SIZE',
    20: 'NIR_INTRINSIC_ALIGN_MUL',
    21: 'NIR_INTRINSIC_ALIGN_OFFSET',
    22: 'NIR_INTRINSIC_DESC_TYPE',
    23: 'NIR_INTRINSIC_SRC_TYPE',
    24: 'NIR_INTRINSIC_DEST_TYPE',
    25: 'NIR_INTRINSIC_SRC_BASE_TYPE',
    26: 'NIR_INTRINSIC_SRC_BASE_TYPE2',
    27: 'NIR_INTRINSIC_DEST_BASE_TYPE',
    28: 'NIR_INTRINSIC_SWIZZLE_MASK',
    29: 'NIR_INTRINSIC_FETCH_INACTIVE',
    30: 'NIR_INTRINSIC_OFFSET0',
    31: 'NIR_INTRINSIC_OFFSET1',
    32: 'NIR_INTRINSIC_ST64',
    33: 'NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD',
    34: 'NIR_INTRINSIC_DST_ACCESS',
    35: 'NIR_INTRINSIC_SRC_ACCESS',
    36: 'NIR_INTRINSIC_DRIVER_LOCATION',
    37: 'NIR_INTRINSIC_MEMORY_SEMANTICS',
    38: 'NIR_INTRINSIC_MEMORY_MODES',
    39: 'NIR_INTRINSIC_MEMORY_SCOPE',
    40: 'NIR_INTRINSIC_EXECUTION_SCOPE',
    41: 'NIR_INTRINSIC_IO_SEMANTICS',
    42: 'NIR_INTRINSIC_IO_XFB',
    43: 'NIR_INTRINSIC_IO_XFB2',
    44: 'NIR_INTRINSIC_RAY_QUERY_VALUE',
    45: 'NIR_INTRINSIC_COMMITTED',
    46: 'NIR_INTRINSIC_ROUNDING_MODE',
    47: 'NIR_INTRINSIC_SATURATE',
    48: 'NIR_INTRINSIC_SYNCHRONOUS',
    49: 'NIR_INTRINSIC_VALUE_ID',
    50: 'NIR_INTRINSIC_SIGN_EXTEND',
    51: 'NIR_INTRINSIC_FLAGS',
    52: 'NIR_INTRINSIC_ATOMIC_OP',
    53: 'NIR_INTRINSIC_RESOURCE_BLOCK_INTEL',
    54: 'NIR_INTRINSIC_RESOURCE_ACCESS_INTEL',
    55: 'NIR_INTRINSIC_NUM_COMPONENTS',
    56: 'NIR_INTRINSIC_NUM_ARRAY_ELEMS',
    57: 'NIR_INTRINSIC_BIT_SIZE',
    58: 'NIR_INTRINSIC_DIVERGENT',
    59: 'NIR_INTRINSIC_LEGACY_FABS',
    60: 'NIR_INTRINSIC_LEGACY_FNEG',
    61: 'NIR_INTRINSIC_LEGACY_FSAT',
    62: 'NIR_INTRINSIC_CMAT_DESC',
    63: 'NIR_INTRINSIC_MATRIX_LAYOUT',
    64: 'NIR_INTRINSIC_CMAT_SIGNED_MASK',
    65: 'NIR_INTRINSIC_ALU_OP',
    66: 'NIR_INTRINSIC_NEG_LO_AMD',
    67: 'NIR_INTRINSIC_NEG_HI_AMD',
    68: 'NIR_INTRINSIC_SYSTOLIC_DEPTH',
    69: 'NIR_INTRINSIC_REPEAT_COUNT',
    70: 'NIR_INTRINSIC_DST_CMAT_DESC',
    71: 'NIR_INTRINSIC_SRC_CMAT_DESC',
    72: 'NIR_INTRINSIC_EXPLICIT_COORD',
    73: 'NIR_INTRINSIC_FMT_IDX',
    74: 'NIR_INTRINSIC_PREAMBLE_CLASS',
    75: 'NIR_INTRINSIC_NUM_INDEX_FLAGS',
}
NIR_INTRINSIC_BASE = 0
NIR_INTRINSIC_WRITE_MASK = 1
NIR_INTRINSIC_STREAM_ID = 2
NIR_INTRINSIC_UCP_ID = 3
NIR_INTRINSIC_RANGE_BASE = 4
NIR_INTRINSIC_RANGE = 5
NIR_INTRINSIC_DESC_SET = 6
NIR_INTRINSIC_BINDING = 7
NIR_INTRINSIC_COMPONENT = 8
NIR_INTRINSIC_COLUMN = 9
NIR_INTRINSIC_INTERP_MODE = 10
NIR_INTRINSIC_REDUCTION_OP = 11
NIR_INTRINSIC_CLUSTER_SIZE = 12
NIR_INTRINSIC_PARAM_IDX = 13
NIR_INTRINSIC_IMAGE_DIM = 14
NIR_INTRINSIC_IMAGE_ARRAY = 15
NIR_INTRINSIC_FORMAT = 16
NIR_INTRINSIC_ACCESS = 17
NIR_INTRINSIC_CALL_IDX = 18
NIR_INTRINSIC_STACK_SIZE = 19
NIR_INTRINSIC_ALIGN_MUL = 20
NIR_INTRINSIC_ALIGN_OFFSET = 21
NIR_INTRINSIC_DESC_TYPE = 22
NIR_INTRINSIC_SRC_TYPE = 23
NIR_INTRINSIC_DEST_TYPE = 24
NIR_INTRINSIC_SRC_BASE_TYPE = 25
NIR_INTRINSIC_SRC_BASE_TYPE2 = 26
NIR_INTRINSIC_DEST_BASE_TYPE = 27
NIR_INTRINSIC_SWIZZLE_MASK = 28
NIR_INTRINSIC_FETCH_INACTIVE = 29
NIR_INTRINSIC_OFFSET0 = 30
NIR_INTRINSIC_OFFSET1 = 31
NIR_INTRINSIC_ST64 = 32
NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD = 33
NIR_INTRINSIC_DST_ACCESS = 34
NIR_INTRINSIC_SRC_ACCESS = 35
NIR_INTRINSIC_DRIVER_LOCATION = 36
NIR_INTRINSIC_MEMORY_SEMANTICS = 37
NIR_INTRINSIC_MEMORY_MODES = 38
NIR_INTRINSIC_MEMORY_SCOPE = 39
NIR_INTRINSIC_EXECUTION_SCOPE = 40
NIR_INTRINSIC_IO_SEMANTICS = 41
NIR_INTRINSIC_IO_XFB = 42
NIR_INTRINSIC_IO_XFB2 = 43
NIR_INTRINSIC_RAY_QUERY_VALUE = 44
NIR_INTRINSIC_COMMITTED = 45
NIR_INTRINSIC_ROUNDING_MODE = 46
NIR_INTRINSIC_SATURATE = 47
NIR_INTRINSIC_SYNCHRONOUS = 48
NIR_INTRINSIC_VALUE_ID = 49
NIR_INTRINSIC_SIGN_EXTEND = 50
NIR_INTRINSIC_FLAGS = 51
NIR_INTRINSIC_ATOMIC_OP = 52
NIR_INTRINSIC_RESOURCE_BLOCK_INTEL = 53
NIR_INTRINSIC_RESOURCE_ACCESS_INTEL = 54
NIR_INTRINSIC_NUM_COMPONENTS = 55
NIR_INTRINSIC_NUM_ARRAY_ELEMS = 56
NIR_INTRINSIC_BIT_SIZE = 57
NIR_INTRINSIC_DIVERGENT = 58
NIR_INTRINSIC_LEGACY_FABS = 59
NIR_INTRINSIC_LEGACY_FNEG = 60
NIR_INTRINSIC_LEGACY_FSAT = 61
NIR_INTRINSIC_CMAT_DESC = 62
NIR_INTRINSIC_MATRIX_LAYOUT = 63
NIR_INTRINSIC_CMAT_SIGNED_MASK = 64
NIR_INTRINSIC_ALU_OP = 65
NIR_INTRINSIC_NEG_LO_AMD = 66
NIR_INTRINSIC_NEG_HI_AMD = 67
NIR_INTRINSIC_SYSTOLIC_DEPTH = 68
NIR_INTRINSIC_REPEAT_COUNT = 69
NIR_INTRINSIC_DST_CMAT_DESC = 70
NIR_INTRINSIC_SRC_CMAT_DESC = 71
NIR_INTRINSIC_EXPLICIT_COORD = 72
NIR_INTRINSIC_FMT_IDX = 73
NIR_INTRINSIC_PREAMBLE_CLASS = 74
NIR_INTRINSIC_NUM_INDEX_FLAGS = 75
c__EA_nir_intrinsic_index_flag = ctypes.c_uint32 # enum
nir_intrinsic_index_flag = c__EA_nir_intrinsic_index_flag
nir_intrinsic_index_flag__enumvalues = c__EA_nir_intrinsic_index_flag__enumvalues
try: nir_intrinsic_index_names = (ctypes.POINTER(ctypes.c_char) * 75).in_dll(_libraries['libtinymesa_cpu.so'], 'nir_intrinsic_index_names')
except (AttributeError, ValueError): pass
class struct_nir_intrinsic_instr(Structure):
    pass

struct_nir_intrinsic_instr._pack_ = 1 # source:False
struct_nir_intrinsic_instr._fields_ = [
    ('instr', nir_instr),
    ('intrinsic', nir_intrinsic_op),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('def', nir_def),
    ('num_components', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('const_index', ctypes.c_int32 * 8),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('src', struct_nir_src * 0),
]

nir_intrinsic_instr = struct_nir_intrinsic_instr
try:
    nir_intrinsic_get_var = _libraries['FIXME_STUB'].nir_intrinsic_get_var
    nir_intrinsic_get_var.restype = ctypes.POINTER(struct_nir_variable)
    nir_intrinsic_get_var.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_memory_semantics'
c__EA_nir_memory_semantics__enumvalues = {
    1: 'NIR_MEMORY_ACQUIRE',
    2: 'NIR_MEMORY_RELEASE',
    3: 'NIR_MEMORY_ACQ_REL',
    4: 'NIR_MEMORY_MAKE_AVAILABLE',
    8: 'NIR_MEMORY_MAKE_VISIBLE',
}
NIR_MEMORY_ACQUIRE = 1
NIR_MEMORY_RELEASE = 2
NIR_MEMORY_ACQ_REL = 3
NIR_MEMORY_MAKE_AVAILABLE = 4
NIR_MEMORY_MAKE_VISIBLE = 8
c__EA_nir_memory_semantics = ctypes.c_uint32 # enum
nir_memory_semantics = c__EA_nir_memory_semantics
nir_memory_semantics__enumvalues = c__EA_nir_memory_semantics__enumvalues

# values for enumeration 'c__EA_nir_intrinsic_semantic_flag'
c__EA_nir_intrinsic_semantic_flag__enumvalues = {
    1: 'NIR_INTRINSIC_CAN_ELIMINATE',
    2: 'NIR_INTRINSIC_CAN_REORDER',
    4: 'NIR_INTRINSIC_SUBGROUP',
    8: 'NIR_INTRINSIC_QUADGROUP',
}
NIR_INTRINSIC_CAN_ELIMINATE = 1
NIR_INTRINSIC_CAN_REORDER = 2
NIR_INTRINSIC_SUBGROUP = 4
NIR_INTRINSIC_QUADGROUP = 8
c__EA_nir_intrinsic_semantic_flag = ctypes.c_uint32 # enum
nir_intrinsic_semantic_flag = c__EA_nir_intrinsic_semantic_flag
nir_intrinsic_semantic_flag__enumvalues = c__EA_nir_intrinsic_semantic_flag__enumvalues
class struct_nir_io_semantics(Structure):
    pass

struct_nir_io_semantics._pack_ = 1 # source:False
struct_nir_io_semantics._fields_ = [
    ('location', ctypes.c_uint32, 7),
    ('num_slots', ctypes.c_uint32, 6),
    ('dual_source_blend_index', ctypes.c_uint32, 1),
    ('fb_fetch_output', ctypes.c_uint32, 1),
    ('fb_fetch_output_coherent', ctypes.c_uint32, 1),
    ('gs_streams', ctypes.c_uint32, 8),
    ('medium_precision', ctypes.c_uint32, 1),
    ('per_view', ctypes.c_uint32, 1),
    ('high_16bits', ctypes.c_uint32, 1),
    ('high_dvec2', ctypes.c_uint32, 1),
    ('no_varying', ctypes.c_uint32, 1),
    ('no_sysval_output', ctypes.c_uint32, 1),
    ('interp_explicit_strict', ctypes.c_uint32, 1),
    ('_pad', ctypes.c_uint32, 1),
]

nir_io_semantics = struct_nir_io_semantics
class struct_nir_io_xfb(Structure):
    pass

class struct_nir_io_xfb_0(Structure):
    pass

struct_nir_io_xfb_0._pack_ = 1 # source:False
struct_nir_io_xfb_0._fields_ = [
    ('num_components', ctypes.c_ubyte, 4),
    ('buffer', ctypes.c_ubyte, 4),
    ('offset', ctypes.c_ubyte, 8),
]

struct_nir_io_xfb._pack_ = 1 # source:False
struct_nir_io_xfb._fields_ = [
    ('out', struct_nir_io_xfb_0 * 2),
]

nir_io_xfb = struct_nir_io_xfb
try:
    nir_instr_xfb_write_mask = _libraries['libtinymesa_cpu.so'].nir_instr_xfb_write_mask
    nir_instr_xfb_write_mask.restype = ctypes.c_uint32
    nir_instr_xfb_write_mask.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
class struct_nir_intrinsic_info(Structure):
    pass

struct_nir_intrinsic_info._pack_ = 1 # source:False
struct_nir_intrinsic_info._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('num_srcs', ctypes.c_ubyte),
    ('src_components', ctypes.c_byte * 11),
    ('has_dest', ctypes.c_bool),
    ('dest_components', ctypes.c_ubyte),
    ('dest_bit_sizes', ctypes.c_ubyte),
    ('bit_size_src', ctypes.c_byte),
    ('num_indices', ctypes.c_ubyte),
    ('indices', ctypes.c_ubyte * 8),
    ('index_map', ctypes.c_ubyte * 75),
    ('flags', nir_intrinsic_semantic_flag),
]

nir_intrinsic_info = struct_nir_intrinsic_info
try: nir_intrinsic_infos = (struct_nir_intrinsic_info * 732).in_dll(_libraries['libtinymesa_cpu.so'], 'nir_intrinsic_infos')
except (AttributeError, ValueError): pass
try:
    nir_intrinsic_src_components = _libraries['libtinymesa_cpu.so'].nir_intrinsic_src_components
    nir_intrinsic_src_components.restype = ctypes.c_uint32
    nir_intrinsic_src_components.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_intrinsic_dest_components = _libraries['libtinymesa_cpu.so'].nir_intrinsic_dest_components
    nir_intrinsic_dest_components.restype = ctypes.c_uint32
    nir_intrinsic_dest_components.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_instr_src_type = _libraries['libtinymesa_cpu.so'].nir_intrinsic_instr_src_type
    nir_intrinsic_instr_src_type.restype = nir_alu_type
    nir_intrinsic_instr_src_type.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_intrinsic_instr_dest_type = _libraries['libtinymesa_cpu.so'].nir_intrinsic_instr_dest_type
    nir_intrinsic_instr_dest_type.restype = nir_alu_type
    nir_intrinsic_instr_dest_type.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_copy_const_indices = _libraries['libtinymesa_cpu.so'].nir_intrinsic_copy_const_indices
    nir_intrinsic_copy_const_indices.restype = None
    nir_intrinsic_copy_const_indices.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_set_align = _libraries['FIXME_STUB'].nir_intrinsic_set_align
    nir_intrinsic_set_align.restype = None
    nir_intrinsic_set_align.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_combined_align = _libraries['FIXME_STUB'].nir_combined_align
    nir_combined_align.restype = uint32_t
    nir_combined_align.argtypes = [uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_intrinsic_align = _libraries['FIXME_STUB'].nir_intrinsic_align
    nir_intrinsic_align.restype = ctypes.c_uint32
    nir_intrinsic_align.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_has_align = _libraries['FIXME_STUB'].nir_intrinsic_has_align
    nir_intrinsic_has_align.restype = ctypes.c_bool
    nir_intrinsic_has_align.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_image_intrinsic_coord_components = _libraries['libtinymesa_cpu.so'].nir_image_intrinsic_coord_components
    nir_image_intrinsic_coord_components.restype = ctypes.c_uint32
    nir_image_intrinsic_coord_components.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_rewrite_image_intrinsic = _libraries['libtinymesa_cpu.so'].nir_rewrite_image_intrinsic
    nir_rewrite_image_intrinsic.restype = None
    nir_rewrite_image_intrinsic.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_def), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_intrinsic_can_reorder = _libraries['libtinymesa_cpu.so'].nir_intrinsic_can_reorder
    nir_intrinsic_can_reorder.restype = ctypes.c_bool
    nir_intrinsic_can_reorder.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_writes_external_memory = _libraries['libtinymesa_cpu.so'].nir_intrinsic_writes_external_memory
    nir_intrinsic_writes_external_memory.restype = ctypes.c_bool
    nir_intrinsic_writes_external_memory.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_has_semantic = _libraries['FIXME_STUB'].nir_intrinsic_has_semantic
    nir_intrinsic_has_semantic.restype = ctypes.c_bool
    nir_intrinsic_has_semantic.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), nir_intrinsic_semantic_flag]
except AttributeError:
    pass
try:
    nir_intrinsic_is_ray_query = _libraries['FIXME_STUB'].nir_intrinsic_is_ray_query
    nir_intrinsic_is_ray_query.restype = ctypes.c_bool
    nir_intrinsic_is_ray_query.argtypes = [nir_intrinsic_op]
except AttributeError:
    pass

# values for enumeration 'nir_tex_src_type'
nir_tex_src_type__enumvalues = {
    0: 'nir_tex_src_coord',
    1: 'nir_tex_src_projector',
    2: 'nir_tex_src_comparator',
    3: 'nir_tex_src_offset',
    4: 'nir_tex_src_bias',
    5: 'nir_tex_src_lod',
    6: 'nir_tex_src_min_lod',
    7: 'nir_tex_src_lod_bias_min_agx',
    8: 'nir_tex_src_ms_index',
    9: 'nir_tex_src_ms_mcs_intel',
    10: 'nir_tex_src_ddx',
    11: 'nir_tex_src_ddy',
    12: 'nir_tex_src_texture_deref',
    13: 'nir_tex_src_sampler_deref',
    14: 'nir_tex_src_texture_offset',
    15: 'nir_tex_src_sampler_offset',
    16: 'nir_tex_src_texture_handle',
    17: 'nir_tex_src_sampler_handle',
    18: 'nir_tex_src_sampler_deref_intrinsic',
    19: 'nir_tex_src_texture_deref_intrinsic',
    20: 'nir_tex_src_plane',
    21: 'nir_tex_src_backend1',
    22: 'nir_tex_src_backend2',
    23: 'nir_num_tex_src_types',
}
nir_tex_src_coord = 0
nir_tex_src_projector = 1
nir_tex_src_comparator = 2
nir_tex_src_offset = 3
nir_tex_src_bias = 4
nir_tex_src_lod = 5
nir_tex_src_min_lod = 6
nir_tex_src_lod_bias_min_agx = 7
nir_tex_src_ms_index = 8
nir_tex_src_ms_mcs_intel = 9
nir_tex_src_ddx = 10
nir_tex_src_ddy = 11
nir_tex_src_texture_deref = 12
nir_tex_src_sampler_deref = 13
nir_tex_src_texture_offset = 14
nir_tex_src_sampler_offset = 15
nir_tex_src_texture_handle = 16
nir_tex_src_sampler_handle = 17
nir_tex_src_sampler_deref_intrinsic = 18
nir_tex_src_texture_deref_intrinsic = 19
nir_tex_src_plane = 20
nir_tex_src_backend1 = 21
nir_tex_src_backend2 = 22
nir_num_tex_src_types = 23
nir_tex_src_type = ctypes.c_uint32 # enum
class struct_nir_tex_src(Structure):
    pass

struct_nir_tex_src._pack_ = 1 # source:False
struct_nir_tex_src._fields_ = [
    ('src', nir_src),
    ('src_type', nir_tex_src_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

nir_tex_src = struct_nir_tex_src

# values for enumeration 'nir_texop'
nir_texop__enumvalues = {
    0: 'nir_texop_tex',
    1: 'nir_texop_txb',
    2: 'nir_texop_txl',
    3: 'nir_texop_txd',
    4: 'nir_texop_txf',
    5: 'nir_texop_txf_ms',
    6: 'nir_texop_txf_ms_fb',
    7: 'nir_texop_txf_ms_mcs_intel',
    8: 'nir_texop_txs',
    9: 'nir_texop_lod',
    10: 'nir_texop_tg4',
    11: 'nir_texop_query_levels',
    12: 'nir_texop_texture_samples',
    13: 'nir_texop_samples_identical',
    14: 'nir_texop_tex_prefetch',
    15: 'nir_texop_lod_bias',
    16: 'nir_texop_fragment_fetch_amd',
    17: 'nir_texop_fragment_mask_fetch_amd',
    18: 'nir_texop_descriptor_amd',
    19: 'nir_texop_sampler_descriptor_amd',
    20: 'nir_texop_image_min_lod_agx',
    21: 'nir_texop_has_custom_border_color_agx',
    22: 'nir_texop_custom_border_color_agx',
    23: 'nir_texop_hdr_dim_nv',
    24: 'nir_texop_tex_type_nv',
}
nir_texop_tex = 0
nir_texop_txb = 1
nir_texop_txl = 2
nir_texop_txd = 3
nir_texop_txf = 4
nir_texop_txf_ms = 5
nir_texop_txf_ms_fb = 6
nir_texop_txf_ms_mcs_intel = 7
nir_texop_txs = 8
nir_texop_lod = 9
nir_texop_tg4 = 10
nir_texop_query_levels = 11
nir_texop_texture_samples = 12
nir_texop_samples_identical = 13
nir_texop_tex_prefetch = 14
nir_texop_lod_bias = 15
nir_texop_fragment_fetch_amd = 16
nir_texop_fragment_mask_fetch_amd = 17
nir_texop_descriptor_amd = 18
nir_texop_sampler_descriptor_amd = 19
nir_texop_image_min_lod_agx = 20
nir_texop_has_custom_border_color_agx = 21
nir_texop_custom_border_color_agx = 22
nir_texop_hdr_dim_nv = 23
nir_texop_tex_type_nv = 24
nir_texop = ctypes.c_uint32 # enum
class struct_nir_tex_instr(Structure):
    pass

struct_nir_tex_instr._pack_ = 1 # source:False
struct_nir_tex_instr._fields_ = [
    ('instr', nir_instr),
    ('sampler_dim', glsl_sampler_dim),
    ('dest_type', nir_alu_type),
    ('op', nir_texop),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('def', nir_def),
    ('src', ctypes.POINTER(struct_nir_tex_src)),
    ('num_srcs', ctypes.c_uint32),
    ('coord_components', ctypes.c_uint32),
    ('is_array', ctypes.c_bool),
    ('is_shadow', ctypes.c_bool),
    ('is_new_style_shadow', ctypes.c_bool),
    ('is_sparse', ctypes.c_bool),
    ('component', ctypes.c_uint32, 2),
    ('array_is_lowered_cube', ctypes.c_uint32, 1),
    ('is_gather_implicit_lod', ctypes.c_uint32, 1),
    ('skip_helpers', ctypes.c_uint32, 1),
    ('PADDING_1', ctypes.c_uint8, 3),
    ('tg4_offsets', ctypes.c_byte * 2 * 4),
    ('texture_non_uniform', ctypes.c_bool),
    ('sampler_non_uniform', ctypes.c_bool),
    ('offset_non_uniform', ctypes.c_bool),
    ('texture_index', ctypes.c_uint32),
    ('sampler_index', ctypes.c_uint32),
    ('backend_flags', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

nir_tex_instr = struct_nir_tex_instr
try:
    nir_tex_instr_need_sampler = _libraries['libtinymesa_cpu.so'].nir_tex_instr_need_sampler
    nir_tex_instr_need_sampler.restype = ctypes.c_bool
    nir_tex_instr_need_sampler.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_result_size = _libraries['libtinymesa_cpu.so'].nir_tex_instr_result_size
    nir_tex_instr_result_size.restype = ctypes.c_uint32
    nir_tex_instr_result_size.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_dest_size = _libraries['FIXME_STUB'].nir_tex_instr_dest_size
    nir_tex_instr_dest_size.restype = ctypes.c_uint32
    nir_tex_instr_dest_size.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_is_query = _libraries['libtinymesa_cpu.so'].nir_tex_instr_is_query
    nir_tex_instr_is_query.restype = ctypes.c_bool
    nir_tex_instr_is_query.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_has_implicit_derivative = _libraries['libtinymesa_cpu.so'].nir_tex_instr_has_implicit_derivative
    nir_tex_instr_has_implicit_derivative.restype = ctypes.c_bool
    nir_tex_instr_has_implicit_derivative.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_src_type = _libraries['libtinymesa_cpu.so'].nir_tex_instr_src_type
    nir_tex_instr_src_type.restype = nir_alu_type
    nir_tex_instr_src_type.argtypes = [ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_tex_instr_src_size = _libraries['libtinymesa_cpu.so'].nir_tex_instr_src_size
    nir_tex_instr_src_size.restype = ctypes.c_uint32
    nir_tex_instr_src_size.argtypes = [ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_tex_instr_src_index = _libraries['FIXME_STUB'].nir_tex_instr_src_index
    nir_tex_instr_src_index.restype = ctypes.c_int32
    nir_tex_instr_src_index.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_tex_instr_add_src = _libraries['libtinymesa_cpu.so'].nir_tex_instr_add_src
    nir_tex_instr_add_src.restype = None
    nir_tex_instr_add_src.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_tex_instr_remove_src = _libraries['libtinymesa_cpu.so'].nir_tex_instr_remove_src
    nir_tex_instr_remove_src.restype = None
    nir_tex_instr_remove_src.argtypes = [ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_get_tex_src = _libraries['FIXME_STUB'].nir_get_tex_src
    nir_get_tex_src.restype = ctypes.POINTER(struct_nir_def)
    nir_get_tex_src.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_get_tex_deref = _libraries['FIXME_STUB'].nir_get_tex_deref
    nir_get_tex_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_get_tex_deref.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_steal_tex_src = _libraries['FIXME_STUB'].nir_steal_tex_src
    nir_steal_tex_src.restype = ctypes.POINTER(struct_nir_def)
    nir_steal_tex_src.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_steal_tex_deref = _libraries['FIXME_STUB'].nir_steal_tex_deref
    nir_steal_tex_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_steal_tex_deref.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_tex_instr_has_explicit_tg4_offsets = _libraries['libtinymesa_cpu.so'].nir_tex_instr_has_explicit_tg4_offsets
    nir_tex_instr_has_explicit_tg4_offsets.restype = ctypes.c_bool
    nir_tex_instr_has_explicit_tg4_offsets.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
class struct_nir_load_const_instr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instr', nir_instr),
    ('def', nir_def),
    ('value', union_c__UA_nir_const_value * 0),
     ]

nir_load_const_instr = struct_nir_load_const_instr

# values for enumeration 'c__EA_nir_jump_type'
c__EA_nir_jump_type__enumvalues = {
    0: 'nir_jump_return',
    1: 'nir_jump_halt',
    2: 'nir_jump_break',
    3: 'nir_jump_continue',
    4: 'nir_jump_goto',
    5: 'nir_jump_goto_if',
}
nir_jump_return = 0
nir_jump_halt = 1
nir_jump_break = 2
nir_jump_continue = 3
nir_jump_goto = 4
nir_jump_goto_if = 5
c__EA_nir_jump_type = ctypes.c_uint32 # enum
nir_jump_type = c__EA_nir_jump_type
nir_jump_type__enumvalues = c__EA_nir_jump_type__enumvalues
class struct_nir_jump_instr(Structure):
    pass

struct_nir_jump_instr._pack_ = 1 # source:False
struct_nir_jump_instr._fields_ = [
    ('instr', nir_instr),
    ('type', nir_jump_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('condition', nir_src),
    ('target', ctypes.POINTER(struct_nir_block)),
    ('else_target', ctypes.POINTER(struct_nir_block)),
]

nir_jump_instr = struct_nir_jump_instr
class struct_nir_undef_instr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instr', nir_instr),
    ('def', nir_def),
     ]

nir_undef_instr = struct_nir_undef_instr
class struct_nir_phi_src(Structure):
    pass

struct_nir_phi_src._pack_ = 1 # source:False
struct_nir_phi_src._fields_ = [
    ('node', struct_exec_node),
    ('pred', ctypes.POINTER(struct_nir_block)),
    ('src', nir_src),
]

nir_phi_src = struct_nir_phi_src
class struct_nir_phi_instr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instr', nir_instr),
    ('srcs', struct_exec_list),
    ('def', nir_def),
     ]

nir_phi_instr = struct_nir_phi_instr
try:
    nir_phi_get_src_from_block = _libraries['FIXME_STUB'].nir_phi_get_src_from_block
    nir_phi_get_src_from_block.restype = ctypes.POINTER(struct_nir_phi_src)
    nir_phi_get_src_from_block.argtypes = [ctypes.POINTER(struct_nir_phi_instr), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
class struct_nir_parallel_copy_entry(Structure):
    pass

class union_nir_parallel_copy_entry_dest(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('def', nir_def),
    ('reg', nir_src),
     ]

struct_nir_parallel_copy_entry._pack_ = 1 # source:False
struct_nir_parallel_copy_entry._fields_ = [
    ('node', struct_exec_node),
    ('src_is_reg', ctypes.c_bool),
    ('dest_is_reg', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('src', nir_src),
    ('dest', union_nir_parallel_copy_entry_dest),
]

nir_parallel_copy_entry = struct_nir_parallel_copy_entry
class struct_nir_parallel_copy_instr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instr', nir_instr),
    ('entries', struct_exec_list),
     ]

nir_parallel_copy_instr = struct_nir_parallel_copy_instr
class struct_nir_instr_debug_info(Structure):
    pass

struct_nir_instr_debug_info._pack_ = 1 # source:False
struct_nir_instr_debug_info._fields_ = [
    ('filename', ctypes.POINTER(ctypes.c_char)),
    ('line', ctypes.c_uint32),
    ('column', ctypes.c_uint32),
    ('spirv_offset', ctypes.c_uint32),
    ('nir_line', ctypes.c_uint32),
    ('variable_name', ctypes.POINTER(ctypes.c_char)),
    ('instr', nir_instr),
]

nir_instr_debug_info = struct_nir_instr_debug_info
try:
    nir_instr_as_alu = _libraries['FIXME_STUB'].nir_instr_as_alu
    nir_instr_as_alu.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_instr_as_alu.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_deref = _libraries['FIXME_STUB'].nir_instr_as_deref
    nir_instr_as_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_instr_as_deref.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_call = _libraries['FIXME_STUB'].nir_instr_as_call
    nir_instr_as_call.restype = ctypes.POINTER(struct_nir_call_instr)
    nir_instr_as_call.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_jump = _libraries['FIXME_STUB'].nir_instr_as_jump
    nir_instr_as_jump.restype = ctypes.POINTER(struct_nir_jump_instr)
    nir_instr_as_jump.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_tex = _libraries['FIXME_STUB'].nir_instr_as_tex
    nir_instr_as_tex.restype = ctypes.POINTER(struct_nir_tex_instr)
    nir_instr_as_tex.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_intrinsic = _libraries['FIXME_STUB'].nir_instr_as_intrinsic
    nir_instr_as_intrinsic.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_instr_as_intrinsic.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_load_const = _libraries['FIXME_STUB'].nir_instr_as_load_const
    nir_instr_as_load_const.restype = ctypes.POINTER(struct_nir_load_const_instr)
    nir_instr_as_load_const.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_undef = _libraries['FIXME_STUB'].nir_instr_as_undef
    nir_instr_as_undef.restype = ctypes.POINTER(struct_nir_undef_instr)
    nir_instr_as_undef.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_phi = _libraries['FIXME_STUB'].nir_instr_as_phi
    nir_instr_as_phi.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_instr_as_phi.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_parallel_copy = _libraries['FIXME_STUB'].nir_instr_as_parallel_copy
    nir_instr_as_parallel_copy.restype = ctypes.POINTER(struct_nir_parallel_copy_instr)
    nir_instr_as_parallel_copy.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_src_comp_as_int = _libraries['FIXME_STUB'].nir_src_comp_as_int
    nir_src_comp_as_int.restype = int64_t
    nir_src_comp_as_int.argtypes = [nir_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_src_as_int = _libraries['FIXME_STUB'].nir_src_as_int
    nir_src_as_int.restype = int64_t
    nir_src_as_int.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_comp_as_uint = _libraries['FIXME_STUB'].nir_src_comp_as_uint
    nir_src_comp_as_uint.restype = uint64_t
    nir_src_comp_as_uint.argtypes = [nir_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_src_as_uint = _libraries['FIXME_STUB'].nir_src_as_uint
    nir_src_as_uint.restype = uint64_t
    nir_src_as_uint.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_comp_as_bool = _libraries['FIXME_STUB'].nir_src_comp_as_bool
    nir_src_comp_as_bool.restype = ctypes.c_bool
    nir_src_comp_as_bool.argtypes = [nir_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_src_as_bool = _libraries['FIXME_STUB'].nir_src_as_bool
    nir_src_as_bool.restype = ctypes.c_bool
    nir_src_as_bool.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_comp_as_float = _libraries['FIXME_STUB'].nir_src_comp_as_float
    nir_src_comp_as_float.restype = ctypes.c_double
    nir_src_comp_as_float.argtypes = [nir_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_src_as_float = _libraries['FIXME_STUB'].nir_src_as_float
    nir_src_as_float.restype = ctypes.c_double
    nir_src_as_float.argtypes = [nir_src]
except AttributeError:
    pass
class struct_nir_scalar(Structure):
    pass

struct_nir_scalar._pack_ = 1 # source:False
struct_nir_scalar._fields_ = [
    ('def', ctypes.POINTER(struct_nir_def)),
    ('comp', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

nir_scalar = struct_nir_scalar
try:
    nir_scalar_is_const = _libraries['FIXME_STUB'].nir_scalar_is_const
    nir_scalar_is_const.restype = ctypes.c_bool
    nir_scalar_is_const.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_is_undef = _libraries['FIXME_STUB'].nir_scalar_is_undef
    nir_scalar_is_undef.restype = ctypes.c_bool
    nir_scalar_is_undef.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_const_value = _libraries['FIXME_STUB'].nir_scalar_as_const_value
    nir_scalar_as_const_value.restype = nir_const_value
    nir_scalar_as_const_value.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_int = _libraries['FIXME_STUB'].nir_scalar_as_int
    nir_scalar_as_int.restype = int64_t
    nir_scalar_as_int.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_uint = _libraries['FIXME_STUB'].nir_scalar_as_uint
    nir_scalar_as_uint.restype = uint64_t
    nir_scalar_as_uint.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_bool = _libraries['FIXME_STUB'].nir_scalar_as_bool
    nir_scalar_as_bool.restype = ctypes.c_bool
    nir_scalar_as_bool.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_float = _libraries['FIXME_STUB'].nir_scalar_as_float
    nir_scalar_as_float.restype = ctypes.c_double
    nir_scalar_as_float.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_is_alu = _libraries['FIXME_STUB'].nir_scalar_is_alu
    nir_scalar_is_alu.restype = ctypes.c_bool
    nir_scalar_is_alu.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_alu_op = _libraries['FIXME_STUB'].nir_scalar_alu_op
    nir_scalar_alu_op.restype = nir_op
    nir_scalar_alu_op.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_is_intrinsic = _libraries['FIXME_STUB'].nir_scalar_is_intrinsic
    nir_scalar_is_intrinsic.restype = ctypes.c_bool
    nir_scalar_is_intrinsic.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_intrinsic_op = _libraries['FIXME_STUB'].nir_scalar_intrinsic_op
    nir_scalar_intrinsic_op.restype = nir_intrinsic_op
    nir_scalar_intrinsic_op.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_chase_alu_src = _libraries['FIXME_STUB'].nir_scalar_chase_alu_src
    nir_scalar_chase_alu_src.restype = nir_scalar
    nir_scalar_chase_alu_src.argtypes = [nir_scalar, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_scalar_chase_movs = _libraries['libtinymesa_cpu.so'].nir_scalar_chase_movs
    nir_scalar_chase_movs.restype = nir_scalar
    nir_scalar_chase_movs.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_get_scalar = _libraries['FIXME_STUB'].nir_get_scalar
    nir_get_scalar.restype = nir_scalar
    nir_get_scalar.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_scalar_resolved = _libraries['FIXME_STUB'].nir_scalar_resolved
    nir_scalar_resolved.restype = nir_scalar
    nir_scalar_resolved.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_scalar_equal = _libraries['FIXME_STUB'].nir_scalar_equal
    nir_scalar_equal.restype = ctypes.c_bool
    nir_scalar_equal.argtypes = [nir_scalar, nir_scalar]
except AttributeError:
    pass
try:
    nir_alu_src_as_uint = _libraries['FIXME_STUB'].nir_alu_src_as_uint
    nir_alu_src_as_uint.restype = uint64_t
    nir_alu_src_as_uint.argtypes = [nir_alu_src]
except AttributeError:
    pass
class struct_nir_binding(Structure):
    pass

struct_nir_binding._pack_ = 1 # source:False
struct_nir_binding._fields_ = [
    ('success', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('var', ctypes.POINTER(struct_nir_variable)),
    ('desc_set', ctypes.c_uint32),
    ('binding', ctypes.c_uint32),
    ('num_indices', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('indices', struct_nir_src * 4),
    ('read_first_invocation', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte * 7),
]

nir_binding = struct_nir_binding
try:
    nir_chase_binding = _libraries['libtinymesa_cpu.so'].nir_chase_binding
    nir_chase_binding.restype = nir_binding
    nir_chase_binding.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_get_binding_variable = _libraries['libtinymesa_cpu.so'].nir_get_binding_variable
    nir_get_binding_variable.restype = ctypes.POINTER(struct_nir_variable)
    nir_get_binding_variable.argtypes = [ctypes.POINTER(struct_nir_shader), nir_binding]
except AttributeError:
    pass
nir_cf_node_type = c__EA_nir_cf_node_type
nir_cf_node_type__enumvalues = c__EA_nir_cf_node_type__enumvalues
nir_cf_node = struct_nir_cf_node
nir_block = struct_nir_block
try:
    nir_block_is_reachable = _libraries['FIXME_STUB'].nir_block_is_reachable
    nir_block_is_reachable.restype = ctypes.c_bool
    nir_block_is_reachable.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_first_instr = _libraries['FIXME_STUB'].nir_block_first_instr
    nir_block_first_instr.restype = ctypes.POINTER(struct_nir_instr)
    nir_block_first_instr.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_last_instr = _libraries['FIXME_STUB'].nir_block_last_instr
    nir_block_last_instr.restype = ctypes.POINTER(struct_nir_instr)
    nir_block_last_instr.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_ends_in_jump = _libraries['FIXME_STUB'].nir_block_ends_in_jump
    nir_block_ends_in_jump.restype = ctypes.c_bool
    nir_block_ends_in_jump.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_ends_in_return_or_halt = _libraries['FIXME_STUB'].nir_block_ends_in_return_or_halt
    nir_block_ends_in_return_or_halt.restype = ctypes.c_bool
    nir_block_ends_in_return_or_halt.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_ends_in_break = _libraries['FIXME_STUB'].nir_block_ends_in_break
    nir_block_ends_in_break.restype = ctypes.c_bool
    nir_block_ends_in_break.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_contains_work = _libraries['libtinymesa_cpu.so'].nir_block_contains_work
    nir_block_contains_work.restype = ctypes.c_bool
    nir_block_contains_work.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_first_phi_in_block = _libraries['FIXME_STUB'].nir_first_phi_in_block
    nir_first_phi_in_block.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_first_phi_in_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_next_phi = _libraries['FIXME_STUB'].nir_next_phi
    nir_next_phi.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_next_phi.argtypes = [ctypes.POINTER(struct_nir_phi_instr)]
except AttributeError:
    pass
try:
    nir_block_last_phi_instr = _libraries['FIXME_STUB'].nir_block_last_phi_instr
    nir_block_last_phi_instr.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_block_last_phi_instr.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
nir_selection_control = c__EA_nir_selection_control
nir_selection_control__enumvalues = c__EA_nir_selection_control__enumvalues
nir_if = struct_nir_if
class struct_nir_loop_terminator(Structure):
    pass

struct_nir_loop_terminator._pack_ = 1 # source:False
struct_nir_loop_terminator._fields_ = [
    ('nif', ctypes.POINTER(struct_nir_if)),
    ('conditional_instr', ctypes.POINTER(struct_nir_instr)),
    ('break_block', ctypes.POINTER(struct_nir_block)),
    ('continue_from_block', ctypes.POINTER(struct_nir_block)),
    ('continue_from_then', ctypes.c_bool),
    ('induction_rhs', ctypes.c_bool),
    ('exact_trip_count_unknown', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 5),
    ('loop_terminator_link', struct_list_head),
]

nir_loop_terminator = struct_nir_loop_terminator
class struct_nir_loop_induction_variable(Structure):
    pass

struct_nir_loop_induction_variable._pack_ = 1 # source:False
struct_nir_loop_induction_variable._fields_ = [
    ('basis', ctypes.POINTER(struct_nir_def)),
    ('def', ctypes.POINTER(struct_nir_def)),
    ('init_src', ctypes.POINTER(struct_nir_src)),
    ('update_src', ctypes.POINTER(struct_nir_alu_src)),
]

nir_loop_induction_variable = struct_nir_loop_induction_variable
class struct_nir_loop_info(Structure):
    pass

class struct_hash_table(Structure):
    pass

struct_nir_loop_info._pack_ = 1 # source:False
struct_nir_loop_info._fields_ = [
    ('instr_cost', ctypes.c_uint32),
    ('has_soft_fp64', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('guessed_trip_count', ctypes.c_uint32),
    ('max_trip_count', ctypes.c_uint32),
    ('exact_trip_count_known', ctypes.c_bool),
    ('force_unroll', ctypes.c_bool),
    ('complex_loop', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 5),
    ('limiting_terminator', ctypes.POINTER(struct_nir_loop_terminator)),
    ('loop_terminator_list', struct_list_head),
    ('induction_vars', ctypes.POINTER(struct_hash_table)),
]

class struct_hash_entry(Structure):
    pass

struct_hash_table._pack_ = 1 # source:False
struct_hash_table._fields_ = [
    ('table', ctypes.POINTER(struct_hash_entry)),
    ('key_hash_function', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(None))),
    ('key_equals_function', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(None), ctypes.POINTER(None))),
    ('deleted_key', ctypes.POINTER(None)),
    ('size', ctypes.c_uint32),
    ('rehash', ctypes.c_uint32),
    ('size_magic', ctypes.c_uint64),
    ('rehash_magic', ctypes.c_uint64),
    ('max_entries', ctypes.c_uint32),
    ('size_index', ctypes.c_uint32),
    ('entries', ctypes.c_uint32),
    ('deleted_entries', ctypes.c_uint32),
]

struct_hash_entry._pack_ = 1 # source:False
struct_hash_entry._fields_ = [
    ('hash', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('key', ctypes.POINTER(None)),
    ('data', ctypes.POINTER(None)),
]

nir_loop_info = struct_nir_loop_info

# values for enumeration 'c__EA_nir_loop_control'
c__EA_nir_loop_control__enumvalues = {
    0: 'nir_loop_control_none',
    1: 'nir_loop_control_unroll',
    2: 'nir_loop_control_dont_unroll',
}
nir_loop_control_none = 0
nir_loop_control_unroll = 1
nir_loop_control_dont_unroll = 2
c__EA_nir_loop_control = ctypes.c_uint32 # enum
nir_loop_control = c__EA_nir_loop_control
nir_loop_control__enumvalues = c__EA_nir_loop_control__enumvalues
class struct_nir_loop(Structure):
    pass

struct_nir_loop._pack_ = 1 # source:False
struct_nir_loop._fields_ = [
    ('cf_node', nir_cf_node),
    ('body', struct_exec_list),
    ('continue_list', struct_exec_list),
    ('info', ctypes.POINTER(struct_nir_loop_info)),
    ('control', nir_loop_control),
    ('partially_unrolled', ctypes.c_bool),
    ('divergent_continue', ctypes.c_bool),
    ('divergent_break', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
]

nir_loop = struct_nir_loop
try:
    nir_loop_is_divergent = _libraries['FIXME_STUB'].nir_loop_is_divergent
    nir_loop_is_divergent.restype = ctypes.c_bool
    nir_loop_is_divergent.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
nir_metadata = c__EA_nir_metadata
nir_metadata__enumvalues = c__EA_nir_metadata__enumvalues
nir_function_impl = struct_nir_function_impl
try:
    nir_start_block = _libraries['FIXME_STUB'].nir_start_block
    nir_start_block.restype = ctypes.POINTER(struct_nir_block)
    nir_start_block.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_impl_last_block = _libraries['FIXME_STUB'].nir_impl_last_block
    nir_impl_last_block.restype = ctypes.POINTER(struct_nir_block)
    nir_impl_last_block.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_cf_node_next = _libraries['FIXME_STUB'].nir_cf_node_next
    nir_cf_node_next.restype = ctypes.POINTER(struct_nir_cf_node)
    nir_cf_node_next.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_prev = _libraries['FIXME_STUB'].nir_cf_node_prev
    nir_cf_node_prev.restype = ctypes.POINTER(struct_nir_cf_node)
    nir_cf_node_prev.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_is_first = _libraries['FIXME_STUB'].nir_cf_node_is_first
    nir_cf_node_is_first.restype = ctypes.c_bool
    nir_cf_node_is_first.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_is_last = _libraries['FIXME_STUB'].nir_cf_node_is_last
    nir_cf_node_is_last.restype = ctypes.c_bool
    nir_cf_node_is_last.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_as_block = _libraries['FIXME_STUB'].nir_cf_node_as_block
    nir_cf_node_as_block.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_as_block.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_as_if = _libraries['FIXME_STUB'].nir_cf_node_as_if
    nir_cf_node_as_if.restype = ctypes.POINTER(struct_nir_if)
    nir_cf_node_as_if.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_as_loop = _libraries['FIXME_STUB'].nir_cf_node_as_loop
    nir_cf_node_as_loop.restype = ctypes.POINTER(struct_nir_loop)
    nir_cf_node_as_loop.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_as_function = _libraries['FIXME_STUB'].nir_cf_node_as_function
    nir_cf_node_as_function.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_cf_node_as_function.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_if_first_then_block = _libraries['FIXME_STUB'].nir_if_first_then_block
    nir_if_first_then_block.restype = ctypes.POINTER(struct_nir_block)
    nir_if_first_then_block.argtypes = [ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_if_last_then_block = _libraries['FIXME_STUB'].nir_if_last_then_block
    nir_if_last_then_block.restype = ctypes.POINTER(struct_nir_block)
    nir_if_last_then_block.argtypes = [ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_if_first_else_block = _libraries['FIXME_STUB'].nir_if_first_else_block
    nir_if_first_else_block.restype = ctypes.POINTER(struct_nir_block)
    nir_if_first_else_block.argtypes = [ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_if_last_else_block = _libraries['FIXME_STUB'].nir_if_last_else_block
    nir_if_last_else_block.restype = ctypes.POINTER(struct_nir_block)
    nir_if_last_else_block.argtypes = [ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_loop_first_block = _libraries['FIXME_STUB'].nir_loop_first_block
    nir_loop_first_block.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_first_block.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_last_block = _libraries['FIXME_STUB'].nir_loop_last_block
    nir_loop_last_block.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_last_block.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_has_continue_construct = _libraries['FIXME_STUB'].nir_loop_has_continue_construct
    nir_loop_has_continue_construct.restype = ctypes.c_bool
    nir_loop_has_continue_construct.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_first_continue_block = _libraries['FIXME_STUB'].nir_loop_first_continue_block
    nir_loop_first_continue_block.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_first_continue_block.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_last_continue_block = _libraries['FIXME_STUB'].nir_loop_last_continue_block
    nir_loop_last_continue_block.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_last_continue_block.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_continue_target = _libraries['FIXME_STUB'].nir_loop_continue_target
    nir_loop_continue_target.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_continue_target.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_cf_list_is_empty_block = _libraries['FIXME_STUB'].nir_cf_list_is_empty_block
    nir_cf_list_is_empty_block.restype = ctypes.c_bool
    nir_cf_list_is_empty_block.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
nir_parameter = struct_nir_parameter
nir_function = struct_nir_function
nir_intrin_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
nir_vectorize_cb = ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
nir_shader = struct_nir_shader
try:
    nir_foreach_function_with_impl_first = _libraries['FIXME_STUB'].nir_foreach_function_with_impl_first
    nir_foreach_function_with_impl_first.restype = ctypes.POINTER(struct_nir_function)
    nir_foreach_function_with_impl_first.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_foreach_function_with_impl_next = _libraries['FIXME_STUB'].nir_foreach_function_with_impl_next
    nir_foreach_function_with_impl_next.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_foreach_function_with_impl_next.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_nir_function))]
except AttributeError:
    pass
try:
    nir_shader_get_entrypoint = _libraries['FIXME_STUB'].nir_shader_get_entrypoint
    nir_shader_get_entrypoint.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_shader_get_entrypoint.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_get_function_for_name = _libraries['FIXME_STUB'].nir_shader_get_function_for_name
    nir_shader_get_function_for_name.restype = ctypes.POINTER(struct_nir_function)
    nir_shader_get_function_for_name.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_remove_non_entrypoints = _libraries['libtinymesa_cpu.so'].nir_remove_non_entrypoints
    nir_remove_non_entrypoints.restype = None
    nir_remove_non_entrypoints.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_non_exported = _libraries['libtinymesa_cpu.so'].nir_remove_non_exported
    nir_remove_non_exported.restype = None
    nir_remove_non_exported.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_entrypoints = _libraries['libtinymesa_cpu.so'].nir_remove_entrypoints
    nir_remove_entrypoints.restype = None
    nir_remove_entrypoints.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_fixup_is_exported = _libraries['libtinymesa_cpu.so'].nir_fixup_is_exported
    nir_fixup_is_exported.restype = None
    nir_fixup_is_exported.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
gl_shader_stage = pipe_shader_type
gl_shader_stage__enumvalues = pipe_shader_type__enumvalues
try:
    nir_shader_create = _libraries['libtinymesa_cpu.so'].nir_shader_create
    nir_shader_create.restype = ctypes.POINTER(struct_nir_shader)
    nir_shader_create.argtypes = [ctypes.POINTER(None), gl_shader_stage, ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_shader_info)]
except AttributeError:
    pass
try:
    nir_shader_add_variable = _libraries['libtinymesa_cpu.so'].nir_shader_add_variable
    nir_shader_add_variable.restype = None
    nir_shader_add_variable.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_function_impl_add_variable = _libraries['FIXME_STUB'].nir_function_impl_add_variable
    nir_function_impl_add_variable.restype = None
    nir_function_impl_add_variable.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_create = _libraries['libtinymesa_cpu.so'].nir_variable_create
    nir_variable_create.restype = ctypes.POINTER(struct_nir_variable)
    nir_variable_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_local_variable_create = _libraries['libtinymesa_cpu.so'].nir_local_variable_create
    nir_local_variable_create.restype = ctypes.POINTER(struct_nir_variable)
    nir_local_variable_create.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_state_variable_create = _libraries['libtinymesa_cpu.so'].nir_state_variable_create
    nir_state_variable_create.restype = ctypes.POINTER(struct_nir_variable)
    nir_state_variable_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char), ctypes.c_int16 * 4]
except AttributeError:
    pass
try:
    nir_get_variable_with_location = _libraries['libtinymesa_cpu.so'].nir_get_variable_with_location
    nir_get_variable_with_location.restype = ctypes.POINTER(struct_nir_variable)
    nir_get_variable_with_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_int32, ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_create_variable_with_location = _libraries['libtinymesa_cpu.so'].nir_create_variable_with_location
    nir_create_variable_with_location.restype = ctypes.POINTER(struct_nir_variable)
    nir_create_variable_with_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_int32, ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_find_variable_with_location = _libraries['libtinymesa_cpu.so'].nir_find_variable_with_location
    nir_find_variable_with_location.restype = ctypes.POINTER(struct_nir_variable)
    nir_find_variable_with_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_find_variable_with_driver_location = _libraries['libtinymesa_cpu.so'].nir_find_variable_with_driver_location
    nir_find_variable_with_driver_location.restype = ctypes.POINTER(struct_nir_variable)
    nir_find_variable_with_driver_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_find_state_variable = _libraries['libtinymesa_cpu.so'].nir_find_state_variable
    nir_find_state_variable.restype = ctypes.POINTER(struct_nir_variable)
    nir_find_state_variable.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_int16 * 4]
except AttributeError:
    pass
try:
    nir_find_sampler_variable_with_tex_index = _libraries['libtinymesa_cpu.so'].nir_find_sampler_variable_with_tex_index
    nir_find_sampler_variable_with_tex_index.restype = ctypes.POINTER(struct_nir_variable)
    nir_find_sampler_variable_with_tex_index.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_sort_variables_with_modes = _libraries['libtinymesa_cpu.so'].nir_sort_variables_with_modes
    nir_sort_variables_with_modes.restype = None
    nir_sort_variables_with_modes.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_variable)), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_function_create = _libraries['libtinymesa_cpu.so'].nir_function_create
    nir_function_create.restype = ctypes.POINTER(struct_nir_function)
    nir_function_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_function_set_impl = _libraries['FIXME_STUB'].nir_function_set_impl
    nir_function_set_impl.restype = None
    nir_function_set_impl.argtypes = [ctypes.POINTER(struct_nir_function), ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_function_impl_create = _libraries['libtinymesa_cpu.so'].nir_function_impl_create
    nir_function_impl_create.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_function_impl_create.argtypes = [ctypes.POINTER(struct_nir_function)]
except AttributeError:
    pass
try:
    nir_function_impl_create_bare = _libraries['libtinymesa_cpu.so'].nir_function_impl_create_bare
    nir_function_impl_create_bare.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_function_impl_create_bare.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_block_create = _libraries['libtinymesa_cpu.so'].nir_block_create
    nir_block_create.restype = ctypes.POINTER(struct_nir_block)
    nir_block_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_if_create = _libraries['libtinymesa_cpu.so'].nir_if_create
    nir_if_create.restype = ctypes.POINTER(struct_nir_if)
    nir_if_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_loop_create = _libraries['libtinymesa_cpu.so'].nir_loop_create
    nir_loop_create.restype = ctypes.POINTER(struct_nir_loop)
    nir_loop_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_cf_node_get_function = _libraries['libtinymesa_cpu.so'].nir_cf_node_get_function
    nir_cf_node_get_function.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_cf_node_get_function.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_metadata_require = _libraries['libtinymesa_cpu.so'].nir_metadata_require
    nir_metadata_require.restype = None
    nir_metadata_require.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_metadata]
except AttributeError:
    pass
try:
    nir_shader_preserve_all_metadata = _libraries['libtinymesa_cpu.so'].nir_shader_preserve_all_metadata
    nir_shader_preserve_all_metadata.restype = None
    nir_shader_preserve_all_metadata.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_metadata_invalidate = _libraries['libtinymesa_cpu.so'].nir_metadata_invalidate
    nir_metadata_invalidate.restype = None
    nir_metadata_invalidate.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_progress = _libraries['libtinymesa_cpu.so'].nir_progress
    nir_progress.restype = ctypes.c_bool
    nir_progress.argtypes = [ctypes.c_bool, ctypes.POINTER(struct_nir_function_impl), nir_metadata]
except AttributeError:
    pass
try:
    nir_no_progress = _libraries['FIXME_STUB'].nir_no_progress
    nir_no_progress.restype = ctypes.c_bool
    nir_no_progress.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_alu_instr_create = _libraries['libtinymesa_cpu.so'].nir_alu_instr_create
    nir_alu_instr_create.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_alu_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_op]
except AttributeError:
    pass
try:
    nir_deref_instr_create = _libraries['libtinymesa_cpu.so'].nir_deref_instr_create
    nir_deref_instr_create.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_deref_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_deref_type]
except AttributeError:
    pass
try:
    nir_jump_instr_create = _libraries['libtinymesa_cpu.so'].nir_jump_instr_create
    nir_jump_instr_create.restype = ctypes.POINTER(struct_nir_jump_instr)
    nir_jump_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_jump_type]
except AttributeError:
    pass
try:
    nir_load_const_instr_create = _libraries['libtinymesa_cpu.so'].nir_load_const_instr_create
    nir_load_const_instr_create.restype = ctypes.POINTER(struct_nir_load_const_instr)
    nir_load_const_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_intrinsic_instr_create = _libraries['libtinymesa_cpu.so'].nir_intrinsic_instr_create
    nir_intrinsic_instr_create.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_intrinsic_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrinsic_op]
except AttributeError:
    pass
try:
    nir_call_instr_create = _libraries['libtinymesa_cpu.so'].nir_call_instr_create
    nir_call_instr_create.restype = ctypes.POINTER(struct_nir_call_instr)
    nir_call_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function)]
except AttributeError:
    pass
try:
    nir_tex_instr_create = _libraries['libtinymesa_cpu.so'].nir_tex_instr_create
    nir_tex_instr_create.restype = ctypes.POINTER(struct_nir_tex_instr)
    nir_tex_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_phi_instr_create = _libraries['libtinymesa_cpu.so'].nir_phi_instr_create
    nir_phi_instr_create.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_phi_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_phi_instr_add_src = _libraries['libtinymesa_cpu.so'].nir_phi_instr_add_src
    nir_phi_instr_add_src.restype = ctypes.POINTER(struct_nir_phi_src)
    nir_phi_instr_add_src.argtypes = [ctypes.POINTER(struct_nir_phi_instr), ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_parallel_copy_instr_create = _libraries['libtinymesa_cpu.so'].nir_parallel_copy_instr_create
    nir_parallel_copy_instr_create.restype = ctypes.POINTER(struct_nir_parallel_copy_instr)
    nir_parallel_copy_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_undef_instr_create = _libraries['libtinymesa_cpu.so'].nir_undef_instr_create
    nir_undef_instr_create.restype = ctypes.POINTER(struct_nir_undef_instr)
    nir_undef_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_binop_identity = _libraries['libtinymesa_cpu.so'].nir_alu_binop_identity
    nir_alu_binop_identity.restype = nir_const_value
    nir_alu_binop_identity.argtypes = [nir_op, ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_cursor_option'
c__EA_nir_cursor_option__enumvalues = {
    0: 'nir_cursor_before_block',
    1: 'nir_cursor_after_block',
    2: 'nir_cursor_before_instr',
    3: 'nir_cursor_after_instr',
}
nir_cursor_before_block = 0
nir_cursor_after_block = 1
nir_cursor_before_instr = 2
nir_cursor_after_instr = 3
c__EA_nir_cursor_option = ctypes.c_uint32 # enum
nir_cursor_option = c__EA_nir_cursor_option
nir_cursor_option__enumvalues = c__EA_nir_cursor_option__enumvalues
class struct_nir_cursor(Structure):
    pass

class union_nir_cursor_0(Union):
    pass

union_nir_cursor_0._pack_ = 1 # source:False
union_nir_cursor_0._fields_ = [
    ('block', ctypes.POINTER(struct_nir_block)),
    ('instr', ctypes.POINTER(struct_nir_instr)),
]

struct_nir_cursor._pack_ = 1 # source:False
struct_nir_cursor._anonymous_ = ('_0',)
struct_nir_cursor._fields_ = [
    ('option', nir_cursor_option),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_0', union_nir_cursor_0),
]

nir_cursor = struct_nir_cursor
try:
    nir_cursor_current_block = _libraries['FIXME_STUB'].nir_cursor_current_block
    nir_cursor_current_block.restype = ctypes.POINTER(struct_nir_block)
    nir_cursor_current_block.argtypes = [nir_cursor]
except AttributeError:
    pass
try:
    nir_cursors_equal = _libraries['libtinymesa_cpu.so'].nir_cursors_equal
    nir_cursors_equal.restype = ctypes.c_bool
    nir_cursors_equal.argtypes = [nir_cursor, nir_cursor]
except AttributeError:
    pass
try:
    nir_before_block = _libraries['FIXME_STUB'].nir_before_block
    nir_before_block.restype = nir_cursor
    nir_before_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_after_block = _libraries['FIXME_STUB'].nir_after_block
    nir_after_block.restype = nir_cursor
    nir_after_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_before_instr = _libraries['FIXME_STUB'].nir_before_instr
    nir_before_instr.restype = nir_cursor
    nir_before_instr.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_after_instr = _libraries['FIXME_STUB'].nir_after_instr
    nir_after_instr.restype = nir_cursor
    nir_after_instr.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_before_block_after_phis = _libraries['FIXME_STUB'].nir_before_block_after_phis
    nir_before_block_after_phis.restype = nir_cursor
    nir_before_block_after_phis.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_after_block_before_jump = _libraries['FIXME_STUB'].nir_after_block_before_jump
    nir_after_block_before_jump.restype = nir_cursor
    nir_after_block_before_jump.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_before_src = _libraries['FIXME_STUB'].nir_before_src
    nir_before_src.restype = nir_cursor
    nir_before_src.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_before_cf_node = _libraries['FIXME_STUB'].nir_before_cf_node
    nir_before_cf_node.restype = nir_cursor
    nir_before_cf_node.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_after_cf_node = _libraries['FIXME_STUB'].nir_after_cf_node
    nir_after_cf_node.restype = nir_cursor
    nir_after_cf_node.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_after_phis = _libraries['FIXME_STUB'].nir_after_phis
    nir_after_phis.restype = nir_cursor
    nir_after_phis.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_after_instr_and_phis = _libraries['FIXME_STUB'].nir_after_instr_and_phis
    nir_after_instr_and_phis.restype = nir_cursor
    nir_after_instr_and_phis.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_after_cf_node_and_phis = _libraries['FIXME_STUB'].nir_after_cf_node_and_phis
    nir_after_cf_node_and_phis.restype = nir_cursor
    nir_after_cf_node_and_phis.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_before_cf_list = _libraries['FIXME_STUB'].nir_before_cf_list
    nir_before_cf_list.restype = nir_cursor
    nir_before_cf_list.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    nir_after_cf_list = _libraries['FIXME_STUB'].nir_after_cf_list
    nir_after_cf_list.restype = nir_cursor
    nir_after_cf_list.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    nir_before_impl = _libraries['FIXME_STUB'].nir_before_impl
    nir_before_impl.restype = nir_cursor
    nir_before_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_after_impl = _libraries['FIXME_STUB'].nir_after_impl
    nir_after_impl.restype = nir_cursor
    nir_after_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_instr_insert = _libraries['libtinymesa_cpu.so'].nir_instr_insert
    nir_instr_insert.restype = None
    nir_instr_insert.argtypes = [nir_cursor, ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_move = _libraries['libtinymesa_cpu.so'].nir_instr_move
    nir_instr_move.restype = ctypes.c_bool
    nir_instr_move.argtypes = [nir_cursor, ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_before = _libraries['FIXME_STUB'].nir_instr_insert_before
    nir_instr_insert_before.restype = None
    nir_instr_insert_before.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_after = _libraries['FIXME_STUB'].nir_instr_insert_after
    nir_instr_insert_after.restype = None
    nir_instr_insert_after.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_before_block = _libraries['FIXME_STUB'].nir_instr_insert_before_block
    nir_instr_insert_before_block.restype = None
    nir_instr_insert_before_block.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_after_block = _libraries['FIXME_STUB'].nir_instr_insert_after_block
    nir_instr_insert_after_block.restype = None
    nir_instr_insert_after_block.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_before_cf = _libraries['FIXME_STUB'].nir_instr_insert_before_cf
    nir_instr_insert_before_cf.restype = None
    nir_instr_insert_before_cf.argtypes = [ctypes.POINTER(struct_nir_cf_node), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_after_cf = _libraries['FIXME_STUB'].nir_instr_insert_after_cf
    nir_instr_insert_after_cf.restype = None
    nir_instr_insert_after_cf.argtypes = [ctypes.POINTER(struct_nir_cf_node), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_before_cf_list = _libraries['FIXME_STUB'].nir_instr_insert_before_cf_list
    nir_instr_insert_before_cf_list.restype = None
    nir_instr_insert_before_cf_list.argtypes = [ctypes.POINTER(struct_exec_list), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_after_cf_list = _libraries['FIXME_STUB'].nir_instr_insert_after_cf_list
    nir_instr_insert_after_cf_list.restype = None
    nir_instr_insert_after_cf_list.argtypes = [ctypes.POINTER(struct_exec_list), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_remove_v = _libraries['libtinymesa_cpu.so'].nir_instr_remove_v
    nir_instr_remove_v.restype = None
    nir_instr_remove_v.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_free = _libraries['libtinymesa_cpu.so'].nir_instr_free
    nir_instr_free.restype = None
    nir_instr_free.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_free_list = _libraries['libtinymesa_cpu.so'].nir_instr_free_list
    nir_instr_free_list.restype = None
    nir_instr_free_list.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    nir_instr_remove = _libraries['FIXME_STUB'].nir_instr_remove
    nir_instr_remove.restype = nir_cursor
    nir_instr_remove.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_free_and_dce = _libraries['libtinymesa_cpu.so'].nir_instr_free_and_dce
    nir_instr_free_and_dce.restype = nir_cursor
    nir_instr_free_and_dce.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_def = _libraries['libtinymesa_cpu.so'].nir_instr_def
    nir_instr_def.restype = ctypes.POINTER(struct_nir_def)
    nir_instr_def.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_get_debug_info = _libraries['FIXME_STUB'].nir_instr_get_debug_info
    nir_instr_get_debug_info.restype = ctypes.POINTER(struct_nir_instr_debug_info)
    nir_instr_get_debug_info.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_get_gc_pointer = _libraries['FIXME_STUB'].nir_instr_get_gc_pointer
    nir_instr_get_gc_pointer.restype = ctypes.POINTER(None)
    nir_instr_get_gc_pointer.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
nir_foreach_def_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_def), ctypes.POINTER(None))
nir_foreach_src_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_src), ctypes.POINTER(None))
try:
    nir_foreach_src = _libraries['FIXME_STUB'].nir_foreach_src
    nir_foreach_src.restype = ctypes.c_bool
    nir_foreach_src.argtypes = [ctypes.POINTER(struct_nir_instr), nir_foreach_src_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_foreach_phi_src_leaving_block = _libraries['libtinymesa_cpu.so'].nir_foreach_phi_src_leaving_block
    nir_foreach_phi_src_leaving_block.restype = ctypes.c_bool
    nir_foreach_phi_src_leaving_block.argtypes = [ctypes.POINTER(struct_nir_block), nir_foreach_src_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_src_as_const_value = _libraries['libtinymesa_cpu.so'].nir_src_as_const_value
    nir_src_as_const_value.restype = ctypes.POINTER(union_c__UA_nir_const_value)
    nir_src_as_const_value.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_as_alu_instr = _libraries['FIXME_STUB'].nir_src_as_alu_instr
    nir_src_as_alu_instr.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_src_as_alu_instr.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_as_intrinsic = _libraries['FIXME_STUB'].nir_src_as_intrinsic
    nir_src_as_intrinsic.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_src_as_intrinsic.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_as_string = _libraries['FIXME_STUB'].nir_src_as_string
    nir_src_as_string.restype = ctypes.POINTER(ctypes.c_char)
    nir_src_as_string.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_is_always_uniform = _libraries['libtinymesa_cpu.so'].nir_src_is_always_uniform
    nir_src_is_always_uniform.restype = ctypes.c_bool
    nir_src_is_always_uniform.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_srcs_equal = _libraries['libtinymesa_cpu.so'].nir_srcs_equal
    nir_srcs_equal.restype = ctypes.c_bool
    nir_srcs_equal.argtypes = [nir_src, nir_src]
except AttributeError:
    pass
try:
    nir_instrs_equal = _libraries['libtinymesa_cpu.so'].nir_instrs_equal
    nir_instrs_equal.restype = ctypes.c_bool
    nir_instrs_equal.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_src_get_block = _libraries['libtinymesa_cpu.so'].nir_src_get_block
    nir_src_get_block.restype = ctypes.POINTER(struct_nir_block)
    nir_src_get_block.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_src_rewrite = _libraries['FIXME_STUB'].nir_src_rewrite
    nir_src_rewrite.restype = None
    nir_src_rewrite.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_instr_init_src = _libraries['libtinymesa_cpu.so'].nir_instr_init_src
    nir_instr_init_src.restype = None
    nir_instr_init_src.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_instr_clear_src = _libraries['libtinymesa_cpu.so'].nir_instr_clear_src
    nir_instr_clear_src.restype = None
    nir_instr_clear_src.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_instr_move_src = _libraries['libtinymesa_cpu.so'].nir_instr_move_src
    nir_instr_move_src.restype = None
    nir_instr_move_src.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_instr_is_before = _libraries['libtinymesa_cpu.so'].nir_instr_is_before
    nir_instr_is_before.restype = ctypes.c_bool
    nir_instr_is_before.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_def_init = _libraries['libtinymesa_cpu.so'].nir_def_init
    nir_def_init.restype = None
    nir_def_init.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_def_init_for_type = _libraries['FIXME_STUB'].nir_def_init_for_type
    nir_def_init_for_type.restype = None
    nir_def_init_for_type.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_def_rewrite_uses = _libraries['libtinymesa_cpu.so'].nir_def_rewrite_uses
    nir_def_rewrite_uses.restype = None
    nir_def_rewrite_uses.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_rewrite_uses_src = _libraries['libtinymesa_cpu.so'].nir_def_rewrite_uses_src
    nir_def_rewrite_uses_src.restype = None
    nir_def_rewrite_uses_src.argtypes = [ctypes.POINTER(struct_nir_def), nir_src]
except AttributeError:
    pass
try:
    nir_def_rewrite_uses_after = _libraries['libtinymesa_cpu.so'].nir_def_rewrite_uses_after
    nir_def_rewrite_uses_after.restype = None
    nir_def_rewrite_uses_after.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_def_replace = _libraries['FIXME_STUB'].nir_def_replace
    nir_def_replace.restype = None
    nir_def_replace.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_src_components_read = _libraries['libtinymesa_cpu.so'].nir_src_components_read
    nir_src_components_read.restype = nir_component_mask_t
    nir_src_components_read.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_def_components_read = _libraries['libtinymesa_cpu.so'].nir_def_components_read
    nir_def_components_read.restype = nir_component_mask_t
    nir_def_components_read.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_all_uses_are_fsat = _libraries['libtinymesa_cpu.so'].nir_def_all_uses_are_fsat
    nir_def_all_uses_are_fsat.restype = ctypes.c_bool
    nir_def_all_uses_are_fsat.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_all_uses_ignore_sign_bit = _libraries['libtinymesa_cpu.so'].nir_def_all_uses_ignore_sign_bit
    nir_def_all_uses_ignore_sign_bit.restype = ctypes.c_bool
    nir_def_all_uses_ignore_sign_bit.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_first_component_read = _libraries['FIXME_STUB'].nir_def_first_component_read
    nir_def_first_component_read.restype = ctypes.c_int32
    nir_def_first_component_read.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_last_component_read = _libraries['FIXME_STUB'].nir_def_last_component_read
    nir_def_last_component_read.restype = ctypes.c_int32
    nir_def_last_component_read.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_is_unused = _libraries['FIXME_STUB'].nir_def_is_unused
    nir_def_is_unused.restype = ctypes.c_bool
    nir_def_is_unused.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_sort_unstructured_blocks = _libraries['libtinymesa_cpu.so'].nir_sort_unstructured_blocks
    nir_sort_unstructured_blocks.restype = None
    nir_sort_unstructured_blocks.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_block_unstructured_next = _libraries['libtinymesa_cpu.so'].nir_block_unstructured_next
    nir_block_unstructured_next.restype = ctypes.POINTER(struct_nir_block)
    nir_block_unstructured_next.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_unstructured_start_block = _libraries['libtinymesa_cpu.so'].nir_unstructured_start_block
    nir_unstructured_start_block.restype = ctypes.POINTER(struct_nir_block)
    nir_unstructured_start_block.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_block_cf_tree_next = _libraries['libtinymesa_cpu.so'].nir_block_cf_tree_next
    nir_block_cf_tree_next.restype = ctypes.POINTER(struct_nir_block)
    nir_block_cf_tree_next.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_cf_tree_prev = _libraries['libtinymesa_cpu.so'].nir_block_cf_tree_prev
    nir_block_cf_tree_prev.restype = ctypes.POINTER(struct_nir_block)
    nir_block_cf_tree_prev.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_cf_node_cf_tree_first = _libraries['libtinymesa_cpu.so'].nir_cf_node_cf_tree_first
    nir_cf_node_cf_tree_first.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_cf_tree_first.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_cf_tree_last = _libraries['libtinymesa_cpu.so'].nir_cf_node_cf_tree_last
    nir_cf_node_cf_tree_last.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_cf_tree_last.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_cf_tree_next = _libraries['libtinymesa_cpu.so'].nir_cf_node_cf_tree_next
    nir_cf_node_cf_tree_next.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_cf_tree_next.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_cf_tree_prev = _libraries['libtinymesa_cpu.so'].nir_cf_node_cf_tree_prev
    nir_cf_node_cf_tree_prev.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_cf_tree_prev.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_block_get_following_if = _libraries['libtinymesa_cpu.so'].nir_block_get_following_if
    nir_block_get_following_if.restype = ctypes.POINTER(struct_nir_if)
    nir_block_get_following_if.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_get_following_loop = _libraries['libtinymesa_cpu.so'].nir_block_get_following_loop
    nir_block_get_following_loop.restype = ctypes.POINTER(struct_nir_loop)
    nir_block_get_following_loop.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_get_predecessors_sorted = _libraries['libtinymesa_cpu.so'].nir_block_get_predecessors_sorted
    nir_block_get_predecessors_sorted.restype = ctypes.POINTER(ctypes.POINTER(struct_nir_block))
    nir_block_get_predecessors_sorted.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_index_ssa_defs = _libraries['libtinymesa_cpu.so'].nir_index_ssa_defs
    nir_index_ssa_defs.restype = None
    nir_index_ssa_defs.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_index_instrs = _libraries['libtinymesa_cpu.so'].nir_index_instrs
    nir_index_instrs.restype = ctypes.c_uint32
    nir_index_instrs.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_index_blocks = _libraries['libtinymesa_cpu.so'].nir_index_blocks
    nir_index_blocks.restype = None
    nir_index_blocks.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_shader_clear_pass_flags = _libraries['libtinymesa_cpu.so'].nir_shader_clear_pass_flags
    nir_shader_clear_pass_flags.restype = None
    nir_shader_clear_pass_flags.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_index_vars = _libraries['libtinymesa_cpu.so'].nir_shader_index_vars
    nir_shader_index_vars.restype = ctypes.c_uint32
    nir_shader_index_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_function_impl_index_vars = _libraries['libtinymesa_cpu.so'].nir_function_impl_index_vars
    nir_function_impl_index_vars.restype = ctypes.c_uint32
    nir_function_impl_index_vars.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_print_shader = _libraries['libtinymesa_cpu.so'].nir_print_shader
    nir_print_shader.restype = None
    nir_print_shader.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_print_function_body = _libraries['libtinymesa_cpu.so'].nir_print_function_body
    nir_print_function_body.restype = None
    nir_print_function_body.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_print_shader_annotated = _libraries['libtinymesa_cpu.so'].nir_print_shader_annotated
    nir_print_shader_annotated.restype = None
    nir_print_shader_annotated.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_print_instr = _libraries['libtinymesa_cpu.so'].nir_print_instr
    nir_print_instr.restype = None
    nir_print_instr.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_print_deref = _libraries['libtinymesa_cpu.so'].nir_print_deref
    nir_print_deref.restype = None
    nir_print_deref.argtypes = [ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass

# values for enumeration 'mesa_log_level'
mesa_log_level__enumvalues = {
    0: 'MESA_LOG_ERROR',
    1: 'MESA_LOG_WARN',
    2: 'MESA_LOG_INFO',
    3: 'MESA_LOG_DEBUG',
}
MESA_LOG_ERROR = 0
MESA_LOG_WARN = 1
MESA_LOG_INFO = 2
MESA_LOG_DEBUG = 3
mesa_log_level = ctypes.c_uint32 # enum
try:
    nir_log_shader_annotated_tagged = _libraries['libtinymesa_cpu.so'].nir_log_shader_annotated_tagged
    nir_log_shader_annotated_tagged.restype = None
    nir_log_shader_annotated_tagged.argtypes = [mesa_log_level, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_shader_as_str = _libraries['libtinymesa_cpu.so'].nir_shader_as_str
    nir_shader_as_str.restype = ctypes.POINTER(ctypes.c_char)
    nir_shader_as_str.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_as_str_annotated = _libraries['libtinymesa_cpu.so'].nir_shader_as_str_annotated
    nir_shader_as_str_annotated.restype = ctypes.POINTER(ctypes.c_char)
    nir_shader_as_str_annotated.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_hash_table), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_instr_as_str = _libraries['libtinymesa_cpu.so'].nir_instr_as_str
    nir_instr_as_str.restype = ctypes.POINTER(ctypes.c_char)
    nir_instr_as_str.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_gather_debug_info = _libraries['libtinymesa_cpu.so'].nir_shader_gather_debug_info
    nir_shader_gather_debug_info.restype = ctypes.POINTER(ctypes.c_char)
    nir_shader_gather_debug_info.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char), uint32_t]
except AttributeError:
    pass
try:
    nir_instr_clone = _libraries['libtinymesa_cpu.so'].nir_instr_clone
    nir_instr_clone.restype = ctypes.POINTER(struct_nir_instr)
    nir_instr_clone.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_clone_deep = _libraries['libtinymesa_cpu.so'].nir_instr_clone_deep
    nir_instr_clone_deep.restype = ctypes.POINTER(struct_nir_instr)
    nir_instr_clone_deep.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_alu_instr_clone = _libraries['libtinymesa_cpu.so'].nir_alu_instr_clone
    nir_alu_instr_clone.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_alu_instr_clone.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_shader_clone = _libraries['libtinymesa_cpu.so'].nir_shader_clone
    nir_shader_clone.restype = ctypes.POINTER(struct_nir_shader)
    nir_shader_clone.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_function_clone = _libraries['libtinymesa_cpu.so'].nir_function_clone
    nir_function_clone.restype = ctypes.POINTER(struct_nir_function)
    nir_function_clone.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function)]
except AttributeError:
    pass
try:
    nir_function_impl_clone = _libraries['libtinymesa_cpu.so'].nir_function_impl_clone
    nir_function_impl_clone.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_function_impl_clone.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_function_impl_clone_remap_globals = _libraries['libtinymesa_cpu.so'].nir_function_impl_clone_remap_globals
    nir_function_impl_clone_remap_globals.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_function_impl_clone_remap_globals.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_constant_clone = _libraries['libtinymesa_cpu.so'].nir_constant_clone
    nir_constant_clone.restype = ctypes.POINTER(struct_nir_constant)
    nir_constant_clone.argtypes = [ctypes.POINTER(struct_nir_constant), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_clone = _libraries['libtinymesa_cpu.so'].nir_variable_clone
    nir_variable_clone.restype = ctypes.POINTER(struct_nir_variable)
    nir_variable_clone.argtypes = [ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_replace = _libraries['libtinymesa_cpu.so'].nir_shader_replace
    nir_shader_replace.restype = None
    nir_shader_replace.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_serialize_deserialize = _libraries['libtinymesa_cpu.so'].nir_shader_serialize_deserialize
    nir_shader_serialize_deserialize.restype = None
    nir_shader_serialize_deserialize.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_validate_shader = _libraries['libtinymesa_cpu.so'].nir_validate_shader
    nir_validate_shader.restype = None
    nir_validate_shader.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_validate_ssa_dominance = _libraries['libtinymesa_cpu.so'].nir_validate_ssa_dominance
    nir_validate_ssa_dominance.restype = None
    nir_validate_ssa_dominance.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_metadata_set_validation_flag = _libraries['libtinymesa_cpu.so'].nir_metadata_set_validation_flag
    nir_metadata_set_validation_flag.restype = None
    nir_metadata_set_validation_flag.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_metadata_check_validation_flag = _libraries['libtinymesa_cpu.so'].nir_metadata_check_validation_flag
    nir_metadata_check_validation_flag.restype = None
    nir_metadata_check_validation_flag.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_metadata_require_all = _libraries['libtinymesa_cpu.so'].nir_metadata_require_all
    nir_metadata_require_all.restype = None
    nir_metadata_require_all.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    should_skip_nir = _libraries['FIXME_STUB'].should_skip_nir
    should_skip_nir.restype = ctypes.c_bool
    should_skip_nir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    should_print_nir = _libraries['FIXME_STUB'].should_print_nir
    should_print_nir.restype = ctypes.c_bool
    should_print_nir.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
nir_instr_writemask_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.c_uint32, ctypes.POINTER(None))
class struct_nir_builder(Structure):
    pass

struct_nir_builder._fields_ = [
    ('cursor', nir_cursor),
    ('exact', ctypes.c_bool),
    ('fp_fast_math', ctypes.c_uint32),
    ('shader', ctypes.POINTER(struct_nir_shader)),
    ('impl', ctypes.POINTER(struct_nir_function_impl)),
]

nir_lower_instr_cb = ctypes.CFUNCTYPE(ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
try:
    nir_function_impl_lower_instructions = _libraries['libtinymesa_cpu.so'].nir_function_impl_lower_instructions
    nir_function_impl_lower_instructions.restype = ctypes.c_bool
    nir_function_impl_lower_instructions.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_instr_filter_cb, nir_lower_instr_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_lower_instructions = _libraries['libtinymesa_cpu.so'].nir_shader_lower_instructions
    nir_shader_lower_instructions.restype = ctypes.c_bool
    nir_shader_lower_instructions.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_filter_cb, nir_lower_instr_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_calc_dominance_impl = _libraries['libtinymesa_cpu.so'].nir_calc_dominance_impl
    nir_calc_dominance_impl.restype = None
    nir_calc_dominance_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_calc_dominance = _libraries['libtinymesa_cpu.so'].nir_calc_dominance
    nir_calc_dominance.restype = None
    nir_calc_dominance.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_dominance_lca = _libraries['libtinymesa_cpu.so'].nir_dominance_lca
    nir_dominance_lca.restype = ctypes.POINTER(struct_nir_block)
    nir_dominance_lca.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_dominates = _libraries['libtinymesa_cpu.so'].nir_block_dominates
    nir_block_dominates.restype = ctypes.c_bool
    nir_block_dominates.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_is_unreachable = _libraries['libtinymesa_cpu.so'].nir_block_is_unreachable
    nir_block_is_unreachable.restype = ctypes.c_bool
    nir_block_is_unreachable.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_dump_dom_tree_impl = _libraries['libtinymesa_cpu.so'].nir_dump_dom_tree_impl
    nir_dump_dom_tree_impl.restype = None
    nir_dump_dom_tree_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_dom_tree = _libraries['libtinymesa_cpu.so'].nir_dump_dom_tree
    nir_dump_dom_tree.restype = None
    nir_dump_dom_tree.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_dom_frontier_impl = _libraries['libtinymesa_cpu.so'].nir_dump_dom_frontier_impl
    nir_dump_dom_frontier_impl.restype = None
    nir_dump_dom_frontier_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_dom_frontier = _libraries['libtinymesa_cpu.so'].nir_dump_dom_frontier
    nir_dump_dom_frontier.restype = None
    nir_dump_dom_frontier.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_cfg_impl = _libraries['libtinymesa_cpu.so'].nir_dump_cfg_impl
    nir_dump_cfg_impl.restype = None
    nir_dump_cfg_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_cfg = _libraries['libtinymesa_cpu.so'].nir_dump_cfg
    nir_dump_cfg.restype = None
    nir_dump_cfg.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_gs_count_vertices_and_primitives = _libraries['libtinymesa_cpu.so'].nir_gs_count_vertices_and_primitives
    nir_gs_count_vertices_and_primitives.restype = None
    nir_gs_count_vertices_and_primitives.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_load_grouping'
c__EA_nir_load_grouping__enumvalues = {
    0: 'nir_group_all',
    1: 'nir_group_same_resource_only',
}
nir_group_all = 0
nir_group_same_resource_only = 1
c__EA_nir_load_grouping = ctypes.c_uint32 # enum
nir_load_grouping = c__EA_nir_load_grouping
nir_load_grouping__enumvalues = c__EA_nir_load_grouping__enumvalues
try:
    nir_group_loads = _libraries['libtinymesa_cpu.so'].nir_group_loads
    nir_group_loads.restype = ctypes.c_bool
    nir_group_loads.argtypes = [ctypes.POINTER(struct_nir_shader), nir_load_grouping, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_shrink_vec_array_vars = _libraries['libtinymesa_cpu.so'].nir_shrink_vec_array_vars
    nir_shrink_vec_array_vars.restype = ctypes.c_bool
    nir_shrink_vec_array_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_split_array_vars = _libraries['libtinymesa_cpu.so'].nir_split_array_vars
    nir_split_array_vars.restype = ctypes.c_bool
    nir_split_array_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_split_var_copies = _libraries['libtinymesa_cpu.so'].nir_split_var_copies
    nir_split_var_copies.restype = ctypes.c_bool
    nir_split_var_copies.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_split_per_member_structs = _libraries['libtinymesa_cpu.so'].nir_split_per_member_structs
    nir_split_per_member_structs.restype = ctypes.c_bool
    nir_split_per_member_structs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_split_struct_vars = _libraries['libtinymesa_cpu.so'].nir_split_struct_vars
    nir_split_struct_vars.restype = ctypes.c_bool
    nir_split_struct_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_returns_impl = _libraries['libtinymesa_cpu.so'].nir_lower_returns_impl
    nir_lower_returns_impl.restype = ctypes.c_bool
    nir_lower_returns_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_lower_returns = _libraries['libtinymesa_cpu.so'].nir_lower_returns
    nir_lower_returns.restype = ctypes.c_bool
    nir_lower_returns.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_inline_function_impl = _libraries['libtinymesa_cpu.so'].nir_inline_function_impl
    nir_inline_function_impl.restype = ctypes.POINTER(struct_nir_def)
    nir_inline_function_impl.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_inline_functions = _libraries['libtinymesa_cpu.so'].nir_inline_functions
    nir_inline_functions.restype = ctypes.c_bool
    nir_inline_functions.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_cleanup_functions = _libraries['libtinymesa_cpu.so'].nir_cleanup_functions
    nir_cleanup_functions.restype = None
    nir_cleanup_functions.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_link_shader_functions = _libraries['libtinymesa_cpu.so'].nir_link_shader_functions
    nir_link_shader_functions.restype = ctypes.c_bool
    nir_link_shader_functions.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_calls_to_builtins = _libraries['libtinymesa_cpu.so'].nir_lower_calls_to_builtins
    nir_lower_calls_to_builtins.restype = ctypes.c_bool
    nir_lower_calls_to_builtins.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_find_inlinable_uniforms = _libraries['libtinymesa_cpu.so'].nir_find_inlinable_uniforms
    nir_find_inlinable_uniforms.restype = None
    nir_find_inlinable_uniforms.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_inline_uniforms = _libraries['libtinymesa_cpu.so'].nir_inline_uniforms
    nir_inline_uniforms.restype = ctypes.c_bool
    nir_inline_uniforms.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass
try:
    nir_collect_src_uniforms = _libraries['libtinymesa_cpu.so'].nir_collect_src_uniforms
    nir_collect_src_uniforms.restype = ctypes.c_bool
    nir_collect_src_uniforms.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_add_inlinable_uniforms = _libraries['libtinymesa_cpu.so'].nir_add_inlinable_uniforms
    nir_add_inlinable_uniforms.restype = None
    nir_add_inlinable_uniforms.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_loop_info), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_propagate_invariant = _libraries['libtinymesa_cpu.so'].nir_propagate_invariant
    nir_propagate_invariant.restype = ctypes.c_bool
    nir_propagate_invariant.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_var_copy_instr = _libraries['FIXME_STUB'].nir_lower_var_copy_instr
    nir_lower_var_copy_instr.restype = None
    nir_lower_var_copy_instr.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_deref_copy_instr = _libraries['libtinymesa_cpu.so'].nir_lower_deref_copy_instr
    nir_lower_deref_copy_instr.restype = None
    nir_lower_deref_copy_instr.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_lower_var_copies = _libraries['libtinymesa_cpu.so'].nir_lower_var_copies
    nir_lower_var_copies.restype = ctypes.c_bool
    nir_lower_var_copies.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_memcpy = _libraries['libtinymesa_cpu.so'].nir_opt_memcpy
    nir_opt_memcpy.restype = ctypes.c_bool
    nir_opt_memcpy.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_memcpy = _libraries['libtinymesa_cpu.so'].nir_lower_memcpy
    nir_lower_memcpy.restype = ctypes.c_bool
    nir_lower_memcpy.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_fixup_deref_modes = _libraries['libtinymesa_cpu.so'].nir_fixup_deref_modes
    nir_fixup_deref_modes.restype = ctypes.c_bool
    nir_fixup_deref_modes.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_fixup_deref_types = _libraries['libtinymesa_cpu.so'].nir_fixup_deref_types
    nir_fixup_deref_types.restype = ctypes.c_bool
    nir_fixup_deref_types.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_global_vars_to_local = _libraries['libtinymesa_cpu.so'].nir_lower_global_vars_to_local
    nir_lower_global_vars_to_local.restype = ctypes.c_bool
    nir_lower_global_vars_to_local.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_constant_to_temp = _libraries['libtinymesa_cpu.so'].nir_lower_constant_to_temp
    nir_lower_constant_to_temp.restype = None
    nir_lower_constant_to_temp.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_array_deref_of_vec_options'
c__EA_nir_lower_array_deref_of_vec_options__enumvalues = {
    1: 'nir_lower_direct_array_deref_of_vec_load',
    2: 'nir_lower_indirect_array_deref_of_vec_load',
    4: 'nir_lower_direct_array_deref_of_vec_store',
    8: 'nir_lower_indirect_array_deref_of_vec_store',
}
nir_lower_direct_array_deref_of_vec_load = 1
nir_lower_indirect_array_deref_of_vec_load = 2
nir_lower_direct_array_deref_of_vec_store = 4
nir_lower_indirect_array_deref_of_vec_store = 8
c__EA_nir_lower_array_deref_of_vec_options = ctypes.c_uint32 # enum
nir_lower_array_deref_of_vec_options = c__EA_nir_lower_array_deref_of_vec_options
nir_lower_array_deref_of_vec_options__enumvalues = c__EA_nir_lower_array_deref_of_vec_options__enumvalues
try:
    nir_lower_array_deref_of_vec = _libraries['libtinymesa_cpu.so'].nir_lower_array_deref_of_vec
    nir_lower_array_deref_of_vec.restype = ctypes.c_bool
    nir_lower_array_deref_of_vec.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_variable)), nir_lower_array_deref_of_vec_options]
except AttributeError:
    pass
try:
    nir_lower_indirect_derefs = _libraries['libtinymesa_cpu.so'].nir_lower_indirect_derefs
    nir_lower_indirect_derefs.restype = ctypes.c_bool
    nir_lower_indirect_derefs.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, uint32_t]
except AttributeError:
    pass
try:
    nir_lower_indirect_var_derefs = _libraries['libtinymesa_cpu.so'].nir_lower_indirect_var_derefs
    nir_lower_indirect_var_derefs.restype = ctypes.c_bool
    nir_lower_indirect_var_derefs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_set)]
except AttributeError:
    pass
try:
    nir_lower_locals_to_regs = _libraries['libtinymesa_cpu.so'].nir_lower_locals_to_regs
    nir_lower_locals_to_regs.restype = ctypes.c_bool
    nir_lower_locals_to_regs.argtypes = [ctypes.POINTER(struct_nir_shader), uint8_t]
except AttributeError:
    pass
try:
    nir_lower_io_vars_to_temporaries = _libraries['libtinymesa_cpu.so'].nir_lower_io_vars_to_temporaries
    nir_lower_io_vars_to_temporaries.restype = ctypes.c_bool
    nir_lower_io_vars_to_temporaries.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_vars_to_scratch = _libraries['libtinymesa_cpu.so'].nir_lower_vars_to_scratch
    nir_lower_vars_to_scratch.restype = ctypes.c_bool
    nir_lower_vars_to_scratch.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_int32, glsl_type_size_align_func, glsl_type_size_align_func]
except AttributeError:
    pass
try:
    nir_lower_scratch_to_var = _libraries['libtinymesa_cpu.so'].nir_lower_scratch_to_var
    nir_lower_scratch_to_var.restype = ctypes.c_bool
    nir_lower_scratch_to_var.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_clip_halfz = _libraries['libtinymesa_cpu.so'].nir_lower_clip_halfz
    nir_lower_clip_halfz.restype = ctypes.c_bool
    nir_lower_clip_halfz.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_gather_info = _libraries['libtinymesa_cpu.so'].nir_shader_gather_info
    nir_shader_gather_info.restype = None
    nir_shader_gather_info.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_gather_types = _libraries['libtinymesa_cpu.so'].nir_gather_types
    nir_gather_types.restype = None
    nir_gather_types.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    nir_remove_unused_varyings = _libraries['libtinymesa_cpu.so'].nir_remove_unused_varyings
    nir_remove_unused_varyings.restype = ctypes.c_bool
    nir_remove_unused_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_unused_io_vars = _libraries['libtinymesa_cpu.so'].nir_remove_unused_io_vars
    nir_remove_unused_io_vars.restype = ctypes.c_bool
    nir_remove_unused_io_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nir_compact_varyings = _libraries['libtinymesa_cpu.so'].nir_compact_varyings
    nir_compact_varyings.restype = None
    nir_compact_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_link_xfb_varyings = _libraries['libtinymesa_cpu.so'].nir_link_xfb_varyings
    nir_link_xfb_varyings.restype = None
    nir_link_xfb_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_link_opt_varyings = _libraries['libtinymesa_cpu.so'].nir_link_opt_varyings
    nir_link_opt_varyings.restype = ctypes.c_bool
    nir_link_opt_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_link_varying_precision = _libraries['libtinymesa_cpu.so'].nir_link_varying_precision
    nir_link_varying_precision.restype = None
    nir_link_varying_precision.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_clone_uniform_variable = _libraries['libtinymesa_cpu.so'].nir_clone_uniform_variable
    nir_clone_uniform_variable.restype = ctypes.POINTER(struct_nir_variable)
    nir_clone_uniform_variable.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_variable), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_clone_deref_instr = _libraries['libtinymesa_cpu.so'].nir_clone_deref_instr
    nir_clone_deref_instr.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_clone_deref_instr.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_opt_varyings_progress'
c__EA_nir_opt_varyings_progress__enumvalues = {
    1: 'nir_progress_producer',
    2: 'nir_progress_consumer',
}
nir_progress_producer = 1
nir_progress_consumer = 2
c__EA_nir_opt_varyings_progress = ctypes.c_uint32 # enum
nir_opt_varyings_progress = c__EA_nir_opt_varyings_progress
nir_opt_varyings_progress__enumvalues = c__EA_nir_opt_varyings_progress__enumvalues
try:
    nir_opt_varyings = _libraries['libtinymesa_cpu.so'].nir_opt_varyings
    nir_opt_varyings.restype = nir_opt_varyings_progress
    nir_opt_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
except AttributeError:
    pass

# values for enumeration 'c__EA_gl_varying_slot'
c__EA_gl_varying_slot__enumvalues = {
    0: 'VARYING_SLOT_POS',
    1: 'VARYING_SLOT_COL0',
    2: 'VARYING_SLOT_COL1',
    3: 'VARYING_SLOT_FOGC',
    4: 'VARYING_SLOT_TEX0',
    5: 'VARYING_SLOT_TEX1',
    6: 'VARYING_SLOT_TEX2',
    7: 'VARYING_SLOT_TEX3',
    8: 'VARYING_SLOT_TEX4',
    9: 'VARYING_SLOT_TEX5',
    10: 'VARYING_SLOT_TEX6',
    11: 'VARYING_SLOT_TEX7',
    12: 'VARYING_SLOT_PSIZ',
    13: 'VARYING_SLOT_BFC0',
    14: 'VARYING_SLOT_BFC1',
    15: 'VARYING_SLOT_EDGE',
    16: 'VARYING_SLOT_CLIP_VERTEX',
    17: 'VARYING_SLOT_CLIP_DIST0',
    18: 'VARYING_SLOT_CLIP_DIST1',
    19: 'VARYING_SLOT_CULL_DIST0',
    20: 'VARYING_SLOT_CULL_DIST1',
    21: 'VARYING_SLOT_PRIMITIVE_ID',
    22: 'VARYING_SLOT_LAYER',
    23: 'VARYING_SLOT_VIEWPORT',
    24: 'VARYING_SLOT_FACE',
    25: 'VARYING_SLOT_PNTC',
    26: 'VARYING_SLOT_TESS_LEVEL_OUTER',
    27: 'VARYING_SLOT_TESS_LEVEL_INNER',
    28: 'VARYING_SLOT_BOUNDING_BOX0',
    29: 'VARYING_SLOT_BOUNDING_BOX1',
    30: 'VARYING_SLOT_VIEW_INDEX',
    31: 'VARYING_SLOT_VIEWPORT_MASK',
    24: 'VARYING_SLOT_PRIMITIVE_SHADING_RATE',
    26: 'VARYING_SLOT_PRIMITIVE_COUNT',
    27: 'VARYING_SLOT_PRIMITIVE_INDICES',
    28: 'VARYING_SLOT_TASK_COUNT',
    28: 'VARYING_SLOT_CULL_PRIMITIVE',
    32: 'VARYING_SLOT_VAR0',
    33: 'VARYING_SLOT_VAR1',
    34: 'VARYING_SLOT_VAR2',
    35: 'VARYING_SLOT_VAR3',
    36: 'VARYING_SLOT_VAR4',
    37: 'VARYING_SLOT_VAR5',
    38: 'VARYING_SLOT_VAR6',
    39: 'VARYING_SLOT_VAR7',
    40: 'VARYING_SLOT_VAR8',
    41: 'VARYING_SLOT_VAR9',
    42: 'VARYING_SLOT_VAR10',
    43: 'VARYING_SLOT_VAR11',
    44: 'VARYING_SLOT_VAR12',
    45: 'VARYING_SLOT_VAR13',
    46: 'VARYING_SLOT_VAR14',
    47: 'VARYING_SLOT_VAR15',
    48: 'VARYING_SLOT_VAR16',
    49: 'VARYING_SLOT_VAR17',
    50: 'VARYING_SLOT_VAR18',
    51: 'VARYING_SLOT_VAR19',
    52: 'VARYING_SLOT_VAR20',
    53: 'VARYING_SLOT_VAR21',
    54: 'VARYING_SLOT_VAR22',
    55: 'VARYING_SLOT_VAR23',
    56: 'VARYING_SLOT_VAR24',
    57: 'VARYING_SLOT_VAR25',
    58: 'VARYING_SLOT_VAR26',
    59: 'VARYING_SLOT_VAR27',
    60: 'VARYING_SLOT_VAR28',
    61: 'VARYING_SLOT_VAR29',
    62: 'VARYING_SLOT_VAR30',
    63: 'VARYING_SLOT_VAR31',
    64: 'VARYING_SLOT_PATCH0',
    65: 'VARYING_SLOT_PATCH1',
    66: 'VARYING_SLOT_PATCH2',
    67: 'VARYING_SLOT_PATCH3',
    68: 'VARYING_SLOT_PATCH4',
    69: 'VARYING_SLOT_PATCH5',
    70: 'VARYING_SLOT_PATCH6',
    71: 'VARYING_SLOT_PATCH7',
    72: 'VARYING_SLOT_PATCH8',
    73: 'VARYING_SLOT_PATCH9',
    74: 'VARYING_SLOT_PATCH10',
    75: 'VARYING_SLOT_PATCH11',
    76: 'VARYING_SLOT_PATCH12',
    77: 'VARYING_SLOT_PATCH13',
    78: 'VARYING_SLOT_PATCH14',
    79: 'VARYING_SLOT_PATCH15',
    80: 'VARYING_SLOT_PATCH16',
    81: 'VARYING_SLOT_PATCH17',
    82: 'VARYING_SLOT_PATCH18',
    83: 'VARYING_SLOT_PATCH19',
    84: 'VARYING_SLOT_PATCH20',
    85: 'VARYING_SLOT_PATCH21',
    86: 'VARYING_SLOT_PATCH22',
    87: 'VARYING_SLOT_PATCH23',
    88: 'VARYING_SLOT_PATCH24',
    89: 'VARYING_SLOT_PATCH25',
    90: 'VARYING_SLOT_PATCH26',
    91: 'VARYING_SLOT_PATCH27',
    92: 'VARYING_SLOT_PATCH28',
    93: 'VARYING_SLOT_PATCH29',
    94: 'VARYING_SLOT_PATCH30',
    95: 'VARYING_SLOT_PATCH31',
    96: 'VARYING_SLOT_VAR0_16BIT',
    97: 'VARYING_SLOT_VAR1_16BIT',
    98: 'VARYING_SLOT_VAR2_16BIT',
    99: 'VARYING_SLOT_VAR3_16BIT',
    100: 'VARYING_SLOT_VAR4_16BIT',
    101: 'VARYING_SLOT_VAR5_16BIT',
    102: 'VARYING_SLOT_VAR6_16BIT',
    103: 'VARYING_SLOT_VAR7_16BIT',
    104: 'VARYING_SLOT_VAR8_16BIT',
    105: 'VARYING_SLOT_VAR9_16BIT',
    106: 'VARYING_SLOT_VAR10_16BIT',
    107: 'VARYING_SLOT_VAR11_16BIT',
    108: 'VARYING_SLOT_VAR12_16BIT',
    109: 'VARYING_SLOT_VAR13_16BIT',
    110: 'VARYING_SLOT_VAR14_16BIT',
    111: 'VARYING_SLOT_VAR15_16BIT',
    112: 'NUM_TOTAL_VARYING_SLOTS',
}
VARYING_SLOT_POS = 0
VARYING_SLOT_COL0 = 1
VARYING_SLOT_COL1 = 2
VARYING_SLOT_FOGC = 3
VARYING_SLOT_TEX0 = 4
VARYING_SLOT_TEX1 = 5
VARYING_SLOT_TEX2 = 6
VARYING_SLOT_TEX3 = 7
VARYING_SLOT_TEX4 = 8
VARYING_SLOT_TEX5 = 9
VARYING_SLOT_TEX6 = 10
VARYING_SLOT_TEX7 = 11
VARYING_SLOT_PSIZ = 12
VARYING_SLOT_BFC0 = 13
VARYING_SLOT_BFC1 = 14
VARYING_SLOT_EDGE = 15
VARYING_SLOT_CLIP_VERTEX = 16
VARYING_SLOT_CLIP_DIST0 = 17
VARYING_SLOT_CLIP_DIST1 = 18
VARYING_SLOT_CULL_DIST0 = 19
VARYING_SLOT_CULL_DIST1 = 20
VARYING_SLOT_PRIMITIVE_ID = 21
VARYING_SLOT_LAYER = 22
VARYING_SLOT_VIEWPORT = 23
VARYING_SLOT_FACE = 24
VARYING_SLOT_PNTC = 25
VARYING_SLOT_TESS_LEVEL_OUTER = 26
VARYING_SLOT_TESS_LEVEL_INNER = 27
VARYING_SLOT_BOUNDING_BOX0 = 28
VARYING_SLOT_BOUNDING_BOX1 = 29
VARYING_SLOT_VIEW_INDEX = 30
VARYING_SLOT_VIEWPORT_MASK = 31
VARYING_SLOT_PRIMITIVE_SHADING_RATE = 24
VARYING_SLOT_PRIMITIVE_COUNT = 26
VARYING_SLOT_PRIMITIVE_INDICES = 27
VARYING_SLOT_TASK_COUNT = 28
VARYING_SLOT_CULL_PRIMITIVE = 28
VARYING_SLOT_VAR0 = 32
VARYING_SLOT_VAR1 = 33
VARYING_SLOT_VAR2 = 34
VARYING_SLOT_VAR3 = 35
VARYING_SLOT_VAR4 = 36
VARYING_SLOT_VAR5 = 37
VARYING_SLOT_VAR6 = 38
VARYING_SLOT_VAR7 = 39
VARYING_SLOT_VAR8 = 40
VARYING_SLOT_VAR9 = 41
VARYING_SLOT_VAR10 = 42
VARYING_SLOT_VAR11 = 43
VARYING_SLOT_VAR12 = 44
VARYING_SLOT_VAR13 = 45
VARYING_SLOT_VAR14 = 46
VARYING_SLOT_VAR15 = 47
VARYING_SLOT_VAR16 = 48
VARYING_SLOT_VAR17 = 49
VARYING_SLOT_VAR18 = 50
VARYING_SLOT_VAR19 = 51
VARYING_SLOT_VAR20 = 52
VARYING_SLOT_VAR21 = 53
VARYING_SLOT_VAR22 = 54
VARYING_SLOT_VAR23 = 55
VARYING_SLOT_VAR24 = 56
VARYING_SLOT_VAR25 = 57
VARYING_SLOT_VAR26 = 58
VARYING_SLOT_VAR27 = 59
VARYING_SLOT_VAR28 = 60
VARYING_SLOT_VAR29 = 61
VARYING_SLOT_VAR30 = 62
VARYING_SLOT_VAR31 = 63
VARYING_SLOT_PATCH0 = 64
VARYING_SLOT_PATCH1 = 65
VARYING_SLOT_PATCH2 = 66
VARYING_SLOT_PATCH3 = 67
VARYING_SLOT_PATCH4 = 68
VARYING_SLOT_PATCH5 = 69
VARYING_SLOT_PATCH6 = 70
VARYING_SLOT_PATCH7 = 71
VARYING_SLOT_PATCH8 = 72
VARYING_SLOT_PATCH9 = 73
VARYING_SLOT_PATCH10 = 74
VARYING_SLOT_PATCH11 = 75
VARYING_SLOT_PATCH12 = 76
VARYING_SLOT_PATCH13 = 77
VARYING_SLOT_PATCH14 = 78
VARYING_SLOT_PATCH15 = 79
VARYING_SLOT_PATCH16 = 80
VARYING_SLOT_PATCH17 = 81
VARYING_SLOT_PATCH18 = 82
VARYING_SLOT_PATCH19 = 83
VARYING_SLOT_PATCH20 = 84
VARYING_SLOT_PATCH21 = 85
VARYING_SLOT_PATCH22 = 86
VARYING_SLOT_PATCH23 = 87
VARYING_SLOT_PATCH24 = 88
VARYING_SLOT_PATCH25 = 89
VARYING_SLOT_PATCH26 = 90
VARYING_SLOT_PATCH27 = 91
VARYING_SLOT_PATCH28 = 92
VARYING_SLOT_PATCH29 = 93
VARYING_SLOT_PATCH30 = 94
VARYING_SLOT_PATCH31 = 95
VARYING_SLOT_VAR0_16BIT = 96
VARYING_SLOT_VAR1_16BIT = 97
VARYING_SLOT_VAR2_16BIT = 98
VARYING_SLOT_VAR3_16BIT = 99
VARYING_SLOT_VAR4_16BIT = 100
VARYING_SLOT_VAR5_16BIT = 101
VARYING_SLOT_VAR6_16BIT = 102
VARYING_SLOT_VAR7_16BIT = 103
VARYING_SLOT_VAR8_16BIT = 104
VARYING_SLOT_VAR9_16BIT = 105
VARYING_SLOT_VAR10_16BIT = 106
VARYING_SLOT_VAR11_16BIT = 107
VARYING_SLOT_VAR12_16BIT = 108
VARYING_SLOT_VAR13_16BIT = 109
VARYING_SLOT_VAR14_16BIT = 110
VARYING_SLOT_VAR15_16BIT = 111
NUM_TOTAL_VARYING_SLOTS = 112
c__EA_gl_varying_slot = ctypes.c_uint32 # enum
gl_varying_slot = c__EA_gl_varying_slot
gl_varying_slot__enumvalues = c__EA_gl_varying_slot__enumvalues
try:
    nir_slot_is_sysval_output = _libraries['libtinymesa_cpu.so'].nir_slot_is_sysval_output
    nir_slot_is_sysval_output.restype = ctypes.c_bool
    nir_slot_is_sysval_output.argtypes = [gl_varying_slot, gl_shader_stage]
except AttributeError:
    pass
try:
    nir_slot_is_varying = _libraries['libtinymesa_cpu.so'].nir_slot_is_varying
    nir_slot_is_varying.restype = ctypes.c_bool
    nir_slot_is_varying.argtypes = [gl_varying_slot, gl_shader_stage]
except AttributeError:
    pass
try:
    nir_slot_is_sysval_output_and_varying = _libraries['libtinymesa_cpu.so'].nir_slot_is_sysval_output_and_varying
    nir_slot_is_sysval_output_and_varying.restype = ctypes.c_bool
    nir_slot_is_sysval_output_and_varying.argtypes = [gl_varying_slot, gl_shader_stage]
except AttributeError:
    pass
try:
    nir_remove_varying = _libraries['libtinymesa_cpu.so'].nir_remove_varying
    nir_remove_varying.restype = ctypes.c_bool
    nir_remove_varying.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), gl_shader_stage]
except AttributeError:
    pass
try:
    nir_remove_sysval_output = _libraries['libtinymesa_cpu.so'].nir_remove_sysval_output
    nir_remove_sysval_output.restype = ctypes.c_bool
    nir_remove_sysval_output.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), gl_shader_stage]
except AttributeError:
    pass
try:
    nir_lower_amul = _libraries['libtinymesa_cpu.so'].nir_lower_amul
    nir_lower_amul.restype = ctypes.c_bool
    nir_lower_amul.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_glsl_type), ctypes.c_bool)]
except AttributeError:
    pass
try:
    nir_lower_ubo_vec4 = _libraries['libtinymesa_cpu.so'].nir_lower_ubo_vec4
    nir_lower_ubo_vec4.restype = ctypes.c_bool
    nir_lower_ubo_vec4.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_sort_variables_by_location = _libraries['libtinymesa_cpu.so'].nir_sort_variables_by_location
    nir_sort_variables_by_location.restype = None
    nir_sort_variables_by_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_assign_io_var_locations = _libraries['libtinymesa_cpu.so'].nir_assign_io_var_locations
    nir_assign_io_var_locations.restype = None
    nir_assign_io_var_locations.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.POINTER(ctypes.c_uint32), gl_shader_stage]
except AttributeError:
    pass
try:
    nir_opt_clip_cull_const = _libraries['libtinymesa_cpu.so'].nir_opt_clip_cull_const
    nir_opt_clip_cull_const.restype = ctypes.c_bool
    nir_opt_clip_cull_const.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_io_options'
c__EA_nir_lower_io_options__enumvalues = {
    1: 'nir_lower_io_lower_64bit_to_32',
    2: 'nir_lower_io_lower_64bit_float_to_32',
    4: 'nir_lower_io_lower_64bit_to_32_new',
    8: 'nir_lower_io_use_interpolated_input_intrinsics',
}
nir_lower_io_lower_64bit_to_32 = 1
nir_lower_io_lower_64bit_float_to_32 = 2
nir_lower_io_lower_64bit_to_32_new = 4
nir_lower_io_use_interpolated_input_intrinsics = 8
c__EA_nir_lower_io_options = ctypes.c_uint32 # enum
nir_lower_io_options = c__EA_nir_lower_io_options
nir_lower_io_options__enumvalues = c__EA_nir_lower_io_options__enumvalues
try:
    nir_lower_io = _libraries['libtinymesa_cpu.so'].nir_lower_io
    nir_lower_io.restype = ctypes.c_bool
    nir_lower_io.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_glsl_type), ctypes.c_bool), nir_lower_io_options]
except AttributeError:
    pass
try:
    nir_io_add_const_offset_to_base = _libraries['libtinymesa_cpu.so'].nir_io_add_const_offset_to_base
    nir_io_add_const_offset_to_base.restype = ctypes.c_bool
    nir_io_add_const_offset_to_base.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_io_passes = _libraries['libtinymesa_cpu.so'].nir_lower_io_passes
    nir_lower_io_passes.restype = None
    nir_lower_io_passes.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_io_add_intrinsic_xfb_info = _libraries['libtinymesa_cpu.so'].nir_io_add_intrinsic_xfb_info
    nir_io_add_intrinsic_xfb_info.restype = ctypes.c_bool
    nir_io_add_intrinsic_xfb_info.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_io_indirect_loads = _libraries['libtinymesa_cpu.so'].nir_lower_io_indirect_loads
    nir_lower_io_indirect_loads.restype = ctypes.c_bool
    nir_lower_io_indirect_loads.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_vars_to_explicit_types = _libraries['libtinymesa_cpu.so'].nir_lower_vars_to_explicit_types
    nir_lower_vars_to_explicit_types.restype = ctypes.c_bool
    nir_lower_vars_to_explicit_types.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, glsl_type_size_align_func]
except AttributeError:
    pass
try:
    nir_gather_explicit_io_initializers = _libraries['libtinymesa_cpu.so'].nir_gather_explicit_io_initializers
    nir_gather_explicit_io_initializers.restype = None
    nir_gather_explicit_io_initializers.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(None), size_t, nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_vec3_to_vec4 = _libraries['libtinymesa_cpu.so'].nir_lower_vec3_to_vec4
    nir_lower_vec3_to_vec4.restype = ctypes.c_bool
    nir_lower_vec3_to_vec4.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_address_format'
c__EA_nir_address_format__enumvalues = {
    0: 'nir_address_format_32bit_global',
    1: 'nir_address_format_64bit_global',
    2: 'nir_address_format_2x32bit_global',
    3: 'nir_address_format_64bit_global_32bit_offset',
    4: 'nir_address_format_64bit_bounded_global',
    5: 'nir_address_format_32bit_index_offset',
    6: 'nir_address_format_32bit_index_offset_pack64',
    7: 'nir_address_format_vec2_index_32bit_offset',
    8: 'nir_address_format_62bit_generic',
    9: 'nir_address_format_32bit_offset',
    10: 'nir_address_format_32bit_offset_as_64bit',
    11: 'nir_address_format_logical',
}
nir_address_format_32bit_global = 0
nir_address_format_64bit_global = 1
nir_address_format_2x32bit_global = 2
nir_address_format_64bit_global_32bit_offset = 3
nir_address_format_64bit_bounded_global = 4
nir_address_format_32bit_index_offset = 5
nir_address_format_32bit_index_offset_pack64 = 6
nir_address_format_vec2_index_32bit_offset = 7
nir_address_format_62bit_generic = 8
nir_address_format_32bit_offset = 9
nir_address_format_32bit_offset_as_64bit = 10
nir_address_format_logical = 11
c__EA_nir_address_format = ctypes.c_uint32 # enum
nir_address_format = c__EA_nir_address_format
nir_address_format__enumvalues = c__EA_nir_address_format__enumvalues
try:
    nir_address_format_bit_size = _libraries['libtinymesa_cpu.so'].nir_address_format_bit_size
    nir_address_format_bit_size.restype = ctypes.c_uint32
    nir_address_format_bit_size.argtypes = [nir_address_format]
except AttributeError:
    pass
try:
    nir_address_format_num_components = _libraries['libtinymesa_cpu.so'].nir_address_format_num_components
    nir_address_format_num_components.restype = ctypes.c_uint32
    nir_address_format_num_components.argtypes = [nir_address_format]
except AttributeError:
    pass
try:
    nir_address_format_to_glsl_type = _libraries['FIXME_STUB'].nir_address_format_to_glsl_type
    nir_address_format_to_glsl_type.restype = ctypes.POINTER(struct_glsl_type)
    nir_address_format_to_glsl_type.argtypes = [nir_address_format]
except AttributeError:
    pass
try:
    nir_address_format_null_value = _libraries['libtinymesa_cpu.so'].nir_address_format_null_value
    nir_address_format_null_value.restype = ctypes.POINTER(union_c__UA_nir_const_value)
    nir_address_format_null_value.argtypes = [nir_address_format]
except AttributeError:
    pass
try:
    nir_build_addr_iadd = _libraries['libtinymesa_cpu.so'].nir_build_addr_iadd
    nir_build_addr_iadd.restype = ctypes.POINTER(struct_nir_def)
    nir_build_addr_iadd.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_address_format, nir_variable_mode, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_addr_iadd_imm = _libraries['libtinymesa_cpu.so'].nir_build_addr_iadd_imm
    nir_build_addr_iadd_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_build_addr_iadd_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_address_format, nir_variable_mode, int64_t]
except AttributeError:
    pass
try:
    nir_build_addr_ieq = _libraries['libtinymesa_cpu.so'].nir_build_addr_ieq
    nir_build_addr_ieq.restype = ctypes.POINTER(struct_nir_def)
    nir_build_addr_ieq.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), nir_address_format]
except AttributeError:
    pass
try:
    nir_build_addr_isub = _libraries['libtinymesa_cpu.so'].nir_build_addr_isub
    nir_build_addr_isub.restype = ctypes.POINTER(struct_nir_def)
    nir_build_addr_isub.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), nir_address_format]
except AttributeError:
    pass
try:
    nir_explicit_io_address_from_deref = _libraries['libtinymesa_cpu.so'].nir_explicit_io_address_from_deref
    nir_explicit_io_address_from_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_explicit_io_address_from_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), nir_address_format]
except AttributeError:
    pass
try:
    nir_get_explicit_deref_align = _libraries['libtinymesa_cpu.so'].nir_get_explicit_deref_align
    nir_get_explicit_deref_align.restype = ctypes.c_bool
    nir_get_explicit_deref_align.argtypes = [ctypes.POINTER(struct_nir_deref_instr), ctypes.c_bool, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    nir_lower_explicit_io_instr = _libraries['libtinymesa_cpu.so'].nir_lower_explicit_io_instr
    nir_lower_explicit_io_instr.restype = None
    nir_lower_explicit_io_instr.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_def), nir_address_format]
except AttributeError:
    pass
try:
    nir_lower_explicit_io = _libraries['libtinymesa_cpu.so'].nir_lower_explicit_io
    nir_lower_explicit_io.restype = ctypes.c_bool
    nir_lower_explicit_io.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, nir_address_format]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_mem_access_shift_method'
c__EA_nir_mem_access_shift_method__enumvalues = {
    0: 'nir_mem_access_shift_method_scalar',
    1: 'nir_mem_access_shift_method_shift64',
    2: 'nir_mem_access_shift_method_bytealign_amd',
}
nir_mem_access_shift_method_scalar = 0
nir_mem_access_shift_method_shift64 = 1
nir_mem_access_shift_method_bytealign_amd = 2
c__EA_nir_mem_access_shift_method = ctypes.c_uint32 # enum
nir_mem_access_shift_method = c__EA_nir_mem_access_shift_method
nir_mem_access_shift_method__enumvalues = c__EA_nir_mem_access_shift_method__enumvalues
class struct_nir_mem_access_size_align(Structure):
    pass

struct_nir_mem_access_size_align._pack_ = 1 # source:False
struct_nir_mem_access_size_align._fields_ = [
    ('num_components', ctypes.c_ubyte),
    ('bit_size', ctypes.c_ubyte),
    ('align', ctypes.c_uint16),
    ('shift', nir_mem_access_shift_method),
]

nir_mem_access_size_align = struct_nir_mem_access_size_align

# values for enumeration 'gl_access_qualifier'
gl_access_qualifier__enumvalues = {
    1: 'ACCESS_COHERENT',
    2: 'ACCESS_RESTRICT',
    4: 'ACCESS_VOLATILE',
    8: 'ACCESS_NON_READABLE',
    16: 'ACCESS_NON_WRITEABLE',
    32: 'ACCESS_NON_UNIFORM',
    64: 'ACCESS_CAN_REORDER',
    128: 'ACCESS_NON_TEMPORAL',
    256: 'ACCESS_INCLUDE_HELPERS',
    512: 'ACCESS_IS_SWIZZLED_AMD',
    1024: 'ACCESS_USES_FORMAT_AMD',
    2048: 'ACCESS_FMASK_LOWERED_AMD',
    4096: 'ACCESS_CAN_SPECULATE',
    8192: 'ACCESS_CP_GE_COHERENT_AMD',
    16384: 'ACCESS_IN_BOUNDS',
    32768: 'ACCESS_KEEP_SCALAR',
    65536: 'ACCESS_SMEM_AMD',
}
ACCESS_COHERENT = 1
ACCESS_RESTRICT = 2
ACCESS_VOLATILE = 4
ACCESS_NON_READABLE = 8
ACCESS_NON_WRITEABLE = 16
ACCESS_NON_UNIFORM = 32
ACCESS_CAN_REORDER = 64
ACCESS_NON_TEMPORAL = 128
ACCESS_INCLUDE_HELPERS = 256
ACCESS_IS_SWIZZLED_AMD = 512
ACCESS_USES_FORMAT_AMD = 1024
ACCESS_FMASK_LOWERED_AMD = 2048
ACCESS_CAN_SPECULATE = 4096
ACCESS_CP_GE_COHERENT_AMD = 8192
ACCESS_IN_BOUNDS = 16384
ACCESS_KEEP_SCALAR = 32768
ACCESS_SMEM_AMD = 65536
gl_access_qualifier = ctypes.c_uint32 # enum
nir_lower_mem_access_bit_sizes_cb = ctypes.CFUNCTYPE(struct_nir_mem_access_size_align, c__EA_nir_intrinsic_op, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, gl_access_qualifier, ctypes.POINTER(None))
class struct_nir_lower_mem_access_bit_sizes_options(Structure):
    pass

struct_nir_lower_mem_access_bit_sizes_options._pack_ = 1 # source:False
struct_nir_lower_mem_access_bit_sizes_options._fields_ = [
    ('callback', ctypes.CFUNCTYPE(struct_nir_mem_access_size_align, c__EA_nir_intrinsic_op, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, gl_access_qualifier, ctypes.POINTER(None))),
    ('modes', nir_variable_mode),
    ('may_lower_unaligned_stores_to_atomics', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('cb_data', ctypes.POINTER(None)),
]

nir_lower_mem_access_bit_sizes_options = struct_nir_lower_mem_access_bit_sizes_options
try:
    nir_lower_mem_access_bit_sizes = _libraries['libtinymesa_cpu.so'].nir_lower_mem_access_bit_sizes
    nir_lower_mem_access_bit_sizes.restype = ctypes.c_bool
    nir_lower_mem_access_bit_sizes.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_mem_access_bit_sizes_options)]
except AttributeError:
    pass
try:
    nir_lower_robust_access = _libraries['libtinymesa_cpu.so'].nir_lower_robust_access
    nir_lower_robust_access.restype = ctypes.c_bool
    nir_lower_robust_access.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrin_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
nir_should_vectorize_mem_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
class struct_nir_load_store_vectorize_options(Structure):
    pass

struct_nir_load_store_vectorize_options._pack_ = 1 # source:False
struct_nir_load_store_vectorize_options._fields_ = [
    ('callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))),
    ('modes', nir_variable_mode),
    ('robust_modes', nir_variable_mode),
    ('cb_data', ctypes.POINTER(None)),
    ('has_shared2_amd', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

nir_load_store_vectorize_options = struct_nir_load_store_vectorize_options
try:
    nir_opt_load_store_vectorize = _libraries['libtinymesa_cpu.so'].nir_opt_load_store_vectorize
    nir_opt_load_store_vectorize.restype = ctypes.c_bool
    nir_opt_load_store_vectorize.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_load_store_vectorize_options)]
except AttributeError:
    pass
try:
    nir_opt_load_store_update_alignments = _libraries['libtinymesa_cpu.so'].nir_opt_load_store_update_alignments
    nir_opt_load_store_update_alignments.restype = ctypes.c_bool
    nir_opt_load_store_update_alignments.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
nir_lower_shader_calls_should_remat_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
class struct_nir_lower_shader_calls_options(Structure):
    pass

struct_nir_lower_shader_calls_options._pack_ = 1 # source:False
struct_nir_lower_shader_calls_options._fields_ = [
    ('address_format', nir_address_format),
    ('stack_alignment', ctypes.c_uint32),
    ('localized_loads', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('vectorizer_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))),
    ('vectorizer_data', ctypes.POINTER(None)),
    ('should_remat_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('should_remat_data', ctypes.POINTER(None)),
]

nir_lower_shader_calls_options = struct_nir_lower_shader_calls_options
try:
    nir_lower_shader_calls = _libraries['libtinymesa_cpu.so'].nir_lower_shader_calls
    nir_lower_shader_calls.restype = ctypes.c_bool
    nir_lower_shader_calls.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_shader_calls_options), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct_nir_shader))), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_get_io_offset_src_number = _libraries['libtinymesa_cpu.so'].nir_get_io_offset_src_number
    nir_get_io_offset_src_number.restype = ctypes.c_int32
    nir_get_io_offset_src_number.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_index_src_number = _libraries['libtinymesa_cpu.so'].nir_get_io_index_src_number
    nir_get_io_index_src_number.restype = ctypes.c_int32
    nir_get_io_index_src_number.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_arrayed_index_src_number = _libraries['libtinymesa_cpu.so'].nir_get_io_arrayed_index_src_number
    nir_get_io_arrayed_index_src_number.restype = ctypes.c_int32
    nir_get_io_arrayed_index_src_number.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_offset_src = _libraries['libtinymesa_cpu.so'].nir_get_io_offset_src
    nir_get_io_offset_src.restype = ctypes.POINTER(struct_nir_src)
    nir_get_io_offset_src.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_index_src = _libraries['libtinymesa_cpu.so'].nir_get_io_index_src
    nir_get_io_index_src.restype = ctypes.POINTER(struct_nir_src)
    nir_get_io_index_src.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_arrayed_index_src = _libraries['libtinymesa_cpu.so'].nir_get_io_arrayed_index_src
    nir_get_io_arrayed_index_src.restype = ctypes.POINTER(struct_nir_src)
    nir_get_io_arrayed_index_src.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_shader_call_payload_src = _libraries['libtinymesa_cpu.so'].nir_get_shader_call_payload_src
    nir_get_shader_call_payload_src.restype = ctypes.POINTER(struct_nir_src)
    nir_get_shader_call_payload_src.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_is_output_load = _libraries['libtinymesa_cpu.so'].nir_is_output_load
    nir_is_output_load.restype = ctypes.c_bool
    nir_is_output_load.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_is_arrayed_io = _libraries['libtinymesa_cpu.so'].nir_is_arrayed_io
    nir_is_arrayed_io.restype = ctypes.c_bool
    nir_is_arrayed_io.argtypes = [ctypes.POINTER(struct_nir_variable), gl_shader_stage]
except AttributeError:
    pass
try:
    nir_lower_reg_intrinsics_to_ssa_impl = _libraries['libtinymesa_cpu.so'].nir_lower_reg_intrinsics_to_ssa_impl
    nir_lower_reg_intrinsics_to_ssa_impl.restype = ctypes.c_bool
    nir_lower_reg_intrinsics_to_ssa_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_lower_reg_intrinsics_to_ssa = _libraries['libtinymesa_cpu.so'].nir_lower_reg_intrinsics_to_ssa
    nir_lower_reg_intrinsics_to_ssa.restype = ctypes.c_bool
    nir_lower_reg_intrinsics_to_ssa.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_vars_to_ssa = _libraries['libtinymesa_cpu.so'].nir_lower_vars_to_ssa
    nir_lower_vars_to_ssa.restype = ctypes.c_bool
    nir_lower_vars_to_ssa.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_dead_derefs = _libraries['libtinymesa_cpu.so'].nir_remove_dead_derefs
    nir_remove_dead_derefs.restype = ctypes.c_bool
    nir_remove_dead_derefs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_dead_derefs_impl = _libraries['libtinymesa_cpu.so'].nir_remove_dead_derefs_impl
    nir_remove_dead_derefs_impl.restype = ctypes.c_bool
    nir_remove_dead_derefs_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
class struct_nir_remove_dead_variables_options(Structure):
    pass

struct_nir_remove_dead_variables_options._pack_ = 1 # source:False
struct_nir_remove_dead_variables_options._fields_ = [
    ('can_remove_var', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_variable), ctypes.POINTER(None))),
    ('can_remove_var_data', ctypes.POINTER(None)),
]

nir_remove_dead_variables_options = struct_nir_remove_dead_variables_options
try:
    nir_remove_dead_variables = _libraries['libtinymesa_cpu.so'].nir_remove_dead_variables
    nir_remove_dead_variables.restype = ctypes.c_bool
    nir_remove_dead_variables.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.POINTER(struct_nir_remove_dead_variables_options)]
except AttributeError:
    pass
try:
    nir_lower_variable_initializers = _libraries['libtinymesa_cpu.so'].nir_lower_variable_initializers
    nir_lower_variable_initializers.restype = ctypes.c_bool
    nir_lower_variable_initializers.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_zero_initialize_shared_memory = _libraries['libtinymesa_cpu.so'].nir_zero_initialize_shared_memory
    nir_zero_initialize_shared_memory.restype = ctypes.c_bool
    nir_zero_initialize_shared_memory.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_clear_shared_memory = _libraries['libtinymesa_cpu.so'].nir_clear_shared_memory
    nir_clear_shared_memory.restype = ctypes.c_bool
    nir_clear_shared_memory.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_opt_move_to_top_options'
c__EA_nir_opt_move_to_top_options__enumvalues = {
    1: 'nir_move_to_entry_block_only',
    2: 'nir_move_to_top_input_loads',
    4: 'nir_move_to_top_load_smem_amd',
}
nir_move_to_entry_block_only = 1
nir_move_to_top_input_loads = 2
nir_move_to_top_load_smem_amd = 4
c__EA_nir_opt_move_to_top_options = ctypes.c_uint32 # enum
nir_opt_move_to_top_options = c__EA_nir_opt_move_to_top_options
nir_opt_move_to_top_options__enumvalues = c__EA_nir_opt_move_to_top_options__enumvalues
try:
    nir_opt_move_to_top = _libraries['libtinymesa_cpu.so'].nir_opt_move_to_top
    nir_opt_move_to_top.restype = ctypes.c_bool
    nir_opt_move_to_top.argtypes = [ctypes.POINTER(struct_nir_shader), nir_opt_move_to_top_options]
except AttributeError:
    pass
try:
    nir_move_vec_src_uses_to_dest = _libraries['libtinymesa_cpu.so'].nir_move_vec_src_uses_to_dest
    nir_move_vec_src_uses_to_dest.restype = ctypes.c_bool
    nir_move_vec_src_uses_to_dest.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_move_output_stores_to_end = _libraries['libtinymesa_cpu.so'].nir_move_output_stores_to_end
    nir_move_output_stores_to_end.restype = ctypes.c_bool
    nir_move_output_stores_to_end.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_vec_to_regs = _libraries['libtinymesa_cpu.so'].nir_lower_vec_to_regs
    nir_lower_vec_to_regs.restype = ctypes.c_bool
    nir_lower_vec_to_regs.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_writemask_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'compare_func'
compare_func__enumvalues = {
    0: 'COMPARE_FUNC_NEVER',
    1: 'COMPARE_FUNC_LESS',
    2: 'COMPARE_FUNC_EQUAL',
    3: 'COMPARE_FUNC_LEQUAL',
    4: 'COMPARE_FUNC_GREATER',
    5: 'COMPARE_FUNC_NOTEQUAL',
    6: 'COMPARE_FUNC_GEQUAL',
    7: 'COMPARE_FUNC_ALWAYS',
}
COMPARE_FUNC_NEVER = 0
COMPARE_FUNC_LESS = 1
COMPARE_FUNC_EQUAL = 2
COMPARE_FUNC_LEQUAL = 3
COMPARE_FUNC_GREATER = 4
COMPARE_FUNC_NOTEQUAL = 5
COMPARE_FUNC_GEQUAL = 6
COMPARE_FUNC_ALWAYS = 7
compare_func = ctypes.c_uint32 # enum
try:
    nir_lower_alpha_test = _libraries['libtinymesa_cpu.so'].nir_lower_alpha_test
    nir_lower_alpha_test.restype = ctypes.c_bool
    nir_lower_alpha_test.argtypes = [ctypes.POINTER(struct_nir_shader), compare_func, ctypes.c_bool, ctypes.POINTER(ctypes.c_int16)]
except AttributeError:
    pass
try:
    nir_lower_alpha_to_coverage = _libraries['libtinymesa_cpu.so'].nir_lower_alpha_to_coverage
    nir_lower_alpha_to_coverage.restype = ctypes.c_bool
    nir_lower_alpha_to_coverage.argtypes = [ctypes.POINTER(struct_nir_shader), uint8_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_alpha_to_one = _libraries['libtinymesa_cpu.so'].nir_lower_alpha_to_one
    nir_lower_alpha_to_one.restype = ctypes.c_bool
    nir_lower_alpha_to_one.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_alu = _libraries['libtinymesa_cpu.so'].nir_lower_alu
    nir_lower_alu.restype = ctypes.c_bool
    nir_lower_alu.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_flrp = _libraries['libtinymesa_cpu.so'].nir_lower_flrp
    nir_lower_flrp.restype = ctypes.c_bool
    nir_lower_flrp.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_scale_fdiv = _libraries['libtinymesa_cpu.so'].nir_scale_fdiv
    nir_scale_fdiv.restype = ctypes.c_bool
    nir_scale_fdiv.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_alu_to_scalar = _libraries['libtinymesa_cpu.so'].nir_lower_alu_to_scalar
    nir_lower_alu_to_scalar.restype = ctypes.c_bool
    nir_lower_alu_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_alu_width = _libraries['libtinymesa_cpu.so'].nir_lower_alu_width
    nir_lower_alu_width.restype = ctypes.c_bool
    nir_lower_alu_width.argtypes = [ctypes.POINTER(struct_nir_shader), nir_vectorize_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_alu_vec8_16_srcs = _libraries['libtinymesa_cpu.so'].nir_lower_alu_vec8_16_srcs
    nir_lower_alu_vec8_16_srcs.restype = ctypes.c_bool
    nir_lower_alu_vec8_16_srcs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_bool_to_bitsize = _libraries['libtinymesa_cpu.so'].nir_lower_bool_to_bitsize
    nir_lower_bool_to_bitsize.restype = ctypes.c_bool
    nir_lower_bool_to_bitsize.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_bool_to_float = _libraries['libtinymesa_cpu.so'].nir_lower_bool_to_float
    nir_lower_bool_to_float.restype = ctypes.c_bool
    nir_lower_bool_to_float.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_bool_to_int32 = _libraries['libtinymesa_cpu.so'].nir_lower_bool_to_int32
    nir_lower_bool_to_int32.restype = ctypes.c_bool
    nir_lower_bool_to_int32.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_simplify_convert_alu_types = _libraries['libtinymesa_cpu.so'].nir_opt_simplify_convert_alu_types
    nir_opt_simplify_convert_alu_types.restype = ctypes.c_bool
    nir_opt_simplify_convert_alu_types.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_const_arrays_to_uniforms = _libraries['libtinymesa_cpu.so'].nir_lower_const_arrays_to_uniforms
    nir_lower_const_arrays_to_uniforms.restype = ctypes.c_bool
    nir_lower_const_arrays_to_uniforms.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_lower_convert_alu_types = _libraries['libtinymesa_cpu.so'].nir_lower_convert_alu_types
    nir_lower_convert_alu_types.restype = ctypes.c_bool
    nir_lower_convert_alu_types.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr))]
except AttributeError:
    pass
try:
    nir_lower_constant_convert_alu_types = _libraries['libtinymesa_cpu.so'].nir_lower_constant_convert_alu_types
    nir_lower_constant_convert_alu_types.restype = ctypes.c_bool
    nir_lower_constant_convert_alu_types.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_alu_conversion_to_intrinsic = _libraries['libtinymesa_cpu.so'].nir_lower_alu_conversion_to_intrinsic
    nir_lower_alu_conversion_to_intrinsic.restype = ctypes.c_bool
    nir_lower_alu_conversion_to_intrinsic.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_int_to_float = _libraries['libtinymesa_cpu.so'].nir_lower_int_to_float
    nir_lower_int_to_float.restype = ctypes.c_bool
    nir_lower_int_to_float.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_load_const_to_scalar = _libraries['libtinymesa_cpu.so'].nir_lower_load_const_to_scalar
    nir_lower_load_const_to_scalar.restype = ctypes.c_bool
    nir_lower_load_const_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_read_invocation_to_scalar = _libraries['FIXME_STUB'].nir_lower_read_invocation_to_scalar
    nir_lower_read_invocation_to_scalar.restype = ctypes.c_bool
    nir_lower_read_invocation_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_phis_to_scalar = _libraries['libtinymesa_cpu.so'].nir_lower_phis_to_scalar
    nir_lower_phis_to_scalar.restype = ctypes.c_bool
    nir_lower_phis_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader), nir_vectorize_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_all_phis_to_scalar = _libraries['libtinymesa_cpu.so'].nir_lower_all_phis_to_scalar
    nir_lower_all_phis_to_scalar.restype = ctypes.c_bool
    nir_lower_all_phis_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_io_array_vars_to_elements = _libraries['libtinymesa_cpu.so'].nir_lower_io_array_vars_to_elements
    nir_lower_io_array_vars_to_elements.restype = None
    nir_lower_io_array_vars_to_elements.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_io_array_vars_to_elements_no_indirects = _libraries['libtinymesa_cpu.so'].nir_lower_io_array_vars_to_elements_no_indirects
    nir_lower_io_array_vars_to_elements_no_indirects.restype = ctypes.c_bool
    nir_lower_io_array_vars_to_elements_no_indirects.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_io_to_scalar = _libraries['libtinymesa_cpu.so'].nir_lower_io_to_scalar
    nir_lower_io_to_scalar.restype = ctypes.c_bool
    nir_lower_io_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, nir_instr_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_io_vars_to_scalar = _libraries['libtinymesa_cpu.so'].nir_lower_io_vars_to_scalar
    nir_lower_io_vars_to_scalar.restype = ctypes.c_bool
    nir_lower_io_vars_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_opt_vectorize_io_vars = _libraries['libtinymesa_cpu.so'].nir_opt_vectorize_io_vars
    nir_opt_vectorize_io_vars.restype = ctypes.c_bool
    nir_opt_vectorize_io_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_tess_level_array_vars_to_vec = _libraries['libtinymesa_cpu.so'].nir_lower_tess_level_array_vars_to_vec
    nir_lower_tess_level_array_vars_to_vec.restype = ctypes.c_bool
    nir_lower_tess_level_array_vars_to_vec.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_create_passthrough_tcs_impl = _libraries['libtinymesa_cpu.so'].nir_create_passthrough_tcs_impl
    nir_create_passthrough_tcs_impl.restype = ctypes.POINTER(struct_nir_shader)
    nir_create_passthrough_tcs_impl.argtypes = [ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, uint8_t]
except AttributeError:
    pass
try:
    nir_create_passthrough_tcs = _libraries['libtinymesa_cpu.so'].nir_create_passthrough_tcs
    nir_create_passthrough_tcs.restype = ctypes.POINTER(struct_nir_shader)
    nir_create_passthrough_tcs.argtypes = [ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_nir_shader), uint8_t]
except AttributeError:
    pass
try:
    nir_create_passthrough_gs = _libraries['libtinymesa_cpu.so'].nir_create_passthrough_gs
    nir_create_passthrough_gs.restype = ctypes.POINTER(struct_nir_shader)
    nir_create_passthrough_gs.argtypes = [ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_nir_shader), mesa_prim, mesa_prim, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_fragcolor = _libraries['libtinymesa_cpu.so'].nir_lower_fragcolor
    nir_lower_fragcolor.restype = ctypes.c_bool
    nir_lower_fragcolor.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_lower_fragcoord_wtrans = _libraries['libtinymesa_cpu.so'].nir_lower_fragcoord_wtrans
    nir_lower_fragcoord_wtrans.restype = ctypes.c_bool
    nir_lower_fragcoord_wtrans.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_frag_coord_to_pixel_coord = _libraries['libtinymesa_cpu.so'].nir_opt_frag_coord_to_pixel_coord
    nir_opt_frag_coord_to_pixel_coord.restype = ctypes.c_bool
    nir_opt_frag_coord_to_pixel_coord.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_frag_coord_to_pixel_coord = _libraries['libtinymesa_cpu.so'].nir_lower_frag_coord_to_pixel_coord
    nir_lower_frag_coord_to_pixel_coord.restype = ctypes.c_bool
    nir_lower_frag_coord_to_pixel_coord.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_viewport_transform = _libraries['libtinymesa_cpu.so'].nir_lower_viewport_transform
    nir_lower_viewport_transform.restype = ctypes.c_bool
    nir_lower_viewport_transform.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_uniforms_to_ubo = _libraries['libtinymesa_cpu.so'].nir_lower_uniforms_to_ubo
    nir_lower_uniforms_to_ubo.restype = ctypes.c_bool
    nir_lower_uniforms_to_ubo.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_is_helper_invocation = _libraries['libtinymesa_cpu.so'].nir_lower_is_helper_invocation
    nir_lower_is_helper_invocation.restype = ctypes.c_bool
    nir_lower_is_helper_invocation.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_single_sampled = _libraries['libtinymesa_cpu.so'].nir_lower_single_sampled
    nir_lower_single_sampled.restype = ctypes.c_bool
    nir_lower_single_sampled.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_atomics = _libraries['libtinymesa_cpu.so'].nir_lower_atomics
    nir_lower_atomics.restype = ctypes.c_bool
    nir_lower_atomics.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_filter_cb]
except AttributeError:
    pass
class struct_nir_lower_subgroups_options(Structure):
    pass

struct_nir_lower_subgroups_options._pack_ = 1 # source:False
struct_nir_lower_subgroups_options._fields_ = [
    ('filter', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('filter_data', ctypes.POINTER(None)),
    ('subgroup_size', ctypes.c_ubyte),
    ('ballot_bit_size', ctypes.c_ubyte),
    ('ballot_components', ctypes.c_ubyte),
    ('lower_to_scalar', ctypes.c_bool, 1),
    ('lower_vote_trivial', ctypes.c_bool, 1),
    ('lower_vote_feq', ctypes.c_bool, 1),
    ('lower_vote_ieq', ctypes.c_bool, 1),
    ('lower_vote_bool_eq', ctypes.c_bool, 1),
    ('lower_first_invocation_to_ballot', ctypes.c_bool, 1),
    ('lower_read_first_invocation', ctypes.c_bool, 1),
    ('lower_subgroup_masks', ctypes.c_bool, 1),
    ('lower_relative_shuffle', ctypes.c_bool, 1),
    ('lower_shuffle_to_32bit', ctypes.c_bool, 1),
    ('lower_shuffle_to_swizzle_amd', ctypes.c_bool, 1),
    ('lower_shuffle', ctypes.c_bool, 1),
    ('lower_quad', ctypes.c_bool, 1),
    ('lower_quad_broadcast_dynamic', ctypes.c_bool, 1),
    ('lower_quad_broadcast_dynamic_to_const', ctypes.c_bool, 1),
    ('lower_quad_vote', ctypes.c_bool, 1),
    ('lower_elect', ctypes.c_bool, 1),
    ('lower_read_invocation_to_cond', ctypes.c_bool, 1),
    ('lower_rotate_to_shuffle', ctypes.c_bool, 1),
    ('lower_rotate_clustered_to_shuffle', ctypes.c_bool, 1),
    ('lower_ballot_bit_count_to_mbcnt_amd', ctypes.c_bool, 1),
    ('lower_inverse_ballot', ctypes.c_bool, 1),
    ('lower_reduce', ctypes.c_bool, 1),
    ('lower_boolean_reduce', ctypes.c_bool, 1),
    ('lower_boolean_shuffle', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint16, 15),
]

nir_lower_subgroups_options = struct_nir_lower_subgroups_options
try:
    nir_lower_subgroups = _libraries['libtinymesa_cpu.so'].nir_lower_subgroups
    nir_lower_subgroups.restype = ctypes.c_bool
    nir_lower_subgroups.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_subgroups_options)]
except AttributeError:
    pass
try:
    nir_lower_system_values = _libraries['libtinymesa_cpu.so'].nir_lower_system_values
    nir_lower_system_values.restype = ctypes.c_bool
    nir_lower_system_values.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_build_lowered_load_helper_invocation = _libraries['libtinymesa_cpu.so'].nir_build_lowered_load_helper_invocation
    nir_build_lowered_load_helper_invocation.restype = ctypes.POINTER(struct_nir_def)
    nir_build_lowered_load_helper_invocation.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
class struct_nir_lower_compute_system_values_options(Structure):
    pass

struct_nir_lower_compute_system_values_options._pack_ = 1 # source:False
struct_nir_lower_compute_system_values_options._fields_ = [
    ('has_base_global_invocation_id', ctypes.c_bool, 1),
    ('has_base_workgroup_id', ctypes.c_bool, 1),
    ('has_global_size', ctypes.c_bool, 1),
    ('shuffle_local_ids_for_quad_derivatives', ctypes.c_bool, 1),
    ('lower_local_invocation_index', ctypes.c_bool, 1),
    ('lower_cs_local_id_to_index', ctypes.c_bool, 1),
    ('lower_workgroup_id_to_index', ctypes.c_bool, 1),
    ('global_id_is_32bit', ctypes.c_bool, 1),
    ('shortcut_1d_workgroup_id', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint32, 23),
    ('num_workgroups', ctypes.c_uint32 * 3),
]

nir_lower_compute_system_values_options = struct_nir_lower_compute_system_values_options
try:
    nir_lower_compute_system_values = _libraries['libtinymesa_cpu.so'].nir_lower_compute_system_values
    nir_lower_compute_system_values.restype = ctypes.c_bool
    nir_lower_compute_system_values.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_compute_system_values_options)]
except AttributeError:
    pass
class struct_nir_lower_sysvals_to_varyings_options(Structure):
    pass

struct_nir_lower_sysvals_to_varyings_options._pack_ = 1 # source:False
struct_nir_lower_sysvals_to_varyings_options._fields_ = [
    ('frag_coord', ctypes.c_bool, 1),
    ('front_face', ctypes.c_bool, 1),
    ('point_coord', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint8, 5),
]

nir_lower_sysvals_to_varyings_options = struct_nir_lower_sysvals_to_varyings_options
try:
    nir_lower_sysvals_to_varyings = _libraries['libtinymesa_cpu.so'].nir_lower_sysvals_to_varyings
    nir_lower_sysvals_to_varyings.restype = ctypes.c_bool
    nir_lower_sysvals_to_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_sysvals_to_varyings_options)]
except AttributeError:
    pass

# values for enumeration 'nir_lower_tex_packing'
nir_lower_tex_packing__enumvalues = {
    0: 'nir_lower_tex_packing_none',
    1: 'nir_lower_tex_packing_16',
    2: 'nir_lower_tex_packing_8',
}
nir_lower_tex_packing_none = 0
nir_lower_tex_packing_16 = 1
nir_lower_tex_packing_8 = 2
nir_lower_tex_packing = ctypes.c_uint32 # enum
class struct_nir_lower_tex_options(Structure):
    pass

struct_nir_lower_tex_options._pack_ = 1 # source:False
struct_nir_lower_tex_options._fields_ = [
    ('lower_txp', ctypes.c_uint32),
    ('lower_txp_array', ctypes.c_bool),
    ('lower_txf_offset', ctypes.c_bool),
    ('lower_rect_offset', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('lower_offset_filter', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('lower_rect', ctypes.c_bool),
    ('lower_1d', ctypes.c_bool),
    ('lower_1d_shadow', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte),
    ('lower_y_uv_external', ctypes.c_uint32),
    ('lower_y_vu_external', ctypes.c_uint32),
    ('lower_y_u_v_external', ctypes.c_uint32),
    ('lower_yx_xuxv_external', ctypes.c_uint32),
    ('lower_yx_xvxu_external', ctypes.c_uint32),
    ('lower_xy_uxvx_external', ctypes.c_uint32),
    ('lower_xy_vxux_external', ctypes.c_uint32),
    ('lower_ayuv_external', ctypes.c_uint32),
    ('lower_xyuv_external', ctypes.c_uint32),
    ('lower_yuv_external', ctypes.c_uint32),
    ('lower_yu_yv_external', ctypes.c_uint32),
    ('lower_yv_yu_external', ctypes.c_uint32),
    ('lower_y41x_external', ctypes.c_uint32),
    ('lower_sx10_external', ctypes.c_uint32),
    ('lower_sx12_external', ctypes.c_uint32),
    ('bt709_external', ctypes.c_uint32),
    ('bt2020_external', ctypes.c_uint32),
    ('yuv_full_range_external', ctypes.c_uint32),
    ('saturate_s', ctypes.c_uint32),
    ('saturate_t', ctypes.c_uint32),
    ('saturate_r', ctypes.c_uint32),
    ('swizzle_result', ctypes.c_uint32),
    ('swizzles', ctypes.c_ubyte * 4 * 32),
    ('scale_factors', ctypes.c_float * 32),
    ('lower_srgb', ctypes.c_uint32),
    ('lower_txd_cube_map', ctypes.c_bool),
    ('lower_txd_3d', ctypes.c_bool),
    ('lower_txd_array', ctypes.c_bool),
    ('lower_txd_shadow', ctypes.c_bool),
    ('lower_txd', ctypes.c_bool),
    ('lower_txd_clamp', ctypes.c_bool),
    ('lower_txb_shadow_clamp', ctypes.c_bool),
    ('lower_txd_shadow_clamp', ctypes.c_bool),
    ('lower_txd_offset_clamp', ctypes.c_bool),
    ('lower_txd_clamp_bindless_sampler', ctypes.c_bool),
    ('lower_txd_clamp_if_sampler_index_not_lt_16', ctypes.c_bool),
    ('lower_txs_lod', ctypes.c_bool),
    ('lower_txs_cube_array', ctypes.c_bool),
    ('lower_tg4_broadcom_swizzle', ctypes.c_bool),
    ('lower_tg4_offsets', ctypes.c_bool),
    ('lower_to_fragment_fetch_amd', ctypes.c_bool),
    ('lower_tex_packing_cb', ctypes.CFUNCTYPE(nir_lower_tex_packing, ctypes.POINTER(struct_nir_tex_instr), ctypes.POINTER(None))),
    ('lower_tex_packing_data', ctypes.POINTER(None)),
    ('lower_lod_zero_width', ctypes.c_bool),
    ('lower_sampler_lod_bias', ctypes.c_bool),
    ('lower_invalid_implicit_lod', ctypes.c_bool),
    ('lower_index_to_offset', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('callback_data', ctypes.POINTER(None)),
]

nir_lower_tex_options = struct_nir_lower_tex_options
try:
    nir_lower_tex = _libraries['libtinymesa_cpu.so'].nir_lower_tex
    nir_lower_tex.restype = ctypes.c_bool
    nir_lower_tex.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_tex_options)]
except AttributeError:
    pass
class struct_nir_lower_tex_shadow_swizzle(Structure):
    pass

struct_nir_lower_tex_shadow_swizzle._pack_ = 1 # source:False
struct_nir_lower_tex_shadow_swizzle._fields_ = [
    ('swizzle_r', ctypes.c_uint32, 3),
    ('swizzle_g', ctypes.c_uint32, 3),
    ('swizzle_b', ctypes.c_uint32, 3),
    ('swizzle_a', ctypes.c_uint32, 3),
    ('PADDING_0', ctypes.c_uint32, 20),
]

nir_lower_tex_shadow_swizzle = struct_nir_lower_tex_shadow_swizzle
try:
    nir_lower_tex_shadow = _libraries['libtinymesa_cpu.so'].nir_lower_tex_shadow
    nir_lower_tex_shadow.restype = ctypes.c_bool
    nir_lower_tex_shadow.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.POINTER(compare_func), ctypes.POINTER(struct_nir_lower_tex_shadow_swizzle), ctypes.c_bool]
except AttributeError:
    pass
class struct_nir_lower_image_options(Structure):
    pass

struct_nir_lower_image_options._pack_ = 1 # source:False
struct_nir_lower_image_options._fields_ = [
    ('lower_cube_size', ctypes.c_bool),
    ('lower_to_fragment_mask_load_amd', ctypes.c_bool),
    ('lower_image_samples_to_one', ctypes.c_bool),
]

nir_lower_image_options = struct_nir_lower_image_options
try:
    nir_lower_image = _libraries['libtinymesa_cpu.so'].nir_lower_image
    nir_lower_image.restype = ctypes.c_bool
    nir_lower_image.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_image_options)]
except AttributeError:
    pass
try:
    nir_lower_image_atomics_to_global = _libraries['libtinymesa_cpu.so'].nir_lower_image_atomics_to_global
    nir_lower_image_atomics_to_global.restype = ctypes.c_bool
    nir_lower_image_atomics_to_global.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrin_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_readonly_images_to_tex = _libraries['libtinymesa_cpu.so'].nir_lower_readonly_images_to_tex
    nir_lower_readonly_images_to_tex.restype = ctypes.c_bool
    nir_lower_readonly_images_to_tex.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass

# values for enumeration 'nir_lower_non_uniform_access_type'
nir_lower_non_uniform_access_type__enumvalues = {
    1: 'nir_lower_non_uniform_ubo_access',
    2: 'nir_lower_non_uniform_ssbo_access',
    4: 'nir_lower_non_uniform_texture_access',
    8: 'nir_lower_non_uniform_image_access',
    16: 'nir_lower_non_uniform_get_ssbo_size',
    32: 'nir_lower_non_uniform_texture_offset_access',
    6: 'nir_lower_non_uniform_access_type_count',
}
nir_lower_non_uniform_ubo_access = 1
nir_lower_non_uniform_ssbo_access = 2
nir_lower_non_uniform_texture_access = 4
nir_lower_non_uniform_image_access = 8
nir_lower_non_uniform_get_ssbo_size = 16
nir_lower_non_uniform_texture_offset_access = 32
nir_lower_non_uniform_access_type_count = 6
nir_lower_non_uniform_access_type = ctypes.c_uint32 # enum
nir_lower_non_uniform_src_access_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32, ctypes.POINTER(None))
nir_lower_non_uniform_access_callback = ctypes.CFUNCTYPE(ctypes.c_uint16, ctypes.POINTER(struct_nir_src), ctypes.POINTER(None))
class struct_nir_lower_non_uniform_access_options(Structure):
    pass

struct_nir_lower_non_uniform_access_options._pack_ = 1 # source:False
struct_nir_lower_non_uniform_access_options._fields_ = [
    ('types', nir_lower_non_uniform_access_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('tex_src_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32, ctypes.POINTER(None))),
    ('callback', ctypes.CFUNCTYPE(ctypes.c_uint16, ctypes.POINTER(struct_nir_src), ctypes.POINTER(None))),
    ('callback_data', ctypes.POINTER(None)),
]

nir_lower_non_uniform_access_options = struct_nir_lower_non_uniform_access_options
try:
    nir_has_non_uniform_access = _libraries['libtinymesa_cpu.so'].nir_has_non_uniform_access
    nir_has_non_uniform_access.restype = ctypes.c_bool
    nir_has_non_uniform_access.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_non_uniform_access_type]
except AttributeError:
    pass
try:
    nir_opt_non_uniform_access = _libraries['libtinymesa_cpu.so'].nir_opt_non_uniform_access
    nir_opt_non_uniform_access.restype = ctypes.c_bool
    nir_opt_non_uniform_access.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_non_uniform_access = _libraries['libtinymesa_cpu.so'].nir_lower_non_uniform_access
    nir_lower_non_uniform_access.restype = ctypes.c_bool
    nir_lower_non_uniform_access.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_non_uniform_access_options)]
except AttributeError:
    pass
class struct_nir_lower_idiv_options(Structure):
    pass

struct_nir_lower_idiv_options._pack_ = 1 # source:False
struct_nir_lower_idiv_options._fields_ = [
    ('allow_fp16', ctypes.c_bool),
]

nir_lower_idiv_options = struct_nir_lower_idiv_options
try:
    nir_lower_idiv = _libraries['libtinymesa_cpu.so'].nir_lower_idiv
    nir_lower_idiv.restype = ctypes.c_bool
    nir_lower_idiv.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_idiv_options)]
except AttributeError:
    pass
class struct_nir_input_attachment_options(Structure):
    pass

struct_nir_input_attachment_options._pack_ = 1 # source:False
struct_nir_input_attachment_options._fields_ = [
    ('use_ia_coord_intrin', ctypes.c_bool),
    ('use_fragcoord_sysval', ctypes.c_bool),
    ('use_layer_id_sysval', ctypes.c_bool),
    ('use_view_id_for_layer', ctypes.c_bool),
    ('unscaled_depth_stencil_ir3', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('unscaled_input_attachment_ir3', ctypes.c_uint32),
]

nir_input_attachment_options = struct_nir_input_attachment_options
try:
    nir_lower_input_attachments = _libraries['libtinymesa_cpu.so'].nir_lower_input_attachments
    nir_lower_input_attachments.restype = ctypes.c_bool
    nir_lower_input_attachments.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_input_attachment_options)]
except AttributeError:
    pass
try:
    nir_lower_clip_vs = _libraries['libtinymesa_cpu.so'].nir_lower_clip_vs
    nir_lower_clip_vs.restype = ctypes.c_bool
    nir_lower_clip_vs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, ctypes.c_int16 * 4 * 0]
except AttributeError:
    pass
try:
    nir_lower_clip_gs = _libraries['libtinymesa_cpu.so'].nir_lower_clip_gs
    nir_lower_clip_gs.restype = ctypes.c_bool
    nir_lower_clip_gs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_int16 * 4 * 0]
except AttributeError:
    pass
try:
    nir_lower_clip_fs = _libraries['libtinymesa_cpu.so'].nir_lower_clip_fs
    nir_lower_clip_fs.restype = ctypes.c_bool
    nir_lower_clip_fs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_clip_cull_distance_to_vec4s = _libraries['libtinymesa_cpu.so'].nir_lower_clip_cull_distance_to_vec4s
    nir_lower_clip_cull_distance_to_vec4s.restype = ctypes.c_bool
    nir_lower_clip_cull_distance_to_vec4s.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_clip_cull_distance_array_vars = _libraries['libtinymesa_cpu.so'].nir_lower_clip_cull_distance_array_vars
    nir_lower_clip_cull_distance_array_vars.restype = ctypes.c_bool
    nir_lower_clip_cull_distance_array_vars.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_clip_disable = _libraries['libtinymesa_cpu.so'].nir_lower_clip_disable
    nir_lower_clip_disable.restype = ctypes.c_bool
    nir_lower_clip_disable.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_lower_point_size_mov = _libraries['libtinymesa_cpu.so'].nir_lower_point_size_mov
    nir_lower_point_size_mov.restype = ctypes.c_bool
    nir_lower_point_size_mov.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_int16)]
except AttributeError:
    pass
try:
    nir_lower_frexp = _libraries['libtinymesa_cpu.so'].nir_lower_frexp
    nir_lower_frexp.restype = ctypes.c_bool
    nir_lower_frexp.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_two_sided_color = _libraries['libtinymesa_cpu.so'].nir_lower_two_sided_color
    nir_lower_two_sided_color.restype = ctypes.c_bool
    nir_lower_two_sided_color.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_clamp_color_outputs = _libraries['libtinymesa_cpu.so'].nir_lower_clamp_color_outputs
    nir_lower_clamp_color_outputs.restype = ctypes.c_bool
    nir_lower_clamp_color_outputs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_flatshade = _libraries['libtinymesa_cpu.so'].nir_lower_flatshade
    nir_lower_flatshade.restype = ctypes.c_bool
    nir_lower_flatshade.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_passthrough_edgeflags = _libraries['libtinymesa_cpu.so'].nir_lower_passthrough_edgeflags
    nir_lower_passthrough_edgeflags.restype = ctypes.c_bool
    nir_lower_passthrough_edgeflags.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_patch_vertices = _libraries['libtinymesa_cpu.so'].nir_lower_patch_vertices
    nir_lower_patch_vertices.restype = ctypes.c_bool
    nir_lower_patch_vertices.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.POINTER(ctypes.c_int16)]
except AttributeError:
    pass
class struct_nir_lower_wpos_ytransform_options(Structure):
    pass

struct_nir_lower_wpos_ytransform_options._pack_ = 1 # source:False
struct_nir_lower_wpos_ytransform_options._fields_ = [
    ('state_tokens', ctypes.c_int16 * 4),
    ('fs_coord_origin_upper_left', ctypes.c_bool, 1),
    ('fs_coord_origin_lower_left', ctypes.c_bool, 1),
    ('fs_coord_pixel_center_integer', ctypes.c_bool, 1),
    ('fs_coord_pixel_center_half_integer', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint16, 12),
]

nir_lower_wpos_ytransform_options = struct_nir_lower_wpos_ytransform_options
try:
    nir_lower_wpos_ytransform = _libraries['libtinymesa_cpu.so'].nir_lower_wpos_ytransform
    nir_lower_wpos_ytransform.restype = ctypes.c_bool
    nir_lower_wpos_ytransform.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_wpos_ytransform_options)]
except AttributeError:
    pass
try:
    nir_lower_wpos_center = _libraries['libtinymesa_cpu.so'].nir_lower_wpos_center
    nir_lower_wpos_center.restype = ctypes.c_bool
    nir_lower_wpos_center.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_pntc_ytransform = _libraries['libtinymesa_cpu.so'].nir_lower_pntc_ytransform
    nir_lower_pntc_ytransform.restype = ctypes.c_bool
    nir_lower_pntc_ytransform.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_int16 * 4 * 0]
except AttributeError:
    pass
try:
    nir_lower_wrmasks = _libraries['libtinymesa_cpu.so'].nir_lower_wrmasks
    nir_lower_wrmasks.restype = ctypes.c_bool
    nir_lower_wrmasks.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_fb_read = _libraries['libtinymesa_cpu.so'].nir_lower_fb_read
    nir_lower_fb_read.restype = ctypes.c_bool
    nir_lower_fb_read.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_lower_drawpixels_options(Structure):
    pass

struct_nir_lower_drawpixels_options._pack_ = 1 # source:False
struct_nir_lower_drawpixels_options._fields_ = [
    ('texcoord_state_tokens', ctypes.c_int16 * 4),
    ('scale_state_tokens', ctypes.c_int16 * 4),
    ('bias_state_tokens', ctypes.c_int16 * 4),
    ('drawpix_sampler', ctypes.c_uint32),
    ('pixelmap_sampler', ctypes.c_uint32),
    ('pixel_maps', ctypes.c_bool, 1),
    ('scale_and_bias', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint32, 30),
]

nir_lower_drawpixels_options = struct_nir_lower_drawpixels_options
try:
    nir_lower_drawpixels = _libraries['libtinymesa_cpu.so'].nir_lower_drawpixels
    nir_lower_drawpixels.restype = ctypes.c_bool
    nir_lower_drawpixels.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_drawpixels_options)]
except AttributeError:
    pass
class struct_nir_lower_bitmap_options(Structure):
    pass

struct_nir_lower_bitmap_options._pack_ = 1 # source:False
struct_nir_lower_bitmap_options._fields_ = [
    ('sampler', ctypes.c_uint32),
    ('swizzle_xxxx', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

nir_lower_bitmap_options = struct_nir_lower_bitmap_options
try:
    nir_lower_bitmap = _libraries['libtinymesa_cpu.so'].nir_lower_bitmap
    nir_lower_bitmap.restype = ctypes.c_bool
    nir_lower_bitmap.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_bitmap_options)]
except AttributeError:
    pass
try:
    nir_lower_atomics_to_ssbo = _libraries['libtinymesa_cpu.so'].nir_lower_atomics_to_ssbo
    nir_lower_atomics_to_ssbo.restype = ctypes.c_bool
    nir_lower_atomics_to_ssbo.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_gs_intrinsics_flags'
c__EA_nir_lower_gs_intrinsics_flags__enumvalues = {
    1: 'nir_lower_gs_intrinsics_per_stream',
    2: 'nir_lower_gs_intrinsics_count_primitives',
    4: 'nir_lower_gs_intrinsics_count_vertices_per_primitive',
    8: 'nir_lower_gs_intrinsics_overwrite_incomplete',
}
nir_lower_gs_intrinsics_per_stream = 1
nir_lower_gs_intrinsics_count_primitives = 2
nir_lower_gs_intrinsics_count_vertices_per_primitive = 4
nir_lower_gs_intrinsics_overwrite_incomplete = 8
c__EA_nir_lower_gs_intrinsics_flags = ctypes.c_uint32 # enum
nir_lower_gs_intrinsics_flags = c__EA_nir_lower_gs_intrinsics_flags
nir_lower_gs_intrinsics_flags__enumvalues = c__EA_nir_lower_gs_intrinsics_flags__enumvalues
try:
    nir_lower_gs_intrinsics = _libraries['libtinymesa_cpu.so'].nir_lower_gs_intrinsics
    nir_lower_gs_intrinsics.restype = ctypes.c_bool
    nir_lower_gs_intrinsics.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_gs_intrinsics_flags]
except AttributeError:
    pass
try:
    nir_lower_halt_to_return = _libraries['libtinymesa_cpu.so'].nir_lower_halt_to_return
    nir_lower_halt_to_return.restype = ctypes.c_bool
    nir_lower_halt_to_return.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_tess_coord_z = _libraries['libtinymesa_cpu.so'].nir_lower_tess_coord_z
    nir_lower_tess_coord_z.restype = ctypes.c_bool
    nir_lower_tess_coord_z.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
class struct_nir_lower_task_shader_options(Structure):
    pass

struct_nir_lower_task_shader_options._pack_ = 1 # source:False
struct_nir_lower_task_shader_options._fields_ = [
    ('payload_to_shared_for_atomics', ctypes.c_bool, 1),
    ('payload_to_shared_for_small_types', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint32, 30),
    ('payload_offset_in_bytes', ctypes.c_uint32),
]

nir_lower_task_shader_options = struct_nir_lower_task_shader_options
try:
    nir_lower_task_shader = _libraries['libtinymesa_cpu.so'].nir_lower_task_shader
    nir_lower_task_shader.restype = ctypes.c_bool
    nir_lower_task_shader.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_task_shader_options]
except AttributeError:
    pass
nir_lower_bit_size_callback = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
try:
    nir_lower_bit_size = _libraries['libtinymesa_cpu.so'].nir_lower_bit_size
    nir_lower_bit_size.restype = ctypes.c_bool
    nir_lower_bit_size.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_bit_size_callback, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_64bit_phis = _libraries['libtinymesa_cpu.so'].nir_lower_64bit_phis
    nir_lower_64bit_phis.restype = ctypes.c_bool
    nir_lower_64bit_phis.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_split_conversions_options(Structure):
    pass

struct_nir_split_conversions_options._pack_ = 1 # source:False
struct_nir_split_conversions_options._fields_ = [
    ('callback', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('callback_data', ctypes.POINTER(None)),
    ('has_convert_alu_types', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

nir_split_conversions_options = struct_nir_split_conversions_options
try:
    nir_split_conversions = _libraries['libtinymesa_cpu.so'].nir_split_conversions
    nir_split_conversions.restype = ctypes.c_bool
    nir_split_conversions.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_split_conversions_options)]
except AttributeError:
    pass
try:
    nir_split_64bit_vec3_and_vec4 = _libraries['libtinymesa_cpu.so'].nir_split_64bit_vec3_and_vec4
    nir_split_64bit_vec3_and_vec4.restype = ctypes.c_bool
    nir_split_64bit_vec3_and_vec4.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_int64_op_to_options_mask = _libraries['libtinymesa_cpu.so'].nir_lower_int64_op_to_options_mask
    nir_lower_int64_op_to_options_mask.restype = nir_lower_int64_options
    nir_lower_int64_op_to_options_mask.argtypes = [nir_op]
except AttributeError:
    pass
try:
    nir_lower_int64 = _libraries['libtinymesa_cpu.so'].nir_lower_int64
    nir_lower_int64.restype = ctypes.c_bool
    nir_lower_int64.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_int64_float_conversions = _libraries['libtinymesa_cpu.so'].nir_lower_int64_float_conversions
    nir_lower_int64_float_conversions.restype = ctypes.c_bool
    nir_lower_int64_float_conversions.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_doubles_op_to_options_mask = _libraries['libtinymesa_cpu.so'].nir_lower_doubles_op_to_options_mask
    nir_lower_doubles_op_to_options_mask.restype = nir_lower_doubles_options
    nir_lower_doubles_op_to_options_mask.argtypes = [nir_op]
except AttributeError:
    pass
try:
    nir_lower_doubles = _libraries['libtinymesa_cpu.so'].nir_lower_doubles
    nir_lower_doubles.restype = ctypes.c_bool
    nir_lower_doubles.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader), nir_lower_doubles_options]
except AttributeError:
    pass
try:
    nir_lower_pack = _libraries['libtinymesa_cpu.so'].nir_lower_pack
    nir_lower_pack.restype = ctypes.c_bool
    nir_lower_pack.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_get_io_intrinsic = _libraries['libtinymesa_cpu.so'].nir_get_io_intrinsic
    nir_get_io_intrinsic.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_get_io_intrinsic.argtypes = [ctypes.POINTER(struct_nir_instr), nir_variable_mode, ctypes.POINTER(c__EA_nir_variable_mode)]
except AttributeError:
    pass
try:
    nir_recompute_io_bases = _libraries['libtinymesa_cpu.so'].nir_recompute_io_bases
    nir_recompute_io_bases.restype = ctypes.c_bool
    nir_recompute_io_bases.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_mediump_vars = _libraries['libtinymesa_cpu.so'].nir_lower_mediump_vars
    nir_lower_mediump_vars.restype = ctypes.c_bool
    nir_lower_mediump_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_mediump_io = _libraries['libtinymesa_cpu.so'].nir_lower_mediump_io
    nir_lower_mediump_io.restype = ctypes.c_bool
    nir_lower_mediump_io.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, uint64_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_clear_mediump_io_flag = _libraries['libtinymesa_cpu.so'].nir_clear_mediump_io_flag
    nir_clear_mediump_io_flag.restype = ctypes.c_bool
    nir_clear_mediump_io_flag.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_opt_tex_srcs_options(Structure):
    pass

struct_nir_opt_tex_srcs_options._pack_ = 1 # source:False
struct_nir_opt_tex_srcs_options._fields_ = [
    ('sampler_dims', ctypes.c_uint32),
    ('src_types', ctypes.c_uint32),
]

nir_opt_tex_srcs_options = struct_nir_opt_tex_srcs_options
class struct_nir_opt_16bit_tex_image_options(Structure):
    pass

struct_nir_opt_16bit_tex_image_options._pack_ = 1 # source:False
struct_nir_opt_16bit_tex_image_options._fields_ = [
    ('rounding_mode', nir_rounding_mode),
    ('opt_tex_dest_types', nir_alu_type),
    ('opt_image_dest_types', nir_alu_type),
    ('integer_dest_saturates', ctypes.c_bool),
    ('opt_image_store_data', ctypes.c_bool),
    ('opt_image_srcs', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('opt_srcs_options_count', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('opt_srcs_options', ctypes.POINTER(struct_nir_opt_tex_srcs_options)),
]

nir_opt_16bit_tex_image_options = struct_nir_opt_16bit_tex_image_options
try:
    nir_opt_16bit_tex_image = _libraries['libtinymesa_cpu.so'].nir_opt_16bit_tex_image
    nir_opt_16bit_tex_image.restype = ctypes.c_bool
    nir_opt_16bit_tex_image.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_16bit_tex_image_options)]
except AttributeError:
    pass
class struct_nir_tex_src_type_constraint(Structure):
    pass

struct_nir_tex_src_type_constraint._pack_ = 1 # source:False
struct_nir_tex_src_type_constraint._fields_ = [
    ('legalize_type', ctypes.c_bool),
    ('bit_size', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('match_src', nir_tex_src_type),
]

nir_tex_src_type_constraint = struct_nir_tex_src_type_constraint
nir_tex_src_type_constraints = struct_nir_tex_src_type_constraint * 23
try:
    nir_legalize_16bit_sampler_srcs = _libraries['libtinymesa_cpu.so'].nir_legalize_16bit_sampler_srcs
    nir_legalize_16bit_sampler_srcs.restype = ctypes.c_bool
    nir_legalize_16bit_sampler_srcs.argtypes = [ctypes.POINTER(struct_nir_shader), nir_tex_src_type_constraints]
except AttributeError:
    pass
try:
    nir_lower_point_size = _libraries['libtinymesa_cpu.so'].nir_lower_point_size
    nir_lower_point_size.restype = ctypes.c_bool
    nir_lower_point_size.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_lower_default_point_size = _libraries['libtinymesa_cpu.so'].nir_lower_default_point_size
    nir_lower_default_point_size.restype = ctypes.c_bool
    nir_lower_default_point_size.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_texcoord_replace = _libraries['libtinymesa_cpu.so'].nir_lower_texcoord_replace
    nir_lower_texcoord_replace.restype = ctypes.c_bool
    nir_lower_texcoord_replace.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_texcoord_replace_late = _libraries['libtinymesa_cpu.so'].nir_lower_texcoord_replace_late
    nir_lower_texcoord_replace_late.restype = ctypes.c_bool
    nir_lower_texcoord_replace_late.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_interpolation_options'
c__EA_nir_lower_interpolation_options__enumvalues = {
    2: 'nir_lower_interpolation_at_sample',
    4: 'nir_lower_interpolation_at_offset',
    8: 'nir_lower_interpolation_centroid',
    16: 'nir_lower_interpolation_pixel',
    32: 'nir_lower_interpolation_sample',
}
nir_lower_interpolation_at_sample = 2
nir_lower_interpolation_at_offset = 4
nir_lower_interpolation_centroid = 8
nir_lower_interpolation_pixel = 16
nir_lower_interpolation_sample = 32
c__EA_nir_lower_interpolation_options = ctypes.c_uint32 # enum
nir_lower_interpolation_options = c__EA_nir_lower_interpolation_options
nir_lower_interpolation_options__enumvalues = c__EA_nir_lower_interpolation_options__enumvalues
try:
    nir_lower_interpolation = _libraries['libtinymesa_cpu.so'].nir_lower_interpolation
    nir_lower_interpolation.restype = ctypes.c_bool
    nir_lower_interpolation.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_interpolation_options]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_discard_if_options'
c__EA_nir_lower_discard_if_options__enumvalues = {
    1: 'nir_lower_demote_if_to_cf',
    2: 'nir_lower_terminate_if_to_cf',
    4: 'nir_move_terminate_out_of_loops',
}
nir_lower_demote_if_to_cf = 1
nir_lower_terminate_if_to_cf = 2
nir_move_terminate_out_of_loops = 4
c__EA_nir_lower_discard_if_options = ctypes.c_uint32 # enum
nir_lower_discard_if_options = c__EA_nir_lower_discard_if_options
nir_lower_discard_if_options__enumvalues = c__EA_nir_lower_discard_if_options__enumvalues
try:
    nir_lower_discard_if = _libraries['libtinymesa_cpu.so'].nir_lower_discard_if
    nir_lower_discard_if.restype = ctypes.c_bool
    nir_lower_discard_if.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_discard_if_options]
except AttributeError:
    pass
try:
    nir_lower_terminate_to_demote = _libraries['libtinymesa_cpu.so'].nir_lower_terminate_to_demote
    nir_lower_terminate_to_demote.restype = ctypes.c_bool
    nir_lower_terminate_to_demote.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_memory_model = _libraries['libtinymesa_cpu.so'].nir_lower_memory_model
    nir_lower_memory_model.restype = ctypes.c_bool
    nir_lower_memory_model.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_goto_ifs = _libraries['libtinymesa_cpu.so'].nir_lower_goto_ifs
    nir_lower_goto_ifs.restype = ctypes.c_bool
    nir_lower_goto_ifs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_continue_constructs = _libraries['libtinymesa_cpu.so'].nir_lower_continue_constructs
    nir_lower_continue_constructs.restype = ctypes.c_bool
    nir_lower_continue_constructs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_lower_multiview_options(Structure):
    pass

struct_nir_lower_multiview_options._pack_ = 1 # source:False
struct_nir_lower_multiview_options._fields_ = [
    ('view_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('allowed_per_view_outputs', ctypes.c_uint64),
]

nir_lower_multiview_options = struct_nir_lower_multiview_options
try:
    nir_shader_uses_view_index = _libraries['libtinymesa_cpu.so'].nir_shader_uses_view_index
    nir_shader_uses_view_index.restype = ctypes.c_bool
    nir_shader_uses_view_index.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_can_lower_multiview = _libraries['libtinymesa_cpu.so'].nir_can_lower_multiview
    nir_can_lower_multiview.restype = ctypes.c_bool
    nir_can_lower_multiview.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_multiview_options]
except AttributeError:
    pass
try:
    nir_lower_multiview = _libraries['libtinymesa_cpu.so'].nir_lower_multiview
    nir_lower_multiview.restype = ctypes.c_bool
    nir_lower_multiview.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_multiview_options]
except AttributeError:
    pass
try:
    nir_lower_view_index_to_device_index = _libraries['libtinymesa_cpu.so'].nir_lower_view_index_to_device_index
    nir_lower_view_index_to_device_index.restype = ctypes.c_bool
    nir_lower_view_index_to_device_index.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_fp16_cast_options'
c__EA_nir_lower_fp16_cast_options__enumvalues = {
    1: 'nir_lower_fp16_rtz',
    2: 'nir_lower_fp16_rtne',
    4: 'nir_lower_fp16_ru',
    8: 'nir_lower_fp16_rd',
    15: 'nir_lower_fp16_all',
    16: 'nir_lower_fp16_split_fp64',
}
nir_lower_fp16_rtz = 1
nir_lower_fp16_rtne = 2
nir_lower_fp16_ru = 4
nir_lower_fp16_rd = 8
nir_lower_fp16_all = 15
nir_lower_fp16_split_fp64 = 16
c__EA_nir_lower_fp16_cast_options = ctypes.c_uint32 # enum
nir_lower_fp16_cast_options = c__EA_nir_lower_fp16_cast_options
nir_lower_fp16_cast_options__enumvalues = c__EA_nir_lower_fp16_cast_options__enumvalues
try:
    nir_lower_fp16_casts = _libraries['libtinymesa_cpu.so'].nir_lower_fp16_casts
    nir_lower_fp16_casts.restype = ctypes.c_bool
    nir_lower_fp16_casts.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_fp16_cast_options]
except AttributeError:
    pass
try:
    nir_normalize_cubemap_coords = _libraries['libtinymesa_cpu.so'].nir_normalize_cubemap_coords
    nir_normalize_cubemap_coords.restype = ctypes.c_bool
    nir_normalize_cubemap_coords.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_supports_implicit_lod = _libraries['libtinymesa_cpu.so'].nir_shader_supports_implicit_lod
    nir_shader_supports_implicit_lod.restype = ctypes.c_bool
    nir_shader_supports_implicit_lod.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_live_defs_impl = _libraries['libtinymesa_cpu.so'].nir_live_defs_impl
    nir_live_defs_impl.restype = None
    nir_live_defs_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_get_live_defs = _libraries['libtinymesa_cpu.so'].nir_get_live_defs
    nir_get_live_defs.restype = ctypes.POINTER(ctypes.c_uint32)
    nir_get_live_defs.argtypes = [nir_cursor, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_loop_analyze_impl = _libraries['libtinymesa_cpu.so'].nir_loop_analyze_impl
    nir_loop_analyze_impl.restype = None
    nir_loop_analyze_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_variable_mode, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_defs_interfere = _libraries['libtinymesa_cpu.so'].nir_defs_interfere
    nir_defs_interfere.restype = ctypes.c_bool
    nir_defs_interfere.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_repair_ssa_impl = _libraries['libtinymesa_cpu.so'].nir_repair_ssa_impl
    nir_repair_ssa_impl.restype = ctypes.c_bool
    nir_repair_ssa_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_repair_ssa = _libraries['libtinymesa_cpu.so'].nir_repair_ssa
    nir_repair_ssa.restype = ctypes.c_bool
    nir_repair_ssa.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_convert_loop_to_lcssa = _libraries['libtinymesa_cpu.so'].nir_convert_loop_to_lcssa
    nir_convert_loop_to_lcssa.restype = None
    nir_convert_loop_to_lcssa.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_convert_to_lcssa = _libraries['libtinymesa_cpu.so'].nir_convert_to_lcssa
    nir_convert_to_lcssa.restype = ctypes.c_bool
    nir_convert_to_lcssa.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_divergence_analysis_impl = _libraries['libtinymesa_cpu.so'].nir_divergence_analysis_impl
    nir_divergence_analysis_impl.restype = None
    nir_divergence_analysis_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_divergence_options]
except AttributeError:
    pass
try:
    nir_divergence_analysis = _libraries['libtinymesa_cpu.so'].nir_divergence_analysis
    nir_divergence_analysis.restype = None
    nir_divergence_analysis.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_vertex_divergence_analysis = _libraries['libtinymesa_cpu.so'].nir_vertex_divergence_analysis
    nir_vertex_divergence_analysis.restype = None
    nir_vertex_divergence_analysis.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_has_divergent_loop = _libraries['libtinymesa_cpu.so'].nir_has_divergent_loop
    nir_has_divergent_loop.restype = ctypes.c_bool
    nir_has_divergent_loop.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_rewrite_uses_to_load_reg = _libraries['libtinymesa_cpu.so'].nir_rewrite_uses_to_load_reg
    nir_rewrite_uses_to_load_reg.restype = None
    nir_rewrite_uses_to_load_reg.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_convert_from_ssa = _libraries['libtinymesa_cpu.so'].nir_convert_from_ssa
    nir_convert_from_ssa.restype = ctypes.c_bool
    nir_convert_from_ssa.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_phis_to_regs_block = _libraries['libtinymesa_cpu.so'].nir_lower_phis_to_regs_block
    nir_lower_phis_to_regs_block.restype = ctypes.c_bool
    nir_lower_phis_to_regs_block.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_ssa_defs_to_regs_block = _libraries['libtinymesa_cpu.so'].nir_lower_ssa_defs_to_regs_block
    nir_lower_ssa_defs_to_regs_block.restype = ctypes.c_bool
    nir_lower_ssa_defs_to_regs_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_rematerialize_deref_in_use_blocks = _libraries['libtinymesa_cpu.so'].nir_rematerialize_deref_in_use_blocks
    nir_rematerialize_deref_in_use_blocks.restype = ctypes.c_bool
    nir_rematerialize_deref_in_use_blocks.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_rematerialize_derefs_in_use_blocks_impl = _libraries['libtinymesa_cpu.so'].nir_rematerialize_derefs_in_use_blocks_impl
    nir_rematerialize_derefs_in_use_blocks_impl.restype = ctypes.c_bool
    nir_rematerialize_derefs_in_use_blocks_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_lower_samplers = _libraries['libtinymesa_cpu.so'].nir_lower_samplers
    nir_lower_samplers.restype = ctypes.c_bool
    nir_lower_samplers.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_cl_images = _libraries['libtinymesa_cpu.so'].nir_lower_cl_images
    nir_lower_cl_images.restype = ctypes.c_bool
    nir_lower_cl_images.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_dedup_inline_samplers = _libraries['libtinymesa_cpu.so'].nir_dedup_inline_samplers
    nir_dedup_inline_samplers.restype = ctypes.c_bool
    nir_dedup_inline_samplers.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_lower_ssbo_options(Structure):
    pass

struct_nir_lower_ssbo_options._pack_ = 1 # source:False
struct_nir_lower_ssbo_options._fields_ = [
    ('native_loads', ctypes.c_bool),
    ('native_offset', ctypes.c_bool),
]

nir_lower_ssbo_options = struct_nir_lower_ssbo_options
try:
    nir_lower_ssbo = _libraries['libtinymesa_cpu.so'].nir_lower_ssbo
    nir_lower_ssbo.restype = ctypes.c_bool
    nir_lower_ssbo.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_ssbo_options)]
except AttributeError:
    pass
try:
    nir_lower_helper_writes = _libraries['libtinymesa_cpu.so'].nir_lower_helper_writes
    nir_lower_helper_writes.restype = ctypes.c_bool
    nir_lower_helper_writes.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
class struct_nir_lower_printf_options(Structure):
    pass

struct_nir_lower_printf_options._pack_ = 1 # source:False
struct_nir_lower_printf_options._fields_ = [
    ('max_buffer_size', ctypes.c_uint32),
    ('ptr_bit_size', ctypes.c_uint32),
    ('hash_format_strings', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

nir_lower_printf_options = struct_nir_lower_printf_options
try:
    nir_lower_printf = _libraries['libtinymesa_cpu.so'].nir_lower_printf
    nir_lower_printf.restype = ctypes.c_bool
    nir_lower_printf.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_printf_options)]
except AttributeError:
    pass
try:
    nir_lower_printf_buffer = _libraries['libtinymesa_cpu.so'].nir_lower_printf_buffer
    nir_lower_printf_buffer.restype = ctypes.c_bool
    nir_lower_printf_buffer.argtypes = [ctypes.POINTER(struct_nir_shader), uint64_t, uint32_t]
except AttributeError:
    pass
try:
    nir_opt_comparison_pre_impl = _libraries['libtinymesa_cpu.so'].nir_opt_comparison_pre_impl
    nir_opt_comparison_pre_impl.restype = ctypes.c_bool
    nir_opt_comparison_pre_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_opt_comparison_pre = _libraries['libtinymesa_cpu.so'].nir_opt_comparison_pre
    nir_opt_comparison_pre.restype = ctypes.c_bool
    nir_opt_comparison_pre.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_opt_access_options(Structure):
    pass

struct_nir_opt_access_options._pack_ = 1 # source:False
struct_nir_opt_access_options._fields_ = [
    ('is_vulkan', ctypes.c_bool),
]

nir_opt_access_options = struct_nir_opt_access_options
try:
    nir_opt_access = _libraries['libtinymesa_cpu.so'].nir_opt_access
    nir_opt_access.restype = ctypes.c_bool
    nir_opt_access.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_access_options)]
except AttributeError:
    pass
try:
    nir_opt_algebraic = _libraries['libtinymesa_cpu.so'].nir_opt_algebraic
    nir_opt_algebraic.restype = ctypes.c_bool
    nir_opt_algebraic.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_before_ffma = _libraries['libtinymesa_cpu.so'].nir_opt_algebraic_before_ffma
    nir_opt_algebraic_before_ffma.restype = ctypes.c_bool
    nir_opt_algebraic_before_ffma.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_before_lower_int64 = _libraries['libtinymesa_cpu.so'].nir_opt_algebraic_before_lower_int64
    nir_opt_algebraic_before_lower_int64.restype = ctypes.c_bool
    nir_opt_algebraic_before_lower_int64.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_late = _libraries['libtinymesa_cpu.so'].nir_opt_algebraic_late
    nir_opt_algebraic_late.restype = ctypes.c_bool
    nir_opt_algebraic_late.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_distribute_src_mods = _libraries['libtinymesa_cpu.so'].nir_opt_algebraic_distribute_src_mods
    nir_opt_algebraic_distribute_src_mods.restype = ctypes.c_bool
    nir_opt_algebraic_distribute_src_mods.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_integer_promotion = _libraries['libtinymesa_cpu.so'].nir_opt_algebraic_integer_promotion
    nir_opt_algebraic_integer_promotion.restype = ctypes.c_bool
    nir_opt_algebraic_integer_promotion.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_reassociate_matrix_mul = _libraries['libtinymesa_cpu.so'].nir_opt_reassociate_matrix_mul
    nir_opt_reassociate_matrix_mul.restype = ctypes.c_bool
    nir_opt_reassociate_matrix_mul.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_constant_folding = _libraries['libtinymesa_cpu.so'].nir_opt_constant_folding
    nir_opt_constant_folding.restype = ctypes.c_bool
    nir_opt_constant_folding.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
nir_combine_barrier_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
try:
    nir_opt_combine_barriers = _libraries['libtinymesa_cpu.so'].nir_opt_combine_barriers
    nir_opt_combine_barriers.restype = ctypes.c_bool
    nir_opt_combine_barriers.argtypes = [ctypes.POINTER(struct_nir_shader), nir_combine_barrier_cb, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'c__EA_mesa_scope'
c__EA_mesa_scope__enumvalues = {
    0: 'SCOPE_NONE',
    1: 'SCOPE_INVOCATION',
    2: 'SCOPE_SUBGROUP',
    3: 'SCOPE_SHADER_CALL',
    4: 'SCOPE_WORKGROUP',
    5: 'SCOPE_QUEUE_FAMILY',
    6: 'SCOPE_DEVICE',
}
SCOPE_NONE = 0
SCOPE_INVOCATION = 1
SCOPE_SUBGROUP = 2
SCOPE_SHADER_CALL = 3
SCOPE_WORKGROUP = 4
SCOPE_QUEUE_FAMILY = 5
SCOPE_DEVICE = 6
c__EA_mesa_scope = ctypes.c_uint32 # enum
mesa_scope = c__EA_mesa_scope
mesa_scope__enumvalues = c__EA_mesa_scope__enumvalues
try:
    nir_opt_acquire_release_barriers = _libraries['libtinymesa_cpu.so'].nir_opt_acquire_release_barriers
    nir_opt_acquire_release_barriers.restype = ctypes.c_bool
    nir_opt_acquire_release_barriers.argtypes = [ctypes.POINTER(struct_nir_shader), mesa_scope]
except AttributeError:
    pass
try:
    nir_opt_barrier_modes = _libraries['libtinymesa_cpu.so'].nir_opt_barrier_modes
    nir_opt_barrier_modes.restype = ctypes.c_bool
    nir_opt_barrier_modes.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_minimize_call_live_states = _libraries['libtinymesa_cpu.so'].nir_minimize_call_live_states
    nir_minimize_call_live_states.restype = ctypes.c_bool
    nir_minimize_call_live_states.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_combine_stores = _libraries['libtinymesa_cpu.so'].nir_opt_combine_stores
    nir_opt_combine_stores.restype = ctypes.c_bool
    nir_opt_combine_stores.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_copy_prop_impl = _libraries['libtinymesa_cpu.so'].nir_copy_prop_impl
    nir_copy_prop_impl.restype = ctypes.c_bool
    nir_copy_prop_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_copy_prop = _libraries['libtinymesa_cpu.so'].nir_copy_prop
    nir_copy_prop.restype = ctypes.c_bool
    nir_copy_prop.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_copy_prop_vars = _libraries['libtinymesa_cpu.so'].nir_opt_copy_prop_vars
    nir_opt_copy_prop_vars.restype = ctypes.c_bool
    nir_opt_copy_prop_vars.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_cse = _libraries['libtinymesa_cpu.so'].nir_opt_cse
    nir_opt_cse.restype = ctypes.c_bool
    nir_opt_cse.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_dce = _libraries['libtinymesa_cpu.so'].nir_opt_dce
    nir_opt_dce.restype = ctypes.c_bool
    nir_opt_dce.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_dead_cf = _libraries['libtinymesa_cpu.so'].nir_opt_dead_cf
    nir_opt_dead_cf.restype = ctypes.c_bool
    nir_opt_dead_cf.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_dead_write_vars = _libraries['libtinymesa_cpu.so'].nir_opt_dead_write_vars
    nir_opt_dead_write_vars.restype = ctypes.c_bool
    nir_opt_dead_write_vars.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_deref_impl = _libraries['libtinymesa_cpu.so'].nir_opt_deref_impl
    nir_opt_deref_impl.restype = ctypes.c_bool
    nir_opt_deref_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_opt_deref = _libraries['libtinymesa_cpu.so'].nir_opt_deref
    nir_opt_deref.restype = ctypes.c_bool
    nir_opt_deref.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_find_array_copies = _libraries['libtinymesa_cpu.so'].nir_opt_find_array_copies
    nir_opt_find_array_copies.restype = ctypes.c_bool
    nir_opt_find_array_copies.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_def_is_frag_coord_z = _libraries['libtinymesa_cpu.so'].nir_def_is_frag_coord_z
    nir_def_is_frag_coord_z.restype = ctypes.c_bool
    nir_def_is_frag_coord_z.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_opt_fragdepth = _libraries['libtinymesa_cpu.so'].nir_opt_fragdepth
    nir_opt_fragdepth.restype = ctypes.c_bool
    nir_opt_fragdepth.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_gcm = _libraries['libtinymesa_cpu.so'].nir_opt_gcm
    nir_opt_gcm.restype = ctypes.c_bool
    nir_opt_gcm.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_generate_bfi = _libraries['libtinymesa_cpu.so'].nir_opt_generate_bfi
    nir_opt_generate_bfi.restype = ctypes.c_bool
    nir_opt_generate_bfi.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_idiv_const = _libraries['libtinymesa_cpu.so'].nir_opt_idiv_const
    nir_opt_idiv_const.restype = ctypes.c_bool
    nir_opt_idiv_const.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_opt_mqsad = _libraries['libtinymesa_cpu.so'].nir_opt_mqsad
    nir_opt_mqsad.restype = ctypes.c_bool
    nir_opt_mqsad.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_opt_if_options'
c__EA_nir_opt_if_options__enumvalues = {
    1: 'nir_opt_if_optimize_phi_true_false',
    2: 'nir_opt_if_avoid_64bit_phis',
}
nir_opt_if_optimize_phi_true_false = 1
nir_opt_if_avoid_64bit_phis = 2
c__EA_nir_opt_if_options = ctypes.c_uint32 # enum
nir_opt_if_options = c__EA_nir_opt_if_options
nir_opt_if_options__enumvalues = c__EA_nir_opt_if_options__enumvalues
try:
    nir_opt_if = _libraries['libtinymesa_cpu.so'].nir_opt_if
    nir_opt_if.restype = ctypes.c_bool
    nir_opt_if.argtypes = [ctypes.POINTER(struct_nir_shader), nir_opt_if_options]
except AttributeError:
    pass
try:
    nir_opt_intrinsics = _libraries['libtinymesa_cpu.so'].nir_opt_intrinsics
    nir_opt_intrinsics.restype = ctypes.c_bool
    nir_opt_intrinsics.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_large_constants = _libraries['libtinymesa_cpu.so'].nir_opt_large_constants
    nir_opt_large_constants.restype = ctypes.c_bool
    nir_opt_large_constants.argtypes = [ctypes.POINTER(struct_nir_shader), glsl_type_size_align_func, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_opt_licm = _libraries['libtinymesa_cpu.so'].nir_opt_licm
    nir_opt_licm.restype = ctypes.c_bool
    nir_opt_licm.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_loop = _libraries['libtinymesa_cpu.so'].nir_opt_loop
    nir_opt_loop.restype = ctypes.c_bool
    nir_opt_loop.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_loop_unroll = _libraries['libtinymesa_cpu.so'].nir_opt_loop_unroll
    nir_opt_loop_unroll.restype = ctypes.c_bool
    nir_opt_loop_unroll.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_move_options'
c__EA_nir_move_options__enumvalues = {
    1: 'nir_move_const_undef',
    2: 'nir_move_load_ubo',
    4: 'nir_move_load_input',
    8: 'nir_move_comparisons',
    16: 'nir_move_copies',
    32: 'nir_move_load_ssbo',
    64: 'nir_move_load_uniform',
    128: 'nir_move_alu',
    256: 'nir_dont_move_byte_word_vecs',
}
nir_move_const_undef = 1
nir_move_load_ubo = 2
nir_move_load_input = 4
nir_move_comparisons = 8
nir_move_copies = 16
nir_move_load_ssbo = 32
nir_move_load_uniform = 64
nir_move_alu = 128
nir_dont_move_byte_word_vecs = 256
c__EA_nir_move_options = ctypes.c_uint32 # enum
nir_move_options = c__EA_nir_move_options
nir_move_options__enumvalues = c__EA_nir_move_options__enumvalues
try:
    nir_can_move_instr = _libraries['libtinymesa_cpu.so'].nir_can_move_instr
    nir_can_move_instr.restype = ctypes.c_bool
    nir_can_move_instr.argtypes = [ctypes.POINTER(struct_nir_instr), nir_move_options]
except AttributeError:
    pass
try:
    nir_opt_sink = _libraries['libtinymesa_cpu.so'].nir_opt_sink
    nir_opt_sink.restype = ctypes.c_bool
    nir_opt_sink.argtypes = [ctypes.POINTER(struct_nir_shader), nir_move_options]
except AttributeError:
    pass
try:
    nir_opt_move = _libraries['libtinymesa_cpu.so'].nir_opt_move
    nir_opt_move.restype = ctypes.c_bool
    nir_opt_move.argtypes = [ctypes.POINTER(struct_nir_shader), nir_move_options]
except AttributeError:
    pass
class struct_nir_opt_offsets_options(Structure):
    pass

struct_nir_opt_offsets_options._pack_ = 1 # source:False
struct_nir_opt_offsets_options._fields_ = [
    ('uniform_max', ctypes.c_uint32),
    ('ubo_vec4_max', ctypes.c_uint32),
    ('shared_max', ctypes.c_uint32),
    ('shared_atomic_max', ctypes.c_uint32),
    ('buffer_max', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('max_offset_cb', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))),
    ('max_offset_data', ctypes.POINTER(None)),
    ('allow_offset_wrap', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

nir_opt_offsets_options = struct_nir_opt_offsets_options
try:
    nir_opt_offsets = _libraries['libtinymesa_cpu.so'].nir_opt_offsets
    nir_opt_offsets.restype = ctypes.c_bool
    nir_opt_offsets.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_offsets_options)]
except AttributeError:
    pass
class struct_nir_opt_peephole_select_options(Structure):
    pass

struct_nir_opt_peephole_select_options._pack_ = 1 # source:False
struct_nir_opt_peephole_select_options._fields_ = [
    ('limit', ctypes.c_uint32),
    ('indirect_load_ok', ctypes.c_bool),
    ('expensive_alu_ok', ctypes.c_bool),
    ('discard_ok', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
]

nir_opt_peephole_select_options = struct_nir_opt_peephole_select_options
try:
    nir_opt_peephole_select = _libraries['libtinymesa_cpu.so'].nir_opt_peephole_select
    nir_opt_peephole_select.restype = ctypes.c_bool
    nir_opt_peephole_select.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_peephole_select_options)]
except AttributeError:
    pass
try:
    nir_opt_reassociate_bfi = _libraries['libtinymesa_cpu.so'].nir_opt_reassociate_bfi
    nir_opt_reassociate_bfi.restype = ctypes.c_bool
    nir_opt_reassociate_bfi.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_rematerialize_compares = _libraries['libtinymesa_cpu.so'].nir_opt_rematerialize_compares
    nir_opt_rematerialize_compares.restype = ctypes.c_bool
    nir_opt_rematerialize_compares.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_remove_phis = _libraries['libtinymesa_cpu.so'].nir_opt_remove_phis
    nir_opt_remove_phis.restype = ctypes.c_bool
    nir_opt_remove_phis.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_single_src_phis_block = _libraries['libtinymesa_cpu.so'].nir_remove_single_src_phis_block
    nir_remove_single_src_phis_block.restype = ctypes.c_bool
    nir_remove_single_src_phis_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_opt_phi_precision = _libraries['libtinymesa_cpu.so'].nir_opt_phi_precision
    nir_opt_phi_precision.restype = ctypes.c_bool
    nir_opt_phi_precision.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_phi_to_bool = _libraries['libtinymesa_cpu.so'].nir_opt_phi_to_bool
    nir_opt_phi_to_bool.restype = ctypes.c_bool
    nir_opt_phi_to_bool.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_shrink_stores = _libraries['libtinymesa_cpu.so'].nir_opt_shrink_stores
    nir_opt_shrink_stores.restype = ctypes.c_bool
    nir_opt_shrink_stores.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_shrink_vectors = _libraries['libtinymesa_cpu.so'].nir_opt_shrink_vectors
    nir_opt_shrink_vectors.restype = ctypes.c_bool
    nir_opt_shrink_vectors.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_undef = _libraries['libtinymesa_cpu.so'].nir_opt_undef
    nir_opt_undef.restype = ctypes.c_bool
    nir_opt_undef.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_undef_to_zero = _libraries['libtinymesa_cpu.so'].nir_lower_undef_to_zero
    nir_lower_undef_to_zero.restype = ctypes.c_bool
    nir_lower_undef_to_zero.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_uniform_atomics = _libraries['libtinymesa_cpu.so'].nir_opt_uniform_atomics
    nir_opt_uniform_atomics.restype = ctypes.c_bool
    nir_opt_uniform_atomics.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_uniform_subgroup = _libraries['libtinymesa_cpu.so'].nir_opt_uniform_subgroup
    nir_opt_uniform_subgroup.restype = ctypes.c_bool
    nir_opt_uniform_subgroup.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_subgroups_options)]
except AttributeError:
    pass
try:
    nir_opt_vectorize = _libraries['libtinymesa_cpu.so'].nir_opt_vectorize
    nir_opt_vectorize.restype = ctypes.c_bool
    nir_opt_vectorize.argtypes = [ctypes.POINTER(struct_nir_shader), nir_vectorize_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_opt_vectorize_io = _libraries['libtinymesa_cpu.so'].nir_opt_vectorize_io
    nir_opt_vectorize_io.restype = ctypes.c_bool
    nir_opt_vectorize_io.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_move_discards_to_top = _libraries['libtinymesa_cpu.so'].nir_opt_move_discards_to_top
    nir_opt_move_discards_to_top.restype = ctypes.c_bool
    nir_opt_move_discards_to_top.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_ray_queries = _libraries['libtinymesa_cpu.so'].nir_opt_ray_queries
    nir_opt_ray_queries.restype = ctypes.c_bool
    nir_opt_ray_queries.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_ray_query_ranges = _libraries['libtinymesa_cpu.so'].nir_opt_ray_query_ranges
    nir_opt_ray_query_ranges.restype = ctypes.c_bool
    nir_opt_ray_query_ranges.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_tex_skip_helpers = _libraries['libtinymesa_cpu.so'].nir_opt_tex_skip_helpers
    nir_opt_tex_skip_helpers.restype = ctypes.c_bool
    nir_opt_tex_skip_helpers.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_sweep = _libraries['libtinymesa_cpu.so'].nir_sweep
    nir_sweep.restype = None
    nir_sweep.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_gl_system_value'
c__EA_gl_system_value__enumvalues = {
    0: 'SYSTEM_VALUE_SUBGROUP_SIZE',
    1: 'SYSTEM_VALUE_SUBGROUP_INVOCATION',
    2: 'SYSTEM_VALUE_SUBGROUP_EQ_MASK',
    3: 'SYSTEM_VALUE_SUBGROUP_GE_MASK',
    4: 'SYSTEM_VALUE_SUBGROUP_GT_MASK',
    5: 'SYSTEM_VALUE_SUBGROUP_LE_MASK',
    6: 'SYSTEM_VALUE_SUBGROUP_LT_MASK',
    7: 'SYSTEM_VALUE_NUM_SUBGROUPS',
    8: 'SYSTEM_VALUE_SUBGROUP_ID',
    9: 'SYSTEM_VALUE_VERTEX_ID',
    10: 'SYSTEM_VALUE_INSTANCE_ID',
    11: 'SYSTEM_VALUE_INSTANCE_INDEX',
    12: 'SYSTEM_VALUE_VERTEX_ID_ZERO_BASE',
    13: 'SYSTEM_VALUE_BASE_VERTEX',
    14: 'SYSTEM_VALUE_FIRST_VERTEX',
    15: 'SYSTEM_VALUE_IS_INDEXED_DRAW',
    16: 'SYSTEM_VALUE_BASE_INSTANCE',
    17: 'SYSTEM_VALUE_DRAW_ID',
    18: 'SYSTEM_VALUE_INVOCATION_ID',
    19: 'SYSTEM_VALUE_FRAG_COORD',
    20: 'SYSTEM_VALUE_PIXEL_COORD',
    21: 'SYSTEM_VALUE_FRAG_COORD_Z',
    22: 'SYSTEM_VALUE_FRAG_COORD_W',
    23: 'SYSTEM_VALUE_POINT_COORD',
    24: 'SYSTEM_VALUE_LINE_COORD',
    25: 'SYSTEM_VALUE_FRONT_FACE',
    26: 'SYSTEM_VALUE_FRONT_FACE_FSIGN',
    27: 'SYSTEM_VALUE_SAMPLE_ID',
    28: 'SYSTEM_VALUE_SAMPLE_POS',
    29: 'SYSTEM_VALUE_SAMPLE_POS_OR_CENTER',
    30: 'SYSTEM_VALUE_SAMPLE_MASK_IN',
    31: 'SYSTEM_VALUE_LAYER_ID',
    32: 'SYSTEM_VALUE_HELPER_INVOCATION',
    33: 'SYSTEM_VALUE_COLOR0',
    34: 'SYSTEM_VALUE_COLOR1',
    35: 'SYSTEM_VALUE_TESS_COORD',
    36: 'SYSTEM_VALUE_VERTICES_IN',
    37: 'SYSTEM_VALUE_PRIMITIVE_ID',
    38: 'SYSTEM_VALUE_TESS_LEVEL_OUTER',
    39: 'SYSTEM_VALUE_TESS_LEVEL_INNER',
    40: 'SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT',
    41: 'SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT',
    42: 'SYSTEM_VALUE_LOCAL_INVOCATION_ID',
    43: 'SYSTEM_VALUE_LOCAL_INVOCATION_INDEX',
    44: 'SYSTEM_VALUE_GLOBAL_INVOCATION_ID',
    45: 'SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID',
    46: 'SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX',
    47: 'SYSTEM_VALUE_WORKGROUP_ID',
    48: 'SYSTEM_VALUE_BASE_WORKGROUP_ID',
    49: 'SYSTEM_VALUE_WORKGROUP_INDEX',
    50: 'SYSTEM_VALUE_NUM_WORKGROUPS',
    51: 'SYSTEM_VALUE_WORKGROUP_SIZE',
    52: 'SYSTEM_VALUE_GLOBAL_GROUP_SIZE',
    53: 'SYSTEM_VALUE_WORK_DIM',
    54: 'SYSTEM_VALUE_USER_DATA_AMD',
    55: 'SYSTEM_VALUE_DEVICE_INDEX',
    56: 'SYSTEM_VALUE_VIEW_INDEX',
    57: 'SYSTEM_VALUE_VERTEX_CNT',
    58: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL',
    59: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE',
    60: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID',
    61: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW',
    62: 'SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL',
    63: 'SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID',
    64: 'SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE',
    65: 'SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL',
    66: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD',
    67: 'SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD',
    68: 'SYSTEM_VALUE_RAY_LAUNCH_ID',
    69: 'SYSTEM_VALUE_RAY_LAUNCH_SIZE',
    70: 'SYSTEM_VALUE_RAY_WORLD_ORIGIN',
    71: 'SYSTEM_VALUE_RAY_WORLD_DIRECTION',
    72: 'SYSTEM_VALUE_RAY_OBJECT_ORIGIN',
    73: 'SYSTEM_VALUE_RAY_OBJECT_DIRECTION',
    74: 'SYSTEM_VALUE_RAY_T_MIN',
    75: 'SYSTEM_VALUE_RAY_T_MAX',
    76: 'SYSTEM_VALUE_RAY_OBJECT_TO_WORLD',
    77: 'SYSTEM_VALUE_RAY_WORLD_TO_OBJECT',
    78: 'SYSTEM_VALUE_RAY_HIT_KIND',
    79: 'SYSTEM_VALUE_RAY_FLAGS',
    80: 'SYSTEM_VALUE_RAY_GEOMETRY_INDEX',
    81: 'SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX',
    82: 'SYSTEM_VALUE_CULL_MASK',
    83: 'SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS',
    84: 'SYSTEM_VALUE_MESH_VIEW_COUNT',
    85: 'SYSTEM_VALUE_MESH_VIEW_INDICES',
    86: 'SYSTEM_VALUE_GS_HEADER_IR3',
    87: 'SYSTEM_VALUE_TCS_HEADER_IR3',
    88: 'SYSTEM_VALUE_REL_PATCH_ID_IR3',
    89: 'SYSTEM_VALUE_FRAG_SHADING_RATE',
    90: 'SYSTEM_VALUE_FULLY_COVERED',
    91: 'SYSTEM_VALUE_FRAG_SIZE',
    92: 'SYSTEM_VALUE_FRAG_INVOCATION_COUNT',
    93: 'SYSTEM_VALUE_SHADER_INDEX',
    94: 'SYSTEM_VALUE_COALESCED_INPUT_COUNT',
    95: 'SYSTEM_VALUE_WARPS_PER_SM_NV',
    96: 'SYSTEM_VALUE_SM_COUNT_NV',
    97: 'SYSTEM_VALUE_WARP_ID_NV',
    98: 'SYSTEM_VALUE_SM_ID_NV',
    99: 'SYSTEM_VALUE_MAX',
}
SYSTEM_VALUE_SUBGROUP_SIZE = 0
SYSTEM_VALUE_SUBGROUP_INVOCATION = 1
SYSTEM_VALUE_SUBGROUP_EQ_MASK = 2
SYSTEM_VALUE_SUBGROUP_GE_MASK = 3
SYSTEM_VALUE_SUBGROUP_GT_MASK = 4
SYSTEM_VALUE_SUBGROUP_LE_MASK = 5
SYSTEM_VALUE_SUBGROUP_LT_MASK = 6
SYSTEM_VALUE_NUM_SUBGROUPS = 7
SYSTEM_VALUE_SUBGROUP_ID = 8
SYSTEM_VALUE_VERTEX_ID = 9
SYSTEM_VALUE_INSTANCE_ID = 10
SYSTEM_VALUE_INSTANCE_INDEX = 11
SYSTEM_VALUE_VERTEX_ID_ZERO_BASE = 12
SYSTEM_VALUE_BASE_VERTEX = 13
SYSTEM_VALUE_FIRST_VERTEX = 14
SYSTEM_VALUE_IS_INDEXED_DRAW = 15
SYSTEM_VALUE_BASE_INSTANCE = 16
SYSTEM_VALUE_DRAW_ID = 17
SYSTEM_VALUE_INVOCATION_ID = 18
SYSTEM_VALUE_FRAG_COORD = 19
SYSTEM_VALUE_PIXEL_COORD = 20
SYSTEM_VALUE_FRAG_COORD_Z = 21
SYSTEM_VALUE_FRAG_COORD_W = 22
SYSTEM_VALUE_POINT_COORD = 23
SYSTEM_VALUE_LINE_COORD = 24
SYSTEM_VALUE_FRONT_FACE = 25
SYSTEM_VALUE_FRONT_FACE_FSIGN = 26
SYSTEM_VALUE_SAMPLE_ID = 27
SYSTEM_VALUE_SAMPLE_POS = 28
SYSTEM_VALUE_SAMPLE_POS_OR_CENTER = 29
SYSTEM_VALUE_SAMPLE_MASK_IN = 30
SYSTEM_VALUE_LAYER_ID = 31
SYSTEM_VALUE_HELPER_INVOCATION = 32
SYSTEM_VALUE_COLOR0 = 33
SYSTEM_VALUE_COLOR1 = 34
SYSTEM_VALUE_TESS_COORD = 35
SYSTEM_VALUE_VERTICES_IN = 36
SYSTEM_VALUE_PRIMITIVE_ID = 37
SYSTEM_VALUE_TESS_LEVEL_OUTER = 38
SYSTEM_VALUE_TESS_LEVEL_INNER = 39
SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT = 40
SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT = 41
SYSTEM_VALUE_LOCAL_INVOCATION_ID = 42
SYSTEM_VALUE_LOCAL_INVOCATION_INDEX = 43
SYSTEM_VALUE_GLOBAL_INVOCATION_ID = 44
SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID = 45
SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX = 46
SYSTEM_VALUE_WORKGROUP_ID = 47
SYSTEM_VALUE_BASE_WORKGROUP_ID = 48
SYSTEM_VALUE_WORKGROUP_INDEX = 49
SYSTEM_VALUE_NUM_WORKGROUPS = 50
SYSTEM_VALUE_WORKGROUP_SIZE = 51
SYSTEM_VALUE_GLOBAL_GROUP_SIZE = 52
SYSTEM_VALUE_WORK_DIM = 53
SYSTEM_VALUE_USER_DATA_AMD = 54
SYSTEM_VALUE_DEVICE_INDEX = 55
SYSTEM_VALUE_VIEW_INDEX = 56
SYSTEM_VALUE_VERTEX_CNT = 57
SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL = 58
SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE = 59
SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID = 60
SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW = 61
SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL = 62
SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID = 63
SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE = 64
SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL = 65
SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD = 66
SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD = 67
SYSTEM_VALUE_RAY_LAUNCH_ID = 68
SYSTEM_VALUE_RAY_LAUNCH_SIZE = 69
SYSTEM_VALUE_RAY_WORLD_ORIGIN = 70
SYSTEM_VALUE_RAY_WORLD_DIRECTION = 71
SYSTEM_VALUE_RAY_OBJECT_ORIGIN = 72
SYSTEM_VALUE_RAY_OBJECT_DIRECTION = 73
SYSTEM_VALUE_RAY_T_MIN = 74
SYSTEM_VALUE_RAY_T_MAX = 75
SYSTEM_VALUE_RAY_OBJECT_TO_WORLD = 76
SYSTEM_VALUE_RAY_WORLD_TO_OBJECT = 77
SYSTEM_VALUE_RAY_HIT_KIND = 78
SYSTEM_VALUE_RAY_FLAGS = 79
SYSTEM_VALUE_RAY_GEOMETRY_INDEX = 80
SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX = 81
SYSTEM_VALUE_CULL_MASK = 82
SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS = 83
SYSTEM_VALUE_MESH_VIEW_COUNT = 84
SYSTEM_VALUE_MESH_VIEW_INDICES = 85
SYSTEM_VALUE_GS_HEADER_IR3 = 86
SYSTEM_VALUE_TCS_HEADER_IR3 = 87
SYSTEM_VALUE_REL_PATCH_ID_IR3 = 88
SYSTEM_VALUE_FRAG_SHADING_RATE = 89
SYSTEM_VALUE_FULLY_COVERED = 90
SYSTEM_VALUE_FRAG_SIZE = 91
SYSTEM_VALUE_FRAG_INVOCATION_COUNT = 92
SYSTEM_VALUE_SHADER_INDEX = 93
SYSTEM_VALUE_COALESCED_INPUT_COUNT = 94
SYSTEM_VALUE_WARPS_PER_SM_NV = 95
SYSTEM_VALUE_SM_COUNT_NV = 96
SYSTEM_VALUE_WARP_ID_NV = 97
SYSTEM_VALUE_SM_ID_NV = 98
SYSTEM_VALUE_MAX = 99
c__EA_gl_system_value = ctypes.c_uint32 # enum
gl_system_value = c__EA_gl_system_value
gl_system_value__enumvalues = c__EA_gl_system_value__enumvalues
try:
    nir_intrinsic_from_system_value = _libraries['libtinymesa_cpu.so'].nir_intrinsic_from_system_value
    nir_intrinsic_from_system_value.restype = nir_intrinsic_op
    nir_intrinsic_from_system_value.argtypes = [gl_system_value]
except AttributeError:
    pass
try:
    nir_system_value_from_intrinsic = _libraries['libtinymesa_cpu.so'].nir_system_value_from_intrinsic
    nir_system_value_from_intrinsic.restype = gl_system_value
    nir_system_value_from_intrinsic.argtypes = [nir_intrinsic_op]
except AttributeError:
    pass
try:
    nir_variable_is_in_ubo = _libraries['FIXME_STUB'].nir_variable_is_in_ubo
    nir_variable_is_in_ubo.restype = ctypes.c_bool
    nir_variable_is_in_ubo.argtypes = [ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_is_in_ssbo = _libraries['FIXME_STUB'].nir_variable_is_in_ssbo
    nir_variable_is_in_ssbo.restype = ctypes.c_bool
    nir_variable_is_in_ssbo.argtypes = [ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_is_in_block = _libraries['FIXME_STUB'].nir_variable_is_in_block
    nir_variable_is_in_block.restype = ctypes.c_bool
    nir_variable_is_in_block.argtypes = [ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_count_slots = _libraries['FIXME_STUB'].nir_variable_count_slots
    nir_variable_count_slots.restype = ctypes.c_uint32
    nir_variable_count_slots.argtypes = [ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_deref_count_slots = _libraries['FIXME_STUB'].nir_deref_count_slots
    nir_deref_count_slots.restype = ctypes.c_uint32
    nir_deref_count_slots.argtypes = [ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
class struct_nir_unsigned_upper_bound_config(Structure):
    pass

struct_nir_unsigned_upper_bound_config._pack_ = 1 # source:False
struct_nir_unsigned_upper_bound_config._fields_ = [
    ('min_subgroup_size', ctypes.c_uint32),
    ('max_subgroup_size', ctypes.c_uint32),
    ('max_workgroup_invocations', ctypes.c_uint32),
    ('max_workgroup_count', ctypes.c_uint32 * 3),
    ('max_workgroup_size', ctypes.c_uint32 * 3),
    ('vertex_attrib_max', ctypes.c_uint32 * 32),
]

nir_unsigned_upper_bound_config = struct_nir_unsigned_upper_bound_config
try:
    nir_unsigned_upper_bound = _libraries['libtinymesa_cpu.so'].nir_unsigned_upper_bound
    nir_unsigned_upper_bound.restype = uint32_t
    nir_unsigned_upper_bound.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_hash_table), nir_scalar, ctypes.POINTER(struct_nir_unsigned_upper_bound_config)]
except AttributeError:
    pass
try:
    nir_addition_might_overflow = _libraries['libtinymesa_cpu.so'].nir_addition_might_overflow
    nir_addition_might_overflow.restype = ctypes.c_bool
    nir_addition_might_overflow.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_hash_table), nir_scalar, ctypes.c_uint32, ctypes.POINTER(struct_nir_unsigned_upper_bound_config)]
except AttributeError:
    pass
class struct_nir_opt_preamble_options(Structure):
    pass

struct_nir_opt_preamble_options._pack_ = 1 # source:False
struct_nir_opt_preamble_options._fields_ = [
    ('drawid_uniform', ctypes.c_bool),
    ('subgroup_size_uniform', ctypes.c_bool),
    ('load_workgroup_size_allowed', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 5),
    ('def_size', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_nir_def), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(c__EA_nir_preamble_class))),
    ('preamble_storage_size', ctypes.c_uint32 * 2),
    ('instr_cost_cb', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('rewrite_cost_cb', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(struct_nir_def), ctypes.POINTER(None))),
    ('avoid_instr_cb', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('cb_data', ctypes.POINTER(None)),
]

nir_opt_preamble_options = struct_nir_opt_preamble_options
try:
    nir_opt_preamble = _libraries['libtinymesa_cpu.so'].nir_opt_preamble
    nir_opt_preamble.restype = ctypes.c_bool
    nir_opt_preamble.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_preamble_options), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    nir_shader_get_preamble = _libraries['libtinymesa_cpu.so'].nir_shader_get_preamble
    nir_shader_get_preamble.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_shader_get_preamble.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_point_smooth = _libraries['libtinymesa_cpu.so'].nir_lower_point_smooth
    nir_lower_point_smooth.restype = ctypes.c_bool
    nir_lower_point_smooth.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_poly_line_smooth = _libraries['libtinymesa_cpu.so'].nir_lower_poly_line_smooth
    nir_lower_poly_line_smooth.restype = ctypes.c_bool
    nir_lower_poly_line_smooth.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_mod_analysis = _libraries['libtinymesa_cpu.so'].nir_mod_analysis
    nir_mod_analysis.restype = ctypes.c_bool
    nir_mod_analysis.argtypes = [nir_scalar, nir_alu_type, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    nir_remove_tex_shadow = _libraries['libtinymesa_cpu.so'].nir_remove_tex_shadow
    nir_remove_tex_shadow.restype = ctypes.c_bool
    nir_remove_tex_shadow.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_trivialize_registers = _libraries['libtinymesa_cpu.so'].nir_trivialize_registers
    nir_trivialize_registers.restype = ctypes.c_bool
    nir_trivialize_registers.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_static_workgroup_size = _libraries['libtinymesa_cpu.so'].nir_static_workgroup_size
    nir_static_workgroup_size.restype = ctypes.c_uint32
    nir_static_workgroup_size.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_reg_get_decl = _libraries['FIXME_STUB'].nir_reg_get_decl
    nir_reg_get_decl.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_reg_get_decl.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_next_decl_reg = _libraries['FIXME_STUB'].nir_next_decl_reg
    nir_next_decl_reg.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_next_decl_reg.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_after_reg_decls = _libraries['FIXME_STUB'].nir_after_reg_decls
    nir_after_reg_decls.restype = nir_cursor
    nir_after_reg_decls.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_is_load_reg = _libraries['FIXME_STUB'].nir_is_load_reg
    nir_is_load_reg.restype = ctypes.c_bool
    nir_is_load_reg.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_is_store_reg = _libraries['FIXME_STUB'].nir_is_store_reg
    nir_is_store_reg.restype = ctypes.c_bool
    nir_is_store_reg.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_load_reg_for_def = _libraries['FIXME_STUB'].nir_load_reg_for_def
    nir_load_reg_for_def.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_load_reg_for_def.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_store_reg_for_def = _libraries['FIXME_STUB'].nir_store_reg_for_def
    nir_store_reg_for_def.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_store_reg_for_def.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
class struct_nir_use_dominance_state(Structure):
    pass

nir_use_dominance_state = struct_nir_use_dominance_state
try:
    nir_calc_use_dominance_impl = _libraries['libtinymesa_cpu.so'].nir_calc_use_dominance_impl
    nir_calc_use_dominance_impl.restype = ctypes.POINTER(struct_nir_use_dominance_state)
    nir_calc_use_dominance_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_get_immediate_use_dominator = _libraries['libtinymesa_cpu.so'].nir_get_immediate_use_dominator
    nir_get_immediate_use_dominator.restype = ctypes.POINTER(struct_nir_instr)
    nir_get_immediate_use_dominator.argtypes = [ctypes.POINTER(struct_nir_use_dominance_state), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_use_dominance_lca = _libraries['libtinymesa_cpu.so'].nir_use_dominance_lca
    nir_use_dominance_lca.restype = ctypes.POINTER(struct_nir_instr)
    nir_use_dominance_lca.argtypes = [ctypes.POINTER(struct_nir_use_dominance_state), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_dominates_use = _libraries['libtinymesa_cpu.so'].nir_instr_dominates_use
    nir_instr_dominates_use.restype = ctypes.c_bool
    nir_instr_dominates_use.argtypes = [ctypes.POINTER(struct_nir_use_dominance_state), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_print_use_dominators = _libraries['libtinymesa_cpu.so'].nir_print_use_dominators
    nir_print_use_dominators.restype = None
    nir_print_use_dominators.argtypes = [ctypes.POINTER(struct_nir_use_dominance_state), ctypes.POINTER(ctypes.POINTER(struct_nir_instr)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_verts_in_output_prim = _libraries['FIXME_STUB'].nir_verts_in_output_prim
    nir_verts_in_output_prim.restype = ctypes.c_uint32
    nir_verts_in_output_prim.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_c__SA_nir_output_deps(Structure):
    pass

class struct_c__SA_nir_output_deps_0(Structure):
    pass

struct_c__SA_nir_output_deps_0._pack_ = 1 # source:False
struct_c__SA_nir_output_deps_0._fields_ = [
    ('instr_list', ctypes.POINTER(ctypes.POINTER(struct_nir_instr))),
    ('num_instr', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_c__SA_nir_output_deps._pack_ = 1 # source:False
struct_c__SA_nir_output_deps._fields_ = [
    ('output', struct_c__SA_nir_output_deps_0 * 112),
]

nir_output_deps = struct_c__SA_nir_output_deps
try:
    nir_gather_output_dependencies = _libraries['libtinymesa_cpu.so'].nir_gather_output_dependencies
    nir_gather_output_dependencies.restype = None
    nir_gather_output_dependencies.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_c__SA_nir_output_deps)]
except AttributeError:
    pass
try:
    nir_free_output_dependencies = _libraries['libtinymesa_cpu.so'].nir_free_output_dependencies
    nir_free_output_dependencies.restype = None
    nir_free_output_dependencies.argtypes = [ctypes.POINTER(struct_c__SA_nir_output_deps)]
except AttributeError:
    pass
class struct_c__SA_nir_input_to_output_deps(Structure):
    pass

class struct_c__SA_nir_input_to_output_deps_0(Structure):
    pass

struct_c__SA_nir_input_to_output_deps_0._pack_ = 1 # source:False
struct_c__SA_nir_input_to_output_deps_0._fields_ = [
    ('inputs', ctypes.c_uint32 * 28),
    ('defined', ctypes.c_bool),
    ('uses_ssbo_reads', ctypes.c_bool),
    ('uses_image_reads', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
]

struct_c__SA_nir_input_to_output_deps._pack_ = 1 # source:False
struct_c__SA_nir_input_to_output_deps._fields_ = [
    ('output', struct_c__SA_nir_input_to_output_deps_0 * 112),
]

nir_input_to_output_deps = struct_c__SA_nir_input_to_output_deps
try:
    nir_gather_input_to_output_dependencies = _libraries['libtinymesa_cpu.so'].nir_gather_input_to_output_dependencies
    nir_gather_input_to_output_dependencies.restype = None
    nir_gather_input_to_output_dependencies.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_c__SA_nir_input_to_output_deps)]
except AttributeError:
    pass
try:
    nir_print_input_to_output_deps = _libraries['libtinymesa_cpu.so'].nir_print_input_to_output_deps
    nir_print_input_to_output_deps.restype = None
    nir_print_input_to_output_deps.argtypes = [ctypes.POINTER(struct_c__SA_nir_input_to_output_deps), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
class struct_c__SA_nir_output_clipper_var_groups(Structure):
    pass

struct_c__SA_nir_output_clipper_var_groups._pack_ = 1 # source:False
struct_c__SA_nir_output_clipper_var_groups._fields_ = [
    ('pos_only', ctypes.c_uint32 * 28),
    ('var_only', ctypes.c_uint32 * 28),
    ('both', ctypes.c_uint32 * 28),
]

nir_output_clipper_var_groups = struct_c__SA_nir_output_clipper_var_groups
try:
    nir_gather_output_clipper_var_groups = _libraries['libtinymesa_cpu.so'].nir_gather_output_clipper_var_groups
    nir_gather_output_clipper_var_groups.restype = None
    nir_gather_output_clipper_var_groups.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_c__SA_nir_output_clipper_var_groups)]
except AttributeError:
    pass
nir_builder = struct_nir_builder
try:
    nir_builder_create = _libraries['FIXME_STUB'].nir_builder_create
    nir_builder_create.restype = nir_builder
    nir_builder_create.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_builder_at = _libraries['FIXME_STUB'].nir_builder_at
    nir_builder_at.restype = nir_builder
    nir_builder_at.argtypes = [nir_cursor]
except AttributeError:
    pass
try:
    nir_builder_init_simple_shader = _libraries['libtinymesa_cpu.so'].nir_builder_init_simple_shader
    nir_builder_init_simple_shader.restype = nir_builder
    nir_builder_init_simple_shader.argtypes = [gl_shader_stage, ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
nir_instr_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
nir_intrinsic_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
nir_alu_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_alu_instr), ctypes.POINTER(None))
nir_tex_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_tex_instr), ctypes.POINTER(None))
nir_phi_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_phi_instr), ctypes.POINTER(None))
try:
    nir_function_instructions_pass = _libraries['FIXME_STUB'].nir_function_instructions_pass
    nir_function_instructions_pass.restype = ctypes.c_bool
    nir_function_instructions_pass.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_instr_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_instructions_pass = _libraries['FIXME_STUB'].nir_shader_instructions_pass
    nir_shader_instructions_pass.restype = ctypes.c_bool
    nir_shader_instructions_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_function_intrinsics_pass = _libraries['FIXME_STUB'].nir_function_intrinsics_pass
    nir_function_intrinsics_pass.restype = ctypes.c_bool
    nir_function_intrinsics_pass.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_intrinsic_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_intrinsics_pass = _libraries['FIXME_STUB'].nir_shader_intrinsics_pass
    nir_shader_intrinsics_pass.restype = ctypes.c_bool
    nir_shader_intrinsics_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrinsic_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_alu_pass = _libraries['FIXME_STUB'].nir_shader_alu_pass
    nir_shader_alu_pass.restype = ctypes.c_bool
    nir_shader_alu_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_alu_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_tex_pass = _libraries['FIXME_STUB'].nir_shader_tex_pass
    nir_shader_tex_pass.restype = ctypes.c_bool
    nir_shader_tex_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_tex_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_phi_pass = _libraries['FIXME_STUB'].nir_shader_phi_pass
    nir_shader_phi_pass.restype = ctypes.c_bool
    nir_shader_phi_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_phi_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_builder_instr_insert = _libraries['libtinymesa_cpu.so'].nir_builder_instr_insert
    nir_builder_instr_insert.restype = None
    nir_builder_instr_insert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_builder_instr_insert_at_top = _libraries['libtinymesa_cpu.so'].nir_builder_instr_insert_at_top
    nir_builder_instr_insert_at_top.restype = None
    nir_builder_instr_insert_at_top.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_builder_last_instr = _libraries['FIXME_STUB'].nir_builder_last_instr
    nir_builder_last_instr.restype = ctypes.POINTER(struct_nir_instr)
    nir_builder_last_instr.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_build_alu = _libraries['libtinymesa_cpu.so'].nir_build_alu
    nir_build_alu.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu1 = _libraries['libtinymesa_cpu.so'].nir_build_alu1
    nir_build_alu1.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu1.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu2 = _libraries['libtinymesa_cpu.so'].nir_build_alu2
    nir_build_alu2.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu2.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu3 = _libraries['libtinymesa_cpu.so'].nir_build_alu3
    nir_build_alu3.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu3.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu4 = _libraries['libtinymesa_cpu.so'].nir_build_alu4
    nir_build_alu4.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu4.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu_src_arr = _libraries['libtinymesa_cpu.so'].nir_build_alu_src_arr
    nir_build_alu_src_arr.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu_src_arr.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(ctypes.POINTER(struct_nir_def))]
except AttributeError:
    pass
try:
    nir_build_tex_deref_instr = _libraries['libtinymesa_cpu.so'].nir_build_tex_deref_instr
    nir_build_tex_deref_instr.restype = ctypes.POINTER(struct_nir_def)
    nir_build_tex_deref_instr.argtypes = [ctypes.POINTER(struct_nir_builder), nir_texop, ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), ctypes.c_uint32, ctypes.POINTER(struct_nir_tex_src)]
except AttributeError:
    pass
try:
    nir_builder_cf_insert = _libraries['libtinymesa_cpu.so'].nir_builder_cf_insert
    nir_builder_cf_insert.restype = None
    nir_builder_cf_insert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_builder_is_inside_cf = _libraries['libtinymesa_cpu.so'].nir_builder_is_inside_cf
    nir_builder_is_inside_cf.restype = ctypes.c_bool
    nir_builder_is_inside_cf.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_push_if = _libraries['libtinymesa_cpu.so'].nir_push_if
    nir_push_if.restype = ctypes.POINTER(struct_nir_if)
    nir_push_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_push_else = _libraries['libtinymesa_cpu.so'].nir_push_else
    nir_push_else.restype = ctypes.POINTER(struct_nir_if)
    nir_push_else.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_pop_if = _libraries['libtinymesa_cpu.so'].nir_pop_if
    nir_pop_if.restype = None
    nir_pop_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_if_phi = _libraries['libtinymesa_cpu.so'].nir_if_phi
    nir_if_phi.restype = ctypes.POINTER(struct_nir_def)
    nir_if_phi.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_push_loop = _libraries['libtinymesa_cpu.so'].nir_push_loop
    nir_push_loop.restype = ctypes.POINTER(struct_nir_loop)
    nir_push_loop.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_push_continue = _libraries['libtinymesa_cpu.so'].nir_push_continue
    nir_push_continue.restype = ctypes.POINTER(struct_nir_loop)
    nir_push_continue.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_pop_loop = _libraries['libtinymesa_cpu.so'].nir_pop_loop
    nir_pop_loop.restype = None
    nir_pop_loop.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_undef = _libraries['FIXME_STUB'].nir_undef
    nir_undef.restype = ctypes.POINTER(struct_nir_def)
    nir_undef.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_imm = _libraries['FIXME_STUB'].nir_build_imm
    nir_build_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_build_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(union_c__UA_nir_const_value)]
except AttributeError:
    pass
try:
    nir_imm_zero = _libraries['FIXME_STUB'].nir_imm_zero
    nir_imm_zero.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_zero.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_boolN_t = _libraries['FIXME_STUB'].nir_imm_boolN_t
    nir_imm_boolN_t.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_boolN_t.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_bool, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_bool = _libraries['FIXME_STUB'].nir_imm_bool
    nir_imm_bool.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_bool.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_imm_true = _libraries['FIXME_STUB'].nir_imm_true
    nir_imm_true.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_true.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_imm_false = _libraries['FIXME_STUB'].nir_imm_false
    nir_imm_false.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_false.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_imm_floatN_t = _libraries['FIXME_STUB'].nir_imm_floatN_t
    nir_imm_floatN_t.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_floatN_t.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_double, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_float16 = _libraries['FIXME_STUB'].nir_imm_float16
    nir_imm_float16.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_float16.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_float = _libraries['FIXME_STUB'].nir_imm_float
    nir_imm_float.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_float.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_double = _libraries['FIXME_STUB'].nir_imm_double
    nir_imm_double.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_double.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_imm_vec2 = _libraries['FIXME_STUB'].nir_imm_vec2
    nir_imm_vec2.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_vec2.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_vec3 = _libraries['FIXME_STUB'].nir_imm_vec3
    nir_imm_vec3.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_vec3.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float, ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_vec4 = _libraries['FIXME_STUB'].nir_imm_vec4
    nir_imm_vec4.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_vec4.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_vec4_16 = _libraries['FIXME_STUB'].nir_imm_vec4_16
    nir_imm_vec4_16.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_vec4_16.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_intN_t = _libraries['FIXME_STUB'].nir_imm_intN_t
    nir_imm_intN_t.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_intN_t.argtypes = [ctypes.POINTER(struct_nir_builder), uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_int = _libraries['FIXME_STUB'].nir_imm_int
    nir_imm_int.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_int.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_imm_int64 = _libraries['FIXME_STUB'].nir_imm_int64
    nir_imm_int64.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_int64.argtypes = [ctypes.POINTER(struct_nir_builder), int64_t]
except AttributeError:
    pass
try:
    nir_imm_ivec2 = _libraries['FIXME_STUB'].nir_imm_ivec2
    nir_imm_ivec2.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec2.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_imm_ivec3_intN = _libraries['FIXME_STUB'].nir_imm_ivec3_intN
    nir_imm_ivec3_intN.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec3_intN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_uvec2_intN = _libraries['FIXME_STUB'].nir_imm_uvec2_intN
    nir_imm_uvec2_intN.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_uvec2_intN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_uvec3_intN = _libraries['FIXME_STUB'].nir_imm_uvec3_intN
    nir_imm_uvec3_intN.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_uvec3_intN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_ivec3 = _libraries['FIXME_STUB'].nir_imm_ivec3
    nir_imm_ivec3.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec3.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_imm_ivec4_intN = _libraries['FIXME_STUB'].nir_imm_ivec4_intN
    nir_imm_ivec4_intN.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec4_intN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_ivec4 = _libraries['FIXME_STUB'].nir_imm_ivec4
    nir_imm_ivec4.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec4.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_builder_alu_instr_finish_and_insert = _libraries['libtinymesa_cpu.so'].nir_builder_alu_instr_finish_and_insert
    nir_builder_alu_instr_finish_and_insert.restype = ctypes.POINTER(struct_nir_def)
    nir_builder_alu_instr_finish_and_insert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_load_system_value = _libraries['libtinymesa_cpu.so'].nir_load_system_value
    nir_load_system_value.restype = ctypes.POINTER(struct_nir_def)
    nir_load_system_value.argtypes = [ctypes.POINTER(struct_nir_builder), nir_intrinsic_op, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_type_convert = _libraries['libtinymesa_cpu.so'].nir_type_convert
    nir_type_convert.restype = ctypes.POINTER(struct_nir_def)
    nir_type_convert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_alu_type, nir_alu_type, nir_rounding_mode]
except AttributeError:
    pass
try:
    nir_convert_to_bit_size = _libraries['FIXME_STUB'].nir_convert_to_bit_size
    nir_convert_to_bit_size.restype = ctypes.POINTER(struct_nir_def)
    nir_convert_to_bit_size.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_alu_type, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_i2iN = _libraries['FIXME_STUB'].nir_i2iN
    nir_i2iN.restype = ctypes.POINTER(struct_nir_def)
    nir_i2iN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_u2uN = _libraries['FIXME_STUB'].nir_u2uN
    nir_u2uN.restype = ctypes.POINTER(struct_nir_def)
    nir_u2uN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_b2bN = _libraries['FIXME_STUB'].nir_b2bN
    nir_b2bN.restype = ctypes.POINTER(struct_nir_def)
    nir_b2bN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_f2fN = _libraries['FIXME_STUB'].nir_f2fN
    nir_f2fN.restype = ctypes.POINTER(struct_nir_def)
    nir_f2fN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_i2b = _libraries['FIXME_STUB'].nir_i2b
    nir_i2b.restype = ctypes.POINTER(struct_nir_def)
    nir_i2b.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_b2iN = _libraries['FIXME_STUB'].nir_b2iN
    nir_b2iN.restype = ctypes.POINTER(struct_nir_def)
    nir_b2iN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_b2fN = _libraries['FIXME_STUB'].nir_b2fN
    nir_b2fN.restype = ctypes.POINTER(struct_nir_def)
    nir_b2fN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_i2fN = _libraries['FIXME_STUB'].nir_i2fN
    nir_i2fN.restype = ctypes.POINTER(struct_nir_def)
    nir_i2fN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_u2fN = _libraries['FIXME_STUB'].nir_u2fN
    nir_u2fN.restype = ctypes.POINTER(struct_nir_def)
    nir_u2fN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_f2uN = _libraries['FIXME_STUB'].nir_f2uN
    nir_f2uN.restype = ctypes.POINTER(struct_nir_def)
    nir_f2uN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_f2iN = _libraries['FIXME_STUB'].nir_f2iN
    nir_f2iN.restype = ctypes.POINTER(struct_nir_def)
    nir_f2iN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_vec = _libraries['FIXME_STUB'].nir_vec
    nir_vec.restype = ctypes.POINTER(struct_nir_def)
    nir_vec.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_vec_scalars = _libraries['libtinymesa_cpu.so'].nir_vec_scalars
    nir_vec_scalars.restype = ctypes.POINTER(struct_nir_def)
    nir_vec_scalars.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_scalar), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_mov_alu = _libraries['FIXME_STUB'].nir_mov_alu
    nir_mov_alu.restype = ctypes.POINTER(struct_nir_def)
    nir_mov_alu.argtypes = [ctypes.POINTER(struct_nir_builder), nir_alu_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_swizzle = _libraries['FIXME_STUB'].nir_swizzle
    nir_swizzle.restype = ctypes.POINTER(struct_nir_def)
    nir_swizzle.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_fdot = _libraries['FIXME_STUB'].nir_fdot
    nir_fdot.restype = ctypes.POINTER(struct_nir_def)
    nir_fdot.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_bfdot = _libraries['FIXME_STUB'].nir_bfdot
    nir_bfdot.restype = ctypes.POINTER(struct_nir_def)
    nir_bfdot.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ball_iequal = _libraries['FIXME_STUB'].nir_ball_iequal
    nir_ball_iequal.restype = ctypes.POINTER(struct_nir_def)
    nir_ball_iequal.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ball = _libraries['FIXME_STUB'].nir_ball
    nir_ball.restype = ctypes.POINTER(struct_nir_def)
    nir_ball.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_bany_inequal = _libraries['FIXME_STUB'].nir_bany_inequal
    nir_bany_inequal.restype = ctypes.POINTER(struct_nir_def)
    nir_bany_inequal.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_bany = _libraries['FIXME_STUB'].nir_bany
    nir_bany.restype = ctypes.POINTER(struct_nir_def)
    nir_bany.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_channel = _libraries['FIXME_STUB'].nir_channel
    nir_channel.restype = ctypes.POINTER(struct_nir_def)
    nir_channel.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_channel_or_undef = _libraries['FIXME_STUB'].nir_channel_or_undef
    nir_channel_or_undef.restype = ctypes.POINTER(struct_nir_def)
    nir_channel_or_undef.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_channels = _libraries['FIXME_STUB'].nir_channels
    nir_channels.restype = ctypes.POINTER(struct_nir_def)
    nir_channels.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_component_mask_t]
except AttributeError:
    pass
try:
    _nir_select_from_array_helper = _libraries['FIXME_STUB']._nir_select_from_array_helper
    _nir_select_from_array_helper.restype = ctypes.POINTER(struct_nir_def)
    _nir_select_from_array_helper.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_select_from_ssa_def_array = _libraries['FIXME_STUB'].nir_select_from_ssa_def_array
    nir_select_from_ssa_def_array.restype = ctypes.POINTER(struct_nir_def)
    nir_select_from_ssa_def_array.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.c_uint32, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_vector_extract = _libraries['FIXME_STUB'].nir_vector_extract
    nir_vector_extract.restype = ctypes.POINTER(struct_nir_def)
    nir_vector_extract.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_vector_insert_imm = _libraries['FIXME_STUB'].nir_vector_insert_imm
    nir_vector_insert_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_vector_insert_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_vector_insert = _libraries['FIXME_STUB'].nir_vector_insert
    nir_vector_insert.restype = ctypes.POINTER(struct_nir_def)
    nir_vector_insert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_replicate = _libraries['FIXME_STUB'].nir_replicate
    nir_replicate.restype = ctypes.POINTER(struct_nir_def)
    nir_replicate.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_iadd_imm = _libraries['FIXME_STUB'].nir_iadd_imm
    nir_iadd_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_iadd_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_iadd_imm_nuw = _libraries['FIXME_STUB'].nir_iadd_imm_nuw
    nir_iadd_imm_nuw.restype = ctypes.POINTER(struct_nir_def)
    nir_iadd_imm_nuw.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_iadd_nuw = _libraries['FIXME_STUB'].nir_iadd_nuw
    nir_iadd_nuw.restype = ctypes.POINTER(struct_nir_def)
    nir_iadd_nuw.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_fgt_imm = _libraries['FIXME_STUB'].nir_fgt_imm
    nir_fgt_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fgt_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_fle_imm = _libraries['FIXME_STUB'].nir_fle_imm
    nir_fle_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fle_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_isub_imm = _libraries['FIXME_STUB'].nir_isub_imm
    nir_isub_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_isub_imm.argtypes = [ctypes.POINTER(struct_nir_builder), uint64_t, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_imax_imm = _libraries['FIXME_STUB'].nir_imax_imm
    nir_imax_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_imax_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), int64_t]
except AttributeError:
    pass
try:
    nir_imin_imm = _libraries['FIXME_STUB'].nir_imin_imm
    nir_imin_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_imin_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), int64_t]
except AttributeError:
    pass
try:
    nir_umax_imm = _libraries['FIXME_STUB'].nir_umax_imm
    nir_umax_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_umax_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_umin_imm = _libraries['FIXME_STUB'].nir_umin_imm
    nir_umin_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_umin_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    _nir_mul_imm = _libraries['FIXME_STUB']._nir_mul_imm
    _nir_mul_imm.restype = ctypes.POINTER(struct_nir_def)
    _nir_mul_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_imul_imm = _libraries['FIXME_STUB'].nir_imul_imm
    nir_imul_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_imul_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_amul_imm = _libraries['FIXME_STUB'].nir_amul_imm
    nir_amul_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_amul_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_fadd_imm = _libraries['FIXME_STUB'].nir_fadd_imm
    nir_fadd_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fadd_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_fsub_imm = _libraries['FIXME_STUB'].nir_fsub_imm
    nir_fsub_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fsub_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_double, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_fmul_imm = _libraries['FIXME_STUB'].nir_fmul_imm
    nir_fmul_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fmul_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_fdiv_imm = _libraries['FIXME_STUB'].nir_fdiv_imm
    nir_fdiv_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fdiv_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_fpow_imm = _libraries['FIXME_STUB'].nir_fpow_imm
    nir_fpow_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fpow_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_iand_imm = _libraries['FIXME_STUB'].nir_iand_imm
    nir_iand_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_iand_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_test_mask = _libraries['FIXME_STUB'].nir_test_mask
    nir_test_mask.restype = ctypes.POINTER(struct_nir_def)
    nir_test_mask.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_ior_imm = _libraries['FIXME_STUB'].nir_ior_imm
    nir_ior_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ior_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_ishl_imm = _libraries['FIXME_STUB'].nir_ishl_imm
    nir_ishl_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ishl_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_ishr_imm = _libraries['FIXME_STUB'].nir_ishr_imm
    nir_ishr_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ishr_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_ushr_imm = _libraries['FIXME_STUB'].nir_ushr_imm
    nir_ushr_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ushr_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_imod_imm = _libraries['FIXME_STUB'].nir_imod_imm
    nir_imod_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_imod_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_udiv_imm = _libraries['FIXME_STUB'].nir_udiv_imm
    nir_udiv_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_udiv_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_umod_imm = _libraries['FIXME_STUB'].nir_umod_imm
    nir_umod_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_umod_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_align_imm = _libraries['FIXME_STUB'].nir_align_imm
    nir_align_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_align_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_ibfe_imm = _libraries['FIXME_STUB'].nir_ibfe_imm
    nir_ibfe_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ibfe_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_ubfe_imm = _libraries['FIXME_STUB'].nir_ubfe_imm
    nir_ubfe_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ubfe_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_ubitfield_extract_imm = _libraries['FIXME_STUB'].nir_ubitfield_extract_imm
    nir_ubitfield_extract_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ubitfield_extract_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_ibitfield_extract_imm = _libraries['FIXME_STUB'].nir_ibitfield_extract_imm
    nir_ibitfield_extract_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ibitfield_extract_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_bitfield_insert_imm = _libraries['FIXME_STUB'].nir_bitfield_insert_imm
    nir_bitfield_insert_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_bitfield_insert_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_extract_u8_imm = _libraries['FIXME_STUB'].nir_extract_u8_imm
    nir_extract_u8_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_extract_u8_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_extract_i8_imm = _libraries['FIXME_STUB'].nir_extract_i8_imm
    nir_extract_i8_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_extract_i8_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_fclamp = _libraries['FIXME_STUB'].nir_fclamp
    nir_fclamp.restype = ctypes.POINTER(struct_nir_def)
    nir_fclamp.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_iclamp = _libraries['FIXME_STUB'].nir_iclamp
    nir_iclamp.restype = ctypes.POINTER(struct_nir_def)
    nir_iclamp.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_uclamp = _libraries['FIXME_STUB'].nir_uclamp
    nir_uclamp.restype = ctypes.POINTER(struct_nir_def)
    nir_uclamp.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ffma_imm12 = _libraries['FIXME_STUB'].nir_ffma_imm12
    nir_ffma_imm12.restype = ctypes.POINTER(struct_nir_def)
    nir_ffma_imm12.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double, ctypes.c_double]
except AttributeError:
    pass
try:
    nir_ffma_imm1 = _libraries['FIXME_STUB'].nir_ffma_imm1
    nir_ffma_imm1.restype = ctypes.POINTER(struct_nir_def)
    nir_ffma_imm1.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ffma_imm2 = _libraries['FIXME_STUB'].nir_ffma_imm2
    nir_ffma_imm2.restype = ctypes.POINTER(struct_nir_def)
    nir_ffma_imm2.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_a_minus_bc = _libraries['FIXME_STUB'].nir_a_minus_bc
    nir_a_minus_bc.restype = ctypes.POINTER(struct_nir_def)
    nir_a_minus_bc.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_pack_bits = _libraries['FIXME_STUB'].nir_pack_bits
    nir_pack_bits.restype = ctypes.POINTER(struct_nir_def)
    nir_pack_bits.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_unpack_bits = _libraries['FIXME_STUB'].nir_unpack_bits
    nir_unpack_bits.restype = ctypes.POINTER(struct_nir_def)
    nir_unpack_bits.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_extract_bits = _libraries['FIXME_STUB'].nir_extract_bits
    nir_extract_bits.restype = ctypes.POINTER(struct_nir_def)
    nir_extract_bits.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_bitcast_vector = _libraries['FIXME_STUB'].nir_bitcast_vector
    nir_bitcast_vector.restype = ctypes.POINTER(struct_nir_def)
    nir_bitcast_vector.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_trim_vector = _libraries['FIXME_STUB'].nir_trim_vector
    nir_trim_vector.restype = ctypes.POINTER(struct_nir_def)
    nir_trim_vector.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_pad_vector = _libraries['FIXME_STUB'].nir_pad_vector
    nir_pad_vector.restype = ctypes.POINTER(struct_nir_def)
    nir_pad_vector.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_pad_vector_imm_int = _libraries['FIXME_STUB'].nir_pad_vector_imm_int
    nir_pad_vector_imm_int.restype = ctypes.POINTER(struct_nir_def)
    nir_pad_vector_imm_int.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_pad_vec4 = _libraries['FIXME_STUB'].nir_pad_vec4
    nir_pad_vec4.restype = ctypes.POINTER(struct_nir_def)
    nir_pad_vec4.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_resize_vector = _libraries['FIXME_STUB'].nir_resize_vector
    nir_resize_vector.restype = ctypes.POINTER(struct_nir_def)
    nir_resize_vector.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_shift_channels = _libraries['FIXME_STUB'].nir_shift_channels
    nir_shift_channels.restype = ctypes.POINTER(struct_nir_def)
    nir_shift_channels.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_ssa_for_alu_src = _libraries['libtinymesa_cpu.so'].nir_ssa_for_alu_src
    nir_ssa_for_alu_src.restype = ctypes.POINTER(struct_nir_def)
    nir_ssa_for_alu_src.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_get_ptr_bitsize = _libraries['FIXME_STUB'].nir_get_ptr_bitsize
    nir_get_ptr_bitsize.restype = ctypes.c_uint32
    nir_get_ptr_bitsize.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_build_deref_var = _libraries['FIXME_STUB'].nir_build_deref_var
    nir_build_deref_var.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_build_deref_array = _libraries['FIXME_STUB'].nir_build_deref_array
    nir_build_deref_array.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_array.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_deref_array_imm = _libraries['FIXME_STUB'].nir_build_deref_array_imm
    nir_build_deref_array_imm.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_array_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), int64_t]
except AttributeError:
    pass
try:
    nir_build_deref_ptr_as_array = _libraries['FIXME_STUB'].nir_build_deref_ptr_as_array
    nir_build_deref_ptr_as_array.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_ptr_as_array.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_deref_array_wildcard = _libraries['FIXME_STUB'].nir_build_deref_array_wildcard
    nir_build_deref_array_wildcard.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_array_wildcard.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_build_deref_struct = _libraries['FIXME_STUB'].nir_build_deref_struct
    nir_build_deref_struct.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_struct.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_deref_cast_with_alignment = _libraries['FIXME_STUB'].nir_build_deref_cast_with_alignment
    nir_build_deref_cast_with_alignment.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_cast_with_alignment.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_variable_mode, ctypes.POINTER(struct_glsl_type), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_deref_cast = _libraries['FIXME_STUB'].nir_build_deref_cast
    nir_build_deref_cast.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_cast.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_variable_mode, ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alignment_deref_cast = _libraries['FIXME_STUB'].nir_alignment_deref_cast
    nir_alignment_deref_cast.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_alignment_deref_cast.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_build_deref_follower = _libraries['FIXME_STUB'].nir_build_deref_follower
    nir_build_deref_follower.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_follower.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_load_deref_with_access = _libraries['FIXME_STUB'].nir_load_deref_with_access
    nir_load_deref_with_access.restype = ctypes.POINTER(struct_nir_def)
    nir_load_deref_with_access.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), gl_access_qualifier]
except AttributeError:
    pass
try:
    nir_load_deref = _libraries['FIXME_STUB'].nir_load_deref
    nir_load_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_load_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_store_deref_with_access = _libraries['FIXME_STUB'].nir_store_deref_with_access
    nir_store_deref_with_access.restype = None
    nir_store_deref_with_access.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, gl_access_qualifier]
except AttributeError:
    pass
try:
    nir_store_deref = _libraries['FIXME_STUB'].nir_store_deref
    nir_store_deref.restype = None
    nir_store_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_write_masked_store = _libraries['FIXME_STUB'].nir_build_write_masked_store
    nir_build_write_masked_store.restype = None
    nir_build_write_masked_store.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_write_masked_stores = _libraries['FIXME_STUB'].nir_build_write_masked_stores
    nir_build_write_masked_stores.restype = None
    nir_build_write_masked_stores.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_copy_deref_with_access = _libraries['FIXME_STUB'].nir_copy_deref_with_access
    nir_copy_deref_with_access.restype = None
    nir_copy_deref_with_access.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), gl_access_qualifier, gl_access_qualifier]
except AttributeError:
    pass
try:
    nir_copy_deref = _libraries['FIXME_STUB'].nir_copy_deref
    nir_copy_deref.restype = None
    nir_copy_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_memcpy_deref_with_access = _libraries['FIXME_STUB'].nir_memcpy_deref_with_access
    nir_memcpy_deref_with_access.restype = None
    nir_memcpy_deref_with_access.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), gl_access_qualifier, gl_access_qualifier]
except AttributeError:
    pass
try:
    nir_memcpy_deref = _libraries['FIXME_STUB'].nir_memcpy_deref
    nir_memcpy_deref.restype = None
    nir_memcpy_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_load_var = _libraries['FIXME_STUB'].nir_load_var
    nir_load_var.restype = ctypes.POINTER(struct_nir_def)
    nir_load_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_store_var = _libraries['FIXME_STUB'].nir_store_var
    nir_store_var.restype = None
    nir_store_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_copy_var = _libraries['FIXME_STUB'].nir_copy_var
    nir_copy_var.restype = None
    nir_copy_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_load_array_var = _libraries['FIXME_STUB'].nir_load_array_var
    nir_load_array_var.restype = ctypes.POINTER(struct_nir_def)
    nir_load_array_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_load_array_var_imm = _libraries['FIXME_STUB'].nir_load_array_var_imm
    nir_load_array_var_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_load_array_var_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), int64_t]
except AttributeError:
    pass
try:
    nir_store_array_var = _libraries['FIXME_STUB'].nir_store_array_var
    nir_store_array_var.restype = None
    nir_store_array_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_store_array_var_imm = _libraries['FIXME_STUB'].nir_store_array_var_imm
    nir_store_array_var_imm.restype = None
    nir_store_array_var_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), int64_t, ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_load_global = _libraries['FIXME_STUB'].nir_load_global
    nir_load_global.restype = ctypes.POINTER(struct_nir_def)
    nir_load_global.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_store_global = _libraries['FIXME_STUB'].nir_store_global
    nir_store_global.restype = None
    nir_store_global.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.POINTER(struct_nir_def), nir_component_mask_t]
except AttributeError:
    pass
try:
    nir_load_global_constant = _libraries['FIXME_STUB'].nir_load_global_constant
    nir_load_global_constant.restype = ctypes.POINTER(struct_nir_def)
    nir_load_global_constant.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_load_param = _libraries['FIXME_STUB'].nir_load_param
    nir_load_param.restype = ctypes.POINTER(struct_nir_def)
    nir_load_param.argtypes = [ctypes.POINTER(struct_nir_builder), uint32_t]
except AttributeError:
    pass
try:
    nir_decl_reg = _libraries['FIXME_STUB'].nir_decl_reg
    nir_decl_reg.restype = ctypes.POINTER(struct_nir_def)
    nir_decl_reg.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_load_reg = _libraries['FIXME_STUB'].nir_load_reg
    nir_load_reg.restype = ctypes.POINTER(struct_nir_def)
    nir_load_reg.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_store_reg = _libraries['FIXME_STUB'].nir_store_reg
    nir_store_reg.restype = None
    nir_store_reg.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_tex_src_for_ssa = _libraries['FIXME_STUB'].nir_tex_src_for_ssa
    nir_tex_src_for_ssa.restype = nir_tex_src
    nir_tex_src_for_ssa.argtypes = [nir_tex_src_type, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_deriv = _libraries['FIXME_STUB'].nir_build_deriv
    nir_build_deriv.restype = ctypes.POINTER(struct_nir_def)
    nir_build_deriv.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_intrinsic_op]
except AttributeError:
    pass
try:
    nir_ddx = _libraries['FIXME_STUB'].nir_ddx
    nir_ddx.restype = ctypes.POINTER(struct_nir_def)
    nir_ddx.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddx_fine = _libraries['FIXME_STUB'].nir_ddx_fine
    nir_ddx_fine.restype = ctypes.POINTER(struct_nir_def)
    nir_ddx_fine.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddx_coarse = _libraries['FIXME_STUB'].nir_ddx_coarse
    nir_ddx_coarse.restype = ctypes.POINTER(struct_nir_def)
    nir_ddx_coarse.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddy = _libraries['FIXME_STUB'].nir_ddy
    nir_ddy.restype = ctypes.POINTER(struct_nir_def)
    nir_ddy.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddy_fine = _libraries['FIXME_STUB'].nir_ddy_fine
    nir_ddy_fine.restype = ctypes.POINTER(struct_nir_def)
    nir_ddy_fine.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddy_coarse = _libraries['FIXME_STUB'].nir_ddy_coarse
    nir_ddy_coarse.restype = ctypes.POINTER(struct_nir_def)
    nir_ddy_coarse.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_tex_deref = _libraries['FIXME_STUB'].nir_tex_deref
    nir_tex_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_tex_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_txl_deref = _libraries['FIXME_STUB'].nir_txl_deref
    nir_txl_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_txl_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_txl_zero_deref = _libraries['FIXME_STUB'].nir_txl_zero_deref
    nir_txl_zero_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_txl_zero_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_tex_type_has_lod = _libraries['FIXME_STUB'].nir_tex_type_has_lod
    nir_tex_type_has_lod.restype = ctypes.c_bool
    nir_tex_type_has_lod.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_txf_deref = _libraries['FIXME_STUB'].nir_txf_deref
    nir_txf_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_txf_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_txf_ms_deref = _libraries['FIXME_STUB'].nir_txf_ms_deref
    nir_txf_ms_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_txf_ms_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_txs_deref = _libraries['FIXME_STUB'].nir_txs_deref
    nir_txs_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_txs_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_samples_identical_deref = _libraries['FIXME_STUB'].nir_samples_identical_deref
    nir_samples_identical_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_samples_identical_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_mask = _libraries['FIXME_STUB'].nir_mask
    nir_mask.restype = ctypes.POINTER(struct_nir_def)
    nir_mask.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_load_barycentric = _libraries['FIXME_STUB'].nir_load_barycentric
    nir_load_barycentric.restype = ctypes.POINTER(struct_nir_def)
    nir_load_barycentric.argtypes = [ctypes.POINTER(struct_nir_builder), nir_intrinsic_op, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_jump = _libraries['FIXME_STUB'].nir_jump
    nir_jump.restype = None
    nir_jump.argtypes = [ctypes.POINTER(struct_nir_builder), nir_jump_type]
except AttributeError:
    pass
try:
    nir_goto = _libraries['FIXME_STUB'].nir_goto
    nir_goto.restype = None
    nir_goto.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_goto_if = _libraries['FIXME_STUB'].nir_goto_if
    nir_goto_if.restype = None
    nir_goto_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_break_if = _libraries['FIXME_STUB'].nir_break_if
    nir_break_if.restype = None
    nir_break_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_call = _libraries['FIXME_STUB'].nir_build_call
    nir_build_call.restype = None
    nir_build_call.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_function), size_t, ctypes.POINTER(ctypes.POINTER(struct_nir_def))]
except AttributeError:
    pass
try:
    nir_build_indirect_call = _libraries['FIXME_STUB'].nir_build_indirect_call
    nir_build_indirect_call.restype = None
    nir_build_indirect_call.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_function), ctypes.POINTER(struct_nir_def), size_t, ctypes.POINTER(ctypes.POINTER(struct_nir_def))]
except AttributeError:
    pass
try:
    nir_discard = _libraries['FIXME_STUB'].nir_discard
    nir_discard.restype = None
    nir_discard.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_discard_if = _libraries['FIXME_STUB'].nir_discard_if
    nir_discard_if.restype = None
    nir_discard_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_string = _libraries['FIXME_STUB'].nir_build_string
    nir_build_string.restype = ctypes.POINTER(struct_nir_def)
    nir_build_string.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_compare_func = _libraries['libtinymesa_cpu.so'].nir_compare_func
    nir_compare_func.restype = ctypes.POINTER(struct_nir_def)
    nir_compare_func.argtypes = [ctypes.POINTER(struct_nir_builder), compare_func, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_scoped_memory_barrier = _libraries['FIXME_STUB'].nir_scoped_memory_barrier
    nir_scoped_memory_barrier.restype = None
    nir_scoped_memory_barrier.argtypes = [ctypes.POINTER(struct_nir_builder), mesa_scope, nir_memory_semantics, nir_variable_mode]
except AttributeError:
    pass
try:
    nir_gen_rect_vertices = _libraries['libtinymesa_cpu.so'].nir_gen_rect_vertices
    nir_gen_rect_vertices.restype = ctypes.POINTER(struct_nir_def)
    nir_gen_rect_vertices.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_printf_fmt = _libraries['libtinymesa_cpu.so'].nir_printf_fmt
    nir_printf_fmt.restype = None
    nir_printf_fmt.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_printf_fmt_at_px = _libraries['libtinymesa_cpu.so'].nir_printf_fmt_at_px
    nir_printf_fmt_at_px.restype = None
    nir_printf_fmt_at_px.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_call_serialized = _libraries['libtinymesa_cpu.so'].nir_call_serialized
    nir_call_serialized.restype = ctypes.POINTER(struct_nir_def)
    nir_call_serialized.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.c_uint32), size_t, ctypes.POINTER(ctypes.POINTER(struct_nir_def))]
except AttributeError:
    pass
try:
    nir_serialize = _libraries['libtinymesa_cpu.so'].nir_serialize
    nir_serialize.restype = None
    nir_serialize.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_deserialize = _libraries['libtinymesa_cpu.so'].nir_deserialize
    nir_deserialize.restype = ctypes.POINTER(struct_nir_shader)
    nir_deserialize.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass
try:
    nir_serialize_function = _libraries['libtinymesa_cpu.so'].nir_serialize_function
    nir_serialize_function.restype = None
    nir_serialize_function.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(struct_nir_function)]
except AttributeError:
    pass
try:
    nir_deserialize_function = _libraries['libtinymesa_cpu.so'].nir_deserialize_function
    nir_deserialize_function.restype = ctypes.POINTER(struct_nir_function)
    nir_deserialize_function.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_blob_reader)]
except AttributeError:
    pass

# values for enumeration 'nv_device_type'
nv_device_type__enumvalues = {
    0: 'NV_DEVICE_TYPE_IGP',
    1: 'NV_DEVICE_TYPE_DIS',
    2: 'NV_DEVICE_TYPE_SOC',
}
NV_DEVICE_TYPE_IGP = 0
NV_DEVICE_TYPE_DIS = 1
NV_DEVICE_TYPE_SOC = 2
nv_device_type = ctypes.c_uint32 # enum
class struct_nv_device_info(Structure):
    pass

class struct_nv_device_info_pci(Structure):
    pass

struct_nv_device_info_pci._pack_ = 1 # source:False
struct_nv_device_info_pci._fields_ = [
    ('domain', ctypes.c_uint16),
    ('bus', ctypes.c_ubyte),
    ('dev', ctypes.c_ubyte),
    ('func', ctypes.c_ubyte),
    ('revision_id', ctypes.c_ubyte),
]

struct_nv_device_info._pack_ = 1 # source:False
struct_nv_device_info._fields_ = [
    ('type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('device_id', ctypes.c_uint16),
    ('chipset', ctypes.c_uint16),
    ('device_name', ctypes.c_char * 64),
    ('chipset_name', ctypes.c_char * 16),
    ('pci', struct_nv_device_info_pci),
    ('sm', ctypes.c_ubyte),
    ('gpc_count', ctypes.c_ubyte),
    ('tpc_count', ctypes.c_uint16),
    ('mp_per_tpc', ctypes.c_ubyte),
    ('max_warps_per_mp', ctypes.c_ubyte),
    ('cls_copy', ctypes.c_uint16),
    ('cls_eng2d', ctypes.c_uint16),
    ('cls_eng3d', ctypes.c_uint16),
    ('cls_m2mf', ctypes.c_uint16),
    ('cls_compute', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('vram_size_B', ctypes.c_uint64),
    ('bar_size_B', ctypes.c_uint64),
]

try:
    nv_device_uuid = _libraries['FIXME_STUB'].nv_device_uuid
    nv_device_uuid.restype = None
    nv_device_uuid.argtypes = [ctypes.POINTER(struct_nv_device_info), ctypes.POINTER(ctypes.c_ubyte), size_t, ctypes.c_bool]
except AttributeError:
    pass
class struct_nak_compiler(Structure):
    pass

try:
    nak_compiler_create = _libraries['libtinymesa_cpu.so'].nak_compiler_create
    nak_compiler_create.restype = ctypes.POINTER(struct_nak_compiler)
    nak_compiler_create.argtypes = [ctypes.POINTER(struct_nv_device_info)]
except AttributeError:
    pass
try:
    nak_compiler_destroy = _libraries['libtinymesa_cpu.so'].nak_compiler_destroy
    nak_compiler_destroy.restype = None
    nak_compiler_destroy.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
try:
    nak_debug_flags = _libraries['libtinymesa_cpu.so'].nak_debug_flags
    nak_debug_flags.restype = uint64_t
    nak_debug_flags.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
try:
    nak_nir_options = _libraries['libtinymesa_cpu.so'].nak_nir_options
    nak_nir_options.restype = ctypes.POINTER(struct_nir_shader_compiler_options)
    nak_nir_options.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
try:
    nak_preprocess_nir = _libraries['libtinymesa_cpu.so'].nak_preprocess_nir
    nak_preprocess_nir.restype = None
    nak_preprocess_nir.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
try:
    nak_nir_lower_image_addrs = _libraries['FIXME_STUB'].nak_nir_lower_image_addrs
    nak_nir_lower_image_addrs.restype = ctypes.c_bool
    nak_nir_lower_image_addrs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
class struct_nak_sample_location(Structure):
    pass

struct_nak_sample_location._pack_ = 1 # source:False
struct_nak_sample_location._fields_ = [
    ('x_u4', ctypes.c_ubyte, 4),
    ('y_u4', ctypes.c_ubyte, 4),
]

class struct_nak_sample_mask(Structure):
    pass

struct_nak_sample_mask._pack_ = 1 # source:False
struct_nak_sample_mask._fields_ = [
    ('sample_mask', ctypes.c_uint16),
]

class struct_nak_fs_key(Structure):
    pass

struct_nak_fs_key._pack_ = 1 # source:False
struct_nak_fs_key._fields_ = [
    ('zs_self_dep', ctypes.c_bool),
    ('force_sample_shading', ctypes.c_bool),
    ('uses_underestimate', ctypes.c_bool),
    ('sample_info_cb', ctypes.c_ubyte),
    ('sample_locations_offset', ctypes.c_uint32),
    ('sample_masks_offset', ctypes.c_uint32),
]

try:
    nak_postprocess_nir = _libraries['libtinymesa_cpu.so'].nak_postprocess_nir
    nak_postprocess_nir.restype = None
    nak_postprocess_nir.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except AttributeError:
    pass

# values for enumeration 'nak_ts_domain'
nak_ts_domain__enumvalues = {
    0: 'NAK_TS_DOMAIN_ISOLINE',
    1: 'NAK_TS_DOMAIN_TRIANGLE',
    2: 'NAK_TS_DOMAIN_QUAD',
}
NAK_TS_DOMAIN_ISOLINE = 0
NAK_TS_DOMAIN_TRIANGLE = 1
NAK_TS_DOMAIN_QUAD = 2
nak_ts_domain = ctypes.c_uint32 # enum

# values for enumeration 'nak_ts_spacing'
nak_ts_spacing__enumvalues = {
    0: 'NAK_TS_SPACING_INTEGER',
    1: 'NAK_TS_SPACING_FRACT_ODD',
    2: 'NAK_TS_SPACING_FRACT_EVEN',
}
NAK_TS_SPACING_INTEGER = 0
NAK_TS_SPACING_FRACT_ODD = 1
NAK_TS_SPACING_FRACT_EVEN = 2
nak_ts_spacing = ctypes.c_uint32 # enum

# values for enumeration 'nak_ts_prims'
nak_ts_prims__enumvalues = {
    0: 'NAK_TS_PRIMS_POINTS',
    1: 'NAK_TS_PRIMS_LINES',
    2: 'NAK_TS_PRIMS_TRIANGLES_CW',
    3: 'NAK_TS_PRIMS_TRIANGLES_CCW',
}
NAK_TS_PRIMS_POINTS = 0
NAK_TS_PRIMS_LINES = 1
NAK_TS_PRIMS_TRIANGLES_CW = 2
NAK_TS_PRIMS_TRIANGLES_CCW = 3
nak_ts_prims = ctypes.c_uint32 # enum
class struct_nak_xfb_info(Structure):
    pass

struct_nak_xfb_info._pack_ = 1 # source:False
struct_nak_xfb_info._fields_ = [
    ('stride', ctypes.c_uint32 * 4),
    ('stream', ctypes.c_ubyte * 4),
    ('attr_count', ctypes.c_ubyte * 4),
    ('attr_index', ctypes.c_ubyte * 128 * 4),
]

class struct_nak_shader_info(Structure):
    pass

class union_nak_shader_info_0(Union):
    pass

class struct_nak_shader_info_0_cs(Structure):
    pass

struct_nak_shader_info_0_cs._pack_ = 1 # source:False
struct_nak_shader_info_0_cs._fields_ = [
    ('local_size', ctypes.c_uint16 * 3),
    ('smem_size', ctypes.c_uint16),
    ('_pad', ctypes.c_ubyte * 4),
]

class struct_nak_shader_info_0_fs(Structure):
    pass

struct_nak_shader_info_0_fs._pack_ = 1 # source:False
struct_nak_shader_info_0_fs._fields_ = [
    ('writes_depth', ctypes.c_bool),
    ('reads_sample_mask', ctypes.c_bool),
    ('post_depth_coverage', ctypes.c_bool),
    ('uses_sample_shading', ctypes.c_bool),
    ('early_fragment_tests', ctypes.c_bool),
    ('_pad', ctypes.c_ubyte * 7),
]

class struct_nak_shader_info_0_ts(Structure):
    pass

struct_nak_shader_info_0_ts._pack_ = 1 # source:False
struct_nak_shader_info_0_ts._fields_ = [
    ('domain', ctypes.c_ubyte),
    ('spacing', ctypes.c_ubyte),
    ('prims', ctypes.c_ubyte),
    ('_pad', ctypes.c_ubyte * 9),
]

union_nak_shader_info_0._pack_ = 1 # source:False
union_nak_shader_info_0._fields_ = [
    ('cs', struct_nak_shader_info_0_cs),
    ('fs', struct_nak_shader_info_0_fs),
    ('ts', struct_nak_shader_info_0_ts),
    ('_pad', ctypes.c_ubyte * 12),
]

class struct_nak_shader_info_vtg(Structure):
    pass

struct_nak_shader_info_vtg._pack_ = 1 # source:False
struct_nak_shader_info_vtg._fields_ = [
    ('writes_layer', ctypes.c_bool),
    ('writes_point_size', ctypes.c_bool),
    ('writes_vprs_table_index', ctypes.c_bool),
    ('clip_enable', ctypes.c_ubyte),
    ('cull_enable', ctypes.c_ubyte),
    ('_pad', ctypes.c_ubyte * 3),
    ('xfb', struct_nak_xfb_info),
]

struct_nak_shader_info._pack_ = 1 # source:False
struct_nak_shader_info._anonymous_ = ('_0',)
struct_nak_shader_info._fields_ = [
    ('stage', gl_shader_stage),
    ('sm', ctypes.c_ubyte),
    ('num_gprs', ctypes.c_ubyte),
    ('num_control_barriers', ctypes.c_ubyte),
    ('_pad0', ctypes.c_ubyte),
    ('max_warps_per_sm', ctypes.c_uint32),
    ('num_instrs', ctypes.c_uint32),
    ('num_static_cycles', ctypes.c_uint32),
    ('num_spills_to_mem', ctypes.c_uint32),
    ('num_fills_from_mem', ctypes.c_uint32),
    ('num_spills_to_reg', ctypes.c_uint32),
    ('num_fills_from_reg', ctypes.c_uint32),
    ('slm_size', ctypes.c_uint32),
    ('crs_size', ctypes.c_uint32),
    ('_0', union_nak_shader_info_0),
    ('vtg', struct_nak_shader_info_vtg),
    ('hdr', ctypes.c_uint32 * 32),
]

class struct_nak_shader_bin(Structure):
    pass

struct_nak_shader_bin._pack_ = 1 # source:False
struct_nak_shader_bin._fields_ = [
    ('info', struct_nak_shader_info),
    ('code_size', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('code', ctypes.POINTER(None)),
    ('asm_str', ctypes.POINTER(ctypes.c_char)),
]

try:
    nak_shader_bin_destroy = _libraries['libtinymesa_cpu.so'].nak_shader_bin_destroy
    nak_shader_bin_destroy.restype = None
    nak_shader_bin_destroy.argtypes = [ctypes.POINTER(struct_nak_shader_bin)]
except AttributeError:
    pass
try:
    nak_compile_shader = _libraries['libtinymesa_cpu.so'].nak_compile_shader
    nak_compile_shader.restype = ctypes.POINTER(struct_nak_shader_bin)
    nak_compile_shader.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except AttributeError:
    pass
class struct_nak_qmd_cbuf(Structure):
    pass

struct_nak_qmd_cbuf._pack_ = 1 # source:False
struct_nak_qmd_cbuf._fields_ = [
    ('index', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('addr', ctypes.c_uint64),
]

class struct_nak_qmd_info(Structure):
    pass

struct_nak_qmd_info._pack_ = 1 # source:False
struct_nak_qmd_info._fields_ = [
    ('addr', ctypes.c_uint64),
    ('smem_size', ctypes.c_uint16),
    ('smem_max', ctypes.c_uint16),
    ('global_size', ctypes.c_uint32 * 3),
    ('num_cbufs', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('cbufs', struct_nak_qmd_cbuf * 8),
]

try:
    nak_qmd_size_B = _libraries['libtinymesa_cpu.so'].nak_qmd_size_B
    nak_qmd_size_B.restype = uint32_t
    nak_qmd_size_B.argtypes = [ctypes.POINTER(struct_nv_device_info)]
except AttributeError:
    pass
try:
    nak_fill_qmd = _libraries['libtinymesa_cpu.so'].nak_fill_qmd
    nak_fill_qmd.restype = None
    nak_fill_qmd.argtypes = [ctypes.POINTER(struct_nv_device_info), ctypes.POINTER(struct_nak_shader_info), ctypes.POINTER(struct_nak_qmd_info), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
class struct_nak_qmd_dispatch_size_layout(Structure):
    pass

struct_nak_qmd_dispatch_size_layout._pack_ = 1 # source:False
struct_nak_qmd_dispatch_size_layout._fields_ = [
    ('x_start', ctypes.c_uint16),
    ('x_end', ctypes.c_uint16),
    ('y_start', ctypes.c_uint16),
    ('y_end', ctypes.c_uint16),
    ('z_start', ctypes.c_uint16),
    ('z_end', ctypes.c_uint16),
]

try:
    nak_get_qmd_dispatch_size_layout = _libraries['libtinymesa_cpu.so'].nak_get_qmd_dispatch_size_layout
    nak_get_qmd_dispatch_size_layout.restype = struct_nak_qmd_dispatch_size_layout
    nak_get_qmd_dispatch_size_layout.argtypes = [ctypes.POINTER(struct_nv_device_info)]
except AttributeError:
    pass
class struct_nak_qmd_cbuf_desc_layout(Structure):
    pass

struct_nak_qmd_cbuf_desc_layout._pack_ = 1 # source:False
struct_nak_qmd_cbuf_desc_layout._fields_ = [
    ('addr_shift', ctypes.c_uint16),
    ('addr_lo_start', ctypes.c_uint16),
    ('addr_lo_end', ctypes.c_uint16),
    ('addr_hi_start', ctypes.c_uint16),
    ('addr_hi_end', ctypes.c_uint16),
]

try:
    nak_get_qmd_cbuf_desc_layout = _libraries['libtinymesa_cpu.so'].nak_get_qmd_cbuf_desc_layout
    nak_get_qmd_cbuf_desc_layout.restype = struct_nak_qmd_cbuf_desc_layout
    nak_get_qmd_cbuf_desc_layout.argtypes = [ctypes.POINTER(struct_nv_device_info), uint8_t]
except AttributeError:
    pass
class struct_lp_context_ref(Structure):
    pass

class struct_LLVMOpaqueContext(Structure):
    pass

struct_lp_context_ref._pack_ = 1 # source:False
struct_lp_context_ref._fields_ = [
    ('ref', ctypes.POINTER(struct_LLVMOpaqueContext)),
    ('owned', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

lp_context_ref = struct_lp_context_ref
try:
    lp_context_create = _libraries['FIXME_STUB'].lp_context_create
    lp_context_create.restype = None
    lp_context_create.argtypes = [ctypes.POINTER(struct_lp_context_ref)]
except AttributeError:
    pass
try:
    lp_context_destroy = _libraries['FIXME_STUB'].lp_context_destroy
    lp_context_destroy.restype = None
    lp_context_destroy.argtypes = [ctypes.POINTER(struct_lp_context_ref)]
except AttributeError:
    pass
class struct_lp_passmgr(Structure):
    pass

class struct_LLVMOpaqueModule(Structure):
    pass

LLVMModuleRef = ctypes.POINTER(struct_LLVMOpaqueModule)
try:
    lp_passmgr_create = _libraries['libtinymesa_cpu.so'].lp_passmgr_create
    lp_passmgr_create.restype = ctypes.c_bool
    lp_passmgr_create.argtypes = [LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(struct_lp_passmgr))]
except AttributeError:
    pass
class struct_LLVMOpaqueTargetMachine(Structure):
    pass

LLVMTargetMachineRef = ctypes.POINTER(struct_LLVMOpaqueTargetMachine)
try:
    lp_passmgr_run = _libraries['libtinymesa_cpu.so'].lp_passmgr_run
    lp_passmgr_run.restype = None
    lp_passmgr_run.argtypes = [ctypes.POINTER(struct_lp_passmgr), LLVMModuleRef, LLVMTargetMachineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lp_passmgr_dispose = _libraries['libtinymesa_cpu.so'].lp_passmgr_dispose
    lp_passmgr_dispose.restype = None
    lp_passmgr_dispose.argtypes = [ctypes.POINTER(struct_lp_passmgr)]
except AttributeError:
    pass
class struct_lp_cached_code(Structure):
    pass

struct_lp_cached_code._pack_ = 1 # source:False
struct_lp_cached_code._fields_ = [
    ('data', ctypes.POINTER(None)),
    ('data_size', ctypes.c_uint64),
    ('dont_cache', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('jit_obj_cache', ctypes.POINTER(None)),
]

class struct_lp_generated_code(Structure):
    pass

class struct_LLVMOpaqueTargetLibraryInfotData(Structure):
    pass

LLVMTargetLibraryInfoRef = ctypes.POINTER(struct_LLVMOpaqueTargetLibraryInfotData)
try:
    gallivm_create_target_library_info = _libraries['libtinymesa_cpu.so'].gallivm_create_target_library_info
    gallivm_create_target_library_info.restype = LLVMTargetLibraryInfoRef
    gallivm_create_target_library_info.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    gallivm_dispose_target_library_info = _libraries['libtinymesa_cpu.so'].gallivm_dispose_target_library_info
    gallivm_dispose_target_library_info.restype = None
    gallivm_dispose_target_library_info.argtypes = [LLVMTargetLibraryInfoRef]
except AttributeError:
    pass
try:
    lp_set_target_options = _libraries['libtinymesa_cpu.so'].lp_set_target_options
    lp_set_target_options.restype = None
    lp_set_target_options.argtypes = []
except AttributeError:
    pass
try:
    lp_bld_init_native_targets = _libraries['libtinymesa_cpu.so'].lp_bld_init_native_targets
    lp_bld_init_native_targets.restype = None
    lp_bld_init_native_targets.argtypes = []
except AttributeError:
    pass
class struct_LLVMOpaqueExecutionEngine(Structure):
    pass

class struct_LLVMOpaqueMCJITMemoryManager(Structure):
    pass

LLVMMCJITMemoryManagerRef = ctypes.POINTER(struct_LLVMOpaqueMCJITMemoryManager)
try:
    lp_build_create_jit_compiler_for_module = _libraries['libtinymesa_cpu.so'].lp_build_create_jit_compiler_for_module
    lp_build_create_jit_compiler_for_module.restype = ctypes.c_int32
    lp_build_create_jit_compiler_for_module.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)), ctypes.POINTER(ctypes.POINTER(struct_lp_generated_code)), ctypes.POINTER(struct_lp_cached_code), LLVMModuleRef, LLVMMCJITMemoryManagerRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    lp_free_generated_code = _libraries['libtinymesa_cpu.so'].lp_free_generated_code
    lp_free_generated_code.restype = None
    lp_free_generated_code.argtypes = [ctypes.POINTER(struct_lp_generated_code)]
except AttributeError:
    pass
try:
    lp_get_default_memory_manager = _libraries['libtinymesa_cpu.so'].lp_get_default_memory_manager
    lp_get_default_memory_manager.restype = LLVMMCJITMemoryManagerRef
    lp_get_default_memory_manager.argtypes = []
except AttributeError:
    pass
try:
    lp_free_memory_manager = _libraries['libtinymesa_cpu.so'].lp_free_memory_manager
    lp_free_memory_manager.restype = None
    lp_free_memory_manager.argtypes = [LLVMMCJITMemoryManagerRef]
except AttributeError:
    pass
class struct_LLVMOpaqueValue(Structure):
    pass

LLVMValueRef = ctypes.POINTER(struct_LLVMOpaqueValue)
try:
    lp_get_called_value = _libraries['libtinymesa_cpu.so'].lp_get_called_value
    lp_get_called_value.restype = LLVMValueRef
    lp_get_called_value.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    lp_is_function = _libraries['libtinymesa_cpu.so'].lp_is_function
    lp_is_function.restype = ctypes.c_bool
    lp_is_function.argtypes = [LLVMValueRef]
except AttributeError:
    pass
try:
    lp_free_objcache = _libraries['libtinymesa_cpu.so'].lp_free_objcache
    lp_free_objcache.restype = None
    lp_free_objcache.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    lp_set_module_stack_alignment_override = _libraries['libtinymesa_cpu.so'].lp_set_module_stack_alignment_override
    lp_set_module_stack_alignment_override.restype = None
    lp_set_module_stack_alignment_override.argtypes = [LLVMModuleRef, ctypes.c_uint32]
except AttributeError:
    pass
lp_native_vector_width = 0 # Variable ctypes.c_uint32
class struct_lp_type(Structure):
    pass

struct_lp_type._pack_ = 1 # source:False
struct_lp_type._fields_ = [
    ('floating', ctypes.c_uint64, 1),
    ('fixed', ctypes.c_uint64, 1),
    ('sign', ctypes.c_uint64, 1),
    ('norm', ctypes.c_uint64, 1),
    ('signed_zero_preserve', ctypes.c_uint64, 1),
    ('nan_preserve', ctypes.c_uint64, 1),
    ('width', ctypes.c_uint64, 14),
    ('PADDING_0', ctypes.c_uint16, 12),
    ('length', ctypes.c_uint64, 14),
    ('PADDING_1', ctypes.c_uint32, 18),
]

class struct_lp_build_context(Structure):
    pass

class struct_gallivm_state(Structure):
    pass

class struct_LLVMOpaqueType(Structure):
    pass

struct_lp_build_context._pack_ = 1 # source:False
struct_lp_build_context._fields_ = [
    ('gallivm', ctypes.POINTER(struct_gallivm_state)),
    ('type', struct_lp_type),
    ('elem_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('vec_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('int_elem_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('int_vec_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('undef', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('zero', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('one', ctypes.POINTER(struct_LLVMOpaqueValue)),
]

class struct_LLVMOpaqueTargetData(Structure):
    pass

class struct_LLVMOpaqueBuilder(Structure):
    pass

class struct_LLVMOpaqueDIBuilder(Structure):
    pass

class struct_LLVMOpaqueMetadata(Structure):
    pass

class struct_lp_jit_texture(Structure):
    pass

struct_gallivm_state._pack_ = 1 # source:False
struct_gallivm_state._fields_ = [
    ('module_name', ctypes.POINTER(ctypes.c_char)),
    ('file_name', ctypes.POINTER(ctypes.c_char)),
    ('module', ctypes.POINTER(struct_LLVMOpaqueModule)),
    ('target', ctypes.POINTER(struct_LLVMOpaqueTargetData)),
    ('engine', ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)),
    ('passmgr', ctypes.POINTER(struct_lp_passmgr)),
    ('memorymgr', ctypes.POINTER(struct_LLVMOpaqueMCJITMemoryManager)),
    ('code', ctypes.POINTER(struct_lp_generated_code)),
    ('context', ctypes.POINTER(struct_LLVMOpaqueContext)),
    ('builder', ctypes.POINTER(struct_LLVMOpaqueBuilder)),
    ('di_builder', ctypes.POINTER(struct_LLVMOpaqueDIBuilder)),
    ('cache', ctypes.POINTER(struct_lp_cached_code)),
    ('compiled', ctypes.c_uint32),
    ('coro_malloc_hook', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('coro_free_hook', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('debug_printf_hook', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('coro_malloc_hook_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('coro_free_hook_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('di_function', ctypes.POINTER(struct_LLVMOpaqueMetadata)),
    ('file', ctypes.POINTER(struct_LLVMOpaqueMetadata)),
    ('get_time_hook', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('texture_descriptor', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('texture_dynamic_state', ctypes.POINTER(struct_lp_jit_texture)),
    ('sampler_descriptor', ctypes.POINTER(struct_LLVMOpaqueValue)),
]

class struct_util_format_description(Structure):
    pass

class struct_util_format_block(Structure):
    pass

struct_util_format_block._pack_ = 1 # source:False
struct_util_format_block._fields_ = [
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('depth', ctypes.c_uint32),
    ('bits', ctypes.c_uint32),
]


# values for enumeration 'util_format_layout'
util_format_layout__enumvalues = {
    0: 'UTIL_FORMAT_LAYOUT_PLAIN',
    1: 'UTIL_FORMAT_LAYOUT_SUBSAMPLED',
    2: 'UTIL_FORMAT_LAYOUT_S3TC',
    3: 'UTIL_FORMAT_LAYOUT_RGTC',
    4: 'UTIL_FORMAT_LAYOUT_ETC',
    5: 'UTIL_FORMAT_LAYOUT_BPTC',
    6: 'UTIL_FORMAT_LAYOUT_ASTC',
    7: 'UTIL_FORMAT_LAYOUT_ATC',
    8: 'UTIL_FORMAT_LAYOUT_PLANAR2',
    9: 'UTIL_FORMAT_LAYOUT_PLANAR3',
    10: 'UTIL_FORMAT_LAYOUT_FXT1',
    11: 'UTIL_FORMAT_LAYOUT_OTHER',
}
UTIL_FORMAT_LAYOUT_PLAIN = 0
UTIL_FORMAT_LAYOUT_SUBSAMPLED = 1
UTIL_FORMAT_LAYOUT_S3TC = 2
UTIL_FORMAT_LAYOUT_RGTC = 3
UTIL_FORMAT_LAYOUT_ETC = 4
UTIL_FORMAT_LAYOUT_BPTC = 5
UTIL_FORMAT_LAYOUT_ASTC = 6
UTIL_FORMAT_LAYOUT_ATC = 7
UTIL_FORMAT_LAYOUT_PLANAR2 = 8
UTIL_FORMAT_LAYOUT_PLANAR3 = 9
UTIL_FORMAT_LAYOUT_FXT1 = 10
UTIL_FORMAT_LAYOUT_OTHER = 11
util_format_layout = ctypes.c_uint32 # enum
class struct_util_format_channel_description(Structure):
    pass

struct_util_format_channel_description._pack_ = 1 # source:False
struct_util_format_channel_description._fields_ = [
    ('type', ctypes.c_uint32, 5),
    ('normalized', ctypes.c_uint32, 1),
    ('pure_integer', ctypes.c_uint32, 1),
    ('size', ctypes.c_uint32, 9),
    ('shift', ctypes.c_uint32, 16),
]


# values for enumeration 'util_format_colorspace'
util_format_colorspace__enumvalues = {
    0: 'UTIL_FORMAT_COLORSPACE_RGB',
    1: 'UTIL_FORMAT_COLORSPACE_SRGB',
    2: 'UTIL_FORMAT_COLORSPACE_YUV',
    3: 'UTIL_FORMAT_COLORSPACE_ZS',
}
UTIL_FORMAT_COLORSPACE_RGB = 0
UTIL_FORMAT_COLORSPACE_SRGB = 1
UTIL_FORMAT_COLORSPACE_YUV = 2
UTIL_FORMAT_COLORSPACE_ZS = 3
util_format_colorspace = ctypes.c_uint32 # enum
class union_util_format_description_0(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('srgb_equivalent', pipe_format),
    ('linear_equivalent', pipe_format),
     ]

struct_util_format_description._pack_ = 1 # source:False
struct_util_format_description._anonymous_ = ('_0',)
struct_util_format_description._fields_ = [
    ('format', pipe_format),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('short_name', ctypes.POINTER(ctypes.c_char)),
    ('block', struct_util_format_block),
    ('layout', util_format_layout),
    ('nr_channels', ctypes.c_uint32, 3),
    ('is_array', ctypes.c_uint32, 1),
    ('is_bitmask', ctypes.c_uint32, 1),
    ('is_mixed', ctypes.c_uint32, 1),
    ('is_unorm', ctypes.c_uint32, 1),
    ('is_snorm', ctypes.c_uint32, 1),
    ('PADDING_1', ctypes.c_uint32, 24),
    ('channel', struct_util_format_channel_description * 4),
    ('swizzle', ctypes.c_ubyte * 4),
    ('colorspace', util_format_colorspace),
    ('_0', union_util_format_description_0),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

try:
    lp_type_from_format_desc = _libraries['FIXME_STUB'].lp_type_from_format_desc
    lp_type_from_format_desc.restype = None
    lp_type_from_format_desc.argtypes = [ctypes.POINTER(struct_lp_type), ctypes.POINTER(struct_util_format_description)]
except AttributeError:
    pass
try:
    lp_type_from_format = _libraries['FIXME_STUB'].lp_type_from_format
    lp_type_from_format.restype = None
    lp_type_from_format.argtypes = [ctypes.POINTER(struct_lp_type), pipe_format]
except AttributeError:
    pass
try:
    lp_type_width = _libraries['FIXME_STUB'].lp_type_width
    lp_type_width.restype = ctypes.c_uint32
    lp_type_width.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_type_float = _libraries['FIXME_STUB'].lp_type_float
    lp_type_float.restype = struct_lp_type
    lp_type_float.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_type_float_vec = _libraries['FIXME_STUB'].lp_type_float_vec
    lp_type_float_vec.restype = struct_lp_type
    lp_type_float_vec.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_type_int = _libraries['FIXME_STUB'].lp_type_int
    lp_type_int.restype = struct_lp_type
    lp_type_int.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_type_int_vec = _libraries['FIXME_STUB'].lp_type_int_vec
    lp_type_int_vec.restype = struct_lp_type
    lp_type_int_vec.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_type_uint = _libraries['FIXME_STUB'].lp_type_uint
    lp_type_uint.restype = struct_lp_type
    lp_type_uint.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_type_uint_vec = _libraries['FIXME_STUB'].lp_type_uint_vec
    lp_type_uint_vec.restype = struct_lp_type
    lp_type_uint_vec.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_type_unorm = _libraries['FIXME_STUB'].lp_type_unorm
    lp_type_unorm.restype = struct_lp_type
    lp_type_unorm.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_type_fixed = _libraries['FIXME_STUB'].lp_type_fixed
    lp_type_fixed.restype = struct_lp_type
    lp_type_fixed.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_type_ufixed = _libraries['FIXME_STUB'].lp_type_ufixed
    lp_type_ufixed.restype = struct_lp_type
    lp_type_ufixed.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
LLVMTypeRef = ctypes.POINTER(struct_LLVMOpaqueType)
try:
    lp_build_elem_type = _libraries['libtinymesa_cpu.so'].lp_build_elem_type
    lp_build_elem_type.restype = LLVMTypeRef
    lp_build_elem_type.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_build_vec_type = _libraries['libtinymesa_cpu.so'].lp_build_vec_type
    lp_build_vec_type.restype = LLVMTypeRef
    lp_build_vec_type.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_check_elem_type = _libraries['libtinymesa_cpu.so'].lp_check_elem_type
    lp_check_elem_type.restype = ctypes.c_bool
    lp_check_elem_type.argtypes = [struct_lp_type, LLVMTypeRef]
except AttributeError:
    pass
try:
    lp_check_vec_type = _libraries['libtinymesa_cpu.so'].lp_check_vec_type
    lp_check_vec_type.restype = ctypes.c_bool
    lp_check_vec_type.argtypes = [struct_lp_type, LLVMTypeRef]
except AttributeError:
    pass
try:
    lp_check_value = _libraries['libtinymesa_cpu.so'].lp_check_value
    lp_check_value.restype = ctypes.c_bool
    lp_check_value.argtypes = [struct_lp_type, LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_int_elem_type = _libraries['libtinymesa_cpu.so'].lp_build_int_elem_type
    lp_build_int_elem_type.restype = LLVMTypeRef
    lp_build_int_elem_type.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_build_int_vec_type = _libraries['libtinymesa_cpu.so'].lp_build_int_vec_type
    lp_build_int_vec_type.restype = LLVMTypeRef
    lp_build_int_vec_type.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_float32_vec4_type = _libraries['FIXME_STUB'].lp_float32_vec4_type
    lp_float32_vec4_type.restype = struct_lp_type
    lp_float32_vec4_type.argtypes = []
except AttributeError:
    pass
try:
    lp_int32_vec4_type = _libraries['FIXME_STUB'].lp_int32_vec4_type
    lp_int32_vec4_type.restype = struct_lp_type
    lp_int32_vec4_type.argtypes = []
except AttributeError:
    pass
try:
    lp_unorm8_vec4_type = _libraries['FIXME_STUB'].lp_unorm8_vec4_type
    lp_unorm8_vec4_type.restype = struct_lp_type
    lp_unorm8_vec4_type.argtypes = []
except AttributeError:
    pass
try:
    lp_elem_type = _libraries['libtinymesa_cpu.so'].lp_elem_type
    lp_elem_type.restype = struct_lp_type
    lp_elem_type.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_uint_type = _libraries['libtinymesa_cpu.so'].lp_uint_type
    lp_uint_type.restype = struct_lp_type
    lp_uint_type.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_int_type = _libraries['libtinymesa_cpu.so'].lp_int_type
    lp_int_type.restype = struct_lp_type
    lp_int_type.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_wider_type = _libraries['libtinymesa_cpu.so'].lp_wider_type
    lp_wider_type.restype = struct_lp_type
    lp_wider_type.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_sizeof_llvm_type = _libraries['libtinymesa_cpu.so'].lp_sizeof_llvm_type
    lp_sizeof_llvm_type.restype = ctypes.c_uint32
    lp_sizeof_llvm_type.argtypes = [LLVMTypeRef]
except AttributeError:
    pass

# values for enumeration 'c__EA_LLVMTypeKind'
c__EA_LLVMTypeKind__enumvalues = {
    0: 'LLVMVoidTypeKind',
    1: 'LLVMHalfTypeKind',
    2: 'LLVMFloatTypeKind',
    3: 'LLVMDoubleTypeKind',
    4: 'LLVMX86_FP80TypeKind',
    5: 'LLVMFP128TypeKind',
    6: 'LLVMPPC_FP128TypeKind',
    7: 'LLVMLabelTypeKind',
    8: 'LLVMIntegerTypeKind',
    9: 'LLVMFunctionTypeKind',
    10: 'LLVMStructTypeKind',
    11: 'LLVMArrayTypeKind',
    12: 'LLVMPointerTypeKind',
    13: 'LLVMVectorTypeKind',
    14: 'LLVMMetadataTypeKind',
    16: 'LLVMTokenTypeKind',
    17: 'LLVMScalableVectorTypeKind',
    18: 'LLVMBFloatTypeKind',
    19: 'LLVMX86_AMXTypeKind',
    20: 'LLVMTargetExtTypeKind',
}
LLVMVoidTypeKind = 0
LLVMHalfTypeKind = 1
LLVMFloatTypeKind = 2
LLVMDoubleTypeKind = 3
LLVMX86_FP80TypeKind = 4
LLVMFP128TypeKind = 5
LLVMPPC_FP128TypeKind = 6
LLVMLabelTypeKind = 7
LLVMIntegerTypeKind = 8
LLVMFunctionTypeKind = 9
LLVMStructTypeKind = 10
LLVMArrayTypeKind = 11
LLVMPointerTypeKind = 12
LLVMVectorTypeKind = 13
LLVMMetadataTypeKind = 14
LLVMTokenTypeKind = 16
LLVMScalableVectorTypeKind = 17
LLVMBFloatTypeKind = 18
LLVMX86_AMXTypeKind = 19
LLVMTargetExtTypeKind = 20
c__EA_LLVMTypeKind = ctypes.c_uint32 # enum
LLVMTypeKind = c__EA_LLVMTypeKind
LLVMTypeKind__enumvalues = c__EA_LLVMTypeKind__enumvalues
try:
    lp_typekind_name = _libraries['libtinymesa_cpu.so'].lp_typekind_name
    lp_typekind_name.restype = ctypes.POINTER(ctypes.c_char)
    lp_typekind_name.argtypes = [LLVMTypeKind]
except AttributeError:
    pass
try:
    lp_dump_llvmtype = _libraries['libtinymesa_cpu.so'].lp_dump_llvmtype
    lp_dump_llvmtype.restype = None
    lp_dump_llvmtype.argtypes = [LLVMTypeRef]
except AttributeError:
    pass
try:
    lp_build_context_init = _libraries['libtinymesa_cpu.so'].lp_build_context_init
    lp_build_context_init.restype = None
    lp_build_context_init.argtypes = [ctypes.POINTER(struct_lp_build_context), ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_build_count_ir_module = _libraries['libtinymesa_cpu.so'].lp_build_count_ir_module
    lp_build_count_ir_module.restype = ctypes.c_uint32
    lp_build_count_ir_module.argtypes = [LLVMModuleRef]
except AttributeError:
    pass
class union_lp_jit_texture_0(Union):
    pass

class struct_lp_jit_texture_0_0(Structure):
    pass

struct_lp_jit_texture_0_0._pack_ = 1 # source:False
struct_lp_jit_texture_0_0._fields_ = [
    ('row_stride', ctypes.c_uint32 * 16),
    ('img_stride', ctypes.c_uint32 * 16),
]

union_lp_jit_texture_0._pack_ = 1 # source:False
union_lp_jit_texture_0._anonymous_ = ('_0',)
union_lp_jit_texture_0._fields_ = [
    ('_0', struct_lp_jit_texture_0_0),
    ('residency', ctypes.POINTER(None)),
    ('PADDING_0', ctypes.c_ubyte * 120),
]

struct_lp_jit_texture._pack_ = 1 # source:False
struct_lp_jit_texture._anonymous_ = ('_0',)
struct_lp_jit_texture._fields_ = [
    ('base', ctypes.POINTER(None)),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint16),
    ('depth', ctypes.c_uint16),
    ('_0', union_lp_jit_texture_0),
    ('first_level', ctypes.c_ubyte),
    ('last_level', ctypes.c_ubyte),
    ('mip_offsets', ctypes.c_uint32 * 16),
    ('sampler_index', ctypes.c_uint32),
]

try:
    lp_build_init_native_width = _libraries['libtinymesa_cpu.so'].lp_build_init_native_width
    lp_build_init_native_width.restype = ctypes.c_uint32
    lp_build_init_native_width.argtypes = []
except AttributeError:
    pass
try:
    lp_build_init = _libraries['libtinymesa_cpu.so'].lp_build_init
    lp_build_init.restype = ctypes.c_bool
    lp_build_init.argtypes = []
except AttributeError:
    pass
try:
    gallivm_create = _libraries['libtinymesa_cpu.so'].gallivm_create
    gallivm_create.restype = ctypes.POINTER(struct_gallivm_state)
    gallivm_create.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_lp_context_ref), ctypes.POINTER(struct_lp_cached_code)]
except AttributeError:
    pass
try:
    gallivm_destroy = _libraries['libtinymesa_cpu.so'].gallivm_destroy
    gallivm_destroy.restype = None
    gallivm_destroy.argtypes = [ctypes.POINTER(struct_gallivm_state)]
except AttributeError:
    pass
try:
    gallivm_free_ir = _libraries['libtinymesa_cpu.so'].gallivm_free_ir
    gallivm_free_ir.restype = None
    gallivm_free_ir.argtypes = [ctypes.POINTER(struct_gallivm_state)]
except AttributeError:
    pass
try:
    gallivm_verify_function = _libraries['libtinymesa_cpu.so'].gallivm_verify_function
    gallivm_verify_function.restype = None
    gallivm_verify_function.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError:
    pass
try:
    gallivm_add_global_mapping = _libraries['libtinymesa_cpu.so'].gallivm_add_global_mapping
    gallivm_add_global_mapping.restype = None
    gallivm_add_global_mapping.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    gallivm_compile_module = _libraries['libtinymesa_cpu.so'].gallivm_compile_module
    gallivm_compile_module.restype = None
    gallivm_compile_module.argtypes = [ctypes.POINTER(struct_gallivm_state)]
except AttributeError:
    pass
func_pointer = ctypes.CFUNCTYPE(None)
try:
    gallivm_jit_function = _libraries['libtinymesa_cpu.so'].gallivm_jit_function
    gallivm_jit_function.restype = func_pointer
    gallivm_jit_function.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    gallivm_stub_func = _libraries['libtinymesa_cpu.so'].gallivm_stub_func
    gallivm_stub_func.restype = None
    gallivm_stub_func.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError:
    pass
try:
    gallivm_get_perf_flags = _libraries['libtinymesa_cpu.so'].gallivm_get_perf_flags
    gallivm_get_perf_flags.restype = ctypes.c_uint32
    gallivm_get_perf_flags.argtypes = []
except AttributeError:
    pass
try:
    lp_init_clock_hook = _libraries['libtinymesa_cpu.so'].lp_init_clock_hook
    lp_init_clock_hook.restype = None
    lp_init_clock_hook.argtypes = [ctypes.POINTER(struct_gallivm_state)]
except AttributeError:
    pass
try:
    lp_init_env_options = _libraries['libtinymesa_cpu.so'].lp_init_env_options
    lp_init_env_options.restype = None
    lp_init_env_options.argtypes = []
except AttributeError:
    pass
try:
    lp_bld_ppc_disable_denorms = _libraries['FIXME_STUB'].lp_bld_ppc_disable_denorms
    lp_bld_ppc_disable_denorms.restype = None
    lp_bld_ppc_disable_denorms.argtypes = []
except AttributeError:
    pass
class struct_lp_build_skip_context(Structure):
    pass

class struct_LLVMOpaqueBasicBlock(Structure):
    pass

struct_lp_build_skip_context._pack_ = 1 # source:False
struct_lp_build_skip_context._fields_ = [
    ('gallivm', ctypes.POINTER(struct_gallivm_state)),
    ('block', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
]

try:
    lp_build_flow_skip_begin = _libraries['libtinymesa_cpu.so'].lp_build_flow_skip_begin
    lp_build_flow_skip_begin.restype = None
    lp_build_flow_skip_begin.argtypes = [ctypes.POINTER(struct_lp_build_skip_context), ctypes.POINTER(struct_gallivm_state)]
except AttributeError:
    pass
try:
    lp_build_flow_skip_cond_break = _libraries['libtinymesa_cpu.so'].lp_build_flow_skip_cond_break
    lp_build_flow_skip_cond_break.restype = None
    lp_build_flow_skip_cond_break.argtypes = [ctypes.POINTER(struct_lp_build_skip_context), LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_flow_skip_end = _libraries['libtinymesa_cpu.so'].lp_build_flow_skip_end
    lp_build_flow_skip_end.restype = None
    lp_build_flow_skip_end.argtypes = [ctypes.POINTER(struct_lp_build_skip_context)]
except AttributeError:
    pass
class struct_lp_build_mask_context(Structure):
    pass

struct_lp_build_mask_context._pack_ = 1 # source:False
struct_lp_build_mask_context._fields_ = [
    ('skip', struct_lp_build_skip_context),
    ('reg_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('var_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('var', ctypes.POINTER(struct_LLVMOpaqueValue)),
]

try:
    lp_build_mask_begin = _libraries['libtinymesa_cpu.so'].lp_build_mask_begin
    lp_build_mask_begin.restype = None
    lp_build_mask_begin.argtypes = [ctypes.POINTER(struct_lp_build_mask_context), ctypes.POINTER(struct_gallivm_state), struct_lp_type, LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_mask_value = _libraries['libtinymesa_cpu.so'].lp_build_mask_value
    lp_build_mask_value.restype = LLVMValueRef
    lp_build_mask_value.argtypes = [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError:
    pass
try:
    lp_build_mask_update = _libraries['libtinymesa_cpu.so'].lp_build_mask_update
    lp_build_mask_update.restype = None
    lp_build_mask_update.argtypes = [ctypes.POINTER(struct_lp_build_mask_context), LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_mask_force = _libraries['libtinymesa_cpu.so'].lp_build_mask_force
    lp_build_mask_force.restype = None
    lp_build_mask_force.argtypes = [ctypes.POINTER(struct_lp_build_mask_context), LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_mask_check = _libraries['libtinymesa_cpu.so'].lp_build_mask_check
    lp_build_mask_check.restype = None
    lp_build_mask_check.argtypes = [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError:
    pass
try:
    lp_build_mask_end = _libraries['libtinymesa_cpu.so'].lp_build_mask_end
    lp_build_mask_end.restype = LLVMValueRef
    lp_build_mask_end.argtypes = [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError:
    pass
class struct_lp_build_loop_state(Structure):
    pass

struct_lp_build_loop_state._pack_ = 1 # source:False
struct_lp_build_loop_state._fields_ = [
    ('block', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
    ('counter_var', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('counter', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('counter_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('gallivm', ctypes.POINTER(struct_gallivm_state)),
]

try:
    lp_build_loop_begin = _libraries['libtinymesa_cpu.so'].lp_build_loop_begin
    lp_build_loop_begin.restype = None
    lp_build_loop_begin.argtypes = [ctypes.POINTER(struct_lp_build_loop_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_loop_end = _libraries['libtinymesa_cpu.so'].lp_build_loop_end
    lp_build_loop_end.restype = None
    lp_build_loop_end.argtypes = [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_loop_force_set_counter = _libraries['libtinymesa_cpu.so'].lp_build_loop_force_set_counter
    lp_build_loop_force_set_counter.restype = None
    lp_build_loop_force_set_counter.argtypes = [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_loop_force_reload_counter = _libraries['libtinymesa_cpu.so'].lp_build_loop_force_reload_counter
    lp_build_loop_force_reload_counter.restype = None
    lp_build_loop_force_reload_counter.argtypes = [ctypes.POINTER(struct_lp_build_loop_state)]
except AttributeError:
    pass

# values for enumeration 'c__EA_LLVMIntPredicate'
c__EA_LLVMIntPredicate__enumvalues = {
    32: 'LLVMIntEQ',
    33: 'LLVMIntNE',
    34: 'LLVMIntUGT',
    35: 'LLVMIntUGE',
    36: 'LLVMIntULT',
    37: 'LLVMIntULE',
    38: 'LLVMIntSGT',
    39: 'LLVMIntSGE',
    40: 'LLVMIntSLT',
    41: 'LLVMIntSLE',
}
LLVMIntEQ = 32
LLVMIntNE = 33
LLVMIntUGT = 34
LLVMIntUGE = 35
LLVMIntULT = 36
LLVMIntULE = 37
LLVMIntSGT = 38
LLVMIntSGE = 39
LLVMIntSLT = 40
LLVMIntSLE = 41
c__EA_LLVMIntPredicate = ctypes.c_uint32 # enum
LLVMIntPredicate = c__EA_LLVMIntPredicate
LLVMIntPredicate__enumvalues = c__EA_LLVMIntPredicate__enumvalues
try:
    lp_build_loop_end_cond = _libraries['libtinymesa_cpu.so'].lp_build_loop_end_cond
    lp_build_loop_end_cond.restype = None
    lp_build_loop_end_cond.argtypes = [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef, LLVMValueRef, LLVMIntPredicate]
except AttributeError:
    pass
class struct_lp_build_for_loop_state(Structure):
    pass

struct_lp_build_for_loop_state._pack_ = 1 # source:False
struct_lp_build_for_loop_state._fields_ = [
    ('begin', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
    ('body', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
    ('exit', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
    ('counter_var', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('counter', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('counter_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('step', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('cond', LLVMIntPredicate),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('end', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('gallivm', ctypes.POINTER(struct_gallivm_state)),
]

try:
    lp_build_for_loop_begin = _libraries['libtinymesa_cpu.so'].lp_build_for_loop_begin
    lp_build_for_loop_begin.restype = None
    lp_build_for_loop_begin.argtypes = [ctypes.POINTER(struct_lp_build_for_loop_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMIntPredicate, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_for_loop_end = _libraries['libtinymesa_cpu.so'].lp_build_for_loop_end
    lp_build_for_loop_end.restype = None
    lp_build_for_loop_end.argtypes = [ctypes.POINTER(struct_lp_build_for_loop_state)]
except AttributeError:
    pass
class struct_lp_build_if_state(Structure):
    pass

struct_lp_build_if_state._pack_ = 1 # source:False
struct_lp_build_if_state._fields_ = [
    ('gallivm', ctypes.POINTER(struct_gallivm_state)),
    ('condition', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('entry_block', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
    ('true_block', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
    ('false_block', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
    ('merge_block', ctypes.POINTER(struct_LLVMOpaqueBasicBlock)),
]

try:
    lp_build_if = _libraries['libtinymesa_cpu.so'].lp_build_if
    lp_build_if.restype = None
    lp_build_if.argtypes = [ctypes.POINTER(struct_lp_build_if_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_else = _libraries['libtinymesa_cpu.so'].lp_build_else
    lp_build_else.restype = None
    lp_build_else.argtypes = [ctypes.POINTER(struct_lp_build_if_state)]
except AttributeError:
    pass
try:
    lp_build_endif = _libraries['libtinymesa_cpu.so'].lp_build_endif
    lp_build_endif.restype = None
    lp_build_endif.argtypes = [ctypes.POINTER(struct_lp_build_if_state)]
except AttributeError:
    pass
LLVMBasicBlockRef = ctypes.POINTER(struct_LLVMOpaqueBasicBlock)
try:
    lp_build_insert_new_block = _libraries['libtinymesa_cpu.so'].lp_build_insert_new_block
    lp_build_insert_new_block.restype = LLVMBasicBlockRef
    lp_build_insert_new_block.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
LLVMBuilderRef = ctypes.POINTER(struct_LLVMOpaqueBuilder)
try:
    lp_create_builder_at_entry = _libraries['libtinymesa_cpu.so'].lp_create_builder_at_entry
    lp_create_builder_at_entry.restype = LLVMBuilderRef
    lp_create_builder_at_entry.argtypes = [ctypes.POINTER(struct_gallivm_state)]
except AttributeError:
    pass
try:
    lp_build_alloca = _libraries['libtinymesa_cpu.so'].lp_build_alloca
    lp_build_alloca.restype = LLVMValueRef
    lp_build_alloca.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lp_build_alloca_undef = _libraries['libtinymesa_cpu.so'].lp_build_alloca_undef
    lp_build_alloca_undef.restype = LLVMValueRef
    lp_build_alloca_undef.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lp_build_array_alloca = _libraries['libtinymesa_cpu.so'].lp_build_array_alloca
    lp_build_array_alloca.restype = LLVMValueRef
    lp_build_array_alloca.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
class struct_lp_build_tgsi_params(Structure):
    pass

class struct_lp_bld_tgsi_system_values(Structure):
    pass

class struct_lp_build_sampler_soa(Structure):
    pass

class struct_tgsi_shader_info(Structure):
    pass

class struct_lp_build_gs_iface(Structure):
    pass

class struct_lp_build_tcs_iface(Structure):
    pass

class struct_lp_build_tes_iface(Structure):
    pass

class struct_lp_build_mesh_iface(Structure):
    pass

class struct_lp_build_image_soa(Structure):
    pass

class struct_lp_build_coro_suspend_info(Structure):
    pass

class struct_lp_build_fs_iface(Structure):
    pass

struct_lp_build_tgsi_params._pack_ = 1 # source:False
struct_lp_build_tgsi_params._fields_ = [
    ('type', struct_lp_type),
    ('mask', ctypes.POINTER(struct_lp_build_mask_context)),
    ('consts_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('const_sizes_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('system_values', ctypes.POINTER(struct_lp_bld_tgsi_system_values)),
    ('inputs', ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue) * 4)),
    ('num_inputs', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('context_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('context_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('resources_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('resources_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('thread_data_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('thread_data_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('sampler', ctypes.POINTER(struct_lp_build_sampler_soa)),
    ('info', ctypes.POINTER(struct_tgsi_shader_info)),
    ('gs_iface', ctypes.POINTER(struct_lp_build_gs_iface)),
    ('tcs_iface', ctypes.POINTER(struct_lp_build_tcs_iface)),
    ('tes_iface', ctypes.POINTER(struct_lp_build_tes_iface)),
    ('mesh_iface', ctypes.POINTER(struct_lp_build_mesh_iface)),
    ('ssbo_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('ssbo_sizes_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('image', ctypes.POINTER(struct_lp_build_image_soa)),
    ('shared_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('payload_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('coro', ctypes.POINTER(struct_lp_build_coro_suspend_info)),
    ('fs_iface', ctypes.POINTER(struct_lp_build_fs_iface)),
    ('gs_vertex_streams', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('current_func', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('fns', ctypes.POINTER(struct_hash_table)),
    ('scratch_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('call_context_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
]

struct_lp_bld_tgsi_system_values._pack_ = 1 # source:False
struct_lp_bld_tgsi_system_values._fields_ = [
    ('instance_id', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('base_instance', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('vertex_id', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('vertex_id_nobase', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('prim_id', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('basevertex', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('firstvertex', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('invocation_id', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('draw_id', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('thread_id', ctypes.POINTER(struct_LLVMOpaqueValue) * 3),
    ('block_id', ctypes.POINTER(struct_LLVMOpaqueValue) * 3),
    ('grid_size', ctypes.POINTER(struct_LLVMOpaqueValue) * 3),
    ('front_facing', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('work_dim', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('block_size', ctypes.POINTER(struct_LLVMOpaqueValue) * 3),
    ('tess_coord', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('tess_outer', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('tess_inner', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('vertices_in', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('sample_id', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('sample_pos_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('sample_pos', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('sample_mask_in', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('view_index', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('subgroup_id', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('num_subgroups', ctypes.POINTER(struct_LLVMOpaqueValue)),
]

class struct_lp_sampler_params(Structure):
    pass

class struct_lp_sampler_size_query_params(Structure):
    pass

struct_lp_build_sampler_soa._pack_ = 1 # source:False
struct_lp_build_sampler_soa._fields_ = [
    ('emit_tex_sample', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_sampler_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_params))),
    ('emit_size_query', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_sampler_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params))),
]

class struct_lp_derivatives(Structure):
    pass

struct_lp_sampler_params._pack_ = 1 # source:False
struct_lp_sampler_params._fields_ = [
    ('type', struct_lp_type),
    ('texture_index', ctypes.c_uint32),
    ('sampler_index', ctypes.c_uint32),
    ('texture_index_offset', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('sample_key', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('resources_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('resources_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('thread_data_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('thread_data_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('coords', ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('offsets', ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('ms_index', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('lod', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('min_lod', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('derivs', ctypes.POINTER(struct_lp_derivatives)),
    ('texel', ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('texture_resource', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('sampler_resource', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('exec_mask', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('exec_mask_nz', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

struct_lp_derivatives._pack_ = 1 # source:False
struct_lp_derivatives._fields_ = [
    ('ddx', ctypes.POINTER(struct_LLVMOpaqueValue) * 3),
    ('ddy', ctypes.POINTER(struct_LLVMOpaqueValue) * 3),
]


# values for enumeration 'lp_sampler_lod_property'
lp_sampler_lod_property__enumvalues = {
    0: 'LP_SAMPLER_LOD_SCALAR',
    1: 'LP_SAMPLER_LOD_PER_ELEMENT',
    2: 'LP_SAMPLER_LOD_PER_QUAD',
}
LP_SAMPLER_LOD_SCALAR = 0
LP_SAMPLER_LOD_PER_ELEMENT = 1
LP_SAMPLER_LOD_PER_QUAD = 2
lp_sampler_lod_property = ctypes.c_uint32 # enum
struct_lp_sampler_size_query_params._pack_ = 1 # source:False
struct_lp_sampler_size_query_params._fields_ = [
    ('int_type', struct_lp_type),
    ('texture_unit', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('texture_unit_offset', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('target', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('resources_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('resources_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('is_sviewinfo', ctypes.c_bool),
    ('samples_only', ctypes.c_bool),
    ('ms', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte),
    ('lod_property', lp_sampler_lod_property),
    ('explicit_lod', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('sizes_out', ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('resource', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('exec_mask', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('exec_mask_nz', ctypes.c_bool),
    ('PADDING_3', ctypes.c_ubyte * 3),
    ('format', pipe_format),
]

struct_tgsi_shader_info._pack_ = 1 # source:False
struct_tgsi_shader_info._fields_ = [
    ('num_inputs', ctypes.c_ubyte),
    ('num_outputs', ctypes.c_ubyte),
    ('input_semantic_name', ctypes.c_ubyte * 80),
    ('input_semantic_index', ctypes.c_ubyte * 80),
    ('input_interpolate', ctypes.c_ubyte * 80),
    ('input_interpolate_loc', ctypes.c_ubyte * 80),
    ('input_usage_mask', ctypes.c_ubyte * 80),
    ('output_semantic_name', ctypes.c_ubyte * 80),
    ('output_semantic_index', ctypes.c_ubyte * 80),
    ('output_usagemask', ctypes.c_ubyte * 80),
    ('output_streams', ctypes.c_ubyte * 80),
    ('num_system_values', ctypes.c_ubyte),
    ('system_value_semantic_name', ctypes.c_ubyte * 80),
    ('processor', ctypes.c_ubyte),
    ('file_mask', ctypes.c_uint32 * 15),
    ('file_count', ctypes.c_uint32 * 15),
    ('file_max', ctypes.c_int32 * 15),
    ('const_file_max', ctypes.c_int32 * 32),
    ('const_buffers_declared', ctypes.c_uint32),
    ('samplers_declared', ctypes.c_uint32),
    ('sampler_targets', ctypes.c_ubyte * 128),
    ('sampler_type', ctypes.c_ubyte * 128),
    ('num_stream_output_components', ctypes.c_ubyte * 4),
    ('input_array_first', ctypes.c_ubyte * 80),
    ('output_array_first', ctypes.c_ubyte * 80),
    ('immediate_count', ctypes.c_uint32),
    ('num_instructions', ctypes.c_uint32),
    ('opcode_count', ctypes.c_uint32 * 252),
    ('reads_pervertex_outputs', ctypes.c_bool),
    ('reads_perpatch_outputs', ctypes.c_bool),
    ('reads_tessfactor_outputs', ctypes.c_bool),
    ('reads_z', ctypes.c_bool),
    ('writes_z', ctypes.c_bool),
    ('writes_stencil', ctypes.c_bool),
    ('writes_samplemask', ctypes.c_bool),
    ('writes_edgeflag', ctypes.c_bool),
    ('uses_kill', ctypes.c_bool),
    ('uses_instanceid', ctypes.c_bool),
    ('uses_vertexid', ctypes.c_bool),
    ('uses_vertexid_nobase', ctypes.c_bool),
    ('uses_basevertex', ctypes.c_bool),
    ('uses_primid', ctypes.c_bool),
    ('uses_frontface', ctypes.c_bool),
    ('uses_invocationid', ctypes.c_bool),
    ('uses_grid_size', ctypes.c_bool),
    ('writes_position', ctypes.c_bool),
    ('writes_psize', ctypes.c_bool),
    ('writes_clipvertex', ctypes.c_bool),
    ('writes_viewport_index', ctypes.c_bool),
    ('writes_layer', ctypes.c_bool),
    ('writes_memory', ctypes.c_bool),
    ('uses_fbfetch', ctypes.c_bool),
    ('num_written_culldistance', ctypes.c_uint32),
    ('num_written_clipdistance', ctypes.c_uint32),
    ('images_declared', ctypes.c_uint32),
    ('msaa_images_declared', ctypes.c_uint32),
    ('images_buffers', ctypes.c_uint32),
    ('shader_buffers_declared', ctypes.c_uint32),
    ('shader_buffers_load', ctypes.c_uint32),
    ('shader_buffers_store', ctypes.c_uint32),
    ('shader_buffers_atomic', ctypes.c_uint32),
    ('hw_atomic_declared', ctypes.c_uint32),
    ('indirect_files', ctypes.c_uint32),
    ('dim_indirect_files', ctypes.c_uint32),
    ('properties', ctypes.c_uint32 * 29),
]

struct_lp_build_gs_iface._pack_ = 1 # source:False
struct_lp_build_gs_iface._fields_ = [
    ('fetch_input', ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('emit_vertex', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue) * 4), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('end_primitive', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_uint32)),
    ('gs_epilogue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_uint32)),
]

struct_lp_build_tcs_iface._pack_ = 1 # source:False
struct_lp_build_tcs_iface._fields_ = [
    ('emit_prologue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
    ('emit_epilogue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
    ('emit_barrier', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
    ('emit_store_output', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('emit_fetch_input', ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('emit_fetch_output', ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_uint32)),
]

struct_lp_build_tes_iface._pack_ = 1 # source:False
struct_lp_build_tes_iface._fields_ = [
    ('fetch_vertex_input', ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_lp_build_tes_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('fetch_patch_input', ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_lp_build_tes_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue))),
]

struct_lp_build_mesh_iface._pack_ = 1 # source:False
struct_lp_build_mesh_iface._fields_ = [
    ('emit_store_output', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_mesh_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('emit_vertex_and_primitive_count', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_mesh_iface), ctypes.POINTER(struct_lp_build_context), ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_LLVMOpaqueValue))),
]

class struct_lp_img_params(Structure):
    pass

struct_lp_build_image_soa._pack_ = 1 # source:False
struct_lp_build_image_soa._fields_ = [
    ('emit_op', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_image_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_img_params))),
    ('emit_size_query', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_image_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params))),
]


# values for enumeration 'c__EA_LLVMAtomicRMWBinOp'
c__EA_LLVMAtomicRMWBinOp__enumvalues = {
    0: 'LLVMAtomicRMWBinOpXchg',
    1: 'LLVMAtomicRMWBinOpAdd',
    2: 'LLVMAtomicRMWBinOpSub',
    3: 'LLVMAtomicRMWBinOpAnd',
    4: 'LLVMAtomicRMWBinOpNand',
    5: 'LLVMAtomicRMWBinOpOr',
    6: 'LLVMAtomicRMWBinOpXor',
    7: 'LLVMAtomicRMWBinOpMax',
    8: 'LLVMAtomicRMWBinOpMin',
    9: 'LLVMAtomicRMWBinOpUMax',
    10: 'LLVMAtomicRMWBinOpUMin',
    11: 'LLVMAtomicRMWBinOpFAdd',
    12: 'LLVMAtomicRMWBinOpFSub',
    13: 'LLVMAtomicRMWBinOpFMax',
    14: 'LLVMAtomicRMWBinOpFMin',
    15: 'LLVMAtomicRMWBinOpUIncWrap',
    16: 'LLVMAtomicRMWBinOpUDecWrap',
    17: 'LLVMAtomicRMWBinOpUSubCond',
    18: 'LLVMAtomicRMWBinOpUSubSat',
}
LLVMAtomicRMWBinOpXchg = 0
LLVMAtomicRMWBinOpAdd = 1
LLVMAtomicRMWBinOpSub = 2
LLVMAtomicRMWBinOpAnd = 3
LLVMAtomicRMWBinOpNand = 4
LLVMAtomicRMWBinOpOr = 5
LLVMAtomicRMWBinOpXor = 6
LLVMAtomicRMWBinOpMax = 7
LLVMAtomicRMWBinOpMin = 8
LLVMAtomicRMWBinOpUMax = 9
LLVMAtomicRMWBinOpUMin = 10
LLVMAtomicRMWBinOpFAdd = 11
LLVMAtomicRMWBinOpFSub = 12
LLVMAtomicRMWBinOpFMax = 13
LLVMAtomicRMWBinOpFMin = 14
LLVMAtomicRMWBinOpUIncWrap = 15
LLVMAtomicRMWBinOpUDecWrap = 16
LLVMAtomicRMWBinOpUSubCond = 17
LLVMAtomicRMWBinOpUSubSat = 18
c__EA_LLVMAtomicRMWBinOp = ctypes.c_uint32 # enum
struct_lp_img_params._pack_ = 1 # source:False
struct_lp_img_params._fields_ = [
    ('type', struct_lp_type),
    ('image_index', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('image_index_offset', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('img_op', ctypes.c_uint32),
    ('target', ctypes.c_uint32),
    ('packed_op', ctypes.c_uint32),
    ('op', c__EA_LLVMAtomicRMWBinOp),
    ('exec_mask', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('exec_mask_nz', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 7),
    ('resources_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('resources_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('thread_data_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('thread_data_ptr', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('coords', ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('ms_index', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('indata', ctypes.POINTER(struct_LLVMOpaqueValue) * 4),
    ('indata2', ctypes.POINTER(struct_LLVMOpaqueValue) * 4),
    ('outdata', ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue))),
    ('resource', ctypes.POINTER(struct_LLVMOpaqueValue)),
    ('format', pipe_format),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

struct_lp_build_fs_iface._pack_ = 1 # source:False
struct_lp_build_fs_iface._fields_ = [
    ('interp_fn', ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_lp_build_fs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)))),
    ('fb_fetch', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_fs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)))),
]

try:
    lp_build_nir_soa = _libraries['libtinymesa_cpu.so'].lp_build_nir_soa
    lp_build_nir_soa.restype = None
    lp_build_nir_soa.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_lp_build_tgsi_params), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue) * 4)]
except AttributeError:
    pass
try:
    lp_build_nir_soa_func = _libraries['libtinymesa_cpu.so'].lp_build_nir_soa_func
    lp_build_nir_soa_func.restype = None
    lp_build_nir_soa_func.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct_lp_build_tgsi_params), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue) * 4)]
except AttributeError:
    pass
class struct_lp_build_sampler_aos(Structure):
    pass


# values for enumeration 'tgsi_texture_type'
tgsi_texture_type__enumvalues = {
    0: 'TGSI_TEXTURE_BUFFER',
    1: 'TGSI_TEXTURE_1D',
    2: 'TGSI_TEXTURE_2D',
    3: 'TGSI_TEXTURE_3D',
    4: 'TGSI_TEXTURE_CUBE',
    5: 'TGSI_TEXTURE_RECT',
    6: 'TGSI_TEXTURE_SHADOW1D',
    7: 'TGSI_TEXTURE_SHADOW2D',
    8: 'TGSI_TEXTURE_SHADOWRECT',
    9: 'TGSI_TEXTURE_1D_ARRAY',
    10: 'TGSI_TEXTURE_2D_ARRAY',
    11: 'TGSI_TEXTURE_SHADOW1D_ARRAY',
    12: 'TGSI_TEXTURE_SHADOW2D_ARRAY',
    13: 'TGSI_TEXTURE_SHADOWCUBE',
    14: 'TGSI_TEXTURE_2D_MSAA',
    15: 'TGSI_TEXTURE_2D_ARRAY_MSAA',
    16: 'TGSI_TEXTURE_CUBE_ARRAY',
    17: 'TGSI_TEXTURE_SHADOWCUBE_ARRAY',
    18: 'TGSI_TEXTURE_UNKNOWN',
    19: 'TGSI_TEXTURE_COUNT',
}
TGSI_TEXTURE_BUFFER = 0
TGSI_TEXTURE_1D = 1
TGSI_TEXTURE_2D = 2
TGSI_TEXTURE_3D = 3
TGSI_TEXTURE_CUBE = 4
TGSI_TEXTURE_RECT = 5
TGSI_TEXTURE_SHADOW1D = 6
TGSI_TEXTURE_SHADOW2D = 7
TGSI_TEXTURE_SHADOWRECT = 8
TGSI_TEXTURE_1D_ARRAY = 9
TGSI_TEXTURE_2D_ARRAY = 10
TGSI_TEXTURE_SHADOW1D_ARRAY = 11
TGSI_TEXTURE_SHADOW2D_ARRAY = 12
TGSI_TEXTURE_SHADOWCUBE = 13
TGSI_TEXTURE_2D_MSAA = 14
TGSI_TEXTURE_2D_ARRAY_MSAA = 15
TGSI_TEXTURE_CUBE_ARRAY = 16
TGSI_TEXTURE_SHADOWCUBE_ARRAY = 17
TGSI_TEXTURE_UNKNOWN = 18
TGSI_TEXTURE_COUNT = 19
tgsi_texture_type = ctypes.c_uint32 # enum

# values for enumeration 'lp_build_tex_modifier'
lp_build_tex_modifier__enumvalues = {
    0: 'LP_BLD_TEX_MODIFIER_NONE',
    1: 'LP_BLD_TEX_MODIFIER_PROJECTED',
    2: 'LP_BLD_TEX_MODIFIER_LOD_BIAS',
    3: 'LP_BLD_TEX_MODIFIER_EXPLICIT_LOD',
    4: 'LP_BLD_TEX_MODIFIER_EXPLICIT_DERIV',
    5: 'LP_BLD_TEX_MODIFIER_LOD_ZERO',
}
LP_BLD_TEX_MODIFIER_NONE = 0
LP_BLD_TEX_MODIFIER_PROJECTED = 1
LP_BLD_TEX_MODIFIER_LOD_BIAS = 2
LP_BLD_TEX_MODIFIER_EXPLICIT_LOD = 3
LP_BLD_TEX_MODIFIER_EXPLICIT_DERIV = 4
LP_BLD_TEX_MODIFIER_LOD_ZERO = 5
lp_build_tex_modifier = ctypes.c_uint32 # enum
struct_lp_build_sampler_aos._pack_ = 1 # source:False
struct_lp_build_sampler_aos._fields_ = [
    ('emit_fetch_texel', ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueValue), ctypes.POINTER(struct_lp_build_sampler_aos), ctypes.POINTER(struct_lp_build_context), tgsi_texture_type, ctypes.c_uint32, ctypes.POINTER(struct_LLVMOpaqueValue), struct_lp_derivatives, lp_build_tex_modifier)),
]

try:
    lp_build_nir_aos = _libraries['FIXME_STUB'].lp_build_nir_aos
    lp_build_nir_aos.restype = None
    lp_build_nir_aos.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), struct_lp_type, ctypes.c_ubyte * 4, LLVMValueRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.POINTER(struct_lp_build_sampler_aos)]
except AttributeError:
    pass
class struct_lp_build_fn(Structure):
    pass

struct_lp_build_fn._pack_ = 1 # source:False
struct_lp_build_fn._fields_ = [
    ('fn_type', ctypes.POINTER(struct_LLVMOpaqueType)),
    ('fn', ctypes.POINTER(struct_LLVMOpaqueValue)),
]

try:
    lp_build_nir_soa_prepasses = _libraries['libtinymesa_cpu.so'].lp_build_nir_soa_prepasses
    lp_build_nir_soa_prepasses.restype = None
    lp_build_nir_soa_prepasses.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    lp_build_opt_nir = _libraries['libtinymesa_cpu.so'].lp_build_opt_nir
    lp_build_opt_nir.restype = None
    lp_build_opt_nir.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    lp_nir_array_build_gather_values = _libraries['FIXME_STUB'].lp_nir_array_build_gather_values
    lp_nir_array_build_gather_values.restype = LLVMValueRef
    lp_nir_array_build_gather_values.argtypes = [LLVMBuilderRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueValue)), ctypes.c_uint32]
except AttributeError:
    pass
LLVMAtomicRMWBinOp = c__EA_LLVMAtomicRMWBinOp
LLVMAtomicRMWBinOp__enumvalues = c__EA_LLVMAtomicRMWBinOp__enumvalues
try:
    lp_translate_atomic_op = _libraries['libtinymesa_cpu.so'].lp_translate_atomic_op
    lp_translate_atomic_op.restype = LLVMAtomicRMWBinOp
    lp_translate_atomic_op.argtypes = [nir_atomic_op]
except AttributeError:
    pass
try:
    lp_build_nir_sample_key = _libraries['libtinymesa_cpu.so'].lp_build_nir_sample_key
    lp_build_nir_sample_key.restype = uint32_t
    lp_build_nir_sample_key.argtypes = [gl_shader_stage, ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    lp_img_op_from_intrinsic = _libraries['libtinymesa_cpu.so'].lp_img_op_from_intrinsic
    lp_img_op_from_intrinsic.restype = None
    lp_img_op_from_intrinsic.argtypes = [ctypes.POINTER(struct_lp_img_params), ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    lp_packed_img_op_from_intrinsic = _libraries['libtinymesa_cpu.so'].lp_packed_img_op_from_intrinsic
    lp_packed_img_op_from_intrinsic.restype = uint32_t
    lp_packed_img_op_from_intrinsic.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass

# values for enumeration 'lp_nir_call_context_args'
lp_nir_call_context_args__enumvalues = {
    0: 'LP_NIR_CALL_CONTEXT_CONTEXT',
    1: 'LP_NIR_CALL_CONTEXT_RESOURCES',
    2: 'LP_NIR_CALL_CONTEXT_SHARED',
    3: 'LP_NIR_CALL_CONTEXT_SCRATCH',
    4: 'LP_NIR_CALL_CONTEXT_WORK_DIM',
    5: 'LP_NIR_CALL_CONTEXT_THREAD_ID_0',
    6: 'LP_NIR_CALL_CONTEXT_THREAD_ID_1',
    7: 'LP_NIR_CALL_CONTEXT_THREAD_ID_2',
    8: 'LP_NIR_CALL_CONTEXT_BLOCK_ID_0',
    9: 'LP_NIR_CALL_CONTEXT_BLOCK_ID_1',
    10: 'LP_NIR_CALL_CONTEXT_BLOCK_ID_2',
    11: 'LP_NIR_CALL_CONTEXT_GRID_SIZE_0',
    12: 'LP_NIR_CALL_CONTEXT_GRID_SIZE_1',
    13: 'LP_NIR_CALL_CONTEXT_GRID_SIZE_2',
    14: 'LP_NIR_CALL_CONTEXT_BLOCK_SIZE_0',
    15: 'LP_NIR_CALL_CONTEXT_BLOCK_SIZE_1',
    16: 'LP_NIR_CALL_CONTEXT_BLOCK_SIZE_2',
    17: 'LP_NIR_CALL_CONTEXT_MAX_ARGS',
}
LP_NIR_CALL_CONTEXT_CONTEXT = 0
LP_NIR_CALL_CONTEXT_RESOURCES = 1
LP_NIR_CALL_CONTEXT_SHARED = 2
LP_NIR_CALL_CONTEXT_SCRATCH = 3
LP_NIR_CALL_CONTEXT_WORK_DIM = 4
LP_NIR_CALL_CONTEXT_THREAD_ID_0 = 5
LP_NIR_CALL_CONTEXT_THREAD_ID_1 = 6
LP_NIR_CALL_CONTEXT_THREAD_ID_2 = 7
LP_NIR_CALL_CONTEXT_BLOCK_ID_0 = 8
LP_NIR_CALL_CONTEXT_BLOCK_ID_1 = 9
LP_NIR_CALL_CONTEXT_BLOCK_ID_2 = 10
LP_NIR_CALL_CONTEXT_GRID_SIZE_0 = 11
LP_NIR_CALL_CONTEXT_GRID_SIZE_1 = 12
LP_NIR_CALL_CONTEXT_GRID_SIZE_2 = 13
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_0 = 14
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_1 = 15
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_2 = 16
LP_NIR_CALL_CONTEXT_MAX_ARGS = 17
lp_nir_call_context_args = ctypes.c_uint32 # enum
try:
    lp_build_cs_func_call_context = _libraries['libtinymesa_cpu.so'].lp_build_cs_func_call_context
    lp_build_cs_func_call_context.restype = LLVMTypeRef
    lp_build_cs_func_call_context.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.c_int32, LLVMTypeRef, LLVMTypeRef]
except AttributeError:
    pass
try:
    lp_build_struct_get_ptr2 = _libraries['libtinymesa_cpu.so'].lp_build_struct_get_ptr2
    lp_build_struct_get_ptr2.restype = LLVMValueRef
    lp_build_struct_get_ptr2.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lp_build_struct_get2 = _libraries['libtinymesa_cpu.so'].lp_build_struct_get2
    lp_build_struct_get2.restype = LLVMValueRef
    lp_build_struct_get2.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lp_build_array_get_ptr2 = _libraries['libtinymesa_cpu.so'].lp_build_array_get_ptr2
    lp_build_array_get_ptr2.restype = LLVMValueRef
    lp_build_array_get_ptr2.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_array_get2 = _libraries['libtinymesa_cpu.so'].lp_build_array_get2
    lp_build_array_get2.restype = LLVMValueRef
    lp_build_array_get2.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_pointer_get2 = _libraries['libtinymesa_cpu.so'].lp_build_pointer_get2
    lp_build_pointer_get2.restype = LLVMValueRef
    lp_build_pointer_get2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_pointer_get_unaligned2 = _libraries['libtinymesa_cpu.so'].lp_build_pointer_get_unaligned2
    lp_build_pointer_get_unaligned2.restype = LLVMValueRef
    lp_build_pointer_get_unaligned2.argtypes = [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_build_pointer_set = _libraries['libtinymesa_cpu.so'].lp_build_pointer_set
    lp_build_pointer_set.restype = None
    lp_build_pointer_set.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError:
    pass
try:
    lp_build_pointer_set_unaligned = _libraries['libtinymesa_cpu.so'].lp_build_pointer_set_unaligned
    lp_build_pointer_set_unaligned.restype = None
    lp_build_pointer_set_unaligned.argtypes = [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
class struct_lp_jit_buffer(Structure):
    pass

class union_lp_jit_buffer_0(Union):
    pass

union_lp_jit_buffer_0._pack_ = 1 # source:False
union_lp_jit_buffer_0._fields_ = [
    ('u', ctypes.POINTER(ctypes.c_uint32)),
    ('f', ctypes.POINTER(ctypes.c_float)),
]

struct_lp_jit_buffer._pack_ = 1 # source:False
struct_lp_jit_buffer._anonymous_ = ('_0',)
struct_lp_jit_buffer._fields_ = [
    ('_0', union_lp_jit_buffer_0),
    ('num_elements', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'c__Ea_LP_JIT_BUFFER_BASE'
c__Ea_LP_JIT_BUFFER_BASE__enumvalues = {
    0: 'LP_JIT_BUFFER_BASE',
    1: 'LP_JIT_BUFFER_NUM_ELEMENTS',
    2: 'LP_JIT_BUFFER_NUM_FIELDS',
}
LP_JIT_BUFFER_BASE = 0
LP_JIT_BUFFER_NUM_ELEMENTS = 1
LP_JIT_BUFFER_NUM_FIELDS = 2
c__Ea_LP_JIT_BUFFER_BASE = ctypes.c_uint32 # enum
try:
    lp_llvm_descriptor_base = _libraries['libtinymesa_cpu.so'].lp_llvm_descriptor_base
    lp_llvm_descriptor_base.restype = LLVMValueRef
    lp_llvm_descriptor_base.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_llvm_buffer_base = _libraries['libtinymesa_cpu.so'].lp_llvm_buffer_base
    lp_llvm_buffer_base.restype = LLVMValueRef
    lp_llvm_buffer_base.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_llvm_buffer_num_elements = _libraries['libtinymesa_cpu.so'].lp_llvm_buffer_num_elements
    lp_llvm_buffer_num_elements.restype = LLVMValueRef
    lp_llvm_buffer_num_elements.argtypes = [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__Ea_LP_JIT_TEXTURE_BASE'
c__Ea_LP_JIT_TEXTURE_BASE__enumvalues = {
    0: 'LP_JIT_TEXTURE_BASE',
    1: 'LP_JIT_TEXTURE_WIDTH',
    2: 'LP_JIT_TEXTURE_HEIGHT',
    3: 'LP_JIT_TEXTURE_DEPTH',
    4: 'LP_JIT_TEXTURE_ROW_STRIDE',
    5: 'LP_JIT_TEXTURE_IMG_STRIDE',
    6: 'LP_JIT_TEXTURE_FIRST_LEVEL',
    7: 'LP_JIT_TEXTURE_LAST_LEVEL',
    8: 'LP_JIT_TEXTURE_MIP_OFFSETS',
    9: 'LP_JIT_SAMPLER_INDEX_DUMMY',
    10: 'LP_JIT_TEXTURE_NUM_FIELDS',
}
LP_JIT_TEXTURE_BASE = 0
LP_JIT_TEXTURE_WIDTH = 1
LP_JIT_TEXTURE_HEIGHT = 2
LP_JIT_TEXTURE_DEPTH = 3
LP_JIT_TEXTURE_ROW_STRIDE = 4
LP_JIT_TEXTURE_IMG_STRIDE = 5
LP_JIT_TEXTURE_FIRST_LEVEL = 6
LP_JIT_TEXTURE_LAST_LEVEL = 7
LP_JIT_TEXTURE_MIP_OFFSETS = 8
LP_JIT_SAMPLER_INDEX_DUMMY = 9
LP_JIT_TEXTURE_NUM_FIELDS = 10
c__Ea_LP_JIT_TEXTURE_BASE = ctypes.c_uint32 # enum
class struct_lp_jit_sampler(Structure):
    pass

struct_lp_jit_sampler._pack_ = 1 # source:False
struct_lp_jit_sampler._fields_ = [
    ('min_lod', ctypes.c_float),
    ('max_lod', ctypes.c_float),
    ('lod_bias', ctypes.c_float),
    ('border_color', ctypes.c_float * 4),
]


# values for enumeration 'c__Ea_LP_JIT_SAMPLER_MIN_LOD'
c__Ea_LP_JIT_SAMPLER_MIN_LOD__enumvalues = {
    0: 'LP_JIT_SAMPLER_MIN_LOD',
    1: 'LP_JIT_SAMPLER_MAX_LOD',
    2: 'LP_JIT_SAMPLER_LOD_BIAS',
    3: 'LP_JIT_SAMPLER_BORDER_COLOR',
    4: 'LP_JIT_SAMPLER_NUM_FIELDS',
}
LP_JIT_SAMPLER_MIN_LOD = 0
LP_JIT_SAMPLER_MAX_LOD = 1
LP_JIT_SAMPLER_LOD_BIAS = 2
LP_JIT_SAMPLER_BORDER_COLOR = 3
LP_JIT_SAMPLER_NUM_FIELDS = 4
c__Ea_LP_JIT_SAMPLER_MIN_LOD = ctypes.c_uint32 # enum
class struct_lp_jit_image(Structure):
    pass

struct_lp_jit_image._pack_ = 1 # source:False
struct_lp_jit_image._fields_ = [
    ('base', ctypes.POINTER(None)),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint16),
    ('depth', ctypes.c_uint16),
    ('num_samples', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('sample_stride', ctypes.c_uint32),
    ('row_stride', ctypes.c_uint32),
    ('img_stride', ctypes.c_uint32),
    ('residency', ctypes.POINTER(None)),
    ('base_offset', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]


# values for enumeration 'c__Ea_LP_JIT_IMAGE_BASE'
c__Ea_LP_JIT_IMAGE_BASE__enumvalues = {
    0: 'LP_JIT_IMAGE_BASE',
    1: 'LP_JIT_IMAGE_WIDTH',
    2: 'LP_JIT_IMAGE_HEIGHT',
    3: 'LP_JIT_IMAGE_DEPTH',
    4: 'LP_JIT_IMAGE_NUM_SAMPLES',
    5: 'LP_JIT_IMAGE_SAMPLE_STRIDE',
    6: 'LP_JIT_IMAGE_ROW_STRIDE',
    7: 'LP_JIT_IMAGE_IMG_STRIDE',
    8: 'LP_JIT_IMAGE_RESIDENCY',
    9: 'LP_JIT_IMAGE_BASE_OFFSET',
    10: 'LP_JIT_IMAGE_NUM_FIELDS',
}
LP_JIT_IMAGE_BASE = 0
LP_JIT_IMAGE_WIDTH = 1
LP_JIT_IMAGE_HEIGHT = 2
LP_JIT_IMAGE_DEPTH = 3
LP_JIT_IMAGE_NUM_SAMPLES = 4
LP_JIT_IMAGE_SAMPLE_STRIDE = 5
LP_JIT_IMAGE_ROW_STRIDE = 6
LP_JIT_IMAGE_IMG_STRIDE = 7
LP_JIT_IMAGE_RESIDENCY = 8
LP_JIT_IMAGE_BASE_OFFSET = 9
LP_JIT_IMAGE_NUM_FIELDS = 10
c__Ea_LP_JIT_IMAGE_BASE = ctypes.c_uint32 # enum
class struct_lp_jit_resources(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('constants', struct_lp_jit_buffer * 16),
    ('ssbos', struct_lp_jit_buffer * 32),
    ('textures', struct_lp_jit_texture * 128),
    ('samplers', struct_lp_jit_sampler * 32),
    ('images', struct_lp_jit_image * 64),
     ]


# values for enumeration 'c__Ea_LP_JIT_RES_CONSTANTS'
c__Ea_LP_JIT_RES_CONSTANTS__enumvalues = {
    0: 'LP_JIT_RES_CONSTANTS',
    1: 'LP_JIT_RES_SSBOS',
    2: 'LP_JIT_RES_TEXTURES',
    3: 'LP_JIT_RES_SAMPLERS',
    4: 'LP_JIT_RES_IMAGES',
    5: 'LP_JIT_RES_COUNT',
}
LP_JIT_RES_CONSTANTS = 0
LP_JIT_RES_SSBOS = 1
LP_JIT_RES_TEXTURES = 2
LP_JIT_RES_SAMPLERS = 3
LP_JIT_RES_IMAGES = 4
LP_JIT_RES_COUNT = 5
c__Ea_LP_JIT_RES_CONSTANTS = ctypes.c_uint32 # enum
try:
    lp_build_jit_resources_type = _libraries['libtinymesa_cpu.so'].lp_build_jit_resources_type
    lp_build_jit_resources_type.restype = LLVMTypeRef
    lp_build_jit_resources_type.argtypes = [ctypes.POINTER(struct_gallivm_state)]
except AttributeError:
    pass

# values for enumeration 'c__Ea_LP_JIT_VERTEX_HEADER_VERTEX_ID'
c__Ea_LP_JIT_VERTEX_HEADER_VERTEX_ID__enumvalues = {
    0: 'LP_JIT_VERTEX_HEADER_VERTEX_ID',
    1: 'LP_JIT_VERTEX_HEADER_CLIP_POS',
    2: 'LP_JIT_VERTEX_HEADER_DATA',
}
LP_JIT_VERTEX_HEADER_VERTEX_ID = 0
LP_JIT_VERTEX_HEADER_CLIP_POS = 1
LP_JIT_VERTEX_HEADER_DATA = 2
c__Ea_LP_JIT_VERTEX_HEADER_VERTEX_ID = ctypes.c_uint32 # enum
try:
    lp_build_create_jit_vertex_header_type = _libraries['libtinymesa_cpu.so'].lp_build_create_jit_vertex_header_type
    lp_build_create_jit_vertex_header_type.restype = LLVMTypeRef
    lp_build_create_jit_vertex_header_type.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.c_int32]
except AttributeError:
    pass
class struct_lp_sampler_dynamic_state(Structure):
    pass

try:
    lp_build_jit_fill_sampler_dynamic_state = _libraries['libtinymesa_cpu.so'].lp_build_jit_fill_sampler_dynamic_state
    lp_build_jit_fill_sampler_dynamic_state.restype = None
    lp_build_jit_fill_sampler_dynamic_state.argtypes = [ctypes.POINTER(struct_lp_sampler_dynamic_state)]
except AttributeError:
    pass
try:
    lp_build_jit_fill_image_dynamic_state = _libraries['libtinymesa_cpu.so'].lp_build_jit_fill_image_dynamic_state
    lp_build_jit_fill_image_dynamic_state.restype = None
    lp_build_jit_fill_image_dynamic_state.argtypes = [ctypes.POINTER(struct_lp_sampler_dynamic_state)]
except AttributeError:
    pass
try:
    lp_build_sample_function_type = _libraries['libtinymesa_cpu.so'].lp_build_sample_function_type
    lp_build_sample_function_type.restype = LLVMTypeRef
    lp_build_sample_function_type.argtypes = [ctypes.POINTER(struct_gallivm_state), uint32_t]
except AttributeError:
    pass
try:
    lp_build_size_function_type = _libraries['libtinymesa_cpu.so'].lp_build_size_function_type
    lp_build_size_function_type.restype = LLVMTypeRef
    lp_build_size_function_type.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params)]
except AttributeError:
    pass
try:
    lp_build_image_function_type = _libraries['libtinymesa_cpu.so'].lp_build_image_function_type
    lp_build_image_function_type.restype = LLVMTypeRef
    lp_build_image_function_type.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_img_params), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
class struct_lp_texture_handle_state(Structure):
    pass

class struct_lp_static_texture_state(Structure):
    pass


# values for enumeration 'c_uint32'
c_uint32__enumvalues = {
    0: 'PIPE_BUFFER',
    1: 'PIPE_TEXTURE_1D',
    2: 'PIPE_TEXTURE_2D',
    3: 'PIPE_TEXTURE_3D',
    4: 'PIPE_TEXTURE_CUBE',
    5: 'PIPE_TEXTURE_RECT',
    6: 'PIPE_TEXTURE_1D_ARRAY',
    7: 'PIPE_TEXTURE_2D_ARRAY',
    8: 'PIPE_TEXTURE_CUBE_ARRAY',
    9: 'PIPE_MAX_TEXTURE_TYPES',
}
PIPE_BUFFER = 0
PIPE_TEXTURE_1D = 1
PIPE_TEXTURE_2D = 2
PIPE_TEXTURE_3D = 3
PIPE_TEXTURE_CUBE = 4
PIPE_TEXTURE_RECT = 5
PIPE_TEXTURE_1D_ARRAY = 6
PIPE_TEXTURE_2D_ARRAY = 7
PIPE_TEXTURE_CUBE_ARRAY = 8
PIPE_MAX_TEXTURE_TYPES = 9
c_uint32 = ctypes.c_uint32 # enum
struct_lp_static_texture_state._pack_ = 1 # source:False
struct_lp_static_texture_state._fields_ = [
    ('format', pipe_format),
    ('res_format', pipe_format),
    ('swizzle_r', ctypes.c_uint32, 3),
    ('swizzle_g', ctypes.c_uint32, 3),
    ('swizzle_b', ctypes.c_uint32, 3),
    ('swizzle_a', ctypes.c_uint32, 3),
    ('target', c_uint32, 5),
    ('res_target', c_uint32, 5),
    ('pot_width', ctypes.c_uint32, 1),
    ('pot_height', ctypes.c_uint32, 1),
    ('pot_depth', ctypes.c_uint32, 1),
    ('level_zero_only', ctypes.c_uint32, 1),
    ('tiled', ctypes.c_uint32, 1),
    ('tiled_samples', ctypes.c_uint32, 5),
]

struct_lp_texture_handle_state._pack_ = 1 # source:False
struct_lp_texture_handle_state._fields_ = [
    ('static_state', struct_lp_static_texture_state),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('dynamic_state', struct_lp_jit_texture),
]

class struct_lp_texture_functions(Structure):
    pass

struct_lp_texture_functions._pack_ = 1 # source:False
struct_lp_texture_functions._fields_ = [
    ('sample_functions', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
    ('sampler_count', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('fetch_functions', ctypes.POINTER(ctypes.POINTER(None))),
    ('size_function', ctypes.POINTER(None)),
    ('samples_function', ctypes.POINTER(None)),
    ('image_functions', ctypes.POINTER(ctypes.POINTER(None))),
    ('state', struct_lp_texture_handle_state),
    ('sampled', ctypes.c_bool),
    ('storage', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 6),
    ('matrix', ctypes.POINTER(None)),
]

class struct_lp_texture_handle(Structure):
    pass

struct_lp_texture_handle._pack_ = 1 # source:False
struct_lp_texture_handle._fields_ = [
    ('functions', ctypes.POINTER(None)),
    ('sampler_index', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_lp_jit_bindless_texture(Structure):
    pass

struct_lp_jit_bindless_texture._pack_ = 1 # source:False
struct_lp_jit_bindless_texture._fields_ = [
    ('base', ctypes.POINTER(None)),
    ('residency', ctypes.POINTER(None)),
    ('sampler_index', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_lp_descriptor(Structure):
    pass

class union_lp_descriptor_0(Union):
    pass

class struct_lp_descriptor_0_0(Structure):
    pass

struct_lp_descriptor_0_0._pack_ = 1 # source:False
struct_lp_descriptor_0_0._fields_ = [
    ('texture', struct_lp_jit_bindless_texture),
    ('sampler', struct_lp_jit_sampler),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_lp_descriptor_0_1(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('image', struct_lp_jit_image),
     ]

union_lp_descriptor_0._pack_ = 1 # source:False
union_lp_descriptor_0._anonymous_ = ('_0', '_1',)
union_lp_descriptor_0._fields_ = [
    ('_0', struct_lp_descriptor_0_0),
    ('_1', struct_lp_descriptor_0_1),
    ('buffer', struct_lp_jit_buffer),
    ('accel_struct', ctypes.c_uint64),
    ('PADDING_0', ctypes.c_ubyte * 48),
]

struct_lp_descriptor._pack_ = 1 # source:False
struct_lp_descriptor._anonymous_ = ('_0',)
struct_lp_descriptor._fields_ = [
    ('_0', union_lp_descriptor_0),
    ('functions', ctypes.POINTER(None)),
]

try:
    lp_mantissa = _libraries['libtinymesa_cpu.so'].lp_mantissa
    lp_mantissa.restype = ctypes.c_uint32
    lp_mantissa.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_const_shift = _libraries['libtinymesa_cpu.so'].lp_const_shift
    lp_const_shift.restype = ctypes.c_uint32
    lp_const_shift.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_const_offset = _libraries['libtinymesa_cpu.so'].lp_const_offset
    lp_const_offset.restype = ctypes.c_uint32
    lp_const_offset.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_const_scale = _libraries['libtinymesa_cpu.so'].lp_const_scale
    lp_const_scale.restype = ctypes.c_double
    lp_const_scale.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_const_min = _libraries['libtinymesa_cpu.so'].lp_const_min
    lp_const_min.restype = ctypes.c_double
    lp_const_min.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_const_max = _libraries['libtinymesa_cpu.so'].lp_const_max
    lp_const_max.restype = ctypes.c_double
    lp_const_max.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_const_eps = _libraries['libtinymesa_cpu.so'].lp_const_eps
    lp_const_eps.restype = ctypes.c_double
    lp_const_eps.argtypes = [struct_lp_type]
except AttributeError:
    pass
try:
    lp_build_undef = _libraries['libtinymesa_cpu.so'].lp_build_undef
    lp_build_undef.restype = LLVMValueRef
    lp_build_undef.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_build_zero = _libraries['libtinymesa_cpu.so'].lp_build_zero
    lp_build_zero.restype = LLVMValueRef
    lp_build_zero.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_build_one = _libraries['libtinymesa_cpu.so'].lp_build_one
    lp_build_one.restype = LLVMValueRef
    lp_build_one.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_build_const_elem = _libraries['libtinymesa_cpu.so'].lp_build_const_elem
    lp_build_const_elem.restype = LLVMValueRef
    lp_build_const_elem.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double]
except AttributeError:
    pass
try:
    lp_build_const_vec = _libraries['libtinymesa_cpu.so'].lp_build_const_vec
    lp_build_const_vec.restype = LLVMValueRef
    lp_build_const_vec.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double]
except AttributeError:
    pass
try:
    lp_build_const_int_vec = _libraries['libtinymesa_cpu.so'].lp_build_const_int_vec
    lp_build_const_int_vec.restype = LLVMValueRef
    lp_build_const_int_vec.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_int64]
except AttributeError:
    pass
try:
    lp_build_const_channel_vec = _libraries['libtinymesa_cpu.so'].lp_build_const_channel_vec
    lp_build_const_channel_vec.restype = LLVMValueRef
    lp_build_const_channel_vec.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError:
    pass
try:
    lp_build_const_aos = _libraries['libtinymesa_cpu.so'].lp_build_const_aos
    lp_build_const_aos.restype = LLVMValueRef
    lp_build_const_aos.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    lp_build_const_mask_aos = _libraries['libtinymesa_cpu.so'].lp_build_const_mask_aos
    lp_build_const_mask_aos.restype = LLVMValueRef
    lp_build_const_mask_aos.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    lp_build_const_mask_aos_swizzled = _libraries['libtinymesa_cpu.so'].lp_build_const_mask_aos_swizzled
    lp_build_const_mask_aos_swizzled.restype = LLVMValueRef
    lp_build_const_mask_aos_swizzled.argtypes = [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    lp_build_const_int32 = _libraries['FIXME_STUB'].lp_build_const_int32
    lp_build_const_int32.restype = LLVMValueRef
    lp_build_const_int32.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.c_int32]
except AttributeError:
    pass
try:
    lp_build_const_int64 = _libraries['FIXME_STUB'].lp_build_const_int64
    lp_build_const_int64.restype = LLVMValueRef
    lp_build_const_int64.argtypes = [ctypes.POINTER(struct_gallivm_state), int64_t]
except AttributeError:
    pass
try:
    lp_build_const_float = _libraries['FIXME_STUB'].lp_build_const_float
    lp_build_const_float.restype = LLVMValueRef
    lp_build_const_float.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.c_float]
except AttributeError:
    pass
try:
    lp_build_const_double = _libraries['FIXME_STUB'].lp_build_const_double
    lp_build_const_double.restype = LLVMValueRef
    lp_build_const_double.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.c_float]
except AttributeError:
    pass
try:
    lp_build_const_int_pointer = _libraries['FIXME_STUB'].lp_build_const_int_pointer
    lp_build_const_int_pointer.restype = LLVMValueRef
    lp_build_const_int_pointer.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    lp_build_const_string = _libraries['libtinymesa_cpu.so'].lp_build_const_string
    lp_build_const_string.restype = LLVMValueRef
    lp_build_const_string.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lp_build_const_func_pointer = _libraries['libtinymesa_cpu.so'].lp_build_const_func_pointer
    lp_build_const_func_pointer.restype = LLVMValueRef
    lp_build_const_func_pointer.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(None), LLVMTypeRef, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueType)), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    lp_build_const_func_pointer_from_type = _libraries['libtinymesa_cpu.so'].lp_build_const_func_pointer_from_type
    lp_build_const_func_pointer_from_type.restype = LLVMValueRef
    lp_build_const_func_pointer_from_type.argtypes = [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(None), LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
__all__ = \
    ['ACCESS_CAN_REORDER', 'ACCESS_CAN_SPECULATE', 'ACCESS_COHERENT',
    'ACCESS_CP_GE_COHERENT_AMD', 'ACCESS_FMASK_LOWERED_AMD',
    'ACCESS_INCLUDE_HELPERS', 'ACCESS_IN_BOUNDS',
    'ACCESS_IS_SWIZZLED_AMD', 'ACCESS_KEEP_SCALAR',
    'ACCESS_NON_READABLE', 'ACCESS_NON_TEMPORAL',
    'ACCESS_NON_UNIFORM', 'ACCESS_NON_WRITEABLE', 'ACCESS_RESTRICT',
    'ACCESS_SMEM_AMD', 'ACCESS_USES_FORMAT_AMD', 'ACCESS_VOLATILE',
    'COMPARE_FUNC_ALWAYS', 'COMPARE_FUNC_EQUAL',
    'COMPARE_FUNC_GEQUAL', 'COMPARE_FUNC_GREATER',
    'COMPARE_FUNC_LEQUAL', 'COMPARE_FUNC_LESS', 'COMPARE_FUNC_NEVER',
    'COMPARE_FUNC_NOTEQUAL', 'DERIVATIVE_GROUP_LINEAR',
    'DERIVATIVE_GROUP_NONE', 'DERIVATIVE_GROUP_QUADS',
    'FRAG_DEPTH_LAYOUT_ANY', 'FRAG_DEPTH_LAYOUT_GREATER',
    'FRAG_DEPTH_LAYOUT_LESS', 'FRAG_DEPTH_LAYOUT_NONE',
    'FRAG_DEPTH_LAYOUT_UNCHANGED', 'FRAG_STENCIL_LAYOUT_ANY',
    'FRAG_STENCIL_LAYOUT_GREATER', 'FRAG_STENCIL_LAYOUT_LESS',
    'FRAG_STENCIL_LAYOUT_NONE', 'FRAG_STENCIL_LAYOUT_UNCHANGED',
    'GLSL_CMAT_USE_A', 'GLSL_CMAT_USE_ACCUMULATOR', 'GLSL_CMAT_USE_B',
    'GLSL_CMAT_USE_NONE', 'GLSL_INTERFACE_PACKING_PACKED',
    'GLSL_INTERFACE_PACKING_SHARED', 'GLSL_INTERFACE_PACKING_STD140',
    'GLSL_INTERFACE_PACKING_STD430',
    'GLSL_MATRIX_LAYOUT_COLUMN_MAJOR', 'GLSL_MATRIX_LAYOUT_INHERITED',
    'GLSL_MATRIX_LAYOUT_ROW_MAJOR', 'GLSL_PRECISION_HIGH',
    'GLSL_PRECISION_LOW', 'GLSL_PRECISION_MEDIUM',
    'GLSL_PRECISION_NONE', 'GLSL_SAMPLER_DIM_1D',
    'GLSL_SAMPLER_DIM_2D', 'GLSL_SAMPLER_DIM_3D',
    'GLSL_SAMPLER_DIM_BUF', 'GLSL_SAMPLER_DIM_CUBE',
    'GLSL_SAMPLER_DIM_EXTERNAL', 'GLSL_SAMPLER_DIM_MS',
    'GLSL_SAMPLER_DIM_RECT', 'GLSL_SAMPLER_DIM_SUBPASS',
    'GLSL_SAMPLER_DIM_SUBPASS_MS', 'GLSL_TYPE_ARRAY',
    'GLSL_TYPE_ATOMIC_UINT', 'GLSL_TYPE_BFLOAT16', 'GLSL_TYPE_BOOL',
    'GLSL_TYPE_COOPERATIVE_MATRIX', 'GLSL_TYPE_DOUBLE',
    'GLSL_TYPE_ERROR', 'GLSL_TYPE_FLOAT', 'GLSL_TYPE_FLOAT16',
    'GLSL_TYPE_FLOAT_E4M3FN', 'GLSL_TYPE_FLOAT_E5M2',
    'GLSL_TYPE_IMAGE', 'GLSL_TYPE_INT', 'GLSL_TYPE_INT16',
    'GLSL_TYPE_INT64', 'GLSL_TYPE_INT8', 'GLSL_TYPE_INTERFACE',
    'GLSL_TYPE_SAMPLER', 'GLSL_TYPE_STRUCT', 'GLSL_TYPE_SUBROUTINE',
    'GLSL_TYPE_TEXTURE', 'GLSL_TYPE_UINT', 'GLSL_TYPE_UINT16',
    'GLSL_TYPE_UINT64', 'GLSL_TYPE_UINT8', 'GLSL_TYPE_VOID',
    'LLVMArrayTypeKind', 'LLVMAtomicRMWBinOp',
    'LLVMAtomicRMWBinOpAdd', 'LLVMAtomicRMWBinOpAnd',
    'LLVMAtomicRMWBinOpFAdd', 'LLVMAtomicRMWBinOpFMax',
    'LLVMAtomicRMWBinOpFMin', 'LLVMAtomicRMWBinOpFSub',
    'LLVMAtomicRMWBinOpMax', 'LLVMAtomicRMWBinOpMin',
    'LLVMAtomicRMWBinOpNand', 'LLVMAtomicRMWBinOpOr',
    'LLVMAtomicRMWBinOpSub', 'LLVMAtomicRMWBinOpUDecWrap',
    'LLVMAtomicRMWBinOpUIncWrap', 'LLVMAtomicRMWBinOpUMax',
    'LLVMAtomicRMWBinOpUMin', 'LLVMAtomicRMWBinOpUSubCond',
    'LLVMAtomicRMWBinOpUSubSat', 'LLVMAtomicRMWBinOpXchg',
    'LLVMAtomicRMWBinOpXor', 'LLVMAtomicRMWBinOp__enumvalues',
    'LLVMBFloatTypeKind', 'LLVMBasicBlockRef', 'LLVMBuilderRef',
    'LLVMDoubleTypeKind', 'LLVMFP128TypeKind', 'LLVMFloatTypeKind',
    'LLVMFunctionTypeKind', 'LLVMHalfTypeKind', 'LLVMIntEQ',
    'LLVMIntNE', 'LLVMIntPredicate', 'LLVMIntPredicate__enumvalues',
    'LLVMIntSGE', 'LLVMIntSGT', 'LLVMIntSLE', 'LLVMIntSLT',
    'LLVMIntUGE', 'LLVMIntUGT', 'LLVMIntULE', 'LLVMIntULT',
    'LLVMIntegerTypeKind', 'LLVMLabelTypeKind',
    'LLVMMCJITMemoryManagerRef', 'LLVMMetadataTypeKind',
    'LLVMModuleRef', 'LLVMPPC_FP128TypeKind', 'LLVMPointerTypeKind',
    'LLVMScalableVectorTypeKind', 'LLVMStructTypeKind',
    'LLVMTargetExtTypeKind', 'LLVMTargetLibraryInfoRef',
    'LLVMTargetMachineRef', 'LLVMTokenTypeKind', 'LLVMTypeKind',
    'LLVMTypeKind__enumvalues', 'LLVMTypeRef', 'LLVMValueRef',
    'LLVMVectorTypeKind', 'LLVMVoidTypeKind', 'LLVMX86_AMXTypeKind',
    'LLVMX86_FP80TypeKind', 'LP_BLD_TEX_MODIFIER_EXPLICIT_DERIV',
    'LP_BLD_TEX_MODIFIER_EXPLICIT_LOD',
    'LP_BLD_TEX_MODIFIER_LOD_BIAS', 'LP_BLD_TEX_MODIFIER_LOD_ZERO',
    'LP_BLD_TEX_MODIFIER_NONE', 'LP_BLD_TEX_MODIFIER_PROJECTED',
    'LP_JIT_BUFFER_BASE', 'LP_JIT_BUFFER_NUM_ELEMENTS',
    'LP_JIT_BUFFER_NUM_FIELDS', 'LP_JIT_IMAGE_BASE',
    'LP_JIT_IMAGE_BASE_OFFSET', 'LP_JIT_IMAGE_DEPTH',
    'LP_JIT_IMAGE_HEIGHT', 'LP_JIT_IMAGE_IMG_STRIDE',
    'LP_JIT_IMAGE_NUM_FIELDS', 'LP_JIT_IMAGE_NUM_SAMPLES',
    'LP_JIT_IMAGE_RESIDENCY', 'LP_JIT_IMAGE_ROW_STRIDE',
    'LP_JIT_IMAGE_SAMPLE_STRIDE', 'LP_JIT_IMAGE_WIDTH',
    'LP_JIT_RES_CONSTANTS', 'LP_JIT_RES_COUNT', 'LP_JIT_RES_IMAGES',
    'LP_JIT_RES_SAMPLERS', 'LP_JIT_RES_SSBOS', 'LP_JIT_RES_TEXTURES',
    'LP_JIT_SAMPLER_BORDER_COLOR', 'LP_JIT_SAMPLER_INDEX_DUMMY',
    'LP_JIT_SAMPLER_LOD_BIAS', 'LP_JIT_SAMPLER_MAX_LOD',
    'LP_JIT_SAMPLER_MIN_LOD', 'LP_JIT_SAMPLER_NUM_FIELDS',
    'LP_JIT_TEXTURE_BASE', 'LP_JIT_TEXTURE_DEPTH',
    'LP_JIT_TEXTURE_FIRST_LEVEL', 'LP_JIT_TEXTURE_HEIGHT',
    'LP_JIT_TEXTURE_IMG_STRIDE', 'LP_JIT_TEXTURE_LAST_LEVEL',
    'LP_JIT_TEXTURE_MIP_OFFSETS', 'LP_JIT_TEXTURE_NUM_FIELDS',
    'LP_JIT_TEXTURE_ROW_STRIDE', 'LP_JIT_TEXTURE_WIDTH',
    'LP_JIT_VERTEX_HEADER_CLIP_POS', 'LP_JIT_VERTEX_HEADER_DATA',
    'LP_JIT_VERTEX_HEADER_VERTEX_ID',
    'LP_NIR_CALL_CONTEXT_BLOCK_ID_0',
    'LP_NIR_CALL_CONTEXT_BLOCK_ID_1',
    'LP_NIR_CALL_CONTEXT_BLOCK_ID_2',
    'LP_NIR_CALL_CONTEXT_BLOCK_SIZE_0',
    'LP_NIR_CALL_CONTEXT_BLOCK_SIZE_1',
    'LP_NIR_CALL_CONTEXT_BLOCK_SIZE_2', 'LP_NIR_CALL_CONTEXT_CONTEXT',
    'LP_NIR_CALL_CONTEXT_GRID_SIZE_0',
    'LP_NIR_CALL_CONTEXT_GRID_SIZE_1',
    'LP_NIR_CALL_CONTEXT_GRID_SIZE_2', 'LP_NIR_CALL_CONTEXT_MAX_ARGS',
    'LP_NIR_CALL_CONTEXT_RESOURCES', 'LP_NIR_CALL_CONTEXT_SCRATCH',
    'LP_NIR_CALL_CONTEXT_SHARED', 'LP_NIR_CALL_CONTEXT_THREAD_ID_0',
    'LP_NIR_CALL_CONTEXT_THREAD_ID_1',
    'LP_NIR_CALL_CONTEXT_THREAD_ID_2', 'LP_NIR_CALL_CONTEXT_WORK_DIM',
    'LP_SAMPLER_LOD_PER_ELEMENT', 'LP_SAMPLER_LOD_PER_QUAD',
    'LP_SAMPLER_LOD_SCALAR', 'MESA_LOG_DEBUG', 'MESA_LOG_ERROR',
    'MESA_LOG_INFO', 'MESA_LOG_WARN', 'MESA_PRIM_COUNT',
    'MESA_PRIM_LINES', 'MESA_PRIM_LINES_ADJACENCY',
    'MESA_PRIM_LINE_LOOP', 'MESA_PRIM_LINE_STRIP',
    'MESA_PRIM_LINE_STRIP_ADJACENCY', 'MESA_PRIM_MAX',
    'MESA_PRIM_PATCHES', 'MESA_PRIM_POINTS', 'MESA_PRIM_POLYGON',
    'MESA_PRIM_QUADS', 'MESA_PRIM_QUAD_STRIP', 'MESA_PRIM_TRIANGLES',
    'MESA_PRIM_TRIANGLES_ADJACENCY', 'MESA_PRIM_TRIANGLE_FAN',
    'MESA_PRIM_TRIANGLE_STRIP', 'MESA_PRIM_TRIANGLE_STRIP_ADJACENCY',
    'MESA_PRIM_UNKNOWN', 'MESA_SHADER_ANY_HIT',
    'MESA_SHADER_CALLABLE', 'MESA_SHADER_CLOSEST_HIT',
    'MESA_SHADER_COMPUTE', 'MESA_SHADER_FRAGMENT',
    'MESA_SHADER_GEOMETRY', 'MESA_SHADER_INTERSECTION',
    'MESA_SHADER_KERNEL', 'MESA_SHADER_MESH', 'MESA_SHADER_MISS',
    'MESA_SHADER_NONE', 'MESA_SHADER_RAYGEN', 'MESA_SHADER_TASK',
    'MESA_SHADER_TESS_CTRL', 'MESA_SHADER_TESS_EVAL',
    'MESA_SHADER_VERTEX', 'NAK_TS_DOMAIN_ISOLINE',
    'NAK_TS_DOMAIN_QUAD', 'NAK_TS_DOMAIN_TRIANGLE',
    'NAK_TS_PRIMS_LINES', 'NAK_TS_PRIMS_POINTS',
    'NAK_TS_PRIMS_TRIANGLES_CCW', 'NAK_TS_PRIMS_TRIANGLES_CW',
    'NAK_TS_SPACING_FRACT_EVEN', 'NAK_TS_SPACING_FRACT_ODD',
    'NAK_TS_SPACING_INTEGER', 'NIR_CMAT_A_SIGNED',
    'NIR_CMAT_B_SIGNED', 'NIR_CMAT_C_SIGNED',
    'NIR_CMAT_RESULT_SIGNED', 'NIR_INTRINSIC_ACCESS',
    'NIR_INTRINSIC_ALIGN_MUL', 'NIR_INTRINSIC_ALIGN_OFFSET',
    'NIR_INTRINSIC_ALU_OP', 'NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD',
    'NIR_INTRINSIC_ATOMIC_OP', 'NIR_INTRINSIC_BASE',
    'NIR_INTRINSIC_BINDING', 'NIR_INTRINSIC_BIT_SIZE',
    'NIR_INTRINSIC_CALL_IDX', 'NIR_INTRINSIC_CAN_ELIMINATE',
    'NIR_INTRINSIC_CAN_REORDER', 'NIR_INTRINSIC_CLUSTER_SIZE',
    'NIR_INTRINSIC_CMAT_DESC', 'NIR_INTRINSIC_CMAT_SIGNED_MASK',
    'NIR_INTRINSIC_COLUMN', 'NIR_INTRINSIC_COMMITTED',
    'NIR_INTRINSIC_COMPONENT', 'NIR_INTRINSIC_DESC_SET',
    'NIR_INTRINSIC_DESC_TYPE', 'NIR_INTRINSIC_DEST_BASE_TYPE',
    'NIR_INTRINSIC_DEST_TYPE', 'NIR_INTRINSIC_DIVERGENT',
    'NIR_INTRINSIC_DRIVER_LOCATION', 'NIR_INTRINSIC_DST_ACCESS',
    'NIR_INTRINSIC_DST_CMAT_DESC', 'NIR_INTRINSIC_EXECUTION_SCOPE',
    'NIR_INTRINSIC_EXPLICIT_COORD', 'NIR_INTRINSIC_FETCH_INACTIVE',
    'NIR_INTRINSIC_FLAGS', 'NIR_INTRINSIC_FMT_IDX',
    'NIR_INTRINSIC_FORMAT', 'NIR_INTRINSIC_IMAGE_ARRAY',
    'NIR_INTRINSIC_IMAGE_DIM', 'NIR_INTRINSIC_INTERP_MODE',
    'NIR_INTRINSIC_IO_SEMANTICS', 'NIR_INTRINSIC_IO_XFB',
    'NIR_INTRINSIC_IO_XFB2', 'NIR_INTRINSIC_LEGACY_FABS',
    'NIR_INTRINSIC_LEGACY_FNEG', 'NIR_INTRINSIC_LEGACY_FSAT',
    'NIR_INTRINSIC_MATRIX_LAYOUT', 'NIR_INTRINSIC_MEMORY_MODES',
    'NIR_INTRINSIC_MEMORY_SCOPE', 'NIR_INTRINSIC_MEMORY_SEMANTICS',
    'NIR_INTRINSIC_NEG_HI_AMD', 'NIR_INTRINSIC_NEG_LO_AMD',
    'NIR_INTRINSIC_NUM_ARRAY_ELEMS', 'NIR_INTRINSIC_NUM_COMPONENTS',
    'NIR_INTRINSIC_NUM_INDEX_FLAGS', 'NIR_INTRINSIC_OFFSET0',
    'NIR_INTRINSIC_OFFSET1', 'NIR_INTRINSIC_PARAM_IDX',
    'NIR_INTRINSIC_PREAMBLE_CLASS', 'NIR_INTRINSIC_QUADGROUP',
    'NIR_INTRINSIC_RANGE', 'NIR_INTRINSIC_RANGE_BASE',
    'NIR_INTRINSIC_RAY_QUERY_VALUE', 'NIR_INTRINSIC_REDUCTION_OP',
    'NIR_INTRINSIC_REPEAT_COUNT',
    'NIR_INTRINSIC_RESOURCE_ACCESS_INTEL',
    'NIR_INTRINSIC_RESOURCE_BLOCK_INTEL',
    'NIR_INTRINSIC_ROUNDING_MODE', 'NIR_INTRINSIC_SATURATE',
    'NIR_INTRINSIC_SIGN_EXTEND', 'NIR_INTRINSIC_SRC_ACCESS',
    'NIR_INTRINSIC_SRC_BASE_TYPE', 'NIR_INTRINSIC_SRC_BASE_TYPE2',
    'NIR_INTRINSIC_SRC_CMAT_DESC', 'NIR_INTRINSIC_SRC_TYPE',
    'NIR_INTRINSIC_ST64', 'NIR_INTRINSIC_STACK_SIZE',
    'NIR_INTRINSIC_STREAM_ID', 'NIR_INTRINSIC_SUBGROUP',
    'NIR_INTRINSIC_SWIZZLE_MASK', 'NIR_INTRINSIC_SYNCHRONOUS',
    'NIR_INTRINSIC_SYSTOLIC_DEPTH', 'NIR_INTRINSIC_UCP_ID',
    'NIR_INTRINSIC_VALUE_ID', 'NIR_INTRINSIC_WRITE_MASK',
    'NIR_MEMORY_ACQUIRE', 'NIR_MEMORY_ACQ_REL',
    'NIR_MEMORY_MAKE_AVAILABLE', 'NIR_MEMORY_MAKE_VISIBLE',
    'NIR_MEMORY_RELEASE', 'NIR_OP_IS_2SRC_COMMUTATIVE',
    'NIR_OP_IS_ASSOCIATIVE', 'NIR_OP_IS_SELECTION',
    'NUM_TOTAL_VARYING_SLOTS', 'NV_DEVICE_TYPE_DIS',
    'NV_DEVICE_TYPE_IGP', 'NV_DEVICE_TYPE_SOC', 'PIPE_BUFFER',
    'PIPE_FORMAT_A16_FLOAT', 'PIPE_FORMAT_A16_SINT',
    'PIPE_FORMAT_A16_SNORM', 'PIPE_FORMAT_A16_UINT',
    'PIPE_FORMAT_A16_UNORM', 'PIPE_FORMAT_A1B5G5R5_UINT',
    'PIPE_FORMAT_A1B5G5R5_UNORM', 'PIPE_FORMAT_A1R5G5B5_UINT',
    'PIPE_FORMAT_A1R5G5B5_UNORM', 'PIPE_FORMAT_A2B10G10R10_UINT',
    'PIPE_FORMAT_A2B10G10R10_UNORM', 'PIPE_FORMAT_A2R10G10B10_UINT',
    'PIPE_FORMAT_A2R10G10B10_UNORM', 'PIPE_FORMAT_A32_FLOAT',
    'PIPE_FORMAT_A32_SINT', 'PIPE_FORMAT_A32_UINT',
    'PIPE_FORMAT_A4B4G4R4_UINT', 'PIPE_FORMAT_A4B4G4R4_UNORM',
    'PIPE_FORMAT_A4R4G4B4_UINT', 'PIPE_FORMAT_A4R4G4B4_UNORM',
    'PIPE_FORMAT_A4R4_UNORM', 'PIPE_FORMAT_A8B8G8R8_SINT',
    'PIPE_FORMAT_A8B8G8R8_SNORM', 'PIPE_FORMAT_A8B8G8R8_SRGB',
    'PIPE_FORMAT_A8B8G8R8_SSCALED', 'PIPE_FORMAT_A8B8G8R8_UINT',
    'PIPE_FORMAT_A8B8G8R8_UNORM', 'PIPE_FORMAT_A8B8G8R8_USCALED',
    'PIPE_FORMAT_A8R8G8B8_SINT', 'PIPE_FORMAT_A8R8G8B8_SNORM',
    'PIPE_FORMAT_A8R8G8B8_SRGB', 'PIPE_FORMAT_A8R8G8B8_UINT',
    'PIPE_FORMAT_A8R8G8B8_UNORM', 'PIPE_FORMAT_A8R8_UNORM',
    'PIPE_FORMAT_A8_SINT', 'PIPE_FORMAT_A8_SNORM',
    'PIPE_FORMAT_A8_UINT', 'PIPE_FORMAT_A8_UNORM',
    'PIPE_FORMAT_ASTC_10x10', 'PIPE_FORMAT_ASTC_10x10_FLOAT',
    'PIPE_FORMAT_ASTC_10x10_SRGB', 'PIPE_FORMAT_ASTC_10x5',
    'PIPE_FORMAT_ASTC_10x5_FLOAT', 'PIPE_FORMAT_ASTC_10x5_SRGB',
    'PIPE_FORMAT_ASTC_10x6', 'PIPE_FORMAT_ASTC_10x6_FLOAT',
    'PIPE_FORMAT_ASTC_10x6_SRGB', 'PIPE_FORMAT_ASTC_10x8',
    'PIPE_FORMAT_ASTC_10x8_FLOAT', 'PIPE_FORMAT_ASTC_10x8_SRGB',
    'PIPE_FORMAT_ASTC_12x10', 'PIPE_FORMAT_ASTC_12x10_FLOAT',
    'PIPE_FORMAT_ASTC_12x10_SRGB', 'PIPE_FORMAT_ASTC_12x12',
    'PIPE_FORMAT_ASTC_12x12_FLOAT', 'PIPE_FORMAT_ASTC_12x12_SRGB',
    'PIPE_FORMAT_ASTC_3x3x3', 'PIPE_FORMAT_ASTC_3x3x3_SRGB',
    'PIPE_FORMAT_ASTC_4x3x3', 'PIPE_FORMAT_ASTC_4x3x3_SRGB',
    'PIPE_FORMAT_ASTC_4x4', 'PIPE_FORMAT_ASTC_4x4_FLOAT',
    'PIPE_FORMAT_ASTC_4x4_SRGB', 'PIPE_FORMAT_ASTC_4x4x3',
    'PIPE_FORMAT_ASTC_4x4x3_SRGB', 'PIPE_FORMAT_ASTC_4x4x4',
    'PIPE_FORMAT_ASTC_4x4x4_SRGB', 'PIPE_FORMAT_ASTC_5x4',
    'PIPE_FORMAT_ASTC_5x4_FLOAT', 'PIPE_FORMAT_ASTC_5x4_SRGB',
    'PIPE_FORMAT_ASTC_5x4x4', 'PIPE_FORMAT_ASTC_5x4x4_SRGB',
    'PIPE_FORMAT_ASTC_5x5', 'PIPE_FORMAT_ASTC_5x5_FLOAT',
    'PIPE_FORMAT_ASTC_5x5_SRGB', 'PIPE_FORMAT_ASTC_5x5x4',
    'PIPE_FORMAT_ASTC_5x5x4_SRGB', 'PIPE_FORMAT_ASTC_5x5x5',
    'PIPE_FORMAT_ASTC_5x5x5_SRGB', 'PIPE_FORMAT_ASTC_6x5',
    'PIPE_FORMAT_ASTC_6x5_FLOAT', 'PIPE_FORMAT_ASTC_6x5_SRGB',
    'PIPE_FORMAT_ASTC_6x5x5', 'PIPE_FORMAT_ASTC_6x5x5_SRGB',
    'PIPE_FORMAT_ASTC_6x6', 'PIPE_FORMAT_ASTC_6x6_FLOAT',
    'PIPE_FORMAT_ASTC_6x6_SRGB', 'PIPE_FORMAT_ASTC_6x6x5',
    'PIPE_FORMAT_ASTC_6x6x5_SRGB', 'PIPE_FORMAT_ASTC_6x6x6',
    'PIPE_FORMAT_ASTC_6x6x6_SRGB', 'PIPE_FORMAT_ASTC_8x5',
    'PIPE_FORMAT_ASTC_8x5_FLOAT', 'PIPE_FORMAT_ASTC_8x5_SRGB',
    'PIPE_FORMAT_ASTC_8x6', 'PIPE_FORMAT_ASTC_8x6_FLOAT',
    'PIPE_FORMAT_ASTC_8x6_SRGB', 'PIPE_FORMAT_ASTC_8x8',
    'PIPE_FORMAT_ASTC_8x8_FLOAT', 'PIPE_FORMAT_ASTC_8x8_SRGB',
    'PIPE_FORMAT_ATC_RGB', 'PIPE_FORMAT_ATC_RGBA_EXPLICIT',
    'PIPE_FORMAT_ATC_RGBA_INTERPOLATED', 'PIPE_FORMAT_AYUV',
    'PIPE_FORMAT_B10G10R10A2_SINT', 'PIPE_FORMAT_B10G10R10A2_SNORM',
    'PIPE_FORMAT_B10G10R10A2_SSCALED', 'PIPE_FORMAT_B10G10R10A2_UINT',
    'PIPE_FORMAT_B10G10R10A2_UNORM',
    'PIPE_FORMAT_B10G10R10A2_USCALED', 'PIPE_FORMAT_B10G10R10X2_SINT',
    'PIPE_FORMAT_B10G10R10X2_SNORM', 'PIPE_FORMAT_B10G10R10X2_UNORM',
    'PIPE_FORMAT_B2G3R3_UINT', 'PIPE_FORMAT_B2G3R3_UNORM',
    'PIPE_FORMAT_B4G4R4A4_UINT', 'PIPE_FORMAT_B4G4R4A4_UNORM',
    'PIPE_FORMAT_B4G4R4X4_UNORM', 'PIPE_FORMAT_B5G5R5A1_UINT',
    'PIPE_FORMAT_B5G5R5A1_UNORM', 'PIPE_FORMAT_B5G5R5X1_UNORM',
    'PIPE_FORMAT_B5G6R5_SRGB', 'PIPE_FORMAT_B5G6R5_UINT',
    'PIPE_FORMAT_B5G6R5_UNORM', 'PIPE_FORMAT_B8G8R8A8_SINT',
    'PIPE_FORMAT_B8G8R8A8_SNORM', 'PIPE_FORMAT_B8G8R8A8_SRGB',
    'PIPE_FORMAT_B8G8R8A8_SSCALED', 'PIPE_FORMAT_B8G8R8A8_UINT',
    'PIPE_FORMAT_B8G8R8A8_UNORM', 'PIPE_FORMAT_B8G8R8A8_USCALED',
    'PIPE_FORMAT_B8G8R8X8_SINT', 'PIPE_FORMAT_B8G8R8X8_SNORM',
    'PIPE_FORMAT_B8G8R8X8_SRGB', 'PIPE_FORMAT_B8G8R8X8_UINT',
    'PIPE_FORMAT_B8G8R8X8_UNORM', 'PIPE_FORMAT_B8G8R8_SINT',
    'PIPE_FORMAT_B8G8R8_SNORM', 'PIPE_FORMAT_B8G8R8_SRGB',
    'PIPE_FORMAT_B8G8R8_SSCALED', 'PIPE_FORMAT_B8G8R8_UINT',
    'PIPE_FORMAT_B8G8R8_UNORM', 'PIPE_FORMAT_B8G8R8_USCALED',
    'PIPE_FORMAT_B8G8_R8G8_UNORM', 'PIPE_FORMAT_B8R8_G8R8_UNORM',
    'PIPE_FORMAT_BPTC_RGBA_UNORM', 'PIPE_FORMAT_BPTC_RGB_FLOAT',
    'PIPE_FORMAT_BPTC_RGB_UFLOAT', 'PIPE_FORMAT_BPTC_SRGBA',
    'PIPE_FORMAT_COUNT', 'PIPE_FORMAT_DXT1_RGB',
    'PIPE_FORMAT_DXT1_RGBA', 'PIPE_FORMAT_DXT1_SRGB',
    'PIPE_FORMAT_DXT1_SRGBA', 'PIPE_FORMAT_DXT3_RGBA',
    'PIPE_FORMAT_DXT3_SRGBA', 'PIPE_FORMAT_DXT5_RGBA',
    'PIPE_FORMAT_DXT5_SRGBA', 'PIPE_FORMAT_ETC1_RGB8',
    'PIPE_FORMAT_ETC2_R11_SNORM', 'PIPE_FORMAT_ETC2_R11_UNORM',
    'PIPE_FORMAT_ETC2_RG11_SNORM', 'PIPE_FORMAT_ETC2_RG11_UNORM',
    'PIPE_FORMAT_ETC2_RGB8', 'PIPE_FORMAT_ETC2_RGB8A1',
    'PIPE_FORMAT_ETC2_RGBA8', 'PIPE_FORMAT_ETC2_SRGB8',
    'PIPE_FORMAT_ETC2_SRGB8A1', 'PIPE_FORMAT_ETC2_SRGBA8',
    'PIPE_FORMAT_FXT1_RGB', 'PIPE_FORMAT_FXT1_RGBA',
    'PIPE_FORMAT_G16R16_SINT', 'PIPE_FORMAT_G16R16_SNORM',
    'PIPE_FORMAT_G16R16_UNORM', 'PIPE_FORMAT_G8B8_G8R8_UNORM',
    'PIPE_FORMAT_G8R8_B8R8_UNORM', 'PIPE_FORMAT_G8R8_G8B8_UNORM',
    'PIPE_FORMAT_G8R8_SINT', 'PIPE_FORMAT_G8R8_SNORM',
    'PIPE_FORMAT_G8R8_UNORM', 'PIPE_FORMAT_G8_B8R8_420_UNORM',
    'PIPE_FORMAT_G8_B8R8_422_UNORM', 'PIPE_FORMAT_G8_B8_R8_420_UNORM',
    'PIPE_FORMAT_I16_FLOAT', 'PIPE_FORMAT_I16_SINT',
    'PIPE_FORMAT_I16_SNORM', 'PIPE_FORMAT_I16_UINT',
    'PIPE_FORMAT_I16_UNORM', 'PIPE_FORMAT_I32_FLOAT',
    'PIPE_FORMAT_I32_SINT', 'PIPE_FORMAT_I32_UINT',
    'PIPE_FORMAT_I8_SINT', 'PIPE_FORMAT_I8_SNORM',
    'PIPE_FORMAT_I8_UINT', 'PIPE_FORMAT_I8_UNORM', 'PIPE_FORMAT_IYUV',
    'PIPE_FORMAT_L16A16_FLOAT', 'PIPE_FORMAT_L16A16_SINT',
    'PIPE_FORMAT_L16A16_SNORM', 'PIPE_FORMAT_L16A16_UINT',
    'PIPE_FORMAT_L16A16_UNORM', 'PIPE_FORMAT_L16_FLOAT',
    'PIPE_FORMAT_L16_SINT', 'PIPE_FORMAT_L16_SNORM',
    'PIPE_FORMAT_L16_UINT', 'PIPE_FORMAT_L16_UNORM',
    'PIPE_FORMAT_L32A32_FLOAT', 'PIPE_FORMAT_L32A32_SINT',
    'PIPE_FORMAT_L32A32_UINT', 'PIPE_FORMAT_L32_FLOAT',
    'PIPE_FORMAT_L32_SINT', 'PIPE_FORMAT_L32_UINT',
    'PIPE_FORMAT_L4A4_UNORM', 'PIPE_FORMAT_L8A8_SINT',
    'PIPE_FORMAT_L8A8_SNORM', 'PIPE_FORMAT_L8A8_SRGB',
    'PIPE_FORMAT_L8A8_UINT', 'PIPE_FORMAT_L8A8_UNORM',
    'PIPE_FORMAT_L8_SINT', 'PIPE_FORMAT_L8_SNORM',
    'PIPE_FORMAT_L8_SRGB', 'PIPE_FORMAT_L8_UINT',
    'PIPE_FORMAT_L8_UNORM', 'PIPE_FORMAT_LATC1_SNORM',
    'PIPE_FORMAT_LATC1_UNORM', 'PIPE_FORMAT_LATC2_SNORM',
    'PIPE_FORMAT_LATC2_UNORM', 'PIPE_FORMAT_NONE', 'PIPE_FORMAT_NV12',
    'PIPE_FORMAT_NV15', 'PIPE_FORMAT_NV16', 'PIPE_FORMAT_NV20',
    'PIPE_FORMAT_NV21', 'PIPE_FORMAT_P010', 'PIPE_FORMAT_P012',
    'PIPE_FORMAT_P016', 'PIPE_FORMAT_P030',
    'PIPE_FORMAT_R10G10B10A2_SINT', 'PIPE_FORMAT_R10G10B10A2_SNORM',
    'PIPE_FORMAT_R10G10B10A2_SSCALED', 'PIPE_FORMAT_R10G10B10A2_UINT',
    'PIPE_FORMAT_R10G10B10A2_UNORM',
    'PIPE_FORMAT_R10G10B10A2_USCALED', 'PIPE_FORMAT_R10G10B10X2_SINT',
    'PIPE_FORMAT_R10G10B10X2_SNORM', 'PIPE_FORMAT_R10G10B10X2_UNORM',
    'PIPE_FORMAT_R10G10B10X2_USCALED',
    'PIPE_FORMAT_R10G10B10_420_UNORM_PACKED',
    'PIPE_FORMAT_R10SG10SB10SA2U_NORM',
    'PIPE_FORMAT_R10_G10B10_420_UNORM',
    'PIPE_FORMAT_R10_G10B10_422_UNORM', 'PIPE_FORMAT_R11G11B10_FLOAT',
    'PIPE_FORMAT_R16A16_FLOAT', 'PIPE_FORMAT_R16A16_SINT',
    'PIPE_FORMAT_R16A16_SNORM', 'PIPE_FORMAT_R16A16_UINT',
    'PIPE_FORMAT_R16A16_UNORM', 'PIPE_FORMAT_R16G16B16A16_FLOAT',
    'PIPE_FORMAT_R16G16B16A16_SINT', 'PIPE_FORMAT_R16G16B16A16_SNORM',
    'PIPE_FORMAT_R16G16B16A16_SSCALED',
    'PIPE_FORMAT_R16G16B16A16_UINT', 'PIPE_FORMAT_R16G16B16A16_UNORM',
    'PIPE_FORMAT_R16G16B16A16_USCALED',
    'PIPE_FORMAT_R16G16B16X16_FLOAT', 'PIPE_FORMAT_R16G16B16X16_SINT',
    'PIPE_FORMAT_R16G16B16X16_SNORM', 'PIPE_FORMAT_R16G16B16X16_UINT',
    'PIPE_FORMAT_R16G16B16X16_UNORM', 'PIPE_FORMAT_R16G16B16_FLOAT',
    'PIPE_FORMAT_R16G16B16_SINT', 'PIPE_FORMAT_R16G16B16_SNORM',
    'PIPE_FORMAT_R16G16B16_SSCALED', 'PIPE_FORMAT_R16G16B16_UINT',
    'PIPE_FORMAT_R16G16B16_UNORM', 'PIPE_FORMAT_R16G16B16_USCALED',
    'PIPE_FORMAT_R16G16_FLOAT', 'PIPE_FORMAT_R16G16_SINT',
    'PIPE_FORMAT_R16G16_SNORM', 'PIPE_FORMAT_R16G16_SSCALED',
    'PIPE_FORMAT_R16G16_UINT', 'PIPE_FORMAT_R16G16_UNORM',
    'PIPE_FORMAT_R16G16_USCALED', 'PIPE_FORMAT_R16_FLOAT',
    'PIPE_FORMAT_R16_SINT', 'PIPE_FORMAT_R16_SNORM',
    'PIPE_FORMAT_R16_SSCALED', 'PIPE_FORMAT_R16_UINT',
    'PIPE_FORMAT_R16_UNORM', 'PIPE_FORMAT_R16_USCALED',
    'PIPE_FORMAT_R1_UNORM', 'PIPE_FORMAT_R32A32_FLOAT',
    'PIPE_FORMAT_R32A32_SINT', 'PIPE_FORMAT_R32A32_UINT',
    'PIPE_FORMAT_R32G32B32A32_FIXED',
    'PIPE_FORMAT_R32G32B32A32_FLOAT', 'PIPE_FORMAT_R32G32B32A32_SINT',
    'PIPE_FORMAT_R32G32B32A32_SNORM',
    'PIPE_FORMAT_R32G32B32A32_SSCALED',
    'PIPE_FORMAT_R32G32B32A32_UINT', 'PIPE_FORMAT_R32G32B32A32_UNORM',
    'PIPE_FORMAT_R32G32B32A32_USCALED',
    'PIPE_FORMAT_R32G32B32X32_FLOAT', 'PIPE_FORMAT_R32G32B32X32_SINT',
    'PIPE_FORMAT_R32G32B32X32_UINT', 'PIPE_FORMAT_R32G32B32_FIXED',
    'PIPE_FORMAT_R32G32B32_FLOAT', 'PIPE_FORMAT_R32G32B32_SINT',
    'PIPE_FORMAT_R32G32B32_SNORM', 'PIPE_FORMAT_R32G32B32_SSCALED',
    'PIPE_FORMAT_R32G32B32_UINT', 'PIPE_FORMAT_R32G32B32_UNORM',
    'PIPE_FORMAT_R32G32B32_USCALED', 'PIPE_FORMAT_R32G32_FIXED',
    'PIPE_FORMAT_R32G32_FLOAT', 'PIPE_FORMAT_R32G32_SINT',
    'PIPE_FORMAT_R32G32_SNORM', 'PIPE_FORMAT_R32G32_SSCALED',
    'PIPE_FORMAT_R32G32_UINT', 'PIPE_FORMAT_R32G32_UNORM',
    'PIPE_FORMAT_R32G32_USCALED', 'PIPE_FORMAT_R32_FIXED',
    'PIPE_FORMAT_R32_FLOAT', 'PIPE_FORMAT_R32_SINT',
    'PIPE_FORMAT_R32_SNORM', 'PIPE_FORMAT_R32_SSCALED',
    'PIPE_FORMAT_R32_UINT', 'PIPE_FORMAT_R32_UNORM',
    'PIPE_FORMAT_R32_USCALED', 'PIPE_FORMAT_R3G3B2_UINT',
    'PIPE_FORMAT_R3G3B2_UNORM', 'PIPE_FORMAT_R4A4_UNORM',
    'PIPE_FORMAT_R4G4B4A4_UINT', 'PIPE_FORMAT_R4G4B4A4_UNORM',
    'PIPE_FORMAT_R4G4B4X4_UNORM', 'PIPE_FORMAT_R5G5B5A1_UINT',
    'PIPE_FORMAT_R5G5B5A1_UNORM', 'PIPE_FORMAT_R5G5B5X1_UNORM',
    'PIPE_FORMAT_R5G6B5_SRGB', 'PIPE_FORMAT_R5G6B5_UINT',
    'PIPE_FORMAT_R5G6B5_UNORM', 'PIPE_FORMAT_R5SG5SB6U_NORM',
    'PIPE_FORMAT_R64G64B64A64_FLOAT', 'PIPE_FORMAT_R64G64B64A64_SINT',
    'PIPE_FORMAT_R64G64B64A64_UINT', 'PIPE_FORMAT_R64G64B64_FLOAT',
    'PIPE_FORMAT_R64G64B64_SINT', 'PIPE_FORMAT_R64G64B64_UINT',
    'PIPE_FORMAT_R64G64_FLOAT', 'PIPE_FORMAT_R64G64_SINT',
    'PIPE_FORMAT_R64G64_UINT', 'PIPE_FORMAT_R64_FLOAT',
    'PIPE_FORMAT_R64_SINT', 'PIPE_FORMAT_R64_UINT',
    'PIPE_FORMAT_R8A8_SINT', 'PIPE_FORMAT_R8A8_SNORM',
    'PIPE_FORMAT_R8A8_UINT', 'PIPE_FORMAT_R8A8_UNORM',
    'PIPE_FORMAT_R8B8_R8G8_UNORM', 'PIPE_FORMAT_R8G8B8A8_SINT',
    'PIPE_FORMAT_R8G8B8A8_SNORM', 'PIPE_FORMAT_R8G8B8A8_SRGB',
    'PIPE_FORMAT_R8G8B8A8_SSCALED', 'PIPE_FORMAT_R8G8B8A8_UINT',
    'PIPE_FORMAT_R8G8B8A8_UNORM', 'PIPE_FORMAT_R8G8B8A8_USCALED',
    'PIPE_FORMAT_R8G8B8X8_SINT', 'PIPE_FORMAT_R8G8B8X8_SNORM',
    'PIPE_FORMAT_R8G8B8X8_SRGB', 'PIPE_FORMAT_R8G8B8X8_UINT',
    'PIPE_FORMAT_R8G8B8X8_UNORM',
    'PIPE_FORMAT_R8G8B8_420_UNORM_PACKED', 'PIPE_FORMAT_R8G8B8_SINT',
    'PIPE_FORMAT_R8G8B8_SNORM', 'PIPE_FORMAT_R8G8B8_SRGB',
    'PIPE_FORMAT_R8G8B8_SSCALED', 'PIPE_FORMAT_R8G8B8_UINT',
    'PIPE_FORMAT_R8G8B8_UNORM', 'PIPE_FORMAT_R8G8B8_USCALED',
    'PIPE_FORMAT_R8G8Bx_SNORM', 'PIPE_FORMAT_R8G8_B8G8_UNORM',
    'PIPE_FORMAT_R8G8_R8B8_UNORM', 'PIPE_FORMAT_R8G8_SINT',
    'PIPE_FORMAT_R8G8_SNORM', 'PIPE_FORMAT_R8G8_SRGB',
    'PIPE_FORMAT_R8G8_SSCALED', 'PIPE_FORMAT_R8G8_UINT',
    'PIPE_FORMAT_R8G8_UNORM', 'PIPE_FORMAT_R8G8_USCALED',
    'PIPE_FORMAT_R8SG8SB8UX8U_NORM', 'PIPE_FORMAT_R8_B8G8_420_UNORM',
    'PIPE_FORMAT_R8_B8G8_422_UNORM', 'PIPE_FORMAT_R8_B8_G8_420_UNORM',
    'PIPE_FORMAT_R8_G8B8_420_UNORM', 'PIPE_FORMAT_R8_G8B8_422_UNORM',
    'PIPE_FORMAT_R8_G8_B8_420_UNORM', 'PIPE_FORMAT_R8_G8_B8_UNORM',
    'PIPE_FORMAT_R8_SINT', 'PIPE_FORMAT_R8_SNORM',
    'PIPE_FORMAT_R8_SRGB', 'PIPE_FORMAT_R8_SSCALED',
    'PIPE_FORMAT_R8_UINT', 'PIPE_FORMAT_R8_UNORM',
    'PIPE_FORMAT_R8_USCALED', 'PIPE_FORMAT_R9G9B9E5_FLOAT',
    'PIPE_FORMAT_RGTC1_SNORM', 'PIPE_FORMAT_RGTC1_UNORM',
    'PIPE_FORMAT_RGTC2_SNORM', 'PIPE_FORMAT_RGTC2_UNORM',
    'PIPE_FORMAT_S8X24_UINT', 'PIPE_FORMAT_S8_UINT',
    'PIPE_FORMAT_S8_UINT_Z24_UNORM', 'PIPE_FORMAT_UYVY',
    'PIPE_FORMAT_VYUY', 'PIPE_FORMAT_X1B5G5R5_UNORM',
    'PIPE_FORMAT_X1R5G5B5_UNORM', 'PIPE_FORMAT_X24S8_UINT',
    'PIPE_FORMAT_X32_S8X24_UINT',
    'PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM',
    'PIPE_FORMAT_X4R12X4G12_UNORM', 'PIPE_FORMAT_X4R12_UNORM',
    'PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM',
    'PIPE_FORMAT_X6R10X6G10_UNORM', 'PIPE_FORMAT_X6R10_UNORM',
    'PIPE_FORMAT_X8B8G8R8_SINT', 'PIPE_FORMAT_X8B8G8R8_SNORM',
    'PIPE_FORMAT_X8B8G8R8_SRGB', 'PIPE_FORMAT_X8B8G8R8_UNORM',
    'PIPE_FORMAT_X8R8G8B8_SINT', 'PIPE_FORMAT_X8R8G8B8_SNORM',
    'PIPE_FORMAT_X8R8G8B8_SRGB', 'PIPE_FORMAT_X8R8G8B8_UNORM',
    'PIPE_FORMAT_X8Z24_UNORM', 'PIPE_FORMAT_XYUV',
    'PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED',
    'PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM',
    'PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM',
    'PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM',
    'PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM',
    'PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM',
    'PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM',
    'PIPE_FORMAT_Y16_U16V16_422_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_420_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_422_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_444_UNORM', 'PIPE_FORMAT_Y210',
    'PIPE_FORMAT_Y212', 'PIPE_FORMAT_Y216', 'PIPE_FORMAT_Y410',
    'PIPE_FORMAT_Y412', 'PIPE_FORMAT_Y416',
    'PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED', 'PIPE_FORMAT_Y8_400_UNORM',
    'PIPE_FORMAT_Y8_U8_V8_422_UNORM',
    'PIPE_FORMAT_Y8_U8_V8_440_UNORM',
    'PIPE_FORMAT_Y8_U8_V8_444_UNORM', 'PIPE_FORMAT_Y8_UNORM',
    'PIPE_FORMAT_YUYV', 'PIPE_FORMAT_YV12', 'PIPE_FORMAT_YV16',
    'PIPE_FORMAT_YVYU', 'PIPE_FORMAT_Z16_UNORM',
    'PIPE_FORMAT_Z16_UNORM_S8_UINT', 'PIPE_FORMAT_Z24X8_UNORM',
    'PIPE_FORMAT_Z24_UNORM_S8_UINT',
    'PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8',
    'PIPE_FORMAT_Z32_FLOAT', 'PIPE_FORMAT_Z32_FLOAT_S8X24_UINT',
    'PIPE_FORMAT_Z32_UNORM', 'PIPE_MAX_TEXTURE_TYPES',
    'PIPE_SHADER_COMPUTE', 'PIPE_SHADER_FRAGMENT',
    'PIPE_SHADER_GEOMETRY', 'PIPE_SHADER_MESH',
    'PIPE_SHADER_MESH_TYPES', 'PIPE_SHADER_TASK',
    'PIPE_SHADER_TESS_CTRL', 'PIPE_SHADER_TESS_EVAL',
    'PIPE_SHADER_TYPES', 'PIPE_SHADER_VERTEX', 'PIPE_TEXTURE_1D',
    'PIPE_TEXTURE_1D_ARRAY', 'PIPE_TEXTURE_2D',
    'PIPE_TEXTURE_2D_ARRAY', 'PIPE_TEXTURE_3D', 'PIPE_TEXTURE_CUBE',
    'PIPE_TEXTURE_CUBE_ARRAY', 'PIPE_TEXTURE_RECT',
    'RALLOC_PRINT_INFO_SUMMARY_ONLY', 'SCOPE_DEVICE',
    'SCOPE_INVOCATION', 'SCOPE_NONE', 'SCOPE_QUEUE_FAMILY',
    'SCOPE_SHADER_CALL', 'SCOPE_SUBGROUP', 'SCOPE_WORKGROUP',
    'SUBGROUP_SIZE_API_CONSTANT', 'SUBGROUP_SIZE_FULL_SUBGROUPS',
    'SUBGROUP_SIZE_REQUIRE_128', 'SUBGROUP_SIZE_REQUIRE_16',
    'SUBGROUP_SIZE_REQUIRE_32', 'SUBGROUP_SIZE_REQUIRE_4',
    'SUBGROUP_SIZE_REQUIRE_64', 'SUBGROUP_SIZE_REQUIRE_8',
    'SUBGROUP_SIZE_UNIFORM', 'SUBGROUP_SIZE_VARYING',
    'SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID',
    'SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD',
    'SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL',
    'SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE',
    'SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL',
    'SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID',
    'SYSTEM_VALUE_BASE_INSTANCE', 'SYSTEM_VALUE_BASE_VERTEX',
    'SYSTEM_VALUE_BASE_WORKGROUP_ID',
    'SYSTEM_VALUE_COALESCED_INPUT_COUNT', 'SYSTEM_VALUE_COLOR0',
    'SYSTEM_VALUE_COLOR1', 'SYSTEM_VALUE_CULL_MASK',
    'SYSTEM_VALUE_DEVICE_INDEX', 'SYSTEM_VALUE_DRAW_ID',
    'SYSTEM_VALUE_FIRST_VERTEX', 'SYSTEM_VALUE_FRAG_COORD',
    'SYSTEM_VALUE_FRAG_COORD_W', 'SYSTEM_VALUE_FRAG_COORD_Z',
    'SYSTEM_VALUE_FRAG_INVOCATION_COUNT',
    'SYSTEM_VALUE_FRAG_SHADING_RATE', 'SYSTEM_VALUE_FRAG_SIZE',
    'SYSTEM_VALUE_FRONT_FACE', 'SYSTEM_VALUE_FRONT_FACE_FSIGN',
    'SYSTEM_VALUE_FULLY_COVERED', 'SYSTEM_VALUE_GLOBAL_GROUP_SIZE',
    'SYSTEM_VALUE_GLOBAL_INVOCATION_ID',
    'SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX',
    'SYSTEM_VALUE_GS_HEADER_IR3', 'SYSTEM_VALUE_HELPER_INVOCATION',
    'SYSTEM_VALUE_INSTANCE_ID', 'SYSTEM_VALUE_INSTANCE_INDEX',
    'SYSTEM_VALUE_INVOCATION_ID', 'SYSTEM_VALUE_IS_INDEXED_DRAW',
    'SYSTEM_VALUE_LAYER_ID', 'SYSTEM_VALUE_LINE_COORD',
    'SYSTEM_VALUE_LOCAL_INVOCATION_ID',
    'SYSTEM_VALUE_LOCAL_INVOCATION_INDEX', 'SYSTEM_VALUE_MAX',
    'SYSTEM_VALUE_MESH_VIEW_COUNT', 'SYSTEM_VALUE_MESH_VIEW_INDICES',
    'SYSTEM_VALUE_NUM_SUBGROUPS', 'SYSTEM_VALUE_NUM_WORKGROUPS',
    'SYSTEM_VALUE_PIXEL_COORD', 'SYSTEM_VALUE_POINT_COORD',
    'SYSTEM_VALUE_PRIMITIVE_ID', 'SYSTEM_VALUE_RAY_FLAGS',
    'SYSTEM_VALUE_RAY_GEOMETRY_INDEX', 'SYSTEM_VALUE_RAY_HIT_KIND',
    'SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX',
    'SYSTEM_VALUE_RAY_LAUNCH_ID', 'SYSTEM_VALUE_RAY_LAUNCH_SIZE',
    'SYSTEM_VALUE_RAY_OBJECT_DIRECTION',
    'SYSTEM_VALUE_RAY_OBJECT_ORIGIN',
    'SYSTEM_VALUE_RAY_OBJECT_TO_WORLD',
    'SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS',
    'SYSTEM_VALUE_RAY_T_MAX', 'SYSTEM_VALUE_RAY_T_MIN',
    'SYSTEM_VALUE_RAY_WORLD_DIRECTION',
    'SYSTEM_VALUE_RAY_WORLD_ORIGIN',
    'SYSTEM_VALUE_RAY_WORLD_TO_OBJECT',
    'SYSTEM_VALUE_REL_PATCH_ID_IR3', 'SYSTEM_VALUE_SAMPLE_ID',
    'SYSTEM_VALUE_SAMPLE_MASK_IN', 'SYSTEM_VALUE_SAMPLE_POS',
    'SYSTEM_VALUE_SAMPLE_POS_OR_CENTER', 'SYSTEM_VALUE_SHADER_INDEX',
    'SYSTEM_VALUE_SM_COUNT_NV', 'SYSTEM_VALUE_SM_ID_NV',
    'SYSTEM_VALUE_SUBGROUP_EQ_MASK', 'SYSTEM_VALUE_SUBGROUP_GE_MASK',
    'SYSTEM_VALUE_SUBGROUP_GT_MASK', 'SYSTEM_VALUE_SUBGROUP_ID',
    'SYSTEM_VALUE_SUBGROUP_INVOCATION',
    'SYSTEM_VALUE_SUBGROUP_LE_MASK', 'SYSTEM_VALUE_SUBGROUP_LT_MASK',
    'SYSTEM_VALUE_SUBGROUP_SIZE', 'SYSTEM_VALUE_TCS_HEADER_IR3',
    'SYSTEM_VALUE_TESS_COORD', 'SYSTEM_VALUE_TESS_LEVEL_INNER',
    'SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT',
    'SYSTEM_VALUE_TESS_LEVEL_OUTER',
    'SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT',
    'SYSTEM_VALUE_USER_DATA_AMD', 'SYSTEM_VALUE_VERTEX_CNT',
    'SYSTEM_VALUE_VERTEX_ID', 'SYSTEM_VALUE_VERTEX_ID_ZERO_BASE',
    'SYSTEM_VALUE_VERTICES_IN', 'SYSTEM_VALUE_VIEW_INDEX',
    'SYSTEM_VALUE_WARPS_PER_SM_NV', 'SYSTEM_VALUE_WARP_ID_NV',
    'SYSTEM_VALUE_WORKGROUP_ID', 'SYSTEM_VALUE_WORKGROUP_INDEX',
    'SYSTEM_VALUE_WORKGROUP_SIZE', 'SYSTEM_VALUE_WORK_DIM',
    'TESS_PRIMITIVE_ISOLINES', 'TESS_PRIMITIVE_QUADS',
    'TESS_PRIMITIVE_TRIANGLES', 'TESS_PRIMITIVE_UNSPECIFIED',
    'TGSI_TEXTURE_1D', 'TGSI_TEXTURE_1D_ARRAY', 'TGSI_TEXTURE_2D',
    'TGSI_TEXTURE_2D_ARRAY', 'TGSI_TEXTURE_2D_ARRAY_MSAA',
    'TGSI_TEXTURE_2D_MSAA', 'TGSI_TEXTURE_3D', 'TGSI_TEXTURE_BUFFER',
    'TGSI_TEXTURE_COUNT', 'TGSI_TEXTURE_CUBE',
    'TGSI_TEXTURE_CUBE_ARRAY', 'TGSI_TEXTURE_RECT',
    'TGSI_TEXTURE_SHADOW1D', 'TGSI_TEXTURE_SHADOW1D_ARRAY',
    'TGSI_TEXTURE_SHADOW2D', 'TGSI_TEXTURE_SHADOW2D_ARRAY',
    'TGSI_TEXTURE_SHADOWCUBE', 'TGSI_TEXTURE_SHADOWCUBE_ARRAY',
    'TGSI_TEXTURE_SHADOWRECT', 'TGSI_TEXTURE_UNKNOWN',
    'UTIL_FORMAT_COLORSPACE_RGB', 'UTIL_FORMAT_COLORSPACE_SRGB',
    'UTIL_FORMAT_COLORSPACE_YUV', 'UTIL_FORMAT_COLORSPACE_ZS',
    'UTIL_FORMAT_LAYOUT_ASTC', 'UTIL_FORMAT_LAYOUT_ATC',
    'UTIL_FORMAT_LAYOUT_BPTC', 'UTIL_FORMAT_LAYOUT_ETC',
    'UTIL_FORMAT_LAYOUT_FXT1', 'UTIL_FORMAT_LAYOUT_OTHER',
    'UTIL_FORMAT_LAYOUT_PLAIN', 'UTIL_FORMAT_LAYOUT_PLANAR2',
    'UTIL_FORMAT_LAYOUT_PLANAR3', 'UTIL_FORMAT_LAYOUT_RGTC',
    'UTIL_FORMAT_LAYOUT_S3TC', 'UTIL_FORMAT_LAYOUT_SUBSAMPLED',
    'VARYING_SLOT_BFC0', 'VARYING_SLOT_BFC1',
    'VARYING_SLOT_BOUNDING_BOX0', 'VARYING_SLOT_BOUNDING_BOX1',
    'VARYING_SLOT_CLIP_DIST0', 'VARYING_SLOT_CLIP_DIST1',
    'VARYING_SLOT_CLIP_VERTEX', 'VARYING_SLOT_COL0',
    'VARYING_SLOT_COL1', 'VARYING_SLOT_CULL_DIST0',
    'VARYING_SLOT_CULL_DIST1', 'VARYING_SLOT_CULL_PRIMITIVE',
    'VARYING_SLOT_EDGE', 'VARYING_SLOT_FACE', 'VARYING_SLOT_FOGC',
    'VARYING_SLOT_LAYER', 'VARYING_SLOT_PATCH0',
    'VARYING_SLOT_PATCH1', 'VARYING_SLOT_PATCH10',
    'VARYING_SLOT_PATCH11', 'VARYING_SLOT_PATCH12',
    'VARYING_SLOT_PATCH13', 'VARYING_SLOT_PATCH14',
    'VARYING_SLOT_PATCH15', 'VARYING_SLOT_PATCH16',
    'VARYING_SLOT_PATCH17', 'VARYING_SLOT_PATCH18',
    'VARYING_SLOT_PATCH19', 'VARYING_SLOT_PATCH2',
    'VARYING_SLOT_PATCH20', 'VARYING_SLOT_PATCH21',
    'VARYING_SLOT_PATCH22', 'VARYING_SLOT_PATCH23',
    'VARYING_SLOT_PATCH24', 'VARYING_SLOT_PATCH25',
    'VARYING_SLOT_PATCH26', 'VARYING_SLOT_PATCH27',
    'VARYING_SLOT_PATCH28', 'VARYING_SLOT_PATCH29',
    'VARYING_SLOT_PATCH3', 'VARYING_SLOT_PATCH30',
    'VARYING_SLOT_PATCH31', 'VARYING_SLOT_PATCH4',
    'VARYING_SLOT_PATCH5', 'VARYING_SLOT_PATCH6',
    'VARYING_SLOT_PATCH7', 'VARYING_SLOT_PATCH8',
    'VARYING_SLOT_PATCH9', 'VARYING_SLOT_PNTC', 'VARYING_SLOT_POS',
    'VARYING_SLOT_PRIMITIVE_COUNT', 'VARYING_SLOT_PRIMITIVE_ID',
    'VARYING_SLOT_PRIMITIVE_INDICES',
    'VARYING_SLOT_PRIMITIVE_SHADING_RATE', 'VARYING_SLOT_PSIZ',
    'VARYING_SLOT_TASK_COUNT', 'VARYING_SLOT_TESS_LEVEL_INNER',
    'VARYING_SLOT_TESS_LEVEL_OUTER', 'VARYING_SLOT_TEX0',
    'VARYING_SLOT_TEX1', 'VARYING_SLOT_TEX2', 'VARYING_SLOT_TEX3',
    'VARYING_SLOT_TEX4', 'VARYING_SLOT_TEX5', 'VARYING_SLOT_TEX6',
    'VARYING_SLOT_TEX7', 'VARYING_SLOT_VAR0',
    'VARYING_SLOT_VAR0_16BIT', 'VARYING_SLOT_VAR1',
    'VARYING_SLOT_VAR10', 'VARYING_SLOT_VAR10_16BIT',
    'VARYING_SLOT_VAR11', 'VARYING_SLOT_VAR11_16BIT',
    'VARYING_SLOT_VAR12', 'VARYING_SLOT_VAR12_16BIT',
    'VARYING_SLOT_VAR13', 'VARYING_SLOT_VAR13_16BIT',
    'VARYING_SLOT_VAR14', 'VARYING_SLOT_VAR14_16BIT',
    'VARYING_SLOT_VAR15', 'VARYING_SLOT_VAR15_16BIT',
    'VARYING_SLOT_VAR16', 'VARYING_SLOT_VAR17', 'VARYING_SLOT_VAR18',
    'VARYING_SLOT_VAR19', 'VARYING_SLOT_VAR1_16BIT',
    'VARYING_SLOT_VAR2', 'VARYING_SLOT_VAR20', 'VARYING_SLOT_VAR21',
    'VARYING_SLOT_VAR22', 'VARYING_SLOT_VAR23', 'VARYING_SLOT_VAR24',
    'VARYING_SLOT_VAR25', 'VARYING_SLOT_VAR26', 'VARYING_SLOT_VAR27',
    'VARYING_SLOT_VAR28', 'VARYING_SLOT_VAR29',
    'VARYING_SLOT_VAR2_16BIT', 'VARYING_SLOT_VAR3',
    'VARYING_SLOT_VAR30', 'VARYING_SLOT_VAR31',
    'VARYING_SLOT_VAR3_16BIT', 'VARYING_SLOT_VAR4',
    'VARYING_SLOT_VAR4_16BIT', 'VARYING_SLOT_VAR5',
    'VARYING_SLOT_VAR5_16BIT', 'VARYING_SLOT_VAR6',
    'VARYING_SLOT_VAR6_16BIT', 'VARYING_SLOT_VAR7',
    'VARYING_SLOT_VAR7_16BIT', 'VARYING_SLOT_VAR8',
    'VARYING_SLOT_VAR8_16BIT', 'VARYING_SLOT_VAR9',
    'VARYING_SLOT_VAR9_16BIT', 'VARYING_SLOT_VIEWPORT',
    'VARYING_SLOT_VIEWPORT_MASK', 'VARYING_SLOT_VIEW_INDEX',
    '_nir_mul_imm', '_nir_select_from_array_helper',
    '_nir_shader_variable_has_mode', '_nir_src_set_parent',
    'blob_align', 'blob_copy_bytes', 'blob_finish',
    'blob_finish_get_buffer', 'blob_init', 'blob_init_fixed',
    'blob_overwrite_bytes', 'blob_overwrite_intptr',
    'blob_overwrite_uint32', 'blob_overwrite_uint8',
    'blob_read_bytes', 'blob_read_intptr', 'blob_read_string',
    'blob_read_uint16', 'blob_read_uint32', 'blob_read_uint64',
    'blob_read_uint8', 'blob_reader_align', 'blob_reader_init',
    'blob_reserve_bytes', 'blob_reserve_intptr',
    'blob_reserve_uint32', 'blob_skip_bytes', 'blob_write_bytes',
    'blob_write_intptr', 'blob_write_string', 'blob_write_uint16',
    'blob_write_uint32', 'blob_write_uint64', 'blob_write_uint8',
    'c__EA_LLVMAtomicRMWBinOp', 'c__EA_LLVMIntPredicate',
    'c__EA_LLVMTypeKind', 'c__EA_gl_system_value',
    'c__EA_gl_varying_slot', 'c__EA_mesa_scope',
    'c__EA_nir_address_format', 'c__EA_nir_alu_type',
    'c__EA_nir_atomic_op', 'c__EA_nir_cf_node_type',
    'c__EA_nir_cmat_signed', 'c__EA_nir_cursor_option',
    'c__EA_nir_depth_layout',
    'c__EA_nir_deref_instr_has_complex_use_options',
    'c__EA_nir_deref_type', 'c__EA_nir_divergence_options',
    'c__EA_nir_instr_type', 'c__EA_nir_intrinsic_index_flag',
    'c__EA_nir_intrinsic_op', 'c__EA_nir_intrinsic_semantic_flag',
    'c__EA_nir_io_options', 'c__EA_nir_jump_type',
    'c__EA_nir_load_grouping', 'c__EA_nir_loop_control',
    'c__EA_nir_lower_array_deref_of_vec_options',
    'c__EA_nir_lower_discard_if_options',
    'c__EA_nir_lower_doubles_options',
    'c__EA_nir_lower_fp16_cast_options',
    'c__EA_nir_lower_gs_intrinsics_flags',
    'c__EA_nir_lower_int64_options',
    'c__EA_nir_lower_interpolation_options',
    'c__EA_nir_lower_io_options', 'c__EA_nir_lower_packing_op',
    'c__EA_nir_mem_access_shift_method', 'c__EA_nir_memory_semantics',
    'c__EA_nir_metadata', 'c__EA_nir_move_options', 'c__EA_nir_op',
    'c__EA_nir_op_algebraic_property', 'c__EA_nir_opt_if_options',
    'c__EA_nir_opt_move_to_top_options',
    'c__EA_nir_opt_varyings_progress', 'c__EA_nir_preamble_class',
    'c__EA_nir_ray_query_value', 'c__EA_nir_resource_data_intel',
    'c__EA_nir_rounding_mode', 'c__EA_nir_selection_control',
    'c__EA_nir_var_declaration_type', 'c__EA_nir_variable_mode',
    'c__Ea_GLSL_PRECISION_NONE', 'c__Ea_LP_JIT_BUFFER_BASE',
    'c__Ea_LP_JIT_IMAGE_BASE', 'c__Ea_LP_JIT_RES_CONSTANTS',
    'c__Ea_LP_JIT_SAMPLER_MIN_LOD', 'c__Ea_LP_JIT_TEXTURE_BASE',
    'c__Ea_LP_JIT_VERTEX_HEADER_VERTEX_ID',
    'c__Ea_RALLOC_PRINT_INFO_SUMMARY_ONLY', 'c_bool', 'c_uint32',
    'c_uint64', 'compare_func', 'decode_type_from_blob',
    'encode_type_to_blob', 'func_pointer',
    'gallivm_add_global_mapping', 'gallivm_compile_module',
    'gallivm_create', 'gallivm_create_target_library_info',
    'gallivm_destroy', 'gallivm_dispose_target_library_info',
    'gallivm_free_ir', 'gallivm_get_perf_flags',
    'gallivm_jit_function', 'gallivm_stub_func',
    'gallivm_verify_function', 'gc_alloc_size', 'gc_context',
    'gc_ctx', 'gc_free', 'gc_get_context', 'gc_mark_live',
    'gc_sweep_end', 'gc_sweep_start', 'gc_zalloc_size',
    'gl_access_qualifier', 'gl_derivative_group', 'gl_shader_stage',
    'gl_shader_stage__enumvalues', 'gl_subgroup_size',
    'gl_system_value', 'gl_system_value__enumvalues',
    'gl_varying_slot', 'gl_varying_slot__enumvalues',
    'glsl_apply_signedness_to_base_type', 'glsl_array_size',
    'glsl_array_type', 'glsl_atomic_size', 'glsl_atomic_uint_type',
    'glsl_bare_sampler_type', 'glsl_bare_shadow_sampler_type',
    'glsl_base_type', 'glsl_base_type_bit_size',
    'glsl_base_type_get_bit_size', 'glsl_base_type_is_16bit',
    'glsl_base_type_is_64bit', 'glsl_base_type_is_float',
    'glsl_base_type_is_integer', 'glsl_bf16vec_type',
    'glsl_bfloat16_t_type', 'glsl_bfloatN_t_type', 'glsl_bool_type',
    'glsl_bvec2_type', 'glsl_bvec4_type', 'glsl_bvec_type',
    'glsl_channel_type', 'glsl_cmat_type', 'glsl_cmat_use',
    'glsl_contains_array', 'glsl_contains_atomic',
    'glsl_contains_double', 'glsl_contains_integer',
    'glsl_contains_opaque', 'glsl_contains_sampler',
    'glsl_contains_subroutine', 'glsl_count_attribute_slots',
    'glsl_count_dword_slots', 'glsl_count_vec4_slots',
    'glsl_double_type', 'glsl_dvec2_type', 'glsl_dvec4_type',
    'glsl_dvec_type', 'glsl_e4m3fn_t_type', 'glsl_e4m3fnvec_type',
    'glsl_e5m2_t_type', 'glsl_e5m2vec_type',
    'glsl_explicit_matrix_type', 'glsl_f16vec_type',
    'glsl_float16_t_type', 'glsl_float16_type', 'glsl_floatN_t_type',
    'glsl_float_type', 'glsl_get_aoa_size', 'glsl_get_array_element',
    'glsl_get_bare_type', 'glsl_get_base_glsl_type',
    'glsl_get_base_type', 'glsl_get_bit_size',
    'glsl_get_cl_alignment', 'glsl_get_cl_size',
    'glsl_get_cl_type_size_align', 'glsl_get_cmat_description',
    'glsl_get_cmat_element', 'glsl_get_column_type',
    'glsl_get_component_slots', 'glsl_get_component_slots_aligned',
    'glsl_get_components', 'glsl_get_explicit_alignment',
    'glsl_get_explicit_interface_type', 'glsl_get_explicit_size',
    'glsl_get_explicit_std140_type', 'glsl_get_explicit_std430_type',
    'glsl_get_explicit_stride',
    'glsl_get_explicit_type_for_size_align', 'glsl_get_field_index',
    'glsl_get_field_type', 'glsl_get_ifc_packing',
    'glsl_get_internal_ifc_packing', 'glsl_get_length',
    'glsl_get_matrix_columns', 'glsl_get_mul_type',
    'glsl_get_natural_size_align_bytes', 'glsl_get_row_type',
    'glsl_get_sampler_coordinate_components', 'glsl_get_sampler_dim',
    'glsl_get_sampler_dim_coordinate_components',
    'glsl_get_sampler_result_type', 'glsl_get_scalar_type',
    'glsl_get_std140_base_alignment', 'glsl_get_std140_size',
    'glsl_get_std430_array_stride', 'glsl_get_std430_base_alignment',
    'glsl_get_std430_size', 'glsl_get_struct_elem_name',
    'glsl_get_struct_field', 'glsl_get_struct_field_data',
    'glsl_get_struct_field_offset', 'glsl_get_struct_location_offset',
    'glsl_get_type_name', 'glsl_get_vec4_size_align_bytes',
    'glsl_get_vector_elements', 'glsl_get_word_size_align_bytes',
    'glsl_i16vec_type', 'glsl_i64vec_type', 'glsl_i8vec_type',
    'glsl_image_type', 'glsl_int16_t_type', 'glsl_int16_type',
    'glsl_int64_t_type', 'glsl_int8_t_type', 'glsl_intN_t_type',
    'glsl_int_type', 'glsl_interface_packing', 'glsl_interface_type',
    'glsl_ivec2_type', 'glsl_ivec4_type', 'glsl_ivec_type',
    'glsl_matrix_layout', 'glsl_matrix_type',
    'glsl_matrix_type_is_row_major', 'glsl_record_compare',
    'glsl_replace_vector_type', 'glsl_sampler_dim',
    'glsl_sampler_type', 'glsl_sampler_type_is_array',
    'glsl_sampler_type_is_shadow', 'glsl_sampler_type_to_texture',
    'glsl_scalar_type', 'glsl_signed_base_type_of',
    'glsl_simple_explicit_type', 'glsl_simple_type',
    'glsl_size_align_handle_array_and_structs', 'glsl_struct_field',
    'glsl_struct_type', 'glsl_struct_type_is_packed',
    'glsl_struct_type_with_explicit_alignment',
    'glsl_subroutine_type', 'glsl_texture_type',
    'glsl_texture_type_to_sampler', 'glsl_transposed_type',
    'glsl_type', 'glsl_type_compare_no_precision',
    'glsl_type_contains_32bit', 'glsl_type_contains_64bit',
    'glsl_type_contains_image', 'glsl_type_get_image_count',
    'glsl_type_get_sampler_count', 'glsl_type_get_texture_count',
    'glsl_type_is_16bit', 'glsl_type_is_32bit', 'glsl_type_is_64bit',
    'glsl_type_is_array', 'glsl_type_is_array_of_arrays',
    'glsl_type_is_array_or_matrix', 'glsl_type_is_atomic_uint',
    'glsl_type_is_bare_sampler', 'glsl_type_is_bfloat_16',
    'glsl_type_is_boolean', 'glsl_type_is_cmat',
    'glsl_type_is_double', 'glsl_type_is_dual_slot',
    'glsl_type_is_e4m3fn', 'glsl_type_is_e5m2', 'glsl_type_is_error',
    'glsl_type_is_float', 'glsl_type_is_float_16',
    'glsl_type_is_float_16_32', 'glsl_type_is_float_16_32_64',
    'glsl_type_is_image', 'glsl_type_is_int_16_32',
    'glsl_type_is_int_16_32_64', 'glsl_type_is_integer',
    'glsl_type_is_integer_16', 'glsl_type_is_integer_16_32',
    'glsl_type_is_integer_16_32_64', 'glsl_type_is_integer_32',
    'glsl_type_is_integer_32_64', 'glsl_type_is_integer_64',
    'glsl_type_is_interface', 'glsl_type_is_leaf',
    'glsl_type_is_matrix', 'glsl_type_is_numeric',
    'glsl_type_is_packed', 'glsl_type_is_sampler',
    'glsl_type_is_scalar', 'glsl_type_is_struct',
    'glsl_type_is_struct_or_ifc', 'glsl_type_is_subroutine',
    'glsl_type_is_texture', 'glsl_type_is_uint_16_32',
    'glsl_type_is_uint_16_32_64', 'glsl_type_is_unsized_array',
    'glsl_type_is_vector', 'glsl_type_is_vector_or_scalar',
    'glsl_type_is_void', 'glsl_type_replace_vec3_with_vec4',
    'glsl_type_singleton_decref', 'glsl_type_singleton_init_or_ref',
    'glsl_type_size_align_func', 'glsl_type_to_16bit',
    'glsl_type_uniform_locations', 'glsl_type_wrap_in_arrays',
    'glsl_u16vec_type', 'glsl_u64vec_type', 'glsl_u8vec_type',
    'glsl_uint16_t_type', 'glsl_uint16_type', 'glsl_uint64_t_type',
    'glsl_uint8_t_type', 'glsl_uintN_t_type', 'glsl_uint_type',
    'glsl_unsigned_base_type_of', 'glsl_uvec2_type',
    'glsl_uvec4_type', 'glsl_uvec_type', 'glsl_varying_count',
    'glsl_vec2_type', 'glsl_vec4_type', 'glsl_vec_type',
    'glsl_vector_type', 'glsl_void_type', 'glsl_without_array',
    'glsl_without_array_or_matrix', 'int64_t', 'intptr_t',
    'linear_alloc_child', 'linear_alloc_child_array',
    'linear_asprintf', 'linear_asprintf_append',
    'linear_asprintf_rewrite_tail', 'linear_context',
    'linear_context_with_opts', 'linear_ctx', 'linear_free_context',
    'linear_opts', 'linear_strcat', 'linear_strdup',
    'linear_vasprintf', 'linear_vasprintf_append',
    'linear_vasprintf_rewrite_tail', 'linear_zalloc_child',
    'linear_zalloc_child_array', 'lp_bld_init_native_targets',
    'lp_bld_ppc_disable_denorms', 'lp_build_alloca',
    'lp_build_alloca_undef', 'lp_build_array_alloca',
    'lp_build_array_get2', 'lp_build_array_get_ptr2',
    'lp_build_const_aos', 'lp_build_const_channel_vec',
    'lp_build_const_double', 'lp_build_const_elem',
    'lp_build_const_float', 'lp_build_const_func_pointer',
    'lp_build_const_func_pointer_from_type', 'lp_build_const_int32',
    'lp_build_const_int64', 'lp_build_const_int_pointer',
    'lp_build_const_int_vec', 'lp_build_const_mask_aos',
    'lp_build_const_mask_aos_swizzled', 'lp_build_const_string',
    'lp_build_const_vec', 'lp_build_context_init',
    'lp_build_count_ir_module',
    'lp_build_create_jit_compiler_for_module',
    'lp_build_create_jit_vertex_header_type',
    'lp_build_cs_func_call_context', 'lp_build_elem_type',
    'lp_build_else', 'lp_build_endif', 'lp_build_flow_skip_begin',
    'lp_build_flow_skip_cond_break', 'lp_build_flow_skip_end',
    'lp_build_for_loop_begin', 'lp_build_for_loop_end', 'lp_build_if',
    'lp_build_image_function_type', 'lp_build_init',
    'lp_build_init_native_width', 'lp_build_insert_new_block',
    'lp_build_int_elem_type', 'lp_build_int_vec_type',
    'lp_build_jit_fill_image_dynamic_state',
    'lp_build_jit_fill_sampler_dynamic_state',
    'lp_build_jit_resources_type', 'lp_build_loop_begin',
    'lp_build_loop_end', 'lp_build_loop_end_cond',
    'lp_build_loop_force_reload_counter',
    'lp_build_loop_force_set_counter', 'lp_build_mask_begin',
    'lp_build_mask_check', 'lp_build_mask_end', 'lp_build_mask_force',
    'lp_build_mask_update', 'lp_build_mask_value', 'lp_build_nir_aos',
    'lp_build_nir_sample_key', 'lp_build_nir_soa',
    'lp_build_nir_soa_func', 'lp_build_nir_soa_prepasses',
    'lp_build_one', 'lp_build_opt_nir', 'lp_build_pointer_get2',
    'lp_build_pointer_get_unaligned2', 'lp_build_pointer_set',
    'lp_build_pointer_set_unaligned', 'lp_build_sample_function_type',
    'lp_build_size_function_type', 'lp_build_struct_get2',
    'lp_build_struct_get_ptr2', 'lp_build_tex_modifier',
    'lp_build_undef', 'lp_build_vec_type', 'lp_build_zero',
    'lp_check_elem_type', 'lp_check_value', 'lp_check_vec_type',
    'lp_const_eps', 'lp_const_max', 'lp_const_min', 'lp_const_offset',
    'lp_const_scale', 'lp_const_shift', 'lp_context_create',
    'lp_context_destroy', 'lp_context_ref',
    'lp_create_builder_at_entry', 'lp_dump_llvmtype', 'lp_elem_type',
    'lp_float32_vec4_type', 'lp_free_generated_code',
    'lp_free_memory_manager', 'lp_free_objcache',
    'lp_get_called_value', 'lp_get_default_memory_manager',
    'lp_img_op_from_intrinsic', 'lp_init_clock_hook',
    'lp_init_env_options', 'lp_int32_vec4_type', 'lp_int_type',
    'lp_is_function', 'lp_llvm_buffer_base',
    'lp_llvm_buffer_num_elements', 'lp_llvm_descriptor_base',
    'lp_mantissa', 'lp_native_vector_width',
    'lp_nir_array_build_gather_values', 'lp_nir_call_context_args',
    'lp_packed_img_op_from_intrinsic', 'lp_passmgr_create',
    'lp_passmgr_dispose', 'lp_passmgr_run', 'lp_sampler_lod_property',
    'lp_set_module_stack_alignment_override', 'lp_set_target_options',
    'lp_sizeof_llvm_type', 'lp_translate_atomic_op', 'lp_type_fixed',
    'lp_type_float', 'lp_type_float_vec', 'lp_type_from_format',
    'lp_type_from_format_desc', 'lp_type_int', 'lp_type_int_vec',
    'lp_type_ufixed', 'lp_type_uint', 'lp_type_uint_vec',
    'lp_type_unorm', 'lp_type_width', 'lp_typekind_name',
    'lp_uint_type', 'lp_unorm8_vec4_type', 'lp_wider_type',
    'mesa_log_level', 'mesa_prim', 'mesa_scope',
    'mesa_scope__enumvalues', 'nak_compile_shader',
    'nak_compiler_create', 'nak_compiler_destroy', 'nak_debug_flags',
    'nak_fill_qmd', 'nak_get_qmd_cbuf_desc_layout',
    'nak_get_qmd_dispatch_size_layout', 'nak_nir_lower_image_addrs',
    'nak_nir_options', 'nak_postprocess_nir', 'nak_preprocess_nir',
    'nak_qmd_size_B', 'nak_shader_bin_destroy', 'nak_ts_domain',
    'nak_ts_prims', 'nak_ts_spacing', 'nir_a_minus_bc',
    'nir_add_inlinable_uniforms', 'nir_addition_might_overflow',
    'nir_address_format', 'nir_address_format_2x32bit_global',
    'nir_address_format_32bit_global',
    'nir_address_format_32bit_index_offset',
    'nir_address_format_32bit_index_offset_pack64',
    'nir_address_format_32bit_offset',
    'nir_address_format_32bit_offset_as_64bit',
    'nir_address_format_62bit_generic',
    'nir_address_format_64bit_bounded_global',
    'nir_address_format_64bit_global',
    'nir_address_format_64bit_global_32bit_offset',
    'nir_address_format__enumvalues', 'nir_address_format_bit_size',
    'nir_address_format_logical', 'nir_address_format_null_value',
    'nir_address_format_num_components',
    'nir_address_format_to_glsl_type',
    'nir_address_format_vec2_index_32bit_offset', 'nir_after_block',
    'nir_after_block_before_jump', 'nir_after_cf_list',
    'nir_after_cf_node', 'nir_after_cf_node_and_phis',
    'nir_after_impl', 'nir_after_instr', 'nir_after_instr_and_phis',
    'nir_after_phis', 'nir_after_reg_decls', 'nir_align_imm',
    'nir_alignment_deref_cast', 'nir_alu_binop_identity',
    'nir_alu_instr', 'nir_alu_instr_channel_used',
    'nir_alu_instr_clone', 'nir_alu_instr_create',
    'nir_alu_instr_is_comparison', 'nir_alu_instr_is_inf_preserve',
    'nir_alu_instr_is_nan_preserve',
    'nir_alu_instr_is_signed_zero_inf_nan_preserve',
    'nir_alu_instr_is_signed_zero_preserve',
    'nir_alu_instr_src_read_mask', 'nir_alu_pass_cb', 'nir_alu_src',
    'nir_alu_src_as_uint', 'nir_alu_src_copy',
    'nir_alu_src_is_trivial_ssa', 'nir_alu_srcs_equal',
    'nir_alu_srcs_negative_equal',
    'nir_alu_srcs_negative_equal_typed', 'nir_alu_type',
    'nir_alu_type__enumvalues', 'nir_amul_imm',
    'nir_assign_io_var_locations', 'nir_atomic_op',
    'nir_atomic_op__enumvalues', 'nir_atomic_op_cmpxchg',
    'nir_atomic_op_dec_wrap', 'nir_atomic_op_fadd',
    'nir_atomic_op_fcmpxchg', 'nir_atomic_op_fmax',
    'nir_atomic_op_fmin', 'nir_atomic_op_iadd', 'nir_atomic_op_iand',
    'nir_atomic_op_imax', 'nir_atomic_op_imin',
    'nir_atomic_op_inc_wrap', 'nir_atomic_op_ior',
    'nir_atomic_op_ixor', 'nir_atomic_op_ordered_add_gfx12_amd',
    'nir_atomic_op_to_alu', 'nir_atomic_op_type',
    'nir_atomic_op_umax', 'nir_atomic_op_umin', 'nir_atomic_op_xchg',
    'nir_b2bN', 'nir_b2fN', 'nir_b2iN', 'nir_ball', 'nir_ball_iequal',
    'nir_bany', 'nir_bany_inequal', 'nir_before_block',
    'nir_before_block_after_phis', 'nir_before_cf_list',
    'nir_before_cf_node', 'nir_before_impl', 'nir_before_instr',
    'nir_before_src', 'nir_bfdot', 'nir_binding',
    'nir_bitcast_vector', 'nir_bitfield_insert_imm', 'nir_block',
    'nir_block_cf_tree_next', 'nir_block_cf_tree_prev',
    'nir_block_contains_work', 'nir_block_create',
    'nir_block_dominates', 'nir_block_ends_in_break',
    'nir_block_ends_in_jump', 'nir_block_ends_in_return_or_halt',
    'nir_block_first_instr', 'nir_block_get_following_if',
    'nir_block_get_following_loop',
    'nir_block_get_predecessors_sorted', 'nir_block_is_reachable',
    'nir_block_is_unreachable', 'nir_block_last_instr',
    'nir_block_last_phi_instr', 'nir_block_unstructured_next',
    'nir_break_if', 'nir_build_addr_iadd', 'nir_build_addr_iadd_imm',
    'nir_build_addr_ieq', 'nir_build_addr_isub', 'nir_build_alu',
    'nir_build_alu1', 'nir_build_alu2', 'nir_build_alu3',
    'nir_build_alu4', 'nir_build_alu_src_arr', 'nir_build_call',
    'nir_build_deref_array', 'nir_build_deref_array_imm',
    'nir_build_deref_array_wildcard', 'nir_build_deref_cast',
    'nir_build_deref_cast_with_alignment', 'nir_build_deref_follower',
    'nir_build_deref_ptr_as_array', 'nir_build_deref_struct',
    'nir_build_deref_var', 'nir_build_deriv', 'nir_build_imm',
    'nir_build_indirect_call',
    'nir_build_lowered_load_helper_invocation', 'nir_build_string',
    'nir_build_tex_deref_instr', 'nir_build_write_masked_store',
    'nir_build_write_masked_stores', 'nir_builder',
    'nir_builder_alu_instr_finish_and_insert', 'nir_builder_at',
    'nir_builder_cf_insert', 'nir_builder_create',
    'nir_builder_init_simple_shader', 'nir_builder_instr_insert',
    'nir_builder_instr_insert_at_top', 'nir_builder_is_inside_cf',
    'nir_builder_last_instr', 'nir_calc_dominance',
    'nir_calc_dominance_impl', 'nir_calc_use_dominance_impl',
    'nir_call_instr', 'nir_call_instr_create', 'nir_call_serialized',
    'nir_can_lower_multiview', 'nir_can_move_instr',
    'nir_cf_list_is_empty_block', 'nir_cf_node',
    'nir_cf_node_as_block', 'nir_cf_node_as_function',
    'nir_cf_node_as_if', 'nir_cf_node_as_loop', 'nir_cf_node_block',
    'nir_cf_node_cf_tree_first', 'nir_cf_node_cf_tree_last',
    'nir_cf_node_cf_tree_next', 'nir_cf_node_cf_tree_prev',
    'nir_cf_node_function', 'nir_cf_node_get_function',
    'nir_cf_node_if', 'nir_cf_node_is_first', 'nir_cf_node_is_last',
    'nir_cf_node_loop', 'nir_cf_node_next', 'nir_cf_node_prev',
    'nir_cf_node_type', 'nir_cf_node_type__enumvalues', 'nir_channel',
    'nir_channel_or_undef', 'nir_channels', 'nir_chase_binding',
    'nir_cleanup_functions', 'nir_clear_mediump_io_flag',
    'nir_clear_shared_memory', 'nir_clone_deref_instr',
    'nir_clone_uniform_variable', 'nir_cmat_signed',
    'nir_cmat_signed__enumvalues', 'nir_collect_src_uniforms',
    'nir_combine_barrier_cb', 'nir_combined_align',
    'nir_compact_varyings', 'nir_compare_func', 'nir_component_mask',
    'nir_component_mask_can_reinterpret',
    'nir_component_mask_reinterpret', 'nir_component_mask_t',
    'nir_const_value', 'nir_const_value_as_bool',
    'nir_const_value_as_float', 'nir_const_value_as_int',
    'nir_const_value_as_uint', 'nir_const_value_for_bool',
    'nir_const_value_for_float', 'nir_const_value_for_int',
    'nir_const_value_for_raw_uint', 'nir_const_value_for_uint',
    'nir_const_value_negative_equal', 'nir_constant',
    'nir_constant_clone', 'nir_convert_from_ssa',
    'nir_convert_loop_to_lcssa', 'nir_convert_to_bit_size',
    'nir_convert_to_lcssa', 'nir_copy_deref',
    'nir_copy_deref_with_access', 'nir_copy_prop',
    'nir_copy_prop_impl', 'nir_copy_var', 'nir_create_passthrough_gs',
    'nir_create_passthrough_tcs', 'nir_create_passthrough_tcs_impl',
    'nir_create_variable_with_location', 'nir_cursor',
    'nir_cursor_after_block', 'nir_cursor_after_instr',
    'nir_cursor_before_block', 'nir_cursor_before_instr',
    'nir_cursor_current_block', 'nir_cursor_option',
    'nir_cursor_option__enumvalues', 'nir_cursors_equal', 'nir_ddx',
    'nir_ddx_coarse', 'nir_ddx_fine', 'nir_ddy', 'nir_ddy_coarse',
    'nir_ddy_fine', 'nir_debug', 'nir_debug_print_shader',
    'nir_decl_reg', 'nir_dedup_inline_samplers', 'nir_def',
    'nir_def_all_uses_are_fsat', 'nir_def_all_uses_ignore_sign_bit',
    'nir_def_components_read', 'nir_def_first_component_read',
    'nir_def_init', 'nir_def_init_for_type',
    'nir_def_is_frag_coord_z', 'nir_def_is_unused',
    'nir_def_last_component_read', 'nir_def_only_used_by_if',
    'nir_def_replace', 'nir_def_rewrite_uses',
    'nir_def_rewrite_uses_after', 'nir_def_rewrite_uses_src',
    'nir_def_used_by_if', 'nir_defs_interfere', 'nir_depth_layout',
    'nir_depth_layout__enumvalues', 'nir_depth_layout_any',
    'nir_depth_layout_greater', 'nir_depth_layout_less',
    'nir_depth_layout_none', 'nir_depth_layout_unchanged',
    'nir_deref_cast_is_trivial', 'nir_deref_count_slots',
    'nir_deref_instr', 'nir_deref_instr_array_stride',
    'nir_deref_instr_create', 'nir_deref_instr_get_variable',
    'nir_deref_instr_has_complex_use',
    'nir_deref_instr_has_complex_use_allow_atomics',
    'nir_deref_instr_has_complex_use_allow_memcpy_dst',
    'nir_deref_instr_has_complex_use_allow_memcpy_src',
    'nir_deref_instr_has_complex_use_options',
    'nir_deref_instr_has_complex_use_options__enumvalues',
    'nir_deref_instr_has_indirect',
    'nir_deref_instr_is_known_out_of_bounds',
    'nir_deref_instr_parent', 'nir_deref_instr_remove_if_unused',
    'nir_deref_mode_is', 'nir_deref_mode_is_in_set',
    'nir_deref_mode_is_one_of', 'nir_deref_mode_may_be',
    'nir_deref_mode_must_be', 'nir_deref_type',
    'nir_deref_type__enumvalues', 'nir_deref_type_array',
    'nir_deref_type_array_wildcard', 'nir_deref_type_cast',
    'nir_deref_type_ptr_as_array', 'nir_deref_type_struct',
    'nir_deref_type_var', 'nir_deserialize',
    'nir_deserialize_function', 'nir_discard', 'nir_discard_if',
    'nir_divergence_analysis', 'nir_divergence_analysis_impl',
    'nir_divergence_ignore_undef_if_phi_srcs',
    'nir_divergence_multiple_workgroup_per_compute_subgroup',
    'nir_divergence_options', 'nir_divergence_options__enumvalues',
    'nir_divergence_shader_record_ptr_uniform',
    'nir_divergence_single_frag_shading_rate_per_subgroup',
    'nir_divergence_single_patch_per_tcs_subgroup',
    'nir_divergence_single_patch_per_tes_subgroup',
    'nir_divergence_single_prim_per_subgroup',
    'nir_divergence_uniform_load_tears',
    'nir_divergence_view_index_uniform', 'nir_dominance_lca',
    'nir_dont_move_byte_word_vecs', 'nir_dump_cfg',
    'nir_dump_cfg_impl', 'nir_dump_dom_frontier',
    'nir_dump_dom_frontier_impl', 'nir_dump_dom_tree',
    'nir_dump_dom_tree_impl', 'nir_explicit_io_address_from_deref',
    'nir_extract_bits', 'nir_extract_i8_imm', 'nir_extract_u8_imm',
    'nir_f2fN', 'nir_f2iN', 'nir_f2uN', 'nir_fadd_imm', 'nir_fclamp',
    'nir_fdiv_imm', 'nir_fdot', 'nir_ffma_imm1', 'nir_ffma_imm12',
    'nir_ffma_imm2', 'nir_fgt_imm', 'nir_find_inlinable_uniforms',
    'nir_find_sampler_variable_with_tex_index',
    'nir_find_state_variable',
    'nir_find_variable_with_driver_location',
    'nir_find_variable_with_location', 'nir_first_phi_in_block',
    'nir_fixup_deref_modes', 'nir_fixup_deref_types',
    'nir_fixup_is_exported', 'nir_fle_imm', 'nir_fmul_imm',
    'nir_foreach_def_cb', 'nir_foreach_function_with_impl_first',
    'nir_foreach_function_with_impl_next',
    'nir_foreach_phi_src_leaving_block', 'nir_foreach_src',
    'nir_foreach_src_cb', 'nir_fpow_imm',
    'nir_free_output_dependencies', 'nir_fsub_imm', 'nir_function',
    'nir_function_clone', 'nir_function_create', 'nir_function_impl',
    'nir_function_impl_add_variable', 'nir_function_impl_clone',
    'nir_function_impl_clone_remap_globals',
    'nir_function_impl_create', 'nir_function_impl_create_bare',
    'nir_function_impl_index_vars',
    'nir_function_impl_lower_instructions',
    'nir_function_instructions_pass', 'nir_function_intrinsics_pass',
    'nir_function_set_impl', 'nir_gather_explicit_io_initializers',
    'nir_gather_input_to_output_dependencies',
    'nir_gather_output_clipper_var_groups',
    'nir_gather_output_dependencies', 'nir_gather_types',
    'nir_gen_rect_vertices', 'nir_get_binding_variable',
    'nir_get_explicit_deref_align',
    'nir_get_glsl_base_type_for_nir_type',
    'nir_get_immediate_use_dominator', 'nir_get_io_arrayed_index_src',
    'nir_get_io_arrayed_index_src_number', 'nir_get_io_index_src',
    'nir_get_io_index_src_number', 'nir_get_io_intrinsic',
    'nir_get_io_offset_src', 'nir_get_io_offset_src_number',
    'nir_get_live_defs', 'nir_get_nir_type_for_glsl_base_type',
    'nir_get_nir_type_for_glsl_type', 'nir_get_ptr_bitsize',
    'nir_get_rounding_mode_from_float_controls', 'nir_get_scalar',
    'nir_get_shader_call_payload_src', 'nir_get_tex_deref',
    'nir_get_tex_src', 'nir_get_variable_with_location', 'nir_goto',
    'nir_goto_if', 'nir_group_all', 'nir_group_loads',
    'nir_group_same_resource_only',
    'nir_gs_count_vertices_and_primitives',
    'nir_has_any_rounding_mode_enabled',
    'nir_has_any_rounding_mode_rtne', 'nir_has_any_rounding_mode_rtz',
    'nir_has_divergent_loop', 'nir_has_non_uniform_access', 'nir_i2b',
    'nir_i2fN', 'nir_i2iN', 'nir_iadd_imm', 'nir_iadd_imm_nuw',
    'nir_iadd_nuw', 'nir_iand_imm', 'nir_ibfe_imm',
    'nir_ibitfield_extract_imm', 'nir_iclamp', 'nir_if',
    'nir_if_create', 'nir_if_first_else_block',
    'nir_if_first_then_block', 'nir_if_last_else_block',
    'nir_if_last_then_block', 'nir_if_phi',
    'nir_image_intrinsic_coord_components', 'nir_imax_imm',
    'nir_imin_imm', 'nir_imm_bool', 'nir_imm_boolN_t',
    'nir_imm_double', 'nir_imm_false', 'nir_imm_float',
    'nir_imm_float16', 'nir_imm_floatN_t', 'nir_imm_int',
    'nir_imm_int64', 'nir_imm_intN_t', 'nir_imm_ivec2',
    'nir_imm_ivec3', 'nir_imm_ivec3_intN', 'nir_imm_ivec4',
    'nir_imm_ivec4_intN', 'nir_imm_true', 'nir_imm_uvec2_intN',
    'nir_imm_uvec3_intN', 'nir_imm_vec2', 'nir_imm_vec3',
    'nir_imm_vec4', 'nir_imm_vec4_16', 'nir_imm_zero', 'nir_imod_imm',
    'nir_impl_last_block', 'nir_imul_imm', 'nir_index_blocks',
    'nir_index_instrs', 'nir_index_ssa_defs',
    'nir_inline_function_impl', 'nir_inline_functions',
    'nir_inline_uniforms', 'nir_input_attachment_options',
    'nir_input_to_output_deps', 'nir_instr', 'nir_instr_as_alu',
    'nir_instr_as_call', 'nir_instr_as_deref',
    'nir_instr_as_intrinsic', 'nir_instr_as_jump',
    'nir_instr_as_load_const', 'nir_instr_as_parallel_copy',
    'nir_instr_as_phi', 'nir_instr_as_str', 'nir_instr_as_tex',
    'nir_instr_as_undef', 'nir_instr_clear_src', 'nir_instr_clone',
    'nir_instr_clone_deep', 'nir_instr_debug_info', 'nir_instr_def',
    'nir_instr_dominates_use', 'nir_instr_filter_cb',
    'nir_instr_free', 'nir_instr_free_and_dce', 'nir_instr_free_list',
    'nir_instr_get_debug_info', 'nir_instr_get_gc_pointer',
    'nir_instr_init_src', 'nir_instr_insert',
    'nir_instr_insert_after', 'nir_instr_insert_after_block',
    'nir_instr_insert_after_cf', 'nir_instr_insert_after_cf_list',
    'nir_instr_insert_before', 'nir_instr_insert_before_block',
    'nir_instr_insert_before_cf', 'nir_instr_insert_before_cf_list',
    'nir_instr_is_before', 'nir_instr_is_first', 'nir_instr_is_last',
    'nir_instr_move', 'nir_instr_move_src', 'nir_instr_next',
    'nir_instr_pass_cb', 'nir_instr_prev', 'nir_instr_remove',
    'nir_instr_remove_v', 'nir_instr_type',
    'nir_instr_type__enumvalues', 'nir_instr_type_alu',
    'nir_instr_type_call', 'nir_instr_type_deref',
    'nir_instr_type_intrinsic', 'nir_instr_type_jump',
    'nir_instr_type_load_const', 'nir_instr_type_parallel_copy',
    'nir_instr_type_phi', 'nir_instr_type_tex',
    'nir_instr_type_undef', 'nir_instr_writemask_filter_cb',
    'nir_instr_xfb_write_mask', 'nir_instrs_equal',
    'nir_intrin_filter_cb', 'nir_intrinsic_accept_ray_intersection',
    'nir_intrinsic_addr_mode_is', 'nir_intrinsic_al2p_nv',
    'nir_intrinsic_ald_nv', 'nir_intrinsic_align',
    'nir_intrinsic_alpha_to_coverage', 'nir_intrinsic_as_uniform',
    'nir_intrinsic_ast_nv',
    'nir_intrinsic_atomic_add_gen_prim_count_amd',
    'nir_intrinsic_atomic_add_gs_emit_prim_count_amd',
    'nir_intrinsic_atomic_add_shader_invocation_count_amd',
    'nir_intrinsic_atomic_add_xfb_prim_count_amd',
    'nir_intrinsic_atomic_counter_add',
    'nir_intrinsic_atomic_counter_add_deref',
    'nir_intrinsic_atomic_counter_and',
    'nir_intrinsic_atomic_counter_and_deref',
    'nir_intrinsic_atomic_counter_comp_swap',
    'nir_intrinsic_atomic_counter_comp_swap_deref',
    'nir_intrinsic_atomic_counter_exchange',
    'nir_intrinsic_atomic_counter_exchange_deref',
    'nir_intrinsic_atomic_counter_inc',
    'nir_intrinsic_atomic_counter_inc_deref',
    'nir_intrinsic_atomic_counter_max',
    'nir_intrinsic_atomic_counter_max_deref',
    'nir_intrinsic_atomic_counter_min',
    'nir_intrinsic_atomic_counter_min_deref',
    'nir_intrinsic_atomic_counter_or',
    'nir_intrinsic_atomic_counter_or_deref',
    'nir_intrinsic_atomic_counter_post_dec',
    'nir_intrinsic_atomic_counter_post_dec_deref',
    'nir_intrinsic_atomic_counter_pre_dec',
    'nir_intrinsic_atomic_counter_pre_dec_deref',
    'nir_intrinsic_atomic_counter_read',
    'nir_intrinsic_atomic_counter_read_deref',
    'nir_intrinsic_atomic_counter_xor',
    'nir_intrinsic_atomic_counter_xor_deref', 'nir_intrinsic_ballot',
    'nir_intrinsic_ballot_bit_count_exclusive',
    'nir_intrinsic_ballot_bit_count_inclusive',
    'nir_intrinsic_ballot_bit_count_reduce',
    'nir_intrinsic_ballot_bitfield_extract',
    'nir_intrinsic_ballot_find_lsb', 'nir_intrinsic_ballot_find_msb',
    'nir_intrinsic_ballot_relaxed', 'nir_intrinsic_bar_break_nv',
    'nir_intrinsic_bar_set_nv', 'nir_intrinsic_bar_sync_nv',
    'nir_intrinsic_barrier',
    'nir_intrinsic_begin_invocation_interlock',
    'nir_intrinsic_bindgen_return',
    'nir_intrinsic_bindless_image_agx',
    'nir_intrinsic_bindless_image_atomic',
    'nir_intrinsic_bindless_image_atomic_swap',
    'nir_intrinsic_bindless_image_descriptor_amd',
    'nir_intrinsic_bindless_image_format',
    'nir_intrinsic_bindless_image_fragment_mask_load_amd',
    'nir_intrinsic_bindless_image_levels',
    'nir_intrinsic_bindless_image_load',
    'nir_intrinsic_bindless_image_load_raw_intel',
    'nir_intrinsic_bindless_image_order',
    'nir_intrinsic_bindless_image_samples',
    'nir_intrinsic_bindless_image_samples_identical',
    'nir_intrinsic_bindless_image_size',
    'nir_intrinsic_bindless_image_sparse_load',
    'nir_intrinsic_bindless_image_store',
    'nir_intrinsic_bindless_image_store_block_agx',
    'nir_intrinsic_bindless_image_store_raw_intel',
    'nir_intrinsic_bindless_image_texel_address',
    'nir_intrinsic_bindless_resource_ir3',
    'nir_intrinsic_brcst_active_ir3',
    'nir_intrinsic_btd_retire_intel', 'nir_intrinsic_btd_spawn_intel',
    'nir_intrinsic_btd_stack_push_intel',
    'nir_intrinsic_bvh64_intersect_ray_amd',
    'nir_intrinsic_bvh8_intersect_ray_amd',
    'nir_intrinsic_bvh_stack_rtn_amd', 'nir_intrinsic_can_reorder',
    'nir_intrinsic_cmat_binary_op', 'nir_intrinsic_cmat_bitcast',
    'nir_intrinsic_cmat_construct', 'nir_intrinsic_cmat_convert',
    'nir_intrinsic_cmat_copy', 'nir_intrinsic_cmat_extract',
    'nir_intrinsic_cmat_insert', 'nir_intrinsic_cmat_length',
    'nir_intrinsic_cmat_load', 'nir_intrinsic_cmat_muladd',
    'nir_intrinsic_cmat_muladd_amd', 'nir_intrinsic_cmat_muladd_nv',
    'nir_intrinsic_cmat_scalar_op', 'nir_intrinsic_cmat_store',
    'nir_intrinsic_cmat_transpose', 'nir_intrinsic_cmat_unary_op',
    'nir_intrinsic_convert_alu_types',
    'nir_intrinsic_convert_cmat_intel',
    'nir_intrinsic_copy_const_indices', 'nir_intrinsic_copy_deref',
    'nir_intrinsic_copy_fs_outputs_nv',
    'nir_intrinsic_copy_global_to_uniform_ir3',
    'nir_intrinsic_copy_push_const_to_uniform_ir3',
    'nir_intrinsic_copy_ubo_to_uniform_ir3', 'nir_intrinsic_ddx',
    'nir_intrinsic_ddx_coarse', 'nir_intrinsic_ddx_fine',
    'nir_intrinsic_ddy', 'nir_intrinsic_ddy_coarse',
    'nir_intrinsic_ddy_fine', 'nir_intrinsic_debug_break',
    'nir_intrinsic_decl_reg', 'nir_intrinsic_demote',
    'nir_intrinsic_demote_if', 'nir_intrinsic_demote_samples',
    'nir_intrinsic_deref_atomic', 'nir_intrinsic_deref_atomic_swap',
    'nir_intrinsic_deref_buffer_array_length',
    'nir_intrinsic_deref_implicit_array_length',
    'nir_intrinsic_deref_mode_is', 'nir_intrinsic_deref_texture_src',
    'nir_intrinsic_dest_components', 'nir_intrinsic_doorbell_agx',
    'nir_intrinsic_dpas_intel', 'nir_intrinsic_dpp16_shift_amd',
    'nir_intrinsic_elect', 'nir_intrinsic_elect_any_ir3',
    'nir_intrinsic_emit_primitive_poly', 'nir_intrinsic_emit_vertex',
    'nir_intrinsic_emit_vertex_nv',
    'nir_intrinsic_emit_vertex_with_counter',
    'nir_intrinsic_end_invocation_interlock',
    'nir_intrinsic_end_primitive', 'nir_intrinsic_end_primitive_nv',
    'nir_intrinsic_end_primitive_with_counter',
    'nir_intrinsic_enqueue_node_payloads',
    'nir_intrinsic_exclusive_scan',
    'nir_intrinsic_exclusive_scan_clusters_ir3',
    'nir_intrinsic_execute_callable',
    'nir_intrinsic_execute_closest_hit_amd',
    'nir_intrinsic_execute_miss_amd', 'nir_intrinsic_export_agx',
    'nir_intrinsic_export_amd',
    'nir_intrinsic_export_dual_src_blend_amd',
    'nir_intrinsic_export_row_amd',
    'nir_intrinsic_fence_helper_exit_agx',
    'nir_intrinsic_fence_mem_to_tex_agx',
    'nir_intrinsic_fence_pbe_to_tex_agx',
    'nir_intrinsic_fence_pbe_to_tex_pixel_agx',
    'nir_intrinsic_final_primitive_nv',
    'nir_intrinsic_finalize_incoming_node_payload',
    'nir_intrinsic_first_invocation',
    'nir_intrinsic_from_system_value', 'nir_intrinsic_fs_out_nv',
    'nir_intrinsic_gds_atomic_add_amd', 'nir_intrinsic_get_ssbo_size',
    'nir_intrinsic_get_ubo_size', 'nir_intrinsic_get_var',
    'nir_intrinsic_global_atomic', 'nir_intrinsic_global_atomic_2x32',
    'nir_intrinsic_global_atomic_agx',
    'nir_intrinsic_global_atomic_amd',
    'nir_intrinsic_global_atomic_swap',
    'nir_intrinsic_global_atomic_swap_2x32',
    'nir_intrinsic_global_atomic_swap_agx',
    'nir_intrinsic_global_atomic_swap_amd', 'nir_intrinsic_has_align',
    'nir_intrinsic_has_semantic',
    'nir_intrinsic_ignore_ray_intersection',
    'nir_intrinsic_imadsp_nv', 'nir_intrinsic_image_atomic',
    'nir_intrinsic_image_atomic_swap',
    'nir_intrinsic_image_deref_atomic',
    'nir_intrinsic_image_deref_atomic_swap',
    'nir_intrinsic_image_deref_descriptor_amd',
    'nir_intrinsic_image_deref_format',
    'nir_intrinsic_image_deref_fragment_mask_load_amd',
    'nir_intrinsic_image_deref_levels',
    'nir_intrinsic_image_deref_load',
    'nir_intrinsic_image_deref_load_info_nv',
    'nir_intrinsic_image_deref_load_param_intel',
    'nir_intrinsic_image_deref_load_raw_intel',
    'nir_intrinsic_image_deref_order',
    'nir_intrinsic_image_deref_samples',
    'nir_intrinsic_image_deref_samples_identical',
    'nir_intrinsic_image_deref_size',
    'nir_intrinsic_image_deref_sparse_load',
    'nir_intrinsic_image_deref_store',
    'nir_intrinsic_image_deref_store_block_agx',
    'nir_intrinsic_image_deref_store_raw_intel',
    'nir_intrinsic_image_deref_texel_address',
    'nir_intrinsic_image_descriptor_amd',
    'nir_intrinsic_image_format',
    'nir_intrinsic_image_fragment_mask_load_amd',
    'nir_intrinsic_image_levels', 'nir_intrinsic_image_load',
    'nir_intrinsic_image_load_raw_intel', 'nir_intrinsic_image_order',
    'nir_intrinsic_image_samples',
    'nir_intrinsic_image_samples_identical',
    'nir_intrinsic_image_size', 'nir_intrinsic_image_sparse_load',
    'nir_intrinsic_image_store',
    'nir_intrinsic_image_store_block_agx',
    'nir_intrinsic_image_store_raw_intel',
    'nir_intrinsic_image_texel_address',
    'nir_intrinsic_inclusive_scan',
    'nir_intrinsic_inclusive_scan_clusters_ir3',
    'nir_intrinsic_index_flag',
    'nir_intrinsic_index_flag__enumvalues',
    'nir_intrinsic_index_names', 'nir_intrinsic_info',
    'nir_intrinsic_infos', 'nir_intrinsic_initialize_node_payloads',
    'nir_intrinsic_instr', 'nir_intrinsic_instr_create',
    'nir_intrinsic_instr_dest_type', 'nir_intrinsic_instr_src_type',
    'nir_intrinsic_interp_deref_at_centroid',
    'nir_intrinsic_interp_deref_at_offset',
    'nir_intrinsic_interp_deref_at_sample',
    'nir_intrinsic_interp_deref_at_vertex',
    'nir_intrinsic_inverse_ballot', 'nir_intrinsic_ipa_nv',
    'nir_intrinsic_is_helper_invocation',
    'nir_intrinsic_is_ray_query',
    'nir_intrinsic_is_sparse_resident_zink',
    'nir_intrinsic_is_sparse_texels_resident',
    'nir_intrinsic_is_subgroup_invocation_lt_amd',
    'nir_intrinsic_isberd_nv', 'nir_intrinsic_lane_permute_16_amd',
    'nir_intrinsic_last_invocation',
    'nir_intrinsic_launch_mesh_workgroups',
    'nir_intrinsic_launch_mesh_workgroups_with_payload_deref',
    'nir_intrinsic_ldc_nv', 'nir_intrinsic_ldcx_nv',
    'nir_intrinsic_ldtram_nv', 'nir_intrinsic_load_aa_line_width',
    'nir_intrinsic_load_accel_struct_amd',
    'nir_intrinsic_load_active_samples_agx',
    'nir_intrinsic_load_active_subgroup_count_agx',
    'nir_intrinsic_load_active_subgroup_invocation_agx',
    'nir_intrinsic_load_agx',
    'nir_intrinsic_load_alpha_reference_amd',
    'nir_intrinsic_load_api_sample_mask_agx',
    'nir_intrinsic_load_attrib_clamp_agx',
    'nir_intrinsic_load_attribute_pan',
    'nir_intrinsic_load_back_face_agx',
    'nir_intrinsic_load_barycentric_at_offset',
    'nir_intrinsic_load_barycentric_at_offset_nv',
    'nir_intrinsic_load_barycentric_at_sample',
    'nir_intrinsic_load_barycentric_centroid',
    'nir_intrinsic_load_barycentric_coord_at_offset',
    'nir_intrinsic_load_barycentric_coord_at_sample',
    'nir_intrinsic_load_barycentric_coord_centroid',
    'nir_intrinsic_load_barycentric_coord_pixel',
    'nir_intrinsic_load_barycentric_coord_sample',
    'nir_intrinsic_load_barycentric_model',
    'nir_intrinsic_load_barycentric_optimize_amd',
    'nir_intrinsic_load_barycentric_pixel',
    'nir_intrinsic_load_barycentric_sample',
    'nir_intrinsic_load_base_global_invocation_id',
    'nir_intrinsic_load_base_instance',
    'nir_intrinsic_load_base_vertex',
    'nir_intrinsic_load_base_workgroup_id',
    'nir_intrinsic_load_blend_const_color_a_float',
    'nir_intrinsic_load_blend_const_color_aaaa8888_unorm',
    'nir_intrinsic_load_blend_const_color_b_float',
    'nir_intrinsic_load_blend_const_color_g_float',
    'nir_intrinsic_load_blend_const_color_r_float',
    'nir_intrinsic_load_blend_const_color_rgba',
    'nir_intrinsic_load_blend_const_color_rgba8888_unorm',
    'nir_intrinsic_load_btd_global_arg_addr_intel',
    'nir_intrinsic_load_btd_local_arg_addr_intel',
    'nir_intrinsic_load_btd_resume_sbt_addr_intel',
    'nir_intrinsic_load_btd_shader_type_intel',
    'nir_intrinsic_load_btd_stack_id_intel',
    'nir_intrinsic_load_buffer_amd',
    'nir_intrinsic_load_callable_sbt_addr_intel',
    'nir_intrinsic_load_callable_sbt_stride_intel',
    'nir_intrinsic_load_clamp_vertex_color_amd',
    'nir_intrinsic_load_clip_half_line_width_amd',
    'nir_intrinsic_load_clip_z_coeff_agx',
    'nir_intrinsic_load_coalesced_input_count',
    'nir_intrinsic_load_coefficients_agx',
    'nir_intrinsic_load_color0', 'nir_intrinsic_load_color1',
    'nir_intrinsic_load_const_buf_base_addr_lvp',
    'nir_intrinsic_load_const_ir3', 'nir_intrinsic_load_constant',
    'nir_intrinsic_load_constant_agx',
    'nir_intrinsic_load_constant_base_ptr',
    'nir_intrinsic_load_converted_output_pan',
    'nir_intrinsic_load_core_id_agx',
    'nir_intrinsic_load_cull_any_enabled_amd',
    'nir_intrinsic_load_cull_back_face_enabled_amd',
    'nir_intrinsic_load_cull_ccw_amd',
    'nir_intrinsic_load_cull_front_face_enabled_amd',
    'nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd',
    'nir_intrinsic_load_cull_mask',
    'nir_intrinsic_load_cull_mask_and_flags_amd',
    'nir_intrinsic_load_cull_small_line_precision_amd',
    'nir_intrinsic_load_cull_small_lines_enabled_amd',
    'nir_intrinsic_load_cull_small_triangle_precision_amd',
    'nir_intrinsic_load_cull_small_triangles_enabled_amd',
    'nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd',
    'nir_intrinsic_load_debug_log_desc_amd',
    'nir_intrinsic_load_depth_never_agx', 'nir_intrinsic_load_deref',
    'nir_intrinsic_load_deref_block_intel',
    'nir_intrinsic_load_draw_id',
    'nir_intrinsic_load_esgs_vertex_stride_amd',
    'nir_intrinsic_load_exported_agx',
    'nir_intrinsic_load_fb_layers_v3d',
    'nir_intrinsic_load_fbfetch_image_desc_amd',
    'nir_intrinsic_load_fbfetch_image_fmask_desc_amd',
    'nir_intrinsic_load_fep_w_v3d', 'nir_intrinsic_load_first_vertex',
    'nir_intrinsic_load_fixed_point_size_agx',
    'nir_intrinsic_load_flat_mask',
    'nir_intrinsic_load_force_vrs_rates_amd',
    'nir_intrinsic_load_frag_coord',
    'nir_intrinsic_load_frag_coord_unscaled_ir3',
    'nir_intrinsic_load_frag_coord_w',
    'nir_intrinsic_load_frag_coord_z',
    'nir_intrinsic_load_frag_coord_zw_pan',
    'nir_intrinsic_load_frag_invocation_count',
    'nir_intrinsic_load_frag_offset_ir3',
    'nir_intrinsic_load_frag_shading_rate',
    'nir_intrinsic_load_frag_size',
    'nir_intrinsic_load_frag_size_ir3',
    'nir_intrinsic_load_from_texture_handle_agx',
    'nir_intrinsic_load_front_face',
    'nir_intrinsic_load_front_face_fsign',
    'nir_intrinsic_load_fs_input_interp_deltas',
    'nir_intrinsic_load_fs_msaa_intel',
    'nir_intrinsic_load_fully_covered',
    'nir_intrinsic_load_geometry_param_buffer_poly',
    'nir_intrinsic_load_global', 'nir_intrinsic_load_global_2x32',
    'nir_intrinsic_load_global_amd',
    'nir_intrinsic_load_global_base_ptr',
    'nir_intrinsic_load_global_block_intel',
    'nir_intrinsic_load_global_bounded',
    'nir_intrinsic_load_global_constant',
    'nir_intrinsic_load_global_constant_bounded',
    'nir_intrinsic_load_global_constant_offset',
    'nir_intrinsic_load_global_constant_uniform_block_intel',
    'nir_intrinsic_load_global_etna',
    'nir_intrinsic_load_global_invocation_id',
    'nir_intrinsic_load_global_invocation_index',
    'nir_intrinsic_load_global_ir3', 'nir_intrinsic_load_global_size',
    'nir_intrinsic_load_gs_header_ir3',
    'nir_intrinsic_load_gs_vertex_offset_amd',
    'nir_intrinsic_load_gs_wave_id_amd',
    'nir_intrinsic_load_helper_arg_hi_agx',
    'nir_intrinsic_load_helper_arg_lo_agx',
    'nir_intrinsic_load_helper_invocation',
    'nir_intrinsic_load_helper_op_id_agx',
    'nir_intrinsic_load_hit_attrib_amd',
    'nir_intrinsic_load_hs_out_patch_data_offset_amd',
    'nir_intrinsic_load_hs_patch_stride_ir3',
    'nir_intrinsic_load_initial_edgeflags_amd',
    'nir_intrinsic_load_inline_data_intel',
    'nir_intrinsic_load_input',
    'nir_intrinsic_load_input_assembly_buffer_poly',
    'nir_intrinsic_load_input_attachment_conv_pan',
    'nir_intrinsic_load_input_attachment_coord',
    'nir_intrinsic_load_input_attachment_target_pan',
    'nir_intrinsic_load_input_topology_poly',
    'nir_intrinsic_load_input_vertex',
    'nir_intrinsic_load_instance_id',
    'nir_intrinsic_load_interpolated_input',
    'nir_intrinsic_load_intersection_opaque_amd',
    'nir_intrinsic_load_invocation_id',
    'nir_intrinsic_load_is_first_fan_agx',
    'nir_intrinsic_load_is_indexed_draw',
    'nir_intrinsic_load_kernel_input', 'nir_intrinsic_load_layer_id',
    'nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd',
    'nir_intrinsic_load_leaf_opaque_intel',
    'nir_intrinsic_load_leaf_procedural_intel',
    'nir_intrinsic_load_line_coord', 'nir_intrinsic_load_line_width',
    'nir_intrinsic_load_local_invocation_id',
    'nir_intrinsic_load_local_invocation_index',
    'nir_intrinsic_load_local_pixel_agx',
    'nir_intrinsic_load_local_shared_r600',
    'nir_intrinsic_load_lshs_vertex_stride_amd',
    'nir_intrinsic_load_max_polygon_intel',
    'nir_intrinsic_load_merged_wave_info_amd',
    'nir_intrinsic_load_mesh_view_count',
    'nir_intrinsic_load_mesh_view_indices',
    'nir_intrinsic_load_multisampled_pan',
    'nir_intrinsic_load_noperspective_varyings_pan',
    'nir_intrinsic_load_num_subgroups',
    'nir_intrinsic_load_num_vertices',
    'nir_intrinsic_load_num_vertices_per_primitive_amd',
    'nir_intrinsic_load_num_workgroups',
    'nir_intrinsic_load_ordered_id_amd', 'nir_intrinsic_load_output',
    'nir_intrinsic_load_packed_passthrough_primitive_amd',
    'nir_intrinsic_load_param',
    'nir_intrinsic_load_patch_vertices_in',
    'nir_intrinsic_load_per_primitive_input',
    'nir_intrinsic_load_per_primitive_output',
    'nir_intrinsic_load_per_primitive_remap_intel',
    'nir_intrinsic_load_per_vertex_input',
    'nir_intrinsic_load_per_vertex_output',
    'nir_intrinsic_load_per_view_output',
    'nir_intrinsic_load_persp_center_rhw_ir3',
    'nir_intrinsic_load_pipeline_stat_query_enabled_amd',
    'nir_intrinsic_load_pixel_coord',
    'nir_intrinsic_load_point_coord',
    'nir_intrinsic_load_point_coord_maybe_flipped',
    'nir_intrinsic_load_poly_line_smooth_enabled',
    'nir_intrinsic_load_polygon_stipple_agx',
    'nir_intrinsic_load_polygon_stipple_buffer_amd',
    'nir_intrinsic_load_preamble',
    'nir_intrinsic_load_prim_gen_query_enabled_amd',
    'nir_intrinsic_load_prim_xfb_query_enabled_amd',
    'nir_intrinsic_load_primitive_id',
    'nir_intrinsic_load_primitive_location_ir3',
    'nir_intrinsic_load_printf_buffer_address',
    'nir_intrinsic_load_printf_buffer_size',
    'nir_intrinsic_load_provoking_last',
    'nir_intrinsic_load_provoking_vtx_amd',
    'nir_intrinsic_load_provoking_vtx_in_prim_amd',
    'nir_intrinsic_load_push_constant',
    'nir_intrinsic_load_push_constant_zink',
    'nir_intrinsic_load_r600_indirect_per_vertex_input',
    'nir_intrinsic_load_rasterization_primitive_amd',
    'nir_intrinsic_load_rasterization_samples_amd',
    'nir_intrinsic_load_rasterization_stream',
    'nir_intrinsic_load_raw_output_pan',
    'nir_intrinsic_load_raw_vertex_id_pan',
    'nir_intrinsic_load_raw_vertex_offset_pan',
    'nir_intrinsic_load_ray_base_mem_addr_intel',
    'nir_intrinsic_load_ray_flags',
    'nir_intrinsic_load_ray_geometry_index',
    'nir_intrinsic_load_ray_hit_kind',
    'nir_intrinsic_load_ray_hit_sbt_addr_intel',
    'nir_intrinsic_load_ray_hit_sbt_stride_intel',
    'nir_intrinsic_load_ray_hw_stack_size_intel',
    'nir_intrinsic_load_ray_instance_custom_index',
    'nir_intrinsic_load_ray_launch_id',
    'nir_intrinsic_load_ray_launch_size',
    'nir_intrinsic_load_ray_miss_sbt_addr_intel',
    'nir_intrinsic_load_ray_miss_sbt_stride_intel',
    'nir_intrinsic_load_ray_num_dss_rt_stacks_intel',
    'nir_intrinsic_load_ray_object_direction',
    'nir_intrinsic_load_ray_object_origin',
    'nir_intrinsic_load_ray_object_to_world',
    'nir_intrinsic_load_ray_query_global_intel',
    'nir_intrinsic_load_ray_sw_stack_size_intel',
    'nir_intrinsic_load_ray_t_max', 'nir_intrinsic_load_ray_t_min',
    'nir_intrinsic_load_ray_tracing_stack_base_lvp',
    'nir_intrinsic_load_ray_triangle_vertex_positions',
    'nir_intrinsic_load_ray_world_direction',
    'nir_intrinsic_load_ray_world_origin',
    'nir_intrinsic_load_ray_world_to_object',
    'nir_intrinsic_load_readonly_output_pan',
    'nir_intrinsic_load_reg', 'nir_intrinsic_load_reg_indirect',
    'nir_intrinsic_load_rel_patch_id_ir3',
    'nir_intrinsic_load_reloc_const_intel',
    'nir_intrinsic_load_resume_shader_address_amd',
    'nir_intrinsic_load_ring_attr_amd',
    'nir_intrinsic_load_ring_attr_offset_amd',
    'nir_intrinsic_load_ring_es2gs_offset_amd',
    'nir_intrinsic_load_ring_esgs_amd',
    'nir_intrinsic_load_ring_gs2vs_offset_amd',
    'nir_intrinsic_load_ring_gsvs_amd',
    'nir_intrinsic_load_ring_mesh_scratch_amd',
    'nir_intrinsic_load_ring_mesh_scratch_offset_amd',
    'nir_intrinsic_load_ring_task_draw_amd',
    'nir_intrinsic_load_ring_task_payload_amd',
    'nir_intrinsic_load_ring_tess_factors_amd',
    'nir_intrinsic_load_ring_tess_factors_offset_amd',
    'nir_intrinsic_load_ring_tess_offchip_amd',
    'nir_intrinsic_load_ring_tess_offchip_offset_amd',
    'nir_intrinsic_load_root_agx',
    'nir_intrinsic_load_rt_arg_scratch_offset_amd',
    'nir_intrinsic_load_rt_conversion_pan',
    'nir_intrinsic_load_sample_id',
    'nir_intrinsic_load_sample_id_no_per_sample',
    'nir_intrinsic_load_sample_mask',
    'nir_intrinsic_load_sample_mask_in',
    'nir_intrinsic_load_sample_pos',
    'nir_intrinsic_load_sample_pos_from_id',
    'nir_intrinsic_load_sample_pos_or_center',
    'nir_intrinsic_load_sample_positions_agx',
    'nir_intrinsic_load_sample_positions_amd',
    'nir_intrinsic_load_sample_positions_pan',
    'nir_intrinsic_load_sampler_handle_agx',
    'nir_intrinsic_load_sampler_lod_parameters',
    'nir_intrinsic_load_samples_log2_agx',
    'nir_intrinsic_load_sbt_base_amd',
    'nir_intrinsic_load_sbt_offset_amd',
    'nir_intrinsic_load_sbt_stride_amd',
    'nir_intrinsic_load_scalar_arg_amd', 'nir_intrinsic_load_scratch',
    'nir_intrinsic_load_scratch_base_ptr',
    'nir_intrinsic_load_shader_call_data_offset_lvp',
    'nir_intrinsic_load_shader_index',
    'nir_intrinsic_load_shader_output_pan',
    'nir_intrinsic_load_shader_part_tests_zs_agx',
    'nir_intrinsic_load_shader_record_ptr',
    'nir_intrinsic_load_shared', 'nir_intrinsic_load_shared2_amd',
    'nir_intrinsic_load_shared_base_ptr',
    'nir_intrinsic_load_shared_block_intel',
    'nir_intrinsic_load_shared_ir3',
    'nir_intrinsic_load_shared_lock_nv',
    'nir_intrinsic_load_shared_uniform_block_intel',
    'nir_intrinsic_load_simd_width_intel',
    'nir_intrinsic_load_sm_count_nv', 'nir_intrinsic_load_sm_id_nv',
    'nir_intrinsic_load_smem_amd', 'nir_intrinsic_load_ssbo',
    'nir_intrinsic_load_ssbo_address',
    'nir_intrinsic_load_ssbo_block_intel',
    'nir_intrinsic_load_ssbo_intel', 'nir_intrinsic_load_ssbo_ir3',
    'nir_intrinsic_load_ssbo_uniform_block_intel',
    'nir_intrinsic_load_stack',
    'nir_intrinsic_load_stat_query_address_agx',
    'nir_intrinsic_load_streamout_buffer_amd',
    'nir_intrinsic_load_streamout_config_amd',
    'nir_intrinsic_load_streamout_offset_amd',
    'nir_intrinsic_load_streamout_write_index_amd',
    'nir_intrinsic_load_subgroup_eq_mask',
    'nir_intrinsic_load_subgroup_ge_mask',
    'nir_intrinsic_load_subgroup_gt_mask',
    'nir_intrinsic_load_subgroup_id',
    'nir_intrinsic_load_subgroup_id_shift_ir3',
    'nir_intrinsic_load_subgroup_invocation',
    'nir_intrinsic_load_subgroup_le_mask',
    'nir_intrinsic_load_subgroup_lt_mask',
    'nir_intrinsic_load_subgroup_size',
    'nir_intrinsic_load_sysval_agx', 'nir_intrinsic_load_sysval_nv',
    'nir_intrinsic_load_task_payload',
    'nir_intrinsic_load_task_ring_entry_amd',
    'nir_intrinsic_load_tcs_header_ir3',
    'nir_intrinsic_load_tcs_in_param_base_r600',
    'nir_intrinsic_load_tcs_mem_attrib_stride',
    'nir_intrinsic_load_tcs_num_patches_amd',
    'nir_intrinsic_load_tcs_out_param_base_r600',
    'nir_intrinsic_load_tcs_primitive_mode_amd',
    'nir_intrinsic_load_tcs_rel_patch_id_r600',
    'nir_intrinsic_load_tcs_tess_factor_base_r600',
    'nir_intrinsic_load_tcs_tess_levels_to_tes_amd',
    'nir_intrinsic_load_tess_coord',
    'nir_intrinsic_load_tess_coord_xy',
    'nir_intrinsic_load_tess_factor_base_ir3',
    'nir_intrinsic_load_tess_level_inner',
    'nir_intrinsic_load_tess_level_inner_default',
    'nir_intrinsic_load_tess_level_outer',
    'nir_intrinsic_load_tess_level_outer_default',
    'nir_intrinsic_load_tess_param_base_ir3',
    'nir_intrinsic_load_tess_param_buffer_poly',
    'nir_intrinsic_load_tess_rel_patch_id_amd',
    'nir_intrinsic_load_tex_sprite_mask_agx',
    'nir_intrinsic_load_texture_handle_agx',
    'nir_intrinsic_load_texture_scale',
    'nir_intrinsic_load_texture_size_etna',
    'nir_intrinsic_load_tlb_color_brcm',
    'nir_intrinsic_load_topology_id_intel',
    'nir_intrinsic_load_typed_buffer_amd',
    'nir_intrinsic_load_uav_ir3', 'nir_intrinsic_load_ubo',
    'nir_intrinsic_load_ubo_uniform_block_intel',
    'nir_intrinsic_load_ubo_vec4', 'nir_intrinsic_load_uniform',
    'nir_intrinsic_load_user_clip_plane',
    'nir_intrinsic_load_user_data_amd',
    'nir_intrinsic_load_uvs_index_agx',
    'nir_intrinsic_load_vbo_base_agx',
    'nir_intrinsic_load_vector_arg_amd',
    'nir_intrinsic_load_vertex_id',
    'nir_intrinsic_load_vertex_id_zero_base',
    'nir_intrinsic_load_view_index',
    'nir_intrinsic_load_viewport_offset',
    'nir_intrinsic_load_viewport_scale',
    'nir_intrinsic_load_viewport_x_offset',
    'nir_intrinsic_load_viewport_x_scale',
    'nir_intrinsic_load_viewport_y_offset',
    'nir_intrinsic_load_viewport_y_scale',
    'nir_intrinsic_load_viewport_z_offset',
    'nir_intrinsic_load_viewport_z_scale',
    'nir_intrinsic_load_vs_output_buffer_poly',
    'nir_intrinsic_load_vs_outputs_poly',
    'nir_intrinsic_load_vs_primitive_stride_ir3',
    'nir_intrinsic_load_vs_vertex_stride_ir3',
    'nir_intrinsic_load_vulkan_descriptor',
    'nir_intrinsic_load_warp_id_nv',
    'nir_intrinsic_load_warps_per_sm_nv',
    'nir_intrinsic_load_work_dim', 'nir_intrinsic_load_workgroup_id',
    'nir_intrinsic_load_workgroup_index',
    'nir_intrinsic_load_workgroup_num_input_primitives_amd',
    'nir_intrinsic_load_workgroup_num_input_vertices_amd',
    'nir_intrinsic_load_workgroup_size',
    'nir_intrinsic_load_xfb_address',
    'nir_intrinsic_load_xfb_index_buffer',
    'nir_intrinsic_load_xfb_size',
    'nir_intrinsic_load_xfb_state_address_gfx12_amd',
    'nir_intrinsic_masked_swizzle_amd', 'nir_intrinsic_mbcnt_amd',
    'nir_intrinsic_memcpy_deref', 'nir_intrinsic_nop',
    'nir_intrinsic_nop_amd', 'nir_intrinsic_op',
    'nir_intrinsic_op__enumvalues',
    'nir_intrinsic_optimization_barrier_sgpr_amd',
    'nir_intrinsic_optimization_barrier_vgpr_amd',
    'nir_intrinsic_ordered_add_loop_gfx12_amd',
    'nir_intrinsic_ordered_xfb_counter_add_gfx11_amd',
    'nir_intrinsic_overwrite_tes_arguments_amd',
    'nir_intrinsic_overwrite_vs_arguments_amd',
    'nir_intrinsic_pass_cb', 'nir_intrinsic_pin_cx_handle_nv',
    'nir_intrinsic_preamble_end_ir3',
    'nir_intrinsic_preamble_start_ir3',
    'nir_intrinsic_prefetch_sam_ir3',
    'nir_intrinsic_prefetch_tex_ir3',
    'nir_intrinsic_prefetch_ubo_ir3', 'nir_intrinsic_printf',
    'nir_intrinsic_printf_abort', 'nir_intrinsic_quad_ballot_agx',
    'nir_intrinsic_quad_broadcast',
    'nir_intrinsic_quad_swap_diagonal',
    'nir_intrinsic_quad_swap_horizontal',
    'nir_intrinsic_quad_swap_vertical',
    'nir_intrinsic_quad_swizzle_amd', 'nir_intrinsic_quad_vote_all',
    'nir_intrinsic_quad_vote_any',
    'nir_intrinsic_r600_indirect_vertex_at_index',
    'nir_intrinsic_ray_intersection_ir3',
    'nir_intrinsic_read_attribute_payload_intel',
    'nir_intrinsic_read_first_invocation',
    'nir_intrinsic_read_getlast_ir3', 'nir_intrinsic_read_invocation',
    'nir_intrinsic_read_invocation_cond_ir3', 'nir_intrinsic_reduce',
    'nir_intrinsic_reduce_clusters_ir3',
    'nir_intrinsic_report_ray_intersection',
    'nir_intrinsic_resource_intel', 'nir_intrinsic_rotate',
    'nir_intrinsic_rq_confirm_intersection',
    'nir_intrinsic_rq_generate_intersection',
    'nir_intrinsic_rq_initialize', 'nir_intrinsic_rq_load',
    'nir_intrinsic_rq_proceed', 'nir_intrinsic_rq_terminate',
    'nir_intrinsic_rt_execute_callable', 'nir_intrinsic_rt_resume',
    'nir_intrinsic_rt_return_amd', 'nir_intrinsic_rt_trace_ray',
    'nir_intrinsic_sample_mask_agx',
    'nir_intrinsic_select_vertex_poly', 'nir_intrinsic_semantic_flag',
    'nir_intrinsic_semantic_flag__enumvalues',
    'nir_intrinsic_sendmsg_amd', 'nir_intrinsic_set_align',
    'nir_intrinsic_set_vertex_and_primitive_count',
    'nir_intrinsic_shader_clock', 'nir_intrinsic_shared_append_amd',
    'nir_intrinsic_shared_atomic', 'nir_intrinsic_shared_atomic_swap',
    'nir_intrinsic_shared_consume_amd', 'nir_intrinsic_shuffle',
    'nir_intrinsic_shuffle_down',
    'nir_intrinsic_shuffle_down_uniform_ir3',
    'nir_intrinsic_shuffle_up',
    'nir_intrinsic_shuffle_up_uniform_ir3',
    'nir_intrinsic_shuffle_xor',
    'nir_intrinsic_shuffle_xor_uniform_ir3',
    'nir_intrinsic_sleep_amd',
    'nir_intrinsic_sparse_residency_code_and',
    'nir_intrinsic_src_components', 'nir_intrinsic_ssa_bar_nv',
    'nir_intrinsic_ssbo_atomic', 'nir_intrinsic_ssbo_atomic_ir3',
    'nir_intrinsic_ssbo_atomic_swap',
    'nir_intrinsic_ssbo_atomic_swap_ir3',
    'nir_intrinsic_stack_map_agx', 'nir_intrinsic_stack_unmap_agx',
    'nir_intrinsic_store_agx', 'nir_intrinsic_store_buffer_amd',
    'nir_intrinsic_store_combined_output_pan',
    'nir_intrinsic_store_const_ir3', 'nir_intrinsic_store_deref',
    'nir_intrinsic_store_deref_block_intel',
    'nir_intrinsic_store_global', 'nir_intrinsic_store_global_2x32',
    'nir_intrinsic_store_global_amd',
    'nir_intrinsic_store_global_block_intel',
    'nir_intrinsic_store_global_etna',
    'nir_intrinsic_store_global_ir3',
    'nir_intrinsic_store_hit_attrib_amd',
    'nir_intrinsic_store_local_pixel_agx',
    'nir_intrinsic_store_local_shared_r600',
    'nir_intrinsic_store_output',
    'nir_intrinsic_store_per_primitive_output',
    'nir_intrinsic_store_per_primitive_payload_intel',
    'nir_intrinsic_store_per_vertex_output',
    'nir_intrinsic_store_per_view_output',
    'nir_intrinsic_store_preamble',
    'nir_intrinsic_store_raw_output_pan', 'nir_intrinsic_store_reg',
    'nir_intrinsic_store_reg_indirect',
    'nir_intrinsic_store_scalar_arg_amd',
    'nir_intrinsic_store_scratch', 'nir_intrinsic_store_shared',
    'nir_intrinsic_store_shared2_amd',
    'nir_intrinsic_store_shared_block_intel',
    'nir_intrinsic_store_shared_ir3',
    'nir_intrinsic_store_shared_unlock_nv',
    'nir_intrinsic_store_ssbo',
    'nir_intrinsic_store_ssbo_block_intel',
    'nir_intrinsic_store_ssbo_intel', 'nir_intrinsic_store_ssbo_ir3',
    'nir_intrinsic_store_stack', 'nir_intrinsic_store_task_payload',
    'nir_intrinsic_store_tf_r600',
    'nir_intrinsic_store_tlb_sample_color_v3d',
    'nir_intrinsic_store_uvs_agx',
    'nir_intrinsic_store_vector_arg_amd',
    'nir_intrinsic_store_zs_agx',
    'nir_intrinsic_strict_wqm_coord_amd', 'nir_intrinsic_subfm_nv',
    'nir_intrinsic_suclamp_nv', 'nir_intrinsic_sueau_nv',
    'nir_intrinsic_suldga_nv', 'nir_intrinsic_sustga_nv',
    'nir_intrinsic_task_payload_atomic',
    'nir_intrinsic_task_payload_atomic_swap',
    'nir_intrinsic_terminate', 'nir_intrinsic_terminate_if',
    'nir_intrinsic_terminate_ray', 'nir_intrinsic_trace_ray',
    'nir_intrinsic_trace_ray_intel', 'nir_intrinsic_unit_test_amd',
    'nir_intrinsic_unit_test_divergent_amd',
    'nir_intrinsic_unit_test_uniform_amd',
    'nir_intrinsic_unpin_cx_handle_nv', 'nir_intrinsic_use',
    'nir_intrinsic_vild_nv', 'nir_intrinsic_vote_all',
    'nir_intrinsic_vote_any', 'nir_intrinsic_vote_feq',
    'nir_intrinsic_vote_ieq', 'nir_intrinsic_vulkan_resource_index',
    'nir_intrinsic_vulkan_resource_reindex',
    'nir_intrinsic_write_invocation_amd',
    'nir_intrinsic_writes_external_memory',
    'nir_intrinsic_xfb_counter_sub_gfx11_amd',
    'nir_io_16bit_input_output_support',
    'nir_io_add_const_offset_to_base',
    'nir_io_add_intrinsic_xfb_info',
    'nir_io_always_interpolate_convergent_fs_inputs',
    'nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups',
    'nir_io_compaction_rotates_color_channels',
    'nir_io_dont_use_pos_for_non_fs_varyings',
    'nir_io_has_flexible_input_interpolation_except_flat',
    'nir_io_has_intrinsics', 'nir_io_mediump_is_32bit',
    'nir_io_mix_convergent_flat_with_interpolated', 'nir_io_options',
    'nir_io_options__enumvalues', 'nir_io_prefer_scalar_fs_inputs',
    'nir_io_radv_intrinsic_component_workaround', 'nir_io_semantics',
    'nir_io_separate_clip_cull_distance_arrays',
    'nir_io_vectorizer_ignores_types', 'nir_io_xfb', 'nir_ior_imm',
    'nir_is_arrayed_io', 'nir_is_denorm_flush_to_zero',
    'nir_is_denorm_preserve', 'nir_is_float_control_inf_preserve',
    'nir_is_float_control_nan_preserve',
    'nir_is_float_control_signed_zero_inf_nan_preserve',
    'nir_is_float_control_signed_zero_preserve', 'nir_is_load_reg',
    'nir_is_output_load', 'nir_is_rounding_mode_rtne',
    'nir_is_rounding_mode_rtz', 'nir_is_same_comp_swizzle',
    'nir_is_sequential_comp_swizzle', 'nir_is_store_reg',
    'nir_ishl_imm', 'nir_ishr_imm', 'nir_isub_imm', 'nir_jump',
    'nir_jump_break', 'nir_jump_continue', 'nir_jump_goto',
    'nir_jump_goto_if', 'nir_jump_halt', 'nir_jump_instr',
    'nir_jump_instr_create', 'nir_jump_return', 'nir_jump_type',
    'nir_jump_type__enumvalues', 'nir_last_intrinsic',
    'nir_last_opcode', 'nir_legalize_16bit_sampler_srcs',
    'nir_link_opt_varyings', 'nir_link_shader_functions',
    'nir_link_varying_precision', 'nir_link_xfb_varyings',
    'nir_live_defs_impl', 'nir_load_array_var',
    'nir_load_array_var_imm', 'nir_load_barycentric',
    'nir_load_const_instr', 'nir_load_const_instr_create',
    'nir_load_deref', 'nir_load_deref_with_access', 'nir_load_global',
    'nir_load_global_constant', 'nir_load_grouping',
    'nir_load_grouping__enumvalues', 'nir_load_param', 'nir_load_reg',
    'nir_load_reg_for_def', 'nir_load_store_vectorize_options',
    'nir_load_system_value', 'nir_load_var',
    'nir_local_variable_create', 'nir_log_shader_annotated_tagged',
    'nir_loop', 'nir_loop_analyze_impl', 'nir_loop_continue_target',
    'nir_loop_control', 'nir_loop_control__enumvalues',
    'nir_loop_control_dont_unroll', 'nir_loop_control_none',
    'nir_loop_control_unroll', 'nir_loop_create',
    'nir_loop_first_block', 'nir_loop_first_continue_block',
    'nir_loop_has_continue_construct', 'nir_loop_induction_variable',
    'nir_loop_info', 'nir_loop_is_divergent', 'nir_loop_last_block',
    'nir_loop_last_continue_block', 'nir_loop_terminator',
    'nir_lower_64bit_phis', 'nir_lower_all_phis_to_scalar',
    'nir_lower_alpha_test', 'nir_lower_alpha_to_coverage',
    'nir_lower_alpha_to_one', 'nir_lower_alu',
    'nir_lower_alu_conversion_to_intrinsic',
    'nir_lower_alu_to_scalar', 'nir_lower_alu_vec8_16_srcs',
    'nir_lower_alu_width', 'nir_lower_amul',
    'nir_lower_array_deref_of_vec',
    'nir_lower_array_deref_of_vec_options',
    'nir_lower_array_deref_of_vec_options__enumvalues',
    'nir_lower_atomics', 'nir_lower_atomics_to_ssbo',
    'nir_lower_bcsel64', 'nir_lower_bit_count64',
    'nir_lower_bit_size', 'nir_lower_bit_size_callback',
    'nir_lower_bitfield_extract64', 'nir_lower_bitfield_reverse64',
    'nir_lower_bitmap', 'nir_lower_bitmap_options',
    'nir_lower_bool_to_bitsize', 'nir_lower_bool_to_float',
    'nir_lower_bool_to_int32', 'nir_lower_calls_to_builtins',
    'nir_lower_cl_images', 'nir_lower_clamp_color_outputs',
    'nir_lower_clip_cull_distance_array_vars',
    'nir_lower_clip_cull_distance_to_vec4s', 'nir_lower_clip_disable',
    'nir_lower_clip_fs', 'nir_lower_clip_gs', 'nir_lower_clip_halfz',
    'nir_lower_clip_vs', 'nir_lower_compute_system_values',
    'nir_lower_compute_system_values_options',
    'nir_lower_const_arrays_to_uniforms',
    'nir_lower_constant_convert_alu_types',
    'nir_lower_constant_to_temp', 'nir_lower_continue_constructs',
    'nir_lower_conv64', 'nir_lower_convert_alu_types',
    'nir_lower_dceil', 'nir_lower_ddiv',
    'nir_lower_default_point_size', 'nir_lower_demote_if_to_cf',
    'nir_lower_deref_copy_instr', 'nir_lower_dfloor',
    'nir_lower_dfract', 'nir_lower_direct_array_deref_of_vec_load',
    'nir_lower_direct_array_deref_of_vec_store',
    'nir_lower_discard_if', 'nir_lower_discard_if_options',
    'nir_lower_discard_if_options__enumvalues', 'nir_lower_divmod64',
    'nir_lower_dminmax', 'nir_lower_dmod', 'nir_lower_doubles',
    'nir_lower_doubles_op_to_options_mask',
    'nir_lower_doubles_options',
    'nir_lower_doubles_options__enumvalues', 'nir_lower_drawpixels',
    'nir_lower_drawpixels_options', 'nir_lower_drcp',
    'nir_lower_dround_even', 'nir_lower_drsq', 'nir_lower_dsat',
    'nir_lower_dsign', 'nir_lower_dsqrt', 'nir_lower_dsub',
    'nir_lower_dtrunc', 'nir_lower_explicit_io',
    'nir_lower_explicit_io_instr', 'nir_lower_extract64',
    'nir_lower_fb_read', 'nir_lower_find_lsb64',
    'nir_lower_flatshade', 'nir_lower_flrp', 'nir_lower_fp16_all',
    'nir_lower_fp16_cast_options',
    'nir_lower_fp16_cast_options__enumvalues', 'nir_lower_fp16_casts',
    'nir_lower_fp16_rd', 'nir_lower_fp16_rtne', 'nir_lower_fp16_rtz',
    'nir_lower_fp16_ru', 'nir_lower_fp16_split_fp64',
    'nir_lower_fp64_full_software',
    'nir_lower_frag_coord_to_pixel_coord', 'nir_lower_fragcolor',
    'nir_lower_fragcoord_wtrans', 'nir_lower_frexp',
    'nir_lower_global_vars_to_local', 'nir_lower_goto_ifs',
    'nir_lower_gs_intrinsics',
    'nir_lower_gs_intrinsics_count_primitives',
    'nir_lower_gs_intrinsics_count_vertices_per_primitive',
    'nir_lower_gs_intrinsics_flags',
    'nir_lower_gs_intrinsics_flags__enumvalues',
    'nir_lower_gs_intrinsics_overwrite_incomplete',
    'nir_lower_gs_intrinsics_per_stream', 'nir_lower_halt_to_return',
    'nir_lower_helper_writes', 'nir_lower_iabs64',
    'nir_lower_iadd3_64', 'nir_lower_iadd64', 'nir_lower_iadd_sat64',
    'nir_lower_icmp64', 'nir_lower_idiv', 'nir_lower_idiv_options',
    'nir_lower_image', 'nir_lower_image_atomics_to_global',
    'nir_lower_image_options', 'nir_lower_imul64',
    'nir_lower_imul_2x32_64', 'nir_lower_imul_high64',
    'nir_lower_indirect_array_deref_of_vec_load',
    'nir_lower_indirect_array_deref_of_vec_store',
    'nir_lower_indirect_derefs', 'nir_lower_indirect_var_derefs',
    'nir_lower_ineg64', 'nir_lower_input_attachments',
    'nir_lower_instr_cb', 'nir_lower_int64',
    'nir_lower_int64_float_conversions',
    'nir_lower_int64_op_to_options_mask', 'nir_lower_int64_options',
    'nir_lower_int64_options__enumvalues', 'nir_lower_int_to_float',
    'nir_lower_interpolation', 'nir_lower_interpolation_at_offset',
    'nir_lower_interpolation_at_sample',
    'nir_lower_interpolation_centroid',
    'nir_lower_interpolation_options',
    'nir_lower_interpolation_options__enumvalues',
    'nir_lower_interpolation_pixel', 'nir_lower_interpolation_sample',
    'nir_lower_io', 'nir_lower_io_array_vars_to_elements',
    'nir_lower_io_array_vars_to_elements_no_indirects',
    'nir_lower_io_indirect_loads',
    'nir_lower_io_lower_64bit_float_to_32',
    'nir_lower_io_lower_64bit_to_32',
    'nir_lower_io_lower_64bit_to_32_new', 'nir_lower_io_options',
    'nir_lower_io_options__enumvalues', 'nir_lower_io_passes',
    'nir_lower_io_to_scalar',
    'nir_lower_io_use_interpolated_input_intrinsics',
    'nir_lower_io_vars_to_scalar', 'nir_lower_io_vars_to_temporaries',
    'nir_lower_is_helper_invocation', 'nir_lower_isign64',
    'nir_lower_load_const_to_scalar', 'nir_lower_locals_to_regs',
    'nir_lower_logic64', 'nir_lower_mediump_io',
    'nir_lower_mediump_vars', 'nir_lower_mem_access_bit_sizes',
    'nir_lower_mem_access_bit_sizes_cb',
    'nir_lower_mem_access_bit_sizes_options', 'nir_lower_memcpy',
    'nir_lower_memory_model', 'nir_lower_minmax64',
    'nir_lower_multiview', 'nir_lower_multiview_options',
    'nir_lower_non_uniform_access',
    'nir_lower_non_uniform_access_callback',
    'nir_lower_non_uniform_access_options',
    'nir_lower_non_uniform_access_type',
    'nir_lower_non_uniform_access_type_count',
    'nir_lower_non_uniform_get_ssbo_size',
    'nir_lower_non_uniform_image_access',
    'nir_lower_non_uniform_src_access_callback',
    'nir_lower_non_uniform_ssbo_access',
    'nir_lower_non_uniform_texture_access',
    'nir_lower_non_uniform_texture_offset_access',
    'nir_lower_non_uniform_ubo_access', 'nir_lower_pack',
    'nir_lower_packing_num_ops', 'nir_lower_packing_op',
    'nir_lower_packing_op__enumvalues',
    'nir_lower_packing_op_pack_32_2x16',
    'nir_lower_packing_op_pack_32_4x8',
    'nir_lower_packing_op_pack_64_2x32',
    'nir_lower_packing_op_pack_64_4x16',
    'nir_lower_packing_op_unpack_32_2x16',
    'nir_lower_packing_op_unpack_32_4x8',
    'nir_lower_packing_op_unpack_64_2x32',
    'nir_lower_packing_op_unpack_64_4x16',
    'nir_lower_passthrough_edgeflags', 'nir_lower_patch_vertices',
    'nir_lower_phis_to_regs_block', 'nir_lower_phis_to_scalar',
    'nir_lower_pntc_ytransform', 'nir_lower_point_size',
    'nir_lower_point_size_mov', 'nir_lower_point_smooth',
    'nir_lower_poly_line_smooth', 'nir_lower_printf',
    'nir_lower_printf_buffer', 'nir_lower_printf_options',
    'nir_lower_read_invocation_to_scalar',
    'nir_lower_readonly_images_to_tex',
    'nir_lower_reg_intrinsics_to_ssa',
    'nir_lower_reg_intrinsics_to_ssa_impl', 'nir_lower_returns',
    'nir_lower_returns_impl', 'nir_lower_robust_access',
    'nir_lower_samplers', 'nir_lower_scan_reduce_bitwise64',
    'nir_lower_scan_reduce_iadd64', 'nir_lower_scratch_to_var',
    'nir_lower_shader_calls', 'nir_lower_shader_calls_options',
    'nir_lower_shader_calls_should_remat_func', 'nir_lower_shift64',
    'nir_lower_single_sampled', 'nir_lower_ssa_defs_to_regs_block',
    'nir_lower_ssbo', 'nir_lower_ssbo_options',
    'nir_lower_subgroup_shuffle64', 'nir_lower_subgroups',
    'nir_lower_subgroups_options', 'nir_lower_system_values',
    'nir_lower_sysvals_to_varyings',
    'nir_lower_sysvals_to_varyings_options', 'nir_lower_task_shader',
    'nir_lower_task_shader_options', 'nir_lower_terminate_if_to_cf',
    'nir_lower_terminate_to_demote', 'nir_lower_tess_coord_z',
    'nir_lower_tess_level_array_vars_to_vec', 'nir_lower_tex',
    'nir_lower_tex_options', 'nir_lower_tex_packing',
    'nir_lower_tex_packing_16', 'nir_lower_tex_packing_8',
    'nir_lower_tex_packing_none', 'nir_lower_tex_shadow',
    'nir_lower_tex_shadow_swizzle', 'nir_lower_texcoord_replace',
    'nir_lower_texcoord_replace_late', 'nir_lower_two_sided_color',
    'nir_lower_uadd_sat64', 'nir_lower_ubo_vec4',
    'nir_lower_ufind_msb64', 'nir_lower_undef_to_zero',
    'nir_lower_uniforms_to_ubo', 'nir_lower_usub_sat64',
    'nir_lower_var_copies', 'nir_lower_var_copy_instr',
    'nir_lower_variable_initializers',
    'nir_lower_vars_to_explicit_types', 'nir_lower_vars_to_scratch',
    'nir_lower_vars_to_ssa', 'nir_lower_vec3_to_vec4',
    'nir_lower_vec_to_regs', 'nir_lower_view_index_to_device_index',
    'nir_lower_viewport_transform', 'nir_lower_vote_ieq64',
    'nir_lower_wpos_center', 'nir_lower_wpos_ytransform',
    'nir_lower_wpos_ytransform_options', 'nir_lower_wrmasks',
    'nir_mask', 'nir_mem_access_shift_method',
    'nir_mem_access_shift_method__enumvalues',
    'nir_mem_access_shift_method_bytealign_amd',
    'nir_mem_access_shift_method_scalar',
    'nir_mem_access_shift_method_shift64',
    'nir_mem_access_size_align', 'nir_memcpy_deref',
    'nir_memcpy_deref_with_access', 'nir_memory_semantics',
    'nir_memory_semantics__enumvalues', 'nir_metadata',
    'nir_metadata__enumvalues', 'nir_metadata_all',
    'nir_metadata_block_index', 'nir_metadata_check_validation_flag',
    'nir_metadata_control_flow', 'nir_metadata_divergence',
    'nir_metadata_dominance', 'nir_metadata_instr_index',
    'nir_metadata_invalidate', 'nir_metadata_live_defs',
    'nir_metadata_loop_analysis', 'nir_metadata_none',
    'nir_metadata_not_properly_reset', 'nir_metadata_require',
    'nir_metadata_require_all', 'nir_metadata_set_validation_flag',
    'nir_minimize_call_live_states', 'nir_mod_analysis',
    'nir_mov_alu', 'nir_move_alu', 'nir_move_comparisons',
    'nir_move_const_undef', 'nir_move_copies', 'nir_move_load_input',
    'nir_move_load_ssbo', 'nir_move_load_ubo',
    'nir_move_load_uniform', 'nir_move_options',
    'nir_move_options__enumvalues', 'nir_move_output_stores_to_end',
    'nir_move_terminate_out_of_loops', 'nir_move_to_entry_block_only',
    'nir_move_to_top_input_loads', 'nir_move_to_top_load_smem_amd',
    'nir_move_vec_src_uses_to_dest', 'nir_next_decl_reg',
    'nir_next_phi', 'nir_no_progress', 'nir_normalize_cubemap_coords',
    'nir_num_intrinsics', 'nir_num_opcodes', 'nir_num_tex_src_types',
    'nir_num_variable_modes', 'nir_op', 'nir_op__enumvalues',
    'nir_op_algebraic_property',
    'nir_op_algebraic_property__enumvalues', 'nir_op_alignbyte_amd',
    'nir_op_amul', 'nir_op_andg_ir3', 'nir_op_b16all_fequal16',
    'nir_op_b16all_fequal2', 'nir_op_b16all_fequal3',
    'nir_op_b16all_fequal4', 'nir_op_b16all_fequal5',
    'nir_op_b16all_fequal8', 'nir_op_b16all_iequal16',
    'nir_op_b16all_iequal2', 'nir_op_b16all_iequal3',
    'nir_op_b16all_iequal4', 'nir_op_b16all_iequal5',
    'nir_op_b16all_iequal8', 'nir_op_b16any_fnequal16',
    'nir_op_b16any_fnequal2', 'nir_op_b16any_fnequal3',
    'nir_op_b16any_fnequal4', 'nir_op_b16any_fnequal5',
    'nir_op_b16any_fnequal8', 'nir_op_b16any_inequal16',
    'nir_op_b16any_inequal2', 'nir_op_b16any_inequal3',
    'nir_op_b16any_inequal4', 'nir_op_b16any_inequal5',
    'nir_op_b16any_inequal8', 'nir_op_b16csel', 'nir_op_b2b1',
    'nir_op_b2b16', 'nir_op_b2b32', 'nir_op_b2b8', 'nir_op_b2f16',
    'nir_op_b2f32', 'nir_op_b2f64', 'nir_op_b2i1', 'nir_op_b2i16',
    'nir_op_b2i32', 'nir_op_b2i64', 'nir_op_b2i8',
    'nir_op_b32all_fequal16', 'nir_op_b32all_fequal2',
    'nir_op_b32all_fequal3', 'nir_op_b32all_fequal4',
    'nir_op_b32all_fequal5', 'nir_op_b32all_fequal8',
    'nir_op_b32all_iequal16', 'nir_op_b32all_iequal2',
    'nir_op_b32all_iequal3', 'nir_op_b32all_iequal4',
    'nir_op_b32all_iequal5', 'nir_op_b32all_iequal8',
    'nir_op_b32any_fnequal16', 'nir_op_b32any_fnequal2',
    'nir_op_b32any_fnequal3', 'nir_op_b32any_fnequal4',
    'nir_op_b32any_fnequal5', 'nir_op_b32any_fnequal8',
    'nir_op_b32any_inequal16', 'nir_op_b32any_inequal2',
    'nir_op_b32any_inequal3', 'nir_op_b32any_inequal4',
    'nir_op_b32any_inequal5', 'nir_op_b32any_inequal8',
    'nir_op_b32csel', 'nir_op_b32fcsel_mdg', 'nir_op_b8all_fequal16',
    'nir_op_b8all_fequal2', 'nir_op_b8all_fequal3',
    'nir_op_b8all_fequal4', 'nir_op_b8all_fequal5',
    'nir_op_b8all_fequal8', 'nir_op_b8all_iequal16',
    'nir_op_b8all_iequal2', 'nir_op_b8all_iequal3',
    'nir_op_b8all_iequal4', 'nir_op_b8all_iequal5',
    'nir_op_b8all_iequal8', 'nir_op_b8any_fnequal16',
    'nir_op_b8any_fnequal2', 'nir_op_b8any_fnequal3',
    'nir_op_b8any_fnequal4', 'nir_op_b8any_fnequal5',
    'nir_op_b8any_fnequal8', 'nir_op_b8any_inequal16',
    'nir_op_b8any_inequal2', 'nir_op_b8any_inequal3',
    'nir_op_b8any_inequal4', 'nir_op_b8any_inequal5',
    'nir_op_b8any_inequal8', 'nir_op_b8csel', 'nir_op_ball_fequal16',
    'nir_op_ball_fequal2', 'nir_op_ball_fequal3',
    'nir_op_ball_fequal4', 'nir_op_ball_fequal5',
    'nir_op_ball_fequal8', 'nir_op_ball_iequal16',
    'nir_op_ball_iequal2', 'nir_op_ball_iequal3',
    'nir_op_ball_iequal4', 'nir_op_ball_iequal5',
    'nir_op_ball_iequal8', 'nir_op_bany_fnequal16',
    'nir_op_bany_fnequal2', 'nir_op_bany_fnequal3',
    'nir_op_bany_fnequal4', 'nir_op_bany_fnequal5',
    'nir_op_bany_fnequal8', 'nir_op_bany_inequal16',
    'nir_op_bany_inequal2', 'nir_op_bany_inequal3',
    'nir_op_bany_inequal4', 'nir_op_bany_inequal5',
    'nir_op_bany_inequal8', 'nir_op_bcsel', 'nir_op_bf2f',
    'nir_op_bfdot16', 'nir_op_bfdot2', 'nir_op_bfdot2_bfadd',
    'nir_op_bfdot3', 'nir_op_bfdot4', 'nir_op_bfdot5',
    'nir_op_bfdot8', 'nir_op_bffma', 'nir_op_bfi', 'nir_op_bfm',
    'nir_op_bfmul', 'nir_op_bit_count', 'nir_op_bitfield_insert',
    'nir_op_bitfield_reverse', 'nir_op_bitfield_select',
    'nir_op_bitnz', 'nir_op_bitnz16', 'nir_op_bitnz32',
    'nir_op_bitnz8', 'nir_op_bitz', 'nir_op_bitz16', 'nir_op_bitz32',
    'nir_op_bitz8', 'nir_op_bounds_agx', 'nir_op_byte_perm_amd',
    'nir_op_cube_amd', 'nir_op_e4m3fn2f', 'nir_op_e5m22f',
    'nir_op_extr_agx', 'nir_op_extract_i16', 'nir_op_extract_i8',
    'nir_op_extract_u16', 'nir_op_extract_u8', 'nir_op_f2bf',
    'nir_op_f2e4m3fn', 'nir_op_f2e4m3fn_sat', 'nir_op_f2e4m3fn_satfn',
    'nir_op_f2e5m2', 'nir_op_f2e5m2_sat', 'nir_op_f2f16',
    'nir_op_f2f16_rtne', 'nir_op_f2f16_rtz', 'nir_op_f2f32',
    'nir_op_f2f64', 'nir_op_f2fmp', 'nir_op_f2i1', 'nir_op_f2i16',
    'nir_op_f2i32', 'nir_op_f2i64', 'nir_op_f2i8', 'nir_op_f2imp',
    'nir_op_f2snorm_16_v3d', 'nir_op_f2u1', 'nir_op_f2u16',
    'nir_op_f2u32', 'nir_op_f2u64', 'nir_op_f2u8', 'nir_op_f2ump',
    'nir_op_f2unorm_16_v3d', 'nir_op_fabs', 'nir_op_fadd',
    'nir_op_fall_equal16', 'nir_op_fall_equal2', 'nir_op_fall_equal3',
    'nir_op_fall_equal4', 'nir_op_fall_equal5', 'nir_op_fall_equal8',
    'nir_op_fany_nequal16', 'nir_op_fany_nequal2',
    'nir_op_fany_nequal3', 'nir_op_fany_nequal4',
    'nir_op_fany_nequal5', 'nir_op_fany_nequal8', 'nir_op_fceil',
    'nir_op_fclamp_pos', 'nir_op_fcos', 'nir_op_fcos_amd',
    'nir_op_fcos_mdg', 'nir_op_fcsel', 'nir_op_fcsel_ge',
    'nir_op_fcsel_gt', 'nir_op_fdiv', 'nir_op_fdot16',
    'nir_op_fdot16_replicated', 'nir_op_fdot2',
    'nir_op_fdot2_replicated', 'nir_op_fdot3',
    'nir_op_fdot3_replicated', 'nir_op_fdot4',
    'nir_op_fdot4_replicated', 'nir_op_fdot5',
    'nir_op_fdot5_replicated', 'nir_op_fdot8',
    'nir_op_fdot8_replicated', 'nir_op_fdph',
    'nir_op_fdph_replicated', 'nir_op_feq', 'nir_op_feq16',
    'nir_op_feq32', 'nir_op_feq8', 'nir_op_fequ', 'nir_op_fequ16',
    'nir_op_fequ32', 'nir_op_fequ8', 'nir_op_fexp2', 'nir_op_ffloor',
    'nir_op_ffma', 'nir_op_ffmaz', 'nir_op_ffract', 'nir_op_fge',
    'nir_op_fge16', 'nir_op_fge32', 'nir_op_fge8', 'nir_op_fgeu',
    'nir_op_fgeu16', 'nir_op_fgeu32', 'nir_op_fgeu8',
    'nir_op_find_lsb', 'nir_op_fisfinite', 'nir_op_fisfinite32',
    'nir_op_fisnormal', 'nir_op_flog2', 'nir_op_flrp', 'nir_op_flt',
    'nir_op_flt16', 'nir_op_flt32', 'nir_op_flt8', 'nir_op_fltu',
    'nir_op_fltu16', 'nir_op_fltu32', 'nir_op_fltu8', 'nir_op_fmax',
    'nir_op_fmax_agx', 'nir_op_fmin', 'nir_op_fmin_agx',
    'nir_op_fmod', 'nir_op_fmul', 'nir_op_fmulz', 'nir_op_fneg',
    'nir_op_fneo', 'nir_op_fneo16', 'nir_op_fneo32', 'nir_op_fneo8',
    'nir_op_fneu', 'nir_op_fneu16', 'nir_op_fneu32', 'nir_op_fneu8',
    'nir_op_ford', 'nir_op_ford16', 'nir_op_ford32', 'nir_op_ford8',
    'nir_op_fpow', 'nir_op_fquantize2f16', 'nir_op_frcp',
    'nir_op_frem', 'nir_op_frexp_exp', 'nir_op_frexp_sig',
    'nir_op_fround_even', 'nir_op_frsq', 'nir_op_fsat',
    'nir_op_fsat_signed', 'nir_op_fsign', 'nir_op_fsin',
    'nir_op_fsin_agx', 'nir_op_fsin_amd', 'nir_op_fsin_mdg',
    'nir_op_fsqrt', 'nir_op_fsub', 'nir_op_fsum2', 'nir_op_fsum3',
    'nir_op_fsum4', 'nir_op_ftrunc', 'nir_op_funord',
    'nir_op_funord16', 'nir_op_funord32', 'nir_op_funord8',
    'nir_op_i2f16', 'nir_op_i2f32', 'nir_op_i2f64', 'nir_op_i2fmp',
    'nir_op_i2i1', 'nir_op_i2i16', 'nir_op_i2i32', 'nir_op_i2i64',
    'nir_op_i2i8', 'nir_op_i2imp', 'nir_op_i32csel_ge',
    'nir_op_i32csel_gt', 'nir_op_iabs', 'nir_op_iadd', 'nir_op_iadd3',
    'nir_op_iadd_sat', 'nir_op_iand', 'nir_op_ibfe',
    'nir_op_ibitfield_extract', 'nir_op_icsel_eqz', 'nir_op_idiv',
    'nir_op_ieq', 'nir_op_ieq16', 'nir_op_ieq32', 'nir_op_ieq8',
    'nir_op_ifind_msb', 'nir_op_ifind_msb_rev', 'nir_op_ige',
    'nir_op_ige16', 'nir_op_ige32', 'nir_op_ige8', 'nir_op_ihadd',
    'nir_op_ilea_agx', 'nir_op_ilt', 'nir_op_ilt16', 'nir_op_ilt32',
    'nir_op_ilt8', 'nir_op_imad', 'nir_op_imad24_ir3',
    'nir_op_imadsh_mix16', 'nir_op_imadshl_agx', 'nir_op_imax',
    'nir_op_imin', 'nir_op_imod', 'nir_op_imsubshl_agx',
    'nir_op_imul', 'nir_op_imul24', 'nir_op_imul24_relaxed',
    'nir_op_imul_2x32_64', 'nir_op_imul_32x16', 'nir_op_imul_high',
    'nir_op_ine', 'nir_op_ine16', 'nir_op_ine32', 'nir_op_ine8',
    'nir_op_ineg', 'nir_op_info', 'nir_op_infos', 'nir_op_inot',
    'nir_op_insert_u16', 'nir_op_insert_u8', 'nir_op_interleave_agx',
    'nir_op_ior', 'nir_op_irem', 'nir_op_irhadd',
    'nir_op_is_selection', 'nir_op_is_vec', 'nir_op_is_vec_or_mov',
    'nir_op_ishl', 'nir_op_ishr', 'nir_op_isign', 'nir_op_isub',
    'nir_op_isub_sat', 'nir_op_ixor', 'nir_op_ldexp',
    'nir_op_ldexp16_pan', 'nir_op_lea_nv', 'nir_op_mov',
    'nir_op_mqsad_4x8', 'nir_op_msad_4x8',
    'nir_op_pack_2x16_to_snorm_2x8_v3d',
    'nir_op_pack_2x16_to_unorm_10_2_v3d',
    'nir_op_pack_2x16_to_unorm_2x10_v3d',
    'nir_op_pack_2x16_to_unorm_2x8_v3d',
    'nir_op_pack_2x32_to_2x16_v3d', 'nir_op_pack_32_2x16',
    'nir_op_pack_32_2x16_split', 'nir_op_pack_32_4x8',
    'nir_op_pack_32_4x8_split', 'nir_op_pack_32_to_r11g11b10_v3d',
    'nir_op_pack_4x16_to_4x8_v3d', 'nir_op_pack_64_2x32',
    'nir_op_pack_64_2x32_split', 'nir_op_pack_64_4x16',
    'nir_op_pack_double_2x32_dxil', 'nir_op_pack_half_2x16',
    'nir_op_pack_half_2x16_rtz_split', 'nir_op_pack_half_2x16_split',
    'nir_op_pack_sint_2x16', 'nir_op_pack_snorm_2x16',
    'nir_op_pack_snorm_4x8', 'nir_op_pack_uint_2x16',
    'nir_op_pack_uint_32_to_r10g10b10a2_v3d',
    'nir_op_pack_unorm_2x16', 'nir_op_pack_unorm_4x8',
    'nir_op_pack_uvec2_to_uint', 'nir_op_pack_uvec4_to_uint',
    'nir_op_prmt_nv', 'nir_op_sdot_2x16_iadd',
    'nir_op_sdot_2x16_iadd_sat', 'nir_op_sdot_4x8_iadd',
    'nir_op_sdot_4x8_iadd_sat', 'nir_op_seq', 'nir_op_sge',
    'nir_op_shfr', 'nir_op_shlg_ir3', 'nir_op_shlm_ir3',
    'nir_op_shrg_ir3', 'nir_op_shrm_ir3', 'nir_op_slt', 'nir_op_sne',
    'nir_op_sudot_4x8_iadd', 'nir_op_sudot_4x8_iadd_sat',
    'nir_op_u2f16', 'nir_op_u2f32', 'nir_op_u2f64', 'nir_op_u2fmp',
    'nir_op_u2u1', 'nir_op_u2u16', 'nir_op_u2u32', 'nir_op_u2u64',
    'nir_op_u2u8', 'nir_op_uabs_isub', 'nir_op_uabs_usub',
    'nir_op_uadd_carry', 'nir_op_uadd_sat', 'nir_op_ubfe',
    'nir_op_ubitfield_extract', 'nir_op_uclz', 'nir_op_udiv',
    'nir_op_udiv_aligned_4', 'nir_op_udot_2x16_uadd',
    'nir_op_udot_2x16_uadd_sat', 'nir_op_udot_4x8_uadd',
    'nir_op_udot_4x8_uadd_sat', 'nir_op_ufind_msb',
    'nir_op_ufind_msb_rev', 'nir_op_uge', 'nir_op_uge16',
    'nir_op_uge32', 'nir_op_uge8', 'nir_op_uhadd', 'nir_op_ulea_agx',
    'nir_op_ult', 'nir_op_ult16', 'nir_op_ult32', 'nir_op_ult8',
    'nir_op_umad24', 'nir_op_umad24_relaxed', 'nir_op_umax',
    'nir_op_umax_4x8_vc4', 'nir_op_umin', 'nir_op_umin_4x8_vc4',
    'nir_op_umod', 'nir_op_umul24', 'nir_op_umul24_relaxed',
    'nir_op_umul_2x32_64', 'nir_op_umul_32x16', 'nir_op_umul_high',
    'nir_op_umul_low', 'nir_op_umul_unorm_4x8_vc4',
    'nir_op_unpack_32_2x16', 'nir_op_unpack_32_2x16_split_x',
    'nir_op_unpack_32_2x16_split_y', 'nir_op_unpack_32_4x8',
    'nir_op_unpack_64_2x32', 'nir_op_unpack_64_2x32_split_x',
    'nir_op_unpack_64_2x32_split_y', 'nir_op_unpack_64_4x16',
    'nir_op_unpack_double_2x32_dxil', 'nir_op_unpack_half_2x16',
    'nir_op_unpack_half_2x16_split_x',
    'nir_op_unpack_half_2x16_split_y', 'nir_op_unpack_snorm_2x16',
    'nir_op_unpack_snorm_4x8', 'nir_op_unpack_unorm_2x16',
    'nir_op_unpack_unorm_4x8', 'nir_op_urhadd', 'nir_op_urol',
    'nir_op_uror', 'nir_op_usadd_4x8_vc4', 'nir_op_ushr',
    'nir_op_ussub_4x8_vc4', 'nir_op_usub_borrow', 'nir_op_usub_sat',
    'nir_op_vec', 'nir_op_vec16', 'nir_op_vec2', 'nir_op_vec3',
    'nir_op_vec4', 'nir_op_vec5', 'nir_op_vec8',
    'nir_opt_16bit_tex_image', 'nir_opt_16bit_tex_image_options',
    'nir_opt_access', 'nir_opt_access_options',
    'nir_opt_acquire_release_barriers', 'nir_opt_algebraic',
    'nir_opt_algebraic_before_ffma',
    'nir_opt_algebraic_before_lower_int64',
    'nir_opt_algebraic_distribute_src_mods',
    'nir_opt_algebraic_integer_promotion', 'nir_opt_algebraic_late',
    'nir_opt_barrier_modes', 'nir_opt_clip_cull_const',
    'nir_opt_combine_barriers', 'nir_opt_combine_stores',
    'nir_opt_comparison_pre', 'nir_opt_comparison_pre_impl',
    'nir_opt_constant_folding', 'nir_opt_copy_prop_vars',
    'nir_opt_cse', 'nir_opt_dce', 'nir_opt_dead_cf',
    'nir_opt_dead_write_vars', 'nir_opt_deref', 'nir_opt_deref_impl',
    'nir_opt_find_array_copies', 'nir_opt_frag_coord_to_pixel_coord',
    'nir_opt_fragdepth', 'nir_opt_gcm', 'nir_opt_generate_bfi',
    'nir_opt_idiv_const', 'nir_opt_if', 'nir_opt_if_avoid_64bit_phis',
    'nir_opt_if_optimize_phi_true_false', 'nir_opt_if_options',
    'nir_opt_if_options__enumvalues', 'nir_opt_intrinsics',
    'nir_opt_large_constants', 'nir_opt_licm',
    'nir_opt_load_store_update_alignments',
    'nir_opt_load_store_vectorize', 'nir_opt_loop',
    'nir_opt_loop_unroll', 'nir_opt_memcpy', 'nir_opt_move',
    'nir_opt_move_discards_to_top', 'nir_opt_move_to_top',
    'nir_opt_move_to_top_options',
    'nir_opt_move_to_top_options__enumvalues', 'nir_opt_mqsad',
    'nir_opt_non_uniform_access', 'nir_opt_offsets',
    'nir_opt_offsets_options', 'nir_opt_peephole_select',
    'nir_opt_peephole_select_options', 'nir_opt_phi_precision',
    'nir_opt_phi_to_bool', 'nir_opt_preamble',
    'nir_opt_preamble_options', 'nir_opt_ray_queries',
    'nir_opt_ray_query_ranges', 'nir_opt_reassociate_bfi',
    'nir_opt_reassociate_matrix_mul',
    'nir_opt_rematerialize_compares', 'nir_opt_remove_phis',
    'nir_opt_shrink_stores', 'nir_opt_shrink_vectors',
    'nir_opt_simplify_convert_alu_types', 'nir_opt_sink',
    'nir_opt_tex_skip_helpers', 'nir_opt_tex_srcs_options',
    'nir_opt_undef', 'nir_opt_uniform_atomics',
    'nir_opt_uniform_subgroup', 'nir_opt_varyings',
    'nir_opt_varyings_progress',
    'nir_opt_varyings_progress__enumvalues', 'nir_opt_vectorize',
    'nir_opt_vectorize_io', 'nir_opt_vectorize_io_vars',
    'nir_output_clipper_var_groups', 'nir_output_deps',
    'nir_pack_bits', 'nir_pad_vec4', 'nir_pad_vector',
    'nir_pad_vector_imm_int', 'nir_parallel_copy_entry',
    'nir_parallel_copy_instr', 'nir_parallel_copy_instr_create',
    'nir_parameter', 'nir_phi_get_src_from_block', 'nir_phi_instr',
    'nir_phi_instr_add_src', 'nir_phi_instr_create',
    'nir_phi_pass_cb', 'nir_phi_src', 'nir_pop_if', 'nir_pop_loop',
    'nir_preamble_class', 'nir_preamble_class__enumvalues',
    'nir_preamble_class_general', 'nir_preamble_class_image',
    'nir_preamble_num_classes', 'nir_print_deref',
    'nir_print_function_body', 'nir_print_input_to_output_deps',
    'nir_print_instr', 'nir_print_shader',
    'nir_print_shader_annotated', 'nir_print_use_dominators',
    'nir_printf_fmt', 'nir_printf_fmt_at_px',
    'nir_process_debug_variable', 'nir_progress',
    'nir_progress_consumer', 'nir_progress_producer',
    'nir_propagate_invariant', 'nir_push_continue', 'nir_push_else',
    'nir_push_if', 'nir_push_loop', 'nir_ray_query_value',
    'nir_ray_query_value__enumvalues', 'nir_ray_query_value_flags',
    'nir_ray_query_value_intersection_barycentrics',
    'nir_ray_query_value_intersection_candidate_aabb_opaque',
    'nir_ray_query_value_intersection_front_face',
    'nir_ray_query_value_intersection_geometry_index',
    'nir_ray_query_value_intersection_instance_custom_index',
    'nir_ray_query_value_intersection_instance_id',
    'nir_ray_query_value_intersection_instance_sbt_index',
    'nir_ray_query_value_intersection_object_ray_direction',
    'nir_ray_query_value_intersection_object_ray_origin',
    'nir_ray_query_value_intersection_object_to_world',
    'nir_ray_query_value_intersection_primitive_index',
    'nir_ray_query_value_intersection_t',
    'nir_ray_query_value_intersection_triangle_vertex_positions',
    'nir_ray_query_value_intersection_type',
    'nir_ray_query_value_intersection_world_to_object',
    'nir_ray_query_value_tmin',
    'nir_ray_query_value_world_ray_direction',
    'nir_ray_query_value_world_ray_origin', 'nir_recompute_io_bases',
    'nir_reg_get_decl', 'nir_rematerialize_deref_in_use_blocks',
    'nir_rematerialize_derefs_in_use_blocks_impl',
    'nir_remove_dead_derefs', 'nir_remove_dead_derefs_impl',
    'nir_remove_dead_variables', 'nir_remove_dead_variables_options',
    'nir_remove_entrypoints', 'nir_remove_non_entrypoints',
    'nir_remove_non_exported', 'nir_remove_single_src_phis_block',
    'nir_remove_sysval_output', 'nir_remove_tex_shadow',
    'nir_remove_unused_io_vars', 'nir_remove_unused_varyings',
    'nir_remove_varying', 'nir_repair_ssa', 'nir_repair_ssa_impl',
    'nir_replicate', 'nir_resize_vector', 'nir_resource_data_intel',
    'nir_resource_data_intel__enumvalues',
    'nir_resource_intel_bindless', 'nir_resource_intel_non_uniform',
    'nir_resource_intel_pushable', 'nir_resource_intel_sampler',
    'nir_resource_intel_sampler_embedded',
    'nir_rewrite_image_intrinsic', 'nir_rewrite_uses_to_load_reg',
    'nir_round_down_components', 'nir_round_up_components',
    'nir_rounding_mode', 'nir_rounding_mode__enumvalues',
    'nir_rounding_mode_rd', 'nir_rounding_mode_rtne',
    'nir_rounding_mode_rtz', 'nir_rounding_mode_ru',
    'nir_rounding_mode_undef', 'nir_samples_identical_deref',
    'nir_scalar', 'nir_scalar_alu_op', 'nir_scalar_as_bool',
    'nir_scalar_as_const_value', 'nir_scalar_as_float',
    'nir_scalar_as_int', 'nir_scalar_as_uint',
    'nir_scalar_chase_alu_src', 'nir_scalar_chase_movs',
    'nir_scalar_equal', 'nir_scalar_intrinsic_op',
    'nir_scalar_is_alu', 'nir_scalar_is_const',
    'nir_scalar_is_intrinsic', 'nir_scalar_is_undef',
    'nir_scalar_resolved', 'nir_scale_fdiv',
    'nir_scoped_memory_barrier', 'nir_select_from_ssa_def_array',
    'nir_selection_control', 'nir_selection_control__enumvalues',
    'nir_selection_control_divergent_always_taken',
    'nir_selection_control_dont_flatten',
    'nir_selection_control_flatten', 'nir_selection_control_none',
    'nir_serialize', 'nir_serialize_function', 'nir_shader',
    'nir_shader_add_variable', 'nir_shader_alu_pass',
    'nir_shader_as_str', 'nir_shader_as_str_annotated',
    'nir_shader_clear_pass_flags', 'nir_shader_clone',
    'nir_shader_compiler_options', 'nir_shader_create',
    'nir_shader_gather_debug_info', 'nir_shader_gather_info',
    'nir_shader_get_entrypoint', 'nir_shader_get_function_for_name',
    'nir_shader_get_preamble', 'nir_shader_index_vars',
    'nir_shader_instructions_pass', 'nir_shader_intrinsics_pass',
    'nir_shader_lower_instructions', 'nir_shader_phi_pass',
    'nir_shader_preserve_all_metadata', 'nir_shader_replace',
    'nir_shader_serialize_deserialize',
    'nir_shader_supports_implicit_lod', 'nir_shader_tex_pass',
    'nir_shader_uses_view_index', 'nir_shift_channels',
    'nir_should_vectorize_mem_func', 'nir_shrink_vec_array_vars',
    'nir_slot_is_sysval_output',
    'nir_slot_is_sysval_output_and_varying', 'nir_slot_is_varying',
    'nir_sort_unstructured_blocks', 'nir_sort_variables_by_location',
    'nir_sort_variables_with_modes', 'nir_split_64bit_vec3_and_vec4',
    'nir_split_array_vars', 'nir_split_conversions',
    'nir_split_conversions_options', 'nir_split_per_member_structs',
    'nir_split_struct_vars', 'nir_split_var_copies', 'nir_src',
    'nir_src_as_alu_instr', 'nir_src_as_bool',
    'nir_src_as_const_value', 'nir_src_as_deref', 'nir_src_as_float',
    'nir_src_as_int', 'nir_src_as_intrinsic', 'nir_src_as_string',
    'nir_src_as_uint', 'nir_src_bit_size', 'nir_src_comp_as_bool',
    'nir_src_comp_as_float', 'nir_src_comp_as_int',
    'nir_src_comp_as_uint', 'nir_src_components_read',
    'nir_src_for_ssa', 'nir_src_get_block', 'nir_src_init',
    'nir_src_is_always_uniform', 'nir_src_is_const',
    'nir_src_is_divergent', 'nir_src_is_if', 'nir_src_is_undef',
    'nir_src_num_components', 'nir_src_parent_if',
    'nir_src_parent_instr', 'nir_src_rewrite',
    'nir_src_set_parent_if', 'nir_src_set_parent_instr',
    'nir_srcs_equal', 'nir_ssa_alu_instr_src_components',
    'nir_ssa_for_alu_src', 'nir_start_block', 'nir_state_slot',
    'nir_state_variable_create', 'nir_static_workgroup_size',
    'nir_steal_tex_deref', 'nir_steal_tex_src', 'nir_store_array_var',
    'nir_store_array_var_imm', 'nir_store_deref',
    'nir_store_deref_with_access', 'nir_store_global',
    'nir_store_reg', 'nir_store_reg_for_def', 'nir_store_var',
    'nir_sweep', 'nir_swizzle', 'nir_system_value_from_intrinsic',
    'nir_test_mask', 'nir_tex_deref', 'nir_tex_instr',
    'nir_tex_instr_add_src', 'nir_tex_instr_create',
    'nir_tex_instr_dest_size',
    'nir_tex_instr_has_explicit_tg4_offsets',
    'nir_tex_instr_has_implicit_derivative', 'nir_tex_instr_is_query',
    'nir_tex_instr_need_sampler', 'nir_tex_instr_remove_src',
    'nir_tex_instr_result_size', 'nir_tex_instr_src_index',
    'nir_tex_instr_src_size', 'nir_tex_instr_src_type',
    'nir_tex_pass_cb', 'nir_tex_src', 'nir_tex_src_backend1',
    'nir_tex_src_backend2', 'nir_tex_src_bias',
    'nir_tex_src_comparator', 'nir_tex_src_coord', 'nir_tex_src_ddx',
    'nir_tex_src_ddy', 'nir_tex_src_for_ssa', 'nir_tex_src_lod',
    'nir_tex_src_lod_bias_min_agx', 'nir_tex_src_min_lod',
    'nir_tex_src_ms_index', 'nir_tex_src_ms_mcs_intel',
    'nir_tex_src_offset', 'nir_tex_src_plane',
    'nir_tex_src_projector', 'nir_tex_src_sampler_deref',
    'nir_tex_src_sampler_deref_intrinsic',
    'nir_tex_src_sampler_handle', 'nir_tex_src_sampler_offset',
    'nir_tex_src_texture_deref',
    'nir_tex_src_texture_deref_intrinsic',
    'nir_tex_src_texture_handle', 'nir_tex_src_texture_offset',
    'nir_tex_src_type', 'nir_tex_src_type_constraint',
    'nir_tex_src_type_constraints', 'nir_tex_type_has_lod',
    'nir_texop', 'nir_texop_custom_border_color_agx',
    'nir_texop_descriptor_amd', 'nir_texop_fragment_fetch_amd',
    'nir_texop_fragment_mask_fetch_amd',
    'nir_texop_has_custom_border_color_agx', 'nir_texop_hdr_dim_nv',
    'nir_texop_image_min_lod_agx', 'nir_texop_lod',
    'nir_texop_lod_bias', 'nir_texop_query_levels',
    'nir_texop_sampler_descriptor_amd', 'nir_texop_samples_identical',
    'nir_texop_tex', 'nir_texop_tex_prefetch',
    'nir_texop_tex_type_nv', 'nir_texop_texture_samples',
    'nir_texop_tg4', 'nir_texop_txb', 'nir_texop_txd',
    'nir_texop_txf', 'nir_texop_txf_ms', 'nir_texop_txf_ms_fb',
    'nir_texop_txf_ms_mcs_intel', 'nir_texop_txl', 'nir_texop_txs',
    'nir_trim_vector', 'nir_trivialize_registers', 'nir_txf_deref',
    'nir_txf_ms_deref', 'nir_txl_deref', 'nir_txl_zero_deref',
    'nir_txs_deref', 'nir_type_bool', 'nir_type_bool1',
    'nir_type_bool16', 'nir_type_bool32', 'nir_type_bool8',
    'nir_type_conversion_op', 'nir_type_convert', 'nir_type_float',
    'nir_type_float16', 'nir_type_float32', 'nir_type_float64',
    'nir_type_int', 'nir_type_int1', 'nir_type_int16',
    'nir_type_int32', 'nir_type_int64', 'nir_type_int8',
    'nir_type_invalid', 'nir_type_uint', 'nir_type_uint1',
    'nir_type_uint16', 'nir_type_uint32', 'nir_type_uint64',
    'nir_type_uint8', 'nir_u2fN', 'nir_u2uN', 'nir_ubfe_imm',
    'nir_ubitfield_extract_imm', 'nir_uclamp', 'nir_udiv_imm',
    'nir_umax_imm', 'nir_umin_imm', 'nir_umod_imm', 'nir_undef',
    'nir_undef_instr', 'nir_undef_instr_create', 'nir_unpack_bits',
    'nir_unsigned_upper_bound', 'nir_unsigned_upper_bound_config',
    'nir_unstructured_start_block', 'nir_use_dominance_lca',
    'nir_use_dominance_state', 'nir_ushr_imm', 'nir_validate_shader',
    'nir_validate_ssa_dominance', 'nir_var_all',
    'nir_var_declaration_type',
    'nir_var_declaration_type__enumvalues',
    'nir_var_declared_implicitly', 'nir_var_declared_normally',
    'nir_var_function_in', 'nir_var_function_inout',
    'nir_var_function_out', 'nir_var_function_temp', 'nir_var_hidden',
    'nir_var_image', 'nir_var_mem_constant', 'nir_var_mem_generic',
    'nir_var_mem_global', 'nir_var_mem_node_payload',
    'nir_var_mem_node_payload_in', 'nir_var_mem_push_const',
    'nir_var_mem_shared', 'nir_var_mem_ssbo',
    'nir_var_mem_task_payload', 'nir_var_mem_ubo',
    'nir_var_ray_hit_attrib', 'nir_var_read_only_modes',
    'nir_var_shader_call_data', 'nir_var_shader_in',
    'nir_var_shader_out', 'nir_var_shader_temp',
    'nir_var_system_value', 'nir_var_uniform',
    'nir_var_vec_indexable_modes', 'nir_variable',
    'nir_variable_clone', 'nir_variable_count_slots',
    'nir_variable_create', 'nir_variable_data',
    'nir_variable_is_global', 'nir_variable_is_in_block',
    'nir_variable_is_in_ssbo', 'nir_variable_is_in_ubo',
    'nir_variable_mode', 'nir_variable_mode__enumvalues', 'nir_vec',
    'nir_vec_scalars', 'nir_vector_extract', 'nir_vector_insert',
    'nir_vector_insert_imm', 'nir_vectorize_cb',
    'nir_vertex_divergence_analysis', 'nir_verts_in_output_prim',
    'nir_zero_initialize_shared_memory', 'nv_device_type',
    'nv_device_uuid', 'pipe_format', 'pipe_shader_type',
    'ralloc_adopt', 'ralloc_array_size', 'ralloc_asprintf',
    'ralloc_asprintf_append', 'ralloc_asprintf_rewrite_tail',
    'ralloc_context', 'ralloc_free', 'ralloc_memdup', 'ralloc_parent',
    'ralloc_parent_of_linear_context', 'ralloc_print_info',
    'ralloc_set_destructor', 'ralloc_size', 'ralloc_steal',
    'ralloc_steal_linear_context', 'ralloc_str_append',
    'ralloc_strcat', 'ralloc_strdup', 'ralloc_strncat',
    'ralloc_strndup', 'ralloc_total_size', 'ralloc_vasprintf',
    'ralloc_vasprintf_append', 'ralloc_vasprintf_rewrite_tail',
    'reralloc_array_size', 'reralloc_size', 'rerzalloc_array_size',
    'rerzalloc_size', 'rzalloc_array_size', 'rzalloc_size',
    'should_print_nir', 'should_skip_nir', 'size_t',
    'struct_LLVMOpaqueBasicBlock', 'struct_LLVMOpaqueBuilder',
    'struct_LLVMOpaqueContext', 'struct_LLVMOpaqueDIBuilder',
    'struct_LLVMOpaqueExecutionEngine',
    'struct_LLVMOpaqueMCJITMemoryManager',
    'struct_LLVMOpaqueMetadata', 'struct_LLVMOpaqueModule',
    'struct_LLVMOpaqueTargetData',
    'struct_LLVMOpaqueTargetLibraryInfotData',
    'struct_LLVMOpaqueTargetMachine', 'struct_LLVMOpaqueType',
    'struct_LLVMOpaqueValue', 'struct__IO_FILE', 'struct__IO_codecvt',
    'struct__IO_marker', 'struct__IO_wide_data',
    'struct___va_list_tag', 'struct_blob', 'struct_blob_reader',
    'struct_c__SA_linear_opts',
    'struct_c__SA_nir_input_to_output_deps',
    'struct_c__SA_nir_input_to_output_deps_0',
    'struct_c__SA_nir_output_clipper_var_groups',
    'struct_c__SA_nir_output_deps', 'struct_c__SA_nir_output_deps_0',
    'struct_exec_list', 'struct_exec_node', 'struct_gallivm_state',
    'struct_gc_ctx', 'struct_glsl_cmat_description',
    'struct_glsl_struct_field', 'struct_glsl_struct_field_0_0',
    'struct_glsl_type', 'struct_hash_entry', 'struct_hash_table',
    'struct_linear_ctx', 'struct_list_head',
    'struct_lp_bld_tgsi_system_values', 'struct_lp_build_context',
    'struct_lp_build_coro_suspend_info', 'struct_lp_build_fn',
    'struct_lp_build_for_loop_state', 'struct_lp_build_fs_iface',
    'struct_lp_build_gs_iface', 'struct_lp_build_if_state',
    'struct_lp_build_image_soa', 'struct_lp_build_loop_state',
    'struct_lp_build_mask_context', 'struct_lp_build_mesh_iface',
    'struct_lp_build_sampler_aos', 'struct_lp_build_sampler_soa',
    'struct_lp_build_skip_context', 'struct_lp_build_tcs_iface',
    'struct_lp_build_tes_iface', 'struct_lp_build_tgsi_params',
    'struct_lp_cached_code', 'struct_lp_context_ref',
    'struct_lp_derivatives', 'struct_lp_descriptor',
    'struct_lp_descriptor_0_0', 'struct_lp_descriptor_0_1',
    'struct_lp_generated_code', 'struct_lp_img_params',
    'struct_lp_jit_bindless_texture', 'struct_lp_jit_buffer',
    'struct_lp_jit_image', 'struct_lp_jit_resources',
    'struct_lp_jit_sampler', 'struct_lp_jit_texture',
    'struct_lp_jit_texture_0_0', 'struct_lp_passmgr',
    'struct_lp_sampler_dynamic_state', 'struct_lp_sampler_params',
    'struct_lp_sampler_size_query_params',
    'struct_lp_static_texture_state', 'struct_lp_texture_functions',
    'struct_lp_texture_handle', 'struct_lp_texture_handle_state',
    'struct_lp_type', 'struct_nak_compiler', 'struct_nak_fs_key',
    'struct_nak_qmd_cbuf', 'struct_nak_qmd_cbuf_desc_layout',
    'struct_nak_qmd_dispatch_size_layout', 'struct_nak_qmd_info',
    'struct_nak_sample_location', 'struct_nak_sample_mask',
    'struct_nak_shader_bin', 'struct_nak_shader_info',
    'struct_nak_shader_info_0_cs', 'struct_nak_shader_info_0_fs',
    'struct_nak_shader_info_0_ts', 'struct_nak_shader_info_vtg',
    'struct_nak_xfb_info', 'struct_nir_alu_instr',
    'struct_nir_alu_src', 'struct_nir_binding', 'struct_nir_block',
    'struct_nir_builder', 'struct_nir_call_instr',
    'struct_nir_cf_node', 'struct_nir_constant', 'struct_nir_cursor',
    'struct_nir_def', 'struct_nir_deref_instr',
    'struct_nir_deref_instr_1_arr', 'struct_nir_deref_instr_1_cast',
    'struct_nir_deref_instr_1_strct', 'struct_nir_function',
    'struct_nir_function_impl', 'struct_nir_if',
    'struct_nir_input_attachment_options', 'struct_nir_instr',
    'struct_nir_instr_debug_info', 'struct_nir_intrinsic_info',
    'struct_nir_intrinsic_instr', 'struct_nir_io_semantics',
    'struct_nir_io_xfb', 'struct_nir_io_xfb_0',
    'struct_nir_jump_instr', 'struct_nir_load_const_instr',
    'struct_nir_load_store_vectorize_options', 'struct_nir_loop',
    'struct_nir_loop_induction_variable', 'struct_nir_loop_info',
    'struct_nir_loop_terminator', 'struct_nir_lower_bitmap_options',
    'struct_nir_lower_compute_system_values_options',
    'struct_nir_lower_drawpixels_options',
    'struct_nir_lower_idiv_options', 'struct_nir_lower_image_options',
    'struct_nir_lower_mem_access_bit_sizes_options',
    'struct_nir_lower_multiview_options',
    'struct_nir_lower_non_uniform_access_options',
    'struct_nir_lower_printf_options',
    'struct_nir_lower_shader_calls_options',
    'struct_nir_lower_ssbo_options',
    'struct_nir_lower_subgroups_options',
    'struct_nir_lower_sysvals_to_varyings_options',
    'struct_nir_lower_task_shader_options',
    'struct_nir_lower_tex_options',
    'struct_nir_lower_tex_shadow_swizzle',
    'struct_nir_lower_wpos_ytransform_options',
    'struct_nir_mem_access_size_align', 'struct_nir_op_info',
    'struct_nir_opt_16bit_tex_image_options',
    'struct_nir_opt_access_options', 'struct_nir_opt_offsets_options',
    'struct_nir_opt_peephole_select_options',
    'struct_nir_opt_preamble_options',
    'struct_nir_opt_tex_srcs_options',
    'struct_nir_parallel_copy_entry',
    'struct_nir_parallel_copy_instr', 'struct_nir_parameter',
    'struct_nir_phi_instr', 'struct_nir_phi_src',
    'struct_nir_remove_dead_variables_options', 'struct_nir_scalar',
    'struct_nir_shader', 'struct_nir_shader_compiler_options',
    'struct_nir_split_conversions_options', 'struct_nir_src',
    'struct_nir_state_slot', 'struct_nir_tex_instr',
    'struct_nir_tex_src', 'struct_nir_tex_src_type_constraint',
    'struct_nir_undef_instr',
    'struct_nir_unsigned_upper_bound_config',
    'struct_nir_use_dominance_state', 'struct_nir_variable',
    'struct_nir_variable_data', 'struct_nir_variable_data_0_image',
    'struct_nir_variable_data_0_sampler',
    'struct_nir_variable_data_0_xfb', 'struct_nir_xfb_info',
    'struct_nv_device_info', 'struct_nv_device_info_pci',
    'struct_set', 'struct_set_entry', 'struct_shader_info',
    'struct_shader_info_0_cs', 'struct_shader_info_0_fs',
    'struct_shader_info_0_gs', 'struct_shader_info_0_mesh',
    'struct_shader_info_0_tess', 'struct_shader_info_0_vs',
    'struct_tgsi_shader_info', 'struct_u_printf_info',
    'struct_util_format_block',
    'struct_util_format_channel_description',
    'struct_util_format_description', 'tess_primitive_mode',
    'tgsi_texture_type', 'u_printf_info', 'uint16_t', 'uint32_t',
    'uint64_t', 'uint8_t', 'union_c__UA_nir_const_value',
    'union_glsl_struct_field_0', 'union_glsl_type_fields',
    'union_lp_descriptor_0', 'union_lp_jit_buffer_0',
    'union_lp_jit_texture_0', 'union_nak_shader_info_0',
    'union_nir_cursor_0', 'union_nir_deref_instr_0',
    'union_nir_deref_instr_1', 'union_nir_parallel_copy_entry_dest',
    'union_nir_variable_data_0', 'union_shader_info_0',
    'union_util_format_description_0', 'util_format_colorspace',
    'util_format_layout', 'va_list']
lvp_nir_options = gzip.decompress(base64.b64decode('H4sIAAAAAAAAA2NgZGRkYGAAkYxgCsQFsxigwgwQBoxmhCqFq2WEKwIrAEGIkQxoAEMALwCqVsCiGUwLMHA0QPn29nBJkswHANb8YpH4AAAA'))
def __getattr__(nm): raise AttributeError('LLVMpipe requires tinymesa_cpu' if 'tinymesa_cpu' not in dll._name else f'attribute {nm} not found') if dll else FileNotFoundError(f'libtinymesa not found (MESA_PATH={BASE}). See https://github.com/sirhcm/tinymesa (tinymesa-32dc66c, mesa-25.2.4)')
