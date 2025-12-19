# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-D__HIP_PLATFORM_AMD__', '-I/opt/rocm/include', '-x', 'c++']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util, os
PATHS_TO_TRY = [
  '/opt/rocm/lib/libamd_comgr.so',
  os.getenv('ROCM_PATH', '')+'/lib/libamd_comgr.so',
  '/usr/local/lib/libamd_comgr.dylib',
  '/opt/homebrew/lib/libamd_comgr.dylib',
]
def _try_dlopen_amd_comgr():
  library = ctypes.util.find_library("amd_comgr")
  if library: return ctypes.CDLL(library)
  for candidate in PATHS_TO_TRY:
    try: return ctypes.CDLL(candidate)
    except OSError: pass
  return None


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



_libraries = {}
_libraries['libamd_comgr.so'] = _try_dlopen_amd_comgr()
c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

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






# values for enumeration 'amd_comgr_status_s'
amd_comgr_status_s__enumvalues = {
    0: 'AMD_COMGR_STATUS_SUCCESS',
    1: 'AMD_COMGR_STATUS_ERROR',
    2: 'AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT',
    3: 'AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES',
}
AMD_COMGR_STATUS_SUCCESS = 0
AMD_COMGR_STATUS_ERROR = 1
AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = 2
AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = 3
amd_comgr_status_s = ctypes.c_uint32 # enum
amd_comgr_status_t = amd_comgr_status_s
amd_comgr_status_t__enumvalues = amd_comgr_status_s__enumvalues

# values for enumeration 'amd_comgr_language_s'
amd_comgr_language_s__enumvalues = {
    0: 'AMD_COMGR_LANGUAGE_NONE',
    1: 'AMD_COMGR_LANGUAGE_OPENCL_1_2',
    2: 'AMD_COMGR_LANGUAGE_OPENCL_2_0',
    3: 'AMD_COMGR_LANGUAGE_HIP',
    4: 'AMD_COMGR_LANGUAGE_LLVM_IR',
    4: 'AMD_COMGR_LANGUAGE_LAST',
}
AMD_COMGR_LANGUAGE_NONE = 0
AMD_COMGR_LANGUAGE_OPENCL_1_2 = 1
AMD_COMGR_LANGUAGE_OPENCL_2_0 = 2
AMD_COMGR_LANGUAGE_HIP = 3
AMD_COMGR_LANGUAGE_LLVM_IR = 4
AMD_COMGR_LANGUAGE_LAST = 4
amd_comgr_language_s = ctypes.c_uint32 # enum
amd_comgr_language_t = amd_comgr_language_s
amd_comgr_language_t__enumvalues = amd_comgr_language_s__enumvalues
try:
    amd_comgr_status_string = _libraries['libamd_comgr.so'].amd_comgr_status_string
    amd_comgr_status_string.restype = amd_comgr_status_t
    amd_comgr_status_string.argtypes = [amd_comgr_status_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    amd_comgr_get_version = _libraries['libamd_comgr.so'].amd_comgr_get_version
    amd_comgr_get_version.restype = None
    amd_comgr_get_version.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass

# values for enumeration 'amd_comgr_data_kind_s'
amd_comgr_data_kind_s__enumvalues = {
    0: 'AMD_COMGR_DATA_KIND_UNDEF',
    1: 'AMD_COMGR_DATA_KIND_SOURCE',
    2: 'AMD_COMGR_DATA_KIND_INCLUDE',
    3: 'AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER',
    4: 'AMD_COMGR_DATA_KIND_DIAGNOSTIC',
    5: 'AMD_COMGR_DATA_KIND_LOG',
    6: 'AMD_COMGR_DATA_KIND_BC',
    7: 'AMD_COMGR_DATA_KIND_RELOCATABLE',
    8: 'AMD_COMGR_DATA_KIND_EXECUTABLE',
    9: 'AMD_COMGR_DATA_KIND_BYTES',
    16: 'AMD_COMGR_DATA_KIND_FATBIN',
    17: 'AMD_COMGR_DATA_KIND_AR',
    18: 'AMD_COMGR_DATA_KIND_BC_BUNDLE',
    19: 'AMD_COMGR_DATA_KIND_AR_BUNDLE',
    20: 'AMD_COMGR_DATA_KIND_OBJ_BUNDLE',
    21: 'AMD_COMGR_DATA_KIND_SPIRV',
    21: 'AMD_COMGR_DATA_KIND_LAST',
}
AMD_COMGR_DATA_KIND_UNDEF = 0
AMD_COMGR_DATA_KIND_SOURCE = 1
AMD_COMGR_DATA_KIND_INCLUDE = 2
AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER = 3
AMD_COMGR_DATA_KIND_DIAGNOSTIC = 4
AMD_COMGR_DATA_KIND_LOG = 5
AMD_COMGR_DATA_KIND_BC = 6
AMD_COMGR_DATA_KIND_RELOCATABLE = 7
AMD_COMGR_DATA_KIND_EXECUTABLE = 8
AMD_COMGR_DATA_KIND_BYTES = 9
AMD_COMGR_DATA_KIND_FATBIN = 16
AMD_COMGR_DATA_KIND_AR = 17
AMD_COMGR_DATA_KIND_BC_BUNDLE = 18
AMD_COMGR_DATA_KIND_AR_BUNDLE = 19
AMD_COMGR_DATA_KIND_OBJ_BUNDLE = 20
AMD_COMGR_DATA_KIND_SPIRV = 21
AMD_COMGR_DATA_KIND_LAST = 21
amd_comgr_data_kind_s = ctypes.c_uint32 # enum
amd_comgr_data_kind_t = amd_comgr_data_kind_s
amd_comgr_data_kind_t__enumvalues = amd_comgr_data_kind_s__enumvalues
class struct_amd_comgr_data_s(Structure):
    pass

struct_amd_comgr_data_s._pack_ = 1 # source:False
struct_amd_comgr_data_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_data_t = struct_amd_comgr_data_s
class struct_amd_comgr_data_set_s(Structure):
    pass

struct_amd_comgr_data_set_s._pack_ = 1 # source:False
struct_amd_comgr_data_set_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_data_set_t = struct_amd_comgr_data_set_s
class struct_amd_comgr_action_info_s(Structure):
    pass

struct_amd_comgr_action_info_s._pack_ = 1 # source:False
struct_amd_comgr_action_info_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_action_info_t = struct_amd_comgr_action_info_s
class struct_amd_comgr_metadata_node_s(Structure):
    pass

struct_amd_comgr_metadata_node_s._pack_ = 1 # source:False
struct_amd_comgr_metadata_node_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_metadata_node_t = struct_amd_comgr_metadata_node_s
class struct_amd_comgr_symbol_s(Structure):
    pass

struct_amd_comgr_symbol_s._pack_ = 1 # source:False
struct_amd_comgr_symbol_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_symbol_t = struct_amd_comgr_symbol_s
class struct_amd_comgr_disassembly_info_s(Structure):
    pass

struct_amd_comgr_disassembly_info_s._pack_ = 1 # source:False
struct_amd_comgr_disassembly_info_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_disassembly_info_t = struct_amd_comgr_disassembly_info_s
class struct_amd_comgr_symbolizer_info_s(Structure):
    pass

struct_amd_comgr_symbolizer_info_s._pack_ = 1 # source:False
struct_amd_comgr_symbolizer_info_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_symbolizer_info_t = struct_amd_comgr_symbolizer_info_s
try:
    amd_comgr_get_isa_count = _libraries['libamd_comgr.so'].amd_comgr_get_isa_count
    amd_comgr_get_isa_count.restype = amd_comgr_status_t
    amd_comgr_get_isa_count.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    amd_comgr_get_isa_name = _libraries['libamd_comgr.so'].amd_comgr_get_isa_name
    amd_comgr_get_isa_name.restype = amd_comgr_status_t
    amd_comgr_get_isa_name.argtypes = [size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    amd_comgr_get_isa_metadata = _libraries['libamd_comgr.so'].amd_comgr_get_isa_metadata
    amd_comgr_get_isa_metadata.restype = amd_comgr_status_t
    amd_comgr_get_isa_metadata.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_amd_comgr_metadata_node_s)]
except AttributeError:
    pass
try:
    amd_comgr_create_data = _libraries['libamd_comgr.so'].amd_comgr_create_data
    amd_comgr_create_data.restype = amd_comgr_status_t
    amd_comgr_create_data.argtypes = [amd_comgr_data_kind_t, ctypes.POINTER(struct_amd_comgr_data_s)]
except AttributeError:
    pass
try:
    amd_comgr_release_data = _libraries['libamd_comgr.so'].amd_comgr_release_data
    amd_comgr_release_data.restype = amd_comgr_status_t
    amd_comgr_release_data.argtypes = [amd_comgr_data_t]
except AttributeError:
    pass
try:
    amd_comgr_get_data_kind = _libraries['libamd_comgr.so'].amd_comgr_get_data_kind
    amd_comgr_get_data_kind.restype = amd_comgr_status_t
    amd_comgr_get_data_kind.argtypes = [amd_comgr_data_t, ctypes.POINTER(amd_comgr_data_kind_s)]
except AttributeError:
    pass
try:
    amd_comgr_set_data = _libraries['libamd_comgr.so'].amd_comgr_set_data
    amd_comgr_set_data.restype = amd_comgr_status_t
    amd_comgr_set_data.argtypes = [amd_comgr_data_t, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
uint64_t = ctypes.c_uint64
try:
    amd_comgr_set_data_from_file_slice = _libraries['libamd_comgr.so'].amd_comgr_set_data_from_file_slice
    amd_comgr_set_data_from_file_slice.restype = amd_comgr_status_t
    amd_comgr_set_data_from_file_slice.argtypes = [amd_comgr_data_t, ctypes.c_int32, uint64_t, uint64_t]
except AttributeError:
    pass
try:
    amd_comgr_set_data_name = _libraries['libamd_comgr.so'].amd_comgr_set_data_name
    amd_comgr_set_data_name.restype = amd_comgr_status_t
    amd_comgr_set_data_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_get_data = _libraries['libamd_comgr.so'].amd_comgr_get_data
    amd_comgr_get_data.restype = amd_comgr_status_t
    amd_comgr_get_data.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_get_data_name = _libraries['libamd_comgr.so'].amd_comgr_get_data_name
    amd_comgr_get_data_name.restype = amd_comgr_status_t
    amd_comgr_get_data_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_get_data_isa_name = _libraries['libamd_comgr.so'].amd_comgr_get_data_isa_name
    amd_comgr_get_data_isa_name.restype = amd_comgr_status_t
    amd_comgr_get_data_isa_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_create_symbolizer_info = _libraries['libamd_comgr.so'].amd_comgr_create_symbolizer_info
    amd_comgr_create_symbolizer_info.restype = amd_comgr_status_t
    amd_comgr_create_symbolizer_info.argtypes = [amd_comgr_data_t, ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)), ctypes.POINTER(struct_amd_comgr_symbolizer_info_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_symbolizer_info = _libraries['libamd_comgr.so'].amd_comgr_destroy_symbolizer_info
    amd_comgr_destroy_symbolizer_info.restype = amd_comgr_status_t
    amd_comgr_destroy_symbolizer_info.argtypes = [amd_comgr_symbolizer_info_t]
except AttributeError:
    pass
try:
    amd_comgr_symbolize = _libraries['libamd_comgr.so'].amd_comgr_symbolize
    amd_comgr_symbolize.restype = amd_comgr_status_t
    amd_comgr_symbolize.argtypes = [amd_comgr_symbolizer_info_t, uint64_t, ctypes.c_bool, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    amd_comgr_get_data_metadata = _libraries['libamd_comgr.so'].amd_comgr_get_data_metadata
    amd_comgr_get_data_metadata.restype = amd_comgr_status_t
    amd_comgr_get_data_metadata.argtypes = [amd_comgr_data_t, ctypes.POINTER(struct_amd_comgr_metadata_node_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_metadata = _libraries['libamd_comgr.so'].amd_comgr_destroy_metadata
    amd_comgr_destroy_metadata.restype = amd_comgr_status_t
    amd_comgr_destroy_metadata.argtypes = [amd_comgr_metadata_node_t]
except AttributeError:
    pass
try:
    amd_comgr_create_data_set = _libraries['libamd_comgr.so'].amd_comgr_create_data_set
    amd_comgr_create_data_set.restype = amd_comgr_status_t
    amd_comgr_create_data_set.argtypes = [ctypes.POINTER(struct_amd_comgr_data_set_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_data_set = _libraries['libamd_comgr.so'].amd_comgr_destroy_data_set
    amd_comgr_destroy_data_set.restype = amd_comgr_status_t
    amd_comgr_destroy_data_set.argtypes = [amd_comgr_data_set_t]
except AttributeError:
    pass
try:
    amd_comgr_data_set_add = _libraries['libamd_comgr.so'].amd_comgr_data_set_add
    amd_comgr_data_set_add.restype = amd_comgr_status_t
    amd_comgr_data_set_add.argtypes = [amd_comgr_data_set_t, amd_comgr_data_t]
except AttributeError:
    pass
try:
    amd_comgr_data_set_remove = _libraries['libamd_comgr.so'].amd_comgr_data_set_remove
    amd_comgr_data_set_remove.restype = amd_comgr_status_t
    amd_comgr_data_set_remove.argtypes = [amd_comgr_data_set_t, amd_comgr_data_kind_t]
except AttributeError:
    pass
try:
    amd_comgr_action_data_count = _libraries['libamd_comgr.so'].amd_comgr_action_data_count
    amd_comgr_action_data_count.restype = amd_comgr_status_t
    amd_comgr_action_data_count.argtypes = [amd_comgr_data_set_t, amd_comgr_data_kind_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_action_data_get_data = _libraries['libamd_comgr.so'].amd_comgr_action_data_get_data
    amd_comgr_action_data_get_data.restype = amd_comgr_status_t
    amd_comgr_action_data_get_data.argtypes = [amd_comgr_data_set_t, amd_comgr_data_kind_t, size_t, ctypes.POINTER(struct_amd_comgr_data_s)]
except AttributeError:
    pass
try:
    amd_comgr_create_action_info = _libraries['libamd_comgr.so'].amd_comgr_create_action_info
    amd_comgr_create_action_info.restype = amd_comgr_status_t
    amd_comgr_create_action_info.argtypes = [ctypes.POINTER(struct_amd_comgr_action_info_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_action_info = _libraries['libamd_comgr.so'].amd_comgr_destroy_action_info
    amd_comgr_destroy_action_info.restype = amd_comgr_status_t
    amd_comgr_destroy_action_info.argtypes = [amd_comgr_action_info_t]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_isa_name = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_isa_name
    amd_comgr_action_info_set_isa_name.restype = amd_comgr_status_t
    amd_comgr_action_info_set_isa_name.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_isa_name = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_isa_name
    amd_comgr_action_info_get_isa_name.restype = amd_comgr_status_t
    amd_comgr_action_info_get_isa_name.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_language = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_language
    amd_comgr_action_info_set_language.restype = amd_comgr_status_t
    amd_comgr_action_info_set_language.argtypes = [amd_comgr_action_info_t, amd_comgr_language_t]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_language = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_language
    amd_comgr_action_info_get_language.restype = amd_comgr_status_t
    amd_comgr_action_info_get_language.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(amd_comgr_language_s)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_option_list = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_option_list
    amd_comgr_action_info_set_option_list.restype = amd_comgr_status_t
    amd_comgr_action_info_set_option_list.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char) * 0, size_t]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_option_list_count = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_option_list_count
    amd_comgr_action_info_get_option_list_count.restype = amd_comgr_status_t
    amd_comgr_action_info_get_option_list_count.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_option_list_item = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_option_list_item
    amd_comgr_action_info_get_option_list_item.restype = amd_comgr_status_t
    amd_comgr_action_info_get_option_list_item.argtypes = [amd_comgr_action_info_t, size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_bundle_entry_ids = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_bundle_entry_ids
    amd_comgr_action_info_set_bundle_entry_ids.restype = amd_comgr_status_t
    amd_comgr_action_info_set_bundle_entry_ids.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char) * 0, size_t]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_bundle_entry_id_count = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_bundle_entry_id_count
    amd_comgr_action_info_get_bundle_entry_id_count.restype = amd_comgr_status_t
    amd_comgr_action_info_get_bundle_entry_id_count.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_bundle_entry_id = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_bundle_entry_id
    amd_comgr_action_info_get_bundle_entry_id.restype = amd_comgr_status_t
    amd_comgr_action_info_get_bundle_entry_id.argtypes = [amd_comgr_action_info_t, size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_device_lib_linking = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_device_lib_linking
    amd_comgr_action_info_set_device_lib_linking.restype = amd_comgr_status_t
    amd_comgr_action_info_set_device_lib_linking.argtypes = [amd_comgr_action_info_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_working_directory_path = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_working_directory_path
    amd_comgr_action_info_set_working_directory_path.restype = amd_comgr_status_t
    amd_comgr_action_info_set_working_directory_path.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_working_directory_path = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_working_directory_path
    amd_comgr_action_info_get_working_directory_path.restype = amd_comgr_status_t
    amd_comgr_action_info_get_working_directory_path.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_logging = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_logging
    amd_comgr_action_info_set_logging.restype = amd_comgr_status_t
    amd_comgr_action_info_set_logging.argtypes = [amd_comgr_action_info_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_logging = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_logging
    amd_comgr_action_info_get_logging.restype = amd_comgr_status_t
    amd_comgr_action_info_get_logging.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass

# values for enumeration 'amd_comgr_action_kind_s'
amd_comgr_action_kind_s__enumvalues = {
    0: 'AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR',
    1: 'AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS',
    2: 'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC',
    3: 'AMD_COMGR_ACTION_LINK_BC_TO_BC',
    4: 'AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE',
    5: 'AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY',
    6: 'AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE',
    7: 'AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE',
    8: 'AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE',
    9: 'AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE',
    10: 'AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE',
    11: 'AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE',
    12: 'AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC',
    13: 'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE',
    14: 'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE',
    15: 'AMD_COMGR_ACTION_UNBUNDLE',
    19: 'AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC',
    19: 'AMD_COMGR_ACTION_LAST',
}
AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR = 0
AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS = 1
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = 2
AMD_COMGR_ACTION_LINK_BC_TO_BC = 3
AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 4
AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY = 5
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE = 6
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 7
AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE = 8
AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE = 9
AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE = 10
AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE = 11
AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 12
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE = 13
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE = 14
AMD_COMGR_ACTION_UNBUNDLE = 15
AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC = 19
AMD_COMGR_ACTION_LAST = 19
amd_comgr_action_kind_s = ctypes.c_uint32 # enum
amd_comgr_action_kind_t = amd_comgr_action_kind_s
amd_comgr_action_kind_t__enumvalues = amd_comgr_action_kind_s__enumvalues
try:
    amd_comgr_do_action = _libraries['libamd_comgr.so'].amd_comgr_do_action
    amd_comgr_do_action.restype = amd_comgr_status_t
    amd_comgr_do_action.argtypes = [amd_comgr_action_kind_t, amd_comgr_action_info_t, amd_comgr_data_set_t, amd_comgr_data_set_t]
except AttributeError:
    pass

# values for enumeration 'amd_comgr_metadata_kind_s'
amd_comgr_metadata_kind_s__enumvalues = {
    0: 'AMD_COMGR_METADATA_KIND_NULL',
    1: 'AMD_COMGR_METADATA_KIND_STRING',
    2: 'AMD_COMGR_METADATA_KIND_MAP',
    3: 'AMD_COMGR_METADATA_KIND_LIST',
    3: 'AMD_COMGR_METADATA_KIND_LAST',
}
AMD_COMGR_METADATA_KIND_NULL = 0
AMD_COMGR_METADATA_KIND_STRING = 1
AMD_COMGR_METADATA_KIND_MAP = 2
AMD_COMGR_METADATA_KIND_LIST = 3
AMD_COMGR_METADATA_KIND_LAST = 3
amd_comgr_metadata_kind_s = ctypes.c_uint32 # enum
amd_comgr_metadata_kind_t = amd_comgr_metadata_kind_s
amd_comgr_metadata_kind_t__enumvalues = amd_comgr_metadata_kind_s__enumvalues
try:
    amd_comgr_get_metadata_kind = _libraries['libamd_comgr.so'].amd_comgr_get_metadata_kind
    amd_comgr_get_metadata_kind.restype = amd_comgr_status_t
    amd_comgr_get_metadata_kind.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(amd_comgr_metadata_kind_s)]
except AttributeError:
    pass
try:
    amd_comgr_get_metadata_string = _libraries['libamd_comgr.so'].amd_comgr_get_metadata_string
    amd_comgr_get_metadata_string.restype = amd_comgr_status_t
    amd_comgr_get_metadata_string.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_get_metadata_map_size = _libraries['libamd_comgr.so'].amd_comgr_get_metadata_map_size
    amd_comgr_get_metadata_map_size.restype = amd_comgr_status_t
    amd_comgr_get_metadata_map_size.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_iterate_map_metadata = _libraries['libamd_comgr.so'].amd_comgr_iterate_map_metadata
    amd_comgr_iterate_map_metadata.restype = amd_comgr_status_t
    amd_comgr_iterate_map_metadata.argtypes = [amd_comgr_metadata_node_t, ctypes.CFUNCTYPE(amd_comgr_status_s, struct_amd_comgr_metadata_node_s, struct_amd_comgr_metadata_node_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    amd_comgr_metadata_lookup = _libraries['libamd_comgr.so'].amd_comgr_metadata_lookup
    amd_comgr_metadata_lookup.restype = amd_comgr_status_t
    amd_comgr_metadata_lookup.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_amd_comgr_metadata_node_s)]
except AttributeError:
    pass
try:
    amd_comgr_get_metadata_list_size = _libraries['libamd_comgr.so'].amd_comgr_get_metadata_list_size
    amd_comgr_get_metadata_list_size.restype = amd_comgr_status_t
    amd_comgr_get_metadata_list_size.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_index_list_metadata = _libraries['libamd_comgr.so'].amd_comgr_index_list_metadata
    amd_comgr_index_list_metadata.restype = amd_comgr_status_t
    amd_comgr_index_list_metadata.argtypes = [amd_comgr_metadata_node_t, size_t, ctypes.POINTER(struct_amd_comgr_metadata_node_s)]
except AttributeError:
    pass
try:
    amd_comgr_iterate_symbols = _libraries['libamd_comgr.so'].amd_comgr_iterate_symbols
    amd_comgr_iterate_symbols.restype = amd_comgr_status_t
    amd_comgr_iterate_symbols.argtypes = [amd_comgr_data_t, ctypes.CFUNCTYPE(amd_comgr_status_s, struct_amd_comgr_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    amd_comgr_symbol_lookup = _libraries['libamd_comgr.so'].amd_comgr_symbol_lookup
    amd_comgr_symbol_lookup.restype = amd_comgr_status_t
    amd_comgr_symbol_lookup.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_amd_comgr_symbol_s)]
except AttributeError:
    pass

# values for enumeration 'amd_comgr_symbol_type_s'
amd_comgr_symbol_type_s__enumvalues = {
    -1: 'AMD_COMGR_SYMBOL_TYPE_UNKNOWN',
    0: 'AMD_COMGR_SYMBOL_TYPE_NOTYPE',
    1: 'AMD_COMGR_SYMBOL_TYPE_OBJECT',
    2: 'AMD_COMGR_SYMBOL_TYPE_FUNC',
    3: 'AMD_COMGR_SYMBOL_TYPE_SECTION',
    4: 'AMD_COMGR_SYMBOL_TYPE_FILE',
    5: 'AMD_COMGR_SYMBOL_TYPE_COMMON',
    10: 'AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL',
}
AMD_COMGR_SYMBOL_TYPE_UNKNOWN = -1
AMD_COMGR_SYMBOL_TYPE_NOTYPE = 0
AMD_COMGR_SYMBOL_TYPE_OBJECT = 1
AMD_COMGR_SYMBOL_TYPE_FUNC = 2
AMD_COMGR_SYMBOL_TYPE_SECTION = 3
AMD_COMGR_SYMBOL_TYPE_FILE = 4
AMD_COMGR_SYMBOL_TYPE_COMMON = 5
AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL = 10
amd_comgr_symbol_type_s = ctypes.c_int32 # enum
amd_comgr_symbol_type_t = amd_comgr_symbol_type_s
amd_comgr_symbol_type_t__enumvalues = amd_comgr_symbol_type_s__enumvalues

# values for enumeration 'amd_comgr_symbol_info_s'
amd_comgr_symbol_info_s__enumvalues = {
    0: 'AMD_COMGR_SYMBOL_INFO_NAME_LENGTH',
    1: 'AMD_COMGR_SYMBOL_INFO_NAME',
    2: 'AMD_COMGR_SYMBOL_INFO_TYPE',
    3: 'AMD_COMGR_SYMBOL_INFO_SIZE',
    4: 'AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED',
    5: 'AMD_COMGR_SYMBOL_INFO_VALUE',
    5: 'AMD_COMGR_SYMBOL_INFO_LAST',
}
AMD_COMGR_SYMBOL_INFO_NAME_LENGTH = 0
AMD_COMGR_SYMBOL_INFO_NAME = 1
AMD_COMGR_SYMBOL_INFO_TYPE = 2
AMD_COMGR_SYMBOL_INFO_SIZE = 3
AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED = 4
AMD_COMGR_SYMBOL_INFO_VALUE = 5
AMD_COMGR_SYMBOL_INFO_LAST = 5
amd_comgr_symbol_info_s = ctypes.c_uint32 # enum
amd_comgr_symbol_info_t = amd_comgr_symbol_info_s
amd_comgr_symbol_info_t__enumvalues = amd_comgr_symbol_info_s__enumvalues
try:
    amd_comgr_symbol_get_info = _libraries['libamd_comgr.so'].amd_comgr_symbol_get_info
    amd_comgr_symbol_get_info.restype = amd_comgr_status_t
    amd_comgr_symbol_get_info.argtypes = [amd_comgr_symbol_t, amd_comgr_symbol_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    amd_comgr_create_disassembly_info = _libraries['libamd_comgr.so'].amd_comgr_create_disassembly_info
    amd_comgr_create_disassembly_info.restype = amd_comgr_status_t
    amd_comgr_create_disassembly_info.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char), ctypes.c_uint64, ctypes.POINTER(None)), ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)), ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.POINTER(None)), ctypes.POINTER(struct_amd_comgr_disassembly_info_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_disassembly_info = _libraries['libamd_comgr.so'].amd_comgr_destroy_disassembly_info
    amd_comgr_destroy_disassembly_info.restype = amd_comgr_status_t
    amd_comgr_destroy_disassembly_info.argtypes = [amd_comgr_disassembly_info_t]
except AttributeError:
    pass
try:
    amd_comgr_disassemble_instruction = _libraries['libamd_comgr.so'].amd_comgr_disassemble_instruction
    amd_comgr_disassemble_instruction.restype = amd_comgr_status_t
    amd_comgr_disassemble_instruction.argtypes = [amd_comgr_disassembly_info_t, uint64_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_demangle_symbol_name = _libraries['libamd_comgr.so'].amd_comgr_demangle_symbol_name
    amd_comgr_demangle_symbol_name.restype = amd_comgr_status_t
    amd_comgr_demangle_symbol_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(struct_amd_comgr_data_s)]
except AttributeError:
    pass
try:
    amd_comgr_populate_mangled_names = _libraries['libamd_comgr.so'].amd_comgr_populate_mangled_names
    amd_comgr_populate_mangled_names.restype = amd_comgr_status_t
    amd_comgr_populate_mangled_names.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_get_mangled_name = _libraries['libamd_comgr.so'].amd_comgr_get_mangled_name
    amd_comgr_get_mangled_name.restype = amd_comgr_status_t
    amd_comgr_get_mangled_name.argtypes = [amd_comgr_data_t, size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_populate_name_expression_map = _libraries['libamd_comgr.so'].amd_comgr_populate_name_expression_map
    amd_comgr_populate_name_expression_map.restype = amd_comgr_status_t
    amd_comgr_populate_name_expression_map.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_map_name_expression_to_symbol_name = _libraries['libamd_comgr.so'].amd_comgr_map_name_expression_to_symbol_name
    amd_comgr_map_name_expression_to_symbol_name.restype = amd_comgr_status_t
    amd_comgr_map_name_expression_to_symbol_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
class struct_code_object_info_s(Structure):
    pass

struct_code_object_info_s._pack_ = 1 # source:False
struct_code_object_info_s._fields_ = [
    ('isa', ctypes.POINTER(ctypes.c_char)),
    ('size', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
]

amd_comgr_code_object_info_t = struct_code_object_info_s
try:
    amd_comgr_lookup_code_object = _libraries['libamd_comgr.so'].amd_comgr_lookup_code_object
    amd_comgr_lookup_code_object.restype = amd_comgr_status_t
    amd_comgr_lookup_code_object.argtypes = [amd_comgr_data_t, ctypes.POINTER(struct_code_object_info_s), size_t]
except AttributeError:
    pass
try:
    amd_comgr_map_elf_virtual_address_to_code_object_offset = _libraries['libamd_comgr.so'].amd_comgr_map_elf_virtual_address_to_code_object_offset
    amd_comgr_map_elf_virtual_address_to_code_object_offset.restype = amd_comgr_status_t
    amd_comgr_map_elf_virtual_address_to_code_object_offset.argtypes = [amd_comgr_data_t, uint64_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
__all__ = \
    ['AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS',
    'AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE',
    'AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY',
    'AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE',
    'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC',
    'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE',
    'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE',
    'AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC',
    'AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE',
    'AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE',
    'AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE',
    'AMD_COMGR_ACTION_LAST', 'AMD_COMGR_ACTION_LINK_BC_TO_BC',
    'AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE',
    'AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE',
    'AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR',
    'AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC',
    'AMD_COMGR_ACTION_UNBUNDLE', 'AMD_COMGR_DATA_KIND_AR',
    'AMD_COMGR_DATA_KIND_AR_BUNDLE', 'AMD_COMGR_DATA_KIND_BC',
    'AMD_COMGR_DATA_KIND_BC_BUNDLE', 'AMD_COMGR_DATA_KIND_BYTES',
    'AMD_COMGR_DATA_KIND_DIAGNOSTIC',
    'AMD_COMGR_DATA_KIND_EXECUTABLE', 'AMD_COMGR_DATA_KIND_FATBIN',
    'AMD_COMGR_DATA_KIND_INCLUDE', 'AMD_COMGR_DATA_KIND_LAST',
    'AMD_COMGR_DATA_KIND_LOG', 'AMD_COMGR_DATA_KIND_OBJ_BUNDLE',
    'AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER',
    'AMD_COMGR_DATA_KIND_RELOCATABLE', 'AMD_COMGR_DATA_KIND_SOURCE',
    'AMD_COMGR_DATA_KIND_SPIRV', 'AMD_COMGR_DATA_KIND_UNDEF',
    'AMD_COMGR_LANGUAGE_HIP', 'AMD_COMGR_LANGUAGE_LAST',
    'AMD_COMGR_LANGUAGE_LLVM_IR', 'AMD_COMGR_LANGUAGE_NONE',
    'AMD_COMGR_LANGUAGE_OPENCL_1_2', 'AMD_COMGR_LANGUAGE_OPENCL_2_0',
    'AMD_COMGR_METADATA_KIND_LAST', 'AMD_COMGR_METADATA_KIND_LIST',
    'AMD_COMGR_METADATA_KIND_MAP', 'AMD_COMGR_METADATA_KIND_NULL',
    'AMD_COMGR_METADATA_KIND_STRING', 'AMD_COMGR_STATUS_ERROR',
    'AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT',
    'AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES',
    'AMD_COMGR_STATUS_SUCCESS', 'AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED',
    'AMD_COMGR_SYMBOL_INFO_LAST', 'AMD_COMGR_SYMBOL_INFO_NAME',
    'AMD_COMGR_SYMBOL_INFO_NAME_LENGTH', 'AMD_COMGR_SYMBOL_INFO_SIZE',
    'AMD_COMGR_SYMBOL_INFO_TYPE', 'AMD_COMGR_SYMBOL_INFO_VALUE',
    'AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL',
    'AMD_COMGR_SYMBOL_TYPE_COMMON', 'AMD_COMGR_SYMBOL_TYPE_FILE',
    'AMD_COMGR_SYMBOL_TYPE_FUNC', 'AMD_COMGR_SYMBOL_TYPE_NOTYPE',
    'AMD_COMGR_SYMBOL_TYPE_OBJECT', 'AMD_COMGR_SYMBOL_TYPE_SECTION',
    'AMD_COMGR_SYMBOL_TYPE_UNKNOWN', 'amd_comgr_action_data_count',
    'amd_comgr_action_data_get_data',
    'amd_comgr_action_info_get_bundle_entry_id',
    'amd_comgr_action_info_get_bundle_entry_id_count',
    'amd_comgr_action_info_get_isa_name',
    'amd_comgr_action_info_get_language',
    'amd_comgr_action_info_get_logging',
    'amd_comgr_action_info_get_option_list_count',
    'amd_comgr_action_info_get_option_list_item',
    'amd_comgr_action_info_get_working_directory_path',
    'amd_comgr_action_info_set_bundle_entry_ids',
    'amd_comgr_action_info_set_device_lib_linking',
    'amd_comgr_action_info_set_isa_name',
    'amd_comgr_action_info_set_language',
    'amd_comgr_action_info_set_logging',
    'amd_comgr_action_info_set_option_list',
    'amd_comgr_action_info_set_working_directory_path',
    'amd_comgr_action_info_t', 'amd_comgr_action_kind_s',
    'amd_comgr_action_kind_t', 'amd_comgr_action_kind_t__enumvalues',
    'amd_comgr_code_object_info_t', 'amd_comgr_create_action_info',
    'amd_comgr_create_data', 'amd_comgr_create_data_set',
    'amd_comgr_create_disassembly_info',
    'amd_comgr_create_symbolizer_info', 'amd_comgr_data_kind_s',
    'amd_comgr_data_kind_t', 'amd_comgr_data_kind_t__enumvalues',
    'amd_comgr_data_set_add', 'amd_comgr_data_set_remove',
    'amd_comgr_data_set_t', 'amd_comgr_data_t',
    'amd_comgr_demangle_symbol_name', 'amd_comgr_destroy_action_info',
    'amd_comgr_destroy_data_set',
    'amd_comgr_destroy_disassembly_info',
    'amd_comgr_destroy_metadata', 'amd_comgr_destroy_symbolizer_info',
    'amd_comgr_disassemble_instruction',
    'amd_comgr_disassembly_info_t', 'amd_comgr_do_action',
    'amd_comgr_get_data', 'amd_comgr_get_data_isa_name',
    'amd_comgr_get_data_kind', 'amd_comgr_get_data_metadata',
    'amd_comgr_get_data_name', 'amd_comgr_get_isa_count',
    'amd_comgr_get_isa_metadata', 'amd_comgr_get_isa_name',
    'amd_comgr_get_mangled_name', 'amd_comgr_get_metadata_kind',
    'amd_comgr_get_metadata_list_size',
    'amd_comgr_get_metadata_map_size',
    'amd_comgr_get_metadata_string', 'amd_comgr_get_version',
    'amd_comgr_index_list_metadata', 'amd_comgr_iterate_map_metadata',
    'amd_comgr_iterate_symbols', 'amd_comgr_language_s',
    'amd_comgr_language_t', 'amd_comgr_language_t__enumvalues',
    'amd_comgr_lookup_code_object',
    'amd_comgr_map_elf_virtual_address_to_code_object_offset',
    'amd_comgr_map_name_expression_to_symbol_name',
    'amd_comgr_metadata_kind_s', 'amd_comgr_metadata_kind_t',
    'amd_comgr_metadata_kind_t__enumvalues',
    'amd_comgr_metadata_lookup', 'amd_comgr_metadata_node_t',
    'amd_comgr_populate_mangled_names',
    'amd_comgr_populate_name_expression_map',
    'amd_comgr_release_data', 'amd_comgr_set_data',
    'amd_comgr_set_data_from_file_slice', 'amd_comgr_set_data_name',
    'amd_comgr_status_s', 'amd_comgr_status_string',
    'amd_comgr_status_t', 'amd_comgr_status_t__enumvalues',
    'amd_comgr_symbol_get_info', 'amd_comgr_symbol_info_s',
    'amd_comgr_symbol_info_t', 'amd_comgr_symbol_info_t__enumvalues',
    'amd_comgr_symbol_lookup', 'amd_comgr_symbol_t',
    'amd_comgr_symbol_type_s', 'amd_comgr_symbol_type_t',
    'amd_comgr_symbol_type_t__enumvalues', 'amd_comgr_symbolize',
    'amd_comgr_symbolizer_info_t', 'size_t',
    'struct_amd_comgr_action_info_s', 'struct_amd_comgr_data_s',
    'struct_amd_comgr_data_set_s',
    'struct_amd_comgr_disassembly_info_s',
    'struct_amd_comgr_metadata_node_s', 'struct_amd_comgr_symbol_s',
    'struct_amd_comgr_symbolizer_info_s', 'struct_code_object_info_s',
    'uint64_t']
