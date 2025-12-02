# pylint: skip-file
# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util
PATHS_TO_TRY = [
  '/usr/local/lib/librocprof-trace-decoder.so',
  '/usr/local/lib/librocprof-trace-decoder.dylib',
]
def _try_dlopen_rocprof_trace_decoder():
  library = ctypes.util.find_library("rocprof-trace-decoder")
  if library:
    try: return ctypes.CDLL(library)
    except OSError: pass
  for candidate in PATHS_TO_TRY:
    try: return ctypes.CDLL(candidate)
    except OSError: pass
  return None


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



class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['FIXME_STUB'] = _try_dlopen_rocprof_trace_decoder() #  ctypes.CDLL('FIXME_STUB')



# values for enumeration 'rocprofiler_thread_trace_decoder_info_t'
rocprofiler_thread_trace_decoder_info_t__enumvalues = {
    0: 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE',
    1: 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST',
    2: 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE',
    3: 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE',
    4: 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST',
}
ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE = 0
ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST = 1
ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE = 2
ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE = 3
ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST = 4
rocprofiler_thread_trace_decoder_info_t = ctypes.c_uint32 # enum
class struct_rocprofiler_thread_trace_decoder_pc_t(Structure):
    pass

struct_rocprofiler_thread_trace_decoder_pc_t._pack_ = 1 # source:False
struct_rocprofiler_thread_trace_decoder_pc_t._fields_ = [
    ('address', ctypes.c_uint64),
    ('code_object_id', ctypes.c_uint64),
]

rocprofiler_thread_trace_decoder_pc_t = struct_rocprofiler_thread_trace_decoder_pc_t
class struct_rocprofiler_thread_trace_decoder_perfevent_t(Structure):
    pass

struct_rocprofiler_thread_trace_decoder_perfevent_t._pack_ = 1 # source:False
struct_rocprofiler_thread_trace_decoder_perfevent_t._fields_ = [
    ('time', ctypes.c_int64),
    ('events0', ctypes.c_uint16),
    ('events1', ctypes.c_uint16),
    ('events2', ctypes.c_uint16),
    ('events3', ctypes.c_uint16),
    ('CU', ctypes.c_ubyte),
    ('bank', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 6),
]

rocprofiler_thread_trace_decoder_perfevent_t = struct_rocprofiler_thread_trace_decoder_perfevent_t
class struct_rocprofiler_thread_trace_decoder_occupancy_t(Structure):
    pass

struct_rocprofiler_thread_trace_decoder_occupancy_t._pack_ = 1 # source:False
struct_rocprofiler_thread_trace_decoder_occupancy_t._fields_ = [
    ('pc', rocprofiler_thread_trace_decoder_pc_t),
    ('time', ctypes.c_uint64),
    ('reserved', ctypes.c_ubyte),
    ('cu', ctypes.c_ubyte),
    ('simd', ctypes.c_ubyte),
    ('wave_id', ctypes.c_ubyte),
    ('start', ctypes.c_uint32, 1),
    ('_rsvd', ctypes.c_uint32, 31),
]

rocprofiler_thread_trace_decoder_occupancy_t = struct_rocprofiler_thread_trace_decoder_occupancy_t

# values for enumeration 'rocprofiler_thread_trace_decoder_wstate_type_t'
rocprofiler_thread_trace_decoder_wstate_type_t__enumvalues = {
    0: 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY',
    1: 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE',
    2: 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC',
    3: 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT',
    4: 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL',
    5: 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST',
}
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY = 0
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE = 1
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC = 2
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT = 3
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL = 4
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST = 5
rocprofiler_thread_trace_decoder_wstate_type_t = ctypes.c_uint32 # enum
class struct_rocprofiler_thread_trace_decoder_wave_state_t(Structure):
    pass

struct_rocprofiler_thread_trace_decoder_wave_state_t._pack_ = 1 # source:False
struct_rocprofiler_thread_trace_decoder_wave_state_t._fields_ = [
    ('type', ctypes.c_int32),
    ('duration', ctypes.c_int32),
]

rocprofiler_thread_trace_decoder_wave_state_t = struct_rocprofiler_thread_trace_decoder_wave_state_t

# values for enumeration 'rocprofiler_thread_trace_decoder_inst_category_t'
rocprofiler_thread_trace_decoder_inst_category_t__enumvalues = {
    0: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE',
    1: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM',
    2: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU',
    3: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM',
    4: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT',
    5: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS',
    6: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU',
    7: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP',
    8: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT',
    9: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED',
    10: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT',
    11: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE',
    12: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH',
    13: 'ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST',
}
ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE = 0
ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM = 1
ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU = 2
ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM = 3
ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT = 4
ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS = 5
ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU = 6
ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP = 7
ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT = 8
ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED = 9
ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT = 10
ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE = 11
ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH = 12
ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST = 13
rocprofiler_thread_trace_decoder_inst_category_t = ctypes.c_uint32 # enum
class struct_rocprofiler_thread_trace_decoder_inst_t(Structure):
    pass

struct_rocprofiler_thread_trace_decoder_inst_t._pack_ = 1 # source:False
struct_rocprofiler_thread_trace_decoder_inst_t._fields_ = [
    ('category', ctypes.c_uint32, 8),
    ('stall', ctypes.c_uint32, 24),
    ('duration', ctypes.c_int32),
    ('time', ctypes.c_int64),
    ('pc', rocprofiler_thread_trace_decoder_pc_t),
]

rocprofiler_thread_trace_decoder_inst_t = struct_rocprofiler_thread_trace_decoder_inst_t
class struct_rocprofiler_thread_trace_decoder_wave_t(Structure):
    pass

struct_rocprofiler_thread_trace_decoder_wave_t._pack_ = 1 # source:False
struct_rocprofiler_thread_trace_decoder_wave_t._fields_ = [
    ('cu', ctypes.c_ubyte),
    ('simd', ctypes.c_ubyte),
    ('wave_id', ctypes.c_ubyte),
    ('contexts', ctypes.c_ubyte),
    ('_rsvd1', ctypes.c_uint32),
    ('_rsvd2', ctypes.c_uint32),
    ('_rsvd3', ctypes.c_uint32),
    ('begin_time', ctypes.c_int64),
    ('end_time', ctypes.c_int64),
    ('timeline_size', ctypes.c_uint64),
    ('instructions_size', ctypes.c_uint64),
    ('timeline_array', ctypes.POINTER(struct_rocprofiler_thread_trace_decoder_wave_state_t)),
    ('instructions_array', ctypes.POINTER(struct_rocprofiler_thread_trace_decoder_inst_t)),
]

rocprofiler_thread_trace_decoder_wave_t = struct_rocprofiler_thread_trace_decoder_wave_t
class struct_rocprofiler_thread_trace_decoder_realtime_t(Structure):
    pass

struct_rocprofiler_thread_trace_decoder_realtime_t._pack_ = 1 # source:False
struct_rocprofiler_thread_trace_decoder_realtime_t._fields_ = [
    ('shader_clock', ctypes.c_int64),
    ('realtime_clock', ctypes.c_uint64),
    ('reserved', ctypes.c_uint64),
]

rocprofiler_thread_trace_decoder_realtime_t = struct_rocprofiler_thread_trace_decoder_realtime_t

# values for enumeration 'rocprofiler_thread_trace_decoder_shaderdata_flags_t'
rocprofiler_thread_trace_decoder_shaderdata_flags_t__enumvalues = {
    0: 'ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM',
    1: 'ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV',
}
ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM = 0
ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV = 1
rocprofiler_thread_trace_decoder_shaderdata_flags_t = ctypes.c_uint32 # enum
class struct_rocprofiler_thread_trace_decoder_shaderdata_t(Structure):
    pass

struct_rocprofiler_thread_trace_decoder_shaderdata_t._pack_ = 1 # source:False
struct_rocprofiler_thread_trace_decoder_shaderdata_t._fields_ = [
    ('time', ctypes.c_int64),
    ('value', ctypes.c_uint64),
    ('cu', ctypes.c_ubyte),
    ('simd', ctypes.c_ubyte),
    ('wave_id', ctypes.c_ubyte),
    ('flags', ctypes.c_ubyte),
    ('reserved', ctypes.c_uint32),
]

rocprofiler_thread_trace_decoder_shaderdata_t = struct_rocprofiler_thread_trace_decoder_shaderdata_t

# values for enumeration 'rocprofiler_thread_trace_decoder_record_type_t'
rocprofiler_thread_trace_decoder_record_type_t__enumvalues = {
    0: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP',
    1: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY',
    2: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT',
    3: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE',
    4: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO',
    5: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG',
    6: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA',
    7: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME',
    8: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY',
    9: 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST',
}
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP = 0
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY = 1
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT = 2
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE = 3
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO = 4
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG = 5
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA = 6
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME = 7
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY = 8
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST = 9
rocprofiler_thread_trace_decoder_record_type_t = ctypes.c_uint32 # enum

# values for enumeration 'c__EA_rocprofiler_thread_trace_decoder_status_t'
c__EA_rocprofiler_thread_trace_decoder_status_t__enumvalues = {
    0: 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS',
    1: 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR',
    2: 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES',
    3: 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT',
    4: 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA',
    5: 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST',
}
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS = 0
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR = 1
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES = 2
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT = 3
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA = 4
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST = 5
c__EA_rocprofiler_thread_trace_decoder_status_t = ctypes.c_uint32 # enum
rocprofiler_thread_trace_decoder_status_t = c__EA_rocprofiler_thread_trace_decoder_status_t
rocprofiler_thread_trace_decoder_status_t__enumvalues = c__EA_rocprofiler_thread_trace_decoder_status_t__enumvalues
rocprof_trace_decoder_trace_callback_t = ctypes.CFUNCTYPE(c__EA_rocprofiler_thread_trace_decoder_status_t, rocprofiler_thread_trace_decoder_record_type_t, ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None))
rocprof_trace_decoder_isa_callback_t = ctypes.CFUNCTYPE(c__EA_rocprofiler_thread_trace_decoder_status_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), struct_rocprofiler_thread_trace_decoder_pc_t, ctypes.POINTER(None))
rocprof_trace_decoder_se_data_callback_t = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None))
try:
    rocprof_trace_decoder_parse_data = _libraries['FIXME_STUB'].rocprof_trace_decoder_parse_data
    rocprof_trace_decoder_parse_data.restype = rocprofiler_thread_trace_decoder_status_t
    rocprof_trace_decoder_parse_data.argtypes = [rocprof_trace_decoder_se_data_callback_t, rocprof_trace_decoder_trace_callback_t, rocprof_trace_decoder_isa_callback_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    rocprof_trace_decoder_get_info_string = _libraries['FIXME_STUB'].rocprof_trace_decoder_get_info_string
    rocprof_trace_decoder_get_info_string.restype = ctypes.POINTER(ctypes.c_char)
    rocprof_trace_decoder_get_info_string.argtypes = [rocprofiler_thread_trace_decoder_info_t]
except AttributeError:
    pass
try:
    rocprof_trace_decoder_get_status_string = _libraries['FIXME_STUB'].rocprof_trace_decoder_get_status_string
    rocprof_trace_decoder_get_status_string.restype = ctypes.POINTER(ctypes.c_char)
    rocprof_trace_decoder_get_status_string.argtypes = [rocprofiler_thread_trace_decoder_status_t]
except AttributeError:
    pass
rocprofiler_thread_trace_decoder_debug_callback_t = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None))
uint64_t = ctypes.c_uint64
try:
    rocprof_trace_decoder_dump_data = _libraries['FIXME_STUB'].rocprof_trace_decoder_dump_data
    rocprof_trace_decoder_dump_data.restype = rocprofiler_thread_trace_decoder_status_t
    rocprof_trace_decoder_dump_data.argtypes = [ctypes.POINTER(ctypes.c_char), uint64_t, rocprofiler_thread_trace_decoder_debug_callback_t, ctypes.POINTER(None)]
except AttributeError:
    pass
class union_rocprof_trace_decoder_gfx9_header_t(Union):
    pass

class struct_rocprof_trace_decoder_gfx9_header_t_0(Structure):
    pass

struct_rocprof_trace_decoder_gfx9_header_t_0._pack_ = 1 # source:False
struct_rocprof_trace_decoder_gfx9_header_t_0._fields_ = [
    ('legacy_version', ctypes.c_uint64, 13),
    ('gfx9_version2', ctypes.c_uint64, 3),
    ('DSIMDM', ctypes.c_uint64, 4),
    ('DCU', ctypes.c_uint64, 5),
    ('reserved1', ctypes.c_uint64, 1),
    ('SEID', ctypes.c_uint64, 6),
    ('reserved2', ctypes.c_uint64, 32),
]

union_rocprof_trace_decoder_gfx9_header_t._pack_ = 1 # source:False
union_rocprof_trace_decoder_gfx9_header_t._anonymous_ = ('_0',)
union_rocprof_trace_decoder_gfx9_header_t._fields_ = [
    ('_0', struct_rocprof_trace_decoder_gfx9_header_t_0),
    ('raw', ctypes.c_uint64),
]

rocprof_trace_decoder_gfx9_header_t = union_rocprof_trace_decoder_gfx9_header_t
class union_rocprof_trace_decoder_instrument_enable_t(Union):
    pass

class struct_rocprof_trace_decoder_instrument_enable_t_0(Structure):
    pass

struct_rocprof_trace_decoder_instrument_enable_t_0._pack_ = 1 # source:False
struct_rocprof_trace_decoder_instrument_enable_t_0._fields_ = [
    ('char1', ctypes.c_uint32, 8),
    ('char2', ctypes.c_uint32, 8),
    ('char3', ctypes.c_uint32, 8),
    ('char4', ctypes.c_uint32, 8),
]

union_rocprof_trace_decoder_instrument_enable_t._pack_ = 1 # source:False
union_rocprof_trace_decoder_instrument_enable_t._anonymous_ = ('_0',)
union_rocprof_trace_decoder_instrument_enable_t._fields_ = [
    ('_0', struct_rocprof_trace_decoder_instrument_enable_t_0),
    ('u32All', ctypes.c_uint32),
]

rocprof_trace_decoder_instrument_enable_t = union_rocprof_trace_decoder_instrument_enable_t
class union_rocprof_trace_decoder_packet_header_t(Union):
    pass

class struct_rocprof_trace_decoder_packet_header_t_0(Structure):
    pass

struct_rocprof_trace_decoder_packet_header_t_0._pack_ = 1 # source:False
struct_rocprof_trace_decoder_packet_header_t_0._fields_ = [
    ('opcode', ctypes.c_uint32, 8),
    ('type', ctypes.c_uint32, 4),
    ('data20', ctypes.c_uint32, 20),
]

union_rocprof_trace_decoder_packet_header_t._pack_ = 1 # source:False
union_rocprof_trace_decoder_packet_header_t._anonymous_ = ('_0',)
union_rocprof_trace_decoder_packet_header_t._fields_ = [
    ('_0', struct_rocprof_trace_decoder_packet_header_t_0),
    ('u32All', ctypes.c_uint32),
]

rocprof_trace_decoder_packet_header_t = union_rocprof_trace_decoder_packet_header_t

# values for enumeration 'rocprof_trace_decoder_packet_opcode_t'
rocprof_trace_decoder_packet_opcode_t__enumvalues = {
    4: 'ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ',
    5: 'ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP',
    6: 'ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO',
}
ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ = 4
ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP = 5
ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO = 6
rocprof_trace_decoder_packet_opcode_t = ctypes.c_uint32 # enum

# values for enumeration 'rocprof_trace_decoder_agent_info_type_t'
rocprof_trace_decoder_agent_info_type_t__enumvalues = {
    0: 'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ',
    1: 'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL',
    2: 'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST',
}
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ = 0
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL = 1
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST = 2
rocprof_trace_decoder_agent_info_type_t = ctypes.c_uint32 # enum
class union_rocprof_trace_decoder_codeobj_marker_tail_t(Union):
    pass

class struct_rocprof_trace_decoder_codeobj_marker_tail_t_0(Structure):
    pass

struct_rocprof_trace_decoder_codeobj_marker_tail_t_0._pack_ = 1 # source:False
struct_rocprof_trace_decoder_codeobj_marker_tail_t_0._fields_ = [
    ('isUnload', ctypes.c_uint32, 1),
    ('bFromStart', ctypes.c_uint32, 1),
    ('legacy_id', ctypes.c_uint32, 30),
]

union_rocprof_trace_decoder_codeobj_marker_tail_t._pack_ = 1 # source:False
union_rocprof_trace_decoder_codeobj_marker_tail_t._anonymous_ = ('_0',)
union_rocprof_trace_decoder_codeobj_marker_tail_t._fields_ = [
    ('_0', struct_rocprof_trace_decoder_codeobj_marker_tail_t_0),
    ('raw', ctypes.c_uint32),
]

rocprof_trace_decoder_codeobj_marker_tail_t = union_rocprof_trace_decoder_codeobj_marker_tail_t

# values for enumeration 'rocprof_trace_decoder_codeobj_marker_type_t'
rocprof_trace_decoder_codeobj_marker_type_t__enumvalues = {
    0: 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL',
    1: 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO',
    2: 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO',
    3: 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI',
    4: 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI',
    5: 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO',
    6: 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI',
    7: 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST',
}
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL = 0
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO = 1
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO = 2
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI = 3
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI = 4
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO = 5
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI = 6
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST = 7
rocprof_trace_decoder_codeobj_marker_type_t = ctypes.c_uint32 # enum
__all__ = \
    ['ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST',
    'ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST',
    'ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE',
    'ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE',
    'ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU',
    'ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA',
    'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE',
    'ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM',
    'ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV',
    'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR',
    'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT',
    'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA',
    'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES',
    'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST',
    'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS',
    'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY',
    'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC',
    'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE',
    'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST',
    'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL',
    'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT',
    'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL',
    'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST',
    'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ',
    'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI',
    'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO',
    'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI',
    'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO',
    'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST',
    'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI',
    'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO',
    'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL',
    'ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO',
    'ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ',
    'ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP',
    'c__EA_rocprofiler_thread_trace_decoder_status_t',
    'rocprof_trace_decoder_agent_info_type_t',
    'rocprof_trace_decoder_codeobj_marker_tail_t',
    'rocprof_trace_decoder_codeobj_marker_type_t',
    'rocprof_trace_decoder_dump_data',
    'rocprof_trace_decoder_get_info_string',
    'rocprof_trace_decoder_get_status_string',
    'rocprof_trace_decoder_gfx9_header_t',
    'rocprof_trace_decoder_instrument_enable_t',
    'rocprof_trace_decoder_isa_callback_t',
    'rocprof_trace_decoder_packet_header_t',
    'rocprof_trace_decoder_packet_opcode_t',
    'rocprof_trace_decoder_parse_data',
    'rocprof_trace_decoder_se_data_callback_t',
    'rocprof_trace_decoder_trace_callback_t',
    'rocprofiler_thread_trace_decoder_debug_callback_t',
    'rocprofiler_thread_trace_decoder_info_t',
    'rocprofiler_thread_trace_decoder_inst_category_t',
    'rocprofiler_thread_trace_decoder_inst_t',
    'rocprofiler_thread_trace_decoder_occupancy_t',
    'rocprofiler_thread_trace_decoder_pc_t',
    'rocprofiler_thread_trace_decoder_perfevent_t',
    'rocprofiler_thread_trace_decoder_realtime_t',
    'rocprofiler_thread_trace_decoder_record_type_t',
    'rocprofiler_thread_trace_decoder_shaderdata_flags_t',
    'rocprofiler_thread_trace_decoder_shaderdata_t',
    'rocprofiler_thread_trace_decoder_status_t',
    'rocprofiler_thread_trace_decoder_status_t__enumvalues',
    'rocprofiler_thread_trace_decoder_wave_state_t',
    'rocprofiler_thread_trace_decoder_wave_t',
    'rocprofiler_thread_trace_decoder_wstate_type_t',
    'struct_rocprof_trace_decoder_codeobj_marker_tail_t_0',
    'struct_rocprof_trace_decoder_gfx9_header_t_0',
    'struct_rocprof_trace_decoder_instrument_enable_t_0',
    'struct_rocprof_trace_decoder_packet_header_t_0',
    'struct_rocprofiler_thread_trace_decoder_inst_t',
    'struct_rocprofiler_thread_trace_decoder_occupancy_t',
    'struct_rocprofiler_thread_trace_decoder_pc_t',
    'struct_rocprofiler_thread_trace_decoder_perfevent_t',
    'struct_rocprofiler_thread_trace_decoder_realtime_t',
    'struct_rocprofiler_thread_trace_decoder_shaderdata_t',
    'struct_rocprofiler_thread_trace_decoder_wave_state_t',
    'struct_rocprofiler_thread_trace_decoder_wave_t', 'uint64_t',
    'union_rocprof_trace_decoder_codeobj_marker_tail_t',
    'union_rocprof_trace_decoder_gfx9_header_t',
    'union_rocprof_trace_decoder_instrument_enable_t',
    'union_rocprof_trace_decoder_packet_header_t']
