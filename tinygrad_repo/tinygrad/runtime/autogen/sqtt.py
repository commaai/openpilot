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



SQTT_FILE_MAGIC_NUMBER = 0x50303042 # macro
SQTT_FILE_VERSION_MAJOR = 1 # macro
SQTT_FILE_VERSION_MINOR = 5 # macro
SQTT_GPU_NAME_MAX_SIZE = 256 # macro
SQTT_MAX_NUM_SE = 32 # macro
SQTT_SA_PER_SE = 2 # macro
SQTT_ACTIVE_PIXEL_PACKER_MASK_DWORDS = 4 # macro
class struct_sqtt_data_info(Structure):
    pass

class union_sqtt_data_info_0(Union):
    pass

union_sqtt_data_info_0._pack_ = 1 # source:False
union_sqtt_data_info_0._fields_ = [
    ('gfx9_write_counter', ctypes.c_uint32),
    ('gfx10_dropped_cntr', ctypes.c_uint32),
]

struct_sqtt_data_info._pack_ = 1 # source:False
struct_sqtt_data_info._anonymous_ = ('_0',)
struct_sqtt_data_info._fields_ = [
    ('cur_offset', ctypes.c_uint32),
    ('trace_status', ctypes.c_uint32),
    ('_0', union_sqtt_data_info_0),
]

class struct_sqtt_data_se(Structure):
    pass

struct_sqtt_data_se._pack_ = 1 # source:False
struct_sqtt_data_se._fields_ = [
    ('info', struct_sqtt_data_info),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('data_ptr', ctypes.POINTER(None)),
    ('shader_engine', ctypes.c_uint32),
    ('compute_unit', ctypes.c_uint32),
]


# values for enumeration 'sqtt_version'
sqtt_version__enumvalues = {
    0: 'SQTT_VERSION_NONE',
    5: 'SQTT_VERSION_2_2',
    6: 'SQTT_VERSION_2_3',
    7: 'SQTT_VERSION_2_4',
    11: 'SQTT_VERSION_3_2',
}
SQTT_VERSION_NONE = 0
SQTT_VERSION_2_2 = 5
SQTT_VERSION_2_3 = 6
SQTT_VERSION_2_4 = 7
SQTT_VERSION_3_2 = 11
sqtt_version = ctypes.c_uint32 # enum

# values for enumeration 'sqtt_file_chunk_type'
sqtt_file_chunk_type__enumvalues = {
    0: 'SQTT_FILE_CHUNK_TYPE_ASIC_INFO',
    1: 'SQTT_FILE_CHUNK_TYPE_SQTT_DESC',
    2: 'SQTT_FILE_CHUNK_TYPE_SQTT_DATA',
    3: 'SQTT_FILE_CHUNK_TYPE_API_INFO',
    4: 'SQTT_FILE_CHUNK_TYPE_RESERVED',
    5: 'SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS',
    6: 'SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION',
    7: 'SQTT_FILE_CHUNK_TYPE_CPU_INFO',
    8: 'SQTT_FILE_CHUNK_TYPE_SPM_DB',
    9: 'SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE',
    10: 'SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS',
    11: 'SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION',
    12: 'SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE',
    13: 'SQTT_FILE_CHUNK_TYPE_COUNT',
}
SQTT_FILE_CHUNK_TYPE_ASIC_INFO = 0
SQTT_FILE_CHUNK_TYPE_SQTT_DESC = 1
SQTT_FILE_CHUNK_TYPE_SQTT_DATA = 2
SQTT_FILE_CHUNK_TYPE_API_INFO = 3
SQTT_FILE_CHUNK_TYPE_RESERVED = 4
SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS = 5
SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION = 6
SQTT_FILE_CHUNK_TYPE_CPU_INFO = 7
SQTT_FILE_CHUNK_TYPE_SPM_DB = 8
SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE = 9
SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS = 10
SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION = 11
SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE = 12
SQTT_FILE_CHUNK_TYPE_COUNT = 13
sqtt_file_chunk_type = ctypes.c_uint32 # enum
class struct_sqtt_file_chunk_id(Structure):
    pass

struct_sqtt_file_chunk_id._pack_ = 1 # source:False
struct_sqtt_file_chunk_id._fields_ = [
    ('type', ctypes.c_int32, 8),
    ('index', ctypes.c_int32, 8),
    ('reserved', ctypes.c_int32, 16),
]

class struct_sqtt_file_chunk_header(Structure):
    pass

struct_sqtt_file_chunk_header._pack_ = 1 # source:False
struct_sqtt_file_chunk_header._fields_ = [
    ('chunk_id', struct_sqtt_file_chunk_id),
    ('minor_version', ctypes.c_uint16),
    ('major_version', ctypes.c_uint16),
    ('size_in_bytes', ctypes.c_int32),
    ('padding', ctypes.c_int32),
]

class struct_sqtt_file_header_flags(Structure):
    pass

class union_sqtt_file_header_flags_0(Union):
    pass

class struct_sqtt_file_header_flags_0_0(Structure):
    pass

struct_sqtt_file_header_flags_0_0._pack_ = 1 # source:False
struct_sqtt_file_header_flags_0_0._fields_ = [
    ('is_semaphore_queue_timing_etw', ctypes.c_uint32, 1),
    ('no_queue_semaphore_timestamps', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 30),
]

union_sqtt_file_header_flags_0._pack_ = 1 # source:False
union_sqtt_file_header_flags_0._anonymous_ = ('_0',)
union_sqtt_file_header_flags_0._fields_ = [
    ('_0', struct_sqtt_file_header_flags_0_0),
    ('value', ctypes.c_uint32),
]

struct_sqtt_file_header_flags._pack_ = 1 # source:False
struct_sqtt_file_header_flags._anonymous_ = ('_0',)
struct_sqtt_file_header_flags._fields_ = [
    ('_0', union_sqtt_file_header_flags_0),
]

class struct_sqtt_file_header(Structure):
    pass

struct_sqtt_file_header._pack_ = 1 # source:False
struct_sqtt_file_header._fields_ = [
    ('magic_number', ctypes.c_uint32),
    ('version_major', ctypes.c_uint32),
    ('version_minor', ctypes.c_uint32),
    ('flags', struct_sqtt_file_header_flags),
    ('chunk_offset', ctypes.c_int32),
    ('second', ctypes.c_int32),
    ('minute', ctypes.c_int32),
    ('hour', ctypes.c_int32),
    ('day_in_month', ctypes.c_int32),
    ('month', ctypes.c_int32),
    ('year', ctypes.c_int32),
    ('day_in_week', ctypes.c_int32),
    ('day_in_year', ctypes.c_int32),
    ('is_daylight_savings', ctypes.c_int32),
]

class struct_sqtt_file_chunk_cpu_info(Structure):
    pass

struct_sqtt_file_chunk_cpu_info._pack_ = 1 # source:False
struct_sqtt_file_chunk_cpu_info._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('vendor_id', ctypes.c_uint32 * 4),
    ('processor_brand', ctypes.c_uint32 * 12),
    ('reserved', ctypes.c_uint32 * 2),
    ('cpu_timestamp_freq', ctypes.c_uint64),
    ('clock_speed', ctypes.c_uint32),
    ('num_logical_cores', ctypes.c_uint32),
    ('num_physical_cores', ctypes.c_uint32),
    ('system_ram_size', ctypes.c_uint32),
]


# values for enumeration 'sqtt_file_chunk_asic_info_flags'
sqtt_file_chunk_asic_info_flags__enumvalues = {
    1: 'SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING',
    2: 'SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED',
}
SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING = 1
SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED = 2
sqtt_file_chunk_asic_info_flags = ctypes.c_uint32 # enum

# values for enumeration 'sqtt_gpu_type'
sqtt_gpu_type__enumvalues = {
    0: 'SQTT_GPU_TYPE_UNKNOWN',
    1: 'SQTT_GPU_TYPE_INTEGRATED',
    2: 'SQTT_GPU_TYPE_DISCRETE',
    3: 'SQTT_GPU_TYPE_VIRTUAL',
}
SQTT_GPU_TYPE_UNKNOWN = 0
SQTT_GPU_TYPE_INTEGRATED = 1
SQTT_GPU_TYPE_DISCRETE = 2
SQTT_GPU_TYPE_VIRTUAL = 3
sqtt_gpu_type = ctypes.c_uint32 # enum

# values for enumeration 'sqtt_gfxip_level'
sqtt_gfxip_level__enumvalues = {
    0: 'SQTT_GFXIP_LEVEL_NONE',
    1: 'SQTT_GFXIP_LEVEL_GFXIP_6',
    2: 'SQTT_GFXIP_LEVEL_GFXIP_7',
    3: 'SQTT_GFXIP_LEVEL_GFXIP_8',
    4: 'SQTT_GFXIP_LEVEL_GFXIP_8_1',
    5: 'SQTT_GFXIP_LEVEL_GFXIP_9',
    7: 'SQTT_GFXIP_LEVEL_GFXIP_10_1',
    9: 'SQTT_GFXIP_LEVEL_GFXIP_10_3',
    12: 'SQTT_GFXIP_LEVEL_GFXIP_11_0',
}
SQTT_GFXIP_LEVEL_NONE = 0
SQTT_GFXIP_LEVEL_GFXIP_6 = 1
SQTT_GFXIP_LEVEL_GFXIP_7 = 2
SQTT_GFXIP_LEVEL_GFXIP_8 = 3
SQTT_GFXIP_LEVEL_GFXIP_8_1 = 4
SQTT_GFXIP_LEVEL_GFXIP_9 = 5
SQTT_GFXIP_LEVEL_GFXIP_10_1 = 7
SQTT_GFXIP_LEVEL_GFXIP_10_3 = 9
SQTT_GFXIP_LEVEL_GFXIP_11_0 = 12
sqtt_gfxip_level = ctypes.c_uint32 # enum

# values for enumeration 'sqtt_memory_type'
sqtt_memory_type__enumvalues = {
    0: 'SQTT_MEMORY_TYPE_UNKNOWN',
    1: 'SQTT_MEMORY_TYPE_DDR',
    2: 'SQTT_MEMORY_TYPE_DDR2',
    3: 'SQTT_MEMORY_TYPE_DDR3',
    4: 'SQTT_MEMORY_TYPE_DDR4',
    5: 'SQTT_MEMORY_TYPE_DDR5',
    16: 'SQTT_MEMORY_TYPE_GDDR3',
    17: 'SQTT_MEMORY_TYPE_GDDR4',
    18: 'SQTT_MEMORY_TYPE_GDDR5',
    19: 'SQTT_MEMORY_TYPE_GDDR6',
    32: 'SQTT_MEMORY_TYPE_HBM',
    33: 'SQTT_MEMORY_TYPE_HBM2',
    34: 'SQTT_MEMORY_TYPE_HBM3',
    48: 'SQTT_MEMORY_TYPE_LPDDR4',
    49: 'SQTT_MEMORY_TYPE_LPDDR5',
}
SQTT_MEMORY_TYPE_UNKNOWN = 0
SQTT_MEMORY_TYPE_DDR = 1
SQTT_MEMORY_TYPE_DDR2 = 2
SQTT_MEMORY_TYPE_DDR3 = 3
SQTT_MEMORY_TYPE_DDR4 = 4
SQTT_MEMORY_TYPE_DDR5 = 5
SQTT_MEMORY_TYPE_GDDR3 = 16
SQTT_MEMORY_TYPE_GDDR4 = 17
SQTT_MEMORY_TYPE_GDDR5 = 18
SQTT_MEMORY_TYPE_GDDR6 = 19
SQTT_MEMORY_TYPE_HBM = 32
SQTT_MEMORY_TYPE_HBM2 = 33
SQTT_MEMORY_TYPE_HBM3 = 34
SQTT_MEMORY_TYPE_LPDDR4 = 48
SQTT_MEMORY_TYPE_LPDDR5 = 49
sqtt_memory_type = ctypes.c_uint32 # enum
class struct_sqtt_file_chunk_asic_info(Structure):
    pass

struct_sqtt_file_chunk_asic_info._pack_ = 1 # source:False
struct_sqtt_file_chunk_asic_info._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('flags', ctypes.c_uint64),
    ('trace_shader_core_clock', ctypes.c_uint64),
    ('trace_memory_clock', ctypes.c_uint64),
    ('device_id', ctypes.c_int32),
    ('device_revision_id', ctypes.c_int32),
    ('vgprs_per_simd', ctypes.c_int32),
    ('sgprs_per_simd', ctypes.c_int32),
    ('shader_engines', ctypes.c_int32),
    ('compute_unit_per_shader_engine', ctypes.c_int32),
    ('simd_per_compute_unit', ctypes.c_int32),
    ('wavefronts_per_simd', ctypes.c_int32),
    ('minimum_vgpr_alloc', ctypes.c_int32),
    ('vgpr_alloc_granularity', ctypes.c_int32),
    ('minimum_sgpr_alloc', ctypes.c_int32),
    ('sgpr_alloc_granularity', ctypes.c_int32),
    ('hardware_contexts', ctypes.c_int32),
    ('gpu_type', sqtt_gpu_type),
    ('gfxip_level', sqtt_gfxip_level),
    ('gpu_index', ctypes.c_int32),
    ('gds_size', ctypes.c_int32),
    ('gds_per_shader_engine', ctypes.c_int32),
    ('ce_ram_size', ctypes.c_int32),
    ('ce_ram_size_graphics', ctypes.c_int32),
    ('ce_ram_size_compute', ctypes.c_int32),
    ('max_number_of_dedicated_cus', ctypes.c_int32),
    ('vram_size', ctypes.c_int64),
    ('vram_bus_width', ctypes.c_int32),
    ('l2_cache_size', ctypes.c_int32),
    ('l1_cache_size', ctypes.c_int32),
    ('lds_size', ctypes.c_int32),
    ('gpu_name', ctypes.c_char * 256),
    ('alu_per_clock', ctypes.c_float),
    ('texture_per_clock', ctypes.c_float),
    ('prims_per_clock', ctypes.c_float),
    ('pixels_per_clock', ctypes.c_float),
    ('gpu_timestamp_frequency', ctypes.c_uint64),
    ('max_shader_core_clock', ctypes.c_uint64),
    ('max_memory_clock', ctypes.c_uint64),
    ('memory_ops_per_clock', ctypes.c_uint32),
    ('memory_chip_type', sqtt_memory_type),
    ('lds_granularity', ctypes.c_uint32),
    ('cu_mask', ctypes.c_uint16 * 2 * 32),
    ('reserved1', ctypes.c_char * 128),
    ('active_pixel_packer_mask', ctypes.c_uint32 * 4),
    ('reserved2', ctypes.c_char * 16),
    ('gl1_cache_size', ctypes.c_uint32),
    ('instruction_cache_size', ctypes.c_uint32),
    ('scalar_cache_size', ctypes.c_uint32),
    ('mall_cache_size', ctypes.c_uint32),
    ('padding', ctypes.c_char * 4),
]


# values for enumeration 'sqtt_api_type'
sqtt_api_type__enumvalues = {
    0: 'SQTT_API_TYPE_DIRECTX_12',
    1: 'SQTT_API_TYPE_VULKAN',
    2: 'SQTT_API_TYPE_GENERIC',
    3: 'SQTT_API_TYPE_OPENCL',
}
SQTT_API_TYPE_DIRECTX_12 = 0
SQTT_API_TYPE_VULKAN = 1
SQTT_API_TYPE_GENERIC = 2
SQTT_API_TYPE_OPENCL = 3
sqtt_api_type = ctypes.c_uint32 # enum

# values for enumeration 'sqtt_instruction_trace_mode'
sqtt_instruction_trace_mode__enumvalues = {
    0: 'SQTT_INSTRUCTION_TRACE_DISABLED',
    1: 'SQTT_INSTRUCTION_TRACE_FULL_FRAME',
    2: 'SQTT_INSTRUCTION_TRACE_API_PSO',
}
SQTT_INSTRUCTION_TRACE_DISABLED = 0
SQTT_INSTRUCTION_TRACE_FULL_FRAME = 1
SQTT_INSTRUCTION_TRACE_API_PSO = 2
sqtt_instruction_trace_mode = ctypes.c_uint32 # enum

# values for enumeration 'sqtt_profiling_mode'
sqtt_profiling_mode__enumvalues = {
    0: 'SQTT_PROFILING_MODE_PRESENT',
    1: 'SQTT_PROFILING_MODE_USER_MARKERS',
    2: 'SQTT_PROFILING_MODE_INDEX',
    3: 'SQTT_PROFILING_MODE_TAG',
}
SQTT_PROFILING_MODE_PRESENT = 0
SQTT_PROFILING_MODE_USER_MARKERS = 1
SQTT_PROFILING_MODE_INDEX = 2
SQTT_PROFILING_MODE_TAG = 3
sqtt_profiling_mode = ctypes.c_uint32 # enum
class union_sqtt_profiling_mode_data(Union):
    pass

class struct_sqtt_profiling_mode_data_user_marker_profiling_data(Structure):
    pass

struct_sqtt_profiling_mode_data_user_marker_profiling_data._pack_ = 1 # source:False
struct_sqtt_profiling_mode_data_user_marker_profiling_data._fields_ = [
    ('start', ctypes.c_char * 256),
    ('end', ctypes.c_char * 256),
]

class struct_sqtt_profiling_mode_data_index_profiling_data(Structure):
    pass

struct_sqtt_profiling_mode_data_index_profiling_data._pack_ = 1 # source:False
struct_sqtt_profiling_mode_data_index_profiling_data._fields_ = [
    ('start', ctypes.c_uint32),
    ('end', ctypes.c_uint32),
]

class struct_sqtt_profiling_mode_data_tag_profiling_data(Structure):
    pass

struct_sqtt_profiling_mode_data_tag_profiling_data._pack_ = 1 # source:False
struct_sqtt_profiling_mode_data_tag_profiling_data._fields_ = [
    ('begin_hi', ctypes.c_uint32),
    ('begin_lo', ctypes.c_uint32),
    ('end_hi', ctypes.c_uint32),
    ('end_lo', ctypes.c_uint32),
]

union_sqtt_profiling_mode_data._pack_ = 1 # source:False
union_sqtt_profiling_mode_data._fields_ = [
    ('user_marker_profiling_data', struct_sqtt_profiling_mode_data_user_marker_profiling_data),
    ('index_profiling_data', struct_sqtt_profiling_mode_data_index_profiling_data),
    ('tag_profiling_data', struct_sqtt_profiling_mode_data_tag_profiling_data),
    ('PADDING_0', ctypes.c_ubyte * 496),
]

class union_sqtt_instruction_trace_data(Union):
    pass

class struct_sqtt_instruction_trace_data_api_pso_data(Structure):
    pass

struct_sqtt_instruction_trace_data_api_pso_data._pack_ = 1 # source:False
struct_sqtt_instruction_trace_data_api_pso_data._fields_ = [
    ('api_pso_filter', ctypes.c_uint64),
]

class struct_sqtt_instruction_trace_data_shader_engine_filter(Structure):
    pass

struct_sqtt_instruction_trace_data_shader_engine_filter._pack_ = 1 # source:False
struct_sqtt_instruction_trace_data_shader_engine_filter._fields_ = [
    ('mask', ctypes.c_uint32),
]

union_sqtt_instruction_trace_data._pack_ = 1 # source:False
union_sqtt_instruction_trace_data._fields_ = [
    ('api_pso_data', struct_sqtt_instruction_trace_data_api_pso_data),
    ('shader_engine_filter', struct_sqtt_instruction_trace_data_shader_engine_filter),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_sqtt_file_chunk_api_info(Structure):
    pass

struct_sqtt_file_chunk_api_info._pack_ = 1 # source:False
struct_sqtt_file_chunk_api_info._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('api_type', sqtt_api_type),
    ('major_version', ctypes.c_uint16),
    ('minor_version', ctypes.c_uint16),
    ('profiling_mode', sqtt_profiling_mode),
    ('reserved', ctypes.c_uint32),
    ('profiling_mode_data', union_sqtt_profiling_mode_data),
    ('instruction_trace_mode', sqtt_instruction_trace_mode),
    ('reserved2', ctypes.c_uint32),
    ('instruction_trace_data', union_sqtt_instruction_trace_data),
]

class struct_sqtt_code_object_database_record(Structure):
    pass

struct_sqtt_code_object_database_record._pack_ = 1 # source:False
struct_sqtt_code_object_database_record._fields_ = [
    ('size', ctypes.c_uint32),
]

class struct_sqtt_file_chunk_code_object_database(Structure):
    pass

struct_sqtt_file_chunk_code_object_database._pack_ = 1 # source:False
struct_sqtt_file_chunk_code_object_database._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('offset', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('record_count', ctypes.c_uint32),
]

class struct_sqtt_code_object_loader_events_record(Structure):
    pass

struct_sqtt_code_object_loader_events_record._pack_ = 1 # source:False
struct_sqtt_code_object_loader_events_record._fields_ = [
    ('loader_event_type', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('base_address', ctypes.c_uint64),
    ('code_object_hash', ctypes.c_uint64 * 2),
    ('time_stamp', ctypes.c_uint64),
]

class struct_sqtt_file_chunk_code_object_loader_events(Structure):
    pass

struct_sqtt_file_chunk_code_object_loader_events._pack_ = 1 # source:False
struct_sqtt_file_chunk_code_object_loader_events._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('offset', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('record_size', ctypes.c_uint32),
    ('record_count', ctypes.c_uint32),
]

class struct_sqtt_pso_correlation_record(Structure):
    pass

struct_sqtt_pso_correlation_record._pack_ = 1 # source:False
struct_sqtt_pso_correlation_record._fields_ = [
    ('api_pso_hash', ctypes.c_uint64),
    ('pipeline_hash', ctypes.c_uint64 * 2),
    ('api_level_obj_name', ctypes.c_char * 64),
]

class struct_sqtt_file_chunk_pso_correlation(Structure):
    pass

struct_sqtt_file_chunk_pso_correlation._pack_ = 1 # source:False
struct_sqtt_file_chunk_pso_correlation._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('offset', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('record_size', ctypes.c_uint32),
    ('record_count', ctypes.c_uint32),
]

class struct_sqtt_file_chunk_sqtt_desc(Structure):
    pass

class union_sqtt_file_chunk_sqtt_desc_0(Union):
    pass

class struct_sqtt_file_chunk_sqtt_desc_0_v0(Structure):
    pass

struct_sqtt_file_chunk_sqtt_desc_0_v0._pack_ = 1 # source:False
struct_sqtt_file_chunk_sqtt_desc_0_v0._fields_ = [
    ('instrumentation_version', ctypes.c_int32),
]

class struct_sqtt_file_chunk_sqtt_desc_0_v1(Structure):
    pass

struct_sqtt_file_chunk_sqtt_desc_0_v1._pack_ = 1 # source:False
struct_sqtt_file_chunk_sqtt_desc_0_v1._fields_ = [
    ('instrumentation_spec_version', ctypes.c_int16),
    ('instrumentation_api_version', ctypes.c_int16),
    ('compute_unit_index', ctypes.c_int32),
]

union_sqtt_file_chunk_sqtt_desc_0._pack_ = 1 # source:False
union_sqtt_file_chunk_sqtt_desc_0._fields_ = [
    ('v0', struct_sqtt_file_chunk_sqtt_desc_0_v0),
    ('v1', struct_sqtt_file_chunk_sqtt_desc_0_v1),
]

struct_sqtt_file_chunk_sqtt_desc._pack_ = 1 # source:False
struct_sqtt_file_chunk_sqtt_desc._anonymous_ = ('_0',)
struct_sqtt_file_chunk_sqtt_desc._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('shader_engine_index', ctypes.c_int32),
    ('sqtt_version', sqtt_version),
    ('_0', union_sqtt_file_chunk_sqtt_desc_0),
]

class struct_sqtt_file_chunk_sqtt_data(Structure):
    pass

struct_sqtt_file_chunk_sqtt_data._pack_ = 1 # source:False
struct_sqtt_file_chunk_sqtt_data._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('offset', ctypes.c_int32),
    ('size', ctypes.c_int32),
]

class struct_sqtt_file_chunk_queue_event_timings(Structure):
    pass

struct_sqtt_file_chunk_queue_event_timings._pack_ = 1 # source:False
struct_sqtt_file_chunk_queue_event_timings._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('queue_info_table_record_count', ctypes.c_uint32),
    ('queue_info_table_size', ctypes.c_uint32),
    ('queue_event_table_record_count', ctypes.c_uint32),
    ('queue_event_table_size', ctypes.c_uint32),
]


# values for enumeration 'sqtt_queue_type'
sqtt_queue_type__enumvalues = {
    0: 'SQTT_QUEUE_TYPE_UNKNOWN',
    1: 'SQTT_QUEUE_TYPE_UNIVERSAL',
    2: 'SQTT_QUEUE_TYPE_COMPUTE',
    3: 'SQTT_QUEUE_TYPE_DMA',
}
SQTT_QUEUE_TYPE_UNKNOWN = 0
SQTT_QUEUE_TYPE_UNIVERSAL = 1
SQTT_QUEUE_TYPE_COMPUTE = 2
SQTT_QUEUE_TYPE_DMA = 3
sqtt_queue_type = ctypes.c_uint32 # enum

# values for enumeration 'sqtt_engine_type'
sqtt_engine_type__enumvalues = {
    0: 'SQTT_ENGINE_TYPE_UNKNOWN',
    1: 'SQTT_ENGINE_TYPE_UNIVERSAL',
    2: 'SQTT_ENGINE_TYPE_COMPUTE',
    3: 'SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE',
    4: 'SQTT_ENGINE_TYPE_DMA',
    7: 'SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL',
    8: 'SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS',
}
SQTT_ENGINE_TYPE_UNKNOWN = 0
SQTT_ENGINE_TYPE_UNIVERSAL = 1
SQTT_ENGINE_TYPE_COMPUTE = 2
SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE = 3
SQTT_ENGINE_TYPE_DMA = 4
SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL = 7
SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS = 8
sqtt_engine_type = ctypes.c_uint32 # enum
class struct_sqtt_queue_hardware_info(Structure):
    pass

class union_sqtt_queue_hardware_info_0(Union):
    pass

class struct_sqtt_queue_hardware_info_0_0(Structure):
    pass

struct_sqtt_queue_hardware_info_0_0._pack_ = 1 # source:False
struct_sqtt_queue_hardware_info_0_0._fields_ = [
    ('queue_type', ctypes.c_int32, 8),
    ('engine_type', ctypes.c_int32, 8),
    ('reserved', ctypes.c_int32, 16),
]

union_sqtt_queue_hardware_info_0._pack_ = 1 # source:False
union_sqtt_queue_hardware_info_0._anonymous_ = ('_0',)
union_sqtt_queue_hardware_info_0._fields_ = [
    ('_0', struct_sqtt_queue_hardware_info_0_0),
    ('value', ctypes.c_uint32),
]

struct_sqtt_queue_hardware_info._pack_ = 1 # source:False
struct_sqtt_queue_hardware_info._anonymous_ = ('_0',)
struct_sqtt_queue_hardware_info._fields_ = [
    ('_0', union_sqtt_queue_hardware_info_0),
]

class struct_sqtt_queue_info_record(Structure):
    pass

struct_sqtt_queue_info_record._pack_ = 1 # source:False
struct_sqtt_queue_info_record._fields_ = [
    ('queue_id', ctypes.c_uint64),
    ('queue_context', ctypes.c_uint64),
    ('hardware_info', struct_sqtt_queue_hardware_info),
    ('reserved', ctypes.c_uint32),
]


# values for enumeration 'sqtt_queue_event_type'
sqtt_queue_event_type__enumvalues = {
    0: 'SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT',
    1: 'SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE',
    2: 'SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE',
    3: 'SQTT_QUEUE_TIMING_EVENT_PRESENT',
}
SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT = 0
SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE = 1
SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE = 2
SQTT_QUEUE_TIMING_EVENT_PRESENT = 3
sqtt_queue_event_type = ctypes.c_uint32 # enum
class struct_sqtt_queue_event_record(Structure):
    pass

struct_sqtt_queue_event_record._pack_ = 1 # source:False
struct_sqtt_queue_event_record._fields_ = [
    ('event_type', sqtt_queue_event_type),
    ('sqtt_cb_id', ctypes.c_uint32),
    ('frame_index', ctypes.c_uint64),
    ('queue_info_index', ctypes.c_uint32),
    ('submit_sub_index', ctypes.c_uint32),
    ('api_id', ctypes.c_uint64),
    ('cpu_timestamp', ctypes.c_uint64),
    ('gpu_timestamps', ctypes.c_uint64 * 2),
]

class struct_sqtt_file_chunk_clock_calibration(Structure):
    pass

struct_sqtt_file_chunk_clock_calibration._pack_ = 1 # source:False
struct_sqtt_file_chunk_clock_calibration._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('cpu_timestamp', ctypes.c_uint64),
    ('gpu_timestamp', ctypes.c_uint64),
    ('reserved', ctypes.c_uint64),
]


# values for enumeration 'elf_gfxip_level'
elf_gfxip_level__enumvalues = {
    40: 'EF_AMDGPU_MACH_AMDGCN_GFX801',
    44: 'EF_AMDGPU_MACH_AMDGCN_GFX900',
    51: 'EF_AMDGPU_MACH_AMDGCN_GFX1010',
    54: 'EF_AMDGPU_MACH_AMDGCN_GFX1030',
    65: 'EF_AMDGPU_MACH_AMDGCN_GFX1100',
}
EF_AMDGPU_MACH_AMDGCN_GFX801 = 40
EF_AMDGPU_MACH_AMDGCN_GFX900 = 44
EF_AMDGPU_MACH_AMDGCN_GFX1010 = 51
EF_AMDGPU_MACH_AMDGCN_GFX1030 = 54
EF_AMDGPU_MACH_AMDGCN_GFX1100 = 65
elf_gfxip_level = ctypes.c_uint32 # enum
class struct_sqtt_file_chunk_spm_db(Structure):
    pass

struct_sqtt_file_chunk_spm_db._pack_ = 1 # source:False
struct_sqtt_file_chunk_spm_db._fields_ = [
    ('header', struct_sqtt_file_chunk_header),
    ('flags', ctypes.c_uint32),
    ('preamble_size', ctypes.c_uint32),
    ('num_timestamps', ctypes.c_uint32),
    ('num_spm_counter_info', ctypes.c_uint32),
    ('spm_counter_info_size', ctypes.c_uint32),
    ('sample_interval', ctypes.c_uint32),
]


# values for enumeration 'rgp_sqtt_marker_identifier'
rgp_sqtt_marker_identifier__enumvalues = {
    0: 'RGP_SQTT_MARKER_IDENTIFIER_EVENT',
    1: 'RGP_SQTT_MARKER_IDENTIFIER_CB_START',
    2: 'RGP_SQTT_MARKER_IDENTIFIER_CB_END',
    3: 'RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START',
    4: 'RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END',
    5: 'RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT',
    6: 'RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API',
    7: 'RGP_SQTT_MARKER_IDENTIFIER_SYNC',
    8: 'RGP_SQTT_MARKER_IDENTIFIER_PRESENT',
    9: 'RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION',
    10: 'RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS',
    11: 'RGP_SQTT_MARKER_IDENTIFIER_RESERVED2',
    12: 'RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE',
    13: 'RGP_SQTT_MARKER_IDENTIFIER_RESERVED4',
    14: 'RGP_SQTT_MARKER_IDENTIFIER_RESERVED5',
    15: 'RGP_SQTT_MARKER_IDENTIFIER_RESERVED6',
}
RGP_SQTT_MARKER_IDENTIFIER_EVENT = 0
RGP_SQTT_MARKER_IDENTIFIER_CB_START = 1
RGP_SQTT_MARKER_IDENTIFIER_CB_END = 2
RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START = 3
RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END = 4
RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT = 5
RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API = 6
RGP_SQTT_MARKER_IDENTIFIER_SYNC = 7
RGP_SQTT_MARKER_IDENTIFIER_PRESENT = 8
RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION = 9
RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS = 10
RGP_SQTT_MARKER_IDENTIFIER_RESERVED2 = 11
RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE = 12
RGP_SQTT_MARKER_IDENTIFIER_RESERVED4 = 13
RGP_SQTT_MARKER_IDENTIFIER_RESERVED5 = 14
RGP_SQTT_MARKER_IDENTIFIER_RESERVED6 = 15
rgp_sqtt_marker_identifier = ctypes.c_uint32 # enum
class union_rgp_sqtt_marker_cb_id(Union):
    pass

class struct_rgp_sqtt_marker_cb_id_per_frame_cb_id(Structure):
    pass

struct_rgp_sqtt_marker_cb_id_per_frame_cb_id._pack_ = 1 # source:False
struct_rgp_sqtt_marker_cb_id_per_frame_cb_id._fields_ = [
    ('per_frame', ctypes.c_uint32, 1),
    ('frame_index', ctypes.c_uint32, 7),
    ('cb_index', ctypes.c_uint32, 12),
    ('reserved', ctypes.c_uint32, 12),
]

class struct_rgp_sqtt_marker_cb_id_global_cb_id(Structure):
    pass

struct_rgp_sqtt_marker_cb_id_global_cb_id._pack_ = 1 # source:False
struct_rgp_sqtt_marker_cb_id_global_cb_id._fields_ = [
    ('per_frame', ctypes.c_uint32, 1),
    ('cb_index', ctypes.c_uint32, 19),
    ('reserved', ctypes.c_uint32, 12),
]

union_rgp_sqtt_marker_cb_id._pack_ = 1 # source:False
union_rgp_sqtt_marker_cb_id._fields_ = [
    ('per_frame_cb_id', struct_rgp_sqtt_marker_cb_id_per_frame_cb_id),
    ('global_cb_id', struct_rgp_sqtt_marker_cb_id_global_cb_id),
    ('all', ctypes.c_uint32),
]

class struct_rgp_sqtt_marker_cb_start(Structure):
    pass

class union_rgp_sqtt_marker_cb_start_0(Union):
    pass

class struct_rgp_sqtt_marker_cb_start_0_0(Structure):
    pass

struct_rgp_sqtt_marker_cb_start_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_cb_start_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('ext_dwords', ctypes.c_uint32, 3),
    ('cb_id', ctypes.c_uint32, 20),
    ('queue', ctypes.c_uint32, 5),
]

union_rgp_sqtt_marker_cb_start_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_cb_start_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_cb_start_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_cb_start_0_0),
    ('dword01', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_cb_start_1(Union):
    pass

union_rgp_sqtt_marker_cb_start_1._pack_ = 1 # source:False
union_rgp_sqtt_marker_cb_start_1._fields_ = [
    ('device_id_low', ctypes.c_uint32),
    ('dword02', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_cb_start_2(Union):
    pass

union_rgp_sqtt_marker_cb_start_2._pack_ = 1 # source:False
union_rgp_sqtt_marker_cb_start_2._fields_ = [
    ('device_id_high', ctypes.c_uint32),
    ('dword03', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_cb_start_3(Union):
    pass

union_rgp_sqtt_marker_cb_start_3._pack_ = 1 # source:False
union_rgp_sqtt_marker_cb_start_3._fields_ = [
    ('queue_flags', ctypes.c_uint32),
    ('dword04', ctypes.c_uint32),
]

struct_rgp_sqtt_marker_cb_start._pack_ = 1 # source:False
struct_rgp_sqtt_marker_cb_start._anonymous_ = ('_0', '_1', '_2', '_3',)
struct_rgp_sqtt_marker_cb_start._fields_ = [
    ('_0', union_rgp_sqtt_marker_cb_start_0),
    ('_1', union_rgp_sqtt_marker_cb_start_1),
    ('_2', union_rgp_sqtt_marker_cb_start_2),
    ('_3', union_rgp_sqtt_marker_cb_start_3),
]

class struct_rgp_sqtt_marker_cb_end(Structure):
    pass

class union_rgp_sqtt_marker_cb_end_0(Union):
    pass

class struct_rgp_sqtt_marker_cb_end_0_0(Structure):
    pass

struct_rgp_sqtt_marker_cb_end_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_cb_end_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('ext_dwords', ctypes.c_uint32, 3),
    ('cb_id', ctypes.c_uint32, 20),
    ('reserved', ctypes.c_uint32, 5),
]

union_rgp_sqtt_marker_cb_end_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_cb_end_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_cb_end_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_cb_end_0_0),
    ('dword01', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_cb_end_1(Union):
    pass

union_rgp_sqtt_marker_cb_end_1._pack_ = 1 # source:False
union_rgp_sqtt_marker_cb_end_1._fields_ = [
    ('device_id_low', ctypes.c_uint32),
    ('dword02', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_cb_end_2(Union):
    pass

union_rgp_sqtt_marker_cb_end_2._pack_ = 1 # source:False
union_rgp_sqtt_marker_cb_end_2._fields_ = [
    ('device_id_high', ctypes.c_uint32),
    ('dword03', ctypes.c_uint32),
]

struct_rgp_sqtt_marker_cb_end._pack_ = 1 # source:False
struct_rgp_sqtt_marker_cb_end._anonymous_ = ('_0', '_1', '_2',)
struct_rgp_sqtt_marker_cb_end._fields_ = [
    ('_0', union_rgp_sqtt_marker_cb_end_0),
    ('_1', union_rgp_sqtt_marker_cb_end_1),
    ('_2', union_rgp_sqtt_marker_cb_end_2),
]


# values for enumeration 'rgp_sqtt_marker_general_api_type'
rgp_sqtt_marker_general_api_type__enumvalues = {
    0: 'ApiCmdBindPipeline',
    1: 'ApiCmdBindDescriptorSets',
    2: 'ApiCmdBindIndexBuffer',
    3: 'ApiCmdBindVertexBuffers',
    4: 'ApiCmdDraw',
    5: 'ApiCmdDrawIndexed',
    6: 'ApiCmdDrawIndirect',
    7: 'ApiCmdDrawIndexedIndirect',
    8: 'ApiCmdDrawIndirectCountAMD',
    9: 'ApiCmdDrawIndexedIndirectCountAMD',
    10: 'ApiCmdDispatch',
    11: 'ApiCmdDispatchIndirect',
    12: 'ApiCmdCopyBuffer',
    13: 'ApiCmdCopyImage',
    14: 'ApiCmdBlitImage',
    15: 'ApiCmdCopyBufferToImage',
    16: 'ApiCmdCopyImageToBuffer',
    17: 'ApiCmdUpdateBuffer',
    18: 'ApiCmdFillBuffer',
    19: 'ApiCmdClearColorImage',
    20: 'ApiCmdClearDepthStencilImage',
    21: 'ApiCmdClearAttachments',
    22: 'ApiCmdResolveImage',
    23: 'ApiCmdWaitEvents',
    24: 'ApiCmdPipelineBarrier',
    25: 'ApiCmdBeginQuery',
    26: 'ApiCmdEndQuery',
    27: 'ApiCmdResetQueryPool',
    28: 'ApiCmdWriteTimestamp',
    29: 'ApiCmdCopyQueryPoolResults',
    30: 'ApiCmdPushConstants',
    31: 'ApiCmdBeginRenderPass',
    32: 'ApiCmdNextSubpass',
    33: 'ApiCmdEndRenderPass',
    34: 'ApiCmdExecuteCommands',
    35: 'ApiCmdSetViewport',
    36: 'ApiCmdSetScissor',
    37: 'ApiCmdSetLineWidth',
    38: 'ApiCmdSetDepthBias',
    39: 'ApiCmdSetBlendConstants',
    40: 'ApiCmdSetDepthBounds',
    41: 'ApiCmdSetStencilCompareMask',
    42: 'ApiCmdSetStencilWriteMask',
    43: 'ApiCmdSetStencilReference',
    44: 'ApiCmdDrawIndirectCount',
    45: 'ApiCmdDrawIndexedIndirectCount',
    47: 'ApiCmdDrawMeshTasksEXT',
    48: 'ApiCmdDrawMeshTasksIndirectCountEXT',
    49: 'ApiCmdDrawMeshTasksIndirectEXT',
    8388608: 'ApiRayTracingSeparateCompiled',
    4294967295: 'ApiInvalid',
}
ApiCmdBindPipeline = 0
ApiCmdBindDescriptorSets = 1
ApiCmdBindIndexBuffer = 2
ApiCmdBindVertexBuffers = 3
ApiCmdDraw = 4
ApiCmdDrawIndexed = 5
ApiCmdDrawIndirect = 6
ApiCmdDrawIndexedIndirect = 7
ApiCmdDrawIndirectCountAMD = 8
ApiCmdDrawIndexedIndirectCountAMD = 9
ApiCmdDispatch = 10
ApiCmdDispatchIndirect = 11
ApiCmdCopyBuffer = 12
ApiCmdCopyImage = 13
ApiCmdBlitImage = 14
ApiCmdCopyBufferToImage = 15
ApiCmdCopyImageToBuffer = 16
ApiCmdUpdateBuffer = 17
ApiCmdFillBuffer = 18
ApiCmdClearColorImage = 19
ApiCmdClearDepthStencilImage = 20
ApiCmdClearAttachments = 21
ApiCmdResolveImage = 22
ApiCmdWaitEvents = 23
ApiCmdPipelineBarrier = 24
ApiCmdBeginQuery = 25
ApiCmdEndQuery = 26
ApiCmdResetQueryPool = 27
ApiCmdWriteTimestamp = 28
ApiCmdCopyQueryPoolResults = 29
ApiCmdPushConstants = 30
ApiCmdBeginRenderPass = 31
ApiCmdNextSubpass = 32
ApiCmdEndRenderPass = 33
ApiCmdExecuteCommands = 34
ApiCmdSetViewport = 35
ApiCmdSetScissor = 36
ApiCmdSetLineWidth = 37
ApiCmdSetDepthBias = 38
ApiCmdSetBlendConstants = 39
ApiCmdSetDepthBounds = 40
ApiCmdSetStencilCompareMask = 41
ApiCmdSetStencilWriteMask = 42
ApiCmdSetStencilReference = 43
ApiCmdDrawIndirectCount = 44
ApiCmdDrawIndexedIndirectCount = 45
ApiCmdDrawMeshTasksEXT = 47
ApiCmdDrawMeshTasksIndirectCountEXT = 48
ApiCmdDrawMeshTasksIndirectEXT = 49
ApiRayTracingSeparateCompiled = 8388608
ApiInvalid = 4294967295
rgp_sqtt_marker_general_api_type = ctypes.c_uint32 # enum
class struct_rgp_sqtt_marker_general_api(Structure):
    pass

class union_rgp_sqtt_marker_general_api_0(Union):
    pass

class struct_rgp_sqtt_marker_general_api_0_0(Structure):
    pass

struct_rgp_sqtt_marker_general_api_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_general_api_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('ext_dwords', ctypes.c_uint32, 3),
    ('api_type', ctypes.c_uint32, 20),
    ('is_end', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 4),
]

union_rgp_sqtt_marker_general_api_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_general_api_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_general_api_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_general_api_0_0),
    ('dword01', ctypes.c_uint32),
]

struct_rgp_sqtt_marker_general_api._pack_ = 1 # source:False
struct_rgp_sqtt_marker_general_api._anonymous_ = ('_0',)
struct_rgp_sqtt_marker_general_api._fields_ = [
    ('_0', union_rgp_sqtt_marker_general_api_0),
]


# values for enumeration 'rgp_sqtt_marker_event_type'
rgp_sqtt_marker_event_type__enumvalues = {
    0: 'EventCmdDraw',
    1: 'EventCmdDrawIndexed',
    2: 'EventCmdDrawIndirect',
    3: 'EventCmdDrawIndexedIndirect',
    4: 'EventCmdDrawIndirectCountAMD',
    5: 'EventCmdDrawIndexedIndirectCountAMD',
    6: 'EventCmdDispatch',
    7: 'EventCmdDispatchIndirect',
    8: 'EventCmdCopyBuffer',
    9: 'EventCmdCopyImage',
    10: 'EventCmdBlitImage',
    11: 'EventCmdCopyBufferToImage',
    12: 'EventCmdCopyImageToBuffer',
    13: 'EventCmdUpdateBuffer',
    14: 'EventCmdFillBuffer',
    15: 'EventCmdClearColorImage',
    16: 'EventCmdClearDepthStencilImage',
    17: 'EventCmdClearAttachments',
    18: 'EventCmdResolveImage',
    19: 'EventCmdWaitEvents',
    20: 'EventCmdPipelineBarrier',
    21: 'EventCmdResetQueryPool',
    22: 'EventCmdCopyQueryPoolResults',
    23: 'EventRenderPassColorClear',
    24: 'EventRenderPassDepthStencilClear',
    25: 'EventRenderPassResolve',
    26: 'EventInternalUnknown',
    27: 'EventCmdDrawIndirectCount',
    28: 'EventCmdDrawIndexedIndirectCount',
    30: 'EventCmdTraceRaysKHR',
    31: 'EventCmdTraceRaysIndirectKHR',
    32: 'EventCmdBuildAccelerationStructuresKHR',
    33: 'EventCmdBuildAccelerationStructuresIndirectKHR',
    34: 'EventCmdCopyAccelerationStructureKHR',
    35: 'EventCmdCopyAccelerationStructureToMemoryKHR',
    36: 'EventCmdCopyMemoryToAccelerationStructureKHR',
    41: 'EventCmdDrawMeshTasksEXT',
    42: 'EventCmdDrawMeshTasksIndirectCountEXT',
    43: 'EventCmdDrawMeshTasksIndirectEXT',
    32767: 'EventUnknown',
    4294967295: 'EventInvalid',
}
EventCmdDraw = 0
EventCmdDrawIndexed = 1
EventCmdDrawIndirect = 2
EventCmdDrawIndexedIndirect = 3
EventCmdDrawIndirectCountAMD = 4
EventCmdDrawIndexedIndirectCountAMD = 5
EventCmdDispatch = 6
EventCmdDispatchIndirect = 7
EventCmdCopyBuffer = 8
EventCmdCopyImage = 9
EventCmdBlitImage = 10
EventCmdCopyBufferToImage = 11
EventCmdCopyImageToBuffer = 12
EventCmdUpdateBuffer = 13
EventCmdFillBuffer = 14
EventCmdClearColorImage = 15
EventCmdClearDepthStencilImage = 16
EventCmdClearAttachments = 17
EventCmdResolveImage = 18
EventCmdWaitEvents = 19
EventCmdPipelineBarrier = 20
EventCmdResetQueryPool = 21
EventCmdCopyQueryPoolResults = 22
EventRenderPassColorClear = 23
EventRenderPassDepthStencilClear = 24
EventRenderPassResolve = 25
EventInternalUnknown = 26
EventCmdDrawIndirectCount = 27
EventCmdDrawIndexedIndirectCount = 28
EventCmdTraceRaysKHR = 30
EventCmdTraceRaysIndirectKHR = 31
EventCmdBuildAccelerationStructuresKHR = 32
EventCmdBuildAccelerationStructuresIndirectKHR = 33
EventCmdCopyAccelerationStructureKHR = 34
EventCmdCopyAccelerationStructureToMemoryKHR = 35
EventCmdCopyMemoryToAccelerationStructureKHR = 36
EventCmdDrawMeshTasksEXT = 41
EventCmdDrawMeshTasksIndirectCountEXT = 42
EventCmdDrawMeshTasksIndirectEXT = 43
EventUnknown = 32767
EventInvalid = 4294967295
rgp_sqtt_marker_event_type = ctypes.c_uint32 # enum
class struct_rgp_sqtt_marker_event(Structure):
    pass

class union_rgp_sqtt_marker_event_0(Union):
    pass

class struct_rgp_sqtt_marker_event_0_0(Structure):
    pass

struct_rgp_sqtt_marker_event_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_event_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('ext_dwords', ctypes.c_uint32, 3),
    ('api_type', ctypes.c_uint32, 24),
    ('has_thread_dims', ctypes.c_uint32, 1),
]

union_rgp_sqtt_marker_event_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_event_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_event_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_event_0_0),
    ('dword01', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_event_1(Union):
    pass

class struct_rgp_sqtt_marker_event_1_0(Structure):
    pass

struct_rgp_sqtt_marker_event_1_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_event_1_0._fields_ = [
    ('cb_id', ctypes.c_uint32, 20),
    ('vertex_offset_reg_idx', ctypes.c_uint32, 4),
    ('instance_offset_reg_idx', ctypes.c_uint32, 4),
    ('draw_index_reg_idx', ctypes.c_uint32, 4),
]

union_rgp_sqtt_marker_event_1._pack_ = 1 # source:False
union_rgp_sqtt_marker_event_1._anonymous_ = ('_0',)
union_rgp_sqtt_marker_event_1._fields_ = [
    ('_0', struct_rgp_sqtt_marker_event_1_0),
    ('dword02', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_event_2(Union):
    pass

union_rgp_sqtt_marker_event_2._pack_ = 1 # source:False
union_rgp_sqtt_marker_event_2._fields_ = [
    ('cmd_id', ctypes.c_uint32),
    ('dword03', ctypes.c_uint32),
]

struct_rgp_sqtt_marker_event._pack_ = 1 # source:False
struct_rgp_sqtt_marker_event._anonymous_ = ('_0', '_1', '_2',)
struct_rgp_sqtt_marker_event._fields_ = [
    ('_0', union_rgp_sqtt_marker_event_0),
    ('_1', union_rgp_sqtt_marker_event_1),
    ('_2', union_rgp_sqtt_marker_event_2),
]

class struct_rgp_sqtt_marker_event_with_dims(Structure):
    pass

struct_rgp_sqtt_marker_event_with_dims._pack_ = 1 # source:False
struct_rgp_sqtt_marker_event_with_dims._fields_ = [
    ('event', struct_rgp_sqtt_marker_event),
    ('thread_x', ctypes.c_uint32),
    ('thread_y', ctypes.c_uint32),
    ('thread_z', ctypes.c_uint32),
]

class struct_rgp_sqtt_marker_barrier_start(Structure):
    pass

class union_rgp_sqtt_marker_barrier_start_0(Union):
    pass

class struct_rgp_sqtt_marker_barrier_start_0_0(Structure):
    pass

struct_rgp_sqtt_marker_barrier_start_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_barrier_start_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('ext_dwords', ctypes.c_uint32, 3),
    ('cb_id', ctypes.c_uint32, 20),
    ('reserved', ctypes.c_uint32, 5),
]

union_rgp_sqtt_marker_barrier_start_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_barrier_start_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_barrier_start_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_barrier_start_0_0),
    ('dword01', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_barrier_start_1(Union):
    pass

class struct_rgp_sqtt_marker_barrier_start_1_0(Structure):
    pass

struct_rgp_sqtt_marker_barrier_start_1_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_barrier_start_1_0._fields_ = [
    ('driver_reason', ctypes.c_uint32, 31),
    ('internal', ctypes.c_uint32, 1),
]

union_rgp_sqtt_marker_barrier_start_1._pack_ = 1 # source:False
union_rgp_sqtt_marker_barrier_start_1._anonymous_ = ('_0',)
union_rgp_sqtt_marker_barrier_start_1._fields_ = [
    ('_0', struct_rgp_sqtt_marker_barrier_start_1_0),
    ('dword02', ctypes.c_uint32),
]

struct_rgp_sqtt_marker_barrier_start._pack_ = 1 # source:False
struct_rgp_sqtt_marker_barrier_start._anonymous_ = ('_0', '_1',)
struct_rgp_sqtt_marker_barrier_start._fields_ = [
    ('_0', union_rgp_sqtt_marker_barrier_start_0),
    ('_1', union_rgp_sqtt_marker_barrier_start_1),
]

class struct_rgp_sqtt_marker_barrier_end(Structure):
    pass

class union_rgp_sqtt_marker_barrier_end_0(Union):
    pass

class struct_rgp_sqtt_marker_barrier_end_0_0(Structure):
    pass

struct_rgp_sqtt_marker_barrier_end_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_barrier_end_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('ext_dwords', ctypes.c_uint32, 3),
    ('cb_id', ctypes.c_uint32, 20),
    ('wait_on_eop_ts', ctypes.c_uint32, 1),
    ('vs_partial_flush', ctypes.c_uint32, 1),
    ('ps_partial_flush', ctypes.c_uint32, 1),
    ('cs_partial_flush', ctypes.c_uint32, 1),
    ('pfp_sync_me', ctypes.c_uint32, 1),
]

union_rgp_sqtt_marker_barrier_end_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_barrier_end_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_barrier_end_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_barrier_end_0_0),
    ('dword01', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_barrier_end_1(Union):
    pass

class struct_rgp_sqtt_marker_barrier_end_1_0(Structure):
    pass

struct_rgp_sqtt_marker_barrier_end_1_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_barrier_end_1_0._fields_ = [
    ('sync_cp_dma', ctypes.c_uint32, 1),
    ('inval_tcp', ctypes.c_uint32, 1),
    ('inval_sqI', ctypes.c_uint32, 1),
    ('inval_sqK', ctypes.c_uint32, 1),
    ('flush_tcc', ctypes.c_uint32, 1),
    ('inval_tcc', ctypes.c_uint32, 1),
    ('flush_cb', ctypes.c_uint32, 1),
    ('inval_cb', ctypes.c_uint32, 1),
    ('flush_db', ctypes.c_uint32, 1),
    ('inval_db', ctypes.c_uint32, 1),
    ('num_layout_transitions', ctypes.c_uint32, 16),
    ('inval_gl1', ctypes.c_uint32, 1),
    ('wait_on_ts', ctypes.c_uint32, 1),
    ('eop_ts_bottom_of_pipe', ctypes.c_uint32, 1),
    ('eos_ts_ps_done', ctypes.c_uint32, 1),
    ('eos_ts_cs_done', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 1),
]

union_rgp_sqtt_marker_barrier_end_1._pack_ = 1 # source:False
union_rgp_sqtt_marker_barrier_end_1._anonymous_ = ('_0',)
union_rgp_sqtt_marker_barrier_end_1._fields_ = [
    ('_0', struct_rgp_sqtt_marker_barrier_end_1_0),
    ('dword02', ctypes.c_uint32),
]

struct_rgp_sqtt_marker_barrier_end._pack_ = 1 # source:False
struct_rgp_sqtt_marker_barrier_end._anonymous_ = ('_0', '_1',)
struct_rgp_sqtt_marker_barrier_end._fields_ = [
    ('_0', union_rgp_sqtt_marker_barrier_end_0),
    ('_1', union_rgp_sqtt_marker_barrier_end_1),
]

class struct_rgp_sqtt_marker_layout_transition(Structure):
    pass

class union_rgp_sqtt_marker_layout_transition_0(Union):
    pass

class struct_rgp_sqtt_marker_layout_transition_0_0(Structure):
    pass

struct_rgp_sqtt_marker_layout_transition_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_layout_transition_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('ext_dwords', ctypes.c_uint32, 3),
    ('depth_stencil_expand', ctypes.c_uint32, 1),
    ('htile_hiz_range_expand', ctypes.c_uint32, 1),
    ('depth_stencil_resummarize', ctypes.c_uint32, 1),
    ('dcc_decompress', ctypes.c_uint32, 1),
    ('fmask_decompress', ctypes.c_uint32, 1),
    ('fast_clear_eliminate', ctypes.c_uint32, 1),
    ('fmask_color_expand', ctypes.c_uint32, 1),
    ('init_mask_ram', ctypes.c_uint32, 1),
    ('reserved1', ctypes.c_uint32, 17),
]

union_rgp_sqtt_marker_layout_transition_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_layout_transition_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_layout_transition_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_layout_transition_0_0),
    ('dword01', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_layout_transition_1(Union):
    pass

class struct_rgp_sqtt_marker_layout_transition_1_0(Structure):
    pass

struct_rgp_sqtt_marker_layout_transition_1_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_layout_transition_1_0._fields_ = [
    ('reserved2', ctypes.c_uint32, 32),
]

union_rgp_sqtt_marker_layout_transition_1._pack_ = 1 # source:False
union_rgp_sqtt_marker_layout_transition_1._anonymous_ = ('_0',)
union_rgp_sqtt_marker_layout_transition_1._fields_ = [
    ('_0', struct_rgp_sqtt_marker_layout_transition_1_0),
    ('dword02', ctypes.c_uint32),
]

struct_rgp_sqtt_marker_layout_transition._pack_ = 1 # source:False
struct_rgp_sqtt_marker_layout_transition._anonymous_ = ('_0', '_1',)
struct_rgp_sqtt_marker_layout_transition._fields_ = [
    ('_0', union_rgp_sqtt_marker_layout_transition_0),
    ('_1', union_rgp_sqtt_marker_layout_transition_1),
]

class struct_rgp_sqtt_marker_user_event(Structure):
    pass

class union_rgp_sqtt_marker_user_event_0(Union):
    pass

class struct_rgp_sqtt_marker_user_event_0_0(Structure):
    pass

struct_rgp_sqtt_marker_user_event_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_user_event_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('reserved0', ctypes.c_uint32, 8),
    ('data_type', ctypes.c_uint32, 8),
    ('reserved1', ctypes.c_uint32, 12),
]

union_rgp_sqtt_marker_user_event_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_user_event_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_user_event_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_user_event_0_0),
    ('dword01', ctypes.c_uint32),
]

struct_rgp_sqtt_marker_user_event._pack_ = 1 # source:False
struct_rgp_sqtt_marker_user_event._anonymous_ = ('_0',)
struct_rgp_sqtt_marker_user_event._fields_ = [
    ('_0', union_rgp_sqtt_marker_user_event_0),
]

class struct_rgp_sqtt_marker_user_event_with_length(Structure):
    pass

struct_rgp_sqtt_marker_user_event_with_length._pack_ = 1 # source:False
struct_rgp_sqtt_marker_user_event_with_length._fields_ = [
    ('user_event', struct_rgp_sqtt_marker_user_event),
    ('length', ctypes.c_uint32),
]


# values for enumeration 'rgp_sqtt_marker_user_event_type'
rgp_sqtt_marker_user_event_type__enumvalues = {
    0: 'UserEventTrigger',
    1: 'UserEventPop',
    2: 'UserEventPush',
    3: 'UserEventObjectName',
}
UserEventTrigger = 0
UserEventPop = 1
UserEventPush = 2
UserEventObjectName = 3
rgp_sqtt_marker_user_event_type = ctypes.c_uint32 # enum
class struct_rgp_sqtt_marker_pipeline_bind(Structure):
    pass

class union_rgp_sqtt_marker_pipeline_bind_0(Union):
    pass

class struct_rgp_sqtt_marker_pipeline_bind_0_0(Structure):
    pass

struct_rgp_sqtt_marker_pipeline_bind_0_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_pipeline_bind_0_0._fields_ = [
    ('identifier', ctypes.c_uint32, 4),
    ('ext_dwords', ctypes.c_uint32, 3),
    ('bind_point', ctypes.c_uint32, 1),
    ('cb_id', ctypes.c_uint32, 20),
    ('reserved', ctypes.c_uint32, 4),
]

union_rgp_sqtt_marker_pipeline_bind_0._pack_ = 1 # source:False
union_rgp_sqtt_marker_pipeline_bind_0._anonymous_ = ('_0',)
union_rgp_sqtt_marker_pipeline_bind_0._fields_ = [
    ('_0', struct_rgp_sqtt_marker_pipeline_bind_0_0),
    ('dword01', ctypes.c_uint32),
]

class union_rgp_sqtt_marker_pipeline_bind_1(Union):
    pass

class struct_rgp_sqtt_marker_pipeline_bind_1_0(Structure):
    pass

struct_rgp_sqtt_marker_pipeline_bind_1_0._pack_ = 1 # source:False
struct_rgp_sqtt_marker_pipeline_bind_1_0._fields_ = [
    ('dword02', ctypes.c_uint32),
    ('dword03', ctypes.c_uint32),
]

union_rgp_sqtt_marker_pipeline_bind_1._pack_ = 1 # source:False
union_rgp_sqtt_marker_pipeline_bind_1._anonymous_ = ('_0',)
union_rgp_sqtt_marker_pipeline_bind_1._fields_ = [
    ('api_pso_hash', ctypes.c_uint32 * 2),
    ('_0', struct_rgp_sqtt_marker_pipeline_bind_1_0),
]

struct_rgp_sqtt_marker_pipeline_bind._pack_ = 1 # source:False
struct_rgp_sqtt_marker_pipeline_bind._anonymous_ = ('_0', '_1',)
struct_rgp_sqtt_marker_pipeline_bind._fields_ = [
    ('_0', union_rgp_sqtt_marker_pipeline_bind_0),
    ('_1', union_rgp_sqtt_marker_pipeline_bind_1),
]

__all__ = \
    ['ApiCmdBeginQuery', 'ApiCmdBeginRenderPass',
    'ApiCmdBindDescriptorSets', 'ApiCmdBindIndexBuffer',
    'ApiCmdBindPipeline', 'ApiCmdBindVertexBuffers',
    'ApiCmdBlitImage', 'ApiCmdClearAttachments',
    'ApiCmdClearColorImage', 'ApiCmdClearDepthStencilImage',
    'ApiCmdCopyBuffer', 'ApiCmdCopyBufferToImage', 'ApiCmdCopyImage',
    'ApiCmdCopyImageToBuffer', 'ApiCmdCopyQueryPoolResults',
    'ApiCmdDispatch', 'ApiCmdDispatchIndirect', 'ApiCmdDraw',
    'ApiCmdDrawIndexed', 'ApiCmdDrawIndexedIndirect',
    'ApiCmdDrawIndexedIndirectCount',
    'ApiCmdDrawIndexedIndirectCountAMD', 'ApiCmdDrawIndirect',
    'ApiCmdDrawIndirectCount', 'ApiCmdDrawIndirectCountAMD',
    'ApiCmdDrawMeshTasksEXT', 'ApiCmdDrawMeshTasksIndirectCountEXT',
    'ApiCmdDrawMeshTasksIndirectEXT', 'ApiCmdEndQuery',
    'ApiCmdEndRenderPass', 'ApiCmdExecuteCommands',
    'ApiCmdFillBuffer', 'ApiCmdNextSubpass', 'ApiCmdPipelineBarrier',
    'ApiCmdPushConstants', 'ApiCmdResetQueryPool',
    'ApiCmdResolveImage', 'ApiCmdSetBlendConstants',
    'ApiCmdSetDepthBias', 'ApiCmdSetDepthBounds',
    'ApiCmdSetLineWidth', 'ApiCmdSetScissor',
    'ApiCmdSetStencilCompareMask', 'ApiCmdSetStencilReference',
    'ApiCmdSetStencilWriteMask', 'ApiCmdSetViewport',
    'ApiCmdUpdateBuffer', 'ApiCmdWaitEvents', 'ApiCmdWriteTimestamp',
    'ApiInvalid', 'ApiRayTracingSeparateCompiled',
    'EF_AMDGPU_MACH_AMDGCN_GFX1010', 'EF_AMDGPU_MACH_AMDGCN_GFX1030',
    'EF_AMDGPU_MACH_AMDGCN_GFX1100', 'EF_AMDGPU_MACH_AMDGCN_GFX801',
    'EF_AMDGPU_MACH_AMDGCN_GFX900', 'EventCmdBlitImage',
    'EventCmdBuildAccelerationStructuresIndirectKHR',
    'EventCmdBuildAccelerationStructuresKHR',
    'EventCmdClearAttachments', 'EventCmdClearColorImage',
    'EventCmdClearDepthStencilImage',
    'EventCmdCopyAccelerationStructureKHR',
    'EventCmdCopyAccelerationStructureToMemoryKHR',
    'EventCmdCopyBuffer', 'EventCmdCopyBufferToImage',
    'EventCmdCopyImage', 'EventCmdCopyImageToBuffer',
    'EventCmdCopyMemoryToAccelerationStructureKHR',
    'EventCmdCopyQueryPoolResults', 'EventCmdDispatch',
    'EventCmdDispatchIndirect', 'EventCmdDraw', 'EventCmdDrawIndexed',
    'EventCmdDrawIndexedIndirect', 'EventCmdDrawIndexedIndirectCount',
    'EventCmdDrawIndexedIndirectCountAMD', 'EventCmdDrawIndirect',
    'EventCmdDrawIndirectCount', 'EventCmdDrawIndirectCountAMD',
    'EventCmdDrawMeshTasksEXT',
    'EventCmdDrawMeshTasksIndirectCountEXT',
    'EventCmdDrawMeshTasksIndirectEXT', 'EventCmdFillBuffer',
    'EventCmdPipelineBarrier', 'EventCmdResetQueryPool',
    'EventCmdResolveImage', 'EventCmdTraceRaysIndirectKHR',
    'EventCmdTraceRaysKHR', 'EventCmdUpdateBuffer',
    'EventCmdWaitEvents', 'EventInternalUnknown', 'EventInvalid',
    'EventRenderPassColorClear', 'EventRenderPassDepthStencilClear',
    'EventRenderPassResolve', 'EventUnknown',
    'RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END',
    'RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START',
    'RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE',
    'RGP_SQTT_MARKER_IDENTIFIER_CB_END',
    'RGP_SQTT_MARKER_IDENTIFIER_CB_START',
    'RGP_SQTT_MARKER_IDENTIFIER_EVENT',
    'RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API',
    'RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION',
    'RGP_SQTT_MARKER_IDENTIFIER_PRESENT',
    'RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS',
    'RGP_SQTT_MARKER_IDENTIFIER_RESERVED2',
    'RGP_SQTT_MARKER_IDENTIFIER_RESERVED4',
    'RGP_SQTT_MARKER_IDENTIFIER_RESERVED5',
    'RGP_SQTT_MARKER_IDENTIFIER_RESERVED6',
    'RGP_SQTT_MARKER_IDENTIFIER_SYNC',
    'RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT',
    'SQTT_ACTIVE_PIXEL_PACKER_MASK_DWORDS',
    'SQTT_API_TYPE_DIRECTX_12', 'SQTT_API_TYPE_GENERIC',
    'SQTT_API_TYPE_OPENCL', 'SQTT_API_TYPE_VULKAN',
    'SQTT_ENGINE_TYPE_COMPUTE', 'SQTT_ENGINE_TYPE_DMA',
    'SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE',
    'SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS',
    'SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL',
    'SQTT_ENGINE_TYPE_UNIVERSAL', 'SQTT_ENGINE_TYPE_UNKNOWN',
    'SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED',
    'SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING',
    'SQTT_FILE_CHUNK_TYPE_API_INFO', 'SQTT_FILE_CHUNK_TYPE_ASIC_INFO',
    'SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION',
    'SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE',
    'SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS',
    'SQTT_FILE_CHUNK_TYPE_COUNT', 'SQTT_FILE_CHUNK_TYPE_CPU_INFO',
    'SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE',
    'SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION',
    'SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS',
    'SQTT_FILE_CHUNK_TYPE_RESERVED', 'SQTT_FILE_CHUNK_TYPE_SPM_DB',
    'SQTT_FILE_CHUNK_TYPE_SQTT_DATA',
    'SQTT_FILE_CHUNK_TYPE_SQTT_DESC', 'SQTT_FILE_MAGIC_NUMBER',
    'SQTT_FILE_VERSION_MAJOR', 'SQTT_FILE_VERSION_MINOR',
    'SQTT_GFXIP_LEVEL_GFXIP_10_1', 'SQTT_GFXIP_LEVEL_GFXIP_10_3',
    'SQTT_GFXIP_LEVEL_GFXIP_11_0', 'SQTT_GFXIP_LEVEL_GFXIP_6',
    'SQTT_GFXIP_LEVEL_GFXIP_7', 'SQTT_GFXIP_LEVEL_GFXIP_8',
    'SQTT_GFXIP_LEVEL_GFXIP_8_1', 'SQTT_GFXIP_LEVEL_GFXIP_9',
    'SQTT_GFXIP_LEVEL_NONE', 'SQTT_GPU_NAME_MAX_SIZE',
    'SQTT_GPU_TYPE_DISCRETE', 'SQTT_GPU_TYPE_INTEGRATED',
    'SQTT_GPU_TYPE_UNKNOWN', 'SQTT_GPU_TYPE_VIRTUAL',
    'SQTT_INSTRUCTION_TRACE_API_PSO',
    'SQTT_INSTRUCTION_TRACE_DISABLED',
    'SQTT_INSTRUCTION_TRACE_FULL_FRAME', 'SQTT_MAX_NUM_SE',
    'SQTT_MEMORY_TYPE_DDR', 'SQTT_MEMORY_TYPE_DDR2',
    'SQTT_MEMORY_TYPE_DDR3', 'SQTT_MEMORY_TYPE_DDR4',
    'SQTT_MEMORY_TYPE_DDR5', 'SQTT_MEMORY_TYPE_GDDR3',
    'SQTT_MEMORY_TYPE_GDDR4', 'SQTT_MEMORY_TYPE_GDDR5',
    'SQTT_MEMORY_TYPE_GDDR6', 'SQTT_MEMORY_TYPE_HBM',
    'SQTT_MEMORY_TYPE_HBM2', 'SQTT_MEMORY_TYPE_HBM3',
    'SQTT_MEMORY_TYPE_LPDDR4', 'SQTT_MEMORY_TYPE_LPDDR5',
    'SQTT_MEMORY_TYPE_UNKNOWN', 'SQTT_PROFILING_MODE_INDEX',
    'SQTT_PROFILING_MODE_PRESENT', 'SQTT_PROFILING_MODE_TAG',
    'SQTT_PROFILING_MODE_USER_MARKERS',
    'SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT',
    'SQTT_QUEUE_TIMING_EVENT_PRESENT',
    'SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE',
    'SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE',
    'SQTT_QUEUE_TYPE_COMPUTE', 'SQTT_QUEUE_TYPE_DMA',
    'SQTT_QUEUE_TYPE_UNIVERSAL', 'SQTT_QUEUE_TYPE_UNKNOWN',
    'SQTT_SA_PER_SE', 'SQTT_VERSION_2_2', 'SQTT_VERSION_2_3',
    'SQTT_VERSION_2_4', 'SQTT_VERSION_3_2', 'SQTT_VERSION_NONE',
    'UserEventObjectName', 'UserEventPop', 'UserEventPush',
    'UserEventTrigger', 'elf_gfxip_level',
    'rgp_sqtt_marker_event_type', 'rgp_sqtt_marker_general_api_type',
    'rgp_sqtt_marker_identifier', 'rgp_sqtt_marker_user_event_type',
    'sqtt_api_type', 'sqtt_engine_type',
    'sqtt_file_chunk_asic_info_flags', 'sqtt_file_chunk_type',
    'sqtt_gfxip_level', 'sqtt_gpu_type',
    'sqtt_instruction_trace_mode', 'sqtt_memory_type',
    'sqtt_profiling_mode', 'sqtt_queue_event_type', 'sqtt_queue_type',
    'sqtt_version', 'struct_rgp_sqtt_marker_barrier_end',
    'struct_rgp_sqtt_marker_barrier_end_0_0',
    'struct_rgp_sqtt_marker_barrier_end_1_0',
    'struct_rgp_sqtt_marker_barrier_start',
    'struct_rgp_sqtt_marker_barrier_start_0_0',
    'struct_rgp_sqtt_marker_barrier_start_1_0',
    'struct_rgp_sqtt_marker_cb_end',
    'struct_rgp_sqtt_marker_cb_end_0_0',
    'struct_rgp_sqtt_marker_cb_id_global_cb_id',
    'struct_rgp_sqtt_marker_cb_id_per_frame_cb_id',
    'struct_rgp_sqtt_marker_cb_start',
    'struct_rgp_sqtt_marker_cb_start_0_0',
    'struct_rgp_sqtt_marker_event',
    'struct_rgp_sqtt_marker_event_0_0',
    'struct_rgp_sqtt_marker_event_1_0',
    'struct_rgp_sqtt_marker_event_with_dims',
    'struct_rgp_sqtt_marker_general_api',
    'struct_rgp_sqtt_marker_general_api_0_0',
    'struct_rgp_sqtt_marker_layout_transition',
    'struct_rgp_sqtt_marker_layout_transition_0_0',
    'struct_rgp_sqtt_marker_layout_transition_1_0',
    'struct_rgp_sqtt_marker_pipeline_bind',
    'struct_rgp_sqtt_marker_pipeline_bind_0_0',
    'struct_rgp_sqtt_marker_pipeline_bind_1_0',
    'struct_rgp_sqtt_marker_user_event',
    'struct_rgp_sqtt_marker_user_event_0_0',
    'struct_rgp_sqtt_marker_user_event_with_length',
    'struct_sqtt_code_object_database_record',
    'struct_sqtt_code_object_loader_events_record',
    'struct_sqtt_data_info', 'struct_sqtt_data_se',
    'struct_sqtt_file_chunk_api_info',
    'struct_sqtt_file_chunk_asic_info',
    'struct_sqtt_file_chunk_clock_calibration',
    'struct_sqtt_file_chunk_code_object_database',
    'struct_sqtt_file_chunk_code_object_loader_events',
    'struct_sqtt_file_chunk_cpu_info',
    'struct_sqtt_file_chunk_header', 'struct_sqtt_file_chunk_id',
    'struct_sqtt_file_chunk_pso_correlation',
    'struct_sqtt_file_chunk_queue_event_timings',
    'struct_sqtt_file_chunk_spm_db',
    'struct_sqtt_file_chunk_sqtt_data',
    'struct_sqtt_file_chunk_sqtt_desc',
    'struct_sqtt_file_chunk_sqtt_desc_0_v0',
    'struct_sqtt_file_chunk_sqtt_desc_0_v1',
    'struct_sqtt_file_header', 'struct_sqtt_file_header_flags',
    'struct_sqtt_file_header_flags_0_0',
    'struct_sqtt_instruction_trace_data_api_pso_data',
    'struct_sqtt_instruction_trace_data_shader_engine_filter',
    'struct_sqtt_profiling_mode_data_index_profiling_data',
    'struct_sqtt_profiling_mode_data_tag_profiling_data',
    'struct_sqtt_profiling_mode_data_user_marker_profiling_data',
    'struct_sqtt_pso_correlation_record',
    'struct_sqtt_queue_event_record',
    'struct_sqtt_queue_hardware_info',
    'struct_sqtt_queue_hardware_info_0_0',
    'struct_sqtt_queue_info_record',
    'union_rgp_sqtt_marker_barrier_end_0',
    'union_rgp_sqtt_marker_barrier_end_1',
    'union_rgp_sqtt_marker_barrier_start_0',
    'union_rgp_sqtt_marker_barrier_start_1',
    'union_rgp_sqtt_marker_cb_end_0',
    'union_rgp_sqtt_marker_cb_end_1',
    'union_rgp_sqtt_marker_cb_end_2', 'union_rgp_sqtt_marker_cb_id',
    'union_rgp_sqtt_marker_cb_start_0',
    'union_rgp_sqtt_marker_cb_start_1',
    'union_rgp_sqtt_marker_cb_start_2',
    'union_rgp_sqtt_marker_cb_start_3',
    'union_rgp_sqtt_marker_event_0', 'union_rgp_sqtt_marker_event_1',
    'union_rgp_sqtt_marker_event_2',
    'union_rgp_sqtt_marker_general_api_0',
    'union_rgp_sqtt_marker_layout_transition_0',
    'union_rgp_sqtt_marker_layout_transition_1',
    'union_rgp_sqtt_marker_pipeline_bind_0',
    'union_rgp_sqtt_marker_pipeline_bind_1',
    'union_rgp_sqtt_marker_user_event_0', 'union_sqtt_data_info_0',
    'union_sqtt_file_chunk_sqtt_desc_0',
    'union_sqtt_file_header_flags_0',
    'union_sqtt_instruction_trace_data',
    'union_sqtt_profiling_mode_data',
    'union_sqtt_queue_hardware_info_0']
