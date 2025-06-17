# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util


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



_libraries = {}
_libraries['libOpenCL.so.1'] = ctypes.CDLL(ctypes.util.find_library('OpenCL'))
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





__OPENCL_CL_H = True # macro
CL_NAME_VERSION_MAX_NAME_SIZE = 64 # macro
CL_SUCCESS = 0 # macro
CL_DEVICE_NOT_FOUND = -1 # macro
CL_DEVICE_NOT_AVAILABLE = -2 # macro
CL_COMPILER_NOT_AVAILABLE = -3 # macro
CL_MEM_OBJECT_ALLOCATION_FAILURE = -4 # macro
CL_OUT_OF_RESOURCES = -5 # macro
CL_OUT_OF_HOST_MEMORY = -6 # macro
CL_PROFILING_INFO_NOT_AVAILABLE = -7 # macro
CL_MEM_COPY_OVERLAP = -8 # macro
CL_IMAGE_FORMAT_MISMATCH = -9 # macro
CL_IMAGE_FORMAT_NOT_SUPPORTED = -10 # macro
CL_BUILD_PROGRAM_FAILURE = -11 # macro
CL_MAP_FAILURE = -12 # macro
CL_MISALIGNED_SUB_BUFFER_OFFSET = -13 # macro
CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14 # macro
CL_COMPILE_PROGRAM_FAILURE = -15 # macro
CL_LINKER_NOT_AVAILABLE = -16 # macro
CL_LINK_PROGRAM_FAILURE = -17 # macro
CL_DEVICE_PARTITION_FAILED = -18 # macro
CL_KERNEL_ARG_INFO_NOT_AVAILABLE = -19 # macro
CL_INVALID_VALUE = -30 # macro
CL_INVALID_DEVICE_TYPE = -31 # macro
CL_INVALID_PLATFORM = -32 # macro
CL_INVALID_DEVICE = -33 # macro
CL_INVALID_CONTEXT = -34 # macro
CL_INVALID_QUEUE_PROPERTIES = -35 # macro
CL_INVALID_COMMAND_QUEUE = -36 # macro
CL_INVALID_HOST_PTR = -37 # macro
CL_INVALID_MEM_OBJECT = -38 # macro
CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39 # macro
CL_INVALID_IMAGE_SIZE = -40 # macro
CL_INVALID_SAMPLER = -41 # macro
CL_INVALID_BINARY = -42 # macro
CL_INVALID_BUILD_OPTIONS = -43 # macro
CL_INVALID_PROGRAM = -44 # macro
CL_INVALID_PROGRAM_EXECUTABLE = -45 # macro
CL_INVALID_KERNEL_NAME = -46 # macro
CL_INVALID_KERNEL_DEFINITION = -47 # macro
CL_INVALID_KERNEL = -48 # macro
CL_INVALID_ARG_INDEX = -49 # macro
CL_INVALID_ARG_VALUE = -50 # macro
CL_INVALID_ARG_SIZE = -51 # macro
CL_INVALID_KERNEL_ARGS = -52 # macro
CL_INVALID_WORK_DIMENSION = -53 # macro
CL_INVALID_WORK_GROUP_SIZE = -54 # macro
CL_INVALID_WORK_ITEM_SIZE = -55 # macro
CL_INVALID_GLOBAL_OFFSET = -56 # macro
CL_INVALID_EVENT_WAIT_LIST = -57 # macro
CL_INVALID_EVENT = -58 # macro
CL_INVALID_OPERATION = -59 # macro
CL_INVALID_GL_OBJECT = -60 # macro
CL_INVALID_BUFFER_SIZE = -61 # macro
CL_INVALID_MIP_LEVEL = -62 # macro
CL_INVALID_GLOBAL_WORK_SIZE = -63 # macro
CL_INVALID_PROPERTY = -64 # macro
CL_INVALID_IMAGE_DESCRIPTOR = -65 # macro
CL_INVALID_COMPILER_OPTIONS = -66 # macro
CL_INVALID_LINKER_OPTIONS = -67 # macro
CL_INVALID_DEVICE_PARTITION_COUNT = -68 # macro
CL_INVALID_PIPE_SIZE = -69 # macro
CL_INVALID_DEVICE_QUEUE = -70 # macro
CL_INVALID_SPEC_ID = -71 # macro
CL_MAX_SIZE_RESTRICTION_EXCEEDED = -72 # macro
CL_FALSE = 0 # macro
CL_TRUE = 1 # macro
CL_BLOCKING = 1 # macro
CL_NON_BLOCKING = 0 # macro
CL_PLATFORM_PROFILE = 0x0900 # macro
CL_PLATFORM_VERSION = 0x0901 # macro
CL_PLATFORM_NAME = 0x0902 # macro
CL_PLATFORM_VENDOR = 0x0903 # macro
CL_PLATFORM_EXTENSIONS = 0x0904 # macro
CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905 # macro
CL_PLATFORM_NUMERIC_VERSION = 0x0906 # macro
CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907 # macro
CL_DEVICE_TYPE_DEFAULT = (1<<0) # macro
CL_DEVICE_TYPE_CPU = (1<<1) # macro
CL_DEVICE_TYPE_GPU = (1<<2) # macro
CL_DEVICE_TYPE_ACCELERATOR = (1<<3) # macro
CL_DEVICE_TYPE_CUSTOM = (1<<4) # macro
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF # macro
CL_DEVICE_TYPE = 0x1000 # macro
CL_DEVICE_VENDOR_ID = 0x1001 # macro
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002 # macro
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003 # macro
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004 # macro
CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100B # macro
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C # macro
CL_DEVICE_ADDRESS_BITS = 0x100D # macro
CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100E # macro
CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100F # macro
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010 # macro
CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011 # macro
CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012 # macro
CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013 # macro
CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014 # macro
CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015 # macro
CL_DEVICE_IMAGE_SUPPORT = 0x1016 # macro
CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017 # macro
CL_DEVICE_MAX_SAMPLERS = 0x1018 # macro
CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019 # macro
CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A # macro
CL_DEVICE_SINGLE_FP_CONFIG = 0x101B # macro
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C # macro
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D # macro
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E # macro
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F # macro
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020 # macro
CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021 # macro
CL_DEVICE_LOCAL_MEM_TYPE = 0x1022 # macro
CL_DEVICE_LOCAL_MEM_SIZE = 0x1023 # macro
CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024 # macro
CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025 # macro
CL_DEVICE_ENDIAN_LITTLE = 0x1026 # macro
CL_DEVICE_AVAILABLE = 0x1027 # macro
CL_DEVICE_COMPILER_AVAILABLE = 0x1028 # macro
CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029 # macro
CL_DEVICE_QUEUE_PROPERTIES = 0x102A # macro
CL_DEVICE_QUEUE_ON_HOST_PROPERTIES = 0x102A # macro
CL_DEVICE_NAME = 0x102B # macro
CL_DEVICE_VENDOR = 0x102C # macro
CL_DRIVER_VERSION = 0x102D # macro
CL_DEVICE_PROFILE = 0x102E # macro
CL_DEVICE_VERSION = 0x102F # macro
CL_DEVICE_EXTENSIONS = 0x1030 # macro
CL_DEVICE_PLATFORM = 0x1031 # macro
CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034 # macro
CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C # macro
CL_DEVICE_OPENCL_C_VERSION = 0x103D # macro
CL_DEVICE_LINKER_AVAILABLE = 0x103E # macro
CL_DEVICE_BUILT_IN_KERNELS = 0x103F # macro
CL_DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040 # macro
CL_DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041 # macro
CL_DEVICE_PARENT_DEVICE = 0x1042 # macro
CL_DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043 # macro
CL_DEVICE_PARTITION_PROPERTIES = 0x1044 # macro
CL_DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045 # macro
CL_DEVICE_PARTITION_TYPE = 0x1046 # macro
CL_DEVICE_REFERENCE_COUNT = 0x1047 # macro
CL_DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048 # macro
CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049 # macro
CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104A # macro
CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104B # macro
CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS = 0x104C # macro
CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE = 0x104D # macro
CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES = 0x104E # macro
CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE = 0x104F # macro
CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = 0x1050 # macro
CL_DEVICE_MAX_ON_DEVICE_QUEUES = 0x1051 # macro
CL_DEVICE_MAX_ON_DEVICE_EVENTS = 0x1052 # macro
CL_DEVICE_SVM_CAPABILITIES = 0x1053 # macro
CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = 0x1054 # macro
CL_DEVICE_MAX_PIPE_ARGS = 0x1055 # macro
CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056 # macro
CL_DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057 # macro
CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058 # macro
CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059 # macro
CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105A # macro
CL_DEVICE_IL_VERSION = 0x105B # macro
CL_DEVICE_MAX_NUM_SUB_GROUPS = 0x105C # macro
CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105D # macro
CL_DEVICE_NUMERIC_VERSION = 0x105E # macro
CL_DEVICE_EXTENSIONS_WITH_VERSION = 0x1060 # macro
CL_DEVICE_ILS_WITH_VERSION = 0x1061 # macro
CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION = 0x1062 # macro
CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES = 0x1063 # macro
CL_DEVICE_ATOMIC_FENCE_CAPABILITIES = 0x1064 # macro
CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT = 0x1065 # macro
CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066 # macro
CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x1067 # macro
CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = 0x1068 # macro
CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT = 0x1069 # macro
CL_DEVICE_OPENCL_C_FEATURES = 0x106F # macro
CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES = 0x1070 # macro
CL_DEVICE_PIPE_SUPPORT = 0x1071 # macro
CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED = 0x1072 # macro
CL_FP_DENORM = (1<<0) # macro
CL_FP_INF_NAN = (1<<1) # macro
CL_FP_ROUND_TO_NEAREST = (1<<2) # macro
CL_FP_ROUND_TO_ZERO = (1<<3) # macro
CL_FP_ROUND_TO_INF = (1<<4) # macro
CL_FP_FMA = (1<<5) # macro
CL_FP_SOFT_FLOAT = (1<<6) # macro
CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = (1<<7) # macro
CL_NONE = 0x0 # macro
CL_READ_ONLY_CACHE = 0x1 # macro
CL_READ_WRITE_CACHE = 0x2 # macro
CL_LOCAL = 0x1 # macro
CL_GLOBAL = 0x2 # macro
CL_EXEC_KERNEL = (1<<0) # macro
CL_EXEC_NATIVE_KERNEL = (1<<1) # macro
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = (1<<0) # macro
CL_QUEUE_PROFILING_ENABLE = (1<<1) # macro
CL_QUEUE_ON_DEVICE = (1<<2) # macro
CL_QUEUE_ON_DEVICE_DEFAULT = (1<<3) # macro
CL_CONTEXT_REFERENCE_COUNT = 0x1080 # macro
CL_CONTEXT_DEVICES = 0x1081 # macro
CL_CONTEXT_PROPERTIES = 0x1082 # macro
CL_CONTEXT_NUM_DEVICES = 0x1083 # macro
CL_CONTEXT_PLATFORM = 0x1084 # macro
CL_CONTEXT_INTEROP_USER_SYNC = 0x1085 # macro
CL_DEVICE_PARTITION_EQUALLY = 0x1086 # macro
CL_DEVICE_PARTITION_BY_COUNTS = 0x1087 # macro
CL_DEVICE_PARTITION_BY_COUNTS_LIST_END = 0x0 # macro
CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN = 0x1088 # macro
CL_DEVICE_AFFINITY_DOMAIN_NUMA = (1<<0) # macro
CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE = (1<<1) # macro
CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE = (1<<2) # macro
CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE = (1<<3) # macro
CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE = (1<<4) # macro
CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE = (1<<5) # macro
CL_DEVICE_SVM_COARSE_GRAIN_BUFFER = (1<<0) # macro
CL_DEVICE_SVM_FINE_GRAIN_BUFFER = (1<<1) # macro
CL_DEVICE_SVM_FINE_GRAIN_SYSTEM = (1<<2) # macro
CL_DEVICE_SVM_ATOMICS = (1<<3) # macro
CL_QUEUE_CONTEXT = 0x1090 # macro
CL_QUEUE_DEVICE = 0x1091 # macro
CL_QUEUE_REFERENCE_COUNT = 0x1092 # macro
CL_QUEUE_PROPERTIES = 0x1093 # macro
CL_QUEUE_SIZE = 0x1094 # macro
CL_QUEUE_DEVICE_DEFAULT = 0x1095 # macro
CL_QUEUE_PROPERTIES_ARRAY = 0x1098 # macro
CL_MEM_READ_WRITE = (1<<0) # macro
CL_MEM_WRITE_ONLY = (1<<1) # macro
CL_MEM_READ_ONLY = (1<<2) # macro
CL_MEM_USE_HOST_PTR = (1<<3) # macro
CL_MEM_ALLOC_HOST_PTR = (1<<4) # macro
CL_MEM_COPY_HOST_PTR = (1<<5) # macro
CL_MEM_HOST_WRITE_ONLY = (1<<7) # macro
CL_MEM_HOST_READ_ONLY = (1<<8) # macro
CL_MEM_HOST_NO_ACCESS = (1<<9) # macro
CL_MEM_SVM_FINE_GRAIN_BUFFER = (1<<10) # macro
CL_MEM_SVM_ATOMICS = (1<<11) # macro
CL_MEM_KERNEL_READ_AND_WRITE = (1<<12) # macro
CL_MIGRATE_MEM_OBJECT_HOST = (1<<0) # macro
CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED = (1<<1) # macro
CL_R = 0x10B0 # macro
CL_A = 0x10B1 # macro
CL_RG = 0x10B2 # macro
CL_RA = 0x10B3 # macro
CL_RGB = 0x10B4 # macro
CL_RGBA = 0x10B5 # macro
CL_BGRA = 0x10B6 # macro
CL_ARGB = 0x10B7 # macro
CL_INTENSITY = 0x10B8 # macro
CL_LUMINANCE = 0x10B9 # macro
CL_Rx = 0x10BA # macro
CL_RGx = 0x10BB # macro
CL_RGBx = 0x10BC # macro
CL_DEPTH = 0x10BD # macro
CL_DEPTH_STENCIL = 0x10BE # macro
CL_sRGB = 0x10BF # macro
CL_sRGBx = 0x10C0 # macro
CL_sRGBA = 0x10C1 # macro
CL_sBGRA = 0x10C2 # macro
CL_ABGR = 0x10C3 # macro
CL_SNORM_INT8 = 0x10D0 # macro
CL_SNORM_INT16 = 0x10D1 # macro
CL_UNORM_INT8 = 0x10D2 # macro
CL_UNORM_INT16 = 0x10D3 # macro
CL_UNORM_SHORT_565 = 0x10D4 # macro
CL_UNORM_SHORT_555 = 0x10D5 # macro
CL_UNORM_INT_101010 = 0x10D6 # macro
CL_SIGNED_INT8 = 0x10D7 # macro
CL_SIGNED_INT16 = 0x10D8 # macro
CL_SIGNED_INT32 = 0x10D9 # macro
CL_UNSIGNED_INT8 = 0x10DA # macro
CL_UNSIGNED_INT16 = 0x10DB # macro
CL_UNSIGNED_INT32 = 0x10DC # macro
CL_HALF_FLOAT = 0x10DD # macro
CL_FLOAT = 0x10DE # macro
CL_UNORM_INT24 = 0x10DF # macro
CL_UNORM_INT_101010_2 = 0x10E0 # macro
CL_MEM_OBJECT_BUFFER = 0x10F0 # macro
CL_MEM_OBJECT_IMAGE2D = 0x10F1 # macro
CL_MEM_OBJECT_IMAGE3D = 0x10F2 # macro
CL_MEM_OBJECT_IMAGE2D_ARRAY = 0x10F3 # macro
CL_MEM_OBJECT_IMAGE1D = 0x10F4 # macro
CL_MEM_OBJECT_IMAGE1D_ARRAY = 0x10F5 # macro
CL_MEM_OBJECT_IMAGE1D_BUFFER = 0x10F6 # macro
CL_MEM_OBJECT_PIPE = 0x10F7 # macro
CL_MEM_TYPE = 0x1100 # macro
CL_MEM_FLAGS = 0x1101 # macro
CL_MEM_SIZE = 0x1102 # macro
CL_MEM_HOST_PTR = 0x1103 # macro
CL_MEM_MAP_COUNT = 0x1104 # macro
CL_MEM_REFERENCE_COUNT = 0x1105 # macro
CL_MEM_CONTEXT = 0x1106 # macro
CL_MEM_ASSOCIATED_MEMOBJECT = 0x1107 # macro
CL_MEM_OFFSET = 0x1108 # macro
CL_MEM_USES_SVM_POINTER = 0x1109 # macro
CL_MEM_PROPERTIES = 0x110A # macro
CL_IMAGE_FORMAT = 0x1110 # macro
CL_IMAGE_ELEMENT_SIZE = 0x1111 # macro
CL_IMAGE_ROW_PITCH = 0x1112 # macro
CL_IMAGE_SLICE_PITCH = 0x1113 # macro
CL_IMAGE_WIDTH = 0x1114 # macro
CL_IMAGE_HEIGHT = 0x1115 # macro
CL_IMAGE_DEPTH = 0x1116 # macro
CL_IMAGE_ARRAY_SIZE = 0x1117 # macro
CL_IMAGE_BUFFER = 0x1118 # macro
CL_IMAGE_NUM_MIP_LEVELS = 0x1119 # macro
CL_IMAGE_NUM_SAMPLES = 0x111A # macro
CL_PIPE_PACKET_SIZE = 0x1120 # macro
CL_PIPE_MAX_PACKETS = 0x1121 # macro
CL_PIPE_PROPERTIES = 0x1122 # macro
CL_ADDRESS_NONE = 0x1130 # macro
CL_ADDRESS_CLAMP_TO_EDGE = 0x1131 # macro
CL_ADDRESS_CLAMP = 0x1132 # macro
CL_ADDRESS_REPEAT = 0x1133 # macro
CL_ADDRESS_MIRRORED_REPEAT = 0x1134 # macro
CL_FILTER_NEAREST = 0x1140 # macro
CL_FILTER_LINEAR = 0x1141 # macro
CL_SAMPLER_REFERENCE_COUNT = 0x1150 # macro
CL_SAMPLER_CONTEXT = 0x1151 # macro
CL_SAMPLER_NORMALIZED_COORDS = 0x1152 # macro
CL_SAMPLER_ADDRESSING_MODE = 0x1153 # macro
CL_SAMPLER_FILTER_MODE = 0x1154 # macro
CL_SAMPLER_MIP_FILTER_MODE = 0x1155 # macro
CL_SAMPLER_LOD_MIN = 0x1156 # macro
CL_SAMPLER_LOD_MAX = 0x1157 # macro
CL_SAMPLER_PROPERTIES = 0x1158 # macro
CL_MAP_READ = (1<<0) # macro
CL_MAP_WRITE = (1<<1) # macro
CL_MAP_WRITE_INVALIDATE_REGION = (1<<2) # macro
CL_PROGRAM_REFERENCE_COUNT = 0x1160 # macro
CL_PROGRAM_CONTEXT = 0x1161 # macro
CL_PROGRAM_NUM_DEVICES = 0x1162 # macro
CL_PROGRAM_DEVICES = 0x1163 # macro
CL_PROGRAM_SOURCE = 0x1164 # macro
CL_PROGRAM_BINARY_SIZES = 0x1165 # macro
CL_PROGRAM_BINARIES = 0x1166 # macro
CL_PROGRAM_NUM_KERNELS = 0x1167 # macro
CL_PROGRAM_KERNEL_NAMES = 0x1168 # macro
CL_PROGRAM_IL = 0x1169 # macro
CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT = 0x116A # macro
CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT = 0x116B # macro
CL_PROGRAM_BUILD_STATUS = 0x1181 # macro
CL_PROGRAM_BUILD_OPTIONS = 0x1182 # macro
CL_PROGRAM_BUILD_LOG = 0x1183 # macro
CL_PROGRAM_BINARY_TYPE = 0x1184 # macro
CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE = 0x1185 # macro
CL_PROGRAM_BINARY_TYPE_NONE = 0x0 # macro
CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 0x1 # macro
CL_PROGRAM_BINARY_TYPE_LIBRARY = 0x2 # macro
CL_PROGRAM_BINARY_TYPE_EXECUTABLE = 0x4 # macro
CL_BUILD_SUCCESS = 0 # macro
CL_BUILD_NONE = -1 # macro
CL_BUILD_ERROR = -2 # macro
CL_BUILD_IN_PROGRESS = -3 # macro
CL_KERNEL_FUNCTION_NAME = 0x1190 # macro
CL_KERNEL_NUM_ARGS = 0x1191 # macro
CL_KERNEL_REFERENCE_COUNT = 0x1192 # macro
CL_KERNEL_CONTEXT = 0x1193 # macro
CL_KERNEL_PROGRAM = 0x1194 # macro
CL_KERNEL_ATTRIBUTES = 0x1195 # macro
CL_KERNEL_ARG_ADDRESS_QUALIFIER = 0x1196 # macro
CL_KERNEL_ARG_ACCESS_QUALIFIER = 0x1197 # macro
CL_KERNEL_ARG_TYPE_NAME = 0x1198 # macro
CL_KERNEL_ARG_TYPE_QUALIFIER = 0x1199 # macro
CL_KERNEL_ARG_NAME = 0x119A # macro
CL_KERNEL_ARG_ADDRESS_GLOBAL = 0x119B # macro
CL_KERNEL_ARG_ADDRESS_LOCAL = 0x119C # macro
CL_KERNEL_ARG_ADDRESS_CONSTANT = 0x119D # macro
CL_KERNEL_ARG_ADDRESS_PRIVATE = 0x119E # macro
CL_KERNEL_ARG_ACCESS_READ_ONLY = 0x11A0 # macro
CL_KERNEL_ARG_ACCESS_WRITE_ONLY = 0x11A1 # macro
CL_KERNEL_ARG_ACCESS_READ_WRITE = 0x11A2 # macro
CL_KERNEL_ARG_ACCESS_NONE = 0x11A3 # macro
CL_KERNEL_ARG_TYPE_NONE = 0 # macro
CL_KERNEL_ARG_TYPE_CONST = (1<<0) # macro
CL_KERNEL_ARG_TYPE_RESTRICT = (1<<1) # macro
CL_KERNEL_ARG_TYPE_VOLATILE = (1<<2) # macro
CL_KERNEL_ARG_TYPE_PIPE = (1<<3) # macro
CL_KERNEL_WORK_GROUP_SIZE = 0x11B0 # macro
CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1 # macro
CL_KERNEL_LOCAL_MEM_SIZE = 0x11B2 # macro
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3 # macro
CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4 # macro
CL_KERNEL_GLOBAL_WORK_SIZE = 0x11B5 # macro
CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE = 0x2033 # macro
CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE = 0x2034 # macro
CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT = 0x11B8 # macro
CL_KERNEL_MAX_NUM_SUB_GROUPS = 0x11B9 # macro
CL_KERNEL_COMPILE_NUM_SUB_GROUPS = 0x11BA # macro
CL_KERNEL_EXEC_INFO_SVM_PTRS = 0x11B6 # macro
CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM = 0x11B7 # macro
CL_EVENT_COMMAND_QUEUE = 0x11D0 # macro
CL_EVENT_COMMAND_TYPE = 0x11D1 # macro
CL_EVENT_REFERENCE_COUNT = 0x11D2 # macro
CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11D3 # macro
CL_EVENT_CONTEXT = 0x11D4 # macro
CL_COMMAND_NDRANGE_KERNEL = 0x11F0 # macro
CL_COMMAND_TASK = 0x11F1 # macro
CL_COMMAND_NATIVE_KERNEL = 0x11F2 # macro
CL_COMMAND_READ_BUFFER = 0x11F3 # macro
CL_COMMAND_WRITE_BUFFER = 0x11F4 # macro
CL_COMMAND_COPY_BUFFER = 0x11F5 # macro
CL_COMMAND_READ_IMAGE = 0x11F6 # macro
CL_COMMAND_WRITE_IMAGE = 0x11F7 # macro
CL_COMMAND_COPY_IMAGE = 0x11F8 # macro
CL_COMMAND_COPY_IMAGE_TO_BUFFER = 0x11F9 # macro
CL_COMMAND_COPY_BUFFER_TO_IMAGE = 0x11FA # macro
CL_COMMAND_MAP_BUFFER = 0x11FB # macro
CL_COMMAND_MAP_IMAGE = 0x11FC # macro
CL_COMMAND_UNMAP_MEM_OBJECT = 0x11FD # macro
CL_COMMAND_MARKER = 0x11FE # macro
CL_COMMAND_ACQUIRE_GL_OBJECTS = 0x11FF # macro
CL_COMMAND_RELEASE_GL_OBJECTS = 0x1200 # macro
CL_COMMAND_READ_BUFFER_RECT = 0x1201 # macro
CL_COMMAND_WRITE_BUFFER_RECT = 0x1202 # macro
CL_COMMAND_COPY_BUFFER_RECT = 0x1203 # macro
CL_COMMAND_USER = 0x1204 # macro
CL_COMMAND_BARRIER = 0x1205 # macro
CL_COMMAND_MIGRATE_MEM_OBJECTS = 0x1206 # macro
CL_COMMAND_FILL_BUFFER = 0x1207 # macro
CL_COMMAND_FILL_IMAGE = 0x1208 # macro
CL_COMMAND_SVM_FREE = 0x1209 # macro
CL_COMMAND_SVM_MEMCPY = 0x120A # macro
CL_COMMAND_SVM_MEMFILL = 0x120B # macro
CL_COMMAND_SVM_MAP = 0x120C # macro
CL_COMMAND_SVM_UNMAP = 0x120D # macro
CL_COMMAND_SVM_MIGRATE_MEM = 0x120E # macro
CL_COMPLETE = 0x0 # macro
CL_RUNNING = 0x1 # macro
CL_SUBMITTED = 0x2 # macro
CL_QUEUED = 0x3 # macro
CL_BUFFER_CREATE_TYPE_REGION = 0x1220 # macro
CL_PROFILING_COMMAND_QUEUED = 0x1280 # macro
CL_PROFILING_COMMAND_SUBMIT = 0x1281 # macro
CL_PROFILING_COMMAND_START = 0x1282 # macro
CL_PROFILING_COMMAND_END = 0x1283 # macro
CL_PROFILING_COMMAND_COMPLETE = 0x1284 # macro
CL_DEVICE_ATOMIC_ORDER_RELAXED = (1<<0) # macro
CL_DEVICE_ATOMIC_ORDER_ACQ_REL = (1<<1) # macro
CL_DEVICE_ATOMIC_ORDER_SEQ_CST = (1<<2) # macro
CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM = (1<<3) # macro
CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP = (1<<4) # macro
CL_DEVICE_ATOMIC_SCOPE_DEVICE = (1<<5) # macro
CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES = (1<<6) # macro
CL_DEVICE_QUEUE_SUPPORTED = (1<<0) # macro
CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT = (1<<1) # macro
CL_KHRONOS_VENDOR_ID_CODEPLAY = 0x10004 # macro
CL_VERSION_MAJOR_BITS = (10) # macro
CL_VERSION_MINOR_BITS = (10) # macro
CL_VERSION_PATCH_BITS = (12) # macro
CL_VERSION_MAJOR_MASK = ((1<<(10))-1) # macro
CL_VERSION_MINOR_MASK = ((1<<(10))-1) # macro
CL_VERSION_PATCH_MASK = ((1<<(12))-1) # macro
def CL_VERSION_MAJOR(version):  # macro
   return ((version)>>((10)+(12)))
def CL_VERSION_MINOR(version):  # macro
   return (((version)>>(12))&((1<<(10))-1))
def CL_VERSION_PATCH(version):  # macro
   return ((version)&((1<<(12))-1))
def CL_MAKE_VERSION(major, minor, patch):  # macro
   return ((((major)&((1<<(10))-1))<<((10)+(12)))|(((minor)&((1<<(10))-1))<<(12))|((patch)&((1<<(12))-1)))
class struct__cl_platform_id(Structure):
    pass

cl_platform_id = ctypes.POINTER(struct__cl_platform_id)
class struct__cl_device_id(Structure):
    pass

cl_device_id = ctypes.POINTER(struct__cl_device_id)
class struct__cl_context(Structure):
    pass

cl_context = ctypes.POINTER(struct__cl_context)
class struct__cl_command_queue(Structure):
    pass

cl_command_queue = ctypes.POINTER(struct__cl_command_queue)
class struct__cl_mem(Structure):
    pass

cl_mem = ctypes.POINTER(struct__cl_mem)
class struct__cl_program(Structure):
    pass

cl_program = ctypes.POINTER(struct__cl_program)
class struct__cl_kernel(Structure):
    pass

cl_kernel = ctypes.POINTER(struct__cl_kernel)
class struct__cl_event(Structure):
    pass

cl_event = ctypes.POINTER(struct__cl_event)
class struct__cl_sampler(Structure):
    pass

cl_sampler = ctypes.POINTER(struct__cl_sampler)
cl_bool = ctypes.c_uint32
cl_bitfield = ctypes.c_uint64
cl_properties = ctypes.c_uint64
cl_device_type = ctypes.c_uint64
cl_platform_info = ctypes.c_uint32
cl_device_info = ctypes.c_uint32
cl_device_fp_config = ctypes.c_uint64
cl_device_mem_cache_type = ctypes.c_uint32
cl_device_local_mem_type = ctypes.c_uint32
cl_device_exec_capabilities = ctypes.c_uint64
cl_device_svm_capabilities = ctypes.c_uint64
cl_command_queue_properties = ctypes.c_uint64
cl_device_partition_property = ctypes.c_int64
cl_device_affinity_domain = ctypes.c_uint64
cl_context_properties = ctypes.c_int64
cl_context_info = ctypes.c_uint32
cl_queue_properties = ctypes.c_uint64
cl_command_queue_info = ctypes.c_uint32
cl_channel_order = ctypes.c_uint32
cl_channel_type = ctypes.c_uint32
cl_mem_flags = ctypes.c_uint64
cl_svm_mem_flags = ctypes.c_uint64
cl_mem_object_type = ctypes.c_uint32
cl_mem_info = ctypes.c_uint32
cl_mem_migration_flags = ctypes.c_uint64
cl_image_info = ctypes.c_uint32
cl_buffer_create_type = ctypes.c_uint32
cl_addressing_mode = ctypes.c_uint32
cl_filter_mode = ctypes.c_uint32
cl_sampler_info = ctypes.c_uint32
cl_map_flags = ctypes.c_uint64
cl_pipe_properties = ctypes.c_int64
cl_pipe_info = ctypes.c_uint32
cl_program_info = ctypes.c_uint32
cl_program_build_info = ctypes.c_uint32
cl_program_binary_type = ctypes.c_uint32
cl_build_status = ctypes.c_int32
cl_kernel_info = ctypes.c_uint32
cl_kernel_arg_info = ctypes.c_uint32
cl_kernel_arg_address_qualifier = ctypes.c_uint32
cl_kernel_arg_access_qualifier = ctypes.c_uint32
cl_kernel_arg_type_qualifier = ctypes.c_uint64
cl_kernel_work_group_info = ctypes.c_uint32
cl_kernel_sub_group_info = ctypes.c_uint32
cl_event_info = ctypes.c_uint32
cl_command_type = ctypes.c_uint32
cl_profiling_info = ctypes.c_uint32
cl_sampler_properties = ctypes.c_uint64
cl_kernel_exec_info = ctypes.c_uint32
cl_device_atomic_capabilities = ctypes.c_uint64
cl_device_device_enqueue_capabilities = ctypes.c_uint64
cl_khronos_vendor_id = ctypes.c_uint32
cl_mem_properties = ctypes.c_uint64
cl_version = ctypes.c_uint32
class struct__cl_image_format(Structure):
    pass

struct__cl_image_format._pack_ = 1 # source:False
struct__cl_image_format._fields_ = [
    ('image_channel_order', ctypes.c_uint32),
    ('image_channel_data_type', ctypes.c_uint32),
]

cl_image_format = struct__cl_image_format
class struct__cl_image_desc(Structure):
    pass

class union__cl_image_desc_0(Union):
    pass

union__cl_image_desc_0._pack_ = 1 # source:False
union__cl_image_desc_0._fields_ = [
    ('buffer', ctypes.POINTER(struct__cl_mem)),
    ('mem_object', ctypes.POINTER(struct__cl_mem)),
]

struct__cl_image_desc._pack_ = 1 # source:False
struct__cl_image_desc._anonymous_ = ('_0',)
struct__cl_image_desc._fields_ = [
    ('image_type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('image_width', ctypes.c_uint64),
    ('image_height', ctypes.c_uint64),
    ('image_depth', ctypes.c_uint64),
    ('image_array_size', ctypes.c_uint64),
    ('image_row_pitch', ctypes.c_uint64),
    ('image_slice_pitch', ctypes.c_uint64),
    ('num_mip_levels', ctypes.c_uint32),
    ('num_samples', ctypes.c_uint32),
    ('_0', union__cl_image_desc_0),
]

cl_image_desc = struct__cl_image_desc
class struct__cl_buffer_region(Structure):
    pass

struct__cl_buffer_region._pack_ = 1 # source:False
struct__cl_buffer_region._fields_ = [
    ('origin', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

cl_buffer_region = struct__cl_buffer_region
class struct__cl_name_version(Structure):
    pass

struct__cl_name_version._pack_ = 1 # source:False
struct__cl_name_version._fields_ = [
    ('version', ctypes.c_uint32),
    ('name', ctypes.c_char * 64),
]

cl_name_version = struct__cl_name_version
cl_int = ctypes.c_int32
cl_uint = ctypes.c_uint32
try:
    clGetPlatformIDs = _libraries['libOpenCL.so.1'].clGetPlatformIDs
    clGetPlatformIDs.restype = cl_int
    clGetPlatformIDs.argtypes = [cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_platform_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    clGetPlatformInfo = _libraries['libOpenCL.so.1'].clGetPlatformInfo
    clGetPlatformInfo.restype = cl_int
    clGetPlatformInfo.argtypes = [cl_platform_id, cl_platform_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetDeviceIDs = _libraries['libOpenCL.so.1'].clGetDeviceIDs
    clGetDeviceIDs.restype = cl_int
    clGetDeviceIDs.argtypes = [cl_platform_id, cl_device_type, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clGetDeviceInfo = _libraries['libOpenCL.so.1'].clGetDeviceInfo
    clGetDeviceInfo.restype = cl_int
    clGetDeviceInfo.argtypes = [cl_device_id, cl_device_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateSubDevices = _libraries['libOpenCL.so.1'].clCreateSubDevices
    clCreateSubDevices.restype = cl_int
    clCreateSubDevices.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_int64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clRetainDevice = _libraries['libOpenCL.so.1'].clRetainDevice
    clRetainDevice.restype = cl_int
    clRetainDevice.argtypes = [cl_device_id]
except AttributeError:
    pass
try:
    clReleaseDevice = _libraries['libOpenCL.so.1'].clReleaseDevice
    clReleaseDevice.restype = cl_int
    clReleaseDevice.argtypes = [cl_device_id]
except AttributeError:
    pass
try:
    clSetDefaultDeviceCommandQueue = _libraries['libOpenCL.so.1'].clSetDefaultDeviceCommandQueue
    clSetDefaultDeviceCommandQueue.restype = cl_int
    clSetDefaultDeviceCommandQueue.argtypes = [cl_context, cl_device_id, cl_command_queue]
except AttributeError:
    pass
try:
    clGetDeviceAndHostTimer = _libraries['libOpenCL.so.1'].clGetDeviceAndHostTimer
    clGetDeviceAndHostTimer.restype = cl_int
    clGetDeviceAndHostTimer.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetHostTimer = _libraries['libOpenCL.so.1'].clGetHostTimer
    clGetHostTimer.restype = cl_int
    clGetHostTimer.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateContext = _libraries['libOpenCL.so.1'].clCreateContext
    clCreateContext.restype = cl_context
    clCreateContext.argtypes = [ctypes.POINTER(ctypes.c_int64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateContextFromType = _libraries['libOpenCL.so.1'].clCreateContextFromType
    clCreateContextFromType.restype = cl_context
    clCreateContextFromType.argtypes = [ctypes.POINTER(ctypes.c_int64), cl_device_type, ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainContext = _libraries['libOpenCL.so.1'].clRetainContext
    clRetainContext.restype = cl_int
    clRetainContext.argtypes = [cl_context]
except AttributeError:
    pass
try:
    clReleaseContext = _libraries['libOpenCL.so.1'].clReleaseContext
    clReleaseContext.restype = cl_int
    clReleaseContext.argtypes = [cl_context]
except AttributeError:
    pass
try:
    clGetContextInfo = _libraries['libOpenCL.so.1'].clGetContextInfo
    clGetContextInfo.restype = cl_int
    clGetContextInfo.argtypes = [cl_context, cl_context_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clSetContextDestructorCallback = _libraries['libOpenCL.so.1'].clSetContextDestructorCallback
    clSetContextDestructorCallback.restype = cl_int
    clSetContextDestructorCallback.argtypes = [cl_context, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_context), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCreateCommandQueueWithProperties = _libraries['libOpenCL.so.1'].clCreateCommandQueueWithProperties
    clCreateCommandQueueWithProperties.restype = cl_command_queue
    clCreateCommandQueueWithProperties.argtypes = [cl_context, cl_device_id, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainCommandQueue = _libraries['libOpenCL.so.1'].clRetainCommandQueue
    clRetainCommandQueue.restype = cl_int
    clRetainCommandQueue.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clReleaseCommandQueue = _libraries['libOpenCL.so.1'].clReleaseCommandQueue
    clReleaseCommandQueue.restype = cl_int
    clReleaseCommandQueue.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clGetCommandQueueInfo = _libraries['libOpenCL.so.1'].clGetCommandQueueInfo
    clGetCommandQueueInfo.restype = cl_int
    clGetCommandQueueInfo.argtypes = [cl_command_queue, cl_command_queue_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateBuffer = _libraries['libOpenCL.so.1'].clCreateBuffer
    clCreateBuffer.restype = cl_mem
    clCreateBuffer.argtypes = [cl_context, cl_mem_flags, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateSubBuffer = _libraries['libOpenCL.so.1'].clCreateSubBuffer
    clCreateSubBuffer.restype = cl_mem
    clCreateSubBuffer.argtypes = [cl_mem, cl_mem_flags, cl_buffer_create_type, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImage = _libraries['libOpenCL.so.1'].clCreateImage
    clCreateImage.restype = cl_mem
    clCreateImage.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(struct__cl_image_desc), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreatePipe = _libraries['libOpenCL.so.1'].clCreatePipe
    clCreatePipe.restype = cl_mem
    clCreatePipe.argtypes = [cl_context, cl_mem_flags, cl_uint, cl_uint, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateBufferWithProperties = _libraries['libOpenCL.so.1'].clCreateBufferWithProperties
    clCreateBufferWithProperties.restype = cl_mem
    clCreateBufferWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), cl_mem_flags, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImageWithProperties = _libraries['libOpenCL.so.1'].clCreateImageWithProperties
    clCreateImageWithProperties.restype = cl_mem
    clCreateImageWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), cl_mem_flags, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(struct__cl_image_desc), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainMemObject = _libraries['libOpenCL.so.1'].clRetainMemObject
    clRetainMemObject.restype = cl_int
    clRetainMemObject.argtypes = [cl_mem]
except AttributeError:
    pass
try:
    clReleaseMemObject = _libraries['libOpenCL.so.1'].clReleaseMemObject
    clReleaseMemObject.restype = cl_int
    clReleaseMemObject.argtypes = [cl_mem]
except AttributeError:
    pass
try:
    clGetSupportedImageFormats = _libraries['libOpenCL.so.1'].clGetSupportedImageFormats
    clGetSupportedImageFormats.restype = cl_int
    clGetSupportedImageFormats.argtypes = [cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clGetMemObjectInfo = _libraries['libOpenCL.so.1'].clGetMemObjectInfo
    clGetMemObjectInfo.restype = cl_int
    clGetMemObjectInfo.argtypes = [cl_mem, cl_mem_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetImageInfo = _libraries['libOpenCL.so.1'].clGetImageInfo
    clGetImageInfo.restype = cl_int
    clGetImageInfo.argtypes = [cl_mem, cl_image_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetPipeInfo = _libraries['libOpenCL.so.1'].clGetPipeInfo
    clGetPipeInfo.restype = cl_int
    clGetPipeInfo.argtypes = [cl_mem, cl_pipe_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clSetMemObjectDestructorCallback = _libraries['libOpenCL.so.1'].clSetMemObjectDestructorCallback
    clSetMemObjectDestructorCallback.restype = cl_int
    clSetMemObjectDestructorCallback.argtypes = [cl_mem, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_mem), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSVMAlloc = _libraries['libOpenCL.so.1'].clSVMAlloc
    clSVMAlloc.restype = ctypes.POINTER(None)
    clSVMAlloc.argtypes = [cl_context, cl_svm_mem_flags, size_t, cl_uint]
except AttributeError:
    pass
try:
    clSVMFree = _libraries['libOpenCL.so.1'].clSVMFree
    clSVMFree.restype = None
    clSVMFree.argtypes = [cl_context, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCreateSamplerWithProperties = _libraries['libOpenCL.so.1'].clCreateSamplerWithProperties
    clCreateSamplerWithProperties.restype = cl_sampler
    clCreateSamplerWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainSampler = _libraries['libOpenCL.so.1'].clRetainSampler
    clRetainSampler.restype = cl_int
    clRetainSampler.argtypes = [cl_sampler]
except AttributeError:
    pass
try:
    clReleaseSampler = _libraries['libOpenCL.so.1'].clReleaseSampler
    clReleaseSampler.restype = cl_int
    clReleaseSampler.argtypes = [cl_sampler]
except AttributeError:
    pass
try:
    clGetSamplerInfo = _libraries['libOpenCL.so.1'].clGetSamplerInfo
    clGetSamplerInfo.restype = cl_int
    clGetSamplerInfo.argtypes = [cl_sampler, cl_sampler_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateProgramWithSource = _libraries['libOpenCL.so.1'].clCreateProgramWithSource
    clCreateProgramWithSource.restype = cl_program
    clCreateProgramWithSource.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithBinary = _libraries['libOpenCL.so.1'].clCreateProgramWithBinary
    clCreateProgramWithBinary.restype = cl_program
    clCreateProgramWithBinary.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithBuiltInKernels = _libraries['libOpenCL.so.1'].clCreateProgramWithBuiltInKernels
    clCreateProgramWithBuiltInKernels.restype = cl_program
    clCreateProgramWithBuiltInKernels.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithIL = _libraries['libOpenCL.so.1'].clCreateProgramWithIL
    clCreateProgramWithIL.restype = cl_program
    clCreateProgramWithIL.argtypes = [cl_context, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainProgram = _libraries['libOpenCL.so.1'].clRetainProgram
    clRetainProgram.restype = cl_int
    clRetainProgram.argtypes = [cl_program]
except AttributeError:
    pass
try:
    clReleaseProgram = _libraries['libOpenCL.so.1'].clReleaseProgram
    clReleaseProgram.restype = cl_int
    clReleaseProgram.argtypes = [cl_program]
except AttributeError:
    pass
try:
    clBuildProgram = _libraries['libOpenCL.so.1'].clBuildProgram
    clBuildProgram.restype = cl_int
    clBuildProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCompileProgram = _libraries['libOpenCL.so.1'].clCompileProgram
    clCompileProgram.restype = cl_int
    clCompileProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_program)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clLinkProgram = _libraries['libOpenCL.so.1'].clLinkProgram
    clLinkProgram.restype = cl_program
    clLinkProgram.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_program)), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clSetProgramReleaseCallback = _libraries['libOpenCL.so.1'].clSetProgramReleaseCallback
    clSetProgramReleaseCallback.restype = cl_int
    clSetProgramReleaseCallback.argtypes = [cl_program, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetProgramSpecializationConstant = _libraries['libOpenCL.so.1'].clSetProgramSpecializationConstant
    clSetProgramSpecializationConstant.restype = cl_int
    clSetProgramSpecializationConstant.argtypes = [cl_program, cl_uint, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clUnloadPlatformCompiler = _libraries['libOpenCL.so.1'].clUnloadPlatformCompiler
    clUnloadPlatformCompiler.restype = cl_int
    clUnloadPlatformCompiler.argtypes = [cl_platform_id]
except AttributeError:
    pass
try:
    clGetProgramInfo = _libraries['libOpenCL.so.1'].clGetProgramInfo
    clGetProgramInfo.restype = cl_int
    clGetProgramInfo.argtypes = [cl_program, cl_program_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetProgramBuildInfo = _libraries['libOpenCL.so.1'].clGetProgramBuildInfo
    clGetProgramBuildInfo.restype = cl_int
    clGetProgramBuildInfo.argtypes = [cl_program, cl_device_id, cl_program_build_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateKernel = _libraries['libOpenCL.so.1'].clCreateKernel
    clCreateKernel.restype = cl_kernel
    clCreateKernel.argtypes = [cl_program, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateKernelsInProgram = _libraries['libOpenCL.so.1'].clCreateKernelsInProgram
    clCreateKernelsInProgram.restype = cl_int
    clCreateKernelsInProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_kernel)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clCloneKernel = _libraries['libOpenCL.so.1'].clCloneKernel
    clCloneKernel.restype = cl_kernel
    clCloneKernel.argtypes = [cl_kernel, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainKernel = _libraries['libOpenCL.so.1'].clRetainKernel
    clRetainKernel.restype = cl_int
    clRetainKernel.argtypes = [cl_kernel]
except AttributeError:
    pass
try:
    clReleaseKernel = _libraries['libOpenCL.so.1'].clReleaseKernel
    clReleaseKernel.restype = cl_int
    clReleaseKernel.argtypes = [cl_kernel]
except AttributeError:
    pass
try:
    clSetKernelArg = _libraries['libOpenCL.so.1'].clSetKernelArg
    clSetKernelArg.restype = cl_int
    clSetKernelArg.argtypes = [cl_kernel, cl_uint, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetKernelArgSVMPointer = _libraries['libOpenCL.so.1'].clSetKernelArgSVMPointer
    clSetKernelArgSVMPointer.restype = cl_int
    clSetKernelArgSVMPointer.argtypes = [cl_kernel, cl_uint, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetKernelExecInfo = _libraries['libOpenCL.so.1'].clSetKernelExecInfo
    clSetKernelExecInfo.restype = cl_int
    clSetKernelExecInfo.argtypes = [cl_kernel, cl_kernel_exec_info, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clGetKernelInfo = _libraries['libOpenCL.so.1'].clGetKernelInfo
    clGetKernelInfo.restype = cl_int
    clGetKernelInfo.argtypes = [cl_kernel, cl_kernel_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelArgInfo = _libraries['libOpenCL.so.1'].clGetKernelArgInfo
    clGetKernelArgInfo.restype = cl_int
    clGetKernelArgInfo.argtypes = [cl_kernel, cl_uint, cl_kernel_arg_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelWorkGroupInfo = _libraries['libOpenCL.so.1'].clGetKernelWorkGroupInfo
    clGetKernelWorkGroupInfo.restype = cl_int
    clGetKernelWorkGroupInfo.argtypes = [cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelSubGroupInfo = _libraries['libOpenCL.so.1'].clGetKernelSubGroupInfo
    clGetKernelSubGroupInfo.restype = cl_int
    clGetKernelSubGroupInfo.argtypes = [cl_kernel, cl_device_id, cl_kernel_sub_group_info, size_t, ctypes.POINTER(None), size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clWaitForEvents = _libraries['libOpenCL.so.1'].clWaitForEvents
    clWaitForEvents.restype = cl_int
    clWaitForEvents.argtypes = [cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clGetEventInfo = _libraries['libOpenCL.so.1'].clGetEventInfo
    clGetEventInfo.restype = cl_int
    clGetEventInfo.argtypes = [cl_event, cl_event_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateUserEvent = _libraries['libOpenCL.so.1'].clCreateUserEvent
    clCreateUserEvent.restype = cl_event
    clCreateUserEvent.argtypes = [cl_context, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainEvent = _libraries['libOpenCL.so.1'].clRetainEvent
    clRetainEvent.restype = cl_int
    clRetainEvent.argtypes = [cl_event]
except AttributeError:
    pass
try:
    clReleaseEvent = _libraries['libOpenCL.so.1'].clReleaseEvent
    clReleaseEvent.restype = cl_int
    clReleaseEvent.argtypes = [cl_event]
except AttributeError:
    pass
try:
    clSetUserEventStatus = _libraries['libOpenCL.so.1'].clSetUserEventStatus
    clSetUserEventStatus.restype = cl_int
    clSetUserEventStatus.argtypes = [cl_event, cl_int]
except AttributeError:
    pass
try:
    clSetEventCallback = _libraries['libOpenCL.so.1'].clSetEventCallback
    clSetEventCallback.restype = cl_int
    clSetEventCallback.argtypes = [cl_event, cl_int, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_event), ctypes.c_int32, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clGetEventProfilingInfo = _libraries['libOpenCL.so.1'].clGetEventProfilingInfo
    clGetEventProfilingInfo.restype = cl_int
    clGetEventProfilingInfo.argtypes = [cl_event, cl_profiling_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clFlush = _libraries['libOpenCL.so.1'].clFlush
    clFlush.restype = cl_int
    clFlush.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clFinish = _libraries['libOpenCL.so.1'].clFinish
    clFinish.restype = cl_int
    clFinish.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clEnqueueReadBuffer = _libraries['libOpenCL.so.1'].clEnqueueReadBuffer
    clEnqueueReadBuffer.restype = cl_int
    clEnqueueReadBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueReadBufferRect = _libraries['libOpenCL.so.1'].clEnqueueReadBufferRect
    clEnqueueReadBufferRect.restype = cl_int
    clEnqueueReadBufferRect.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteBuffer = _libraries['libOpenCL.so.1'].clEnqueueWriteBuffer
    clEnqueueWriteBuffer.restype = cl_int
    clEnqueueWriteBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteBufferRect = _libraries['libOpenCL.so.1'].clEnqueueWriteBufferRect
    clEnqueueWriteBufferRect.restype = cl_int
    clEnqueueWriteBufferRect.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueFillBuffer = _libraries['libOpenCL.so.1'].clEnqueueFillBuffer
    clEnqueueFillBuffer.restype = cl_int
    clEnqueueFillBuffer.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBuffer = _libraries['libOpenCL.so.1'].clEnqueueCopyBuffer
    clEnqueueCopyBuffer.restype = cl_int
    clEnqueueCopyBuffer.argtypes = [cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBufferRect = _libraries['libOpenCL.so.1'].clEnqueueCopyBufferRect
    clEnqueueCopyBufferRect.restype = cl_int
    clEnqueueCopyBufferRect.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueReadImage = _libraries['libOpenCL.so.1'].clEnqueueReadImage
    clEnqueueReadImage.restype = cl_int
    clEnqueueReadImage.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteImage = _libraries['libOpenCL.so.1'].clEnqueueWriteImage
    clEnqueueWriteImage.restype = cl_int
    clEnqueueWriteImage.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueFillImage = _libraries['libOpenCL.so.1'].clEnqueueFillImage
    clEnqueueFillImage.restype = cl_int
    clEnqueueFillImage.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyImage = _libraries['libOpenCL.so.1'].clEnqueueCopyImage
    clEnqueueCopyImage.restype = cl_int
    clEnqueueCopyImage.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyImageToBuffer = _libraries['libOpenCL.so.1'].clEnqueueCopyImageToBuffer
    clEnqueueCopyImageToBuffer.restype = cl_int
    clEnqueueCopyImageToBuffer.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBufferToImage = _libraries['libOpenCL.so.1'].clEnqueueCopyBufferToImage
    clEnqueueCopyBufferToImage.restype = cl_int
    clEnqueueCopyBufferToImage.argtypes = [cl_command_queue, cl_mem, cl_mem, size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMapBuffer = _libraries['libOpenCL.so.1'].clEnqueueMapBuffer
    clEnqueueMapBuffer.restype = ctypes.POINTER(None)
    clEnqueueMapBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueMapImage = _libraries['libOpenCL.so.1'].clEnqueueMapImage
    clEnqueueMapImage.restype = ctypes.POINTER(None)
    clEnqueueMapImage.argtypes = [cl_command_queue, cl_mem, cl_bool, cl_map_flags, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueUnmapMemObject = _libraries['libOpenCL.so.1'].clEnqueueUnmapMemObject
    clEnqueueUnmapMemObject.restype = cl_int
    clEnqueueUnmapMemObject.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMigrateMemObjects = _libraries['libOpenCL.so.1'].clEnqueueMigrateMemObjects
    clEnqueueMigrateMemObjects.restype = cl_int
    clEnqueueMigrateMemObjects.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_mem)), cl_mem_migration_flags, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueNDRangeKernel = _libraries['libOpenCL.so.1'].clEnqueueNDRangeKernel
    clEnqueueNDRangeKernel.restype = cl_int
    clEnqueueNDRangeKernel.argtypes = [cl_command_queue, cl_kernel, cl_uint, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueNativeKernel = _libraries['libOpenCL.so.1'].clEnqueueNativeKernel
    clEnqueueNativeKernel.restype = cl_int
    clEnqueueNativeKernel.argtypes = [cl_command_queue, ctypes.CFUNCTYPE(None, ctypes.POINTER(None)), ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_mem)), ctypes.POINTER(ctypes.POINTER(None)), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMarkerWithWaitList = _libraries['libOpenCL.so.1'].clEnqueueMarkerWithWaitList
    clEnqueueMarkerWithWaitList.restype = cl_int
    clEnqueueMarkerWithWaitList.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueBarrierWithWaitList = _libraries['libOpenCL.so.1'].clEnqueueBarrierWithWaitList
    clEnqueueBarrierWithWaitList.restype = cl_int
    clEnqueueBarrierWithWaitList.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMFree = _libraries['libOpenCL.so.1'].clEnqueueSVMFree
    clEnqueueSVMFree.restype = cl_int
    clEnqueueSVMFree.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(None) * 0, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_command_queue), ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(None)), ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMemcpy = _libraries['libOpenCL.so.1'].clEnqueueSVMMemcpy
    clEnqueueSVMMemcpy.restype = cl_int
    clEnqueueSVMMemcpy.argtypes = [cl_command_queue, cl_bool, ctypes.POINTER(None), ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMemFill = _libraries['libOpenCL.so.1'].clEnqueueSVMMemFill
    clEnqueueSVMMemFill.restype = cl_int
    clEnqueueSVMMemFill.argtypes = [cl_command_queue, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMap = _libraries['libOpenCL.so.1'].clEnqueueSVMMap
    clEnqueueSVMMap.restype = cl_int
    clEnqueueSVMMap.argtypes = [cl_command_queue, cl_bool, cl_map_flags, ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMUnmap = _libraries['libOpenCL.so.1'].clEnqueueSVMUnmap
    clEnqueueSVMUnmap.restype = cl_int
    clEnqueueSVMUnmap.argtypes = [cl_command_queue, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMigrateMem = _libraries['libOpenCL.so.1'].clEnqueueSVMMigrateMem
    clEnqueueSVMMigrateMem.restype = cl_int
    clEnqueueSVMMigrateMem.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), cl_mem_migration_flags, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clGetExtensionFunctionAddressForPlatform = _libraries['libOpenCL.so.1'].clGetExtensionFunctionAddressForPlatform
    clGetExtensionFunctionAddressForPlatform.restype = ctypes.POINTER(None)
    clGetExtensionFunctionAddressForPlatform.argtypes = [cl_platform_id, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    clCreateImage2D = _libraries['libOpenCL.so.1'].clCreateImage2D
    clCreateImage2D.restype = cl_mem
    clCreateImage2D.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), size_t, size_t, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImage3D = _libraries['libOpenCL.so.1'].clCreateImage3D
    clCreateImage3D.restype = cl_mem
    clCreateImage3D.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), size_t, size_t, size_t, size_t, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueMarker = _libraries['libOpenCL.so.1'].clEnqueueMarker
    clEnqueueMarker.restype = cl_int
    clEnqueueMarker.argtypes = [cl_command_queue, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWaitForEvents = _libraries['libOpenCL.so.1'].clEnqueueWaitForEvents
    clEnqueueWaitForEvents.restype = cl_int
    clEnqueueWaitForEvents.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueBarrier = _libraries['libOpenCL.so.1'].clEnqueueBarrier
    clEnqueueBarrier.restype = cl_int
    clEnqueueBarrier.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clUnloadCompiler = _libraries['libOpenCL.so.1'].clUnloadCompiler
    clUnloadCompiler.restype = cl_int
    clUnloadCompiler.argtypes = []
except AttributeError:
    pass
try:
    clGetExtensionFunctionAddress = _libraries['libOpenCL.so.1'].clGetExtensionFunctionAddress
    clGetExtensionFunctionAddress.restype = ctypes.POINTER(None)
    clGetExtensionFunctionAddress.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    clCreateCommandQueue = _libraries['libOpenCL.so.1'].clCreateCommandQueue
    clCreateCommandQueue.restype = cl_command_queue
    clCreateCommandQueue.argtypes = [cl_context, cl_device_id, cl_command_queue_properties, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateSampler = _libraries['libOpenCL.so.1'].clCreateSampler
    clCreateSampler.restype = cl_sampler
    clCreateSampler.argtypes = [cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueTask = _libraries['libOpenCL.so.1'].clEnqueueTask
    clEnqueueTask.restype = cl_int
    clEnqueueTask.argtypes = [cl_command_queue, cl_kernel, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
__all__ = \
    ['CL_A', 'CL_ABGR', 'CL_ADDRESS_CLAMP',
    'CL_ADDRESS_CLAMP_TO_EDGE', 'CL_ADDRESS_MIRRORED_REPEAT',
    'CL_ADDRESS_NONE', 'CL_ADDRESS_REPEAT', 'CL_ARGB', 'CL_BGRA',
    'CL_BLOCKING', 'CL_BUFFER_CREATE_TYPE_REGION', 'CL_BUILD_ERROR',
    'CL_BUILD_IN_PROGRESS', 'CL_BUILD_NONE',
    'CL_BUILD_PROGRAM_FAILURE', 'CL_BUILD_SUCCESS',
    'CL_COMMAND_ACQUIRE_GL_OBJECTS', 'CL_COMMAND_BARRIER',
    'CL_COMMAND_COPY_BUFFER', 'CL_COMMAND_COPY_BUFFER_RECT',
    'CL_COMMAND_COPY_BUFFER_TO_IMAGE', 'CL_COMMAND_COPY_IMAGE',
    'CL_COMMAND_COPY_IMAGE_TO_BUFFER', 'CL_COMMAND_FILL_BUFFER',
    'CL_COMMAND_FILL_IMAGE', 'CL_COMMAND_MAP_BUFFER',
    'CL_COMMAND_MAP_IMAGE', 'CL_COMMAND_MARKER',
    'CL_COMMAND_MIGRATE_MEM_OBJECTS', 'CL_COMMAND_NATIVE_KERNEL',
    'CL_COMMAND_NDRANGE_KERNEL', 'CL_COMMAND_READ_BUFFER',
    'CL_COMMAND_READ_BUFFER_RECT', 'CL_COMMAND_READ_IMAGE',
    'CL_COMMAND_RELEASE_GL_OBJECTS', 'CL_COMMAND_SVM_FREE',
    'CL_COMMAND_SVM_MAP', 'CL_COMMAND_SVM_MEMCPY',
    'CL_COMMAND_SVM_MEMFILL', 'CL_COMMAND_SVM_MIGRATE_MEM',
    'CL_COMMAND_SVM_UNMAP', 'CL_COMMAND_TASK',
    'CL_COMMAND_UNMAP_MEM_OBJECT', 'CL_COMMAND_USER',
    'CL_COMMAND_WRITE_BUFFER', 'CL_COMMAND_WRITE_BUFFER_RECT',
    'CL_COMMAND_WRITE_IMAGE', 'CL_COMPILER_NOT_AVAILABLE',
    'CL_COMPILE_PROGRAM_FAILURE', 'CL_COMPLETE', 'CL_CONTEXT_DEVICES',
    'CL_CONTEXT_INTEROP_USER_SYNC', 'CL_CONTEXT_NUM_DEVICES',
    'CL_CONTEXT_PLATFORM', 'CL_CONTEXT_PROPERTIES',
    'CL_CONTEXT_REFERENCE_COUNT', 'CL_DEPTH', 'CL_DEPTH_STENCIL',
    'CL_DEVICE_ADDRESS_BITS', 'CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE',
    'CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE',
    'CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE',
    'CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE',
    'CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE',
    'CL_DEVICE_AFFINITY_DOMAIN_NUMA',
    'CL_DEVICE_ATOMIC_FENCE_CAPABILITIES',
    'CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES',
    'CL_DEVICE_ATOMIC_ORDER_ACQ_REL',
    'CL_DEVICE_ATOMIC_ORDER_RELAXED',
    'CL_DEVICE_ATOMIC_ORDER_SEQ_CST',
    'CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES',
    'CL_DEVICE_ATOMIC_SCOPE_DEVICE',
    'CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP',
    'CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM', 'CL_DEVICE_AVAILABLE',
    'CL_DEVICE_BUILT_IN_KERNELS',
    'CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION',
    'CL_DEVICE_COMPILER_AVAILABLE',
    'CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES',
    'CL_DEVICE_DOUBLE_FP_CONFIG', 'CL_DEVICE_ENDIAN_LITTLE',
    'CL_DEVICE_ERROR_CORRECTION_SUPPORT',
    'CL_DEVICE_EXECUTION_CAPABILITIES', 'CL_DEVICE_EXTENSIONS',
    'CL_DEVICE_EXTENSIONS_WITH_VERSION',
    'CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT',
    'CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE',
    'CL_DEVICE_GLOBAL_MEM_CACHE_SIZE',
    'CL_DEVICE_GLOBAL_MEM_CACHE_TYPE', 'CL_DEVICE_GLOBAL_MEM_SIZE',
    'CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE',
    'CL_DEVICE_HOST_UNIFIED_MEMORY', 'CL_DEVICE_ILS_WITH_VERSION',
    'CL_DEVICE_IL_VERSION', 'CL_DEVICE_IMAGE2D_MAX_HEIGHT',
    'CL_DEVICE_IMAGE2D_MAX_WIDTH', 'CL_DEVICE_IMAGE3D_MAX_DEPTH',
    'CL_DEVICE_IMAGE3D_MAX_HEIGHT', 'CL_DEVICE_IMAGE3D_MAX_WIDTH',
    'CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT',
    'CL_DEVICE_IMAGE_MAX_ARRAY_SIZE',
    'CL_DEVICE_IMAGE_MAX_BUFFER_SIZE',
    'CL_DEVICE_IMAGE_PITCH_ALIGNMENT', 'CL_DEVICE_IMAGE_SUPPORT',
    'CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED',
    'CL_DEVICE_LINKER_AVAILABLE', 'CL_DEVICE_LOCAL_MEM_SIZE',
    'CL_DEVICE_LOCAL_MEM_TYPE', 'CL_DEVICE_MAX_CLOCK_FREQUENCY',
    'CL_DEVICE_MAX_COMPUTE_UNITS', 'CL_DEVICE_MAX_CONSTANT_ARGS',
    'CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE',
    'CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE',
    'CL_DEVICE_MAX_MEM_ALLOC_SIZE', 'CL_DEVICE_MAX_NUM_SUB_GROUPS',
    'CL_DEVICE_MAX_ON_DEVICE_EVENTS',
    'CL_DEVICE_MAX_ON_DEVICE_QUEUES', 'CL_DEVICE_MAX_PARAMETER_SIZE',
    'CL_DEVICE_MAX_PIPE_ARGS', 'CL_DEVICE_MAX_READ_IMAGE_ARGS',
    'CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS', 'CL_DEVICE_MAX_SAMPLERS',
    'CL_DEVICE_MAX_WORK_GROUP_SIZE',
    'CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS',
    'CL_DEVICE_MAX_WORK_ITEM_SIZES', 'CL_DEVICE_MAX_WRITE_IMAGE_ARGS',
    'CL_DEVICE_MEM_BASE_ADDR_ALIGN',
    'CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE', 'CL_DEVICE_NAME',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_INT',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT',
    'CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT',
    'CL_DEVICE_NOT_AVAILABLE', 'CL_DEVICE_NOT_FOUND',
    'CL_DEVICE_NUMERIC_VERSION', 'CL_DEVICE_OPENCL_C_ALL_VERSIONS',
    'CL_DEVICE_OPENCL_C_FEATURES', 'CL_DEVICE_OPENCL_C_VERSION',
    'CL_DEVICE_PARENT_DEVICE', 'CL_DEVICE_PARTITION_AFFINITY_DOMAIN',
    'CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN',
    'CL_DEVICE_PARTITION_BY_COUNTS',
    'CL_DEVICE_PARTITION_BY_COUNTS_LIST_END',
    'CL_DEVICE_PARTITION_EQUALLY', 'CL_DEVICE_PARTITION_FAILED',
    'CL_DEVICE_PARTITION_MAX_SUB_DEVICES',
    'CL_DEVICE_PARTITION_PROPERTIES', 'CL_DEVICE_PARTITION_TYPE',
    'CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS',
    'CL_DEVICE_PIPE_MAX_PACKET_SIZE', 'CL_DEVICE_PIPE_SUPPORT',
    'CL_DEVICE_PLATFORM',
    'CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT',
    'CL_DEVICE_PREFERRED_INTEROP_USER_SYNC',
    'CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT',
    'CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT',
    'CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE',
    'CL_DEVICE_PRINTF_BUFFER_SIZE', 'CL_DEVICE_PROFILE',
    'CL_DEVICE_PROFILING_TIMER_RESOLUTION',
    'CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE',
    'CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE',
    'CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES',
    'CL_DEVICE_QUEUE_ON_HOST_PROPERTIES',
    'CL_DEVICE_QUEUE_PROPERTIES',
    'CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT',
    'CL_DEVICE_QUEUE_SUPPORTED', 'CL_DEVICE_REFERENCE_COUNT',
    'CL_DEVICE_SINGLE_FP_CONFIG',
    'CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS',
    'CL_DEVICE_SVM_ATOMICS', 'CL_DEVICE_SVM_CAPABILITIES',
    'CL_DEVICE_SVM_COARSE_GRAIN_BUFFER',
    'CL_DEVICE_SVM_FINE_GRAIN_BUFFER',
    'CL_DEVICE_SVM_FINE_GRAIN_SYSTEM', 'CL_DEVICE_TYPE',
    'CL_DEVICE_TYPE_ACCELERATOR', 'CL_DEVICE_TYPE_ALL',
    'CL_DEVICE_TYPE_CPU', 'CL_DEVICE_TYPE_CUSTOM',
    'CL_DEVICE_TYPE_DEFAULT', 'CL_DEVICE_TYPE_GPU',
    'CL_DEVICE_VENDOR', 'CL_DEVICE_VENDOR_ID', 'CL_DEVICE_VERSION',
    'CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT',
    'CL_DRIVER_VERSION', 'CL_EVENT_COMMAND_EXECUTION_STATUS',
    'CL_EVENT_COMMAND_QUEUE', 'CL_EVENT_COMMAND_TYPE',
    'CL_EVENT_CONTEXT', 'CL_EVENT_REFERENCE_COUNT', 'CL_EXEC_KERNEL',
    'CL_EXEC_NATIVE_KERNEL',
    'CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST', 'CL_FALSE',
    'CL_FILTER_LINEAR', 'CL_FILTER_NEAREST', 'CL_FLOAT',
    'CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT', 'CL_FP_DENORM',
    'CL_FP_FMA', 'CL_FP_INF_NAN', 'CL_FP_ROUND_TO_INF',
    'CL_FP_ROUND_TO_NEAREST', 'CL_FP_ROUND_TO_ZERO',
    'CL_FP_SOFT_FLOAT', 'CL_GLOBAL', 'CL_HALF_FLOAT',
    'CL_IMAGE_ARRAY_SIZE', 'CL_IMAGE_BUFFER', 'CL_IMAGE_DEPTH',
    'CL_IMAGE_ELEMENT_SIZE', 'CL_IMAGE_FORMAT',
    'CL_IMAGE_FORMAT_MISMATCH', 'CL_IMAGE_FORMAT_NOT_SUPPORTED',
    'CL_IMAGE_HEIGHT', 'CL_IMAGE_NUM_MIP_LEVELS',
    'CL_IMAGE_NUM_SAMPLES', 'CL_IMAGE_ROW_PITCH',
    'CL_IMAGE_SLICE_PITCH', 'CL_IMAGE_WIDTH', 'CL_INTENSITY',
    'CL_INVALID_ARG_INDEX', 'CL_INVALID_ARG_SIZE',
    'CL_INVALID_ARG_VALUE', 'CL_INVALID_BINARY',
    'CL_INVALID_BUFFER_SIZE', 'CL_INVALID_BUILD_OPTIONS',
    'CL_INVALID_COMMAND_QUEUE', 'CL_INVALID_COMPILER_OPTIONS',
    'CL_INVALID_CONTEXT', 'CL_INVALID_DEVICE',
    'CL_INVALID_DEVICE_PARTITION_COUNT', 'CL_INVALID_DEVICE_QUEUE',
    'CL_INVALID_DEVICE_TYPE', 'CL_INVALID_EVENT',
    'CL_INVALID_EVENT_WAIT_LIST', 'CL_INVALID_GLOBAL_OFFSET',
    'CL_INVALID_GLOBAL_WORK_SIZE', 'CL_INVALID_GL_OBJECT',
    'CL_INVALID_HOST_PTR', 'CL_INVALID_IMAGE_DESCRIPTOR',
    'CL_INVALID_IMAGE_FORMAT_DESCRIPTOR', 'CL_INVALID_IMAGE_SIZE',
    'CL_INVALID_KERNEL', 'CL_INVALID_KERNEL_ARGS',
    'CL_INVALID_KERNEL_DEFINITION', 'CL_INVALID_KERNEL_NAME',
    'CL_INVALID_LINKER_OPTIONS', 'CL_INVALID_MEM_OBJECT',
    'CL_INVALID_MIP_LEVEL', 'CL_INVALID_OPERATION',
    'CL_INVALID_PIPE_SIZE', 'CL_INVALID_PLATFORM',
    'CL_INVALID_PROGRAM', 'CL_INVALID_PROGRAM_EXECUTABLE',
    'CL_INVALID_PROPERTY', 'CL_INVALID_QUEUE_PROPERTIES',
    'CL_INVALID_SAMPLER', 'CL_INVALID_SPEC_ID', 'CL_INVALID_VALUE',
    'CL_INVALID_WORK_DIMENSION', 'CL_INVALID_WORK_GROUP_SIZE',
    'CL_INVALID_WORK_ITEM_SIZE', 'CL_KERNEL_ARG_ACCESS_NONE',
    'CL_KERNEL_ARG_ACCESS_QUALIFIER',
    'CL_KERNEL_ARG_ACCESS_READ_ONLY',
    'CL_KERNEL_ARG_ACCESS_READ_WRITE',
    'CL_KERNEL_ARG_ACCESS_WRITE_ONLY',
    'CL_KERNEL_ARG_ADDRESS_CONSTANT', 'CL_KERNEL_ARG_ADDRESS_GLOBAL',
    'CL_KERNEL_ARG_ADDRESS_LOCAL', 'CL_KERNEL_ARG_ADDRESS_PRIVATE',
    'CL_KERNEL_ARG_ADDRESS_QUALIFIER',
    'CL_KERNEL_ARG_INFO_NOT_AVAILABLE', 'CL_KERNEL_ARG_NAME',
    'CL_KERNEL_ARG_TYPE_CONST', 'CL_KERNEL_ARG_TYPE_NAME',
    'CL_KERNEL_ARG_TYPE_NONE', 'CL_KERNEL_ARG_TYPE_PIPE',
    'CL_KERNEL_ARG_TYPE_QUALIFIER', 'CL_KERNEL_ARG_TYPE_RESTRICT',
    'CL_KERNEL_ARG_TYPE_VOLATILE', 'CL_KERNEL_ATTRIBUTES',
    'CL_KERNEL_COMPILE_NUM_SUB_GROUPS',
    'CL_KERNEL_COMPILE_WORK_GROUP_SIZE', 'CL_KERNEL_CONTEXT',
    'CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM',
    'CL_KERNEL_EXEC_INFO_SVM_PTRS', 'CL_KERNEL_FUNCTION_NAME',
    'CL_KERNEL_GLOBAL_WORK_SIZE', 'CL_KERNEL_LOCAL_MEM_SIZE',
    'CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT',
    'CL_KERNEL_MAX_NUM_SUB_GROUPS',
    'CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE', 'CL_KERNEL_NUM_ARGS',
    'CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE',
    'CL_KERNEL_PRIVATE_MEM_SIZE', 'CL_KERNEL_PROGRAM',
    'CL_KERNEL_REFERENCE_COUNT',
    'CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE',
    'CL_KERNEL_WORK_GROUP_SIZE', 'CL_KHRONOS_VENDOR_ID_CODEPLAY',
    'CL_LINKER_NOT_AVAILABLE', 'CL_LINK_PROGRAM_FAILURE', 'CL_LOCAL',
    'CL_LUMINANCE', 'CL_MAP_FAILURE', 'CL_MAP_READ', 'CL_MAP_WRITE',
    'CL_MAP_WRITE_INVALIDATE_REGION',
    'CL_MAX_SIZE_RESTRICTION_EXCEEDED', 'CL_MEM_ALLOC_HOST_PTR',
    'CL_MEM_ASSOCIATED_MEMOBJECT', 'CL_MEM_CONTEXT',
    'CL_MEM_COPY_HOST_PTR', 'CL_MEM_COPY_OVERLAP', 'CL_MEM_FLAGS',
    'CL_MEM_HOST_NO_ACCESS', 'CL_MEM_HOST_PTR',
    'CL_MEM_HOST_READ_ONLY', 'CL_MEM_HOST_WRITE_ONLY',
    'CL_MEM_KERNEL_READ_AND_WRITE', 'CL_MEM_MAP_COUNT',
    'CL_MEM_OBJECT_ALLOCATION_FAILURE', 'CL_MEM_OBJECT_BUFFER',
    'CL_MEM_OBJECT_IMAGE1D', 'CL_MEM_OBJECT_IMAGE1D_ARRAY',
    'CL_MEM_OBJECT_IMAGE1D_BUFFER', 'CL_MEM_OBJECT_IMAGE2D',
    'CL_MEM_OBJECT_IMAGE2D_ARRAY', 'CL_MEM_OBJECT_IMAGE3D',
    'CL_MEM_OBJECT_PIPE', 'CL_MEM_OFFSET', 'CL_MEM_PROPERTIES',
    'CL_MEM_READ_ONLY', 'CL_MEM_READ_WRITE', 'CL_MEM_REFERENCE_COUNT',
    'CL_MEM_SIZE', 'CL_MEM_SVM_ATOMICS',
    'CL_MEM_SVM_FINE_GRAIN_BUFFER', 'CL_MEM_TYPE',
    'CL_MEM_USES_SVM_POINTER', 'CL_MEM_USE_HOST_PTR',
    'CL_MEM_WRITE_ONLY', 'CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED',
    'CL_MIGRATE_MEM_OBJECT_HOST', 'CL_MISALIGNED_SUB_BUFFER_OFFSET',
    'CL_NAME_VERSION_MAX_NAME_SIZE', 'CL_NONE', 'CL_NON_BLOCKING',
    'CL_OUT_OF_HOST_MEMORY', 'CL_OUT_OF_RESOURCES',
    'CL_PIPE_MAX_PACKETS', 'CL_PIPE_PACKET_SIZE',
    'CL_PIPE_PROPERTIES', 'CL_PLATFORM_EXTENSIONS',
    'CL_PLATFORM_EXTENSIONS_WITH_VERSION',
    'CL_PLATFORM_HOST_TIMER_RESOLUTION', 'CL_PLATFORM_NAME',
    'CL_PLATFORM_NUMERIC_VERSION', 'CL_PLATFORM_PROFILE',
    'CL_PLATFORM_VENDOR', 'CL_PLATFORM_VERSION',
    'CL_PROFILING_COMMAND_COMPLETE', 'CL_PROFILING_COMMAND_END',
    'CL_PROFILING_COMMAND_QUEUED', 'CL_PROFILING_COMMAND_START',
    'CL_PROFILING_COMMAND_SUBMIT', 'CL_PROFILING_INFO_NOT_AVAILABLE',
    'CL_PROGRAM_BINARIES', 'CL_PROGRAM_BINARY_SIZES',
    'CL_PROGRAM_BINARY_TYPE',
    'CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT',
    'CL_PROGRAM_BINARY_TYPE_EXECUTABLE',
    'CL_PROGRAM_BINARY_TYPE_LIBRARY', 'CL_PROGRAM_BINARY_TYPE_NONE',
    'CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE',
    'CL_PROGRAM_BUILD_LOG', 'CL_PROGRAM_BUILD_OPTIONS',
    'CL_PROGRAM_BUILD_STATUS', 'CL_PROGRAM_CONTEXT',
    'CL_PROGRAM_DEVICES', 'CL_PROGRAM_IL', 'CL_PROGRAM_KERNEL_NAMES',
    'CL_PROGRAM_NUM_DEVICES', 'CL_PROGRAM_NUM_KERNELS',
    'CL_PROGRAM_REFERENCE_COUNT',
    'CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT',
    'CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT', 'CL_PROGRAM_SOURCE',
    'CL_QUEUED', 'CL_QUEUE_CONTEXT', 'CL_QUEUE_DEVICE',
    'CL_QUEUE_DEVICE_DEFAULT', 'CL_QUEUE_ON_DEVICE',
    'CL_QUEUE_ON_DEVICE_DEFAULT',
    'CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE',
    'CL_QUEUE_PROFILING_ENABLE', 'CL_QUEUE_PROPERTIES',
    'CL_QUEUE_PROPERTIES_ARRAY', 'CL_QUEUE_REFERENCE_COUNT',
    'CL_QUEUE_SIZE', 'CL_R', 'CL_RA', 'CL_READ_ONLY_CACHE',
    'CL_READ_WRITE_CACHE', 'CL_RG', 'CL_RGB', 'CL_RGBA', 'CL_RGBx',
    'CL_RGx', 'CL_RUNNING', 'CL_Rx', 'CL_SAMPLER_ADDRESSING_MODE',
    'CL_SAMPLER_CONTEXT', 'CL_SAMPLER_FILTER_MODE',
    'CL_SAMPLER_LOD_MAX', 'CL_SAMPLER_LOD_MIN',
    'CL_SAMPLER_MIP_FILTER_MODE', 'CL_SAMPLER_NORMALIZED_COORDS',
    'CL_SAMPLER_PROPERTIES', 'CL_SAMPLER_REFERENCE_COUNT',
    'CL_SIGNED_INT16', 'CL_SIGNED_INT32', 'CL_SIGNED_INT8',
    'CL_SNORM_INT16', 'CL_SNORM_INT8', 'CL_SUBMITTED', 'CL_SUCCESS',
    'CL_TRUE', 'CL_UNORM_INT16', 'CL_UNORM_INT24', 'CL_UNORM_INT8',
    'CL_UNORM_INT_101010', 'CL_UNORM_INT_101010_2',
    'CL_UNORM_SHORT_555', 'CL_UNORM_SHORT_565', 'CL_UNSIGNED_INT16',
    'CL_UNSIGNED_INT32', 'CL_UNSIGNED_INT8', 'CL_VERSION_MAJOR_BITS',
    'CL_VERSION_MAJOR_MASK', 'CL_VERSION_MINOR_BITS',
    'CL_VERSION_MINOR_MASK', 'CL_VERSION_PATCH_BITS',
    'CL_VERSION_PATCH_MASK', 'CL_sBGRA', 'CL_sRGB', 'CL_sRGBA',
    'CL_sRGBx', '__OPENCL_CL_H', 'clBuildProgram', 'clCloneKernel',
    'clCompileProgram', 'clCreateBuffer',
    'clCreateBufferWithProperties', 'clCreateCommandQueue',
    'clCreateCommandQueueWithProperties', 'clCreateContext',
    'clCreateContextFromType', 'clCreateImage', 'clCreateImage2D',
    'clCreateImage3D', 'clCreateImageWithProperties',
    'clCreateKernel', 'clCreateKernelsInProgram', 'clCreatePipe',
    'clCreateProgramWithBinary', 'clCreateProgramWithBuiltInKernels',
    'clCreateProgramWithIL', 'clCreateProgramWithSource',
    'clCreateSampler', 'clCreateSamplerWithProperties',
    'clCreateSubBuffer', 'clCreateSubDevices', 'clCreateUserEvent',
    'clEnqueueBarrier', 'clEnqueueBarrierWithWaitList',
    'clEnqueueCopyBuffer', 'clEnqueueCopyBufferRect',
    'clEnqueueCopyBufferToImage', 'clEnqueueCopyImage',
    'clEnqueueCopyImageToBuffer', 'clEnqueueFillBuffer',
    'clEnqueueFillImage', 'clEnqueueMapBuffer', 'clEnqueueMapImage',
    'clEnqueueMarker', 'clEnqueueMarkerWithWaitList',
    'clEnqueueMigrateMemObjects', 'clEnqueueNDRangeKernel',
    'clEnqueueNativeKernel', 'clEnqueueReadBuffer',
    'clEnqueueReadBufferRect', 'clEnqueueReadImage',
    'clEnqueueSVMFree', 'clEnqueueSVMMap', 'clEnqueueSVMMemFill',
    'clEnqueueSVMMemcpy', 'clEnqueueSVMMigrateMem',
    'clEnqueueSVMUnmap', 'clEnqueueTask', 'clEnqueueUnmapMemObject',
    'clEnqueueWaitForEvents', 'clEnqueueWriteBuffer',
    'clEnqueueWriteBufferRect', 'clEnqueueWriteImage', 'clFinish',
    'clFlush', 'clGetCommandQueueInfo', 'clGetContextInfo',
    'clGetDeviceAndHostTimer', 'clGetDeviceIDs', 'clGetDeviceInfo',
    'clGetEventInfo', 'clGetEventProfilingInfo',
    'clGetExtensionFunctionAddress',
    'clGetExtensionFunctionAddressForPlatform', 'clGetHostTimer',
    'clGetImageInfo', 'clGetKernelArgInfo', 'clGetKernelInfo',
    'clGetKernelSubGroupInfo', 'clGetKernelWorkGroupInfo',
    'clGetMemObjectInfo', 'clGetPipeInfo', 'clGetPlatformIDs',
    'clGetPlatformInfo', 'clGetProgramBuildInfo', 'clGetProgramInfo',
    'clGetSamplerInfo', 'clGetSupportedImageFormats', 'clLinkProgram',
    'clReleaseCommandQueue', 'clReleaseContext', 'clReleaseDevice',
    'clReleaseEvent', 'clReleaseKernel', 'clReleaseMemObject',
    'clReleaseProgram', 'clReleaseSampler', 'clRetainCommandQueue',
    'clRetainContext', 'clRetainDevice', 'clRetainEvent',
    'clRetainKernel', 'clRetainMemObject', 'clRetainProgram',
    'clRetainSampler', 'clSVMAlloc', 'clSVMFree',
    'clSetContextDestructorCallback',
    'clSetDefaultDeviceCommandQueue', 'clSetEventCallback',
    'clSetKernelArg', 'clSetKernelArgSVMPointer',
    'clSetKernelExecInfo', 'clSetMemObjectDestructorCallback',
    'clSetProgramReleaseCallback',
    'clSetProgramSpecializationConstant', 'clSetUserEventStatus',
    'clUnloadCompiler', 'clUnloadPlatformCompiler', 'clWaitForEvents',
    'cl_addressing_mode', 'cl_bitfield', 'cl_bool',
    'cl_buffer_create_type', 'cl_buffer_region', 'cl_build_status',
    'cl_channel_order', 'cl_channel_type', 'cl_command_queue',
    'cl_command_queue_info', 'cl_command_queue_properties',
    'cl_command_type', 'cl_context', 'cl_context_info',
    'cl_context_properties', 'cl_device_affinity_domain',
    'cl_device_atomic_capabilities',
    'cl_device_device_enqueue_capabilities',
    'cl_device_exec_capabilities', 'cl_device_fp_config',
    'cl_device_id', 'cl_device_info', 'cl_device_local_mem_type',
    'cl_device_mem_cache_type', 'cl_device_partition_property',
    'cl_device_svm_capabilities', 'cl_device_type', 'cl_event',
    'cl_event_info', 'cl_filter_mode', 'cl_image_desc',
    'cl_image_format', 'cl_image_info', 'cl_int', 'cl_kernel',
    'cl_kernel_arg_access_qualifier',
    'cl_kernel_arg_address_qualifier', 'cl_kernel_arg_info',
    'cl_kernel_arg_type_qualifier', 'cl_kernel_exec_info',
    'cl_kernel_info', 'cl_kernel_sub_group_info',
    'cl_kernel_work_group_info', 'cl_khronos_vendor_id',
    'cl_map_flags', 'cl_mem', 'cl_mem_flags', 'cl_mem_info',
    'cl_mem_migration_flags', 'cl_mem_object_type',
    'cl_mem_properties', 'cl_name_version', 'cl_pipe_info',
    'cl_pipe_properties', 'cl_platform_id', 'cl_platform_info',
    'cl_profiling_info', 'cl_program', 'cl_program_binary_type',
    'cl_program_build_info', 'cl_program_info', 'cl_properties',
    'cl_queue_properties', 'cl_sampler', 'cl_sampler_info',
    'cl_sampler_properties', 'cl_svm_mem_flags', 'cl_uint',
    'cl_version', 'size_t', 'struct__cl_buffer_region',
    'struct__cl_command_queue', 'struct__cl_context',
    'struct__cl_device_id', 'struct__cl_event',
    'struct__cl_image_desc', 'struct__cl_image_format',
    'struct__cl_kernel', 'struct__cl_mem', 'struct__cl_name_version',
    'struct__cl_platform_id', 'struct__cl_program',
    'struct__cl_sampler', 'union__cl_image_desc_0']
