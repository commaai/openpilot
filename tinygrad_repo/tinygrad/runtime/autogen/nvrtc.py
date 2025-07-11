# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util


_libraries = {}
_libraries['libnvrtc.so'] = ctypes.CDLL(ctypes.util.find_library('nvrtc'))
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



_libraries['libnvJitLink.so'] = ctypes.CDLL(ctypes.util.find_library('nvJitLink'))
c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16




# values for enumeration 'c__EA_nvrtcResult'
c__EA_nvrtcResult__enumvalues = {
    0: 'NVRTC_SUCCESS',
    1: 'NVRTC_ERROR_OUT_OF_MEMORY',
    2: 'NVRTC_ERROR_PROGRAM_CREATION_FAILURE',
    3: 'NVRTC_ERROR_INVALID_INPUT',
    4: 'NVRTC_ERROR_INVALID_PROGRAM',
    5: 'NVRTC_ERROR_INVALID_OPTION',
    6: 'NVRTC_ERROR_COMPILATION',
    7: 'NVRTC_ERROR_BUILTIN_OPERATION_FAILURE',
    8: 'NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION',
    9: 'NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION',
    10: 'NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID',
    11: 'NVRTC_ERROR_INTERNAL_ERROR',
    12: 'NVRTC_ERROR_TIME_FILE_WRITE_FAILED',
}
NVRTC_SUCCESS = 0
NVRTC_ERROR_OUT_OF_MEMORY = 1
NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
NVRTC_ERROR_INVALID_INPUT = 3
NVRTC_ERROR_INVALID_PROGRAM = 4
NVRTC_ERROR_INVALID_OPTION = 5
NVRTC_ERROR_COMPILATION = 6
NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
NVRTC_ERROR_INTERNAL_ERROR = 11
NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12
c__EA_nvrtcResult = ctypes.c_uint32 # enum
nvrtcResult = c__EA_nvrtcResult
nvrtcResult__enumvalues = c__EA_nvrtcResult__enumvalues
try:
    nvrtcGetErrorString = _libraries['libnvrtc.so'].nvrtcGetErrorString
    nvrtcGetErrorString.restype = ctypes.POINTER(ctypes.c_char)
    nvrtcGetErrorString.argtypes = [nvrtcResult]
except AttributeError:
    pass
try:
    nvrtcVersion = _libraries['libnvrtc.so'].nvrtcVersion
    nvrtcVersion.restype = nvrtcResult
    nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    nvrtcGetNumSupportedArchs = _libraries['libnvrtc.so'].nvrtcGetNumSupportedArchs
    nvrtcGetNumSupportedArchs.restype = nvrtcResult
    nvrtcGetNumSupportedArchs.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    nvrtcGetSupportedArchs = _libraries['libnvrtc.so'].nvrtcGetSupportedArchs
    nvrtcGetSupportedArchs.restype = nvrtcResult
    nvrtcGetSupportedArchs.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
class struct__nvrtcProgram(Structure):
    pass

nvrtcProgram = ctypes.POINTER(struct__nvrtcProgram)
try:
    nvrtcCreateProgram = _libraries['libnvrtc.so'].nvrtcCreateProgram
    nvrtcCreateProgram.restype = nvrtcResult
    nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__nvrtcProgram)), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    nvrtcDestroyProgram = _libraries['libnvrtc.so'].nvrtcDestroyProgram
    nvrtcDestroyProgram.restype = nvrtcResult
    nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__nvrtcProgram))]
except AttributeError:
    pass
try:
    nvrtcCompileProgram = _libraries['libnvrtc.so'].nvrtcCompileProgram
    nvrtcCompileProgram.restype = nvrtcResult
    nvrtcCompileProgram.argtypes = [nvrtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    nvrtcGetPTXSize = _libraries['libnvrtc.so'].nvrtcGetPTXSize
    nvrtcGetPTXSize.restype = nvrtcResult
    nvrtcGetPTXSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetPTX = _libraries['libnvrtc.so'].nvrtcGetPTX
    nvrtcGetPTX.restype = nvrtcResult
    nvrtcGetPTX.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetCUBINSize = _libraries['libnvrtc.so'].nvrtcGetCUBINSize
    nvrtcGetCUBINSize.restype = nvrtcResult
    nvrtcGetCUBINSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetCUBIN = _libraries['libnvrtc.so'].nvrtcGetCUBIN
    nvrtcGetCUBIN.restype = nvrtcResult
    nvrtcGetCUBIN.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetNVVMSize = _libraries['libnvrtc.so'].nvrtcGetNVVMSize
    nvrtcGetNVVMSize.restype = nvrtcResult
    nvrtcGetNVVMSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetNVVM = _libraries['libnvrtc.so'].nvrtcGetNVVM
    nvrtcGetNVVM.restype = nvrtcResult
    nvrtcGetNVVM.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetLTOIRSize = _libraries['libnvrtc.so'].nvrtcGetLTOIRSize
    nvrtcGetLTOIRSize.restype = nvrtcResult
    nvrtcGetLTOIRSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetLTOIR = _libraries['libnvrtc.so'].nvrtcGetLTOIR
    nvrtcGetLTOIR.restype = nvrtcResult
    nvrtcGetLTOIR.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetOptiXIRSize = _libraries['libnvrtc.so'].nvrtcGetOptiXIRSize
    nvrtcGetOptiXIRSize.restype = nvrtcResult
    nvrtcGetOptiXIRSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetOptiXIR = _libraries['libnvrtc.so'].nvrtcGetOptiXIR
    nvrtcGetOptiXIR.restype = nvrtcResult
    nvrtcGetOptiXIR.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetProgramLogSize = _libraries['libnvrtc.so'].nvrtcGetProgramLogSize
    nvrtcGetProgramLogSize.restype = nvrtcResult
    nvrtcGetProgramLogSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetProgramLog = _libraries['libnvrtc.so'].nvrtcGetProgramLog
    nvrtcGetProgramLog.restype = nvrtcResult
    nvrtcGetProgramLog.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcAddNameExpression = _libraries['libnvrtc.so'].nvrtcAddNameExpression
    nvrtcAddNameExpression.restype = nvrtcResult
    nvrtcAddNameExpression.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetLoweredName = _libraries['libnvrtc.so'].nvrtcGetLoweredName
    nvrtcGetLoweredName.restype = nvrtcResult
    nvrtcGetLoweredName.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass

# values for enumeration 'c__EA_nvJitLinkResult'
c__EA_nvJitLinkResult__enumvalues = {
    0: 'NVJITLINK_SUCCESS',
    1: 'NVJITLINK_ERROR_UNRECOGNIZED_OPTION',
    2: 'NVJITLINK_ERROR_MISSING_ARCH',
    3: 'NVJITLINK_ERROR_INVALID_INPUT',
    4: 'NVJITLINK_ERROR_PTX_COMPILE',
    5: 'NVJITLINK_ERROR_NVVM_COMPILE',
    6: 'NVJITLINK_ERROR_INTERNAL',
    7: 'NVJITLINK_ERROR_THREADPOOL',
    8: 'NVJITLINK_ERROR_UNRECOGNIZED_INPUT',
}
NVJITLINK_SUCCESS = 0
NVJITLINK_ERROR_UNRECOGNIZED_OPTION = 1
NVJITLINK_ERROR_MISSING_ARCH = 2
NVJITLINK_ERROR_INVALID_INPUT = 3
NVJITLINK_ERROR_PTX_COMPILE = 4
NVJITLINK_ERROR_NVVM_COMPILE = 5
NVJITLINK_ERROR_INTERNAL = 6
NVJITLINK_ERROR_THREADPOOL = 7
NVJITLINK_ERROR_UNRECOGNIZED_INPUT = 8
c__EA_nvJitLinkResult = ctypes.c_uint32 # enum
nvJitLinkResult = c__EA_nvJitLinkResult
nvJitLinkResult__enumvalues = c__EA_nvJitLinkResult__enumvalues

# values for enumeration 'c__EA_nvJitLinkInputType'
c__EA_nvJitLinkInputType__enumvalues = {
    0: 'NVJITLINK_INPUT_NONE',
    1: 'NVJITLINK_INPUT_CUBIN',
    2: 'NVJITLINK_INPUT_PTX',
    3: 'NVJITLINK_INPUT_LTOIR',
    4: 'NVJITLINK_INPUT_FATBIN',
    5: 'NVJITLINK_INPUT_OBJECT',
    6: 'NVJITLINK_INPUT_LIBRARY',
    10: 'NVJITLINK_INPUT_ANY',
}
NVJITLINK_INPUT_NONE = 0
NVJITLINK_INPUT_CUBIN = 1
NVJITLINK_INPUT_PTX = 2
NVJITLINK_INPUT_LTOIR = 3
NVJITLINK_INPUT_FATBIN = 4
NVJITLINK_INPUT_OBJECT = 5
NVJITLINK_INPUT_LIBRARY = 6
NVJITLINK_INPUT_ANY = 10
c__EA_nvJitLinkInputType = ctypes.c_uint32 # enum
nvJitLinkInputType = c__EA_nvJitLinkInputType
nvJitLinkInputType__enumvalues = c__EA_nvJitLinkInputType__enumvalues
class struct_nvJitLink(Structure):
    pass

nvJitLinkHandle = ctypes.POINTER(struct_nvJitLink)
uint32_t = ctypes.c_uint32
try:
    __nvJitLinkCreate_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkCreate_12_4
    __nvJitLinkCreate_12_4.restype = nvJitLinkResult
    __nvJitLinkCreate_12_4.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_nvJitLink)), uint32_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    nvJitLinkCreate = _libraries['libnvJitLink.so'].nvJitLinkCreate
    nvJitLinkCreate.restype = nvJitLinkResult
    nvJitLinkCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_nvJitLink)), uint32_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    __nvJitLinkDestroy_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkDestroy_12_4
    __nvJitLinkDestroy_12_4.restype = nvJitLinkResult
    __nvJitLinkDestroy_12_4.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_nvJitLink))]
except AttributeError:
    pass
try:
    nvJitLinkDestroy = _libraries['libnvJitLink.so'].nvJitLinkDestroy
    nvJitLinkDestroy.restype = nvJitLinkResult
    nvJitLinkDestroy.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_nvJitLink))]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    __nvJitLinkAddData_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkAddData_12_4
    __nvJitLinkAddData_12_4.restype = nvJitLinkResult
    __nvJitLinkAddData_12_4.argtypes = [nvJitLinkHandle, nvJitLinkInputType, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvJitLinkAddData = _libraries['libnvJitLink.so'].nvJitLinkAddData
    nvJitLinkAddData.restype = nvJitLinkResult
    nvJitLinkAddData.argtypes = [nvJitLinkHandle, nvJitLinkInputType, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    __nvJitLinkAddFile_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkAddFile_12_4
    __nvJitLinkAddFile_12_4.restype = nvJitLinkResult
    __nvJitLinkAddFile_12_4.argtypes = [nvJitLinkHandle, nvJitLinkInputType, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvJitLinkAddFile = _libraries['libnvJitLink.so'].nvJitLinkAddFile
    nvJitLinkAddFile.restype = nvJitLinkResult
    nvJitLinkAddFile.argtypes = [nvJitLinkHandle, nvJitLinkInputType, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    __nvJitLinkComplete_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkComplete_12_4
    __nvJitLinkComplete_12_4.restype = nvJitLinkResult
    __nvJitLinkComplete_12_4.argtypes = [nvJitLinkHandle]
except AttributeError:
    pass
try:
    nvJitLinkComplete = _libraries['libnvJitLink.so'].nvJitLinkComplete
    nvJitLinkComplete.restype = nvJitLinkResult
    nvJitLinkComplete.argtypes = [nvJitLinkHandle]
except AttributeError:
    pass
try:
    __nvJitLinkGetLinkedCubinSize_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkGetLinkedCubinSize_12_4
    __nvJitLinkGetLinkedCubinSize_12_4.restype = nvJitLinkResult
    __nvJitLinkGetLinkedCubinSize_12_4.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvJitLinkGetLinkedCubinSize = _libraries['libnvJitLink.so'].nvJitLinkGetLinkedCubinSize
    nvJitLinkGetLinkedCubinSize.restype = nvJitLinkResult
    nvJitLinkGetLinkedCubinSize.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    __nvJitLinkGetLinkedCubin_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkGetLinkedCubin_12_4
    __nvJitLinkGetLinkedCubin_12_4.restype = nvJitLinkResult
    __nvJitLinkGetLinkedCubin_12_4.argtypes = [nvJitLinkHandle, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nvJitLinkGetLinkedCubin = _libraries['libnvJitLink.so'].nvJitLinkGetLinkedCubin
    nvJitLinkGetLinkedCubin.restype = nvJitLinkResult
    nvJitLinkGetLinkedCubin.argtypes = [nvJitLinkHandle, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    __nvJitLinkGetLinkedPtxSize_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkGetLinkedPtxSize_12_4
    __nvJitLinkGetLinkedPtxSize_12_4.restype = nvJitLinkResult
    __nvJitLinkGetLinkedPtxSize_12_4.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvJitLinkGetLinkedPtxSize = _libraries['libnvJitLink.so'].nvJitLinkGetLinkedPtxSize
    nvJitLinkGetLinkedPtxSize.restype = nvJitLinkResult
    nvJitLinkGetLinkedPtxSize.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    __nvJitLinkGetLinkedPtx_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkGetLinkedPtx_12_4
    __nvJitLinkGetLinkedPtx_12_4.restype = nvJitLinkResult
    __nvJitLinkGetLinkedPtx_12_4.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvJitLinkGetLinkedPtx = _libraries['libnvJitLink.so'].nvJitLinkGetLinkedPtx
    nvJitLinkGetLinkedPtx.restype = nvJitLinkResult
    nvJitLinkGetLinkedPtx.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    __nvJitLinkGetErrorLogSize_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkGetErrorLogSize_12_4
    __nvJitLinkGetErrorLogSize_12_4.restype = nvJitLinkResult
    __nvJitLinkGetErrorLogSize_12_4.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvJitLinkGetErrorLogSize = _libraries['libnvJitLink.so'].nvJitLinkGetErrorLogSize
    nvJitLinkGetErrorLogSize.restype = nvJitLinkResult
    nvJitLinkGetErrorLogSize.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    __nvJitLinkGetErrorLog_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkGetErrorLog_12_4
    __nvJitLinkGetErrorLog_12_4.restype = nvJitLinkResult
    __nvJitLinkGetErrorLog_12_4.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvJitLinkGetErrorLog = _libraries['libnvJitLink.so'].nvJitLinkGetErrorLog
    nvJitLinkGetErrorLog.restype = nvJitLinkResult
    nvJitLinkGetErrorLog.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    __nvJitLinkGetInfoLogSize_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkGetInfoLogSize_12_4
    __nvJitLinkGetInfoLogSize_12_4.restype = nvJitLinkResult
    __nvJitLinkGetInfoLogSize_12_4.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvJitLinkGetInfoLogSize = _libraries['libnvJitLink.so'].nvJitLinkGetInfoLogSize
    nvJitLinkGetInfoLogSize.restype = nvJitLinkResult
    nvJitLinkGetInfoLogSize.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    __nvJitLinkGetInfoLog_12_4 = _libraries['libnvJitLink.so'].__nvJitLinkGetInfoLog_12_4
    __nvJitLinkGetInfoLog_12_4.restype = nvJitLinkResult
    __nvJitLinkGetInfoLog_12_4.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvJitLinkGetInfoLog = _libraries['libnvJitLink.so'].nvJitLinkGetInfoLog
    nvJitLinkGetInfoLog.restype = nvJitLinkResult
    nvJitLinkGetInfoLog.argtypes = [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvJitLinkVersion = _libraries['libnvJitLink.so'].nvJitLinkVersion
    nvJitLinkVersion.restype = nvJitLinkResult
    nvJitLinkVersion.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
__all__ = \
    ['NVJITLINK_ERROR_INTERNAL', 'NVJITLINK_ERROR_INVALID_INPUT',
    'NVJITLINK_ERROR_MISSING_ARCH', 'NVJITLINK_ERROR_NVVM_COMPILE',
    'NVJITLINK_ERROR_PTX_COMPILE', 'NVJITLINK_ERROR_THREADPOOL',
    'NVJITLINK_ERROR_UNRECOGNIZED_INPUT',
    'NVJITLINK_ERROR_UNRECOGNIZED_OPTION', 'NVJITLINK_INPUT_ANY',
    'NVJITLINK_INPUT_CUBIN', 'NVJITLINK_INPUT_FATBIN',
    'NVJITLINK_INPUT_LIBRARY', 'NVJITLINK_INPUT_LTOIR',
    'NVJITLINK_INPUT_NONE', 'NVJITLINK_INPUT_OBJECT',
    'NVJITLINK_INPUT_PTX', 'NVJITLINK_SUCCESS',
    'NVRTC_ERROR_BUILTIN_OPERATION_FAILURE',
    'NVRTC_ERROR_COMPILATION', 'NVRTC_ERROR_INTERNAL_ERROR',
    'NVRTC_ERROR_INVALID_INPUT', 'NVRTC_ERROR_INVALID_OPTION',
    'NVRTC_ERROR_INVALID_PROGRAM',
    'NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID',
    'NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION',
    'NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION',
    'NVRTC_ERROR_OUT_OF_MEMORY',
    'NVRTC_ERROR_PROGRAM_CREATION_FAILURE',
    'NVRTC_ERROR_TIME_FILE_WRITE_FAILED', 'NVRTC_SUCCESS',
    '__nvJitLinkAddData_12_4', '__nvJitLinkAddFile_12_4',
    '__nvJitLinkComplete_12_4', '__nvJitLinkCreate_12_4',
    '__nvJitLinkDestroy_12_4', '__nvJitLinkGetErrorLogSize_12_4',
    '__nvJitLinkGetErrorLog_12_4', '__nvJitLinkGetInfoLogSize_12_4',
    '__nvJitLinkGetInfoLog_12_4',
    '__nvJitLinkGetLinkedCubinSize_12_4',
    '__nvJitLinkGetLinkedCubin_12_4',
    '__nvJitLinkGetLinkedPtxSize_12_4',
    '__nvJitLinkGetLinkedPtx_12_4', 'c__EA_nvJitLinkInputType',
    'c__EA_nvJitLinkResult', 'c__EA_nvrtcResult', 'nvJitLinkAddData',
    'nvJitLinkAddFile', 'nvJitLinkComplete', 'nvJitLinkCreate',
    'nvJitLinkDestroy', 'nvJitLinkGetErrorLog',
    'nvJitLinkGetErrorLogSize', 'nvJitLinkGetInfoLog',
    'nvJitLinkGetInfoLogSize', 'nvJitLinkGetLinkedCubin',
    'nvJitLinkGetLinkedCubinSize', 'nvJitLinkGetLinkedPtx',
    'nvJitLinkGetLinkedPtxSize', 'nvJitLinkHandle',
    'nvJitLinkInputType', 'nvJitLinkInputType__enumvalues',
    'nvJitLinkResult', 'nvJitLinkResult__enumvalues',
    'nvJitLinkVersion', 'nvrtcAddNameExpression',
    'nvrtcCompileProgram', 'nvrtcCreateProgram',
    'nvrtcDestroyProgram', 'nvrtcGetCUBIN', 'nvrtcGetCUBINSize',
    'nvrtcGetErrorString', 'nvrtcGetLTOIR', 'nvrtcGetLTOIRSize',
    'nvrtcGetLoweredName', 'nvrtcGetNVVM', 'nvrtcGetNVVMSize',
    'nvrtcGetNumSupportedArchs', 'nvrtcGetOptiXIR',
    'nvrtcGetOptiXIRSize', 'nvrtcGetPTX', 'nvrtcGetPTXSize',
    'nvrtcGetProgramLog', 'nvrtcGetProgramLogSize',
    'nvrtcGetSupportedArchs', 'nvrtcProgram', 'nvrtcResult',
    'nvrtcResult__enumvalues', 'nvrtcVersion', 'size_t',
    'struct__nvrtcProgram', 'struct_nvJitLink', 'uint32_t']
