# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
import sysconfig
dll = DLL('nvrtc', 'nvrtc', f'/usr/local/cuda/targets/{sysconfig.get_config_var("MULTIARCH").rsplit("-", 1)[0]}/lib')
nvrtcResult = CEnum(ctypes.c_uint32)
NVRTC_SUCCESS = nvrtcResult.define('NVRTC_SUCCESS', 0)
NVRTC_ERROR_OUT_OF_MEMORY = nvrtcResult.define('NVRTC_ERROR_OUT_OF_MEMORY', 1)
NVRTC_ERROR_PROGRAM_CREATION_FAILURE = nvrtcResult.define('NVRTC_ERROR_PROGRAM_CREATION_FAILURE', 2)
NVRTC_ERROR_INVALID_INPUT = nvrtcResult.define('NVRTC_ERROR_INVALID_INPUT', 3)
NVRTC_ERROR_INVALID_PROGRAM = nvrtcResult.define('NVRTC_ERROR_INVALID_PROGRAM', 4)
NVRTC_ERROR_INVALID_OPTION = nvrtcResult.define('NVRTC_ERROR_INVALID_OPTION', 5)
NVRTC_ERROR_COMPILATION = nvrtcResult.define('NVRTC_ERROR_COMPILATION', 6)
NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = nvrtcResult.define('NVRTC_ERROR_BUILTIN_OPERATION_FAILURE', 7)
NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = nvrtcResult.define('NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION', 8)
NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = nvrtcResult.define('NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION', 9)
NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = nvrtcResult.define('NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID', 10)
NVRTC_ERROR_INTERNAL_ERROR = nvrtcResult.define('NVRTC_ERROR_INTERNAL_ERROR', 11)

try: (nvrtcGetErrorString:=dll.nvrtcGetErrorString).restype, nvrtcGetErrorString.argtypes = ctypes.POINTER(ctypes.c_char), [nvrtcResult]
except AttributeError: pass

try: (nvrtcVersion:=dll.nvrtcVersion).restype, nvrtcVersion.argtypes = nvrtcResult, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

try: (nvrtcGetNumSupportedArchs:=dll.nvrtcGetNumSupportedArchs).restype, nvrtcGetNumSupportedArchs.argtypes = nvrtcResult, [ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

try: (nvrtcGetSupportedArchs:=dll.nvrtcGetSupportedArchs).restype, nvrtcGetSupportedArchs.argtypes = nvrtcResult, [ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

class struct__nvrtcProgram(Struct): pass
nvrtcProgram = ctypes.POINTER(struct__nvrtcProgram)
try: (nvrtcCreateProgram:=dll.nvrtcCreateProgram).restype, nvrtcCreateProgram.argtypes = nvrtcResult, [ctypes.POINTER(nvrtcProgram), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (nvrtcDestroyProgram:=dll.nvrtcDestroyProgram).restype, nvrtcDestroyProgram.argtypes = nvrtcResult, [ctypes.POINTER(nvrtcProgram)]
except AttributeError: pass

try: (nvrtcCompileProgram:=dll.nvrtcCompileProgram).restype, nvrtcCompileProgram.argtypes = nvrtcResult, [nvrtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

size_t = ctypes.c_uint64
try: (nvrtcGetPTXSize:=dll.nvrtcGetPTXSize).restype, nvrtcGetPTXSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (nvrtcGetPTX:=dll.nvrtcGetPTX).restype, nvrtcGetPTX.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nvrtcGetCUBINSize:=dll.nvrtcGetCUBINSize).restype, nvrtcGetCUBINSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (nvrtcGetCUBIN:=dll.nvrtcGetCUBIN).restype, nvrtcGetCUBIN.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nvrtcGetNVVMSize:=dll.nvrtcGetNVVMSize).restype, nvrtcGetNVVMSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (nvrtcGetNVVM:=dll.nvrtcGetNVVM).restype, nvrtcGetNVVM.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nvrtcGetLTOIRSize:=dll.nvrtcGetLTOIRSize).restype, nvrtcGetLTOIRSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (nvrtcGetLTOIR:=dll.nvrtcGetLTOIR).restype, nvrtcGetLTOIR.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nvrtcGetOptiXIRSize:=dll.nvrtcGetOptiXIRSize).restype, nvrtcGetOptiXIRSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (nvrtcGetOptiXIR:=dll.nvrtcGetOptiXIR).restype, nvrtcGetOptiXIR.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nvrtcGetProgramLogSize:=dll.nvrtcGetProgramLogSize).restype, nvrtcGetProgramLogSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (nvrtcGetProgramLog:=dll.nvrtcGetProgramLog).restype, nvrtcGetProgramLog.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nvrtcAddNameExpression:=dll.nvrtcAddNameExpression).restype, nvrtcAddNameExpression.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nvrtcGetLoweredName:=dll.nvrtcGetLoweredName).restype, nvrtcGetLoweredName.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

__DEPRECATED__ = lambda msg: __attribute__((deprecated(msg)))