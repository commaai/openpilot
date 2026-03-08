# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
import os
dll = c.DLL('hip', os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so')
class ihipModuleSymbol_t(ctypes.Structure): pass
hipFunction_t: TypeAlias = c.POINTER[ihipModuleSymbol_t]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
class ihipStream_t(ctypes.Structure): pass
hipStream_t: TypeAlias = c.POINTER[ihipStream_t]
class ihipEvent_t(ctypes.Structure): pass
hipEvent_t: TypeAlias = c.POINTER[ihipEvent_t]
class hipError_t(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipSuccess = hipError_t.define('hipSuccess', 0)
hipErrorInvalidValue = hipError_t.define('hipErrorInvalidValue', 1)
hipErrorOutOfMemory = hipError_t.define('hipErrorOutOfMemory', 2)
hipErrorMemoryAllocation = hipError_t.define('hipErrorMemoryAllocation', 2)
hipErrorNotInitialized = hipError_t.define('hipErrorNotInitialized', 3)
hipErrorInitializationError = hipError_t.define('hipErrorInitializationError', 3)
hipErrorDeinitialized = hipError_t.define('hipErrorDeinitialized', 4)
hipErrorProfilerDisabled = hipError_t.define('hipErrorProfilerDisabled', 5)
hipErrorProfilerNotInitialized = hipError_t.define('hipErrorProfilerNotInitialized', 6)
hipErrorProfilerAlreadyStarted = hipError_t.define('hipErrorProfilerAlreadyStarted', 7)
hipErrorProfilerAlreadyStopped = hipError_t.define('hipErrorProfilerAlreadyStopped', 8)
hipErrorInvalidConfiguration = hipError_t.define('hipErrorInvalidConfiguration', 9)
hipErrorInvalidPitchValue = hipError_t.define('hipErrorInvalidPitchValue', 12)
hipErrorInvalidSymbol = hipError_t.define('hipErrorInvalidSymbol', 13)
hipErrorInvalidDevicePointer = hipError_t.define('hipErrorInvalidDevicePointer', 17)
hipErrorInvalidMemcpyDirection = hipError_t.define('hipErrorInvalidMemcpyDirection', 21)
hipErrorInsufficientDriver = hipError_t.define('hipErrorInsufficientDriver', 35)
hipErrorMissingConfiguration = hipError_t.define('hipErrorMissingConfiguration', 52)
hipErrorPriorLaunchFailure = hipError_t.define('hipErrorPriorLaunchFailure', 53)
hipErrorInvalidDeviceFunction = hipError_t.define('hipErrorInvalidDeviceFunction', 98)
hipErrorNoDevice = hipError_t.define('hipErrorNoDevice', 100)
hipErrorInvalidDevice = hipError_t.define('hipErrorInvalidDevice', 101)
hipErrorInvalidImage = hipError_t.define('hipErrorInvalidImage', 200)
hipErrorInvalidContext = hipError_t.define('hipErrorInvalidContext', 201)
hipErrorContextAlreadyCurrent = hipError_t.define('hipErrorContextAlreadyCurrent', 202)
hipErrorMapFailed = hipError_t.define('hipErrorMapFailed', 205)
hipErrorMapBufferObjectFailed = hipError_t.define('hipErrorMapBufferObjectFailed', 205)
hipErrorUnmapFailed = hipError_t.define('hipErrorUnmapFailed', 206)
hipErrorArrayIsMapped = hipError_t.define('hipErrorArrayIsMapped', 207)
hipErrorAlreadyMapped = hipError_t.define('hipErrorAlreadyMapped', 208)
hipErrorNoBinaryForGpu = hipError_t.define('hipErrorNoBinaryForGpu', 209)
hipErrorAlreadyAcquired = hipError_t.define('hipErrorAlreadyAcquired', 210)
hipErrorNotMapped = hipError_t.define('hipErrorNotMapped', 211)
hipErrorNotMappedAsArray = hipError_t.define('hipErrorNotMappedAsArray', 212)
hipErrorNotMappedAsPointer = hipError_t.define('hipErrorNotMappedAsPointer', 213)
hipErrorECCNotCorrectable = hipError_t.define('hipErrorECCNotCorrectable', 214)
hipErrorUnsupportedLimit = hipError_t.define('hipErrorUnsupportedLimit', 215)
hipErrorContextAlreadyInUse = hipError_t.define('hipErrorContextAlreadyInUse', 216)
hipErrorPeerAccessUnsupported = hipError_t.define('hipErrorPeerAccessUnsupported', 217)
hipErrorInvalidKernelFile = hipError_t.define('hipErrorInvalidKernelFile', 218)
hipErrorInvalidGraphicsContext = hipError_t.define('hipErrorInvalidGraphicsContext', 219)
hipErrorInvalidSource = hipError_t.define('hipErrorInvalidSource', 300)
hipErrorFileNotFound = hipError_t.define('hipErrorFileNotFound', 301)
hipErrorSharedObjectSymbolNotFound = hipError_t.define('hipErrorSharedObjectSymbolNotFound', 302)
hipErrorSharedObjectInitFailed = hipError_t.define('hipErrorSharedObjectInitFailed', 303)
hipErrorOperatingSystem = hipError_t.define('hipErrorOperatingSystem', 304)
hipErrorInvalidHandle = hipError_t.define('hipErrorInvalidHandle', 400)
hipErrorInvalidResourceHandle = hipError_t.define('hipErrorInvalidResourceHandle', 400)
hipErrorIllegalState = hipError_t.define('hipErrorIllegalState', 401)
hipErrorNotFound = hipError_t.define('hipErrorNotFound', 500)
hipErrorNotReady = hipError_t.define('hipErrorNotReady', 600)
hipErrorIllegalAddress = hipError_t.define('hipErrorIllegalAddress', 700)
hipErrorLaunchOutOfResources = hipError_t.define('hipErrorLaunchOutOfResources', 701)
hipErrorLaunchTimeOut = hipError_t.define('hipErrorLaunchTimeOut', 702)
hipErrorPeerAccessAlreadyEnabled = hipError_t.define('hipErrorPeerAccessAlreadyEnabled', 704)
hipErrorPeerAccessNotEnabled = hipError_t.define('hipErrorPeerAccessNotEnabled', 705)
hipErrorSetOnActiveProcess = hipError_t.define('hipErrorSetOnActiveProcess', 708)
hipErrorContextIsDestroyed = hipError_t.define('hipErrorContextIsDestroyed', 709)
hipErrorAssert = hipError_t.define('hipErrorAssert', 710)
hipErrorHostMemoryAlreadyRegistered = hipError_t.define('hipErrorHostMemoryAlreadyRegistered', 712)
hipErrorHostMemoryNotRegistered = hipError_t.define('hipErrorHostMemoryNotRegistered', 713)
hipErrorLaunchFailure = hipError_t.define('hipErrorLaunchFailure', 719)
hipErrorCooperativeLaunchTooLarge = hipError_t.define('hipErrorCooperativeLaunchTooLarge', 720)
hipErrorNotSupported = hipError_t.define('hipErrorNotSupported', 801)
hipErrorStreamCaptureUnsupported = hipError_t.define('hipErrorStreamCaptureUnsupported', 900)
hipErrorStreamCaptureInvalidated = hipError_t.define('hipErrorStreamCaptureInvalidated', 901)
hipErrorStreamCaptureMerge = hipError_t.define('hipErrorStreamCaptureMerge', 902)
hipErrorStreamCaptureUnmatched = hipError_t.define('hipErrorStreamCaptureUnmatched', 903)
hipErrorStreamCaptureUnjoined = hipError_t.define('hipErrorStreamCaptureUnjoined', 904)
hipErrorStreamCaptureIsolation = hipError_t.define('hipErrorStreamCaptureIsolation', 905)
hipErrorStreamCaptureImplicit = hipError_t.define('hipErrorStreamCaptureImplicit', 906)
hipErrorCapturedEvent = hipError_t.define('hipErrorCapturedEvent', 907)
hipErrorStreamCaptureWrongThread = hipError_t.define('hipErrorStreamCaptureWrongThread', 908)
hipErrorGraphExecUpdateFailure = hipError_t.define('hipErrorGraphExecUpdateFailure', 910)
hipErrorUnknown = hipError_t.define('hipErrorUnknown', 999)
hipErrorRuntimeMemory = hipError_t.define('hipErrorRuntimeMemory', 1052)
hipErrorRuntimeOther = hipError_t.define('hipErrorRuntimeOther', 1053)
hipErrorTbd = hipError_t.define('hipErrorTbd', 1054)

@dll.bind
def hipExtModuleLaunchKernel(f:hipFunction_t, globalWorkSizeX:uint32_t, globalWorkSizeY:uint32_t, globalWorkSizeZ:uint32_t, localWorkSizeX:uint32_t, localWorkSizeY:uint32_t, localWorkSizeZ:uint32_t, sharedMemBytes:size_t, hStream:hipStream_t, kernelParams:c.POINTER[ctypes.c_void_p], extra:c.POINTER[ctypes.c_void_p], startEvent:hipEvent_t, stopEvent:hipEvent_t, flags:uint32_t) -> hipError_t: ...
@dll.bind
def hipHccModuleLaunchKernel(f:hipFunction_t, globalWorkSizeX:uint32_t, globalWorkSizeY:uint32_t, globalWorkSizeZ:uint32_t, localWorkSizeX:uint32_t, localWorkSizeY:uint32_t, localWorkSizeZ:uint32_t, sharedMemBytes:size_t, hStream:hipStream_t, kernelParams:c.POINTER[ctypes.c_void_p], extra:c.POINTER[ctypes.c_void_p], startEvent:hipEvent_t, stopEvent:hipEvent_t) -> hipError_t: ...
@c.record
class dim3(c.Struct):
  SIZE = 12
  x: Annotated[uint32_t, 0]
  y: Annotated[uint32_t, 4]
  z: Annotated[uint32_t, 8]
@dll.bind
def hipExtLaunchKernel(function_address:ctypes.c_void_p, numBlocks:dim3, dimBlocks:dim3, args:c.POINTER[ctypes.c_void_p], sharedMemBytes:size_t, stream:hipStream_t, startEvent:hipEvent_t, stopEvent:hipEvent_t, flags:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
class hiprtcResult(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIPRTC_SUCCESS = hiprtcResult.define('HIPRTC_SUCCESS', 0)
HIPRTC_ERROR_OUT_OF_MEMORY = hiprtcResult.define('HIPRTC_ERROR_OUT_OF_MEMORY', 1)
HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = hiprtcResult.define('HIPRTC_ERROR_PROGRAM_CREATION_FAILURE', 2)
HIPRTC_ERROR_INVALID_INPUT = hiprtcResult.define('HIPRTC_ERROR_INVALID_INPUT', 3)
HIPRTC_ERROR_INVALID_PROGRAM = hiprtcResult.define('HIPRTC_ERROR_INVALID_PROGRAM', 4)
HIPRTC_ERROR_INVALID_OPTION = hiprtcResult.define('HIPRTC_ERROR_INVALID_OPTION', 5)
HIPRTC_ERROR_COMPILATION = hiprtcResult.define('HIPRTC_ERROR_COMPILATION', 6)
HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = hiprtcResult.define('HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE', 7)
HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = hiprtcResult.define('HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION', 8)
HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = hiprtcResult.define('HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION', 9)
HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = hiprtcResult.define('HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID', 10)
HIPRTC_ERROR_INTERNAL_ERROR = hiprtcResult.define('HIPRTC_ERROR_INTERNAL_ERROR', 11)
HIPRTC_ERROR_LINKING = hiprtcResult.define('HIPRTC_ERROR_LINKING', 100)

class hiprtcJIT_option(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIPRTC_JIT_MAX_REGISTERS = hiprtcJIT_option.define('HIPRTC_JIT_MAX_REGISTERS', 0)
HIPRTC_JIT_THREADS_PER_BLOCK = hiprtcJIT_option.define('HIPRTC_JIT_THREADS_PER_BLOCK', 1)
HIPRTC_JIT_WALL_TIME = hiprtcJIT_option.define('HIPRTC_JIT_WALL_TIME', 2)
HIPRTC_JIT_INFO_LOG_BUFFER = hiprtcJIT_option.define('HIPRTC_JIT_INFO_LOG_BUFFER', 3)
HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES = hiprtcJIT_option.define('HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES', 4)
HIPRTC_JIT_ERROR_LOG_BUFFER = hiprtcJIT_option.define('HIPRTC_JIT_ERROR_LOG_BUFFER', 5)
HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = hiprtcJIT_option.define('HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES', 6)
HIPRTC_JIT_OPTIMIZATION_LEVEL = hiprtcJIT_option.define('HIPRTC_JIT_OPTIMIZATION_LEVEL', 7)
HIPRTC_JIT_TARGET_FROM_HIPCONTEXT = hiprtcJIT_option.define('HIPRTC_JIT_TARGET_FROM_HIPCONTEXT', 8)
HIPRTC_JIT_TARGET = hiprtcJIT_option.define('HIPRTC_JIT_TARGET', 9)
HIPRTC_JIT_FALLBACK_STRATEGY = hiprtcJIT_option.define('HIPRTC_JIT_FALLBACK_STRATEGY', 10)
HIPRTC_JIT_GENERATE_DEBUG_INFO = hiprtcJIT_option.define('HIPRTC_JIT_GENERATE_DEBUG_INFO', 11)
HIPRTC_JIT_LOG_VERBOSE = hiprtcJIT_option.define('HIPRTC_JIT_LOG_VERBOSE', 12)
HIPRTC_JIT_GENERATE_LINE_INFO = hiprtcJIT_option.define('HIPRTC_JIT_GENERATE_LINE_INFO', 13)
HIPRTC_JIT_CACHE_MODE = hiprtcJIT_option.define('HIPRTC_JIT_CACHE_MODE', 14)
HIPRTC_JIT_NEW_SM3X_OPT = hiprtcJIT_option.define('HIPRTC_JIT_NEW_SM3X_OPT', 15)
HIPRTC_JIT_FAST_COMPILE = hiprtcJIT_option.define('HIPRTC_JIT_FAST_COMPILE', 16)
HIPRTC_JIT_GLOBAL_SYMBOL_NAMES = hiprtcJIT_option.define('HIPRTC_JIT_GLOBAL_SYMBOL_NAMES', 17)
HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS = hiprtcJIT_option.define('HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS', 18)
HIPRTC_JIT_GLOBAL_SYMBOL_COUNT = hiprtcJIT_option.define('HIPRTC_JIT_GLOBAL_SYMBOL_COUNT', 19)
HIPRTC_JIT_LTO = hiprtcJIT_option.define('HIPRTC_JIT_LTO', 20)
HIPRTC_JIT_FTZ = hiprtcJIT_option.define('HIPRTC_JIT_FTZ', 21)
HIPRTC_JIT_PREC_DIV = hiprtcJIT_option.define('HIPRTC_JIT_PREC_DIV', 22)
HIPRTC_JIT_PREC_SQRT = hiprtcJIT_option.define('HIPRTC_JIT_PREC_SQRT', 23)
HIPRTC_JIT_FMA = hiprtcJIT_option.define('HIPRTC_JIT_FMA', 24)
HIPRTC_JIT_NUM_OPTIONS = hiprtcJIT_option.define('HIPRTC_JIT_NUM_OPTIONS', 25)
HIPRTC_JIT_IR_TO_ISA_OPT_EXT = hiprtcJIT_option.define('HIPRTC_JIT_IR_TO_ISA_OPT_EXT', 10000)
HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT = hiprtcJIT_option.define('HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT', 10001)

class hiprtcJITInputType(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIPRTC_JIT_INPUT_CUBIN = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_CUBIN', 0)
HIPRTC_JIT_INPUT_PTX = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_PTX', 1)
HIPRTC_JIT_INPUT_FATBINARY = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_FATBINARY', 2)
HIPRTC_JIT_INPUT_OBJECT = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_OBJECT', 3)
HIPRTC_JIT_INPUT_LIBRARY = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_LIBRARY', 4)
HIPRTC_JIT_INPUT_NVVM = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_NVVM', 5)
HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = hiprtcJITInputType.define('HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES', 6)
HIPRTC_JIT_INPUT_LLVM_BITCODE = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_LLVM_BITCODE', 100)
HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE', 101)
HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE', 102)
HIPRTC_JIT_NUM_INPUT_TYPES = hiprtcJITInputType.define('HIPRTC_JIT_NUM_INPUT_TYPES', 9)

class ihiprtcLinkState(ctypes.Structure): pass
hiprtcLinkState: TypeAlias = c.POINTER[ihiprtcLinkState]
@dll.bind
def hiprtcGetErrorString(result:hiprtcResult) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def hiprtcVersion(major:c.POINTER[Annotated[int, ctypes.c_int32]], minor:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hiprtcResult: ...
class _hiprtcProgram(ctypes.Structure): pass
hiprtcProgram: TypeAlias = c.POINTER[_hiprtcProgram]
@dll.bind
def hiprtcAddNameExpression(prog:hiprtcProgram, name_expression:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hiprtcResult: ...
@dll.bind
def hiprtcCompileProgram(prog:hiprtcProgram, numOptions:Annotated[int, ctypes.c_int32], options:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> hiprtcResult: ...
@dll.bind
def hiprtcCreateProgram(prog:c.POINTER[hiprtcProgram], src:c.POINTER[Annotated[bytes, ctypes.c_char]], name:c.POINTER[Annotated[bytes, ctypes.c_char]], numHeaders:Annotated[int, ctypes.c_int32], headers:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], includeNames:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> hiprtcResult: ...
@dll.bind
def hiprtcDestroyProgram(prog:c.POINTER[hiprtcProgram]) -> hiprtcResult: ...
@dll.bind
def hiprtcGetLoweredName(prog:hiprtcProgram, name_expression:c.POINTER[Annotated[bytes, ctypes.c_char]], lowered_name:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> hiprtcResult: ...
@dll.bind
def hiprtcGetProgramLog(prog:hiprtcProgram, log:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hiprtcResult: ...
@dll.bind
def hiprtcGetProgramLogSize(prog:hiprtcProgram, logSizeRet:c.POINTER[size_t]) -> hiprtcResult: ...
@dll.bind
def hiprtcGetCode(prog:hiprtcProgram, code:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hiprtcResult: ...
@dll.bind
def hiprtcGetCodeSize(prog:hiprtcProgram, codeSizeRet:c.POINTER[size_t]) -> hiprtcResult: ...
@dll.bind
def hiprtcGetBitcode(prog:hiprtcProgram, bitcode:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hiprtcResult: ...
@dll.bind
def hiprtcGetBitcodeSize(prog:hiprtcProgram, bitcode_size:c.POINTER[size_t]) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkCreate(num_options:Annotated[int, ctypes.c_uint32], option_ptr:c.POINTER[hiprtcJIT_option], option_vals_pptr:c.POINTER[ctypes.c_void_p], hip_link_state_ptr:c.POINTER[hiprtcLinkState]) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkAddFile(hip_link_state:hiprtcLinkState, input_type:hiprtcJITInputType, file_path:c.POINTER[Annotated[bytes, ctypes.c_char]], num_options:Annotated[int, ctypes.c_uint32], options_ptr:c.POINTER[hiprtcJIT_option], option_values:c.POINTER[ctypes.c_void_p]) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkAddData(hip_link_state:hiprtcLinkState, input_type:hiprtcJITInputType, image:ctypes.c_void_p, image_size:size_t, name:c.POINTER[Annotated[bytes, ctypes.c_char]], num_options:Annotated[int, ctypes.c_uint32], options_ptr:c.POINTER[hiprtcJIT_option], option_values:c.POINTER[ctypes.c_void_p]) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkComplete(hip_link_state:hiprtcLinkState, bin_out:c.POINTER[ctypes.c_void_p], size_out:c.POINTER[size_t]) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkDestroy(hip_link_state:hiprtcLinkState) -> hiprtcResult: ...
class _anonenum0(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_SUCCESS = _anonenum0.define('HIP_SUCCESS', 0)
HIP_ERROR_INVALID_VALUE = _anonenum0.define('HIP_ERROR_INVALID_VALUE', 1)
HIP_ERROR_NOT_INITIALIZED = _anonenum0.define('HIP_ERROR_NOT_INITIALIZED', 2)
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = _anonenum0.define('HIP_ERROR_LAUNCH_OUT_OF_RESOURCES', 3)

@c.record
class hipDeviceArch_t(c.Struct):
  SIZE = 4
  hasGlobalInt32Atomics: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 0]
  hasGlobalFloatAtomicExch: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 1]
  hasSharedInt32Atomics: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 2]
  hasSharedFloatAtomicExch: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 3]
  hasFloatAtomicAdd: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 4]
  hasGlobalInt64Atomics: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 5]
  hasSharedInt64Atomics: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 6]
  hasDoubles: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 7]
  hasWarpVote: Annotated[Annotated[int, ctypes.c_uint32], 1, 1, 0]
  hasWarpBallot: Annotated[Annotated[int, ctypes.c_uint32], 1, 1, 1]
  hasWarpShuffle: Annotated[Annotated[int, ctypes.c_uint32], 1, 1, 2]
  hasFunnelShift: Annotated[Annotated[int, ctypes.c_uint32], 1, 1, 3]
  hasThreadFenceSystem: Annotated[Annotated[int, ctypes.c_uint32], 1, 1, 4]
  hasSyncThreadsExt: Annotated[Annotated[int, ctypes.c_uint32], 1, 1, 5]
  hasSurfaceFuncs: Annotated[Annotated[int, ctypes.c_uint32], 1, 1, 6]
  has3dGrid: Annotated[Annotated[int, ctypes.c_uint32], 1, 1, 7]
  hasDynamicParallelism: Annotated[Annotated[int, ctypes.c_uint32], 2, 1, 0]
@c.record
class hipUUID_t(c.Struct):
  SIZE = 16
  bytes: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[16]], 0]
hipUUID: TypeAlias = hipUUID_t
@c.record
class hipDeviceProp_tR0600(c.Struct):
  SIZE = 1472
  name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[256]], 0]
  uuid: Annotated[hipUUID, 256]
  luid: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[8]], 272]
  luidDeviceNodeMask: Annotated[Annotated[int, ctypes.c_uint32], 280]
  totalGlobalMem: Annotated[size_t, 288]
  sharedMemPerBlock: Annotated[size_t, 296]
  regsPerBlock: Annotated[Annotated[int, ctypes.c_int32], 304]
  warpSize: Annotated[Annotated[int, ctypes.c_int32], 308]
  memPitch: Annotated[size_t, 312]
  maxThreadsPerBlock: Annotated[Annotated[int, ctypes.c_int32], 320]
  maxThreadsDim: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 324]
  maxGridSize: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 336]
  clockRate: Annotated[Annotated[int, ctypes.c_int32], 348]
  totalConstMem: Annotated[size_t, 352]
  major: Annotated[Annotated[int, ctypes.c_int32], 360]
  minor: Annotated[Annotated[int, ctypes.c_int32], 364]
  textureAlignment: Annotated[size_t, 368]
  texturePitchAlignment: Annotated[size_t, 376]
  deviceOverlap: Annotated[Annotated[int, ctypes.c_int32], 384]
  multiProcessorCount: Annotated[Annotated[int, ctypes.c_int32], 388]
  kernelExecTimeoutEnabled: Annotated[Annotated[int, ctypes.c_int32], 392]
  integrated: Annotated[Annotated[int, ctypes.c_int32], 396]
  canMapHostMemory: Annotated[Annotated[int, ctypes.c_int32], 400]
  computeMode: Annotated[Annotated[int, ctypes.c_int32], 404]
  maxTexture1D: Annotated[Annotated[int, ctypes.c_int32], 408]
  maxTexture1DMipmap: Annotated[Annotated[int, ctypes.c_int32], 412]
  maxTexture1DLinear: Annotated[Annotated[int, ctypes.c_int32], 416]
  maxTexture2D: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[2]], 420]
  maxTexture2DMipmap: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[2]], 428]
  maxTexture2DLinear: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 436]
  maxTexture2DGather: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[2]], 448]
  maxTexture3D: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 456]
  maxTexture3DAlt: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 468]
  maxTextureCubemap: Annotated[Annotated[int, ctypes.c_int32], 480]
  maxTexture1DLayered: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[2]], 484]
  maxTexture2DLayered: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 492]
  maxTextureCubemapLayered: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[2]], 504]
  maxSurface1D: Annotated[Annotated[int, ctypes.c_int32], 512]
  maxSurface2D: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[2]], 516]
  maxSurface3D: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 524]
  maxSurface1DLayered: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[2]], 536]
  maxSurface2DLayered: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 544]
  maxSurfaceCubemap: Annotated[Annotated[int, ctypes.c_int32], 556]
  maxSurfaceCubemapLayered: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[2]], 560]
  surfaceAlignment: Annotated[size_t, 568]
  concurrentKernels: Annotated[Annotated[int, ctypes.c_int32], 576]
  ECCEnabled: Annotated[Annotated[int, ctypes.c_int32], 580]
  pciBusID: Annotated[Annotated[int, ctypes.c_int32], 584]
  pciDeviceID: Annotated[Annotated[int, ctypes.c_int32], 588]
  pciDomainID: Annotated[Annotated[int, ctypes.c_int32], 592]
  tccDriver: Annotated[Annotated[int, ctypes.c_int32], 596]
  asyncEngineCount: Annotated[Annotated[int, ctypes.c_int32], 600]
  unifiedAddressing: Annotated[Annotated[int, ctypes.c_int32], 604]
  memoryClockRate: Annotated[Annotated[int, ctypes.c_int32], 608]
  memoryBusWidth: Annotated[Annotated[int, ctypes.c_int32], 612]
  l2CacheSize: Annotated[Annotated[int, ctypes.c_int32], 616]
  persistingL2CacheMaxSize: Annotated[Annotated[int, ctypes.c_int32], 620]
  maxThreadsPerMultiProcessor: Annotated[Annotated[int, ctypes.c_int32], 624]
  streamPrioritiesSupported: Annotated[Annotated[int, ctypes.c_int32], 628]
  globalL1CacheSupported: Annotated[Annotated[int, ctypes.c_int32], 632]
  localL1CacheSupported: Annotated[Annotated[int, ctypes.c_int32], 636]
  sharedMemPerMultiprocessor: Annotated[size_t, 640]
  regsPerMultiprocessor: Annotated[Annotated[int, ctypes.c_int32], 648]
  managedMemory: Annotated[Annotated[int, ctypes.c_int32], 652]
  isMultiGpuBoard: Annotated[Annotated[int, ctypes.c_int32], 656]
  multiGpuBoardGroupID: Annotated[Annotated[int, ctypes.c_int32], 660]
  hostNativeAtomicSupported: Annotated[Annotated[int, ctypes.c_int32], 664]
  singleToDoublePrecisionPerfRatio: Annotated[Annotated[int, ctypes.c_int32], 668]
  pageableMemoryAccess: Annotated[Annotated[int, ctypes.c_int32], 672]
  concurrentManagedAccess: Annotated[Annotated[int, ctypes.c_int32], 676]
  computePreemptionSupported: Annotated[Annotated[int, ctypes.c_int32], 680]
  canUseHostPointerForRegisteredMem: Annotated[Annotated[int, ctypes.c_int32], 684]
  cooperativeLaunch: Annotated[Annotated[int, ctypes.c_int32], 688]
  cooperativeMultiDeviceLaunch: Annotated[Annotated[int, ctypes.c_int32], 692]
  sharedMemPerBlockOptin: Annotated[size_t, 696]
  pageableMemoryAccessUsesHostPageTables: Annotated[Annotated[int, ctypes.c_int32], 704]
  directManagedMemAccessFromHost: Annotated[Annotated[int, ctypes.c_int32], 708]
  maxBlocksPerMultiProcessor: Annotated[Annotated[int, ctypes.c_int32], 712]
  accessPolicyMaxWindowSize: Annotated[Annotated[int, ctypes.c_int32], 716]
  reservedSharedMemPerBlock: Annotated[size_t, 720]
  hostRegisterSupported: Annotated[Annotated[int, ctypes.c_int32], 728]
  sparseHipArraySupported: Annotated[Annotated[int, ctypes.c_int32], 732]
  hostRegisterReadOnlySupported: Annotated[Annotated[int, ctypes.c_int32], 736]
  timelineSemaphoreInteropSupported: Annotated[Annotated[int, ctypes.c_int32], 740]
  memoryPoolsSupported: Annotated[Annotated[int, ctypes.c_int32], 744]
  gpuDirectRDMASupported: Annotated[Annotated[int, ctypes.c_int32], 748]
  gpuDirectRDMAFlushWritesOptions: Annotated[Annotated[int, ctypes.c_uint32], 752]
  gpuDirectRDMAWritesOrdering: Annotated[Annotated[int, ctypes.c_int32], 756]
  memoryPoolSupportedHandleTypes: Annotated[Annotated[int, ctypes.c_uint32], 760]
  deferredMappingHipArraySupported: Annotated[Annotated[int, ctypes.c_int32], 764]
  ipcEventSupported: Annotated[Annotated[int, ctypes.c_int32], 768]
  clusterLaunch: Annotated[Annotated[int, ctypes.c_int32], 772]
  unifiedFunctionPointers: Annotated[Annotated[int, ctypes.c_int32], 776]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[63]], 780]
  hipReserved: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[32]], 1032]
  gcnArchName: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[256]], 1160]
  maxSharedMemoryPerMultiProcessor: Annotated[size_t, 1416]
  clockInstructionRate: Annotated[Annotated[int, ctypes.c_int32], 1424]
  arch: Annotated[hipDeviceArch_t, 1428]
  hdpMemFlushCntl: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 1432]
  hdpRegFlushCntl: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 1440]
  cooperativeMultiDeviceUnmatchedFunc: Annotated[Annotated[int, ctypes.c_int32], 1448]
  cooperativeMultiDeviceUnmatchedGridDim: Annotated[Annotated[int, ctypes.c_int32], 1452]
  cooperativeMultiDeviceUnmatchedBlockDim: Annotated[Annotated[int, ctypes.c_int32], 1456]
  cooperativeMultiDeviceUnmatchedSharedMem: Annotated[Annotated[int, ctypes.c_int32], 1460]
  isLargeBar: Annotated[Annotated[int, ctypes.c_int32], 1464]
  asicRevision: Annotated[Annotated[int, ctypes.c_int32], 1468]
class hipMemoryType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemoryTypeUnregistered = hipMemoryType.define('hipMemoryTypeUnregistered', 0)
hipMemoryTypeHost = hipMemoryType.define('hipMemoryTypeHost', 1)
hipMemoryTypeDevice = hipMemoryType.define('hipMemoryTypeDevice', 2)
hipMemoryTypeManaged = hipMemoryType.define('hipMemoryTypeManaged', 3)
hipMemoryTypeArray = hipMemoryType.define('hipMemoryTypeArray', 10)
hipMemoryTypeUnified = hipMemoryType.define('hipMemoryTypeUnified', 11)

@c.record
class hipPointerAttribute_t(c.Struct):
  SIZE = 32
  type: Annotated[hipMemoryType, 0]
  device: Annotated[Annotated[int, ctypes.c_int32], 4]
  devicePointer: Annotated[ctypes.c_void_p, 8]
  hostPointer: Annotated[ctypes.c_void_p, 16]
  isManaged: Annotated[Annotated[int, ctypes.c_int32], 24]
  allocationFlags: Annotated[Annotated[int, ctypes.c_uint32], 28]
class hipDeviceAttribute_t(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipDeviceAttributeCudaCompatibleBegin = hipDeviceAttribute_t.define('hipDeviceAttributeCudaCompatibleBegin', 0)
hipDeviceAttributeEccEnabled = hipDeviceAttribute_t.define('hipDeviceAttributeEccEnabled', 0)
hipDeviceAttributeAccessPolicyMaxWindowSize = hipDeviceAttribute_t.define('hipDeviceAttributeAccessPolicyMaxWindowSize', 1)
hipDeviceAttributeAsyncEngineCount = hipDeviceAttribute_t.define('hipDeviceAttributeAsyncEngineCount', 2)
hipDeviceAttributeCanMapHostMemory = hipDeviceAttribute_t.define('hipDeviceAttributeCanMapHostMemory', 3)
hipDeviceAttributeCanUseHostPointerForRegisteredMem = hipDeviceAttribute_t.define('hipDeviceAttributeCanUseHostPointerForRegisteredMem', 4)
hipDeviceAttributeClockRate = hipDeviceAttribute_t.define('hipDeviceAttributeClockRate', 5)
hipDeviceAttributeComputeMode = hipDeviceAttribute_t.define('hipDeviceAttributeComputeMode', 6)
hipDeviceAttributeComputePreemptionSupported = hipDeviceAttribute_t.define('hipDeviceAttributeComputePreemptionSupported', 7)
hipDeviceAttributeConcurrentKernels = hipDeviceAttribute_t.define('hipDeviceAttributeConcurrentKernels', 8)
hipDeviceAttributeConcurrentManagedAccess = hipDeviceAttribute_t.define('hipDeviceAttributeConcurrentManagedAccess', 9)
hipDeviceAttributeCooperativeLaunch = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeLaunch', 10)
hipDeviceAttributeCooperativeMultiDeviceLaunch = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceLaunch', 11)
hipDeviceAttributeDeviceOverlap = hipDeviceAttribute_t.define('hipDeviceAttributeDeviceOverlap', 12)
hipDeviceAttributeDirectManagedMemAccessFromHost = hipDeviceAttribute_t.define('hipDeviceAttributeDirectManagedMemAccessFromHost', 13)
hipDeviceAttributeGlobalL1CacheSupported = hipDeviceAttribute_t.define('hipDeviceAttributeGlobalL1CacheSupported', 14)
hipDeviceAttributeHostNativeAtomicSupported = hipDeviceAttribute_t.define('hipDeviceAttributeHostNativeAtomicSupported', 15)
hipDeviceAttributeIntegrated = hipDeviceAttribute_t.define('hipDeviceAttributeIntegrated', 16)
hipDeviceAttributeIsMultiGpuBoard = hipDeviceAttribute_t.define('hipDeviceAttributeIsMultiGpuBoard', 17)
hipDeviceAttributeKernelExecTimeout = hipDeviceAttribute_t.define('hipDeviceAttributeKernelExecTimeout', 18)
hipDeviceAttributeL2CacheSize = hipDeviceAttribute_t.define('hipDeviceAttributeL2CacheSize', 19)
hipDeviceAttributeLocalL1CacheSupported = hipDeviceAttribute_t.define('hipDeviceAttributeLocalL1CacheSupported', 20)
hipDeviceAttributeLuid = hipDeviceAttribute_t.define('hipDeviceAttributeLuid', 21)
hipDeviceAttributeLuidDeviceNodeMask = hipDeviceAttribute_t.define('hipDeviceAttributeLuidDeviceNodeMask', 22)
hipDeviceAttributeComputeCapabilityMajor = hipDeviceAttribute_t.define('hipDeviceAttributeComputeCapabilityMajor', 23)
hipDeviceAttributeManagedMemory = hipDeviceAttribute_t.define('hipDeviceAttributeManagedMemory', 24)
hipDeviceAttributeMaxBlocksPerMultiProcessor = hipDeviceAttribute_t.define('hipDeviceAttributeMaxBlocksPerMultiProcessor', 25)
hipDeviceAttributeMaxBlockDimX = hipDeviceAttribute_t.define('hipDeviceAttributeMaxBlockDimX', 26)
hipDeviceAttributeMaxBlockDimY = hipDeviceAttribute_t.define('hipDeviceAttributeMaxBlockDimY', 27)
hipDeviceAttributeMaxBlockDimZ = hipDeviceAttribute_t.define('hipDeviceAttributeMaxBlockDimZ', 28)
hipDeviceAttributeMaxGridDimX = hipDeviceAttribute_t.define('hipDeviceAttributeMaxGridDimX', 29)
hipDeviceAttributeMaxGridDimY = hipDeviceAttribute_t.define('hipDeviceAttributeMaxGridDimY', 30)
hipDeviceAttributeMaxGridDimZ = hipDeviceAttribute_t.define('hipDeviceAttributeMaxGridDimZ', 31)
hipDeviceAttributeMaxSurface1D = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface1D', 32)
hipDeviceAttributeMaxSurface1DLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface1DLayered', 33)
hipDeviceAttributeMaxSurface2D = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface2D', 34)
hipDeviceAttributeMaxSurface2DLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface2DLayered', 35)
hipDeviceAttributeMaxSurface3D = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface3D', 36)
hipDeviceAttributeMaxSurfaceCubemap = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurfaceCubemap', 37)
hipDeviceAttributeMaxSurfaceCubemapLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurfaceCubemapLayered', 38)
hipDeviceAttributeMaxTexture1DWidth = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture1DWidth', 39)
hipDeviceAttributeMaxTexture1DLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture1DLayered', 40)
hipDeviceAttributeMaxTexture1DLinear = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture1DLinear', 41)
hipDeviceAttributeMaxTexture1DMipmap = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture1DMipmap', 42)
hipDeviceAttributeMaxTexture2DWidth = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DWidth', 43)
hipDeviceAttributeMaxTexture2DHeight = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DHeight', 44)
hipDeviceAttributeMaxTexture2DGather = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DGather', 45)
hipDeviceAttributeMaxTexture2DLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DLayered', 46)
hipDeviceAttributeMaxTexture2DLinear = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DLinear', 47)
hipDeviceAttributeMaxTexture2DMipmap = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DMipmap', 48)
hipDeviceAttributeMaxTexture3DWidth = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture3DWidth', 49)
hipDeviceAttributeMaxTexture3DHeight = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture3DHeight', 50)
hipDeviceAttributeMaxTexture3DDepth = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture3DDepth', 51)
hipDeviceAttributeMaxTexture3DAlt = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture3DAlt', 52)
hipDeviceAttributeMaxTextureCubemap = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTextureCubemap', 53)
hipDeviceAttributeMaxTextureCubemapLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTextureCubemapLayered', 54)
hipDeviceAttributeMaxThreadsDim = hipDeviceAttribute_t.define('hipDeviceAttributeMaxThreadsDim', 55)
hipDeviceAttributeMaxThreadsPerBlock = hipDeviceAttribute_t.define('hipDeviceAttributeMaxThreadsPerBlock', 56)
hipDeviceAttributeMaxThreadsPerMultiProcessor = hipDeviceAttribute_t.define('hipDeviceAttributeMaxThreadsPerMultiProcessor', 57)
hipDeviceAttributeMaxPitch = hipDeviceAttribute_t.define('hipDeviceAttributeMaxPitch', 58)
hipDeviceAttributeMemoryBusWidth = hipDeviceAttribute_t.define('hipDeviceAttributeMemoryBusWidth', 59)
hipDeviceAttributeMemoryClockRate = hipDeviceAttribute_t.define('hipDeviceAttributeMemoryClockRate', 60)
hipDeviceAttributeComputeCapabilityMinor = hipDeviceAttribute_t.define('hipDeviceAttributeComputeCapabilityMinor', 61)
hipDeviceAttributeMultiGpuBoardGroupID = hipDeviceAttribute_t.define('hipDeviceAttributeMultiGpuBoardGroupID', 62)
hipDeviceAttributeMultiprocessorCount = hipDeviceAttribute_t.define('hipDeviceAttributeMultiprocessorCount', 63)
hipDeviceAttributeUnused1 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused1', 64)
hipDeviceAttributePageableMemoryAccess = hipDeviceAttribute_t.define('hipDeviceAttributePageableMemoryAccess', 65)
hipDeviceAttributePageableMemoryAccessUsesHostPageTables = hipDeviceAttribute_t.define('hipDeviceAttributePageableMemoryAccessUsesHostPageTables', 66)
hipDeviceAttributePciBusId = hipDeviceAttribute_t.define('hipDeviceAttributePciBusId', 67)
hipDeviceAttributePciDeviceId = hipDeviceAttribute_t.define('hipDeviceAttributePciDeviceId', 68)
hipDeviceAttributePciDomainID = hipDeviceAttribute_t.define('hipDeviceAttributePciDomainID', 69)
hipDeviceAttributePersistingL2CacheMaxSize = hipDeviceAttribute_t.define('hipDeviceAttributePersistingL2CacheMaxSize', 70)
hipDeviceAttributeMaxRegistersPerBlock = hipDeviceAttribute_t.define('hipDeviceAttributeMaxRegistersPerBlock', 71)
hipDeviceAttributeMaxRegistersPerMultiprocessor = hipDeviceAttribute_t.define('hipDeviceAttributeMaxRegistersPerMultiprocessor', 72)
hipDeviceAttributeReservedSharedMemPerBlock = hipDeviceAttribute_t.define('hipDeviceAttributeReservedSharedMemPerBlock', 73)
hipDeviceAttributeMaxSharedMemoryPerBlock = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSharedMemoryPerBlock', 74)
hipDeviceAttributeSharedMemPerBlockOptin = hipDeviceAttribute_t.define('hipDeviceAttributeSharedMemPerBlockOptin', 75)
hipDeviceAttributeSharedMemPerMultiprocessor = hipDeviceAttribute_t.define('hipDeviceAttributeSharedMemPerMultiprocessor', 76)
hipDeviceAttributeSingleToDoublePrecisionPerfRatio = hipDeviceAttribute_t.define('hipDeviceAttributeSingleToDoublePrecisionPerfRatio', 77)
hipDeviceAttributeStreamPrioritiesSupported = hipDeviceAttribute_t.define('hipDeviceAttributeStreamPrioritiesSupported', 78)
hipDeviceAttributeSurfaceAlignment = hipDeviceAttribute_t.define('hipDeviceAttributeSurfaceAlignment', 79)
hipDeviceAttributeTccDriver = hipDeviceAttribute_t.define('hipDeviceAttributeTccDriver', 80)
hipDeviceAttributeTextureAlignment = hipDeviceAttribute_t.define('hipDeviceAttributeTextureAlignment', 81)
hipDeviceAttributeTexturePitchAlignment = hipDeviceAttribute_t.define('hipDeviceAttributeTexturePitchAlignment', 82)
hipDeviceAttributeTotalConstantMemory = hipDeviceAttribute_t.define('hipDeviceAttributeTotalConstantMemory', 83)
hipDeviceAttributeTotalGlobalMem = hipDeviceAttribute_t.define('hipDeviceAttributeTotalGlobalMem', 84)
hipDeviceAttributeUnifiedAddressing = hipDeviceAttribute_t.define('hipDeviceAttributeUnifiedAddressing', 85)
hipDeviceAttributeUnused2 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused2', 86)
hipDeviceAttributeWarpSize = hipDeviceAttribute_t.define('hipDeviceAttributeWarpSize', 87)
hipDeviceAttributeMemoryPoolsSupported = hipDeviceAttribute_t.define('hipDeviceAttributeMemoryPoolsSupported', 88)
hipDeviceAttributeVirtualMemoryManagementSupported = hipDeviceAttribute_t.define('hipDeviceAttributeVirtualMemoryManagementSupported', 89)
hipDeviceAttributeHostRegisterSupported = hipDeviceAttribute_t.define('hipDeviceAttributeHostRegisterSupported', 90)
hipDeviceAttributeMemoryPoolSupportedHandleTypes = hipDeviceAttribute_t.define('hipDeviceAttributeMemoryPoolSupportedHandleTypes', 91)
hipDeviceAttributeCudaCompatibleEnd = hipDeviceAttribute_t.define('hipDeviceAttributeCudaCompatibleEnd', 9999)
hipDeviceAttributeAmdSpecificBegin = hipDeviceAttribute_t.define('hipDeviceAttributeAmdSpecificBegin', 10000)
hipDeviceAttributeClockInstructionRate = hipDeviceAttribute_t.define('hipDeviceAttributeClockInstructionRate', 10000)
hipDeviceAttributeUnused3 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused3', 10001)
hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSharedMemoryPerMultiprocessor', 10002)
hipDeviceAttributeUnused4 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused4', 10003)
hipDeviceAttributeUnused5 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused5', 10004)
hipDeviceAttributeHdpMemFlushCntl = hipDeviceAttribute_t.define('hipDeviceAttributeHdpMemFlushCntl', 10005)
hipDeviceAttributeHdpRegFlushCntl = hipDeviceAttribute_t.define('hipDeviceAttributeHdpRegFlushCntl', 10006)
hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc', 10007)
hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim', 10008)
hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim', 10009)
hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem', 10010)
hipDeviceAttributeIsLargeBar = hipDeviceAttribute_t.define('hipDeviceAttributeIsLargeBar', 10011)
hipDeviceAttributeAsicRevision = hipDeviceAttribute_t.define('hipDeviceAttributeAsicRevision', 10012)
hipDeviceAttributeCanUseStreamWaitValue = hipDeviceAttribute_t.define('hipDeviceAttributeCanUseStreamWaitValue', 10013)
hipDeviceAttributeImageSupport = hipDeviceAttribute_t.define('hipDeviceAttributeImageSupport', 10014)
hipDeviceAttributePhysicalMultiProcessorCount = hipDeviceAttribute_t.define('hipDeviceAttributePhysicalMultiProcessorCount', 10015)
hipDeviceAttributeFineGrainSupport = hipDeviceAttribute_t.define('hipDeviceAttributeFineGrainSupport', 10016)
hipDeviceAttributeWallClockRate = hipDeviceAttribute_t.define('hipDeviceAttributeWallClockRate', 10017)
hipDeviceAttributeAmdSpecificEnd = hipDeviceAttribute_t.define('hipDeviceAttributeAmdSpecificEnd', 19999)
hipDeviceAttributeVendorSpecificBegin = hipDeviceAttribute_t.define('hipDeviceAttributeVendorSpecificBegin', 20000)

class hipDriverProcAddressQueryResult(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_GET_PROC_ADDRESS_SUCCESS = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_SUCCESS', 0)
HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND', 1)
HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT', 2)

class hipComputeMode(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipComputeModeDefault = hipComputeMode.define('hipComputeModeDefault', 0)
hipComputeModeExclusive = hipComputeMode.define('hipComputeModeExclusive', 1)
hipComputeModeProhibited = hipComputeMode.define('hipComputeModeProhibited', 2)
hipComputeModeExclusiveProcess = hipComputeMode.define('hipComputeModeExclusiveProcess', 3)

class hipFlushGPUDirectRDMAWritesOptions(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipFlushGPUDirectRDMAWritesOptionHost = hipFlushGPUDirectRDMAWritesOptions.define('hipFlushGPUDirectRDMAWritesOptionHost', 1)
hipFlushGPUDirectRDMAWritesOptionMemOps = hipFlushGPUDirectRDMAWritesOptions.define('hipFlushGPUDirectRDMAWritesOptionMemOps', 2)

class hipGPUDirectRDMAWritesOrdering(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGPUDirectRDMAWritesOrderingNone = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingNone', 0)
hipGPUDirectRDMAWritesOrderingOwner = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingOwner', 100)
hipGPUDirectRDMAWritesOrderingAllDevices = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingAllDevices', 200)

@dll.bind
def hip_init() -> hipError_t: ...
class ihipCtx_t(ctypes.Structure): pass
hipCtx_t: TypeAlias = c.POINTER[ihipCtx_t]
hipDevice_t: TypeAlias = Annotated[int, ctypes.c_int32]
class hipDeviceP2PAttr(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipDevP2PAttrPerformanceRank = hipDeviceP2PAttr.define('hipDevP2PAttrPerformanceRank', 0)
hipDevP2PAttrAccessSupported = hipDeviceP2PAttr.define('hipDevP2PAttrAccessSupported', 1)
hipDevP2PAttrNativeAtomicSupported = hipDeviceP2PAttr.define('hipDevP2PAttrNativeAtomicSupported', 2)
hipDevP2PAttrHipArrayAccessSupported = hipDeviceP2PAttr.define('hipDevP2PAttrHipArrayAccessSupported', 3)

@c.record
class hipIpcMemHandle_st(c.Struct):
  SIZE = 64
  reserved: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[64]], 0]
hipIpcMemHandle_t: TypeAlias = hipIpcMemHandle_st
@c.record
class hipIpcEventHandle_st(c.Struct):
  SIZE = 64
  reserved: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[64]], 0]
hipIpcEventHandle_t: TypeAlias = hipIpcEventHandle_st
class ihipModule_t(ctypes.Structure): pass
hipModule_t: TypeAlias = c.POINTER[ihipModule_t]
class ihipMemPoolHandle_t(ctypes.Structure): pass
hipMemPool_t: TypeAlias = c.POINTER[ihipMemPoolHandle_t]
@c.record
class hipFuncAttributes(c.Struct):
  SIZE = 56
  binaryVersion: Annotated[Annotated[int, ctypes.c_int32], 0]
  cacheModeCA: Annotated[Annotated[int, ctypes.c_int32], 4]
  constSizeBytes: Annotated[size_t, 8]
  localSizeBytes: Annotated[size_t, 16]
  maxDynamicSharedSizeBytes: Annotated[Annotated[int, ctypes.c_int32], 24]
  maxThreadsPerBlock: Annotated[Annotated[int, ctypes.c_int32], 28]
  numRegs: Annotated[Annotated[int, ctypes.c_int32], 32]
  preferredShmemCarveout: Annotated[Annotated[int, ctypes.c_int32], 36]
  ptxVersion: Annotated[Annotated[int, ctypes.c_int32], 40]
  sharedSizeBytes: Annotated[size_t, 48]
class hipLimit_t(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipLimitStackSize = hipLimit_t.define('hipLimitStackSize', 0)
hipLimitPrintfFifoSize = hipLimit_t.define('hipLimitPrintfFifoSize', 1)
hipLimitMallocHeapSize = hipLimit_t.define('hipLimitMallocHeapSize', 2)
hipLimitRange = hipLimit_t.define('hipLimitRange', 3)

class hipMemoryAdvise(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemAdviseSetReadMostly = hipMemoryAdvise.define('hipMemAdviseSetReadMostly', 1)
hipMemAdviseUnsetReadMostly = hipMemoryAdvise.define('hipMemAdviseUnsetReadMostly', 2)
hipMemAdviseSetPreferredLocation = hipMemoryAdvise.define('hipMemAdviseSetPreferredLocation', 3)
hipMemAdviseUnsetPreferredLocation = hipMemoryAdvise.define('hipMemAdviseUnsetPreferredLocation', 4)
hipMemAdviseSetAccessedBy = hipMemoryAdvise.define('hipMemAdviseSetAccessedBy', 5)
hipMemAdviseUnsetAccessedBy = hipMemoryAdvise.define('hipMemAdviseUnsetAccessedBy', 6)
hipMemAdviseSetCoarseGrain = hipMemoryAdvise.define('hipMemAdviseSetCoarseGrain', 100)
hipMemAdviseUnsetCoarseGrain = hipMemoryAdvise.define('hipMemAdviseUnsetCoarseGrain', 101)

class hipMemRangeCoherencyMode(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemRangeCoherencyModeFineGrain = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeFineGrain', 0)
hipMemRangeCoherencyModeCoarseGrain = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeCoarseGrain', 1)
hipMemRangeCoherencyModeIndeterminate = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeIndeterminate', 2)

class hipMemRangeAttribute(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemRangeAttributeReadMostly = hipMemRangeAttribute.define('hipMemRangeAttributeReadMostly', 1)
hipMemRangeAttributePreferredLocation = hipMemRangeAttribute.define('hipMemRangeAttributePreferredLocation', 2)
hipMemRangeAttributeAccessedBy = hipMemRangeAttribute.define('hipMemRangeAttributeAccessedBy', 3)
hipMemRangeAttributeLastPrefetchLocation = hipMemRangeAttribute.define('hipMemRangeAttributeLastPrefetchLocation', 4)
hipMemRangeAttributeCoherencyMode = hipMemRangeAttribute.define('hipMemRangeAttributeCoherencyMode', 100)

class hipMemPoolAttr(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemPoolReuseFollowEventDependencies = hipMemPoolAttr.define('hipMemPoolReuseFollowEventDependencies', 1)
hipMemPoolReuseAllowOpportunistic = hipMemPoolAttr.define('hipMemPoolReuseAllowOpportunistic', 2)
hipMemPoolReuseAllowInternalDependencies = hipMemPoolAttr.define('hipMemPoolReuseAllowInternalDependencies', 3)
hipMemPoolAttrReleaseThreshold = hipMemPoolAttr.define('hipMemPoolAttrReleaseThreshold', 4)
hipMemPoolAttrReservedMemCurrent = hipMemPoolAttr.define('hipMemPoolAttrReservedMemCurrent', 5)
hipMemPoolAttrReservedMemHigh = hipMemPoolAttr.define('hipMemPoolAttrReservedMemHigh', 6)
hipMemPoolAttrUsedMemCurrent = hipMemPoolAttr.define('hipMemPoolAttrUsedMemCurrent', 7)
hipMemPoolAttrUsedMemHigh = hipMemPoolAttr.define('hipMemPoolAttrUsedMemHigh', 8)

class hipMemLocationType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemLocationTypeInvalid = hipMemLocationType.define('hipMemLocationTypeInvalid', 0)
hipMemLocationTypeDevice = hipMemLocationType.define('hipMemLocationTypeDevice', 1)

@c.record
class hipMemLocation(c.Struct):
  SIZE = 8
  type: Annotated[hipMemLocationType, 0]
  id: Annotated[Annotated[int, ctypes.c_int32], 4]
class hipMemAccessFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemAccessFlagsProtNone = hipMemAccessFlags.define('hipMemAccessFlagsProtNone', 0)
hipMemAccessFlagsProtRead = hipMemAccessFlags.define('hipMemAccessFlagsProtRead', 1)
hipMemAccessFlagsProtReadWrite = hipMemAccessFlags.define('hipMemAccessFlagsProtReadWrite', 3)

@c.record
class hipMemAccessDesc(c.Struct):
  SIZE = 12
  location: Annotated[hipMemLocation, 0]
  flags: Annotated[hipMemAccessFlags, 8]
class hipMemAllocationType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemAllocationTypeInvalid = hipMemAllocationType.define('hipMemAllocationTypeInvalid', 0)
hipMemAllocationTypePinned = hipMemAllocationType.define('hipMemAllocationTypePinned', 1)
hipMemAllocationTypeMax = hipMemAllocationType.define('hipMemAllocationTypeMax', 2147483647)

class hipMemAllocationHandleType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemHandleTypeNone = hipMemAllocationHandleType.define('hipMemHandleTypeNone', 0)
hipMemHandleTypePosixFileDescriptor = hipMemAllocationHandleType.define('hipMemHandleTypePosixFileDescriptor', 1)
hipMemHandleTypeWin32 = hipMemAllocationHandleType.define('hipMemHandleTypeWin32', 2)
hipMemHandleTypeWin32Kmt = hipMemAllocationHandleType.define('hipMemHandleTypeWin32Kmt', 4)

@c.record
class hipMemPoolProps(c.Struct):
  SIZE = 88
  allocType: Annotated[hipMemAllocationType, 0]
  handleTypes: Annotated[hipMemAllocationHandleType, 4]
  location: Annotated[hipMemLocation, 8]
  win32SecurityAttributes: Annotated[ctypes.c_void_p, 16]
  maxSize: Annotated[size_t, 24]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[56]], 32]
@c.record
class hipMemPoolPtrExportData(c.Struct):
  SIZE = 64
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
class hipJitOption(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipJitOptionMaxRegisters = hipJitOption.define('hipJitOptionMaxRegisters', 0)
hipJitOptionThreadsPerBlock = hipJitOption.define('hipJitOptionThreadsPerBlock', 1)
hipJitOptionWallTime = hipJitOption.define('hipJitOptionWallTime', 2)
hipJitOptionInfoLogBuffer = hipJitOption.define('hipJitOptionInfoLogBuffer', 3)
hipJitOptionInfoLogBufferSizeBytes = hipJitOption.define('hipJitOptionInfoLogBufferSizeBytes', 4)
hipJitOptionErrorLogBuffer = hipJitOption.define('hipJitOptionErrorLogBuffer', 5)
hipJitOptionErrorLogBufferSizeBytes = hipJitOption.define('hipJitOptionErrorLogBufferSizeBytes', 6)
hipJitOptionOptimizationLevel = hipJitOption.define('hipJitOptionOptimizationLevel', 7)
hipJitOptionTargetFromContext = hipJitOption.define('hipJitOptionTargetFromContext', 8)
hipJitOptionTarget = hipJitOption.define('hipJitOptionTarget', 9)
hipJitOptionFallbackStrategy = hipJitOption.define('hipJitOptionFallbackStrategy', 10)
hipJitOptionGenerateDebugInfo = hipJitOption.define('hipJitOptionGenerateDebugInfo', 11)
hipJitOptionLogVerbose = hipJitOption.define('hipJitOptionLogVerbose', 12)
hipJitOptionGenerateLineInfo = hipJitOption.define('hipJitOptionGenerateLineInfo', 13)
hipJitOptionCacheMode = hipJitOption.define('hipJitOptionCacheMode', 14)
hipJitOptionSm3xOpt = hipJitOption.define('hipJitOptionSm3xOpt', 15)
hipJitOptionFastCompile = hipJitOption.define('hipJitOptionFastCompile', 16)
hipJitOptionNumOptions = hipJitOption.define('hipJitOptionNumOptions', 17)

class hipFuncAttribute(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipFuncAttributeMaxDynamicSharedMemorySize = hipFuncAttribute.define('hipFuncAttributeMaxDynamicSharedMemorySize', 8)
hipFuncAttributePreferredSharedMemoryCarveout = hipFuncAttribute.define('hipFuncAttributePreferredSharedMemoryCarveout', 9)
hipFuncAttributeMax = hipFuncAttribute.define('hipFuncAttributeMax', 10)

class hipFuncCache_t(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipFuncCachePreferNone = hipFuncCache_t.define('hipFuncCachePreferNone', 0)
hipFuncCachePreferShared = hipFuncCache_t.define('hipFuncCachePreferShared', 1)
hipFuncCachePreferL1 = hipFuncCache_t.define('hipFuncCachePreferL1', 2)
hipFuncCachePreferEqual = hipFuncCache_t.define('hipFuncCachePreferEqual', 3)

class hipSharedMemConfig(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipSharedMemBankSizeDefault = hipSharedMemConfig.define('hipSharedMemBankSizeDefault', 0)
hipSharedMemBankSizeFourByte = hipSharedMemConfig.define('hipSharedMemBankSizeFourByte', 1)
hipSharedMemBankSizeEightByte = hipSharedMemConfig.define('hipSharedMemBankSizeEightByte', 2)

@c.record
class hipLaunchParams_t(c.Struct):
  SIZE = 56
  func: Annotated[ctypes.c_void_p, 0]
  gridDim: Annotated[dim3, 8]
  blockDim: Annotated[dim3, 20]
  args: Annotated[c.POINTER[ctypes.c_void_p], 32]
  sharedMem: Annotated[size_t, 40]
  stream: Annotated[hipStream_t, 48]
hipLaunchParams: TypeAlias = hipLaunchParams_t
@c.record
class hipFunctionLaunchParams_t(c.Struct):
  SIZE = 56
  function: Annotated[hipFunction_t, 0]
  gridDimX: Annotated[Annotated[int, ctypes.c_uint32], 8]
  gridDimY: Annotated[Annotated[int, ctypes.c_uint32], 12]
  gridDimZ: Annotated[Annotated[int, ctypes.c_uint32], 16]
  blockDimX: Annotated[Annotated[int, ctypes.c_uint32], 20]
  blockDimY: Annotated[Annotated[int, ctypes.c_uint32], 24]
  blockDimZ: Annotated[Annotated[int, ctypes.c_uint32], 28]
  sharedMemBytes: Annotated[Annotated[int, ctypes.c_uint32], 32]
  hStream: Annotated[hipStream_t, 40]
  kernelParams: Annotated[c.POINTER[ctypes.c_void_p], 48]
hipFunctionLaunchParams: TypeAlias = hipFunctionLaunchParams_t
class hipExternalMemoryHandleType_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipExternalMemoryHandleTypeOpaqueFd = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueFd', 1)
hipExternalMemoryHandleTypeOpaqueWin32 = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueWin32', 2)
hipExternalMemoryHandleTypeOpaqueWin32Kmt = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueWin32Kmt', 3)
hipExternalMemoryHandleTypeD3D12Heap = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D12Heap', 4)
hipExternalMemoryHandleTypeD3D12Resource = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D12Resource', 5)
hipExternalMemoryHandleTypeD3D11Resource = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D11Resource', 6)
hipExternalMemoryHandleTypeD3D11ResourceKmt = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D11ResourceKmt', 7)
hipExternalMemoryHandleTypeNvSciBuf = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeNvSciBuf', 8)

hipExternalMemoryHandleType: TypeAlias = hipExternalMemoryHandleType_enum
@c.record
class hipExternalMemoryHandleDesc_st(c.Struct):
  SIZE = 104
  type: Annotated[hipExternalMemoryHandleType, 0]
  handle: Annotated[hipExternalMemoryHandleDesc_st_handle, 8]
  size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[16]], 36]
@c.record
class hipExternalMemoryHandleDesc_st_handle(c.Struct):
  SIZE = 16
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  win32: Annotated[hipExternalMemoryHandleDesc_st_handle_win32, 0]
  nvSciBufObject: Annotated[ctypes.c_void_p, 0]
@c.record
class hipExternalMemoryHandleDesc_st_handle_win32(c.Struct):
  SIZE = 16
  handle: Annotated[ctypes.c_void_p, 0]
  name: Annotated[ctypes.c_void_p, 8]
hipExternalMemoryHandleDesc: TypeAlias = hipExternalMemoryHandleDesc_st
@c.record
class hipExternalMemoryBufferDesc_st(c.Struct):
  SIZE = 88
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[16]], 20]
hipExternalMemoryBufferDesc: TypeAlias = hipExternalMemoryBufferDesc_st
@c.record
class hipExternalMemoryMipmappedArrayDesc_st(c.Struct):
  SIZE = 64
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  formatDesc: Annotated[hipChannelFormatDesc, 8]
  extent: Annotated[hipExtent, 32]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 56]
  numLevels: Annotated[Annotated[int, ctypes.c_uint32], 60]
@c.record
class hipChannelFormatDesc(c.Struct):
  SIZE = 20
  x: Annotated[Annotated[int, ctypes.c_int32], 0]
  y: Annotated[Annotated[int, ctypes.c_int32], 4]
  z: Annotated[Annotated[int, ctypes.c_int32], 8]
  w: Annotated[Annotated[int, ctypes.c_int32], 12]
  f: Annotated[hipChannelFormatKind, 16]
class hipChannelFormatKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipChannelFormatKindSigned = hipChannelFormatKind.define('hipChannelFormatKindSigned', 0)
hipChannelFormatKindUnsigned = hipChannelFormatKind.define('hipChannelFormatKindUnsigned', 1)
hipChannelFormatKindFloat = hipChannelFormatKind.define('hipChannelFormatKindFloat', 2)
hipChannelFormatKindNone = hipChannelFormatKind.define('hipChannelFormatKindNone', 3)

@c.record
class hipExtent(c.Struct):
  SIZE = 24
  width: Annotated[size_t, 0]
  height: Annotated[size_t, 8]
  depth: Annotated[size_t, 16]
hipExternalMemoryMipmappedArrayDesc: TypeAlias = hipExternalMemoryMipmappedArrayDesc_st
hipExternalMemory_t: TypeAlias = ctypes.c_void_p
class hipExternalSemaphoreHandleType_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipExternalSemaphoreHandleTypeOpaqueFd = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeOpaqueFd', 1)
hipExternalSemaphoreHandleTypeOpaqueWin32 = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeOpaqueWin32', 2)
hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeOpaqueWin32Kmt', 3)
hipExternalSemaphoreHandleTypeD3D12Fence = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeD3D12Fence', 4)
hipExternalSemaphoreHandleTypeD3D11Fence = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeD3D11Fence', 5)
hipExternalSemaphoreHandleTypeNvSciSync = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeNvSciSync', 6)
hipExternalSemaphoreHandleTypeKeyedMutex = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeKeyedMutex', 7)
hipExternalSemaphoreHandleTypeKeyedMutexKmt = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeKeyedMutexKmt', 8)
hipExternalSemaphoreHandleTypeTimelineSemaphoreFd = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeTimelineSemaphoreFd', 9)
hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32', 10)

hipExternalSemaphoreHandleType: TypeAlias = hipExternalSemaphoreHandleType_enum
@c.record
class hipExternalSemaphoreHandleDesc_st(c.Struct):
  SIZE = 96
  type: Annotated[hipExternalSemaphoreHandleType, 0]
  handle: Annotated[hipExternalSemaphoreHandleDesc_st_handle, 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[16]], 28]
@c.record
class hipExternalSemaphoreHandleDesc_st_handle(c.Struct):
  SIZE = 16
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  win32: Annotated[hipExternalSemaphoreHandleDesc_st_handle_win32, 0]
  NvSciSyncObj: Annotated[ctypes.c_void_p, 0]
@c.record
class hipExternalSemaphoreHandleDesc_st_handle_win32(c.Struct):
  SIZE = 16
  handle: Annotated[ctypes.c_void_p, 0]
  name: Annotated[ctypes.c_void_p, 8]
hipExternalSemaphoreHandleDesc: TypeAlias = hipExternalSemaphoreHandleDesc_st
hipExternalSemaphore_t: TypeAlias = ctypes.c_void_p
@c.record
class hipExternalSemaphoreSignalParams_st(c.Struct):
  SIZE = 144
  params: Annotated[hipExternalSemaphoreSignalParams_st_params, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 72]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[16]], 76]
@c.record
class hipExternalSemaphoreSignalParams_st_params(c.Struct):
  SIZE = 72
  fence: Annotated[hipExternalSemaphoreSignalParams_st_params_fence, 0]
  nvSciSync: Annotated[hipExternalSemaphoreSignalParams_st_params_nvSciSync, 8]
  keyedMutex: Annotated[hipExternalSemaphoreSignalParams_st_params_keyedMutex, 16]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[12]], 24]
@c.record
class hipExternalSemaphoreSignalParams_st_params_fence(c.Struct):
  SIZE = 8
  value: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class hipExternalSemaphoreSignalParams_st_params_nvSciSync(c.Struct):
  SIZE = 8
  fence: Annotated[ctypes.c_void_p, 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class hipExternalSemaphoreSignalParams_st_params_keyedMutex(c.Struct):
  SIZE = 8
  key: Annotated[Annotated[int, ctypes.c_uint64], 0]
hipExternalSemaphoreSignalParams: TypeAlias = hipExternalSemaphoreSignalParams_st
@c.record
class hipExternalSemaphoreWaitParams_st(c.Struct):
  SIZE = 144
  params: Annotated[hipExternalSemaphoreWaitParams_st_params, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 72]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[16]], 76]
@c.record
class hipExternalSemaphoreWaitParams_st_params(c.Struct):
  SIZE = 72
  fence: Annotated[hipExternalSemaphoreWaitParams_st_params_fence, 0]
  nvSciSync: Annotated[hipExternalSemaphoreWaitParams_st_params_nvSciSync, 8]
  keyedMutex: Annotated[hipExternalSemaphoreWaitParams_st_params_keyedMutex, 16]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[10]], 32]
@c.record
class hipExternalSemaphoreWaitParams_st_params_fence(c.Struct):
  SIZE = 8
  value: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class hipExternalSemaphoreWaitParams_st_params_nvSciSync(c.Struct):
  SIZE = 8
  fence: Annotated[ctypes.c_void_p, 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class hipExternalSemaphoreWaitParams_st_params_keyedMutex(c.Struct):
  SIZE = 16
  key: Annotated[Annotated[int, ctypes.c_uint64], 0]
  timeoutMs: Annotated[Annotated[int, ctypes.c_uint32], 8]
hipExternalSemaphoreWaitParams: TypeAlias = hipExternalSemaphoreWaitParams_st
@dll.bind
def __hipGetPCH(pch:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], size:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
class hipGraphicsRegisterFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphicsRegisterFlagsNone = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsNone', 0)
hipGraphicsRegisterFlagsReadOnly = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsReadOnly', 1)
hipGraphicsRegisterFlagsWriteDiscard = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsWriteDiscard', 2)
hipGraphicsRegisterFlagsSurfaceLoadStore = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsSurfaceLoadStore', 4)
hipGraphicsRegisterFlagsTextureGather = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsTextureGather', 8)

class _hipGraphicsResource(ctypes.Structure): pass
hipGraphicsResource: TypeAlias = _hipGraphicsResource
hipGraphicsResource_t: TypeAlias = c.POINTER[_hipGraphicsResource]
class ihipGraph(ctypes.Structure): pass
hipGraph_t: TypeAlias = c.POINTER[ihipGraph]
class hipGraphNode(ctypes.Structure): pass
hipGraphNode_t: TypeAlias = c.POINTER[hipGraphNode]
class hipGraphExec(ctypes.Structure): pass
hipGraphExec_t: TypeAlias = c.POINTER[hipGraphExec]
class hipUserObject(ctypes.Structure): pass
hipUserObject_t: TypeAlias = c.POINTER[hipUserObject]
class hipGraphNodeType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphNodeTypeKernel = hipGraphNodeType.define('hipGraphNodeTypeKernel', 0)
hipGraphNodeTypeMemcpy = hipGraphNodeType.define('hipGraphNodeTypeMemcpy', 1)
hipGraphNodeTypeMemset = hipGraphNodeType.define('hipGraphNodeTypeMemset', 2)
hipGraphNodeTypeHost = hipGraphNodeType.define('hipGraphNodeTypeHost', 3)
hipGraphNodeTypeGraph = hipGraphNodeType.define('hipGraphNodeTypeGraph', 4)
hipGraphNodeTypeEmpty = hipGraphNodeType.define('hipGraphNodeTypeEmpty', 5)
hipGraphNodeTypeWaitEvent = hipGraphNodeType.define('hipGraphNodeTypeWaitEvent', 6)
hipGraphNodeTypeEventRecord = hipGraphNodeType.define('hipGraphNodeTypeEventRecord', 7)
hipGraphNodeTypeExtSemaphoreSignal = hipGraphNodeType.define('hipGraphNodeTypeExtSemaphoreSignal', 8)
hipGraphNodeTypeExtSemaphoreWait = hipGraphNodeType.define('hipGraphNodeTypeExtSemaphoreWait', 9)
hipGraphNodeTypeMemAlloc = hipGraphNodeType.define('hipGraphNodeTypeMemAlloc', 10)
hipGraphNodeTypeMemFree = hipGraphNodeType.define('hipGraphNodeTypeMemFree', 11)
hipGraphNodeTypeMemcpyFromSymbol = hipGraphNodeType.define('hipGraphNodeTypeMemcpyFromSymbol', 12)
hipGraphNodeTypeMemcpyToSymbol = hipGraphNodeType.define('hipGraphNodeTypeMemcpyToSymbol', 13)
hipGraphNodeTypeCount = hipGraphNodeType.define('hipGraphNodeTypeCount', 14)

hipHostFn_t: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_void_p]]
@c.record
class hipHostNodeParams(c.Struct):
  SIZE = 16
  fn: Annotated[hipHostFn_t, 0]
  userData: Annotated[ctypes.c_void_p, 8]
@c.record
class hipKernelNodeParams(c.Struct):
  SIZE = 64
  blockDim: Annotated[dim3, 0]
  extra: Annotated[c.POINTER[ctypes.c_void_p], 16]
  func: Annotated[ctypes.c_void_p, 24]
  gridDim: Annotated[dim3, 32]
  kernelParams: Annotated[c.POINTER[ctypes.c_void_p], 48]
  sharedMemBytes: Annotated[Annotated[int, ctypes.c_uint32], 56]
@c.record
class hipMemsetParams(c.Struct):
  SIZE = 48
  dst: Annotated[ctypes.c_void_p, 0]
  elementSize: Annotated[Annotated[int, ctypes.c_uint32], 8]
  height: Annotated[size_t, 16]
  pitch: Annotated[size_t, 24]
  value: Annotated[Annotated[int, ctypes.c_uint32], 32]
  width: Annotated[size_t, 40]
@c.record
class hipMemAllocNodeParams(c.Struct):
  SIZE = 120
  poolProps: Annotated[hipMemPoolProps, 0]
  accessDescs: Annotated[c.POINTER[hipMemAccessDesc], 88]
  accessDescCount: Annotated[size_t, 96]
  bytesize: Annotated[size_t, 104]
  dptr: Annotated[ctypes.c_void_p, 112]
class hipAccessProperty(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipAccessPropertyNormal = hipAccessProperty.define('hipAccessPropertyNormal', 0)
hipAccessPropertyStreaming = hipAccessProperty.define('hipAccessPropertyStreaming', 1)
hipAccessPropertyPersisting = hipAccessProperty.define('hipAccessPropertyPersisting', 2)

@c.record
class hipAccessPolicyWindow(c.Struct):
  SIZE = 32
  base_ptr: Annotated[ctypes.c_void_p, 0]
  hitProp: Annotated[hipAccessProperty, 8]
  hitRatio: Annotated[Annotated[float, ctypes.c_float], 12]
  missProp: Annotated[hipAccessProperty, 16]
  num_bytes: Annotated[size_t, 24]
class hipLaunchAttributeID(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipLaunchAttributeAccessPolicyWindow = hipLaunchAttributeID.define('hipLaunchAttributeAccessPolicyWindow', 1)
hipLaunchAttributeCooperative = hipLaunchAttributeID.define('hipLaunchAttributeCooperative', 2)
hipLaunchAttributePriority = hipLaunchAttributeID.define('hipLaunchAttributePriority', 8)

@c.record
class hipLaunchAttributeValue(c.Struct):
  SIZE = 32
  accessPolicyWindow: Annotated[hipAccessPolicyWindow, 0]
  cooperative: Annotated[Annotated[int, ctypes.c_int32], 0]
  priority: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class HIP_MEMSET_NODE_PARAMS(c.Struct):
  SIZE = 40
  dst: Annotated[hipDeviceptr_t, 0]
  pitch: Annotated[size_t, 8]
  value: Annotated[Annotated[int, ctypes.c_uint32], 16]
  elementSize: Annotated[Annotated[int, ctypes.c_uint32], 20]
  width: Annotated[size_t, 24]
  height: Annotated[size_t, 32]
hipDeviceptr_t: TypeAlias = ctypes.c_void_p
class hipGraphExecUpdateResult(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphExecUpdateSuccess = hipGraphExecUpdateResult.define('hipGraphExecUpdateSuccess', 0)
hipGraphExecUpdateError = hipGraphExecUpdateResult.define('hipGraphExecUpdateError', 1)
hipGraphExecUpdateErrorTopologyChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorTopologyChanged', 2)
hipGraphExecUpdateErrorNodeTypeChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorNodeTypeChanged', 3)
hipGraphExecUpdateErrorFunctionChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorFunctionChanged', 4)
hipGraphExecUpdateErrorParametersChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorParametersChanged', 5)
hipGraphExecUpdateErrorNotSupported = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorNotSupported', 6)
hipGraphExecUpdateErrorUnsupportedFunctionChange = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorUnsupportedFunctionChange', 7)

class hipStreamCaptureMode(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipStreamCaptureModeGlobal = hipStreamCaptureMode.define('hipStreamCaptureModeGlobal', 0)
hipStreamCaptureModeThreadLocal = hipStreamCaptureMode.define('hipStreamCaptureModeThreadLocal', 1)
hipStreamCaptureModeRelaxed = hipStreamCaptureMode.define('hipStreamCaptureModeRelaxed', 2)

class hipStreamCaptureStatus(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipStreamCaptureStatusNone = hipStreamCaptureStatus.define('hipStreamCaptureStatusNone', 0)
hipStreamCaptureStatusActive = hipStreamCaptureStatus.define('hipStreamCaptureStatusActive', 1)
hipStreamCaptureStatusInvalidated = hipStreamCaptureStatus.define('hipStreamCaptureStatusInvalidated', 2)

class hipStreamUpdateCaptureDependenciesFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipStreamAddCaptureDependencies = hipStreamUpdateCaptureDependenciesFlags.define('hipStreamAddCaptureDependencies', 0)
hipStreamSetCaptureDependencies = hipStreamUpdateCaptureDependenciesFlags.define('hipStreamSetCaptureDependencies', 1)

class hipGraphMemAttributeType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphMemAttrUsedMemCurrent = hipGraphMemAttributeType.define('hipGraphMemAttrUsedMemCurrent', 0)
hipGraphMemAttrUsedMemHigh = hipGraphMemAttributeType.define('hipGraphMemAttrUsedMemHigh', 1)
hipGraphMemAttrReservedMemCurrent = hipGraphMemAttributeType.define('hipGraphMemAttrReservedMemCurrent', 2)
hipGraphMemAttrReservedMemHigh = hipGraphMemAttributeType.define('hipGraphMemAttrReservedMemHigh', 3)

class hipUserObjectFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipUserObjectNoDestructorSync = hipUserObjectFlags.define('hipUserObjectNoDestructorSync', 1)

class hipUserObjectRetainFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphUserObjectMove = hipUserObjectRetainFlags.define('hipGraphUserObjectMove', 1)

class hipGraphInstantiateFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphInstantiateFlagAutoFreeOnLaunch = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagAutoFreeOnLaunch', 1)
hipGraphInstantiateFlagUpload = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagUpload', 2)
hipGraphInstantiateFlagDeviceLaunch = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagDeviceLaunch', 4)
hipGraphInstantiateFlagUseNodePriority = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagUseNodePriority', 8)

class hipGraphDebugDotFlags(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphDebugDotFlagsVerbose = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsVerbose', 1)
hipGraphDebugDotFlagsKernelNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsKernelNodeParams', 4)
hipGraphDebugDotFlagsMemcpyNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsMemcpyNodeParams', 8)
hipGraphDebugDotFlagsMemsetNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsMemsetNodeParams', 16)
hipGraphDebugDotFlagsHostNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsHostNodeParams', 32)
hipGraphDebugDotFlagsEventNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsEventNodeParams', 64)
hipGraphDebugDotFlagsExtSemasSignalNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsExtSemasSignalNodeParams', 128)
hipGraphDebugDotFlagsExtSemasWaitNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsExtSemasWaitNodeParams', 256)
hipGraphDebugDotFlagsKernelNodeAttributes = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsKernelNodeAttributes', 512)
hipGraphDebugDotFlagsHandles = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsHandles', 1024)

class hipGraphInstantiateResult(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphInstantiateSuccess = hipGraphInstantiateResult.define('hipGraphInstantiateSuccess', 0)
hipGraphInstantiateError = hipGraphInstantiateResult.define('hipGraphInstantiateError', 1)
hipGraphInstantiateInvalidStructure = hipGraphInstantiateResult.define('hipGraphInstantiateInvalidStructure', 2)
hipGraphInstantiateNodeOperationNotSupported = hipGraphInstantiateResult.define('hipGraphInstantiateNodeOperationNotSupported', 3)
hipGraphInstantiateMultipleDevicesNotSupported = hipGraphInstantiateResult.define('hipGraphInstantiateMultipleDevicesNotSupported', 4)

@c.record
class hipGraphInstantiateParams(c.Struct):
  SIZE = 32
  errNode_out: Annotated[hipGraphNode_t, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint64], 8]
  result_out: Annotated[hipGraphInstantiateResult, 16]
  uploadStream: Annotated[hipStream_t, 24]
@c.record
class hipMemAllocationProp(c.Struct):
  SIZE = 32
  type: Annotated[hipMemAllocationType, 0]
  requestedHandleType: Annotated[hipMemAllocationHandleType, 4]
  location: Annotated[hipMemLocation, 8]
  win32HandleMetaData: Annotated[ctypes.c_void_p, 16]
  allocFlags: Annotated[hipMemAllocationProp_allocFlags, 24]
@c.record
class hipMemAllocationProp_allocFlags(c.Struct):
  SIZE = 4
  compressionType: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  gpuDirectRDMACapable: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  usage: Annotated[Annotated[int, ctypes.c_uint16], 2]
@c.record
class hipExternalSemaphoreSignalNodeParams(c.Struct):
  SIZE = 24
  extSemArray: Annotated[c.POINTER[hipExternalSemaphore_t], 0]
  paramsArray: Annotated[c.POINTER[hipExternalSemaphoreSignalParams], 8]
  numExtSems: Annotated[Annotated[int, ctypes.c_uint32], 16]
@c.record
class hipExternalSemaphoreWaitNodeParams(c.Struct):
  SIZE = 24
  extSemArray: Annotated[c.POINTER[hipExternalSemaphore_t], 0]
  paramsArray: Annotated[c.POINTER[hipExternalSemaphoreWaitParams], 8]
  numExtSems: Annotated[Annotated[int, ctypes.c_uint32], 16]
class ihipMemGenericAllocationHandle(ctypes.Structure): pass
hipMemGenericAllocationHandle_t: TypeAlias = c.POINTER[ihipMemGenericAllocationHandle]
class hipMemAllocationGranularity_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemAllocationGranularityMinimum = hipMemAllocationGranularity_flags.define('hipMemAllocationGranularityMinimum', 0)
hipMemAllocationGranularityRecommended = hipMemAllocationGranularity_flags.define('hipMemAllocationGranularityRecommended', 1)

class hipMemHandleType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemHandleTypeGeneric = hipMemHandleType.define('hipMemHandleTypeGeneric', 0)

class hipMemOperationType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemOperationTypeMap = hipMemOperationType.define('hipMemOperationTypeMap', 1)
hipMemOperationTypeUnmap = hipMemOperationType.define('hipMemOperationTypeUnmap', 2)

class hipArraySparseSubresourceType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipArraySparseSubresourceTypeSparseLevel = hipArraySparseSubresourceType.define('hipArraySparseSubresourceTypeSparseLevel', 0)
hipArraySparseSubresourceTypeMiptail = hipArraySparseSubresourceType.define('hipArraySparseSubresourceTypeMiptail', 1)

@c.record
class hipArrayMapInfo(c.Struct):
  SIZE = 152
  resourceType: Annotated[hipResourceType, 0]
  resource: Annotated[hipArrayMapInfo_resource, 8]
  subresourceType: Annotated[hipArraySparseSubresourceType, 72]
  subresource: Annotated[hipArrayMapInfo_subresource, 80]
  memOperationType: Annotated[hipMemOperationType, 112]
  memHandleType: Annotated[hipMemHandleType, 116]
  memHandle: Annotated[hipArrayMapInfo_memHandle, 120]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 128]
  deviceBitMask: Annotated[Annotated[int, ctypes.c_uint32], 136]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 140]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 144]
class hipResourceType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipResourceTypeArray = hipResourceType.define('hipResourceTypeArray', 0)
hipResourceTypeMipmappedArray = hipResourceType.define('hipResourceTypeMipmappedArray', 1)
hipResourceTypeLinear = hipResourceType.define('hipResourceTypeLinear', 2)
hipResourceTypePitch2D = hipResourceType.define('hipResourceTypePitch2D', 3)

@c.record
class hipArrayMapInfo_resource(c.Struct):
  SIZE = 64
  mipmap: Annotated[hipMipmappedArray, 0]
  array: Annotated[hipArray_t, 0]
@c.record
class hipMipmappedArray(c.Struct):
  SIZE = 64
  data: Annotated[ctypes.c_void_p, 0]
  desc: Annotated[hipChannelFormatDesc, 8]
  type: Annotated[Annotated[int, ctypes.c_uint32], 28]
  width: Annotated[Annotated[int, ctypes.c_uint32], 32]
  height: Annotated[Annotated[int, ctypes.c_uint32], 36]
  depth: Annotated[Annotated[int, ctypes.c_uint32], 40]
  min_mipmap_level: Annotated[Annotated[int, ctypes.c_uint32], 44]
  max_mipmap_level: Annotated[Annotated[int, ctypes.c_uint32], 48]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 52]
  format: Annotated[hipArray_Format, 56]
  num_channels: Annotated[Annotated[int, ctypes.c_uint32], 60]
class hipArray_Format(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_AD_FORMAT_UNSIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT8', 1)
HIP_AD_FORMAT_UNSIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT16', 2)
HIP_AD_FORMAT_UNSIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT32', 3)
HIP_AD_FORMAT_SIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT8', 8)
HIP_AD_FORMAT_SIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT16', 9)
HIP_AD_FORMAT_SIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT32', 10)
HIP_AD_FORMAT_HALF = hipArray_Format.define('HIP_AD_FORMAT_HALF', 16)
HIP_AD_FORMAT_FLOAT = hipArray_Format.define('HIP_AD_FORMAT_FLOAT', 32)

class hipArray(ctypes.Structure): pass
hipArray_t: TypeAlias = c.POINTER[hipArray]
@c.record
class hipArrayMapInfo_subresource(c.Struct):
  SIZE = 32
  sparseLevel: Annotated[hipArrayMapInfo_subresource_sparseLevel, 0]
  miptail: Annotated[hipArrayMapInfo_subresource_miptail, 0]
@c.record
class hipArrayMapInfo_subresource_sparseLevel(c.Struct):
  SIZE = 32
  level: Annotated[Annotated[int, ctypes.c_uint32], 0]
  layer: Annotated[Annotated[int, ctypes.c_uint32], 4]
  offsetX: Annotated[Annotated[int, ctypes.c_uint32], 8]
  offsetY: Annotated[Annotated[int, ctypes.c_uint32], 12]
  offsetZ: Annotated[Annotated[int, ctypes.c_uint32], 16]
  extentWidth: Annotated[Annotated[int, ctypes.c_uint32], 20]
  extentHeight: Annotated[Annotated[int, ctypes.c_uint32], 24]
  extentDepth: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class hipArrayMapInfo_subresource_miptail(c.Struct):
  SIZE = 24
  layer: Annotated[Annotated[int, ctypes.c_uint32], 0]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
  size: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class hipArrayMapInfo_memHandle(c.Struct):
  SIZE = 8
  memHandle: Annotated[hipMemGenericAllocationHandle_t, 0]
@c.record
class hipMemcpyNodeParams(c.Struct):
  SIZE = 176
  flags: Annotated[Annotated[int, ctypes.c_int32], 0]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 4]
  copyParams: Annotated[hipMemcpy3DParms, 16]
@c.record
class hipMemcpy3DParms(c.Struct):
  SIZE = 160
  srcArray: Annotated[hipArray_t, 0]
  srcPos: Annotated[hipPos, 8]
  srcPtr: Annotated[hipPitchedPtr, 32]
  dstArray: Annotated[hipArray_t, 64]
  dstPos: Annotated[hipPos, 72]
  dstPtr: Annotated[hipPitchedPtr, 96]
  extent: Annotated[hipExtent, 128]
  kind: Annotated[hipMemcpyKind, 152]
@c.record
class hipPos(c.Struct):
  SIZE = 24
  x: Annotated[size_t, 0]
  y: Annotated[size_t, 8]
  z: Annotated[size_t, 16]
@c.record
class hipPitchedPtr(c.Struct):
  SIZE = 32
  ptr: Annotated[ctypes.c_void_p, 0]
  pitch: Annotated[size_t, 8]
  xsize: Annotated[size_t, 16]
  ysize: Annotated[size_t, 24]
class hipMemcpyKind(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipMemcpyHostToHost = hipMemcpyKind.define('hipMemcpyHostToHost', 0)
hipMemcpyHostToDevice = hipMemcpyKind.define('hipMemcpyHostToDevice', 1)
hipMemcpyDeviceToHost = hipMemcpyKind.define('hipMemcpyDeviceToHost', 2)
hipMemcpyDeviceToDevice = hipMemcpyKind.define('hipMemcpyDeviceToDevice', 3)
hipMemcpyDefault = hipMemcpyKind.define('hipMemcpyDefault', 4)
hipMemcpyDeviceToDeviceNoCU = hipMemcpyKind.define('hipMemcpyDeviceToDeviceNoCU', 1024)

@c.record
class hipChildGraphNodeParams(c.Struct):
  SIZE = 8
  graph: Annotated[hipGraph_t, 0]
@c.record
class hipEventWaitNodeParams(c.Struct):
  SIZE = 8
  event: Annotated[hipEvent_t, 0]
@c.record
class hipEventRecordNodeParams(c.Struct):
  SIZE = 8
  event: Annotated[hipEvent_t, 0]
@c.record
class hipMemFreeNodeParams(c.Struct):
  SIZE = 8
  dptr: Annotated[ctypes.c_void_p, 0]
@c.record
class hipGraphNodeParams(c.Struct):
  SIZE = 256
  type: Annotated[hipGraphNodeType, 0]
  reserved0: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[3]], 4]
  reserved1: Annotated[c.Array[Annotated[int, ctypes.c_int64], Literal[29]], 16]
  kernel: Annotated[hipKernelNodeParams, 16]
  memcpy: Annotated[hipMemcpyNodeParams, 16]
  memset: Annotated[hipMemsetParams, 16]
  host: Annotated[hipHostNodeParams, 16]
  graph: Annotated[hipChildGraphNodeParams, 16]
  eventWait: Annotated[hipEventWaitNodeParams, 16]
  eventRecord: Annotated[hipEventRecordNodeParams, 16]
  extSemSignal: Annotated[hipExternalSemaphoreSignalNodeParams, 16]
  extSemWait: Annotated[hipExternalSemaphoreWaitNodeParams, 16]
  alloc: Annotated[hipMemAllocNodeParams, 16]
  free: Annotated[hipMemFreeNodeParams, 16]
  reserved2: Annotated[Annotated[int, ctypes.c_int64], 248]
class hipGraphDependencyType(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipGraphDependencyTypeDefault = hipGraphDependencyType.define('hipGraphDependencyTypeDefault', 0)
hipGraphDependencyTypeProgrammatic = hipGraphDependencyType.define('hipGraphDependencyTypeProgrammatic', 1)

@c.record
class hipGraphEdgeData(c.Struct):
  SIZE = 8
  from_port: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1]
  to_port: Annotated[Annotated[int, ctypes.c_ubyte], 6]
  type: Annotated[Annotated[int, ctypes.c_ubyte], 7]
@dll.bind
def hipInit(flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipDriverGetVersion(driverVersion:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hipError_t: ...
@dll.bind
def hipRuntimeGetVersion(runtimeVersion:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hipError_t: ...
@dll.bind
def hipDeviceGet(device:c.POINTER[hipDevice_t], ordinal:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipDeviceComputeCapability(major:c.POINTER[Annotated[int, ctypes.c_int32]], minor:c.POINTER[Annotated[int, ctypes.c_int32]], device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetName(name:c.POINTER[Annotated[bytes, ctypes.c_char]], len:Annotated[int, ctypes.c_int32], device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetUuid(uuid:c.POINTER[hipUUID], device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetP2PAttribute(value:c.POINTER[Annotated[int, ctypes.c_int32]], attr:hipDeviceP2PAttr, srcDevice:Annotated[int, ctypes.c_int32], dstDevice:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipDeviceGetPCIBusId(pciBusId:c.POINTER[Annotated[bytes, ctypes.c_char]], len:Annotated[int, ctypes.c_int32], device:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipDeviceGetByPCIBusId(device:c.POINTER[Annotated[int, ctypes.c_int32]], pciBusId:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hipError_t: ...
@dll.bind
def hipDeviceTotalMem(bytes:c.POINTER[size_t], device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDeviceSynchronize() -> hipError_t: ...
@dll.bind
def hipDeviceReset() -> hipError_t: ...
@dll.bind
def hipSetDevice(deviceId:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipSetValidDevices(device_arr:c.POINTER[Annotated[int, ctypes.c_int32]], len:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipGetDevice(deviceId:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hipError_t: ...
@dll.bind
def hipGetDeviceCount(count:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hipError_t: ...
@dll.bind
def hipDeviceGetAttribute(pi:c.POINTER[Annotated[int, ctypes.c_int32]], attr:hipDeviceAttribute_t, deviceId:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipDeviceGetDefaultMemPool(mem_pool:c.POINTER[hipMemPool_t], device:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipDeviceSetMemPool(device:Annotated[int, ctypes.c_int32], mem_pool:hipMemPool_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetMemPool(mem_pool:c.POINTER[hipMemPool_t], device:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipGetDevicePropertiesR0600(prop:c.POINTER[hipDeviceProp_tR0600], deviceId:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipDeviceSetCacheConfig(cacheConfig:hipFuncCache_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetCacheConfig(cacheConfig:c.POINTER[hipFuncCache_t]) -> hipError_t: ...
@dll.bind
def hipDeviceGetLimit(pValue:c.POINTER[size_t], limit:hipLimit_t) -> hipError_t: ...
@dll.bind
def hipDeviceSetLimit(limit:hipLimit_t, value:size_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetSharedMemConfig(pConfig:c.POINTER[hipSharedMemConfig]) -> hipError_t: ...
@dll.bind
def hipGetDeviceFlags(flags:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> hipError_t: ...
@dll.bind
def hipDeviceSetSharedMemConfig(config:hipSharedMemConfig) -> hipError_t: ...
@dll.bind
def hipSetDeviceFlags(flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipChooseDeviceR0600(device:c.POINTER[Annotated[int, ctypes.c_int32]], prop:c.POINTER[hipDeviceProp_tR0600]) -> hipError_t: ...
@dll.bind
def hipExtGetLinkTypeAndHopCount(device1:Annotated[int, ctypes.c_int32], device2:Annotated[int, ctypes.c_int32], linktype:c.POINTER[uint32_t], hopcount:c.POINTER[uint32_t]) -> hipError_t: ...
@dll.bind
def hipIpcGetMemHandle(handle:c.POINTER[hipIpcMemHandle_t], devPtr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipIpcOpenMemHandle(devPtr:c.POINTER[ctypes.c_void_p], handle:hipIpcMemHandle_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipIpcCloseMemHandle(devPtr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipIpcGetEventHandle(handle:c.POINTER[hipIpcEventHandle_t], event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipIpcOpenEventHandle(event:c.POINTER[hipEvent_t], handle:hipIpcEventHandle_t) -> hipError_t: ...
@dll.bind
def hipFuncSetAttribute(func:ctypes.c_void_p, attr:hipFuncAttribute, value:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipFuncSetCacheConfig(func:ctypes.c_void_p, config:hipFuncCache_t) -> hipError_t: ...
@dll.bind
def hipFuncSetSharedMemConfig(func:ctypes.c_void_p, config:hipSharedMemConfig) -> hipError_t: ...
@dll.bind
def hipGetLastError() -> hipError_t: ...
@dll.bind
def hipExtGetLastError() -> hipError_t: ...
@dll.bind
def hipPeekAtLastError() -> hipError_t: ...
@dll.bind
def hipGetErrorName(hip_error:hipError_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def hipGetErrorString(hipError:hipError_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def hipDrvGetErrorName(hipError:hipError_t, errorString:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> hipError_t: ...
@dll.bind
def hipDrvGetErrorString(hipError:hipError_t, errorString:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> hipError_t: ...
@dll.bind
def hipStreamCreate(stream:c.POINTER[hipStream_t]) -> hipError_t: ...
@dll.bind
def hipStreamCreateWithFlags(stream:c.POINTER[hipStream_t], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipStreamCreateWithPriority(stream:c.POINTER[hipStream_t], flags:Annotated[int, ctypes.c_uint32], priority:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipDeviceGetStreamPriorityRange(leastPriority:c.POINTER[Annotated[int, ctypes.c_int32]], greatestPriority:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hipError_t: ...
@dll.bind
def hipStreamDestroy(stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipStreamQuery(stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipStreamSynchronize(stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipStreamWaitEvent(stream:hipStream_t, event:hipEvent_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipStreamGetFlags(stream:hipStream_t, flags:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> hipError_t: ...
@dll.bind
def hipStreamGetPriority(stream:hipStream_t, priority:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hipError_t: ...
@dll.bind
def hipStreamGetDevice(stream:hipStream_t, device:c.POINTER[hipDevice_t]) -> hipError_t: ...
@dll.bind
def hipExtStreamCreateWithCUMask(stream:c.POINTER[hipStream_t], cuMaskSize:uint32_t, cuMask:c.POINTER[uint32_t]) -> hipError_t: ...
@dll.bind
def hipExtStreamGetCUMask(stream:hipStream_t, cuMaskSize:uint32_t, cuMask:c.POINTER[uint32_t]) -> hipError_t: ...
hipStreamCallback_t: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[ihipStream_t], hipError_t, ctypes.c_void_p]]
@dll.bind
def hipStreamAddCallback(stream:hipStream_t, callback:hipStreamCallback_t, userData:ctypes.c_void_p, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipStreamWaitValue32(stream:hipStream_t, ptr:ctypes.c_void_p, value:uint32_t, flags:Annotated[int, ctypes.c_uint32], mask:uint32_t) -> hipError_t: ...
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@dll.bind
def hipStreamWaitValue64(stream:hipStream_t, ptr:ctypes.c_void_p, value:uint64_t, flags:Annotated[int, ctypes.c_uint32], mask:uint64_t) -> hipError_t: ...
@dll.bind
def hipStreamWriteValue32(stream:hipStream_t, ptr:ctypes.c_void_p, value:uint32_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipStreamWriteValue64(stream:hipStream_t, ptr:ctypes.c_void_p, value:uint64_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipEventCreateWithFlags(event:c.POINTER[hipEvent_t], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipEventCreate(event:c.POINTER[hipEvent_t]) -> hipError_t: ...
@dll.bind
def hipEventRecord(event:hipEvent_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipEventDestroy(event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipEventSynchronize(event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipEventElapsedTime(ms:c.POINTER[Annotated[float, ctypes.c_float]], start:hipEvent_t, stop:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipEventQuery(event:hipEvent_t) -> hipError_t: ...
class hipPointer_attribute(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_POINTER_ATTRIBUTE_CONTEXT = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_CONTEXT', 1)
HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_MEMORY_TYPE', 2)
HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_DEVICE_POINTER', 3)
HIP_POINTER_ATTRIBUTE_HOST_POINTER = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_HOST_POINTER', 4)
HIP_POINTER_ATTRIBUTE_P2P_TOKENS = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_P2P_TOKENS', 5)
HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS', 6)
HIP_POINTER_ATTRIBUTE_BUFFER_ID = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_BUFFER_ID', 7)
HIP_POINTER_ATTRIBUTE_IS_MANAGED = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_IS_MANAGED', 8)
HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL', 9)
HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE', 10)
HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR', 11)
HIP_POINTER_ATTRIBUTE_RANGE_SIZE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_RANGE_SIZE', 12)
HIP_POINTER_ATTRIBUTE_MAPPED = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_MAPPED', 13)
HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES', 14)
HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE', 15)
HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS', 16)
HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE', 17)

@dll.bind
def hipPointerSetAttribute(value:ctypes.c_void_p, attribute:hipPointer_attribute, ptr:hipDeviceptr_t) -> hipError_t: ...
@dll.bind
def hipPointerGetAttributes(attributes:c.POINTER[hipPointerAttribute_t], ptr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipPointerGetAttribute(data:ctypes.c_void_p, attribute:hipPointer_attribute, ptr:hipDeviceptr_t) -> hipError_t: ...
@dll.bind
def hipDrvPointerGetAttributes(numAttributes:Annotated[int, ctypes.c_uint32], attributes:c.POINTER[hipPointer_attribute], data:c.POINTER[ctypes.c_void_p], ptr:hipDeviceptr_t) -> hipError_t: ...
@dll.bind
def hipImportExternalSemaphore(extSem_out:c.POINTER[hipExternalSemaphore_t], semHandleDesc:c.POINTER[hipExternalSemaphoreHandleDesc]) -> hipError_t: ...
@dll.bind
def hipSignalExternalSemaphoresAsync(extSemArray:c.POINTER[hipExternalSemaphore_t], paramsArray:c.POINTER[hipExternalSemaphoreSignalParams], numExtSems:Annotated[int, ctypes.c_uint32], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipWaitExternalSemaphoresAsync(extSemArray:c.POINTER[hipExternalSemaphore_t], paramsArray:c.POINTER[hipExternalSemaphoreWaitParams], numExtSems:Annotated[int, ctypes.c_uint32], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipDestroyExternalSemaphore(extSem:hipExternalSemaphore_t) -> hipError_t: ...
@dll.bind
def hipImportExternalMemory(extMem_out:c.POINTER[hipExternalMemory_t], memHandleDesc:c.POINTER[hipExternalMemoryHandleDesc]) -> hipError_t: ...
@dll.bind
def hipExternalMemoryGetMappedBuffer(devPtr:c.POINTER[ctypes.c_void_p], extMem:hipExternalMemory_t, bufferDesc:c.POINTER[hipExternalMemoryBufferDesc]) -> hipError_t: ...
@dll.bind
def hipDestroyExternalMemory(extMem:hipExternalMemory_t) -> hipError_t: ...
hipMipmappedArray_t: TypeAlias = c.POINTER[hipMipmappedArray]
@dll.bind
def hipExternalMemoryGetMappedMipmappedArray(mipmap:c.POINTER[hipMipmappedArray_t], extMem:hipExternalMemory_t, mipmapDesc:c.POINTER[hipExternalMemoryMipmappedArrayDesc]) -> hipError_t: ...
@dll.bind
def hipMalloc(ptr:c.POINTER[ctypes.c_void_p], size:size_t) -> hipError_t: ...
@dll.bind
def hipExtMallocWithFlags(ptr:c.POINTER[ctypes.c_void_p], sizeBytes:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipMallocHost(ptr:c.POINTER[ctypes.c_void_p], size:size_t) -> hipError_t: ...
@dll.bind
def hipMemAllocHost(ptr:c.POINTER[ctypes.c_void_p], size:size_t) -> hipError_t: ...
@dll.bind
def hipHostMalloc(ptr:c.POINTER[ctypes.c_void_p], size:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipMallocManaged(dev_ptr:c.POINTER[ctypes.c_void_p], size:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipMemPrefetchAsync(dev_ptr:ctypes.c_void_p, count:size_t, device:Annotated[int, ctypes.c_int32], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemAdvise(dev_ptr:ctypes.c_void_p, count:size_t, advice:hipMemoryAdvise, device:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipMemRangeGetAttribute(data:ctypes.c_void_p, data_size:size_t, attribute:hipMemRangeAttribute, dev_ptr:ctypes.c_void_p, count:size_t) -> hipError_t: ...
@dll.bind
def hipMemRangeGetAttributes(data:c.POINTER[ctypes.c_void_p], data_sizes:c.POINTER[size_t], attributes:c.POINTER[hipMemRangeAttribute], num_attributes:size_t, dev_ptr:ctypes.c_void_p, count:size_t) -> hipError_t: ...
@dll.bind
def hipStreamAttachMemAsync(stream:hipStream_t, dev_ptr:ctypes.c_void_p, length:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipMallocAsync(dev_ptr:c.POINTER[ctypes.c_void_p], size:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipFreeAsync(dev_ptr:ctypes.c_void_p, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemPoolTrimTo(mem_pool:hipMemPool_t, min_bytes_to_hold:size_t) -> hipError_t: ...
@dll.bind
def hipMemPoolSetAttribute(mem_pool:hipMemPool_t, attr:hipMemPoolAttr, value:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipMemPoolGetAttribute(mem_pool:hipMemPool_t, attr:hipMemPoolAttr, value:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipMemPoolSetAccess(mem_pool:hipMemPool_t, desc_list:c.POINTER[hipMemAccessDesc], count:size_t) -> hipError_t: ...
@dll.bind
def hipMemPoolGetAccess(flags:c.POINTER[hipMemAccessFlags], mem_pool:hipMemPool_t, location:c.POINTER[hipMemLocation]) -> hipError_t: ...
@dll.bind
def hipMemPoolCreate(mem_pool:c.POINTER[hipMemPool_t], pool_props:c.POINTER[hipMemPoolProps]) -> hipError_t: ...
@dll.bind
def hipMemPoolDestroy(mem_pool:hipMemPool_t) -> hipError_t: ...
@dll.bind
def hipMallocFromPoolAsync(dev_ptr:c.POINTER[ctypes.c_void_p], size:size_t, mem_pool:hipMemPool_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemPoolExportToShareableHandle(shared_handle:ctypes.c_void_p, mem_pool:hipMemPool_t, handle_type:hipMemAllocationHandleType, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipMemPoolImportFromShareableHandle(mem_pool:c.POINTER[hipMemPool_t], shared_handle:ctypes.c_void_p, handle_type:hipMemAllocationHandleType, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipMemPoolExportPointer(export_data:c.POINTER[hipMemPoolPtrExportData], dev_ptr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipMemPoolImportPointer(dev_ptr:c.POINTER[ctypes.c_void_p], mem_pool:hipMemPool_t, export_data:c.POINTER[hipMemPoolPtrExportData]) -> hipError_t: ...
@dll.bind
def hipHostAlloc(ptr:c.POINTER[ctypes.c_void_p], size:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipHostGetDevicePointer(devPtr:c.POINTER[ctypes.c_void_p], hstPtr:ctypes.c_void_p, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipHostGetFlags(flagsPtr:c.POINTER[Annotated[int, ctypes.c_uint32]], hostPtr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipHostRegister(hostPtr:ctypes.c_void_p, sizeBytes:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipHostUnregister(hostPtr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipMallocPitch(ptr:c.POINTER[ctypes.c_void_p], pitch:c.POINTER[size_t], width:size_t, height:size_t) -> hipError_t: ...
@dll.bind
def hipMemAllocPitch(dptr:c.POINTER[hipDeviceptr_t], pitch:c.POINTER[size_t], widthInBytes:size_t, height:size_t, elementSizeBytes:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipFree(ptr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipFreeHost(ptr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipHostFree(ptr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipMemcpy(dst:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyWithStream(dst:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyHtoD(dst:hipDeviceptr_t, src:ctypes.c_void_p, sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoH(dst:ctypes.c_void_p, src:hipDeviceptr_t, sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoD(dst:hipDeviceptr_t, src:hipDeviceptr_t, sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAtoD(dstDevice:hipDeviceptr_t, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoA(dstArray:hipArray_t, dstOffset:size_t, srcDevice:hipDeviceptr_t, ByteCount:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAtoA(dstArray:hipArray_t, dstOffset:size_t, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyHtoDAsync(dst:hipDeviceptr_t, src:ctypes.c_void_p, sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoHAsync(dst:ctypes.c_void_p, src:hipDeviceptr_t, sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoDAsync(dst:hipDeviceptr_t, src:hipDeviceptr_t, sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAtoHAsync(dstHost:ctypes.c_void_p, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyHtoAAsync(dstArray:hipArray_t, dstOffset:size_t, srcHost:ctypes.c_void_p, ByteCount:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipModuleGetGlobal(dptr:c.POINTER[hipDeviceptr_t], bytes:c.POINTER[size_t], hmod:hipModule_t, name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hipError_t: ...
@dll.bind
def hipGetSymbolAddress(devPtr:c.POINTER[ctypes.c_void_p], symbol:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipGetSymbolSize(size:c.POINTER[size_t], symbol:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipGetProcAddress(symbol:c.POINTER[Annotated[bytes, ctypes.c_char]], pfn:c.POINTER[ctypes.c_void_p], hipVersion:Annotated[int, ctypes.c_int32], flags:uint64_t, symbolStatus:c.POINTER[hipDriverProcAddressQueryResult]) -> hipError_t: ...
@dll.bind
def hipMemcpyToSymbol(symbol:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyToSymbolAsync(symbol:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, offset:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyFromSymbol(dst:ctypes.c_void_p, symbol:ctypes.c_void_p, sizeBytes:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyFromSymbolAsync(dst:ctypes.c_void_p, symbol:ctypes.c_void_p, sizeBytes:size_t, offset:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAsync(dst:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemset(dst:ctypes.c_void_p, value:Annotated[int, ctypes.c_int32], sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemsetD8(dest:hipDeviceptr_t, value:Annotated[int, ctypes.c_ubyte], count:size_t) -> hipError_t: ...
@dll.bind
def hipMemsetD8Async(dest:hipDeviceptr_t, value:Annotated[int, ctypes.c_ubyte], count:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemsetD16(dest:hipDeviceptr_t, value:Annotated[int, ctypes.c_uint16], count:size_t) -> hipError_t: ...
@dll.bind
def hipMemsetD16Async(dest:hipDeviceptr_t, value:Annotated[int, ctypes.c_uint16], count:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemsetD32(dest:hipDeviceptr_t, value:Annotated[int, ctypes.c_int32], count:size_t) -> hipError_t: ...
@dll.bind
def hipMemsetAsync(dst:ctypes.c_void_p, value:Annotated[int, ctypes.c_int32], sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemsetD32Async(dst:hipDeviceptr_t, value:Annotated[int, ctypes.c_int32], count:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemset2D(dst:ctypes.c_void_p, pitch:size_t, value:Annotated[int, ctypes.c_int32], width:size_t, height:size_t) -> hipError_t: ...
@dll.bind
def hipMemset2DAsync(dst:ctypes.c_void_p, pitch:size_t, value:Annotated[int, ctypes.c_int32], width:size_t, height:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemset3D(pitchedDevPtr:hipPitchedPtr, value:Annotated[int, ctypes.c_int32], extent:hipExtent) -> hipError_t: ...
@dll.bind
def hipMemset3DAsync(pitchedDevPtr:hipPitchedPtr, value:Annotated[int, ctypes.c_int32], extent:hipExtent, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemGetInfo(free:c.POINTER[size_t], total:c.POINTER[size_t]) -> hipError_t: ...
@dll.bind
def hipMemPtrGetInfo(ptr:ctypes.c_void_p, size:c.POINTER[size_t]) -> hipError_t: ...
@dll.bind
def hipMallocArray(array:c.POINTER[hipArray_t], desc:c.POINTER[hipChannelFormatDesc], width:size_t, height:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@c.record
class HIP_ARRAY_DESCRIPTOR(c.Struct):
  SIZE = 24
  Width: Annotated[size_t, 0]
  Height: Annotated[size_t, 8]
  Format: Annotated[hipArray_Format, 16]
  NumChannels: Annotated[Annotated[int, ctypes.c_uint32], 20]
@dll.bind
def hipArrayCreate(pHandle:c.POINTER[hipArray_t], pAllocateArray:c.POINTER[HIP_ARRAY_DESCRIPTOR]) -> hipError_t: ...
@dll.bind
def hipArrayDestroy(array:hipArray_t) -> hipError_t: ...
@c.record
class HIP_ARRAY3D_DESCRIPTOR(c.Struct):
  SIZE = 40
  Width: Annotated[size_t, 0]
  Height: Annotated[size_t, 8]
  Depth: Annotated[size_t, 16]
  Format: Annotated[hipArray_Format, 24]
  NumChannels: Annotated[Annotated[int, ctypes.c_uint32], 28]
  Flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
@dll.bind
def hipArray3DCreate(array:c.POINTER[hipArray_t], pAllocateArray:c.POINTER[HIP_ARRAY3D_DESCRIPTOR]) -> hipError_t: ...
@dll.bind
def hipMalloc3D(pitchedDevPtr:c.POINTER[hipPitchedPtr], extent:hipExtent) -> hipError_t: ...
@dll.bind
def hipFreeArray(array:hipArray_t) -> hipError_t: ...
@dll.bind
def hipMalloc3DArray(array:c.POINTER[hipArray_t], desc:c.POINTER[hipChannelFormatDesc], extent:hipExtent, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipArrayGetInfo(desc:c.POINTER[hipChannelFormatDesc], extent:c.POINTER[hipExtent], flags:c.POINTER[Annotated[int, ctypes.c_uint32]], array:hipArray_t) -> hipError_t: ...
@dll.bind
def hipArrayGetDescriptor(pArrayDescriptor:c.POINTER[HIP_ARRAY_DESCRIPTOR], array:hipArray_t) -> hipError_t: ...
@dll.bind
def hipArray3DGetDescriptor(pArrayDescriptor:c.POINTER[HIP_ARRAY3D_DESCRIPTOR], array:hipArray_t) -> hipError_t: ...
@dll.bind
def hipMemcpy2D(dst:ctypes.c_void_p, dpitch:size_t, src:ctypes.c_void_p, spitch:size_t, width:size_t, height:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@c.record
class hip_Memcpy2D(c.Struct):
  SIZE = 128
  srcXInBytes: Annotated[size_t, 0]
  srcY: Annotated[size_t, 8]
  srcMemoryType: Annotated[hipMemoryType, 16]
  srcHost: Annotated[ctypes.c_void_p, 24]
  srcDevice: Annotated[hipDeviceptr_t, 32]
  srcArray: Annotated[hipArray_t, 40]
  srcPitch: Annotated[size_t, 48]
  dstXInBytes: Annotated[size_t, 56]
  dstY: Annotated[size_t, 64]
  dstMemoryType: Annotated[hipMemoryType, 72]
  dstHost: Annotated[ctypes.c_void_p, 80]
  dstDevice: Annotated[hipDeviceptr_t, 88]
  dstArray: Annotated[hipArray_t, 96]
  dstPitch: Annotated[size_t, 104]
  WidthInBytes: Annotated[size_t, 112]
  Height: Annotated[size_t, 120]
@dll.bind
def hipMemcpyParam2D(pCopy:c.POINTER[hip_Memcpy2D]) -> hipError_t: ...
@dll.bind
def hipMemcpyParam2DAsync(pCopy:c.POINTER[hip_Memcpy2D], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpy2DAsync(dst:ctypes.c_void_p, dpitch:size_t, src:ctypes.c_void_p, spitch:size_t, width:size_t, height:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpy2DToArray(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.c_void_p, spitch:size_t, width:size_t, height:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpy2DToArrayAsync(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.c_void_p, spitch:size_t, width:size_t, height:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
hipArray_const_t: TypeAlias = c.POINTER[hipArray]
@dll.bind
def hipMemcpy2DArrayToArray(dst:hipArray_t, wOffsetDst:size_t, hOffsetDst:size_t, src:hipArray_const_t, wOffsetSrc:size_t, hOffsetSrc:size_t, width:size_t, height:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyToArray(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.c_void_p, count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyFromArray(dst:ctypes.c_void_p, srcArray:hipArray_const_t, wOffset:size_t, hOffset:size_t, count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpy2DFromArray(dst:ctypes.c_void_p, dpitch:size_t, src:hipArray_const_t, wOffset:size_t, hOffset:size_t, width:size_t, height:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpy2DFromArrayAsync(dst:ctypes.c_void_p, dpitch:size_t, src:hipArray_const_t, wOffset:size_t, hOffset:size_t, width:size_t, height:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAtoH(dst:ctypes.c_void_p, srcArray:hipArray_t, srcOffset:size_t, count:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyHtoA(dstArray:hipArray_t, dstOffset:size_t, srcHost:ctypes.c_void_p, count:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpy3D(p:c.POINTER[hipMemcpy3DParms]) -> hipError_t: ...
@dll.bind
def hipMemcpy3DAsync(p:c.POINTER[hipMemcpy3DParms], stream:hipStream_t) -> hipError_t: ...
@c.record
class HIP_MEMCPY3D(c.Struct):
  SIZE = 184
  srcXInBytes: Annotated[size_t, 0]
  srcY: Annotated[size_t, 8]
  srcZ: Annotated[size_t, 16]
  srcLOD: Annotated[size_t, 24]
  srcMemoryType: Annotated[hipMemoryType, 32]
  srcHost: Annotated[ctypes.c_void_p, 40]
  srcDevice: Annotated[hipDeviceptr_t, 48]
  srcArray: Annotated[hipArray_t, 56]
  srcPitch: Annotated[size_t, 64]
  srcHeight: Annotated[size_t, 72]
  dstXInBytes: Annotated[size_t, 80]
  dstY: Annotated[size_t, 88]
  dstZ: Annotated[size_t, 96]
  dstLOD: Annotated[size_t, 104]
  dstMemoryType: Annotated[hipMemoryType, 112]
  dstHost: Annotated[ctypes.c_void_p, 120]
  dstDevice: Annotated[hipDeviceptr_t, 128]
  dstArray: Annotated[hipArray_t, 136]
  dstPitch: Annotated[size_t, 144]
  dstHeight: Annotated[size_t, 152]
  WidthInBytes: Annotated[size_t, 160]
  Height: Annotated[size_t, 168]
  Depth: Annotated[size_t, 176]
@dll.bind
def hipDrvMemcpy3D(pCopy:c.POINTER[HIP_MEMCPY3D]) -> hipError_t: ...
@dll.bind
def hipDrvMemcpy3DAsync(pCopy:c.POINTER[HIP_MEMCPY3D], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipDeviceCanAccessPeer(canAccessPeer:c.POINTER[Annotated[int, ctypes.c_int32]], deviceId:Annotated[int, ctypes.c_int32], peerDeviceId:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipDeviceEnablePeerAccess(peerDeviceId:Annotated[int, ctypes.c_int32], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipDeviceDisablePeerAccess(peerDeviceId:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipMemGetAddressRange(pbase:c.POINTER[hipDeviceptr_t], psize:c.POINTER[size_t], dptr:hipDeviceptr_t) -> hipError_t: ...
@dll.bind
def hipMemcpyPeer(dst:ctypes.c_void_p, dstDeviceId:Annotated[int, ctypes.c_int32], src:ctypes.c_void_p, srcDeviceId:Annotated[int, ctypes.c_int32], sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyPeerAsync(dst:ctypes.c_void_p, dstDeviceId:Annotated[int, ctypes.c_int32], src:ctypes.c_void_p, srcDevice:Annotated[int, ctypes.c_int32], sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipCtxCreate(ctx:c.POINTER[hipCtx_t], flags:Annotated[int, ctypes.c_uint32], device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipCtxDestroy(ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipCtxPopCurrent(ctx:c.POINTER[hipCtx_t]) -> hipError_t: ...
@dll.bind
def hipCtxPushCurrent(ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipCtxSetCurrent(ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipCtxGetCurrent(ctx:c.POINTER[hipCtx_t]) -> hipError_t: ...
@dll.bind
def hipCtxGetDevice(device:c.POINTER[hipDevice_t]) -> hipError_t: ...
@dll.bind
def hipCtxGetApiVersion(ctx:hipCtx_t, apiVersion:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hipError_t: ...
@dll.bind
def hipCtxGetCacheConfig(cacheConfig:c.POINTER[hipFuncCache_t]) -> hipError_t: ...
@dll.bind
def hipCtxSetCacheConfig(cacheConfig:hipFuncCache_t) -> hipError_t: ...
@dll.bind
def hipCtxSetSharedMemConfig(config:hipSharedMemConfig) -> hipError_t: ...
@dll.bind
def hipCtxGetSharedMemConfig(pConfig:c.POINTER[hipSharedMemConfig]) -> hipError_t: ...
@dll.bind
def hipCtxSynchronize() -> hipError_t: ...
@dll.bind
def hipCtxGetFlags(flags:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> hipError_t: ...
@dll.bind
def hipCtxEnablePeerAccess(peerCtx:hipCtx_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipCtxDisablePeerAccess(peerCtx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxGetState(dev:hipDevice_t, flags:c.POINTER[Annotated[int, ctypes.c_uint32]], active:c.POINTER[Annotated[int, ctypes.c_int32]]) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxRelease(dev:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxRetain(pctx:c.POINTER[hipCtx_t], dev:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxReset(dev:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxSetFlags(dev:hipDevice_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipModuleLoad(module:c.POINTER[hipModule_t], fname:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hipError_t: ...
@dll.bind
def hipModuleUnload(module:hipModule_t) -> hipError_t: ...
@dll.bind
def hipModuleGetFunction(function:c.POINTER[hipFunction_t], module:hipModule_t, kname:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hipError_t: ...
@dll.bind
def hipFuncGetAttributes(attr:c.POINTER[hipFuncAttributes], func:ctypes.c_void_p) -> hipError_t: ...
class hipFunction_attribute(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK', 0)
HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES', 1)
HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES', 2)
HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', 3)
HIP_FUNC_ATTRIBUTE_NUM_REGS = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_NUM_REGS', 4)
HIP_FUNC_ATTRIBUTE_PTX_VERSION = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_PTX_VERSION', 5)
HIP_FUNC_ATTRIBUTE_BINARY_VERSION = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_BINARY_VERSION', 6)
HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA', 7)
HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES', 8)
HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT', 9)
HIP_FUNC_ATTRIBUTE_MAX = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_MAX', 10)

@dll.bind
def hipFuncGetAttribute(value:c.POINTER[Annotated[int, ctypes.c_int32]], attrib:hipFunction_attribute, hfunc:hipFunction_t) -> hipError_t: ...
@dll.bind
def hipGetFuncBySymbol(functionPtr:c.POINTER[hipFunction_t], symbolPtr:ctypes.c_void_p) -> hipError_t: ...
@c.record
class textureReference(c.Struct):
  SIZE = 88
  normalized: Annotated[Annotated[int, ctypes.c_int32], 0]
  readMode: Annotated[hipTextureReadMode, 4]
  filterMode: Annotated[hipTextureFilterMode, 8]
  addressMode: Annotated[c.Array[hipTextureAddressMode, Literal[3]], 12]
  channelDesc: Annotated[hipChannelFormatDesc, 24]
  sRGB: Annotated[Annotated[int, ctypes.c_int32], 44]
  maxAnisotropy: Annotated[Annotated[int, ctypes.c_uint32], 48]
  mipmapFilterMode: Annotated[hipTextureFilterMode, 52]
  mipmapLevelBias: Annotated[Annotated[float, ctypes.c_float], 56]
  minMipmapLevelClamp: Annotated[Annotated[float, ctypes.c_float], 60]
  maxMipmapLevelClamp: Annotated[Annotated[float, ctypes.c_float], 64]
  textureObject: Annotated[hipTextureObject_t, 72]
  numChannels: Annotated[Annotated[int, ctypes.c_int32], 80]
  format: Annotated[hipArray_Format, 84]
class hipTextureReadMode(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipReadModeElementType = hipTextureReadMode.define('hipReadModeElementType', 0)
hipReadModeNormalizedFloat = hipTextureReadMode.define('hipReadModeNormalizedFloat', 1)

class hipTextureFilterMode(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipFilterModePoint = hipTextureFilterMode.define('hipFilterModePoint', 0)
hipFilterModeLinear = hipTextureFilterMode.define('hipFilterModeLinear', 1)

class hipTextureAddressMode(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipAddressModeWrap = hipTextureAddressMode.define('hipAddressModeWrap', 0)
hipAddressModeClamp = hipTextureAddressMode.define('hipAddressModeClamp', 1)
hipAddressModeMirror = hipTextureAddressMode.define('hipAddressModeMirror', 2)
hipAddressModeBorder = hipTextureAddressMode.define('hipAddressModeBorder', 3)

class __hip_texture(ctypes.Structure): pass
hipTextureObject_t: TypeAlias = c.POINTER[__hip_texture]
@dll.bind
def hipModuleGetTexRef(texRef:c.POINTER[c.POINTER[textureReference]], hmod:hipModule_t, name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hipError_t: ...
@dll.bind
def hipModuleLoadData(module:c.POINTER[hipModule_t], image:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipModuleLoadDataEx(module:c.POINTER[hipModule_t], image:ctypes.c_void_p, numOptions:Annotated[int, ctypes.c_uint32], options:c.POINTER[hipJitOption], optionValues:c.POINTER[ctypes.c_void_p]) -> hipError_t: ...
@dll.bind
def hipModuleLaunchKernel(f:hipFunction_t, gridDimX:Annotated[int, ctypes.c_uint32], gridDimY:Annotated[int, ctypes.c_uint32], gridDimZ:Annotated[int, ctypes.c_uint32], blockDimX:Annotated[int, ctypes.c_uint32], blockDimY:Annotated[int, ctypes.c_uint32], blockDimZ:Annotated[int, ctypes.c_uint32], sharedMemBytes:Annotated[int, ctypes.c_uint32], stream:hipStream_t, kernelParams:c.POINTER[ctypes.c_void_p], extra:c.POINTER[ctypes.c_void_p]) -> hipError_t: ...
@dll.bind
def hipModuleLaunchCooperativeKernel(f:hipFunction_t, gridDimX:Annotated[int, ctypes.c_uint32], gridDimY:Annotated[int, ctypes.c_uint32], gridDimZ:Annotated[int, ctypes.c_uint32], blockDimX:Annotated[int, ctypes.c_uint32], blockDimY:Annotated[int, ctypes.c_uint32], blockDimZ:Annotated[int, ctypes.c_uint32], sharedMemBytes:Annotated[int, ctypes.c_uint32], stream:hipStream_t, kernelParams:c.POINTER[ctypes.c_void_p]) -> hipError_t: ...
@dll.bind
def hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList:c.POINTER[hipFunctionLaunchParams], numDevices:Annotated[int, ctypes.c_uint32], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipLaunchCooperativeKernel(f:ctypes.c_void_p, gridDim:dim3, blockDimX:dim3, kernelParams:c.POINTER[ctypes.c_void_p], sharedMemBytes:Annotated[int, ctypes.c_uint32], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipLaunchCooperativeKernelMultiDevice(launchParamsList:c.POINTER[hipLaunchParams], numDevices:Annotated[int, ctypes.c_int32], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipExtLaunchMultiKernelMultiDevice(launchParamsList:c.POINTER[hipLaunchParams], numDevices:Annotated[int, ctypes.c_int32], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipModuleOccupancyMaxPotentialBlockSize(gridSize:c.POINTER[Annotated[int, ctypes.c_int32]], blockSize:c.POINTER[Annotated[int, ctypes.c_int32]], f:hipFunction_t, dynSharedMemPerBlk:size_t, blockSizeLimit:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize:c.POINTER[Annotated[int, ctypes.c_int32]], blockSize:c.POINTER[Annotated[int, ctypes.c_int32]], f:hipFunction_t, dynSharedMemPerBlk:size_t, blockSizeLimit:Annotated[int, ctypes.c_int32], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:c.POINTER[Annotated[int, ctypes.c_int32]], f:hipFunction_t, blockSize:Annotated[int, ctypes.c_int32], dynSharedMemPerBlk:size_t) -> hipError_t: ...
@dll.bind
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:c.POINTER[Annotated[int, ctypes.c_int32]], f:hipFunction_t, blockSize:Annotated[int, ctypes.c_int32], dynSharedMemPerBlk:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:c.POINTER[Annotated[int, ctypes.c_int32]], f:ctypes.c_void_p, blockSize:Annotated[int, ctypes.c_int32], dynSharedMemPerBlk:size_t) -> hipError_t: ...
@dll.bind
def hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:c.POINTER[Annotated[int, ctypes.c_int32]], f:ctypes.c_void_p, blockSize:Annotated[int, ctypes.c_int32], dynSharedMemPerBlk:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipOccupancyMaxPotentialBlockSize(gridSize:c.POINTER[Annotated[int, ctypes.c_int32]], blockSize:c.POINTER[Annotated[int, ctypes.c_int32]], f:ctypes.c_void_p, dynSharedMemPerBlk:size_t, blockSizeLimit:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipProfilerStart() -> hipError_t: ...
@dll.bind
def hipProfilerStop() -> hipError_t: ...
@dll.bind
def hipConfigureCall(gridDim:dim3, blockDim:dim3, sharedMem:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipSetupArgument(arg:ctypes.c_void_p, size:size_t, offset:size_t) -> hipError_t: ...
@dll.bind
def hipLaunchByPtr(func:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def __hipPushCallConfiguration(gridDim:dim3, blockDim:dim3, sharedMem:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def __hipPopCallConfiguration(gridDim:c.POINTER[dim3], blockDim:c.POINTER[dim3], sharedMem:c.POINTER[size_t], stream:c.POINTER[hipStream_t]) -> hipError_t: ...
@dll.bind
def hipLaunchKernel(function_address:ctypes.c_void_p, numBlocks:dim3, dimBlocks:dim3, args:c.POINTER[ctypes.c_void_p], sharedMemBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipLaunchHostFunc(stream:hipStream_t, fn:hipHostFn_t, userData:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipDrvMemcpy2DUnaligned(pCopy:c.POINTER[hip_Memcpy2D]) -> hipError_t: ...
@c.record
class hipResourceDesc(c.Struct):
  SIZE = 64
  resType: Annotated[hipResourceType, 0]
  res: Annotated[hipResourceDesc_res, 8]
@c.record
class hipResourceDesc_res(c.Struct):
  SIZE = 56
  array: Annotated[hipResourceDesc_res_array, 0]
  mipmap: Annotated[hipResourceDesc_res_mipmap, 0]
  linear: Annotated[hipResourceDesc_res_linear, 0]
  pitch2D: Annotated[hipResourceDesc_res_pitch2D, 0]
@c.record
class hipResourceDesc_res_array(c.Struct):
  SIZE = 8
  array: Annotated[hipArray_t, 0]
@c.record
class hipResourceDesc_res_mipmap(c.Struct):
  SIZE = 8
  mipmap: Annotated[hipMipmappedArray_t, 0]
@c.record
class hipResourceDesc_res_linear(c.Struct):
  SIZE = 40
  devPtr: Annotated[ctypes.c_void_p, 0]
  desc: Annotated[hipChannelFormatDesc, 8]
  sizeInBytes: Annotated[size_t, 32]
@c.record
class hipResourceDesc_res_pitch2D(c.Struct):
  SIZE = 56
  devPtr: Annotated[ctypes.c_void_p, 0]
  desc: Annotated[hipChannelFormatDesc, 8]
  width: Annotated[size_t, 32]
  height: Annotated[size_t, 40]
  pitchInBytes: Annotated[size_t, 48]
@c.record
class hipTextureDesc(c.Struct):
  SIZE = 64
  addressMode: Annotated[c.Array[hipTextureAddressMode, Literal[3]], 0]
  filterMode: Annotated[hipTextureFilterMode, 12]
  readMode: Annotated[hipTextureReadMode, 16]
  sRGB: Annotated[Annotated[int, ctypes.c_int32], 20]
  borderColor: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[4]], 24]
  normalizedCoords: Annotated[Annotated[int, ctypes.c_int32], 40]
  maxAnisotropy: Annotated[Annotated[int, ctypes.c_uint32], 44]
  mipmapFilterMode: Annotated[hipTextureFilterMode, 48]
  mipmapLevelBias: Annotated[Annotated[float, ctypes.c_float], 52]
  minMipmapLevelClamp: Annotated[Annotated[float, ctypes.c_float], 56]
  maxMipmapLevelClamp: Annotated[Annotated[float, ctypes.c_float], 60]
@c.record
class hipResourceViewDesc(c.Struct):
  SIZE = 48
  format: Annotated[hipResourceViewFormat, 0]
  width: Annotated[size_t, 8]
  height: Annotated[size_t, 16]
  depth: Annotated[size_t, 24]
  firstMipmapLevel: Annotated[Annotated[int, ctypes.c_uint32], 32]
  lastMipmapLevel: Annotated[Annotated[int, ctypes.c_uint32], 36]
  firstLayer: Annotated[Annotated[int, ctypes.c_uint32], 40]
  lastLayer: Annotated[Annotated[int, ctypes.c_uint32], 44]
class hipResourceViewFormat(Annotated[int, ctypes.c_uint32], c.Enum): pass
hipResViewFormatNone = hipResourceViewFormat.define('hipResViewFormatNone', 0)
hipResViewFormatUnsignedChar1 = hipResourceViewFormat.define('hipResViewFormatUnsignedChar1', 1)
hipResViewFormatUnsignedChar2 = hipResourceViewFormat.define('hipResViewFormatUnsignedChar2', 2)
hipResViewFormatUnsignedChar4 = hipResourceViewFormat.define('hipResViewFormatUnsignedChar4', 3)
hipResViewFormatSignedChar1 = hipResourceViewFormat.define('hipResViewFormatSignedChar1', 4)
hipResViewFormatSignedChar2 = hipResourceViewFormat.define('hipResViewFormatSignedChar2', 5)
hipResViewFormatSignedChar4 = hipResourceViewFormat.define('hipResViewFormatSignedChar4', 6)
hipResViewFormatUnsignedShort1 = hipResourceViewFormat.define('hipResViewFormatUnsignedShort1', 7)
hipResViewFormatUnsignedShort2 = hipResourceViewFormat.define('hipResViewFormatUnsignedShort2', 8)
hipResViewFormatUnsignedShort4 = hipResourceViewFormat.define('hipResViewFormatUnsignedShort4', 9)
hipResViewFormatSignedShort1 = hipResourceViewFormat.define('hipResViewFormatSignedShort1', 10)
hipResViewFormatSignedShort2 = hipResourceViewFormat.define('hipResViewFormatSignedShort2', 11)
hipResViewFormatSignedShort4 = hipResourceViewFormat.define('hipResViewFormatSignedShort4', 12)
hipResViewFormatUnsignedInt1 = hipResourceViewFormat.define('hipResViewFormatUnsignedInt1', 13)
hipResViewFormatUnsignedInt2 = hipResourceViewFormat.define('hipResViewFormatUnsignedInt2', 14)
hipResViewFormatUnsignedInt4 = hipResourceViewFormat.define('hipResViewFormatUnsignedInt4', 15)
hipResViewFormatSignedInt1 = hipResourceViewFormat.define('hipResViewFormatSignedInt1', 16)
hipResViewFormatSignedInt2 = hipResourceViewFormat.define('hipResViewFormatSignedInt2', 17)
hipResViewFormatSignedInt4 = hipResourceViewFormat.define('hipResViewFormatSignedInt4', 18)
hipResViewFormatHalf1 = hipResourceViewFormat.define('hipResViewFormatHalf1', 19)
hipResViewFormatHalf2 = hipResourceViewFormat.define('hipResViewFormatHalf2', 20)
hipResViewFormatHalf4 = hipResourceViewFormat.define('hipResViewFormatHalf4', 21)
hipResViewFormatFloat1 = hipResourceViewFormat.define('hipResViewFormatFloat1', 22)
hipResViewFormatFloat2 = hipResourceViewFormat.define('hipResViewFormatFloat2', 23)
hipResViewFormatFloat4 = hipResourceViewFormat.define('hipResViewFormatFloat4', 24)
hipResViewFormatUnsignedBlockCompressed1 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed1', 25)
hipResViewFormatUnsignedBlockCompressed2 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed2', 26)
hipResViewFormatUnsignedBlockCompressed3 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed3', 27)
hipResViewFormatUnsignedBlockCompressed4 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed4', 28)
hipResViewFormatSignedBlockCompressed4 = hipResourceViewFormat.define('hipResViewFormatSignedBlockCompressed4', 29)
hipResViewFormatUnsignedBlockCompressed5 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed5', 30)
hipResViewFormatSignedBlockCompressed5 = hipResourceViewFormat.define('hipResViewFormatSignedBlockCompressed5', 31)
hipResViewFormatUnsignedBlockCompressed6H = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed6H', 32)
hipResViewFormatSignedBlockCompressed6H = hipResourceViewFormat.define('hipResViewFormatSignedBlockCompressed6H', 33)
hipResViewFormatUnsignedBlockCompressed7 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed7', 34)

@dll.bind
def hipCreateTextureObject(pTexObject:c.POINTER[hipTextureObject_t], pResDesc:c.POINTER[hipResourceDesc], pTexDesc:c.POINTER[hipTextureDesc], pResViewDesc:c.POINTER[hipResourceViewDesc]) -> hipError_t: ...
@dll.bind
def hipDestroyTextureObject(textureObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipGetChannelDesc(desc:c.POINTER[hipChannelFormatDesc], array:hipArray_const_t) -> hipError_t: ...
@dll.bind
def hipGetTextureObjectResourceDesc(pResDesc:c.POINTER[hipResourceDesc], textureObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipGetTextureObjectResourceViewDesc(pResViewDesc:c.POINTER[hipResourceViewDesc], textureObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipGetTextureObjectTextureDesc(pTexDesc:c.POINTER[hipTextureDesc], textureObject:hipTextureObject_t) -> hipError_t: ...
@c.record
class HIP_RESOURCE_DESC_st(c.Struct):
  SIZE = 144
  resType: Annotated[HIPresourcetype, 0]
  res: Annotated[HIP_RESOURCE_DESC_st_res, 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 136]
HIP_RESOURCE_DESC: TypeAlias = HIP_RESOURCE_DESC_st
class HIPresourcetype_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_RESOURCE_TYPE_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_ARRAY', 0)
HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY', 1)
HIP_RESOURCE_TYPE_LINEAR = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_LINEAR', 2)
HIP_RESOURCE_TYPE_PITCH2D = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_PITCH2D', 3)

HIPresourcetype: TypeAlias = HIPresourcetype_enum
@c.record
class HIP_RESOURCE_DESC_st_res(c.Struct):
  SIZE = 128
  array: Annotated[HIP_RESOURCE_DESC_st_res_array, 0]
  mipmap: Annotated[HIP_RESOURCE_DESC_st_res_mipmap, 0]
  linear: Annotated[HIP_RESOURCE_DESC_st_res_linear, 0]
  pitch2D: Annotated[HIP_RESOURCE_DESC_st_res_pitch2D, 0]
  reserved: Annotated[HIP_RESOURCE_DESC_st_res_reserved, 0]
@c.record
class HIP_RESOURCE_DESC_st_res_array(c.Struct):
  SIZE = 8
  hArray: Annotated[hipArray_t, 0]
@c.record
class HIP_RESOURCE_DESC_st_res_mipmap(c.Struct):
  SIZE = 8
  hMipmappedArray: Annotated[hipMipmappedArray_t, 0]
@c.record
class HIP_RESOURCE_DESC_st_res_linear(c.Struct):
  SIZE = 24
  devPtr: Annotated[hipDeviceptr_t, 0]
  format: Annotated[hipArray_Format, 8]
  numChannels: Annotated[Annotated[int, ctypes.c_uint32], 12]
  sizeInBytes: Annotated[size_t, 16]
@c.record
class HIP_RESOURCE_DESC_st_res_pitch2D(c.Struct):
  SIZE = 40
  devPtr: Annotated[hipDeviceptr_t, 0]
  format: Annotated[hipArray_Format, 8]
  numChannels: Annotated[Annotated[int, ctypes.c_uint32], 12]
  width: Annotated[size_t, 16]
  height: Annotated[size_t, 24]
  pitchInBytes: Annotated[size_t, 32]
@c.record
class HIP_RESOURCE_DESC_st_res_reserved(c.Struct):
  SIZE = 128
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[32]], 0]
@c.record
class HIP_TEXTURE_DESC_st(c.Struct):
  SIZE = 104
  addressMode: Annotated[c.Array[HIPaddress_mode, Literal[3]], 0]
  filterMode: Annotated[HIPfilter_mode, 12]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  maxAnisotropy: Annotated[Annotated[int, ctypes.c_uint32], 20]
  mipmapFilterMode: Annotated[HIPfilter_mode, 24]
  mipmapLevelBias: Annotated[Annotated[float, ctypes.c_float], 28]
  minMipmapLevelClamp: Annotated[Annotated[float, ctypes.c_float], 32]
  maxMipmapLevelClamp: Annotated[Annotated[float, ctypes.c_float], 36]
  borderColor: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[4]], 40]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_int32], Literal[12]], 56]
HIP_TEXTURE_DESC: TypeAlias = HIP_TEXTURE_DESC_st
class HIPaddress_mode_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_TR_ADDRESS_MODE_WRAP = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_WRAP', 0)
HIP_TR_ADDRESS_MODE_CLAMP = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_CLAMP', 1)
HIP_TR_ADDRESS_MODE_MIRROR = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_MIRROR', 2)
HIP_TR_ADDRESS_MODE_BORDER = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_BORDER', 3)

HIPaddress_mode: TypeAlias = HIPaddress_mode_enum
class HIPfilter_mode_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_TR_FILTER_MODE_POINT = HIPfilter_mode_enum.define('HIP_TR_FILTER_MODE_POINT', 0)
HIP_TR_FILTER_MODE_LINEAR = HIPfilter_mode_enum.define('HIP_TR_FILTER_MODE_LINEAR', 1)

HIPfilter_mode: TypeAlias = HIPfilter_mode_enum
@c.record
class HIP_RESOURCE_VIEW_DESC_st(c.Struct):
  SIZE = 112
  format: Annotated[HIPresourceViewFormat, 0]
  width: Annotated[size_t, 8]
  height: Annotated[size_t, 16]
  depth: Annotated[size_t, 24]
  firstMipmapLevel: Annotated[Annotated[int, ctypes.c_uint32], 32]
  lastMipmapLevel: Annotated[Annotated[int, ctypes.c_uint32], 36]
  firstLayer: Annotated[Annotated[int, ctypes.c_uint32], 40]
  lastLayer: Annotated[Annotated[int, ctypes.c_uint32], 44]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[16]], 48]
HIP_RESOURCE_VIEW_DESC: TypeAlias = HIP_RESOURCE_VIEW_DESC_st
class HIPresourceViewFormat_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
HIP_RES_VIEW_FORMAT_NONE = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_NONE', 0)
HIP_RES_VIEW_FORMAT_UINT_1X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_1X8', 1)
HIP_RES_VIEW_FORMAT_UINT_2X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_2X8', 2)
HIP_RES_VIEW_FORMAT_UINT_4X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_4X8', 3)
HIP_RES_VIEW_FORMAT_SINT_1X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_1X8', 4)
HIP_RES_VIEW_FORMAT_SINT_2X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_2X8', 5)
HIP_RES_VIEW_FORMAT_SINT_4X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_4X8', 6)
HIP_RES_VIEW_FORMAT_UINT_1X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_1X16', 7)
HIP_RES_VIEW_FORMAT_UINT_2X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_2X16', 8)
HIP_RES_VIEW_FORMAT_UINT_4X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_4X16', 9)
HIP_RES_VIEW_FORMAT_SINT_1X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_1X16', 10)
HIP_RES_VIEW_FORMAT_SINT_2X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_2X16', 11)
HIP_RES_VIEW_FORMAT_SINT_4X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_4X16', 12)
HIP_RES_VIEW_FORMAT_UINT_1X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_1X32', 13)
HIP_RES_VIEW_FORMAT_UINT_2X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_2X32', 14)
HIP_RES_VIEW_FORMAT_UINT_4X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_4X32', 15)
HIP_RES_VIEW_FORMAT_SINT_1X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_1X32', 16)
HIP_RES_VIEW_FORMAT_SINT_2X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_2X32', 17)
HIP_RES_VIEW_FORMAT_SINT_4X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_4X32', 18)
HIP_RES_VIEW_FORMAT_FLOAT_1X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_1X16', 19)
HIP_RES_VIEW_FORMAT_FLOAT_2X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_2X16', 20)
HIP_RES_VIEW_FORMAT_FLOAT_4X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_4X16', 21)
HIP_RES_VIEW_FORMAT_FLOAT_1X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_1X32', 22)
HIP_RES_VIEW_FORMAT_FLOAT_2X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_2X32', 23)
HIP_RES_VIEW_FORMAT_FLOAT_4X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_4X32', 24)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC1', 25)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC2', 26)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC3', 27)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC4', 28)
HIP_RES_VIEW_FORMAT_SIGNED_BC4 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SIGNED_BC4', 29)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC5', 30)
HIP_RES_VIEW_FORMAT_SIGNED_BC5 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SIGNED_BC5', 31)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H', 32)
HIP_RES_VIEW_FORMAT_SIGNED_BC6H = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SIGNED_BC6H', 33)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC7', 34)

HIPresourceViewFormat: TypeAlias = HIPresourceViewFormat_enum
@dll.bind
def hipTexObjectCreate(pTexObject:c.POINTER[hipTextureObject_t], pResDesc:c.POINTER[HIP_RESOURCE_DESC], pTexDesc:c.POINTER[HIP_TEXTURE_DESC], pResViewDesc:c.POINTER[HIP_RESOURCE_VIEW_DESC]) -> hipError_t: ...
@dll.bind
def hipTexObjectDestroy(texObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipTexObjectGetResourceDesc(pResDesc:c.POINTER[HIP_RESOURCE_DESC], texObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipTexObjectGetResourceViewDesc(pResViewDesc:c.POINTER[HIP_RESOURCE_VIEW_DESC], texObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipTexObjectGetTextureDesc(pTexDesc:c.POINTER[HIP_TEXTURE_DESC], texObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipMallocMipmappedArray(mipmappedArray:c.POINTER[hipMipmappedArray_t], desc:c.POINTER[hipChannelFormatDesc], extent:hipExtent, numLevels:Annotated[int, ctypes.c_uint32], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipFreeMipmappedArray(mipmappedArray:hipMipmappedArray_t) -> hipError_t: ...
hipMipmappedArray_const_t: TypeAlias = c.POINTER[hipMipmappedArray]
@dll.bind
def hipGetMipmappedArrayLevel(levelArray:c.POINTER[hipArray_t], mipmappedArray:hipMipmappedArray_const_t, level:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipMipmappedArrayCreate(pHandle:c.POINTER[hipMipmappedArray_t], pMipmappedArrayDesc:c.POINTER[HIP_ARRAY3D_DESCRIPTOR], numMipmapLevels:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipMipmappedArrayDestroy(hMipmappedArray:hipMipmappedArray_t) -> hipError_t: ...
@dll.bind
def hipMipmappedArrayGetLevel(pLevelArray:c.POINTER[hipArray_t], hMipMappedArray:hipMipmappedArray_t, level:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipBindTextureToMipmappedArray(tex:c.POINTER[textureReference], mipmappedArray:hipMipmappedArray_const_t, desc:c.POINTER[hipChannelFormatDesc]) -> hipError_t: ...
@dll.bind
def hipGetTextureReference(texref:c.POINTER[c.POINTER[textureReference]], symbol:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipTexRefGetBorderColor(pBorderColor:c.POINTER[Annotated[float, ctypes.c_float]], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetArray(pArray:c.POINTER[hipArray_t], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefSetAddressMode(texRef:c.POINTER[textureReference], dim:Annotated[int, ctypes.c_int32], am:hipTextureAddressMode) -> hipError_t: ...
@dll.bind
def hipTexRefSetArray(tex:c.POINTER[textureReference], array:hipArray_const_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipTexRefSetFilterMode(texRef:c.POINTER[textureReference], fm:hipTextureFilterMode) -> hipError_t: ...
@dll.bind
def hipTexRefSetFlags(texRef:c.POINTER[textureReference], Flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipTexRefSetFormat(texRef:c.POINTER[textureReference], fmt:hipArray_Format, NumPackedComponents:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipBindTexture(offset:c.POINTER[size_t], tex:c.POINTER[textureReference], devPtr:ctypes.c_void_p, desc:c.POINTER[hipChannelFormatDesc], size:size_t) -> hipError_t: ...
@dll.bind
def hipBindTexture2D(offset:c.POINTER[size_t], tex:c.POINTER[textureReference], devPtr:ctypes.c_void_p, desc:c.POINTER[hipChannelFormatDesc], width:size_t, height:size_t, pitch:size_t) -> hipError_t: ...
@dll.bind
def hipBindTextureToArray(tex:c.POINTER[textureReference], array:hipArray_const_t, desc:c.POINTER[hipChannelFormatDesc]) -> hipError_t: ...
@dll.bind
def hipGetTextureAlignmentOffset(offset:c.POINTER[size_t], texref:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipUnbindTexture(tex:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetAddress(dev_ptr:c.POINTER[hipDeviceptr_t], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetAddressMode(pam:c.POINTER[hipTextureAddressMode], texRef:c.POINTER[textureReference], dim:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipTexRefGetFilterMode(pfm:c.POINTER[hipTextureFilterMode], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetFlags(pFlags:c.POINTER[Annotated[int, ctypes.c_uint32]], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetFormat(pFormat:c.POINTER[hipArray_Format], pNumChannels:c.POINTER[Annotated[int, ctypes.c_int32]], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetMaxAnisotropy(pmaxAnsio:c.POINTER[Annotated[int, ctypes.c_int32]], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetMipmapFilterMode(pfm:c.POINTER[hipTextureFilterMode], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetMipmapLevelBias(pbias:c.POINTER[Annotated[float, ctypes.c_float]], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp:c.POINTER[Annotated[float, ctypes.c_float]], pmaxMipmapLevelClamp:c.POINTER[Annotated[float, ctypes.c_float]], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefGetMipMappedArray(pArray:c.POINTER[hipMipmappedArray_t], texRef:c.POINTER[textureReference]) -> hipError_t: ...
@dll.bind
def hipTexRefSetAddress(ByteOffset:c.POINTER[size_t], texRef:c.POINTER[textureReference], dptr:hipDeviceptr_t, bytes:size_t) -> hipError_t: ...
@dll.bind
def hipTexRefSetAddress2D(texRef:c.POINTER[textureReference], desc:c.POINTER[HIP_ARRAY_DESCRIPTOR], dptr:hipDeviceptr_t, Pitch:size_t) -> hipError_t: ...
@dll.bind
def hipTexRefSetMaxAnisotropy(texRef:c.POINTER[textureReference], maxAniso:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipTexRefSetBorderColor(texRef:c.POINTER[textureReference], pBorderColor:c.POINTER[Annotated[float, ctypes.c_float]]) -> hipError_t: ...
@dll.bind
def hipTexRefSetMipmapFilterMode(texRef:c.POINTER[textureReference], fm:hipTextureFilterMode) -> hipError_t: ...
@dll.bind
def hipTexRefSetMipmapLevelBias(texRef:c.POINTER[textureReference], bias:Annotated[float, ctypes.c_float]) -> hipError_t: ...
@dll.bind
def hipTexRefSetMipmapLevelClamp(texRef:c.POINTER[textureReference], minMipMapLevelClamp:Annotated[float, ctypes.c_float], maxMipMapLevelClamp:Annotated[float, ctypes.c_float]) -> hipError_t: ...
@dll.bind
def hipTexRefSetMipmappedArray(texRef:c.POINTER[textureReference], mipmappedArray:c.POINTER[hipMipmappedArray], Flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipApiName(id:uint32_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def hipKernelNameRef(f:hipFunction_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def hipKernelNameRefByPtr(hostFunction:ctypes.c_void_p, stream:hipStream_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def hipGetStreamDeviceId(stream:hipStream_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def hipStreamBeginCapture(stream:hipStream_t, mode:hipStreamCaptureMode) -> hipError_t: ...
@dll.bind
def hipStreamBeginCaptureToGraph(stream:hipStream_t, graph:hipGraph_t, dependencies:c.POINTER[hipGraphNode_t], dependencyData:c.POINTER[hipGraphEdgeData], numDependencies:size_t, mode:hipStreamCaptureMode) -> hipError_t: ...
@dll.bind
def hipStreamEndCapture(stream:hipStream_t, pGraph:c.POINTER[hipGraph_t]) -> hipError_t: ...
@dll.bind
def hipStreamGetCaptureInfo(stream:hipStream_t, pCaptureStatus:c.POINTER[hipStreamCaptureStatus], pId:c.POINTER[Annotated[int, ctypes.c_uint64]]) -> hipError_t: ...
@dll.bind
def hipStreamGetCaptureInfo_v2(stream:hipStream_t, captureStatus_out:c.POINTER[hipStreamCaptureStatus], id_out:c.POINTER[Annotated[int, ctypes.c_uint64]], graph_out:c.POINTER[hipGraph_t], dependencies_out:c.POINTER[c.POINTER[hipGraphNode_t]], numDependencies_out:c.POINTER[size_t]) -> hipError_t: ...
@dll.bind
def hipStreamIsCapturing(stream:hipStream_t, pCaptureStatus:c.POINTER[hipStreamCaptureStatus]) -> hipError_t: ...
@dll.bind
def hipStreamUpdateCaptureDependencies(stream:hipStream_t, dependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipThreadExchangeStreamCaptureMode(mode:c.POINTER[hipStreamCaptureMode]) -> hipError_t: ...
@dll.bind
def hipGraphCreate(pGraph:c.POINTER[hipGraph_t], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipGraphDestroy(graph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphAddDependencies(graph:hipGraph_t, _from:c.POINTER[hipGraphNode_t], to:c.POINTER[hipGraphNode_t], numDependencies:size_t) -> hipError_t: ...
@dll.bind
def hipGraphRemoveDependencies(graph:hipGraph_t, _from:c.POINTER[hipGraphNode_t], to:c.POINTER[hipGraphNode_t], numDependencies:size_t) -> hipError_t: ...
@dll.bind
def hipGraphGetEdges(graph:hipGraph_t, _from:c.POINTER[hipGraphNode_t], to:c.POINTER[hipGraphNode_t], numEdges:c.POINTER[size_t]) -> hipError_t: ...
@dll.bind
def hipGraphGetNodes(graph:hipGraph_t, nodes:c.POINTER[hipGraphNode_t], numNodes:c.POINTER[size_t]) -> hipError_t: ...
@dll.bind
def hipGraphGetRootNodes(graph:hipGraph_t, pRootNodes:c.POINTER[hipGraphNode_t], pNumRootNodes:c.POINTER[size_t]) -> hipError_t: ...
@dll.bind
def hipGraphNodeGetDependencies(node:hipGraphNode_t, pDependencies:c.POINTER[hipGraphNode_t], pNumDependencies:c.POINTER[size_t]) -> hipError_t: ...
@dll.bind
def hipGraphNodeGetDependentNodes(node:hipGraphNode_t, pDependentNodes:c.POINTER[hipGraphNode_t], pNumDependentNodes:c.POINTER[size_t]) -> hipError_t: ...
@dll.bind
def hipGraphNodeGetType(node:hipGraphNode_t, pType:c.POINTER[hipGraphNodeType]) -> hipError_t: ...
@dll.bind
def hipGraphDestroyNode(node:hipGraphNode_t) -> hipError_t: ...
@dll.bind
def hipGraphClone(pGraphClone:c.POINTER[hipGraph_t], originalGraph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphNodeFindInClone(pNode:c.POINTER[hipGraphNode_t], originalNode:hipGraphNode_t, clonedGraph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphInstantiate(pGraphExec:c.POINTER[hipGraphExec_t], graph:hipGraph_t, pErrorNode:c.POINTER[hipGraphNode_t], pLogBuffer:c.POINTER[Annotated[bytes, ctypes.c_char]], bufferSize:size_t) -> hipError_t: ...
@dll.bind
def hipGraphInstantiateWithFlags(pGraphExec:c.POINTER[hipGraphExec_t], graph:hipGraph_t, flags:Annotated[int, ctypes.c_uint64]) -> hipError_t: ...
@dll.bind
def hipGraphInstantiateWithParams(pGraphExec:c.POINTER[hipGraphExec_t], graph:hipGraph_t, instantiateParams:c.POINTER[hipGraphInstantiateParams]) -> hipError_t: ...
@dll.bind
def hipGraphLaunch(graphExec:hipGraphExec_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipGraphUpload(graphExec:hipGraphExec_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipGraphAddNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, nodeParams:c.POINTER[hipGraphNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExecDestroy(graphExec:hipGraphExec_t) -> hipError_t: ...
@dll.bind
def hipGraphExecUpdate(hGraphExec:hipGraphExec_t, hGraph:hipGraph_t, hErrorNode_out:c.POINTER[hipGraphNode_t], updateResult_out:c.POINTER[hipGraphExecUpdateResult]) -> hipError_t: ...
@dll.bind
def hipGraphAddKernelNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pNodeParams:c.POINTER[hipKernelNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipKernelNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeSetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipKernelNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExecKernelNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:c.POINTER[hipKernelNodeParams]) -> hipError_t: ...
@dll.bind
def hipDrvGraphAddMemcpyNode(phGraphNode:c.POINTER[hipGraphNode_t], hGraph:hipGraph_t, dependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, copyParams:c.POINTER[HIP_MEMCPY3D], ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipGraphAddMemcpyNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pCopyParams:c.POINTER[hipMemcpy3DParms]) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemcpy3DParms]) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeSetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemcpy3DParms]) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeSetAttribute(hNode:hipGraphNode_t, attr:hipLaunchAttributeID, value:c.POINTER[hipLaunchAttributeValue]) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeGetAttribute(hNode:hipGraphNode_t, attr:hipLaunchAttributeID, value:c.POINTER[hipLaunchAttributeValue]) -> hipError_t: ...
@dll.bind
def hipGraphExecMemcpyNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemcpy3DParms]) -> hipError_t: ...
@dll.bind
def hipGraphAddMemcpyNode1D(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, dst:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeSetParams1D(node:hipGraphNode_t, dst:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphExecMemcpyNodeSetParams1D(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, dst:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphAddMemcpyNodeFromSymbol(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, dst:ctypes.c_void_p, symbol:ctypes.c_void_p, count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeSetParamsFromSymbol(node:hipGraphNode_t, dst:ctypes.c_void_p, symbol:ctypes.c_void_p, count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, dst:ctypes.c_void_p, symbol:ctypes.c_void_p, count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphAddMemcpyNodeToSymbol(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, symbol:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeSetParamsToSymbol(node:hipGraphNode_t, symbol:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, symbol:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphAddMemsetNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pMemsetParams:c.POINTER[hipMemsetParams]) -> hipError_t: ...
@dll.bind
def hipGraphMemsetNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemsetParams]) -> hipError_t: ...
@dll.bind
def hipGraphMemsetNodeSetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemsetParams]) -> hipError_t: ...
@dll.bind
def hipGraphExecMemsetNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemsetParams]) -> hipError_t: ...
@dll.bind
def hipGraphAddHostNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pNodeParams:c.POINTER[hipHostNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphHostNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipHostNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphHostNodeSetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipHostNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExecHostNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:c.POINTER[hipHostNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphAddChildGraphNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, childGraph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphChildGraphNodeGetGraph(node:hipGraphNode_t, pGraph:c.POINTER[hipGraph_t]) -> hipError_t: ...
@dll.bind
def hipGraphExecChildGraphNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, childGraph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphAddEmptyNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t) -> hipError_t: ...
@dll.bind
def hipGraphAddEventRecordNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphEventRecordNodeGetEvent(node:hipGraphNode_t, event_out:c.POINTER[hipEvent_t]) -> hipError_t: ...
@dll.bind
def hipGraphEventRecordNodeSetEvent(node:hipGraphNode_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphExecEventRecordNodeSetEvent(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphAddEventWaitNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphEventWaitNodeGetEvent(node:hipGraphNode_t, event_out:c.POINTER[hipEvent_t]) -> hipError_t: ...
@dll.bind
def hipGraphEventWaitNodeSetEvent(node:hipGraphNode_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphExecEventWaitNodeSetEvent(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphAddMemAllocNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pNodeParams:c.POINTER[hipMemAllocNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphMemAllocNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemAllocNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphAddMemFreeNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, dev_ptr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipGraphMemFreeNodeGetParams(node:hipGraphNode_t, dev_ptr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipDeviceGetGraphMemAttribute(device:Annotated[int, ctypes.c_int32], attr:hipGraphMemAttributeType, value:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipDeviceSetGraphMemAttribute(device:Annotated[int, ctypes.c_int32], attr:hipGraphMemAttributeType, value:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipDeviceGraphMemTrim(device:Annotated[int, ctypes.c_int32]) -> hipError_t: ...
@dll.bind
def hipUserObjectCreate(object_out:c.POINTER[hipUserObject_t], ptr:ctypes.c_void_p, destroy:hipHostFn_t, initialRefcount:Annotated[int, ctypes.c_uint32], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipUserObjectRelease(object:hipUserObject_t, count:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipUserObjectRetain(object:hipUserObject_t, count:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipGraphRetainUserObject(graph:hipGraph_t, object:hipUserObject_t, count:Annotated[int, ctypes.c_uint32], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipGraphReleaseUserObject(graph:hipGraph_t, object:hipUserObject_t, count:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipGraphDebugDotPrint(graph:hipGraph_t, path:c.POINTER[Annotated[bytes, ctypes.c_char]], flags:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeCopyAttributes(hSrc:hipGraphNode_t, hDst:hipGraphNode_t) -> hipError_t: ...
@dll.bind
def hipGraphNodeSetEnabled(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, isEnabled:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipGraphNodeGetEnabled(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, isEnabled:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> hipError_t: ...
@dll.bind
def hipGraphAddExternalSemaphoresWaitNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, nodeParams:c.POINTER[hipExternalSemaphoreWaitNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphAddExternalSemaphoresSignalNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, nodeParams:c.POINTER[hipExternalSemaphoreSignalNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExternalSemaphoresSignalNodeSetParams(hNode:hipGraphNode_t, nodeParams:c.POINTER[hipExternalSemaphoreSignalNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExternalSemaphoresWaitNodeSetParams(hNode:hipGraphNode_t, nodeParams:c.POINTER[hipExternalSemaphoreWaitNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExternalSemaphoresSignalNodeGetParams(hNode:hipGraphNode_t, params_out:c.POINTER[hipExternalSemaphoreSignalNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExternalSemaphoresWaitNodeGetParams(hNode:hipGraphNode_t, params_out:c.POINTER[hipExternalSemaphoreWaitNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:c.POINTER[hipExternalSemaphoreSignalNodeParams]) -> hipError_t: ...
@dll.bind
def hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:c.POINTER[hipExternalSemaphoreWaitNodeParams]) -> hipError_t: ...
@dll.bind
def hipDrvGraphAddMemsetNode(phGraphNode:c.POINTER[hipGraphNode_t], hGraph:hipGraph_t, dependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, memsetParams:c.POINTER[HIP_MEMSET_NODE_PARAMS], ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipMemAddressFree(devPtr:ctypes.c_void_p, size:size_t) -> hipError_t: ...
@dll.bind
def hipMemAddressReserve(ptr:c.POINTER[ctypes.c_void_p], size:size_t, alignment:size_t, addr:ctypes.c_void_p, flags:Annotated[int, ctypes.c_uint64]) -> hipError_t: ...
@dll.bind
def hipMemCreate(handle:c.POINTER[hipMemGenericAllocationHandle_t], size:size_t, prop:c.POINTER[hipMemAllocationProp], flags:Annotated[int, ctypes.c_uint64]) -> hipError_t: ...
@dll.bind
def hipMemExportToShareableHandle(shareableHandle:ctypes.c_void_p, handle:hipMemGenericAllocationHandle_t, handleType:hipMemAllocationHandleType, flags:Annotated[int, ctypes.c_uint64]) -> hipError_t: ...
@dll.bind
def hipMemGetAccess(flags:c.POINTER[Annotated[int, ctypes.c_uint64]], location:c.POINTER[hipMemLocation], ptr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipMemGetAllocationGranularity(granularity:c.POINTER[size_t], prop:c.POINTER[hipMemAllocationProp], option:hipMemAllocationGranularity_flags) -> hipError_t: ...
@dll.bind
def hipMemGetAllocationPropertiesFromHandle(prop:c.POINTER[hipMemAllocationProp], handle:hipMemGenericAllocationHandle_t) -> hipError_t: ...
@dll.bind
def hipMemImportFromShareableHandle(handle:c.POINTER[hipMemGenericAllocationHandle_t], osHandle:ctypes.c_void_p, shHandleType:hipMemAllocationHandleType) -> hipError_t: ...
@dll.bind
def hipMemMap(ptr:ctypes.c_void_p, size:size_t, offset:size_t, handle:hipMemGenericAllocationHandle_t, flags:Annotated[int, ctypes.c_uint64]) -> hipError_t: ...
@dll.bind
def hipMemMapArrayAsync(mapInfoList:c.POINTER[hipArrayMapInfo], count:Annotated[int, ctypes.c_uint32], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemRelease(handle:hipMemGenericAllocationHandle_t) -> hipError_t: ...
@dll.bind
def hipMemRetainAllocationHandle(handle:c.POINTER[hipMemGenericAllocationHandle_t], addr:ctypes.c_void_p) -> hipError_t: ...
@dll.bind
def hipMemSetAccess(ptr:ctypes.c_void_p, size:size_t, desc:c.POINTER[hipMemAccessDesc], count:size_t) -> hipError_t: ...
@dll.bind
def hipMemUnmap(ptr:ctypes.c_void_p, size:size_t) -> hipError_t: ...
@dll.bind
def hipGraphicsMapResources(count:Annotated[int, ctypes.c_int32], resources:c.POINTER[hipGraphicsResource_t], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipGraphicsSubResourceGetMappedArray(array:c.POINTER[hipArray_t], resource:hipGraphicsResource_t, arrayIndex:Annotated[int, ctypes.c_uint32], mipLevel:Annotated[int, ctypes.c_uint32]) -> hipError_t: ...
@dll.bind
def hipGraphicsResourceGetMappedPointer(devPtr:c.POINTER[ctypes.c_void_p], size:c.POINTER[size_t], resource:hipGraphicsResource_t) -> hipError_t: ...
@dll.bind
def hipGraphicsUnmapResources(count:Annotated[int, ctypes.c_int32], resources:c.POINTER[hipGraphicsResource_t], stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipGraphicsUnregisterResource(resource:hipGraphicsResource_t) -> hipError_t: ...
class __hip_surface(ctypes.Structure): pass
hipSurfaceObject_t: TypeAlias = c.POINTER[__hip_surface]
@dll.bind
def hipCreateSurfaceObject(pSurfObject:c.POINTER[hipSurfaceObject_t], pResDesc:c.POINTER[hipResourceDesc]) -> hipError_t: ...
@dll.bind
def hipDestroySurfaceObject(surfaceObject:hipSurfaceObject_t) -> hipError_t: ...
hipmipmappedArray: TypeAlias = c.POINTER[hipMipmappedArray]
hipResourcetype: TypeAlias = HIPresourcetype_enum
c.init_records()
hipGetDeviceProperties = hipGetDevicePropertiesR0600 # type: ignore
hipDeviceProp_t = hipDeviceProp_tR0600 # type: ignore
hipChooseDevice = hipChooseDeviceR0600 # type: ignore
GENERIC_GRID_LAUNCH = 1 # type: ignore
DEPRECATED = lambda msg: __attribute__ ((deprecated(msg))) # type: ignore
hipIpcMemLazyEnablePeerAccess = 0x01 # type: ignore
HIP_IPC_HANDLE_SIZE = 64 # type: ignore
hipStreamDefault = 0x00 # type: ignore
hipStreamNonBlocking = 0x01 # type: ignore
hipEventDefault = 0x0 # type: ignore
hipEventBlockingSync = 0x1 # type: ignore
hipEventDisableTiming = 0x2 # type: ignore
hipEventInterprocess = 0x4 # type: ignore
hipEventDisableSystemFence = 0x20000000 # type: ignore
hipEventReleaseToDevice = 0x40000000 # type: ignore
hipEventReleaseToSystem = 0x80000000 # type: ignore
hipHostMallocDefault = 0x0 # type: ignore
hipHostMallocPortable = 0x1 # type: ignore
hipHostMallocMapped = 0x2 # type: ignore
hipHostMallocWriteCombined = 0x4 # type: ignore
hipHostMallocNumaUser = 0x20000000 # type: ignore
hipHostMallocCoherent = 0x40000000 # type: ignore
hipHostMallocNonCoherent = 0x80000000 # type: ignore
hipMemAttachGlobal = 0x01 # type: ignore
hipMemAttachHost = 0x02 # type: ignore
hipMemAttachSingle = 0x04 # type: ignore
hipDeviceMallocDefault = 0x0 # type: ignore
hipDeviceMallocFinegrained = 0x1 # type: ignore
hipMallocSignalMemory = 0x2 # type: ignore
hipDeviceMallocUncached = 0x3 # type: ignore
hipDeviceMallocContiguous = 0x4 # type: ignore
hipHostRegisterDefault = 0x0 # type: ignore
hipHostRegisterPortable = 0x1 # type: ignore
hipHostRegisterMapped = 0x2 # type: ignore
hipHostRegisterIoMemory = 0x4 # type: ignore
hipHostRegisterReadOnly = 0x08 # type: ignore
hipExtHostRegisterCoarseGrained = 0x8 # type: ignore
hipDeviceScheduleAuto = 0x0 # type: ignore
hipDeviceScheduleSpin = 0x1 # type: ignore
hipDeviceScheduleYield = 0x2 # type: ignore
hipDeviceScheduleBlockingSync = 0x4 # type: ignore
hipDeviceScheduleMask = 0x7 # type: ignore
hipDeviceMapHost = 0x8 # type: ignore
hipDeviceLmemResizeToMax = 0x10 # type: ignore
hipArrayDefault = 0x00 # type: ignore
hipArrayLayered = 0x01 # type: ignore
hipArraySurfaceLoadStore = 0x02 # type: ignore
hipArrayCubemap = 0x04 # type: ignore
hipArrayTextureGather = 0x08 # type: ignore
hipOccupancyDefault = 0x00 # type: ignore
hipOccupancyDisableCachingOverride = 0x01 # type: ignore
hipCooperativeLaunchMultiDeviceNoPreSync = 0x01 # type: ignore
hipCooperativeLaunchMultiDeviceNoPostSync = 0x02 # type: ignore
hipExtAnyOrderLaunch = 0x01 # type: ignore
hipStreamWaitValueGte = 0x0 # type: ignore
hipStreamWaitValueEq = 0x1 # type: ignore
hipStreamWaitValueAnd = 0x2 # type: ignore
hipStreamWaitValueNor = 0x3 # type: ignore
hipExternalMemoryDedicated = 0x1 # type: ignore
hipKernelNodeAttrID = hipLaunchAttributeID # type: ignore
hipKernelNodeAttributeAccessPolicyWindow = hipLaunchAttributeAccessPolicyWindow # type: ignore
hipKernelNodeAttributeCooperative = hipLaunchAttributeCooperative # type: ignore
hipKernelNodeAttributePriority = hipLaunchAttributePriority # type: ignore
hipKernelNodeAttrValue = hipLaunchAttributeValue # type: ignore
hipGraphKernelNodePortDefault = 0 # type: ignore
hipGraphKernelNodePortLaunchCompletion = 2 # type: ignore
hipGraphKernelNodePortProgrammatic = 1 # type: ignore
USE_PEER_NON_UNIFIED = 1 # type: ignore
HIP_TRSA_OVERRIDE_FORMAT = 0x01 # type: ignore
HIP_TRSF_READ_AS_INTEGER = 0x01 # type: ignore
HIP_TRSF_NORMALIZED_COORDINATES = 0x02 # type: ignore
HIP_TRSF_SRGB = 0x10 # type: ignore