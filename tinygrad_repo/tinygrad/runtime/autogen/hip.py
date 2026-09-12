# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
import os
dll = c.DLL('hip', os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so')
hipError_t: dict[int, str] = {(hipSuccess:=0): 'hipSuccess', (hipErrorInvalidValue:=1): 'hipErrorInvalidValue', (hipErrorOutOfMemory:=2): 'hipErrorOutOfMemory', (hipErrorMemoryAllocation:=2): 'hipErrorMemoryAllocation', (hipErrorNotInitialized:=3): 'hipErrorNotInitialized', (hipErrorInitializationError:=3): 'hipErrorInitializationError', (hipErrorDeinitialized:=4): 'hipErrorDeinitialized', (hipErrorProfilerDisabled:=5): 'hipErrorProfilerDisabled', (hipErrorProfilerNotInitialized:=6): 'hipErrorProfilerNotInitialized', (hipErrorProfilerAlreadyStarted:=7): 'hipErrorProfilerAlreadyStarted', (hipErrorProfilerAlreadyStopped:=8): 'hipErrorProfilerAlreadyStopped', (hipErrorInvalidConfiguration:=9): 'hipErrorInvalidConfiguration', (hipErrorInvalidPitchValue:=12): 'hipErrorInvalidPitchValue', (hipErrorInvalidSymbol:=13): 'hipErrorInvalidSymbol', (hipErrorInvalidDevicePointer:=17): 'hipErrorInvalidDevicePointer', (hipErrorInvalidMemcpyDirection:=21): 'hipErrorInvalidMemcpyDirection', (hipErrorInsufficientDriver:=35): 'hipErrorInsufficientDriver', (hipErrorMissingConfiguration:=52): 'hipErrorMissingConfiguration', (hipErrorPriorLaunchFailure:=53): 'hipErrorPriorLaunchFailure', (hipErrorInvalidDeviceFunction:=98): 'hipErrorInvalidDeviceFunction', (hipErrorNoDevice:=100): 'hipErrorNoDevice', (hipErrorInvalidDevice:=101): 'hipErrorInvalidDevice', (hipErrorInvalidImage:=200): 'hipErrorInvalidImage', (hipErrorInvalidContext:=201): 'hipErrorInvalidContext', (hipErrorContextAlreadyCurrent:=202): 'hipErrorContextAlreadyCurrent', (hipErrorMapFailed:=205): 'hipErrorMapFailed', (hipErrorMapBufferObjectFailed:=205): 'hipErrorMapBufferObjectFailed', (hipErrorUnmapFailed:=206): 'hipErrorUnmapFailed', (hipErrorArrayIsMapped:=207): 'hipErrorArrayIsMapped', (hipErrorAlreadyMapped:=208): 'hipErrorAlreadyMapped', (hipErrorNoBinaryForGpu:=209): 'hipErrorNoBinaryForGpu', (hipErrorAlreadyAcquired:=210): 'hipErrorAlreadyAcquired', (hipErrorNotMapped:=211): 'hipErrorNotMapped', (hipErrorNotMappedAsArray:=212): 'hipErrorNotMappedAsArray', (hipErrorNotMappedAsPointer:=213): 'hipErrorNotMappedAsPointer', (hipErrorECCNotCorrectable:=214): 'hipErrorECCNotCorrectable', (hipErrorUnsupportedLimit:=215): 'hipErrorUnsupportedLimit', (hipErrorContextAlreadyInUse:=216): 'hipErrorContextAlreadyInUse', (hipErrorPeerAccessUnsupported:=217): 'hipErrorPeerAccessUnsupported', (hipErrorInvalidKernelFile:=218): 'hipErrorInvalidKernelFile', (hipErrorInvalidGraphicsContext:=219): 'hipErrorInvalidGraphicsContext', (hipErrorInvalidSource:=300): 'hipErrorInvalidSource', (hipErrorFileNotFound:=301): 'hipErrorFileNotFound', (hipErrorSharedObjectSymbolNotFound:=302): 'hipErrorSharedObjectSymbolNotFound', (hipErrorSharedObjectInitFailed:=303): 'hipErrorSharedObjectInitFailed', (hipErrorOperatingSystem:=304): 'hipErrorOperatingSystem', (hipErrorInvalidHandle:=400): 'hipErrorInvalidHandle', (hipErrorInvalidResourceHandle:=400): 'hipErrorInvalidResourceHandle', (hipErrorIllegalState:=401): 'hipErrorIllegalState', (hipErrorNotFound:=500): 'hipErrorNotFound', (hipErrorNotReady:=600): 'hipErrorNotReady', (hipErrorIllegalAddress:=700): 'hipErrorIllegalAddress', (hipErrorLaunchOutOfResources:=701): 'hipErrorLaunchOutOfResources', (hipErrorLaunchTimeOut:=702): 'hipErrorLaunchTimeOut', (hipErrorPeerAccessAlreadyEnabled:=704): 'hipErrorPeerAccessAlreadyEnabled', (hipErrorPeerAccessNotEnabled:=705): 'hipErrorPeerAccessNotEnabled', (hipErrorSetOnActiveProcess:=708): 'hipErrorSetOnActiveProcess', (hipErrorContextIsDestroyed:=709): 'hipErrorContextIsDestroyed', (hipErrorAssert:=710): 'hipErrorAssert', (hipErrorHostMemoryAlreadyRegistered:=712): 'hipErrorHostMemoryAlreadyRegistered', (hipErrorHostMemoryNotRegistered:=713): 'hipErrorHostMemoryNotRegistered', (hipErrorLaunchFailure:=719): 'hipErrorLaunchFailure', (hipErrorCooperativeLaunchTooLarge:=720): 'hipErrorCooperativeLaunchTooLarge', (hipErrorNotSupported:=801): 'hipErrorNotSupported', (hipErrorStreamCaptureUnsupported:=900): 'hipErrorStreamCaptureUnsupported', (hipErrorStreamCaptureInvalidated:=901): 'hipErrorStreamCaptureInvalidated', (hipErrorStreamCaptureMerge:=902): 'hipErrorStreamCaptureMerge', (hipErrorStreamCaptureUnmatched:=903): 'hipErrorStreamCaptureUnmatched', (hipErrorStreamCaptureUnjoined:=904): 'hipErrorStreamCaptureUnjoined', (hipErrorStreamCaptureIsolation:=905): 'hipErrorStreamCaptureIsolation', (hipErrorStreamCaptureImplicit:=906): 'hipErrorStreamCaptureImplicit', (hipErrorCapturedEvent:=907): 'hipErrorCapturedEvent', (hipErrorStreamCaptureWrongThread:=908): 'hipErrorStreamCaptureWrongThread', (hipErrorGraphExecUpdateFailure:=910): 'hipErrorGraphExecUpdateFailure', (hipErrorInvalidChannelDescriptor:=911): 'hipErrorInvalidChannelDescriptor', (hipErrorInvalidTexture:=912): 'hipErrorInvalidTexture', (hipErrorUnknown:=999): 'hipErrorUnknown', (hipErrorRuntimeMemory:=1052): 'hipErrorRuntimeMemory', (hipErrorRuntimeOther:=1053): 'hipErrorRuntimeOther', (hipErrorTbd:=1054): 'hipErrorTbd'}
class ihipModuleSymbol_t(c.Struct): pass
hipFunction_t: TypeAlias = c.POINTER[ihipModuleSymbol_t]
uint32_t: TypeAlias = ctypes.c_uint32
size_t: TypeAlias = ctypes.c_uint64
class ihipStream_t(c.Struct): pass
hipStream_t: TypeAlias = c.POINTER[ihipStream_t]
class ihipEvent_t(c.Struct): pass
hipEvent_t: TypeAlias = c.POINTER[ihipEvent_t]
@dll.bind(ctypes.c_uint32, hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, c.POINTER[ctypes.c_void_p], c.POINTER[ctypes.c_void_p], hipEvent_t, hipEvent_t, uint32_t)
def hipExtModuleLaunchKernel(f:hipFunction_t, globalWorkSizeX:uint32_t, globalWorkSizeY:uint32_t, globalWorkSizeZ:uint32_t, localWorkSizeX:uint32_t, localWorkSizeY:uint32_t, localWorkSizeZ:uint32_t, sharedMemBytes:size_t, hStream:hipStream_t, kernelParams:c.POINTER[ctypes.c_void_p], extra:c.POINTER[ctypes.c_void_p], startEvent:hipEvent_t, stopEvent:hipEvent_t, flags:uint32_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, c.POINTER[ctypes.c_void_p], c.POINTER[ctypes.c_void_p], hipEvent_t, hipEvent_t)
def hipHccModuleLaunchKernel(f:hipFunction_t, globalWorkSizeX:uint32_t, globalWorkSizeY:uint32_t, globalWorkSizeZ:uint32_t, localWorkSizeX:uint32_t, localWorkSizeY:uint32_t, localWorkSizeZ:uint32_t, sharedMemBytes:size_t, hStream:hipStream_t, kernelParams:c.POINTER[ctypes.c_void_p], extra:c.POINTER[ctypes.c_void_p], startEvent:hipEvent_t, stopEvent:hipEvent_t) -> ctypes.c_uint32: ...
@c.record
class dim3(c.Struct):
  SIZE = 12
  x: int
  y: int
  z: int
dim3.register_fields([('x', uint32_t, 0), ('y', uint32_t, 4), ('z', uint32_t, 8)])
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, dim3, dim3, c.POINTER[ctypes.c_void_p], size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int32)
def hipExtLaunchKernel(function_address:ctypes.c_void_p, numBlocks:dim3, dimBlocks:dim3, args:c.POINTER[ctypes.c_void_p], sharedMemBytes:size_t, stream:hipStream_t, startEvent:hipEvent_t, stopEvent:hipEvent_t, flags:int) -> ctypes.c_uint32: ...
hiprtcResult: dict[int, str] = {(HIPRTC_SUCCESS:=0): 'HIPRTC_SUCCESS', (HIPRTC_ERROR_OUT_OF_MEMORY:=1): 'HIPRTC_ERROR_OUT_OF_MEMORY', (HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:=2): 'HIPRTC_ERROR_PROGRAM_CREATION_FAILURE', (HIPRTC_ERROR_INVALID_INPUT:=3): 'HIPRTC_ERROR_INVALID_INPUT', (HIPRTC_ERROR_INVALID_PROGRAM:=4): 'HIPRTC_ERROR_INVALID_PROGRAM', (HIPRTC_ERROR_INVALID_OPTION:=5): 'HIPRTC_ERROR_INVALID_OPTION', (HIPRTC_ERROR_COMPILATION:=6): 'HIPRTC_ERROR_COMPILATION', (HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:=7): 'HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE', (HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:=8): 'HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION', (HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:=9): 'HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION', (HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:=10): 'HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID', (HIPRTC_ERROR_INTERNAL_ERROR:=11): 'HIPRTC_ERROR_INTERNAL_ERROR', (HIPRTC_ERROR_LINKING:=100): 'HIPRTC_ERROR_LINKING'}
class ihiprtcLinkState(c.Struct): pass
hiprtcLinkState: TypeAlias = c.POINTER[ihiprtcLinkState]
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_uint32)
def hiprtcGetErrorString(result:ctypes.c_uint32) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[ctypes.c_int32])
def hiprtcVersion(major:c.POINTER[ctypes.c_int32], minor:c.POINTER[ctypes.c_int32]) -> ctypes.c_uint32: ...
class _hiprtcProgram(c.Struct): pass
hiprtcProgram: TypeAlias = c.POINTER[_hiprtcProgram]
@dll.bind(ctypes.c_uint32, hiprtcProgram, c.POINTER[ctypes.c_char])
def hiprtcAddNameExpression(prog:hiprtcProgram, name_expression:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.c_int32, c.POINTER[c.POINTER[ctypes.c_char]])
def hiprtcCompileProgram(prog:hiprtcProgram, numOptions:int, options:c.POINTER[c.POINTER[ctypes.c_char]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hiprtcProgram], c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char], ctypes.c_int32, c.POINTER[c.POINTER[ctypes.c_char]], c.POINTER[c.POINTER[ctypes.c_char]])
def hiprtcCreateProgram(prog:c.POINTER[hiprtcProgram], src:c.POINTER[ctypes.c_char], name:c.POINTER[ctypes.c_char], numHeaders:int, headers:c.POINTER[c.POINTER[ctypes.c_char]], includeNames:c.POINTER[c.POINTER[ctypes.c_char]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hiprtcProgram])
def hiprtcDestroyProgram(prog:c.POINTER[hiprtcProgram]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, c.POINTER[ctypes.c_char], c.POINTER[c.POINTER[ctypes.c_char]])
def hiprtcGetLoweredName(prog:hiprtcProgram, name_expression:c.POINTER[ctypes.c_char], lowered_name:c.POINTER[c.POINTER[ctypes.c_char]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, c.POINTER[ctypes.c_char])
def hiprtcGetProgramLog(prog:hiprtcProgram, log:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, c.POINTER[size_t])
def hiprtcGetProgramLogSize(prog:hiprtcProgram, logSizeRet:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, c.POINTER[ctypes.c_char])
def hiprtcGetCode(prog:hiprtcProgram, code:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, c.POINTER[size_t])
def hiprtcGetCodeSize(prog:hiprtcProgram, codeSizeRet:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, c.POINTER[ctypes.c_char])
def hiprtcGetBitcode(prog:hiprtcProgram, bitcode:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, c.POINTER[size_t])
def hiprtcGetBitcodeSize(prog:hiprtcProgram, bitcode_size:c.POINTER[size_t]) -> ctypes.c_uint32: ...
hipJitOption: dict[int, str] = {(hipJitOptionMaxRegisters:=0): 'hipJitOptionMaxRegisters', (hipJitOptionThreadsPerBlock:=1): 'hipJitOptionThreadsPerBlock', (hipJitOptionWallTime:=2): 'hipJitOptionWallTime', (hipJitOptionInfoLogBuffer:=3): 'hipJitOptionInfoLogBuffer', (hipJitOptionInfoLogBufferSizeBytes:=4): 'hipJitOptionInfoLogBufferSizeBytes', (hipJitOptionErrorLogBuffer:=5): 'hipJitOptionErrorLogBuffer', (hipJitOptionErrorLogBufferSizeBytes:=6): 'hipJitOptionErrorLogBufferSizeBytes', (hipJitOptionOptimizationLevel:=7): 'hipJitOptionOptimizationLevel', (hipJitOptionTargetFromContext:=8): 'hipJitOptionTargetFromContext', (hipJitOptionTarget:=9): 'hipJitOptionTarget', (hipJitOptionFallbackStrategy:=10): 'hipJitOptionFallbackStrategy', (hipJitOptionGenerateDebugInfo:=11): 'hipJitOptionGenerateDebugInfo', (hipJitOptionLogVerbose:=12): 'hipJitOptionLogVerbose', (hipJitOptionGenerateLineInfo:=13): 'hipJitOptionGenerateLineInfo', (hipJitOptionCacheMode:=14): 'hipJitOptionCacheMode', (hipJitOptionSm3xOpt:=15): 'hipJitOptionSm3xOpt', (hipJitOptionFastCompile:=16): 'hipJitOptionFastCompile', (hipJitOptionGlobalSymbolNames:=17): 'hipJitOptionGlobalSymbolNames', (hipJitOptionGlobalSymbolAddresses:=18): 'hipJitOptionGlobalSymbolAddresses', (hipJitOptionGlobalSymbolCount:=19): 'hipJitOptionGlobalSymbolCount', (hipJitOptionLto:=20): 'hipJitOptionLto', (hipJitOptionFtz:=21): 'hipJitOptionFtz', (hipJitOptionPrecDiv:=22): 'hipJitOptionPrecDiv', (hipJitOptionPrecSqrt:=23): 'hipJitOptionPrecSqrt', (hipJitOptionFma:=24): 'hipJitOptionFma', (hipJitOptionPositionIndependentCode:=25): 'hipJitOptionPositionIndependentCode', (hipJitOptionMinCTAPerSM:=26): 'hipJitOptionMinCTAPerSM', (hipJitOptionMaxThreadsPerBlock:=27): 'hipJitOptionMaxThreadsPerBlock', (hipJitOptionOverrideDirectiveValues:=28): 'hipJitOptionOverrideDirectiveValues', (hipJitOptionNumOptions:=29): 'hipJitOptionNumOptions', (hipJitOptionIRtoISAOptExt:=10000): 'hipJitOptionIRtoISAOptExt', (hipJitOptionIRtoISAOptCountExt:=10001): 'hipJitOptionIRtoISAOptCountExt'}
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_void_p], c.POINTER[hiprtcLinkState])
def hiprtcLinkCreate(num_options:int, option_ptr:c.POINTER[ctypes.c_uint32], option_vals_pptr:c.POINTER[ctypes.c_void_p], hip_link_state_ptr:c.POINTER[hiprtcLinkState]) -> ctypes.c_uint32: ...
hipJitInputType: dict[int, str] = {(hipJitInputCubin:=0): 'hipJitInputCubin', (hipJitInputPtx:=1): 'hipJitInputPtx', (hipJitInputFatBinary:=2): 'hipJitInputFatBinary', (hipJitInputObject:=3): 'hipJitInputObject', (hipJitInputLibrary:=4): 'hipJitInputLibrary', (hipJitInputNvvm:=5): 'hipJitInputNvvm', (hipJitNumLegacyInputTypes:=6): 'hipJitNumLegacyInputTypes', (hipJitInputLLVMBitcode:=100): 'hipJitInputLLVMBitcode', (hipJitInputLLVMBundledBitcode:=101): 'hipJitInputLLVMBundledBitcode', (hipJitInputLLVMArchivesOfBundledBitcode:=102): 'hipJitInputLLVMArchivesOfBundledBitcode', (hipJitInputSpirv:=103): 'hipJitInputSpirv', (hipJitNumInputTypes:=10): 'hipJitNumInputTypes'}
@dll.bind(ctypes.c_uint32, hiprtcLinkState, ctypes.c_uint32, c.POINTER[ctypes.c_char], ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_void_p])
def hiprtcLinkAddFile(hip_link_state:hiprtcLinkState, input_type:ctypes.c_uint32, file_path:c.POINTER[ctypes.c_char], num_options:int, options_ptr:c.POINTER[ctypes.c_uint32], option_values:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcLinkState, ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[ctypes.c_char], ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_void_p])
def hiprtcLinkAddData(hip_link_state:hiprtcLinkState, input_type:ctypes.c_uint32, image:ctypes.c_void_p, image_size:size_t, name:c.POINTER[ctypes.c_char], num_options:int, options_ptr:c.POINTER[ctypes.c_uint32], option_values:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcLinkState, c.POINTER[ctypes.c_void_p], c.POINTER[size_t])
def hiprtcLinkComplete(hip_link_state:hiprtcLinkState, bin_out:c.POINTER[ctypes.c_void_p], size_out:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcLinkState)
def hiprtcLinkDestroy(hip_link_state:hiprtcLinkState) -> ctypes.c_uint32: ...
_anonenum0: dict[int, str] = {(HIP_SUCCESS:=0): 'HIP_SUCCESS', (HIP_ERROR_INVALID_VALUE:=1): 'HIP_ERROR_INVALID_VALUE', (HIP_ERROR_NOT_INITIALIZED:=2): 'HIP_ERROR_NOT_INITIALIZED', (HIP_ERROR_LAUNCH_OUT_OF_RESOURCES:=3): 'HIP_ERROR_LAUNCH_OUT_OF_RESOURCES'}
@c.record
class hipDeviceArch_t(c.Struct):
  SIZE = 4
  hasGlobalInt32Atomics: int
  hasGlobalFloatAtomicExch: int
  hasSharedInt32Atomics: int
  hasSharedFloatAtomicExch: int
  hasFloatAtomicAdd: int
  hasGlobalInt64Atomics: int
  hasSharedInt64Atomics: int
  hasDoubles: int
  hasWarpVote: int
  hasWarpBallot: int
  hasWarpShuffle: int
  hasFunnelShift: int
  hasThreadFenceSystem: int
  hasSyncThreadsExt: int
  hasSurfaceFuncs: int
  has3dGrid: int
  hasDynamicParallelism: int
hipDeviceArch_t.register_fields([('hasGlobalInt32Atomics', ctypes.c_uint32, 0, 1, 0), ('hasGlobalFloatAtomicExch', ctypes.c_uint32, 0, 1, 1), ('hasSharedInt32Atomics', ctypes.c_uint32, 0, 1, 2), ('hasSharedFloatAtomicExch', ctypes.c_uint32, 0, 1, 3), ('hasFloatAtomicAdd', ctypes.c_uint32, 0, 1, 4), ('hasGlobalInt64Atomics', ctypes.c_uint32, 0, 1, 5), ('hasSharedInt64Atomics', ctypes.c_uint32, 0, 1, 6), ('hasDoubles', ctypes.c_uint32, 0, 1, 7), ('hasWarpVote', ctypes.c_uint32, 1, 1, 0), ('hasWarpBallot', ctypes.c_uint32, 1, 1, 1), ('hasWarpShuffle', ctypes.c_uint32, 1, 1, 2), ('hasFunnelShift', ctypes.c_uint32, 1, 1, 3), ('hasThreadFenceSystem', ctypes.c_uint32, 1, 1, 4), ('hasSyncThreadsExt', ctypes.c_uint32, 1, 1, 5), ('hasSurfaceFuncs', ctypes.c_uint32, 1, 1, 6), ('has3dGrid', ctypes.c_uint32, 1, 1, 7), ('hasDynamicParallelism', ctypes.c_uint32, 2, 1, 0)])
@c.record
class hipUUID_t(c.Struct):
  SIZE = 16
  bytes: c.Array[ctypes.c_char, Literal[16]]
hipUUID_t.register_fields([('bytes', c.Array[ctypes.c_char, Literal[16]], 0)])
hipUUID: TypeAlias = hipUUID_t
@c.record
class hipDeviceProp_tR0600(c.Struct):
  SIZE = 1472
  name: c.Array[ctypes.c_char, Literal[256]]
  uuid: hipUUID_t
  luid: c.Array[ctypes.c_char, Literal[8]]
  luidDeviceNodeMask: int
  totalGlobalMem: int
  sharedMemPerBlock: int
  regsPerBlock: int
  warpSize: int
  memPitch: int
  maxThreadsPerBlock: int
  maxThreadsDim: c.Array[ctypes.c_int32, Literal[3]]
  maxGridSize: c.Array[ctypes.c_int32, Literal[3]]
  clockRate: int
  totalConstMem: int
  major: int
  minor: int
  textureAlignment: int
  texturePitchAlignment: int
  deviceOverlap: int
  multiProcessorCount: int
  kernelExecTimeoutEnabled: int
  integrated: int
  canMapHostMemory: int
  computeMode: int
  maxTexture1D: int
  maxTexture1DMipmap: int
  maxTexture1DLinear: int
  maxTexture2D: c.Array[ctypes.c_int32, Literal[2]]
  maxTexture2DMipmap: c.Array[ctypes.c_int32, Literal[2]]
  maxTexture2DLinear: c.Array[ctypes.c_int32, Literal[3]]
  maxTexture2DGather: c.Array[ctypes.c_int32, Literal[2]]
  maxTexture3D: c.Array[ctypes.c_int32, Literal[3]]
  maxTexture3DAlt: c.Array[ctypes.c_int32, Literal[3]]
  maxTextureCubemap: int
  maxTexture1DLayered: c.Array[ctypes.c_int32, Literal[2]]
  maxTexture2DLayered: c.Array[ctypes.c_int32, Literal[3]]
  maxTextureCubemapLayered: c.Array[ctypes.c_int32, Literal[2]]
  maxSurface1D: int
  maxSurface2D: c.Array[ctypes.c_int32, Literal[2]]
  maxSurface3D: c.Array[ctypes.c_int32, Literal[3]]
  maxSurface1DLayered: c.Array[ctypes.c_int32, Literal[2]]
  maxSurface2DLayered: c.Array[ctypes.c_int32, Literal[3]]
  maxSurfaceCubemap: int
  maxSurfaceCubemapLayered: c.Array[ctypes.c_int32, Literal[2]]
  surfaceAlignment: int
  concurrentKernels: int
  ECCEnabled: int
  pciBusID: int
  pciDeviceID: int
  pciDomainID: int
  tccDriver: int
  asyncEngineCount: int
  unifiedAddressing: int
  memoryClockRate: int
  memoryBusWidth: int
  l2CacheSize: int
  persistingL2CacheMaxSize: int
  maxThreadsPerMultiProcessor: int
  streamPrioritiesSupported: int
  globalL1CacheSupported: int
  localL1CacheSupported: int
  sharedMemPerMultiprocessor: int
  regsPerMultiprocessor: int
  managedMemory: int
  isMultiGpuBoard: int
  multiGpuBoardGroupID: int
  hostNativeAtomicSupported: int
  singleToDoublePrecisionPerfRatio: int
  pageableMemoryAccess: int
  concurrentManagedAccess: int
  computePreemptionSupported: int
  canUseHostPointerForRegisteredMem: int
  cooperativeLaunch: int
  cooperativeMultiDeviceLaunch: int
  sharedMemPerBlockOptin: int
  pageableMemoryAccessUsesHostPageTables: int
  directManagedMemAccessFromHost: int
  maxBlocksPerMultiProcessor: int
  accessPolicyMaxWindowSize: int
  reservedSharedMemPerBlock: int
  hostRegisterSupported: int
  sparseHipArraySupported: int
  hostRegisterReadOnlySupported: int
  timelineSemaphoreInteropSupported: int
  memoryPoolsSupported: int
  gpuDirectRDMASupported: int
  gpuDirectRDMAFlushWritesOptions: int
  gpuDirectRDMAWritesOrdering: int
  memoryPoolSupportedHandleTypes: int
  deferredMappingHipArraySupported: int
  ipcEventSupported: int
  clusterLaunch: int
  unifiedFunctionPointers: int
  reserved: c.Array[ctypes.c_int32, Literal[63]]
  hipReserved: c.Array[ctypes.c_int32, Literal[32]]
  gcnArchName: c.Array[ctypes.c_char, Literal[256]]
  maxSharedMemoryPerMultiProcessor: int
  clockInstructionRate: int
  arch: hipDeviceArch_t
  hdpMemFlushCntl: c.POINTER[ctypes.c_uint32]
  hdpRegFlushCntl: c.POINTER[ctypes.c_uint32]
  cooperativeMultiDeviceUnmatchedFunc: int
  cooperativeMultiDeviceUnmatchedGridDim: int
  cooperativeMultiDeviceUnmatchedBlockDim: int
  cooperativeMultiDeviceUnmatchedSharedMem: int
  isLargeBar: int
  asicRevision: int
hipDeviceProp_tR0600.register_fields([('name', c.Array[ctypes.c_char, Literal[256]], 0), ('uuid', hipUUID, 256), ('luid', c.Array[ctypes.c_char, Literal[8]], 272), ('luidDeviceNodeMask', ctypes.c_uint32, 280), ('totalGlobalMem', size_t, 288), ('sharedMemPerBlock', size_t, 296), ('regsPerBlock', ctypes.c_int32, 304), ('warpSize', ctypes.c_int32, 308), ('memPitch', size_t, 312), ('maxThreadsPerBlock', ctypes.c_int32, 320), ('maxThreadsDim', c.Array[ctypes.c_int32, Literal[3]], 324), ('maxGridSize', c.Array[ctypes.c_int32, Literal[3]], 336), ('clockRate', ctypes.c_int32, 348), ('totalConstMem', size_t, 352), ('major', ctypes.c_int32, 360), ('minor', ctypes.c_int32, 364), ('textureAlignment', size_t, 368), ('texturePitchAlignment', size_t, 376), ('deviceOverlap', ctypes.c_int32, 384), ('multiProcessorCount', ctypes.c_int32, 388), ('kernelExecTimeoutEnabled', ctypes.c_int32, 392), ('integrated', ctypes.c_int32, 396), ('canMapHostMemory', ctypes.c_int32, 400), ('computeMode', ctypes.c_int32, 404), ('maxTexture1D', ctypes.c_int32, 408), ('maxTexture1DMipmap', ctypes.c_int32, 412), ('maxTexture1DLinear', ctypes.c_int32, 416), ('maxTexture2D', c.Array[ctypes.c_int32, Literal[2]], 420), ('maxTexture2DMipmap', c.Array[ctypes.c_int32, Literal[2]], 428), ('maxTexture2DLinear', c.Array[ctypes.c_int32, Literal[3]], 436), ('maxTexture2DGather', c.Array[ctypes.c_int32, Literal[2]], 448), ('maxTexture3D', c.Array[ctypes.c_int32, Literal[3]], 456), ('maxTexture3DAlt', c.Array[ctypes.c_int32, Literal[3]], 468), ('maxTextureCubemap', ctypes.c_int32, 480), ('maxTexture1DLayered', c.Array[ctypes.c_int32, Literal[2]], 484), ('maxTexture2DLayered', c.Array[ctypes.c_int32, Literal[3]], 492), ('maxTextureCubemapLayered', c.Array[ctypes.c_int32, Literal[2]], 504), ('maxSurface1D', ctypes.c_int32, 512), ('maxSurface2D', c.Array[ctypes.c_int32, Literal[2]], 516), ('maxSurface3D', c.Array[ctypes.c_int32, Literal[3]], 524), ('maxSurface1DLayered', c.Array[ctypes.c_int32, Literal[2]], 536), ('maxSurface2DLayered', c.Array[ctypes.c_int32, Literal[3]], 544), ('maxSurfaceCubemap', ctypes.c_int32, 556), ('maxSurfaceCubemapLayered', c.Array[ctypes.c_int32, Literal[2]], 560), ('surfaceAlignment', size_t, 568), ('concurrentKernels', ctypes.c_int32, 576), ('ECCEnabled', ctypes.c_int32, 580), ('pciBusID', ctypes.c_int32, 584), ('pciDeviceID', ctypes.c_int32, 588), ('pciDomainID', ctypes.c_int32, 592), ('tccDriver', ctypes.c_int32, 596), ('asyncEngineCount', ctypes.c_int32, 600), ('unifiedAddressing', ctypes.c_int32, 604), ('memoryClockRate', ctypes.c_int32, 608), ('memoryBusWidth', ctypes.c_int32, 612), ('l2CacheSize', ctypes.c_int32, 616), ('persistingL2CacheMaxSize', ctypes.c_int32, 620), ('maxThreadsPerMultiProcessor', ctypes.c_int32, 624), ('streamPrioritiesSupported', ctypes.c_int32, 628), ('globalL1CacheSupported', ctypes.c_int32, 632), ('localL1CacheSupported', ctypes.c_int32, 636), ('sharedMemPerMultiprocessor', size_t, 640), ('regsPerMultiprocessor', ctypes.c_int32, 648), ('managedMemory', ctypes.c_int32, 652), ('isMultiGpuBoard', ctypes.c_int32, 656), ('multiGpuBoardGroupID', ctypes.c_int32, 660), ('hostNativeAtomicSupported', ctypes.c_int32, 664), ('singleToDoublePrecisionPerfRatio', ctypes.c_int32, 668), ('pageableMemoryAccess', ctypes.c_int32, 672), ('concurrentManagedAccess', ctypes.c_int32, 676), ('computePreemptionSupported', ctypes.c_int32, 680), ('canUseHostPointerForRegisteredMem', ctypes.c_int32, 684), ('cooperativeLaunch', ctypes.c_int32, 688), ('cooperativeMultiDeviceLaunch', ctypes.c_int32, 692), ('sharedMemPerBlockOptin', size_t, 696), ('pageableMemoryAccessUsesHostPageTables', ctypes.c_int32, 704), ('directManagedMemAccessFromHost', ctypes.c_int32, 708), ('maxBlocksPerMultiProcessor', ctypes.c_int32, 712), ('accessPolicyMaxWindowSize', ctypes.c_int32, 716), ('reservedSharedMemPerBlock', size_t, 720), ('hostRegisterSupported', ctypes.c_int32, 728), ('sparseHipArraySupported', ctypes.c_int32, 732), ('hostRegisterReadOnlySupported', ctypes.c_int32, 736), ('timelineSemaphoreInteropSupported', ctypes.c_int32, 740), ('memoryPoolsSupported', ctypes.c_int32, 744), ('gpuDirectRDMASupported', ctypes.c_int32, 748), ('gpuDirectRDMAFlushWritesOptions', ctypes.c_uint32, 752), ('gpuDirectRDMAWritesOrdering', ctypes.c_int32, 756), ('memoryPoolSupportedHandleTypes', ctypes.c_uint32, 760), ('deferredMappingHipArraySupported', ctypes.c_int32, 764), ('ipcEventSupported', ctypes.c_int32, 768), ('clusterLaunch', ctypes.c_int32, 772), ('unifiedFunctionPointers', ctypes.c_int32, 776), ('reserved', c.Array[ctypes.c_int32, Literal[63]], 780), ('hipReserved', c.Array[ctypes.c_int32, Literal[32]], 1032), ('gcnArchName', c.Array[ctypes.c_char, Literal[256]], 1160), ('maxSharedMemoryPerMultiProcessor', size_t, 1416), ('clockInstructionRate', ctypes.c_int32, 1424), ('arch', hipDeviceArch_t, 1428), ('hdpMemFlushCntl', c.POINTER[ctypes.c_uint32], 1432), ('hdpRegFlushCntl', c.POINTER[ctypes.c_uint32], 1440), ('cooperativeMultiDeviceUnmatchedFunc', ctypes.c_int32, 1448), ('cooperativeMultiDeviceUnmatchedGridDim', ctypes.c_int32, 1452), ('cooperativeMultiDeviceUnmatchedBlockDim', ctypes.c_int32, 1456), ('cooperativeMultiDeviceUnmatchedSharedMem', ctypes.c_int32, 1460), ('isLargeBar', ctypes.c_int32, 1464), ('asicRevision', ctypes.c_int32, 1468)])
hipMemoryType: dict[int, str] = {(hipMemoryTypeUnregistered:=0): 'hipMemoryTypeUnregistered', (hipMemoryTypeHost:=1): 'hipMemoryTypeHost', (hipMemoryTypeDevice:=2): 'hipMemoryTypeDevice', (hipMemoryTypeManaged:=3): 'hipMemoryTypeManaged', (hipMemoryTypeArray:=10): 'hipMemoryTypeArray', (hipMemoryTypeUnified:=11): 'hipMemoryTypeUnified'}
@c.record
class hipPointerAttribute_t(c.Struct):
  SIZE = 32
  type: int
  device: int
  devicePointer: ctypes.c_void_p
  hostPointer: ctypes.c_void_p
  isManaged: int
  allocationFlags: int
hipPointerAttribute_t.register_fields([('type', ctypes.c_uint32, 0), ('device', ctypes.c_int32, 4), ('devicePointer', ctypes.c_void_p, 8), ('hostPointer', ctypes.c_void_p, 16), ('isManaged', ctypes.c_int32, 24), ('allocationFlags', ctypes.c_uint32, 28)])
hipDeviceAttribute_t: dict[int, str] = {(hipDeviceAttributeCudaCompatibleBegin:=0): 'hipDeviceAttributeCudaCompatibleBegin', (hipDeviceAttributeEccEnabled:=0): 'hipDeviceAttributeEccEnabled', (hipDeviceAttributeAccessPolicyMaxWindowSize:=1): 'hipDeviceAttributeAccessPolicyMaxWindowSize', (hipDeviceAttributeAsyncEngineCount:=2): 'hipDeviceAttributeAsyncEngineCount', (hipDeviceAttributeCanMapHostMemory:=3): 'hipDeviceAttributeCanMapHostMemory', (hipDeviceAttributeCanUseHostPointerForRegisteredMem:=4): 'hipDeviceAttributeCanUseHostPointerForRegisteredMem', (hipDeviceAttributeClockRate:=5): 'hipDeviceAttributeClockRate', (hipDeviceAttributeComputeMode:=6): 'hipDeviceAttributeComputeMode', (hipDeviceAttributeComputePreemptionSupported:=7): 'hipDeviceAttributeComputePreemptionSupported', (hipDeviceAttributeConcurrentKernels:=8): 'hipDeviceAttributeConcurrentKernels', (hipDeviceAttributeConcurrentManagedAccess:=9): 'hipDeviceAttributeConcurrentManagedAccess', (hipDeviceAttributeCooperativeLaunch:=10): 'hipDeviceAttributeCooperativeLaunch', (hipDeviceAttributeCooperativeMultiDeviceLaunch:=11): 'hipDeviceAttributeCooperativeMultiDeviceLaunch', (hipDeviceAttributeDeviceOverlap:=12): 'hipDeviceAttributeDeviceOverlap', (hipDeviceAttributeDirectManagedMemAccessFromHost:=13): 'hipDeviceAttributeDirectManagedMemAccessFromHost', (hipDeviceAttributeGlobalL1CacheSupported:=14): 'hipDeviceAttributeGlobalL1CacheSupported', (hipDeviceAttributeHostNativeAtomicSupported:=15): 'hipDeviceAttributeHostNativeAtomicSupported', (hipDeviceAttributeIntegrated:=16): 'hipDeviceAttributeIntegrated', (hipDeviceAttributeIsMultiGpuBoard:=17): 'hipDeviceAttributeIsMultiGpuBoard', (hipDeviceAttributeKernelExecTimeout:=18): 'hipDeviceAttributeKernelExecTimeout', (hipDeviceAttributeL2CacheSize:=19): 'hipDeviceAttributeL2CacheSize', (hipDeviceAttributeLocalL1CacheSupported:=20): 'hipDeviceAttributeLocalL1CacheSupported', (hipDeviceAttributeLuid:=21): 'hipDeviceAttributeLuid', (hipDeviceAttributeLuidDeviceNodeMask:=22): 'hipDeviceAttributeLuidDeviceNodeMask', (hipDeviceAttributeComputeCapabilityMajor:=23): 'hipDeviceAttributeComputeCapabilityMajor', (hipDeviceAttributeManagedMemory:=24): 'hipDeviceAttributeManagedMemory', (hipDeviceAttributeMaxBlocksPerMultiProcessor:=25): 'hipDeviceAttributeMaxBlocksPerMultiProcessor', (hipDeviceAttributeMaxBlockDimX:=26): 'hipDeviceAttributeMaxBlockDimX', (hipDeviceAttributeMaxBlockDimY:=27): 'hipDeviceAttributeMaxBlockDimY', (hipDeviceAttributeMaxBlockDimZ:=28): 'hipDeviceAttributeMaxBlockDimZ', (hipDeviceAttributeMaxGridDimX:=29): 'hipDeviceAttributeMaxGridDimX', (hipDeviceAttributeMaxGridDimY:=30): 'hipDeviceAttributeMaxGridDimY', (hipDeviceAttributeMaxGridDimZ:=31): 'hipDeviceAttributeMaxGridDimZ', (hipDeviceAttributeMaxSurface1D:=32): 'hipDeviceAttributeMaxSurface1D', (hipDeviceAttributeMaxSurface1DLayered:=33): 'hipDeviceAttributeMaxSurface1DLayered', (hipDeviceAttributeMaxSurface2D:=34): 'hipDeviceAttributeMaxSurface2D', (hipDeviceAttributeMaxSurface2DLayered:=35): 'hipDeviceAttributeMaxSurface2DLayered', (hipDeviceAttributeMaxSurface3D:=36): 'hipDeviceAttributeMaxSurface3D', (hipDeviceAttributeMaxSurfaceCubemap:=37): 'hipDeviceAttributeMaxSurfaceCubemap', (hipDeviceAttributeMaxSurfaceCubemapLayered:=38): 'hipDeviceAttributeMaxSurfaceCubemapLayered', (hipDeviceAttributeMaxTexture1DWidth:=39): 'hipDeviceAttributeMaxTexture1DWidth', (hipDeviceAttributeMaxTexture1DLayered:=40): 'hipDeviceAttributeMaxTexture1DLayered', (hipDeviceAttributeMaxTexture1DLinear:=41): 'hipDeviceAttributeMaxTexture1DLinear', (hipDeviceAttributeMaxTexture1DMipmap:=42): 'hipDeviceAttributeMaxTexture1DMipmap', (hipDeviceAttributeMaxTexture2DWidth:=43): 'hipDeviceAttributeMaxTexture2DWidth', (hipDeviceAttributeMaxTexture2DHeight:=44): 'hipDeviceAttributeMaxTexture2DHeight', (hipDeviceAttributeMaxTexture2DGather:=45): 'hipDeviceAttributeMaxTexture2DGather', (hipDeviceAttributeMaxTexture2DLayered:=46): 'hipDeviceAttributeMaxTexture2DLayered', (hipDeviceAttributeMaxTexture2DLinear:=47): 'hipDeviceAttributeMaxTexture2DLinear', (hipDeviceAttributeMaxTexture2DMipmap:=48): 'hipDeviceAttributeMaxTexture2DMipmap', (hipDeviceAttributeMaxTexture3DWidth:=49): 'hipDeviceAttributeMaxTexture3DWidth', (hipDeviceAttributeMaxTexture3DHeight:=50): 'hipDeviceAttributeMaxTexture3DHeight', (hipDeviceAttributeMaxTexture3DDepth:=51): 'hipDeviceAttributeMaxTexture3DDepth', (hipDeviceAttributeMaxTexture3DAlt:=52): 'hipDeviceAttributeMaxTexture3DAlt', (hipDeviceAttributeMaxTextureCubemap:=53): 'hipDeviceAttributeMaxTextureCubemap', (hipDeviceAttributeMaxTextureCubemapLayered:=54): 'hipDeviceAttributeMaxTextureCubemapLayered', (hipDeviceAttributeMaxThreadsDim:=55): 'hipDeviceAttributeMaxThreadsDim', (hipDeviceAttributeMaxThreadsPerBlock:=56): 'hipDeviceAttributeMaxThreadsPerBlock', (hipDeviceAttributeMaxThreadsPerMultiProcessor:=57): 'hipDeviceAttributeMaxThreadsPerMultiProcessor', (hipDeviceAttributeMaxPitch:=58): 'hipDeviceAttributeMaxPitch', (hipDeviceAttributeMemoryBusWidth:=59): 'hipDeviceAttributeMemoryBusWidth', (hipDeviceAttributeMemoryClockRate:=60): 'hipDeviceAttributeMemoryClockRate', (hipDeviceAttributeComputeCapabilityMinor:=61): 'hipDeviceAttributeComputeCapabilityMinor', (hipDeviceAttributeMultiGpuBoardGroupID:=62): 'hipDeviceAttributeMultiGpuBoardGroupID', (hipDeviceAttributeMultiprocessorCount:=63): 'hipDeviceAttributeMultiprocessorCount', (hipDeviceAttributeUnused1:=64): 'hipDeviceAttributeUnused1', (hipDeviceAttributePageableMemoryAccess:=65): 'hipDeviceAttributePageableMemoryAccess', (hipDeviceAttributePageableMemoryAccessUsesHostPageTables:=66): 'hipDeviceAttributePageableMemoryAccessUsesHostPageTables', (hipDeviceAttributePciBusId:=67): 'hipDeviceAttributePciBusId', (hipDeviceAttributePciDeviceId:=68): 'hipDeviceAttributePciDeviceId', (hipDeviceAttributePciDomainId:=69): 'hipDeviceAttributePciDomainId', (hipDeviceAttributePciDomainID:=69): 'hipDeviceAttributePciDomainID', (hipDeviceAttributePersistingL2CacheMaxSize:=70): 'hipDeviceAttributePersistingL2CacheMaxSize', (hipDeviceAttributeMaxRegistersPerBlock:=71): 'hipDeviceAttributeMaxRegistersPerBlock', (hipDeviceAttributeMaxRegistersPerMultiprocessor:=72): 'hipDeviceAttributeMaxRegistersPerMultiprocessor', (hipDeviceAttributeReservedSharedMemPerBlock:=73): 'hipDeviceAttributeReservedSharedMemPerBlock', (hipDeviceAttributeMaxSharedMemoryPerBlock:=74): 'hipDeviceAttributeMaxSharedMemoryPerBlock', (hipDeviceAttributeSharedMemPerBlockOptin:=75): 'hipDeviceAttributeSharedMemPerBlockOptin', (hipDeviceAttributeSharedMemPerMultiprocessor:=76): 'hipDeviceAttributeSharedMemPerMultiprocessor', (hipDeviceAttributeSingleToDoublePrecisionPerfRatio:=77): 'hipDeviceAttributeSingleToDoublePrecisionPerfRatio', (hipDeviceAttributeStreamPrioritiesSupported:=78): 'hipDeviceAttributeStreamPrioritiesSupported', (hipDeviceAttributeSurfaceAlignment:=79): 'hipDeviceAttributeSurfaceAlignment', (hipDeviceAttributeTccDriver:=80): 'hipDeviceAttributeTccDriver', (hipDeviceAttributeTextureAlignment:=81): 'hipDeviceAttributeTextureAlignment', (hipDeviceAttributeTexturePitchAlignment:=82): 'hipDeviceAttributeTexturePitchAlignment', (hipDeviceAttributeTotalConstantMemory:=83): 'hipDeviceAttributeTotalConstantMemory', (hipDeviceAttributeTotalGlobalMem:=84): 'hipDeviceAttributeTotalGlobalMem', (hipDeviceAttributeUnifiedAddressing:=85): 'hipDeviceAttributeUnifiedAddressing', (hipDeviceAttributeUnused2:=86): 'hipDeviceAttributeUnused2', (hipDeviceAttributeWarpSize:=87): 'hipDeviceAttributeWarpSize', (hipDeviceAttributeMemoryPoolsSupported:=88): 'hipDeviceAttributeMemoryPoolsSupported', (hipDeviceAttributeVirtualMemoryManagementSupported:=89): 'hipDeviceAttributeVirtualMemoryManagementSupported', (hipDeviceAttributeHostRegisterSupported:=90): 'hipDeviceAttributeHostRegisterSupported', (hipDeviceAttributeMemoryPoolSupportedHandleTypes:=91): 'hipDeviceAttributeMemoryPoolSupportedHandleTypes', (hipDeviceAttributeCudaCompatibleEnd:=9999): 'hipDeviceAttributeCudaCompatibleEnd', (hipDeviceAttributeAmdSpecificBegin:=10000): 'hipDeviceAttributeAmdSpecificBegin', (hipDeviceAttributeClockInstructionRate:=10000): 'hipDeviceAttributeClockInstructionRate', (hipDeviceAttributeUnused3:=10001): 'hipDeviceAttributeUnused3', (hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:=10002): 'hipDeviceAttributeMaxSharedMemoryPerMultiprocessor', (hipDeviceAttributeUnused4:=10003): 'hipDeviceAttributeUnused4', (hipDeviceAttributeUnused5:=10004): 'hipDeviceAttributeUnused5', (hipDeviceAttributeHdpMemFlushCntl:=10005): 'hipDeviceAttributeHdpMemFlushCntl', (hipDeviceAttributeHdpRegFlushCntl:=10006): 'hipDeviceAttributeHdpRegFlushCntl', (hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:=10007): 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc', (hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:=10008): 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim', (hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:=10009): 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim', (hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:=10010): 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem', (hipDeviceAttributeIsLargeBar:=10011): 'hipDeviceAttributeIsLargeBar', (hipDeviceAttributeAsicRevision:=10012): 'hipDeviceAttributeAsicRevision', (hipDeviceAttributeCanUseStreamWaitValue:=10013): 'hipDeviceAttributeCanUseStreamWaitValue', (hipDeviceAttributeImageSupport:=10014): 'hipDeviceAttributeImageSupport', (hipDeviceAttributePhysicalMultiProcessorCount:=10015): 'hipDeviceAttributePhysicalMultiProcessorCount', (hipDeviceAttributeFineGrainSupport:=10016): 'hipDeviceAttributeFineGrainSupport', (hipDeviceAttributeWallClockRate:=10017): 'hipDeviceAttributeWallClockRate', (hipDeviceAttributeNumberOfXccs:=10018): 'hipDeviceAttributeNumberOfXccs', (hipDeviceAttributeMaxAvailableVgprsPerThread:=10019): 'hipDeviceAttributeMaxAvailableVgprsPerThread', (hipDeviceAttributePciChipId:=10020): 'hipDeviceAttributePciChipId', (hipDeviceAttributeAmdSpecificEnd:=19999): 'hipDeviceAttributeAmdSpecificEnd', (hipDeviceAttributeVendorSpecificBegin:=20000): 'hipDeviceAttributeVendorSpecificBegin'}
hipDriverProcAddressQueryResult: dict[int, str] = {(HIP_GET_PROC_ADDRESS_SUCCESS:=0): 'HIP_GET_PROC_ADDRESS_SUCCESS', (HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND:=1): 'HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND', (HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT:=2): 'HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT'}
hipComputeMode: dict[int, str] = {(hipComputeModeDefault:=0): 'hipComputeModeDefault', (hipComputeModeExclusive:=1): 'hipComputeModeExclusive', (hipComputeModeProhibited:=2): 'hipComputeModeProhibited', (hipComputeModeExclusiveProcess:=3): 'hipComputeModeExclusiveProcess'}
hipFlushGPUDirectRDMAWritesOptions: dict[int, str] = {(hipFlushGPUDirectRDMAWritesOptionHost:=1): 'hipFlushGPUDirectRDMAWritesOptionHost', (hipFlushGPUDirectRDMAWritesOptionMemOps:=2): 'hipFlushGPUDirectRDMAWritesOptionMemOps'}
hipGPUDirectRDMAWritesOrdering: dict[int, str] = {(hipGPUDirectRDMAWritesOrderingNone:=0): 'hipGPUDirectRDMAWritesOrderingNone', (hipGPUDirectRDMAWritesOrderingOwner:=100): 'hipGPUDirectRDMAWritesOrderingOwner', (hipGPUDirectRDMAWritesOrderingAllDevices:=200): 'hipGPUDirectRDMAWritesOrderingAllDevices'}
@dll.bind(ctypes.c_uint32)
def hip_init() -> ctypes.c_uint32: ...
class ihipCtx_t(c.Struct): pass
hipCtx_t: TypeAlias = c.POINTER[ihipCtx_t]
hipDevice_t: TypeAlias = ctypes.c_int32
hipDeviceP2PAttr: dict[int, str] = {(hipDevP2PAttrPerformanceRank:=0): 'hipDevP2PAttrPerformanceRank', (hipDevP2PAttrAccessSupported:=1): 'hipDevP2PAttrAccessSupported', (hipDevP2PAttrNativeAtomicSupported:=2): 'hipDevP2PAttrNativeAtomicSupported', (hipDevP2PAttrHipArrayAccessSupported:=3): 'hipDevP2PAttrHipArrayAccessSupported'}
hipDriverEntryPointQueryResult: dict[int, str] = {(hipDriverEntryPointSuccess:=0): 'hipDriverEntryPointSuccess', (hipDriverEntryPointSymbolNotFound:=1): 'hipDriverEntryPointSymbolNotFound', (hipDriverEntryPointVersionNotSufficent:=2): 'hipDriverEntryPointVersionNotSufficent'}
@c.record
class hipIpcMemHandle_st(c.Struct):
  SIZE = 64
  reserved: c.Array[ctypes.c_char, Literal[64]]
hipIpcMemHandle_st.register_fields([('reserved', c.Array[ctypes.c_char, Literal[64]], 0)])
hipIpcMemHandle_t: TypeAlias = hipIpcMemHandle_st
@c.record
class hipIpcEventHandle_st(c.Struct):
  SIZE = 64
  reserved: c.Array[ctypes.c_char, Literal[64]]
hipIpcEventHandle_st.register_fields([('reserved', c.Array[ctypes.c_char, Literal[64]], 0)])
hipIpcEventHandle_t: TypeAlias = hipIpcEventHandle_st
class ihipModule_t(c.Struct): pass
hipModule_t: TypeAlias = c.POINTER[ihipModule_t]
class ihipLinkState_t(c.Struct): pass
hipLinkState_t: TypeAlias = c.POINTER[ihipLinkState_t]
class ihipLibrary_t(c.Struct): pass
hipLibrary_t: TypeAlias = c.POINTER[ihipLibrary_t]
class ihipKernel_t(c.Struct): pass
hipKernel_t: TypeAlias = c.POINTER[ihipKernel_t]
class ihipMemPoolHandle_t(c.Struct): pass
hipMemPool_t: TypeAlias = c.POINTER[ihipMemPoolHandle_t]
@c.record
class hipFuncAttributes(c.Struct):
  SIZE = 56
  binaryVersion: int
  cacheModeCA: int
  constSizeBytes: int
  localSizeBytes: int
  maxDynamicSharedSizeBytes: int
  maxThreadsPerBlock: int
  numRegs: int
  preferredShmemCarveout: int
  ptxVersion: int
  sharedSizeBytes: int
hipFuncAttributes.register_fields([('binaryVersion', ctypes.c_int32, 0), ('cacheModeCA', ctypes.c_int32, 4), ('constSizeBytes', size_t, 8), ('localSizeBytes', size_t, 16), ('maxDynamicSharedSizeBytes', ctypes.c_int32, 24), ('maxThreadsPerBlock', ctypes.c_int32, 28), ('numRegs', ctypes.c_int32, 32), ('preferredShmemCarveout', ctypes.c_int32, 36), ('ptxVersion', ctypes.c_int32, 40), ('sharedSizeBytes', size_t, 48)])
hipLimit_t: dict[int, str] = {(hipLimitStackSize:=0): 'hipLimitStackSize', (hipLimitPrintfFifoSize:=1): 'hipLimitPrintfFifoSize', (hipLimitMallocHeapSize:=2): 'hipLimitMallocHeapSize', (hipExtLimitScratchMin:=4096): 'hipExtLimitScratchMin', (hipExtLimitScratchMax:=4097): 'hipExtLimitScratchMax', (hipExtLimitScratchCurrent:=4098): 'hipExtLimitScratchCurrent', (hipLimitRange:=4099): 'hipLimitRange'}
hipStreamBatchMemOpType: dict[int, str] = {(hipStreamMemOpWaitValue32:=1): 'hipStreamMemOpWaitValue32', (hipStreamMemOpWriteValue32:=2): 'hipStreamMemOpWriteValue32', (hipStreamMemOpWaitValue64:=4): 'hipStreamMemOpWaitValue64', (hipStreamMemOpWriteValue64:=5): 'hipStreamMemOpWriteValue64', (hipStreamMemOpBarrier:=6): 'hipStreamMemOpBarrier', (hipStreamMemOpFlushRemoteWrites:=3): 'hipStreamMemOpFlushRemoteWrites'}
@c.record
class hipStreamBatchMemOpParams_union(c.Struct):
  SIZE = 48
  operation: int
  waitValue: hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t
  writeValue: hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t
  flushRemoteWrites: hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t
  memoryBarrier: hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t
  pad: c.Array[ctypes.c_uint64, Literal[6]]
@c.record
class hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t(c.Struct):
  SIZE = 40
  operation: int
  address: ctypes.c_void_p
  value: int
  value64: int
  flags: int
  alias: ctypes.c_void_p
hipDeviceptr_t: TypeAlias = ctypes.c_void_p
uint64_t: TypeAlias = ctypes.c_uint64
hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t.register_fields([('operation', ctypes.c_uint32, 0), ('address', hipDeviceptr_t, 8), ('value', uint32_t, 16), ('value64', uint64_t, 16), ('flags', ctypes.c_uint32, 24), ('alias', hipDeviceptr_t, 32)])
@c.record
class hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t(c.Struct):
  SIZE = 40
  operation: int
  address: ctypes.c_void_p
  value: int
  value64: int
  flags: int
  alias: ctypes.c_void_p
hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t.register_fields([('operation', ctypes.c_uint32, 0), ('address', hipDeviceptr_t, 8), ('value', uint32_t, 16), ('value64', uint64_t, 16), ('flags', ctypes.c_uint32, 24), ('alias', hipDeviceptr_t, 32)])
@c.record
class hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t(c.Struct):
  SIZE = 8
  operation: int
  flags: int
hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t.register_fields([('operation', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4)])
@c.record
class hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t(c.Struct):
  SIZE = 8
  operation: int
  flags: int
hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t.register_fields([('operation', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4)])
hipStreamBatchMemOpParams_union.register_fields([('operation', ctypes.c_uint32, 0), ('waitValue', hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t, 0), ('writeValue', hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t, 0), ('flushRemoteWrites', hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t, 0), ('memoryBarrier', hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t, 0), ('pad', c.Array[uint64_t, Literal[6]], 0)])
hipStreamBatchMemOpParams: TypeAlias = hipStreamBatchMemOpParams_union
@c.record
class hipBatchMemOpNodeParams(c.Struct):
  SIZE = 32
  ctx: c.POINTER[ihipCtx_t]
  count: int
  paramArray: c.POINTER[hipStreamBatchMemOpParams_union]
  flags: int
hipBatchMemOpNodeParams.register_fields([('ctx', hipCtx_t, 0), ('count', ctypes.c_uint32, 8), ('paramArray', c.POINTER[hipStreamBatchMemOpParams], 16), ('flags', ctypes.c_uint32, 24)])
hipMemoryAdvise: dict[int, str] = {(hipMemAdviseSetReadMostly:=1): 'hipMemAdviseSetReadMostly', (hipMemAdviseUnsetReadMostly:=2): 'hipMemAdviseUnsetReadMostly', (hipMemAdviseSetPreferredLocation:=3): 'hipMemAdviseSetPreferredLocation', (hipMemAdviseUnsetPreferredLocation:=4): 'hipMemAdviseUnsetPreferredLocation', (hipMemAdviseSetAccessedBy:=5): 'hipMemAdviseSetAccessedBy', (hipMemAdviseUnsetAccessedBy:=6): 'hipMemAdviseUnsetAccessedBy', (hipMemAdviseSetCoarseGrain:=100): 'hipMemAdviseSetCoarseGrain', (hipMemAdviseUnsetCoarseGrain:=101): 'hipMemAdviseUnsetCoarseGrain'}
hipMemRangeCoherencyMode: dict[int, str] = {(hipMemRangeCoherencyModeFineGrain:=0): 'hipMemRangeCoherencyModeFineGrain', (hipMemRangeCoherencyModeCoarseGrain:=1): 'hipMemRangeCoherencyModeCoarseGrain', (hipMemRangeCoherencyModeIndeterminate:=2): 'hipMemRangeCoherencyModeIndeterminate'}
hipMemRangeAttribute: dict[int, str] = {(hipMemRangeAttributeReadMostly:=1): 'hipMemRangeAttributeReadMostly', (hipMemRangeAttributePreferredLocation:=2): 'hipMemRangeAttributePreferredLocation', (hipMemRangeAttributeAccessedBy:=3): 'hipMemRangeAttributeAccessedBy', (hipMemRangeAttributeLastPrefetchLocation:=4): 'hipMemRangeAttributeLastPrefetchLocation', (hipMemRangeAttributeCoherencyMode:=100): 'hipMemRangeAttributeCoherencyMode'}
hipMemPoolAttr: dict[int, str] = {(hipMemPoolReuseFollowEventDependencies:=1): 'hipMemPoolReuseFollowEventDependencies', (hipMemPoolReuseAllowOpportunistic:=2): 'hipMemPoolReuseAllowOpportunistic', (hipMemPoolReuseAllowInternalDependencies:=3): 'hipMemPoolReuseAllowInternalDependencies', (hipMemPoolAttrReleaseThreshold:=4): 'hipMemPoolAttrReleaseThreshold', (hipMemPoolAttrReservedMemCurrent:=5): 'hipMemPoolAttrReservedMemCurrent', (hipMemPoolAttrReservedMemHigh:=6): 'hipMemPoolAttrReservedMemHigh', (hipMemPoolAttrUsedMemCurrent:=7): 'hipMemPoolAttrUsedMemCurrent', (hipMemPoolAttrUsedMemHigh:=8): 'hipMemPoolAttrUsedMemHigh'}
hipMemAccessFlags: dict[int, str] = {(hipMemAccessFlagsProtNone:=0): 'hipMemAccessFlagsProtNone', (hipMemAccessFlagsProtRead:=1): 'hipMemAccessFlagsProtRead', (hipMemAccessFlagsProtReadWrite:=3): 'hipMemAccessFlagsProtReadWrite'}
@c.record
class hipMemAccessDesc(c.Struct):
  SIZE = 12
  location: hipMemLocation
  flags: int
@c.record
class hipMemLocation(c.Struct):
  SIZE = 8
  type: int
  id: int
hipMemLocationType: dict[int, str] = {(hipMemLocationTypeInvalid:=0): 'hipMemLocationTypeInvalid', (hipMemLocationTypeNone:=0): 'hipMemLocationTypeNone', (hipMemLocationTypeDevice:=1): 'hipMemLocationTypeDevice', (hipMemLocationTypeHost:=2): 'hipMemLocationTypeHost', (hipMemLocationTypeHostNuma:=3): 'hipMemLocationTypeHostNuma', (hipMemLocationTypeHostNumaCurrent:=4): 'hipMemLocationTypeHostNumaCurrent'}
hipMemLocation.register_fields([('type', ctypes.c_uint32, 0), ('id', ctypes.c_int32, 4)])
hipMemAccessDesc.register_fields([('location', hipMemLocation, 0), ('flags', ctypes.c_uint32, 8)])
hipMemAllocationType: dict[int, str] = {(hipMemAllocationTypeInvalid:=0): 'hipMemAllocationTypeInvalid', (hipMemAllocationTypePinned:=1): 'hipMemAllocationTypePinned', (hipMemAllocationTypeUncached:=1073741824): 'hipMemAllocationTypeUncached', (hipMemAllocationTypeMax:=2147483647): 'hipMemAllocationTypeMax'}
hipMemAllocationHandleType: dict[int, str] = {(hipMemHandleTypeNone:=0): 'hipMemHandleTypeNone', (hipMemHandleTypePosixFileDescriptor:=1): 'hipMemHandleTypePosixFileDescriptor', (hipMemHandleTypeWin32:=2): 'hipMemHandleTypeWin32', (hipMemHandleTypeWin32Kmt:=4): 'hipMemHandleTypeWin32Kmt'}
@c.record
class hipMemPoolProps(c.Struct):
  SIZE = 88
  allocType: int
  handleTypes: int
  location: hipMemLocation
  win32SecurityAttributes: ctypes.c_void_p
  maxSize: int
  reserved: c.Array[ctypes.c_ubyte, Literal[56]]
hipMemPoolProps.register_fields([('allocType', ctypes.c_uint32, 0), ('handleTypes', ctypes.c_uint32, 4), ('location', hipMemLocation, 8), ('win32SecurityAttributes', ctypes.c_void_p, 16), ('maxSize', size_t, 24), ('reserved', c.Array[ctypes.c_ubyte, Literal[56]], 32)])
@c.record
class hipMemPoolPtrExportData(c.Struct):
  SIZE = 64
  reserved: c.Array[ctypes.c_ubyte, Literal[64]]
hipMemPoolPtrExportData.register_fields([('reserved', c.Array[ctypes.c_ubyte, Literal[64]], 0)])
hipFuncAttribute: dict[int, str] = {(hipFuncAttributeMaxDynamicSharedMemorySize:=8): 'hipFuncAttributeMaxDynamicSharedMemorySize', (hipFuncAttributePreferredSharedMemoryCarveout:=9): 'hipFuncAttributePreferredSharedMemoryCarveout', (hipFuncAttributeMax:=10): 'hipFuncAttributeMax'}
hipFuncCache_t: dict[int, str] = {(hipFuncCachePreferNone:=0): 'hipFuncCachePreferNone', (hipFuncCachePreferShared:=1): 'hipFuncCachePreferShared', (hipFuncCachePreferL1:=2): 'hipFuncCachePreferL1', (hipFuncCachePreferEqual:=3): 'hipFuncCachePreferEqual'}
hipSharedMemConfig: dict[int, str] = {(hipSharedMemBankSizeDefault:=0): 'hipSharedMemBankSizeDefault', (hipSharedMemBankSizeFourByte:=1): 'hipSharedMemBankSizeFourByte', (hipSharedMemBankSizeEightByte:=2): 'hipSharedMemBankSizeEightByte'}
@c.record
class hipLaunchParams_t(c.Struct):
  SIZE = 56
  func: ctypes.c_void_p
  gridDim: dim3
  blockDim: dim3
  args: c.POINTER[ctypes.c_void_p]
  sharedMem: int
  stream: c.POINTER[ihipStream_t]
hipLaunchParams_t.register_fields([('func', ctypes.c_void_p, 0), ('gridDim', dim3, 8), ('blockDim', dim3, 20), ('args', c.POINTER[ctypes.c_void_p], 32), ('sharedMem', size_t, 40), ('stream', hipStream_t, 48)])
hipLaunchParams: TypeAlias = hipLaunchParams_t
@c.record
class hipFunctionLaunchParams_t(c.Struct):
  SIZE = 56
  function: c.POINTER[ihipModuleSymbol_t]
  gridDimX: int
  gridDimY: int
  gridDimZ: int
  blockDimX: int
  blockDimY: int
  blockDimZ: int
  sharedMemBytes: int
  hStream: c.POINTER[ihipStream_t]
  kernelParams: c.POINTER[ctypes.c_void_p]
hipFunctionLaunchParams_t.register_fields([('function', hipFunction_t, 0), ('gridDimX', ctypes.c_uint32, 8), ('gridDimY', ctypes.c_uint32, 12), ('gridDimZ', ctypes.c_uint32, 16), ('blockDimX', ctypes.c_uint32, 20), ('blockDimY', ctypes.c_uint32, 24), ('blockDimZ', ctypes.c_uint32, 28), ('sharedMemBytes', ctypes.c_uint32, 32), ('hStream', hipStream_t, 40), ('kernelParams', c.POINTER[ctypes.c_void_p], 48)])
hipFunctionLaunchParams: TypeAlias = hipFunctionLaunchParams_t
hipExternalMemoryHandleType_enum: dict[int, str] = {(hipExternalMemoryHandleTypeOpaqueFd:=1): 'hipExternalMemoryHandleTypeOpaqueFd', (hipExternalMemoryHandleTypeOpaqueWin32:=2): 'hipExternalMemoryHandleTypeOpaqueWin32', (hipExternalMemoryHandleTypeOpaqueWin32Kmt:=3): 'hipExternalMemoryHandleTypeOpaqueWin32Kmt', (hipExternalMemoryHandleTypeD3D12Heap:=4): 'hipExternalMemoryHandleTypeD3D12Heap', (hipExternalMemoryHandleTypeD3D12Resource:=5): 'hipExternalMemoryHandleTypeD3D12Resource', (hipExternalMemoryHandleTypeD3D11Resource:=6): 'hipExternalMemoryHandleTypeD3D11Resource', (hipExternalMemoryHandleTypeD3D11ResourceKmt:=7): 'hipExternalMemoryHandleTypeD3D11ResourceKmt', (hipExternalMemoryHandleTypeNvSciBuf:=8): 'hipExternalMemoryHandleTypeNvSciBuf'}
hipExternalMemoryHandleType: TypeAlias = ctypes.c_uint32
@c.record
class hipExternalMemoryHandleDesc_st(c.Struct):
  SIZE = 104
  type: int
  handle: hipExternalMemoryHandleDesc_st_handle
  size: int
  flags: int
  reserved: c.Array[ctypes.c_uint32, Literal[16]]
@c.record
class hipExternalMemoryHandleDesc_st_handle(c.Struct):
  SIZE = 16
  fd: int
  win32: hipExternalMemoryHandleDesc_st_handle_win32
  nvSciBufObject: ctypes.c_void_p
@c.record
class hipExternalMemoryHandleDesc_st_handle_win32(c.Struct):
  SIZE = 16
  handle: ctypes.c_void_p
  name: ctypes.c_void_p
hipExternalMemoryHandleDesc_st_handle_win32.register_fields([('handle', ctypes.c_void_p, 0), ('name', ctypes.c_void_p, 8)])
hipExternalMemoryHandleDesc_st_handle.register_fields([('fd', ctypes.c_int32, 0), ('win32', hipExternalMemoryHandleDesc_st_handle_win32, 0), ('nvSciBufObject', ctypes.c_void_p, 0)])
hipExternalMemoryHandleDesc_st.register_fields([('type', hipExternalMemoryHandleType, 0), ('handle', hipExternalMemoryHandleDesc_st_handle, 8), ('size', ctypes.c_uint64, 24), ('flags', ctypes.c_uint32, 32), ('reserved', c.Array[ctypes.c_uint32, Literal[16]], 36)])
hipExternalMemoryHandleDesc: TypeAlias = hipExternalMemoryHandleDesc_st
@c.record
class hipExternalMemoryBufferDesc_st(c.Struct):
  SIZE = 88
  offset: int
  size: int
  flags: int
  reserved: c.Array[ctypes.c_uint32, Literal[16]]
hipExternalMemoryBufferDesc_st.register_fields([('offset', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('reserved', c.Array[ctypes.c_uint32, Literal[16]], 20)])
hipExternalMemoryBufferDesc: TypeAlias = hipExternalMemoryBufferDesc_st
@c.record
class hipExternalMemoryMipmappedArrayDesc_st(c.Struct):
  SIZE = 64
  offset: int
  formatDesc: hipChannelFormatDesc
  extent: hipExtent
  flags: int
  numLevels: int
@c.record
class hipChannelFormatDesc(c.Struct):
  SIZE = 20
  x: int
  y: int
  z: int
  w: int
  f: int
hipChannelFormatKind: dict[int, str] = {(hipChannelFormatKindSigned:=0): 'hipChannelFormatKindSigned', (hipChannelFormatKindUnsigned:=1): 'hipChannelFormatKindUnsigned', (hipChannelFormatKindFloat:=2): 'hipChannelFormatKindFloat', (hipChannelFormatKindNone:=3): 'hipChannelFormatKindNone'}
hipChannelFormatDesc.register_fields([('x', ctypes.c_int32, 0), ('y', ctypes.c_int32, 4), ('z', ctypes.c_int32, 8), ('w', ctypes.c_int32, 12), ('f', ctypes.c_uint32, 16)])
@c.record
class hipExtent(c.Struct):
  SIZE = 24
  width: int
  height: int
  depth: int
hipExtent.register_fields([('width', size_t, 0), ('height', size_t, 8), ('depth', size_t, 16)])
hipExternalMemoryMipmappedArrayDesc_st.register_fields([('offset', ctypes.c_uint64, 0), ('formatDesc', hipChannelFormatDesc, 8), ('extent', hipExtent, 32), ('flags', ctypes.c_uint32, 56), ('numLevels', ctypes.c_uint32, 60)])
hipExternalMemoryMipmappedArrayDesc: TypeAlias = hipExternalMemoryMipmappedArrayDesc_st
hipExternalMemory_t: TypeAlias = ctypes.c_void_p
hipExternalSemaphoreHandleType_enum: dict[int, str] = {(hipExternalSemaphoreHandleTypeOpaqueFd:=1): 'hipExternalSemaphoreHandleTypeOpaqueFd', (hipExternalSemaphoreHandleTypeOpaqueWin32:=2): 'hipExternalSemaphoreHandleTypeOpaqueWin32', (hipExternalSemaphoreHandleTypeOpaqueWin32Kmt:=3): 'hipExternalSemaphoreHandleTypeOpaqueWin32Kmt', (hipExternalSemaphoreHandleTypeD3D12Fence:=4): 'hipExternalSemaphoreHandleTypeD3D12Fence', (hipExternalSemaphoreHandleTypeD3D11Fence:=5): 'hipExternalSemaphoreHandleTypeD3D11Fence', (hipExternalSemaphoreHandleTypeNvSciSync:=6): 'hipExternalSemaphoreHandleTypeNvSciSync', (hipExternalSemaphoreHandleTypeKeyedMutex:=7): 'hipExternalSemaphoreHandleTypeKeyedMutex', (hipExternalSemaphoreHandleTypeKeyedMutexKmt:=8): 'hipExternalSemaphoreHandleTypeKeyedMutexKmt', (hipExternalSemaphoreHandleTypeTimelineSemaphoreFd:=9): 'hipExternalSemaphoreHandleTypeTimelineSemaphoreFd', (hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32:=10): 'hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32'}
hipExternalSemaphoreHandleType: TypeAlias = ctypes.c_uint32
@c.record
class hipExternalSemaphoreHandleDesc_st(c.Struct):
  SIZE = 96
  type: int
  handle: hipExternalSemaphoreHandleDesc_st_handle
  flags: int
  reserved: c.Array[ctypes.c_uint32, Literal[16]]
@c.record
class hipExternalSemaphoreHandleDesc_st_handle(c.Struct):
  SIZE = 16
  fd: int
  win32: hipExternalSemaphoreHandleDesc_st_handle_win32
  NvSciSyncObj: ctypes.c_void_p
@c.record
class hipExternalSemaphoreHandleDesc_st_handle_win32(c.Struct):
  SIZE = 16
  handle: ctypes.c_void_p
  name: ctypes.c_void_p
hipExternalSemaphoreHandleDesc_st_handle_win32.register_fields([('handle', ctypes.c_void_p, 0), ('name', ctypes.c_void_p, 8)])
hipExternalSemaphoreHandleDesc_st_handle.register_fields([('fd', ctypes.c_int32, 0), ('win32', hipExternalSemaphoreHandleDesc_st_handle_win32, 0), ('NvSciSyncObj', ctypes.c_void_p, 0)])
hipExternalSemaphoreHandleDesc_st.register_fields([('type', hipExternalSemaphoreHandleType, 0), ('handle', hipExternalSemaphoreHandleDesc_st_handle, 8), ('flags', ctypes.c_uint32, 24), ('reserved', c.Array[ctypes.c_uint32, Literal[16]], 28)])
hipExternalSemaphoreHandleDesc: TypeAlias = hipExternalSemaphoreHandleDesc_st
hipExternalSemaphore_t: TypeAlias = ctypes.c_void_p
@c.record
class hipExternalSemaphoreSignalParams_st(c.Struct):
  SIZE = 144
  params: hipExternalSemaphoreSignalParams_st_params
  flags: int
  reserved: c.Array[ctypes.c_uint32, Literal[16]]
@c.record
class hipExternalSemaphoreSignalParams_st_params(c.Struct):
  SIZE = 72
  fence: hipExternalSemaphoreSignalParams_st_params_fence
  nvSciSync: hipExternalSemaphoreSignalParams_st_params_nvSciSync
  keyedMutex: hipExternalSemaphoreSignalParams_st_params_keyedMutex
  reserved: c.Array[ctypes.c_uint32, Literal[12]]
@c.record
class hipExternalSemaphoreSignalParams_st_params_fence(c.Struct):
  SIZE = 8
  value: int
hipExternalSemaphoreSignalParams_st_params_fence.register_fields([('value', ctypes.c_uint64, 0)])
@c.record
class hipExternalSemaphoreSignalParams_st_params_nvSciSync(c.Struct):
  SIZE = 8
  fence: ctypes.c_void_p
  reserved: int
hipExternalSemaphoreSignalParams_st_params_nvSciSync.register_fields([('fence', ctypes.c_void_p, 0), ('reserved', ctypes.c_uint64, 0)])
@c.record
class hipExternalSemaphoreSignalParams_st_params_keyedMutex(c.Struct):
  SIZE = 8
  key: int
hipExternalSemaphoreSignalParams_st_params_keyedMutex.register_fields([('key', ctypes.c_uint64, 0)])
hipExternalSemaphoreSignalParams_st_params.register_fields([('fence', hipExternalSemaphoreSignalParams_st_params_fence, 0), ('nvSciSync', hipExternalSemaphoreSignalParams_st_params_nvSciSync, 8), ('keyedMutex', hipExternalSemaphoreSignalParams_st_params_keyedMutex, 16), ('reserved', c.Array[ctypes.c_uint32, Literal[12]], 24)])
hipExternalSemaphoreSignalParams_st.register_fields([('params', hipExternalSemaphoreSignalParams_st_params, 0), ('flags', ctypes.c_uint32, 72), ('reserved', c.Array[ctypes.c_uint32, Literal[16]], 76)])
hipExternalSemaphoreSignalParams: TypeAlias = hipExternalSemaphoreSignalParams_st
@c.record
class hipExternalSemaphoreWaitParams_st(c.Struct):
  SIZE = 144
  params: hipExternalSemaphoreWaitParams_st_params
  flags: int
  reserved: c.Array[ctypes.c_uint32, Literal[16]]
@c.record
class hipExternalSemaphoreWaitParams_st_params(c.Struct):
  SIZE = 72
  fence: hipExternalSemaphoreWaitParams_st_params_fence
  nvSciSync: hipExternalSemaphoreWaitParams_st_params_nvSciSync
  keyedMutex: hipExternalSemaphoreWaitParams_st_params_keyedMutex
  reserved: c.Array[ctypes.c_uint32, Literal[10]]
@c.record
class hipExternalSemaphoreWaitParams_st_params_fence(c.Struct):
  SIZE = 8
  value: int
hipExternalSemaphoreWaitParams_st_params_fence.register_fields([('value', ctypes.c_uint64, 0)])
@c.record
class hipExternalSemaphoreWaitParams_st_params_nvSciSync(c.Struct):
  SIZE = 8
  fence: ctypes.c_void_p
  reserved: int
hipExternalSemaphoreWaitParams_st_params_nvSciSync.register_fields([('fence', ctypes.c_void_p, 0), ('reserved', ctypes.c_uint64, 0)])
@c.record
class hipExternalSemaphoreWaitParams_st_params_keyedMutex(c.Struct):
  SIZE = 16
  key: int
  timeoutMs: int
hipExternalSemaphoreWaitParams_st_params_keyedMutex.register_fields([('key', ctypes.c_uint64, 0), ('timeoutMs', ctypes.c_uint32, 8)])
hipExternalSemaphoreWaitParams_st_params.register_fields([('fence', hipExternalSemaphoreWaitParams_st_params_fence, 0), ('nvSciSync', hipExternalSemaphoreWaitParams_st_params_nvSciSync, 8), ('keyedMutex', hipExternalSemaphoreWaitParams_st_params_keyedMutex, 16), ('reserved', c.Array[ctypes.c_uint32, Literal[10]], 32)])
hipExternalSemaphoreWaitParams_st.register_fields([('params', hipExternalSemaphoreWaitParams_st_params, 0), ('flags', ctypes.c_uint32, 72), ('reserved', c.Array[ctypes.c_uint32, Literal[16]], 76)])
hipExternalSemaphoreWaitParams: TypeAlias = hipExternalSemaphoreWaitParams_st
@dll.bind(None, c.POINTER[c.POINTER[ctypes.c_char]], c.POINTER[ctypes.c_uint32])
def __hipGetPCH(pch:c.POINTER[c.POINTER[ctypes.c_char]], size:c.POINTER[ctypes.c_uint32]) -> None: ...
hipGraphicsRegisterFlags: dict[int, str] = {(hipGraphicsRegisterFlagsNone:=0): 'hipGraphicsRegisterFlagsNone', (hipGraphicsRegisterFlagsReadOnly:=1): 'hipGraphicsRegisterFlagsReadOnly', (hipGraphicsRegisterFlagsWriteDiscard:=2): 'hipGraphicsRegisterFlagsWriteDiscard', (hipGraphicsRegisterFlagsSurfaceLoadStore:=4): 'hipGraphicsRegisterFlagsSurfaceLoadStore', (hipGraphicsRegisterFlagsTextureGather:=8): 'hipGraphicsRegisterFlagsTextureGather'}
class _hipGraphicsResource(c.Struct): pass
hipGraphicsResource: TypeAlias = _hipGraphicsResource
hipGraphicsResource_t: TypeAlias = c.POINTER[_hipGraphicsResource]
class ihipGraph(c.Struct): pass
hipGraph_t: TypeAlias = c.POINTER[ihipGraph]
class hipGraphNode(c.Struct): pass
hipGraphNode_t: TypeAlias = c.POINTER[hipGraphNode]
class hipGraphExec(c.Struct): pass
hipGraphExec_t: TypeAlias = c.POINTER[hipGraphExec]
class hipUserObject(c.Struct): pass
hipUserObject_t: TypeAlias = c.POINTER[hipUserObject]
hipGraphNodeType: dict[int, str] = {(hipGraphNodeTypeKernel:=0): 'hipGraphNodeTypeKernel', (hipGraphNodeTypeMemcpy:=1): 'hipGraphNodeTypeMemcpy', (hipGraphNodeTypeMemset:=2): 'hipGraphNodeTypeMemset', (hipGraphNodeTypeHost:=3): 'hipGraphNodeTypeHost', (hipGraphNodeTypeGraph:=4): 'hipGraphNodeTypeGraph', (hipGraphNodeTypeEmpty:=5): 'hipGraphNodeTypeEmpty', (hipGraphNodeTypeWaitEvent:=6): 'hipGraphNodeTypeWaitEvent', (hipGraphNodeTypeEventRecord:=7): 'hipGraphNodeTypeEventRecord', (hipGraphNodeTypeExtSemaphoreSignal:=8): 'hipGraphNodeTypeExtSemaphoreSignal', (hipGraphNodeTypeExtSemaphoreWait:=9): 'hipGraphNodeTypeExtSemaphoreWait', (hipGraphNodeTypeMemAlloc:=10): 'hipGraphNodeTypeMemAlloc', (hipGraphNodeTypeMemFree:=11): 'hipGraphNodeTypeMemFree', (hipGraphNodeTypeMemcpyFromSymbol:=12): 'hipGraphNodeTypeMemcpyFromSymbol', (hipGraphNodeTypeMemcpyToSymbol:=13): 'hipGraphNodeTypeMemcpyToSymbol', (hipGraphNodeTypeBatchMemOp:=14): 'hipGraphNodeTypeBatchMemOp', (hipGraphNodeTypeCount:=15): 'hipGraphNodeTypeCount'}
hipHostFn_t: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_void_p]]
@c.record
class hipHostNodeParams(c.Struct):
  SIZE = 16
  fn: c.CFUNCTYPE[None, [ctypes.c_void_p]]
  userData: ctypes.c_void_p
hipHostNodeParams.register_fields([('fn', hipHostFn_t, 0), ('userData', ctypes.c_void_p, 8)])
@c.record
class hipKernelNodeParams(c.Struct):
  SIZE = 64
  blockDim: dim3
  extra: c.POINTER[ctypes.c_void_p]
  func: ctypes.c_void_p
  gridDim: dim3
  kernelParams: c.POINTER[ctypes.c_void_p]
  sharedMemBytes: int
hipKernelNodeParams.register_fields([('blockDim', dim3, 0), ('extra', c.POINTER[ctypes.c_void_p], 16), ('func', ctypes.c_void_p, 24), ('gridDim', dim3, 32), ('kernelParams', c.POINTER[ctypes.c_void_p], 48), ('sharedMemBytes', ctypes.c_uint32, 56)])
@c.record
class hipMemsetParams(c.Struct):
  SIZE = 48
  dst: ctypes.c_void_p
  elementSize: int
  height: int
  pitch: int
  value: int
  width: int
hipMemsetParams.register_fields([('dst', ctypes.c_void_p, 0), ('elementSize', ctypes.c_uint32, 8), ('height', size_t, 16), ('pitch', size_t, 24), ('value', ctypes.c_uint32, 32), ('width', size_t, 40)])
@c.record
class hipMemAllocNodeParams(c.Struct):
  SIZE = 120
  poolProps: hipMemPoolProps
  accessDescs: c.POINTER[hipMemAccessDesc]
  accessDescCount: int
  bytesize: int
  dptr: ctypes.c_void_p
hipMemAllocNodeParams.register_fields([('poolProps', hipMemPoolProps, 0), ('accessDescs', c.POINTER[hipMemAccessDesc], 88), ('accessDescCount', size_t, 96), ('bytesize', size_t, 104), ('dptr', ctypes.c_void_p, 112)])
hipAccessProperty: dict[int, str] = {(hipAccessPropertyNormal:=0): 'hipAccessPropertyNormal', (hipAccessPropertyStreaming:=1): 'hipAccessPropertyStreaming', (hipAccessPropertyPersisting:=2): 'hipAccessPropertyPersisting'}
@c.record
class hipAccessPolicyWindow(c.Struct):
  SIZE = 32
  base_ptr: ctypes.c_void_p
  hitProp: int
  hitRatio: float
  missProp: int
  num_bytes: int
hipAccessPolicyWindow.register_fields([('base_ptr', ctypes.c_void_p, 0), ('hitProp', ctypes.c_uint32, 8), ('hitRatio', ctypes.c_float, 12), ('missProp', ctypes.c_uint32, 16), ('num_bytes', size_t, 24)])
@c.record
class hipLaunchMemSyncDomainMap(c.Struct):
  SIZE = 2
  default_: int
  remote: int
hipLaunchMemSyncDomainMap.register_fields([('default_', ctypes.c_ubyte, 0), ('remote', ctypes.c_ubyte, 1)])
hipLaunchMemSyncDomain: dict[int, str] = {(hipLaunchMemSyncDomainDefault:=0): 'hipLaunchMemSyncDomainDefault', (hipLaunchMemSyncDomainRemote:=1): 'hipLaunchMemSyncDomainRemote'}
hipSynchronizationPolicy: dict[int, str] = {(hipSyncPolicyAuto:=1): 'hipSyncPolicyAuto', (hipSyncPolicySpin:=2): 'hipSyncPolicySpin', (hipSyncPolicyYield:=3): 'hipSyncPolicyYield', (hipSyncPolicyBlockingSync:=4): 'hipSyncPolicyBlockingSync'}
hipLaunchAttributeID: dict[int, str] = {(hipLaunchAttributeAccessPolicyWindow:=1): 'hipLaunchAttributeAccessPolicyWindow', (hipLaunchAttributeCooperative:=2): 'hipLaunchAttributeCooperative', (hipLaunchAttributeSynchronizationPolicy:=3): 'hipLaunchAttributeSynchronizationPolicy', (hipLaunchAttributePriority:=8): 'hipLaunchAttributePriority', (hipLaunchAttributeMemSyncDomainMap:=9): 'hipLaunchAttributeMemSyncDomainMap', (hipLaunchAttributeMemSyncDomain:=10): 'hipLaunchAttributeMemSyncDomain', (hipLaunchAttributeMax:=11): 'hipLaunchAttributeMax'}
@c.record
class hipLaunchAttributeValue(c.Struct):
  SIZE = 64
  pad: c.Array[ctypes.c_char, Literal[64]]
  accessPolicyWindow: hipAccessPolicyWindow
  cooperative: int
  priority: int
  syncPolicy: int
  memSyncDomainMap: hipLaunchMemSyncDomainMap
  memSyncDomain: int
hipLaunchAttributeValue.register_fields([('pad', c.Array[ctypes.c_char, Literal[64]], 0), ('accessPolicyWindow', hipAccessPolicyWindow, 0), ('cooperative', ctypes.c_int32, 0), ('priority', ctypes.c_int32, 0), ('syncPolicy', ctypes.c_uint32, 0), ('memSyncDomainMap', hipLaunchMemSyncDomainMap, 0), ('memSyncDomain', ctypes.c_uint32, 0)])
hipGraphExecUpdateResult: dict[int, str] = {(hipGraphExecUpdateSuccess:=0): 'hipGraphExecUpdateSuccess', (hipGraphExecUpdateError:=1): 'hipGraphExecUpdateError', (hipGraphExecUpdateErrorTopologyChanged:=2): 'hipGraphExecUpdateErrorTopologyChanged', (hipGraphExecUpdateErrorNodeTypeChanged:=3): 'hipGraphExecUpdateErrorNodeTypeChanged', (hipGraphExecUpdateErrorFunctionChanged:=4): 'hipGraphExecUpdateErrorFunctionChanged', (hipGraphExecUpdateErrorParametersChanged:=5): 'hipGraphExecUpdateErrorParametersChanged', (hipGraphExecUpdateErrorNotSupported:=6): 'hipGraphExecUpdateErrorNotSupported', (hipGraphExecUpdateErrorUnsupportedFunctionChange:=7): 'hipGraphExecUpdateErrorUnsupportedFunctionChange'}
hipStreamCaptureMode: dict[int, str] = {(hipStreamCaptureModeGlobal:=0): 'hipStreamCaptureModeGlobal', (hipStreamCaptureModeThreadLocal:=1): 'hipStreamCaptureModeThreadLocal', (hipStreamCaptureModeRelaxed:=2): 'hipStreamCaptureModeRelaxed'}
hipStreamCaptureStatus: dict[int, str] = {(hipStreamCaptureStatusNone:=0): 'hipStreamCaptureStatusNone', (hipStreamCaptureStatusActive:=1): 'hipStreamCaptureStatusActive', (hipStreamCaptureStatusInvalidated:=2): 'hipStreamCaptureStatusInvalidated'}
hipStreamUpdateCaptureDependenciesFlags: dict[int, str] = {(hipStreamAddCaptureDependencies:=0): 'hipStreamAddCaptureDependencies', (hipStreamSetCaptureDependencies:=1): 'hipStreamSetCaptureDependencies'}
hipGraphMemAttributeType: dict[int, str] = {(hipGraphMemAttrUsedMemCurrent:=0): 'hipGraphMemAttrUsedMemCurrent', (hipGraphMemAttrUsedMemHigh:=1): 'hipGraphMemAttrUsedMemHigh', (hipGraphMemAttrReservedMemCurrent:=2): 'hipGraphMemAttrReservedMemCurrent', (hipGraphMemAttrReservedMemHigh:=3): 'hipGraphMemAttrReservedMemHigh'}
hipUserObjectFlags: dict[int, str] = {(hipUserObjectNoDestructorSync:=1): 'hipUserObjectNoDestructorSync'}
hipUserObjectRetainFlags: dict[int, str] = {(hipGraphUserObjectMove:=1): 'hipGraphUserObjectMove'}
hipGraphInstantiateFlags: dict[int, str] = {(hipGraphInstantiateFlagAutoFreeOnLaunch:=1): 'hipGraphInstantiateFlagAutoFreeOnLaunch', (hipGraphInstantiateFlagUpload:=2): 'hipGraphInstantiateFlagUpload', (hipGraphInstantiateFlagDeviceLaunch:=4): 'hipGraphInstantiateFlagDeviceLaunch', (hipGraphInstantiateFlagUseNodePriority:=8): 'hipGraphInstantiateFlagUseNodePriority'}
hipGraphDebugDotFlags: dict[int, str] = {(hipGraphDebugDotFlagsVerbose:=1): 'hipGraphDebugDotFlagsVerbose', (hipGraphDebugDotFlagsKernelNodeParams:=4): 'hipGraphDebugDotFlagsKernelNodeParams', (hipGraphDebugDotFlagsMemcpyNodeParams:=8): 'hipGraphDebugDotFlagsMemcpyNodeParams', (hipGraphDebugDotFlagsMemsetNodeParams:=16): 'hipGraphDebugDotFlagsMemsetNodeParams', (hipGraphDebugDotFlagsHostNodeParams:=32): 'hipGraphDebugDotFlagsHostNodeParams', (hipGraphDebugDotFlagsEventNodeParams:=64): 'hipGraphDebugDotFlagsEventNodeParams', (hipGraphDebugDotFlagsExtSemasSignalNodeParams:=128): 'hipGraphDebugDotFlagsExtSemasSignalNodeParams', (hipGraphDebugDotFlagsExtSemasWaitNodeParams:=256): 'hipGraphDebugDotFlagsExtSemasWaitNodeParams', (hipGraphDebugDotFlagsKernelNodeAttributes:=512): 'hipGraphDebugDotFlagsKernelNodeAttributes', (hipGraphDebugDotFlagsHandles:=1024): 'hipGraphDebugDotFlagsHandles'}
hipGraphInstantiateResult: dict[int, str] = {(hipGraphInstantiateSuccess:=0): 'hipGraphInstantiateSuccess', (hipGraphInstantiateError:=1): 'hipGraphInstantiateError', (hipGraphInstantiateInvalidStructure:=2): 'hipGraphInstantiateInvalidStructure', (hipGraphInstantiateNodeOperationNotSupported:=3): 'hipGraphInstantiateNodeOperationNotSupported', (hipGraphInstantiateMultipleDevicesNotSupported:=4): 'hipGraphInstantiateMultipleDevicesNotSupported'}
@c.record
class hipGraphInstantiateParams(c.Struct):
  SIZE = 32
  errNode_out: c.POINTER[hipGraphNode]
  flags: int
  result_out: int
  uploadStream: c.POINTER[ihipStream_t]
hipGraphInstantiateParams.register_fields([('errNode_out', hipGraphNode_t, 0), ('flags', ctypes.c_uint64, 8), ('result_out', ctypes.c_uint32, 16), ('uploadStream', hipStream_t, 24)])
@c.record
class hipMemAllocationProp(c.Struct):
  SIZE = 32
  type: int
  requestedHandleType: int
  requestedHandleTypes: int
  location: hipMemLocation
  win32HandleMetaData: ctypes.c_void_p
  allocFlags: hipMemAllocationProp_allocFlags
@c.record
class hipMemAllocationProp_allocFlags(c.Struct):
  SIZE = 4
  compressionType: int
  gpuDirectRDMACapable: int
  usage: int
hipMemAllocationProp_allocFlags.register_fields([('compressionType', ctypes.c_ubyte, 0), ('gpuDirectRDMACapable', ctypes.c_ubyte, 1), ('usage', ctypes.c_uint16, 2)])
hipMemAllocationProp.register_fields([('type', ctypes.c_uint32, 0), ('requestedHandleType', ctypes.c_uint32, 4), ('requestedHandleTypes', ctypes.c_uint32, 4), ('location', hipMemLocation, 8), ('win32HandleMetaData', ctypes.c_void_p, 16), ('allocFlags', hipMemAllocationProp_allocFlags, 24)])
@c.record
class hipExternalSemaphoreSignalNodeParams(c.Struct):
  SIZE = 24
  extSemArray: c.POINTER[ctypes.c_void_p]
  paramsArray: c.POINTER[hipExternalSemaphoreSignalParams_st]
  numExtSems: int
hipExternalSemaphoreSignalNodeParams.register_fields([('extSemArray', c.POINTER[hipExternalSemaphore_t], 0), ('paramsArray', c.POINTER[hipExternalSemaphoreSignalParams], 8), ('numExtSems', ctypes.c_uint32, 16)])
@c.record
class hipExternalSemaphoreWaitNodeParams(c.Struct):
  SIZE = 24
  extSemArray: c.POINTER[ctypes.c_void_p]
  paramsArray: c.POINTER[hipExternalSemaphoreWaitParams_st]
  numExtSems: int
hipExternalSemaphoreWaitNodeParams.register_fields([('extSemArray', c.POINTER[hipExternalSemaphore_t], 0), ('paramsArray', c.POINTER[hipExternalSemaphoreWaitParams], 8), ('numExtSems', ctypes.c_uint32, 16)])
class ihipMemGenericAllocationHandle(c.Struct): pass
hipMemGenericAllocationHandle_t: TypeAlias = c.POINTER[ihipMemGenericAllocationHandle]
hipMemAllocationGranularity_flags: dict[int, str] = {(hipMemAllocationGranularityMinimum:=0): 'hipMemAllocationGranularityMinimum', (hipMemAllocationGranularityRecommended:=1): 'hipMemAllocationGranularityRecommended'}
hipMemHandleType: dict[int, str] = {(hipMemHandleTypeGeneric:=0): 'hipMemHandleTypeGeneric'}
hipMemOperationType: dict[int, str] = {(hipMemOperationTypeMap:=1): 'hipMemOperationTypeMap', (hipMemOperationTypeUnmap:=2): 'hipMemOperationTypeUnmap'}
hipArraySparseSubresourceType: dict[int, str] = {(hipArraySparseSubresourceTypeSparseLevel:=0): 'hipArraySparseSubresourceTypeSparseLevel', (hipArraySparseSubresourceTypeMiptail:=1): 'hipArraySparseSubresourceTypeMiptail'}
@c.record
class hipArrayMapInfo(c.Struct):
  SIZE = 152
  resourceType: int
  resource: hipArrayMapInfo_resource
  subresourceType: int
  subresource: hipArrayMapInfo_subresource
  memOperationType: int
  memHandleType: int
  memHandle: hipArrayMapInfo_memHandle
  offset: int
  deviceBitMask: int
  flags: int
  reserved: c.Array[ctypes.c_uint32, Literal[2]]
hipResourceType: dict[int, str] = {(hipResourceTypeArray:=0): 'hipResourceTypeArray', (hipResourceTypeMipmappedArray:=1): 'hipResourceTypeMipmappedArray', (hipResourceTypeLinear:=2): 'hipResourceTypeLinear', (hipResourceTypePitch2D:=3): 'hipResourceTypePitch2D'}
@c.record
class hipArrayMapInfo_resource(c.Struct):
  SIZE = 64
  mipmap: hipMipmappedArray
  array: c.POINTER[hipArray]
@c.record
class hipMipmappedArray(c.Struct):
  SIZE = 64
  data: ctypes.c_void_p
  desc: hipChannelFormatDesc
  type: int
  width: int
  height: int
  depth: int
  min_mipmap_level: int
  max_mipmap_level: int
  flags: int
  format: int
  num_channels: int
hipArray_Format: dict[int, str] = {(HIP_AD_FORMAT_UNSIGNED_INT8:=1): 'HIP_AD_FORMAT_UNSIGNED_INT8', (HIP_AD_FORMAT_UNSIGNED_INT16:=2): 'HIP_AD_FORMAT_UNSIGNED_INT16', (HIP_AD_FORMAT_UNSIGNED_INT32:=3): 'HIP_AD_FORMAT_UNSIGNED_INT32', (HIP_AD_FORMAT_SIGNED_INT8:=8): 'HIP_AD_FORMAT_SIGNED_INT8', (HIP_AD_FORMAT_SIGNED_INT16:=9): 'HIP_AD_FORMAT_SIGNED_INT16', (HIP_AD_FORMAT_SIGNED_INT32:=10): 'HIP_AD_FORMAT_SIGNED_INT32', (HIP_AD_FORMAT_HALF:=16): 'HIP_AD_FORMAT_HALF', (HIP_AD_FORMAT_FLOAT:=32): 'HIP_AD_FORMAT_FLOAT'}
hipMipmappedArray.register_fields([('data', ctypes.c_void_p, 0), ('desc', hipChannelFormatDesc, 8), ('type', ctypes.c_uint32, 28), ('width', ctypes.c_uint32, 32), ('height', ctypes.c_uint32, 36), ('depth', ctypes.c_uint32, 40), ('min_mipmap_level', ctypes.c_uint32, 44), ('max_mipmap_level', ctypes.c_uint32, 48), ('flags', ctypes.c_uint32, 52), ('format', ctypes.c_uint32, 56), ('num_channels', ctypes.c_uint32, 60)])
class hipArray(c.Struct): pass
hipArray_t: TypeAlias = c.POINTER[hipArray]
hipArrayMapInfo_resource.register_fields([('mipmap', hipMipmappedArray, 0), ('array', hipArray_t, 0)])
@c.record
class hipArrayMapInfo_subresource(c.Struct):
  SIZE = 32
  sparseLevel: hipArrayMapInfo_subresource_sparseLevel
  miptail: hipArrayMapInfo_subresource_miptail
@c.record
class hipArrayMapInfo_subresource_sparseLevel(c.Struct):
  SIZE = 32
  level: int
  layer: int
  offsetX: int
  offsetY: int
  offsetZ: int
  extentWidth: int
  extentHeight: int
  extentDepth: int
hipArrayMapInfo_subresource_sparseLevel.register_fields([('level', ctypes.c_uint32, 0), ('layer', ctypes.c_uint32, 4), ('offsetX', ctypes.c_uint32, 8), ('offsetY', ctypes.c_uint32, 12), ('offsetZ', ctypes.c_uint32, 16), ('extentWidth', ctypes.c_uint32, 20), ('extentHeight', ctypes.c_uint32, 24), ('extentDepth', ctypes.c_uint32, 28)])
@c.record
class hipArrayMapInfo_subresource_miptail(c.Struct):
  SIZE = 24
  layer: int
  offset: int
  size: int
hipArrayMapInfo_subresource_miptail.register_fields([('layer', ctypes.c_uint32, 0), ('offset', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16)])
hipArrayMapInfo_subresource.register_fields([('sparseLevel', hipArrayMapInfo_subresource_sparseLevel, 0), ('miptail', hipArrayMapInfo_subresource_miptail, 0)])
@c.record
class hipArrayMapInfo_memHandle(c.Struct):
  SIZE = 8
  memHandle: c.POINTER[ihipMemGenericAllocationHandle]
hipArrayMapInfo_memHandle.register_fields([('memHandle', hipMemGenericAllocationHandle_t, 0)])
hipArrayMapInfo.register_fields([('resourceType', ctypes.c_uint32, 0), ('resource', hipArrayMapInfo_resource, 8), ('subresourceType', ctypes.c_uint32, 72), ('subresource', hipArrayMapInfo_subresource, 80), ('memOperationType', ctypes.c_uint32, 112), ('memHandleType', ctypes.c_uint32, 116), ('memHandle', hipArrayMapInfo_memHandle, 120), ('offset', ctypes.c_uint64, 128), ('deviceBitMask', ctypes.c_uint32, 136), ('flags', ctypes.c_uint32, 140), ('reserved', c.Array[ctypes.c_uint32, Literal[2]], 144)])
@c.record
class hipMemcpyNodeParams(c.Struct):
  SIZE = 176
  flags: int
  reserved: c.Array[ctypes.c_int32, Literal[3]]
  copyParams: hipMemcpy3DParms
@c.record
class hipMemcpy3DParms(c.Struct):
  SIZE = 160
  srcArray: c.POINTER[hipArray]
  srcPos: hipPos
  srcPtr: hipPitchedPtr
  dstArray: c.POINTER[hipArray]
  dstPos: hipPos
  dstPtr: hipPitchedPtr
  extent: hipExtent
  kind: int
@c.record
class hipPos(c.Struct):
  SIZE = 24
  x: int
  y: int
  z: int
hipPos.register_fields([('x', size_t, 0), ('y', size_t, 8), ('z', size_t, 16)])
@c.record
class hipPitchedPtr(c.Struct):
  SIZE = 32
  ptr: ctypes.c_void_p
  pitch: int
  xsize: int
  ysize: int
hipPitchedPtr.register_fields([('ptr', ctypes.c_void_p, 0), ('pitch', size_t, 8), ('xsize', size_t, 16), ('ysize', size_t, 24)])
hipMemcpyKind: dict[int, str] = {(hipMemcpyHostToHost:=0): 'hipMemcpyHostToHost', (hipMemcpyHostToDevice:=1): 'hipMemcpyHostToDevice', (hipMemcpyDeviceToHost:=2): 'hipMemcpyDeviceToHost', (hipMemcpyDeviceToDevice:=3): 'hipMemcpyDeviceToDevice', (hipMemcpyDefault:=4): 'hipMemcpyDefault', (hipMemcpyDeviceToDeviceNoCU:=1024): 'hipMemcpyDeviceToDeviceNoCU'}
hipMemcpy3DParms.register_fields([('srcArray', hipArray_t, 0), ('srcPos', hipPos, 8), ('srcPtr', hipPitchedPtr, 32), ('dstArray', hipArray_t, 64), ('dstPos', hipPos, 72), ('dstPtr', hipPitchedPtr, 96), ('extent', hipExtent, 128), ('kind', ctypes.c_uint32, 152)])
hipMemcpyNodeParams.register_fields([('flags', ctypes.c_int32, 0), ('reserved', c.Array[ctypes.c_int32, Literal[3]], 4), ('copyParams', hipMemcpy3DParms, 16)])
@c.record
class hipChildGraphNodeParams(c.Struct):
  SIZE = 8
  graph: c.POINTER[ihipGraph]
hipChildGraphNodeParams.register_fields([('graph', hipGraph_t, 0)])
@c.record
class hipEventWaitNodeParams(c.Struct):
  SIZE = 8
  event: c.POINTER[ihipEvent_t]
hipEventWaitNodeParams.register_fields([('event', hipEvent_t, 0)])
@c.record
class hipEventRecordNodeParams(c.Struct):
  SIZE = 8
  event: c.POINTER[ihipEvent_t]
hipEventRecordNodeParams.register_fields([('event', hipEvent_t, 0)])
@c.record
class hipMemFreeNodeParams(c.Struct):
  SIZE = 8
  dptr: ctypes.c_void_p
hipMemFreeNodeParams.register_fields([('dptr', ctypes.c_void_p, 0)])
@c.record
class hipGraphNodeParams(c.Struct):
  SIZE = 256
  type: int
  reserved0: c.Array[ctypes.c_int32, Literal[3]]
  reserved1: c.Array[ctypes.c_int64, Literal[29]]
  kernel: hipKernelNodeParams
  memcpy: hipMemcpyNodeParams
  memset: hipMemsetParams
  host: hipHostNodeParams
  graph: hipChildGraphNodeParams
  eventWait: hipEventWaitNodeParams
  eventRecord: hipEventRecordNodeParams
  extSemSignal: hipExternalSemaphoreSignalNodeParams
  extSemWait: hipExternalSemaphoreWaitNodeParams
  alloc: hipMemAllocNodeParams
  free: hipMemFreeNodeParams
  reserved2: int
hipGraphNodeParams.register_fields([('type', ctypes.c_uint32, 0), ('reserved0', c.Array[ctypes.c_int32, Literal[3]], 4), ('reserved1', c.Array[ctypes.c_int64, Literal[29]], 16), ('kernel', hipKernelNodeParams, 16), ('memcpy', hipMemcpyNodeParams, 16), ('memset', hipMemsetParams, 16), ('host', hipHostNodeParams, 16), ('graph', hipChildGraphNodeParams, 16), ('eventWait', hipEventWaitNodeParams, 16), ('eventRecord', hipEventRecordNodeParams, 16), ('extSemSignal', hipExternalSemaphoreSignalNodeParams, 16), ('extSemWait', hipExternalSemaphoreWaitNodeParams, 16), ('alloc', hipMemAllocNodeParams, 16), ('free', hipMemFreeNodeParams, 16), ('reserved2', ctypes.c_int64, 248)])
hipGraphDependencyType: dict[int, str] = {(hipGraphDependencyTypeDefault:=0): 'hipGraphDependencyTypeDefault', (hipGraphDependencyTypeProgrammatic:=1): 'hipGraphDependencyTypeProgrammatic'}
@c.record
class hipGraphEdgeData(c.Struct):
  SIZE = 8
  from_port: int
  reserved: c.Array[ctypes.c_ubyte, Literal[5]]
  to_port: int
  type: int
hipGraphEdgeData.register_fields([('from_port', ctypes.c_ubyte, 0), ('reserved', c.Array[ctypes.c_ubyte, Literal[5]], 1), ('to_port', ctypes.c_ubyte, 6), ('type', ctypes.c_ubyte, 7)])
@c.record
class hipLaunchAttribute_st(c.Struct):
  SIZE = 72
  id: int
  pad: c.Array[ctypes.c_char, Literal[4]]
  val: hipLaunchAttributeValue
  value: hipLaunchAttributeValue
hipLaunchAttribute_st.register_fields([('id', ctypes.c_uint32, 0), ('pad', c.Array[ctypes.c_char, Literal[4]], 4), ('val', hipLaunchAttributeValue, 8), ('value', hipLaunchAttributeValue, 8)])
hipLaunchAttribute: TypeAlias = hipLaunchAttribute_st
@c.record
class hipLaunchConfig_st(c.Struct):
  SIZE = 56
  gridDim: dim3
  blockDim: dim3
  dynamicSmemBytes: int
  stream: c.POINTER[ihipStream_t]
  attrs: c.POINTER[hipLaunchAttribute_st]
  numAttrs: int
hipLaunchConfig_st.register_fields([('gridDim', dim3, 0), ('blockDim', dim3, 12), ('dynamicSmemBytes', size_t, 24), ('stream', hipStream_t, 32), ('attrs', c.POINTER[hipLaunchAttribute], 40), ('numAttrs', ctypes.c_uint32, 48)])
hipLaunchConfig_t: TypeAlias = hipLaunchConfig_st
@c.record
class HIP_LAUNCH_CONFIG_st(c.Struct):
  SIZE = 56
  gridDimX: int
  gridDimY: int
  gridDimZ: int
  blockDimX: int
  blockDimY: int
  blockDimZ: int
  sharedMemBytes: int
  hStream: c.POINTER[ihipStream_t]
  attrs: c.POINTER[hipLaunchAttribute_st]
  numAttrs: int
HIP_LAUNCH_CONFIG_st.register_fields([('gridDimX', ctypes.c_uint32, 0), ('gridDimY', ctypes.c_uint32, 4), ('gridDimZ', ctypes.c_uint32, 8), ('blockDimX', ctypes.c_uint32, 12), ('blockDimY', ctypes.c_uint32, 16), ('blockDimZ', ctypes.c_uint32, 20), ('sharedMemBytes', ctypes.c_uint32, 24), ('hStream', hipStream_t, 32), ('attrs', c.POINTER[hipLaunchAttribute], 40), ('numAttrs', ctypes.c_uint32, 48)])
HIP_LAUNCH_CONFIG: TypeAlias = HIP_LAUNCH_CONFIG_st
hipMemRangeHandleType: dict[int, str] = {(hipMemRangeHandleTypeDmaBufFd:=1): 'hipMemRangeHandleTypeDmaBufFd', (hipMemRangeHandleTypeMax:=2147483647): 'hipMemRangeHandleTypeMax'}
hipMemRangeFlags: dict[int, str] = {(hipMemRangeFlagDmaBufMappingTypePcie:=1): 'hipMemRangeFlagDmaBufMappingTypePcie', (hipMemRangeFlagsMax:=2147483647): 'hipMemRangeFlagsMax'}
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipInit(flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32])
def hipDriverGetVersion(driverVersion:c.POINTER[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32])
def hipRuntimeGetVersion(runtimeVersion:c.POINTER[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipDevice_t], ctypes.c_int32)
def hipDeviceGet(device:c.POINTER[hipDevice_t], ordinal:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[ctypes.c_int32], hipDevice_t)
def hipDeviceComputeCapability(major:c.POINTER[ctypes.c_int32], minor:c.POINTER[ctypes.c_int32], device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_char], ctypes.c_int32, hipDevice_t)
def hipDeviceGetName(name:c.POINTER[ctypes.c_char], len:int, device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipUUID], hipDevice_t)
def hipDeviceGetUuid(uuid:c.POINTER[hipUUID], device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32)
def hipDeviceGetP2PAttribute(value:c.POINTER[ctypes.c_int32], attr:ctypes.c_uint32, srcDevice:int, dstDevice:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_char], ctypes.c_int32, ctypes.c_int32)
def hipDeviceGetPCIBusId(pciBusId:c.POINTER[ctypes.c_char], len:int, device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[ctypes.c_char])
def hipDeviceGetByPCIBusId(device:c.POINTER[ctypes.c_int32], pciBusId:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], hipDevice_t)
def hipDeviceTotalMem(bytes:c.POINTER[size_t], device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipDeviceSynchronize() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipDeviceReset() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def hipSetDevice(deviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], ctypes.c_int32)
def hipSetValidDevices(device_arr:c.POINTER[ctypes.c_int32], len:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32])
def hipGetDevice(deviceId:c.POINTER[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32])
def hipGetDeviceCount(count:c.POINTER[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], ctypes.c_uint32, ctypes.c_int32)
def hipDeviceGetAttribute(pi:c.POINTER[ctypes.c_int32], attr:ctypes.c_uint32, deviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemPool_t], ctypes.c_int32)
def hipDeviceGetDefaultMemPool(mem_pool:c.POINTER[hipMemPool_t], device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, hipMemPool_t)
def hipDeviceSetMemPool(device:int, mem_pool:hipMemPool_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemPool_t], ctypes.c_int32)
def hipDeviceGetMemPool(mem_pool:c.POINTER[hipMemPool_t], device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipDeviceProp_tR0600], ctypes.c_int32)
def hipGetDevicePropertiesR0600(prop:c.POINTER[hipDeviceProp_tR0600], deviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], c.POINTER[hipChannelFormatDesc], ctypes.c_int32)
def hipDeviceGetTexture1DLinearMaxWidth(max_width:c.POINTER[size_t], desc:c.POINTER[hipChannelFormatDesc], device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipDeviceSetCacheConfig(cacheConfig:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32])
def hipDeviceGetCacheConfig(cacheConfig:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], ctypes.c_uint32)
def hipDeviceGetLimit(pValue:c.POINTER[size_t], limit:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, size_t)
def hipDeviceSetLimit(limit:ctypes.c_uint32, value:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32])
def hipDeviceGetSharedMemConfig(pConfig:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32])
def hipGetDeviceFlags(flags:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipDeviceSetSharedMemConfig(config:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipSetDeviceFlags(flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[hipDeviceProp_tR0600])
def hipChooseDeviceR0600(device:c.POINTER[ctypes.c_int32], prop:c.POINTER[hipDeviceProp_tR0600]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32, c.POINTER[uint32_t], c.POINTER[uint32_t])
def hipExtGetLinkTypeAndHopCount(device1:int, device2:int, linktype:c.POINTER[uint32_t], hopcount:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipIpcMemHandle_t], ctypes.c_void_p)
def hipIpcGetMemHandle(handle:c.POINTER[hipIpcMemHandle_t], devPtr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], hipIpcMemHandle_t, ctypes.c_uint32)
def hipIpcOpenMemHandle(devPtr:c.POINTER[ctypes.c_void_p], handle:hipIpcMemHandle_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipIpcCloseMemHandle(devPtr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipIpcEventHandle_t], hipEvent_t)
def hipIpcGetEventHandle(handle:c.POINTER[hipIpcEventHandle_t], event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipEvent_t], hipIpcEventHandle_t)
def hipIpcOpenEventHandle(event:c.POINTER[hipEvent_t], handle:hipIpcEventHandle_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32)
def hipFuncSetAttribute(func:ctypes.c_void_p, attr:ctypes.c_uint32, value:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
def hipFuncSetCacheConfig(func:ctypes.c_void_p, config:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
def hipFuncSetSharedMemConfig(func:ctypes.c_void_p, config:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipGetLastError() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipExtGetLastError() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipPeekAtLastError() -> ctypes.c_uint32: ...
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_uint32)
def hipGetErrorName(hip_error:ctypes.c_uint32) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_uint32)
def hipGetErrorString(hipError:ctypes.c_uint32) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, c.POINTER[c.POINTER[ctypes.c_char]])
def hipDrvGetErrorName(hipError:ctypes.c_uint32, errorString:c.POINTER[c.POINTER[ctypes.c_char]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, c.POINTER[c.POINTER[ctypes.c_char]])
def hipDrvGetErrorString(hipError:ctypes.c_uint32, errorString:c.POINTER[c.POINTER[ctypes.c_char]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipStream_t])
def hipStreamCreate(stream:c.POINTER[hipStream_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipStream_t], ctypes.c_uint32)
def hipStreamCreateWithFlags(stream:c.POINTER[hipStream_t], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipStream_t], ctypes.c_uint32, ctypes.c_int32)
def hipStreamCreateWithPriority(stream:c.POINTER[hipStream_t], flags:int, priority:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[ctypes.c_int32])
def hipDeviceGetStreamPriorityRange(leastPriority:c.POINTER[ctypes.c_int32], greatestPriority:c.POINTER[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t)
def hipStreamDestroy(stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t)
def hipStreamQuery(stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t)
def hipStreamSynchronize(stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, hipEvent_t, ctypes.c_uint32)
def hipStreamWaitEvent(stream:hipStream_t, event:hipEvent_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[ctypes.c_uint32])
def hipStreamGetFlags(stream:hipStream_t, flags:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[ctypes.c_uint64])
def hipStreamGetId(stream:hipStream_t, streamId:c.POINTER[ctypes.c_uint64]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[ctypes.c_int32])
def hipStreamGetPriority(stream:hipStream_t, priority:c.POINTER[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[hipDevice_t])
def hipStreamGetDevice(stream:hipStream_t, device:c.POINTER[hipDevice_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipStream_t], uint32_t, c.POINTER[uint32_t])
def hipExtStreamCreateWithCUMask(stream:c.POINTER[hipStream_t], cuMaskSize:uint32_t, cuMask:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, uint32_t, c.POINTER[uint32_t])
def hipExtStreamGetCUMask(stream:hipStream_t, cuMaskSize:uint32_t, cuMask:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
hipStreamCallback_t: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[ihipStream_t], ctypes.c_uint32, ctypes.c_void_p]]
@dll.bind(ctypes.c_uint32, hipStream_t, hipStreamCallback_t, ctypes.c_void_p, ctypes.c_uint32)
def hipStreamAddCallback(stream:hipStream_t, callback:hipStreamCallback_t, userData:ctypes.c_void_p, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_uint32, c.POINTER[hipLaunchAttributeValue])
def hipStreamSetAttribute(stream:hipStream_t, attr:ctypes.c_uint32, value:c.POINTER[hipLaunchAttributeValue]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_uint32, c.POINTER[hipLaunchAttributeValue])
def hipStreamGetAttribute(stream:hipStream_t, attr:ctypes.c_uint32, value_out:c.POINTER[hipLaunchAttributeValue]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint32, uint32_t)
def hipStreamWaitValue32(stream:hipStream_t, ptr:ctypes.c_void_p, value:uint32_t, flags:int, mask:uint32_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint32, uint64_t)
def hipStreamWaitValue64(stream:hipStream_t, ptr:ctypes.c_void_p, value:uint64_t, flags:int, mask:uint64_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint32)
def hipStreamWriteValue32(stream:hipStream_t, ptr:ctypes.c_void_p, value:uint32_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint32)
def hipStreamWriteValue64(stream:hipStream_t, ptr:ctypes.c_void_p, value:uint64_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_uint32, c.POINTER[hipStreamBatchMemOpParams], ctypes.c_uint32)
def hipStreamBatchMemOp(stream:hipStream_t, count:int, paramArray:c.POINTER[hipStreamBatchMemOpParams], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipBatchMemOpNodeParams])
def hipGraphAddBatchMemOpNode(phGraphNode:c.POINTER[hipGraphNode_t], hGraph:hipGraph_t, dependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, nodeParams:c.POINTER[hipBatchMemOpNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipBatchMemOpNodeParams])
def hipGraphBatchMemOpNodeGetParams(hNode:hipGraphNode_t, nodeParams_out:c.POINTER[hipBatchMemOpNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipBatchMemOpNodeParams])
def hipGraphBatchMemOpNodeSetParams(hNode:hipGraphNode_t, nodeParams:c.POINTER[hipBatchMemOpNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipBatchMemOpNodeParams])
def hipGraphExecBatchMemOpNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:c.POINTER[hipBatchMemOpNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipEvent_t], ctypes.c_uint32)
def hipEventCreateWithFlags(event:c.POINTER[hipEvent_t], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipEvent_t])
def hipEventCreate(event:c.POINTER[hipEvent_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t, hipStream_t, ctypes.c_uint32)
def hipEventRecordWithFlags(event:hipEvent_t, stream:hipStream_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t, hipStream_t)
def hipEventRecord(event:hipEvent_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t)
def hipEventDestroy(event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t)
def hipEventSynchronize(event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_float], hipEvent_t, hipEvent_t)
def hipEventElapsedTime(ms:c.POINTER[ctypes.c_float], start:hipEvent_t, stop:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t)
def hipEventQuery(event:hipEvent_t) -> ctypes.c_uint32: ...
hipPointer_attribute: dict[int, str] = {(HIP_POINTER_ATTRIBUTE_CONTEXT:=1): 'HIP_POINTER_ATTRIBUTE_CONTEXT', (HIP_POINTER_ATTRIBUTE_MEMORY_TYPE:=2): 'HIP_POINTER_ATTRIBUTE_MEMORY_TYPE', (HIP_POINTER_ATTRIBUTE_DEVICE_POINTER:=3): 'HIP_POINTER_ATTRIBUTE_DEVICE_POINTER', (HIP_POINTER_ATTRIBUTE_HOST_POINTER:=4): 'HIP_POINTER_ATTRIBUTE_HOST_POINTER', (HIP_POINTER_ATTRIBUTE_P2P_TOKENS:=5): 'HIP_POINTER_ATTRIBUTE_P2P_TOKENS', (HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS:=6): 'HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS', (HIP_POINTER_ATTRIBUTE_BUFFER_ID:=7): 'HIP_POINTER_ATTRIBUTE_BUFFER_ID', (HIP_POINTER_ATTRIBUTE_IS_MANAGED:=8): 'HIP_POINTER_ATTRIBUTE_IS_MANAGED', (HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL:=9): 'HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL', (HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE:=10): 'HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE', (HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR:=11): 'HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR', (HIP_POINTER_ATTRIBUTE_RANGE_SIZE:=12): 'HIP_POINTER_ATTRIBUTE_RANGE_SIZE', (HIP_POINTER_ATTRIBUTE_MAPPED:=13): 'HIP_POINTER_ATTRIBUTE_MAPPED', (HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:=14): 'HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES', (HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:=15): 'HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE', (HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS:=16): 'HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS', (HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:=17): 'HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE'}
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, hipDeviceptr_t)
def hipPointerSetAttribute(value:ctypes.c_void_p, attribute:ctypes.c_uint32, ptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipPointerAttribute_t], ctypes.c_void_p)
def hipPointerGetAttributes(attributes:c.POINTER[hipPointerAttribute_t], ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, hipDeviceptr_t)
def hipPointerGetAttribute(data:ctypes.c_void_p, attribute:ctypes.c_uint32, ptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_void_p], hipDeviceptr_t)
def hipDrvPointerGetAttributes(numAttributes:int, attributes:c.POINTER[ctypes.c_uint32], data:c.POINTER[ctypes.c_void_p], ptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipExternalSemaphore_t], c.POINTER[hipExternalSemaphoreHandleDesc])
def hipImportExternalSemaphore(extSem_out:c.POINTER[hipExternalSemaphore_t], semHandleDesc:c.POINTER[hipExternalSemaphoreHandleDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipExternalSemaphore_t], c.POINTER[hipExternalSemaphoreSignalParams], ctypes.c_uint32, hipStream_t)
def hipSignalExternalSemaphoresAsync(extSemArray:c.POINTER[hipExternalSemaphore_t], paramsArray:c.POINTER[hipExternalSemaphoreSignalParams], numExtSems:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipExternalSemaphore_t], c.POINTER[hipExternalSemaphoreWaitParams], ctypes.c_uint32, hipStream_t)
def hipWaitExternalSemaphoresAsync(extSemArray:c.POINTER[hipExternalSemaphore_t], paramsArray:c.POINTER[hipExternalSemaphoreWaitParams], numExtSems:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipExternalSemaphore_t)
def hipDestroyExternalSemaphore(extSem:hipExternalSemaphore_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipExternalMemory_t], c.POINTER[hipExternalMemoryHandleDesc])
def hipImportExternalMemory(extMem_out:c.POINTER[hipExternalMemory_t], memHandleDesc:c.POINTER[hipExternalMemoryHandleDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], hipExternalMemory_t, c.POINTER[hipExternalMemoryBufferDesc])
def hipExternalMemoryGetMappedBuffer(devPtr:c.POINTER[ctypes.c_void_p], extMem:hipExternalMemory_t, bufferDesc:c.POINTER[hipExternalMemoryBufferDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipExternalMemory_t)
def hipDestroyExternalMemory(extMem:hipExternalMemory_t) -> ctypes.c_uint32: ...
hipMipmappedArray_t: TypeAlias = c.POINTER[hipMipmappedArray]
@dll.bind(ctypes.c_uint32, c.POINTER[hipMipmappedArray_t], hipExternalMemory_t, c.POINTER[hipExternalMemoryMipmappedArrayDesc])
def hipExternalMemoryGetMappedMipmappedArray(mipmap:c.POINTER[hipMipmappedArray_t], extMem:hipExternalMemory_t, mipmapDesc:c.POINTER[hipExternalMemoryMipmappedArrayDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t)
def hipMalloc(ptr:c.POINTER[ctypes.c_void_p], size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, ctypes.c_uint32)
def hipExtMallocWithFlags(ptr:c.POINTER[ctypes.c_void_p], sizeBytes:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t)
def hipMallocHost(ptr:c.POINTER[ctypes.c_void_p], size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t)
def hipMemAllocHost(ptr:c.POINTER[ctypes.c_void_p], size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, ctypes.c_uint32)
def hipHostMalloc(ptr:c.POINTER[ctypes.c_void_p], size:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, ctypes.c_uint32)
def hipMallocManaged(dev_ptr:c.POINTER[ctypes.c_void_p], size:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_int32, hipStream_t)
def hipMemPrefetchAsync(dev_ptr:ctypes.c_void_p, count:size_t, device:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, hipMemLocation, ctypes.c_uint32, hipStream_t)
def hipMemPrefetchAsync_v2(dev_ptr:ctypes.c_void_p, count:size_t, location:hipMemLocation, flags:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_uint32, ctypes.c_int32)
def hipMemAdvise(dev_ptr:ctypes.c_void_p, count:size_t, advice:ctypes.c_uint32, device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_uint32, hipMemLocation)
def hipMemAdvise_v2(dev_ptr:ctypes.c_void_p, count:size_t, advice:ctypes.c_uint32, location:hipMemLocation) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_uint32, ctypes.c_void_p, size_t)
def hipMemRangeGetAttribute(data:ctypes.c_void_p, data_size:size_t, attribute:ctypes.c_uint32, dev_ptr:ctypes.c_void_p, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], c.POINTER[size_t], c.POINTER[ctypes.c_uint32], size_t, ctypes.c_void_p, size_t)
def hipMemRangeGetAttributes(data:c.POINTER[ctypes.c_void_p], data_sizes:c.POINTER[size_t], attributes:c.POINTER[ctypes.c_uint32], num_attributes:size_t, dev_ptr:ctypes.c_void_p, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipStreamAttachMemAsync(stream:hipStream_t, dev_ptr:ctypes.c_void_p, length:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, hipStream_t)
def hipMallocAsync(dev_ptr:c.POINTER[ctypes.c_void_p], size:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipStream_t)
def hipFreeAsync(dev_ptr:ctypes.c_void_p, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t, size_t)
def hipMemPoolTrimTo(mem_pool:hipMemPool_t, min_bytes_to_hold:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t, ctypes.c_uint32, ctypes.c_void_p)
def hipMemPoolSetAttribute(mem_pool:hipMemPool_t, attr:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t, ctypes.c_uint32, ctypes.c_void_p)
def hipMemPoolGetAttribute(mem_pool:hipMemPool_t, attr:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t, c.POINTER[hipMemAccessDesc], size_t)
def hipMemPoolSetAccess(mem_pool:hipMemPool_t, desc_list:c.POINTER[hipMemAccessDesc], count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], hipMemPool_t, c.POINTER[hipMemLocation])
def hipMemPoolGetAccess(flags:c.POINTER[ctypes.c_uint32], mem_pool:hipMemPool_t, location:c.POINTER[hipMemLocation]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemPool_t], c.POINTER[hipMemPoolProps])
def hipMemPoolCreate(mem_pool:c.POINTER[hipMemPool_t], pool_props:c.POINTER[hipMemPoolProps]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t)
def hipMemPoolDestroy(mem_pool:hipMemPool_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, hipMemPool_t, hipStream_t)
def hipMallocFromPoolAsync(dev_ptr:c.POINTER[ctypes.c_void_p], size:size_t, mem_pool:hipMemPool_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipMemPool_t, ctypes.c_uint32, ctypes.c_uint32)
def hipMemPoolExportToShareableHandle(shared_handle:ctypes.c_void_p, mem_pool:hipMemPool_t, handle_type:ctypes.c_uint32, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemPool_t], ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)
def hipMemPoolImportFromShareableHandle(mem_pool:c.POINTER[hipMemPool_t], shared_handle:ctypes.c_void_p, handle_type:ctypes.c_uint32, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemPoolPtrExportData], ctypes.c_void_p)
def hipMemPoolExportPointer(export_data:c.POINTER[hipMemPoolPtrExportData], dev_ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], hipMemPool_t, c.POINTER[hipMemPoolPtrExportData])
def hipMemPoolImportPointer(dev_ptr:c.POINTER[ctypes.c_void_p], mem_pool:hipMemPool_t, export_data:c.POINTER[hipMemPoolPtrExportData]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, ctypes.c_uint32)
def hipHostAlloc(ptr:c.POINTER[ctypes.c_void_p], size:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], ctypes.c_void_p, ctypes.c_uint32)
def hipHostGetDevicePointer(devPtr:c.POINTER[ctypes.c_void_p], hstPtr:ctypes.c_void_p, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], ctypes.c_void_p)
def hipHostGetFlags(flagsPtr:c.POINTER[ctypes.c_uint32], hostPtr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipHostRegister(hostPtr:ctypes.c_void_p, sizeBytes:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipHostUnregister(hostPtr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], c.POINTER[size_t], size_t, size_t)
def hipMallocPitch(ptr:c.POINTER[ctypes.c_void_p], pitch:c.POINTER[size_t], width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipDeviceptr_t], c.POINTER[size_t], size_t, size_t, ctypes.c_uint32)
def hipMemAllocPitch(dptr:c.POINTER[hipDeviceptr_t], pitch:c.POINTER[size_t], widthInBytes:size_t, height:size_t, elementSizeBytes:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipFree(ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipFreeHost(ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipHostFree(ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipMemcpy(dst:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpyWithStream(dst:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_void_p, size_t)
def hipMemcpyHtoD(dst:hipDeviceptr_t, src:ctypes.c_void_p, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipDeviceptr_t, size_t)
def hipMemcpyDtoH(dst:ctypes.c_void_p, src:hipDeviceptr_t, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, hipDeviceptr_t, size_t)
def hipMemcpyDtoD(dst:hipDeviceptr_t, src:hipDeviceptr_t, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, hipArray_t, size_t, size_t)
def hipMemcpyAtoD(dstDevice:hipDeviceptr_t, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, hipDeviceptr_t, size_t)
def hipMemcpyDtoA(dstArray:hipArray_t, dstOffset:size_t, srcDevice:hipDeviceptr_t, ByteCount:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, hipArray_t, size_t, size_t)
def hipMemcpyAtoA(dstArray:hipArray_t, dstOffset:size_t, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_void_p, size_t, hipStream_t)
def hipMemcpyHtoDAsync(dst:hipDeviceptr_t, src:ctypes.c_void_p, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipDeviceptr_t, size_t, hipStream_t)
def hipMemcpyDtoHAsync(dst:ctypes.c_void_p, src:hipDeviceptr_t, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t)
def hipMemcpyDtoDAsync(dst:hipDeviceptr_t, src:hipDeviceptr_t, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipArray_t, size_t, size_t, hipStream_t)
def hipMemcpyAtoHAsync(dstHost:ctypes.c_void_p, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, ctypes.c_void_p, size_t, hipStream_t)
def hipMemcpyHtoAAsync(dstArray:hipArray_t, dstOffset:size_t, srcHost:ctypes.c_void_p, ByteCount:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipDeviceptr_t], c.POINTER[size_t], hipModule_t, c.POINTER[ctypes.c_char])
def hipModuleGetGlobal(dptr:c.POINTER[hipDeviceptr_t], bytes:c.POINTER[size_t], hmod:hipModule_t, name:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], ctypes.c_void_p)
def hipGetSymbolAddress(devPtr:c.POINTER[ctypes.c_void_p], symbol:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], ctypes.c_void_p)
def hipGetSymbolSize(size:c.POINTER[size_t], symbol:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p], ctypes.c_int32, uint64_t, c.POINTER[ctypes.c_uint32])
def hipGetProcAddress(symbol:c.POINTER[ctypes.c_char], pfn:c.POINTER[ctypes.c_void_p], hipVersion:int, flags:uint64_t, symbolStatus:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipMemcpyToSymbol(symbol:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpyToSymbolAsync(symbol:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, offset:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipMemcpyFromSymbol(dst:ctypes.c_void_p, symbol:ctypes.c_void_p, sizeBytes:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpyFromSymbolAsync(dst:ctypes.c_void_p, symbol:ctypes.c_void_p, sizeBytes:size_t, offset:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpyAsync(dst:ctypes.c_void_p, src:ctypes.c_void_p, sizeBytes:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, size_t)
def hipMemset(dst:ctypes.c_void_p, value:int, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_ubyte, size_t)
def hipMemsetD8(dest:hipDeviceptr_t, value:int, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_ubyte, size_t, hipStream_t)
def hipMemsetD8Async(dest:hipDeviceptr_t, value:int, count:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_uint16, size_t)
def hipMemsetD16(dest:hipDeviceptr_t, value:int, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_uint16, size_t, hipStream_t)
def hipMemsetD16Async(dest:hipDeviceptr_t, value:int, count:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_int32, size_t)
def hipMemsetD32(dest:hipDeviceptr_t, value:int, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, size_t, hipStream_t)
def hipMemsetAsync(dst:ctypes.c_void_p, value:int, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_int32, size_t, hipStream_t)
def hipMemsetD32Async(dst:hipDeviceptr_t, value:int, count:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_int32, size_t, size_t)
def hipMemset2D(dst:ctypes.c_void_p, pitch:size_t, value:int, width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_int32, size_t, size_t, hipStream_t)
def hipMemset2DAsync(dst:ctypes.c_void_p, pitch:size_t, value:int, width:size_t, height:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipPitchedPtr, ctypes.c_int32, hipExtent)
def hipMemset3D(pitchedDevPtr:hipPitchedPtr, value:int, extent:hipExtent) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipPitchedPtr, ctypes.c_int32, hipExtent, hipStream_t)
def hipMemset3DAsync(pitchedDevPtr:hipPitchedPtr, value:int, extent:hipExtent, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_ubyte, size_t, size_t)
def hipMemsetD2D8(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_ubyte, size_t, size_t, hipStream_t)
def hipMemsetD2D8Async(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_uint16, size_t, size_t)
def hipMemsetD2D16(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_uint16, size_t, size_t, hipStream_t)
def hipMemsetD2D16Async(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_uint32, size_t, size_t)
def hipMemsetD2D32(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_uint32, size_t, size_t, hipStream_t)
def hipMemsetD2D32Async(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], c.POINTER[size_t])
def hipMemGetInfo(free:c.POINTER[size_t], total:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, c.POINTER[size_t])
def hipMemPtrGetInfo(ptr:ctypes.c_void_p, size:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipArray_t], c.POINTER[hipChannelFormatDesc], size_t, size_t, ctypes.c_uint32)
def hipMallocArray(array:c.POINTER[hipArray_t], desc:c.POINTER[hipChannelFormatDesc], width:size_t, height:size_t, flags:int) -> ctypes.c_uint32: ...
@c.record
class HIP_ARRAY_DESCRIPTOR(c.Struct):
  SIZE = 24
  Width: int
  Height: int
  Format: int
  NumChannels: int
HIP_ARRAY_DESCRIPTOR.register_fields([('Width', size_t, 0), ('Height', size_t, 8), ('Format', ctypes.c_uint32, 16), ('NumChannels', ctypes.c_uint32, 20)])
@dll.bind(ctypes.c_uint32, c.POINTER[hipArray_t], c.POINTER[HIP_ARRAY_DESCRIPTOR])
def hipArrayCreate(pHandle:c.POINTER[hipArray_t], pAllocateArray:c.POINTER[HIP_ARRAY_DESCRIPTOR]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t)
def hipArrayDestroy(array:hipArray_t) -> ctypes.c_uint32: ...
@c.record
class HIP_ARRAY3D_DESCRIPTOR(c.Struct):
  SIZE = 40
  Width: int
  Height: int
  Depth: int
  Format: int
  NumChannels: int
  Flags: int
HIP_ARRAY3D_DESCRIPTOR.register_fields([('Width', size_t, 0), ('Height', size_t, 8), ('Depth', size_t, 16), ('Format', ctypes.c_uint32, 24), ('NumChannels', ctypes.c_uint32, 28), ('Flags', ctypes.c_uint32, 32)])
@dll.bind(ctypes.c_uint32, c.POINTER[hipArray_t], c.POINTER[HIP_ARRAY3D_DESCRIPTOR])
def hipArray3DCreate(array:c.POINTER[hipArray_t], pAllocateArray:c.POINTER[HIP_ARRAY3D_DESCRIPTOR]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipPitchedPtr], hipExtent)
def hipMalloc3D(pitchedDevPtr:c.POINTER[hipPitchedPtr], extent:hipExtent) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t)
def hipFreeArray(array:hipArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipArray_t], c.POINTER[hipChannelFormatDesc], hipExtent, ctypes.c_uint32)
def hipMalloc3DArray(array:c.POINTER[hipArray_t], desc:c.POINTER[hipChannelFormatDesc], extent:hipExtent, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipChannelFormatDesc], c.POINTER[hipExtent], c.POINTER[ctypes.c_uint32], hipArray_t)
def hipArrayGetInfo(desc:c.POINTER[hipChannelFormatDesc], extent:c.POINTER[hipExtent], flags:c.POINTER[ctypes.c_uint32], array:hipArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[HIP_ARRAY_DESCRIPTOR], hipArray_t)
def hipArrayGetDescriptor(pArrayDescriptor:c.POINTER[HIP_ARRAY_DESCRIPTOR], array:hipArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[HIP_ARRAY3D_DESCRIPTOR], hipArray_t)
def hipArray3DGetDescriptor(pArrayDescriptor:c.POINTER[HIP_ARRAY3D_DESCRIPTOR], array:hipArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpy2D(dst:ctypes.c_void_p, dpitch:size_t, src:ctypes.c_void_p, spitch:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@c.record
class hip_Memcpy2D(c.Struct):
  SIZE = 128
  srcXInBytes: int
  srcY: int
  srcMemoryType: int
  srcHost: ctypes.c_void_p
  srcDevice: ctypes.c_void_p
  srcArray: c.POINTER[hipArray]
  srcPitch: int
  dstXInBytes: int
  dstY: int
  dstMemoryType: int
  dstHost: ctypes.c_void_p
  dstDevice: ctypes.c_void_p
  dstArray: c.POINTER[hipArray]
  dstPitch: int
  WidthInBytes: int
  Height: int
hip_Memcpy2D.register_fields([('srcXInBytes', size_t, 0), ('srcY', size_t, 8), ('srcMemoryType', ctypes.c_uint32, 16), ('srcHost', ctypes.c_void_p, 24), ('srcDevice', hipDeviceptr_t, 32), ('srcArray', hipArray_t, 40), ('srcPitch', size_t, 48), ('dstXInBytes', size_t, 56), ('dstY', size_t, 64), ('dstMemoryType', ctypes.c_uint32, 72), ('dstHost', ctypes.c_void_p, 80), ('dstDevice', hipDeviceptr_t, 88), ('dstArray', hipArray_t, 96), ('dstPitch', size_t, 104), ('WidthInBytes', size_t, 112), ('Height', size_t, 120)])
@dll.bind(ctypes.c_uint32, c.POINTER[hip_Memcpy2D])
def hipMemcpyParam2D(pCopy:c.POINTER[hip_Memcpy2D]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hip_Memcpy2D], hipStream_t)
def hipMemcpyParam2DAsync(pCopy:c.POINTER[hip_Memcpy2D], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpy2DAsync(dst:ctypes.c_void_p, dpitch:size_t, src:ctypes.c_void_p, spitch:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpy2DToArray(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.c_void_p, spitch:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpy2DToArrayAsync(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.c_void_p, spitch:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
hipArray_const_t: TypeAlias = c.POINTER[hipArray]
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpy2DArrayToArray(dst:hipArray_t, wOffsetDst:size_t, hOffsetDst:size_t, src:hipArray_const_t, wOffsetSrc:size_t, hOffsetSrc:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipMemcpyToArray(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.c_void_p, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipArray_const_t, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpyFromArray(dst:ctypes.c_void_p, srcArray:hipArray_const_t, wOffset:size_t, hOffset:size_t, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpy2DFromArray(dst:ctypes.c_void_p, dpitch:size_t, src:hipArray_const_t, wOffset:size_t, hOffset:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpy2DFromArrayAsync(dst:ctypes.c_void_p, dpitch:size_t, src:hipArray_const_t, wOffset:size_t, hOffset:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipArray_t, size_t, size_t)
def hipMemcpyAtoH(dst:ctypes.c_void_p, srcArray:hipArray_t, srcOffset:size_t, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, ctypes.c_void_p, size_t)
def hipMemcpyHtoA(dstArray:hipArray_t, dstOffset:size_t, srcHost:ctypes.c_void_p, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemcpy3DParms])
def hipMemcpy3D(p:c.POINTER[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemcpy3DParms], hipStream_t)
def hipMemcpy3DAsync(p:c.POINTER[hipMemcpy3DParms], stream:hipStream_t) -> ctypes.c_uint32: ...
@c.record
class HIP_MEMCPY3D(c.Struct):
  SIZE = 184
  srcXInBytes: int
  srcY: int
  srcZ: int
  srcLOD: int
  srcMemoryType: int
  srcHost: ctypes.c_void_p
  srcDevice: ctypes.c_void_p
  srcArray: c.POINTER[hipArray]
  srcPitch: int
  srcHeight: int
  dstXInBytes: int
  dstY: int
  dstZ: int
  dstLOD: int
  dstMemoryType: int
  dstHost: ctypes.c_void_p
  dstDevice: ctypes.c_void_p
  dstArray: c.POINTER[hipArray]
  dstPitch: int
  dstHeight: int
  WidthInBytes: int
  Height: int
  Depth: int
HIP_MEMCPY3D.register_fields([('srcXInBytes', size_t, 0), ('srcY', size_t, 8), ('srcZ', size_t, 16), ('srcLOD', size_t, 24), ('srcMemoryType', ctypes.c_uint32, 32), ('srcHost', ctypes.c_void_p, 40), ('srcDevice', hipDeviceptr_t, 48), ('srcArray', hipArray_t, 56), ('srcPitch', size_t, 64), ('srcHeight', size_t, 72), ('dstXInBytes', size_t, 80), ('dstY', size_t, 88), ('dstZ', size_t, 96), ('dstLOD', size_t, 104), ('dstMemoryType', ctypes.c_uint32, 112), ('dstHost', ctypes.c_void_p, 120), ('dstDevice', hipDeviceptr_t, 128), ('dstArray', hipArray_t, 136), ('dstPitch', size_t, 144), ('dstHeight', size_t, 152), ('WidthInBytes', size_t, 160), ('Height', size_t, 168), ('Depth', size_t, 176)])
@dll.bind(ctypes.c_uint32, c.POINTER[HIP_MEMCPY3D])
def hipDrvMemcpy3D(pCopy:c.POINTER[HIP_MEMCPY3D]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[HIP_MEMCPY3D], hipStream_t)
def hipDrvMemcpy3DAsync(pCopy:c.POINTER[HIP_MEMCPY3D], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipDeviceptr_t], c.POINTER[size_t], hipDeviceptr_t)
def hipMemGetAddressRange(pbase:c.POINTER[hipDeviceptr_t], psize:c.POINTER[size_t], dptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@c.record
class hipMemcpyAttributes(c.Struct):
  SIZE = 24
  srcAccessOrder: int
  srcLocHint: hipMemLocation
  dstLocHint: hipMemLocation
  flags: int
hipMemcpySrcAccessOrder: dict[int, str] = {(hipMemcpySrcAccessOrderInvalid:=0): 'hipMemcpySrcAccessOrderInvalid', (hipMemcpySrcAccessOrderStream:=1): 'hipMemcpySrcAccessOrderStream', (hipMemcpySrcAccessOrderDuringApiCall:=2): 'hipMemcpySrcAccessOrderDuringApiCall', (hipMemcpySrcAccessOrderAny:=3): 'hipMemcpySrcAccessOrderAny', (hipMemcpySrcAccessOrderMax:=2147483647): 'hipMemcpySrcAccessOrderMax'}
hipMemcpyAttributes.register_fields([('srcAccessOrder', ctypes.c_uint32, 0), ('srcLocHint', hipMemLocation, 4), ('dstLocHint', hipMemLocation, 12), ('flags', ctypes.c_uint32, 20)])
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], c.POINTER[ctypes.c_void_p], c.POINTER[size_t], size_t, c.POINTER[hipMemcpyAttributes], c.POINTER[size_t], size_t, c.POINTER[size_t], hipStream_t)
def hipMemcpyBatchAsync(dsts:c.POINTER[ctypes.c_void_p], srcs:c.POINTER[ctypes.c_void_p], sizes:c.POINTER[size_t], count:size_t, attrs:c.POINTER[hipMemcpyAttributes], attrsIdxs:c.POINTER[size_t], numAttrs:size_t, failIdx:c.POINTER[size_t], stream:hipStream_t) -> ctypes.c_uint32: ...
@c.record
class hipMemcpy3DBatchOp(c.Struct):
  SIZE = 112
  src: hipMemcpy3DOperand
  dst: hipMemcpy3DOperand
  extent: hipExtent
  srcAccessOrder: int
  flags: int
@c.record
class hipMemcpy3DOperand(c.Struct):
  SIZE = 40
  type: int
  op: hipMemcpy3DOperand_op
hipMemcpy3DOperandType: dict[int, str] = {(hipMemcpyOperandTypePointer:=1): 'hipMemcpyOperandTypePointer', (hipMemcpyOperandTypeArray:=2): 'hipMemcpyOperandTypeArray', (hipMemcpyOperandTypeMax:=2147483647): 'hipMemcpyOperandTypeMax'}
@c.record
class hipMemcpy3DOperand_op(c.Struct):
  SIZE = 32
  ptr: hipMemcpy3DOperand_op_ptr
  array: hipMemcpy3DOperand_op_array
@c.record
class hipMemcpy3DOperand_op_ptr(c.Struct):
  SIZE = 32
  ptr: ctypes.c_void_p
  rowLength: int
  layerHeight: int
  locHint: hipMemLocation
hipMemcpy3DOperand_op_ptr.register_fields([('ptr', ctypes.c_void_p, 0), ('rowLength', size_t, 8), ('layerHeight', size_t, 16), ('locHint', hipMemLocation, 24)])
@c.record
class hipMemcpy3DOperand_op_array(c.Struct):
  SIZE = 32
  array: c.POINTER[hipArray]
  offset: hipOffset3D
@c.record
class hipOffset3D(c.Struct):
  SIZE = 24
  x: int
  y: int
  z: int
hipOffset3D.register_fields([('x', size_t, 0), ('y', size_t, 8), ('z', size_t, 16)])
hipMemcpy3DOperand_op_array.register_fields([('array', hipArray_t, 0), ('offset', hipOffset3D, 8)])
hipMemcpy3DOperand_op.register_fields([('ptr', hipMemcpy3DOperand_op_ptr, 0), ('array', hipMemcpy3DOperand_op_array, 0)])
hipMemcpy3DOperand.register_fields([('type', ctypes.c_uint32, 0), ('op', hipMemcpy3DOperand_op, 8)])
hipMemcpy3DBatchOp.register_fields([('src', hipMemcpy3DOperand, 0), ('dst', hipMemcpy3DOperand, 40), ('extent', hipExtent, 80), ('srcAccessOrder', ctypes.c_uint32, 104), ('flags', ctypes.c_uint32, 108)])
@dll.bind(ctypes.c_uint32, size_t, c.POINTER[hipMemcpy3DBatchOp], c.POINTER[size_t], ctypes.c_uint64, hipStream_t)
def hipMemcpy3DBatchAsync(numOps:size_t, opList:c.POINTER[hipMemcpy3DBatchOp], failIdx:c.POINTER[size_t], flags:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@c.record
class hipMemcpy3DPeerParms(c.Struct):
  SIZE = 168
  srcArray: c.POINTER[hipArray]
  srcPos: hipPos
  srcPtr: hipPitchedPtr
  srcDevice: int
  dstArray: c.POINTER[hipArray]
  dstPos: hipPos
  dstPtr: hipPitchedPtr
  dstDevice: int
  extent: hipExtent
hipMemcpy3DPeerParms.register_fields([('srcArray', hipArray_t, 0), ('srcPos', hipPos, 8), ('srcPtr', hipPitchedPtr, 32), ('srcDevice', ctypes.c_int32, 64), ('dstArray', hipArray_t, 72), ('dstPos', hipPos, 80), ('dstPtr', hipPitchedPtr, 104), ('dstDevice', ctypes.c_int32, 136), ('extent', hipExtent, 144)])
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemcpy3DPeerParms])
def hipMemcpy3DPeer(p:c.POINTER[hipMemcpy3DPeerParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemcpy3DPeerParms], hipStream_t)
def hipMemcpy3DPeerAsync(p:c.POINTER[hipMemcpy3DPeerParms], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], ctypes.c_int32, ctypes.c_int32)
def hipDeviceCanAccessPeer(canAccessPeer:c.POINTER[ctypes.c_int32], deviceId:int, peerDeviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32)
def hipDeviceEnablePeerAccess(peerDeviceId:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def hipDeviceDisablePeerAccess(peerDeviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, size_t)
def hipMemcpyPeer(dst:ctypes.c_void_p, dstDeviceId:int, src:ctypes.c_void_p, srcDeviceId:int, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, size_t, hipStream_t)
def hipMemcpyPeerAsync(dst:ctypes.c_void_p, dstDeviceId:int, src:ctypes.c_void_p, srcDevice:int, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipCtx_t], ctypes.c_uint32, hipDevice_t)
def hipCtxCreate(ctx:c.POINTER[hipCtx_t], flags:int, device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t)
def hipCtxDestroy(ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipCtx_t])
def hipCtxPopCurrent(ctx:c.POINTER[hipCtx_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t)
def hipCtxPushCurrent(ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t)
def hipCtxSetCurrent(ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipCtx_t])
def hipCtxGetCurrent(ctx:c.POINTER[hipCtx_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipDevice_t])
def hipCtxGetDevice(device:c.POINTER[hipDevice_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t, c.POINTER[ctypes.c_uint32])
def hipCtxGetApiVersion(ctx:hipCtx_t, apiVersion:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32])
def hipCtxGetCacheConfig(cacheConfig:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipCtxSetCacheConfig(cacheConfig:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipCtxSetSharedMemConfig(config:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32])
def hipCtxGetSharedMemConfig(pConfig:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipCtxSynchronize() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32])
def hipCtxGetFlags(flags:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t, ctypes.c_uint32)
def hipCtxEnablePeerAccess(peerCtx:hipCtx_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t)
def hipCtxDisablePeerAccess(peerCtx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDevice_t, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_int32])
def hipDevicePrimaryCtxGetState(dev:hipDevice_t, flags:c.POINTER[ctypes.c_uint32], active:c.POINTER[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDevice_t)
def hipDevicePrimaryCtxRelease(dev:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipCtx_t], hipDevice_t)
def hipDevicePrimaryCtxRetain(pctx:c.POINTER[hipCtx_t], dev:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDevice_t)
def hipDevicePrimaryCtxReset(dev:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDevice_t, ctypes.c_uint32)
def hipDevicePrimaryCtxSetFlags(dev:hipDevice_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipModule_t], ctypes.c_void_p)
def hipModuleLoadFatBinary(module:c.POINTER[hipModule_t], fatbin:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipModule_t], c.POINTER[ctypes.c_char])
def hipModuleLoad(module:c.POINTER[hipModule_t], fname:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipModule_t)
def hipModuleUnload(module:hipModule_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipFunction_t], hipModule_t, c.POINTER[ctypes.c_char])
def hipModuleGetFunction(function:c.POINTER[hipFunction_t], module:hipModule_t, kname:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], hipModule_t)
def hipModuleGetFunctionCount(count:c.POINTER[ctypes.c_uint32], mod:hipModule_t) -> ctypes.c_uint32: ...
hipLibraryOption_e: dict[int, str] = {(hipLibraryHostUniversalFunctionAndDataTable:=0): 'hipLibraryHostUniversalFunctionAndDataTable', (hipLibraryBinaryIsPreserved:=1): 'hipLibraryBinaryIsPreserved'}
hipLibraryOption: TypeAlias = ctypes.c_uint32
@dll.bind(ctypes.c_uint32, c.POINTER[hipLibrary_t], ctypes.c_void_p, c.POINTER[c.POINTER[ctypes.c_uint32]], c.POINTER[ctypes.c_void_p], ctypes.c_uint32, c.POINTER[c.POINTER[hipLibraryOption]], c.POINTER[ctypes.c_void_p], ctypes.c_uint32)
def hipLibraryLoadData(library:c.POINTER[hipLibrary_t], code:ctypes.c_void_p, jitOptions:c.POINTER[c.POINTER[ctypes.c_uint32]], jitOptionsValues:c.POINTER[ctypes.c_void_p], numJitOptions:int, libraryOptions:c.POINTER[c.POINTER[hipLibraryOption]], libraryOptionValues:c.POINTER[ctypes.c_void_p], numLibraryOptions:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipLibrary_t], c.POINTER[ctypes.c_char], c.POINTER[c.POINTER[ctypes.c_uint32]], c.POINTER[ctypes.c_void_p], ctypes.c_uint32, c.POINTER[c.POINTER[hipLibraryOption]], c.POINTER[ctypes.c_void_p], ctypes.c_uint32)
def hipLibraryLoadFromFile(library:c.POINTER[hipLibrary_t], fileName:c.POINTER[ctypes.c_char], jitOptions:c.POINTER[c.POINTER[ctypes.c_uint32]], jitOptionsValues:c.POINTER[ctypes.c_void_p], numJitOptions:int, libraryOptions:c.POINTER[c.POINTER[hipLibraryOption]], libraryOptionValues:c.POINTER[ctypes.c_void_p], numLibraryOptions:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLibrary_t)
def hipLibraryUnload(library:hipLibrary_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipKernel_t], hipLibrary_t, c.POINTER[ctypes.c_char])
def hipLibraryGetKernel(pKernel:c.POINTER[hipKernel_t], library:hipLibrary_t, name:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], hipLibrary_t)
def hipLibraryGetKernelCount(count:c.POINTER[ctypes.c_uint32], library:hipLibrary_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipFuncAttributes], ctypes.c_void_p)
def hipFuncGetAttributes(attr:c.POINTER[hipFuncAttributes], func:ctypes.c_void_p) -> ctypes.c_uint32: ...
hipFunction_attribute: dict[int, str] = {(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:=0): 'HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK', (HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:=1): 'HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES', (HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:=2): 'HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES', (HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:=3): 'HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', (HIP_FUNC_ATTRIBUTE_NUM_REGS:=4): 'HIP_FUNC_ATTRIBUTE_NUM_REGS', (HIP_FUNC_ATTRIBUTE_PTX_VERSION:=5): 'HIP_FUNC_ATTRIBUTE_PTX_VERSION', (HIP_FUNC_ATTRIBUTE_BINARY_VERSION:=6): 'HIP_FUNC_ATTRIBUTE_BINARY_VERSION', (HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:=7): 'HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA', (HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:=8): 'HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES', (HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:=9): 'HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT', (HIP_FUNC_ATTRIBUTE_MAX:=10): 'HIP_FUNC_ATTRIBUTE_MAX'}
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], ctypes.c_uint32, hipFunction_t)
def hipFuncGetAttribute(value:c.POINTER[ctypes.c_int32], attrib:ctypes.c_uint32, hfunc:hipFunction_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipFunction_t], ctypes.c_void_p)
def hipGetFuncBySymbol(functionPtr:c.POINTER[hipFunction_t], symbolPtr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p], ctypes.c_uint64, c.POINTER[ctypes.c_uint32])
def hipGetDriverEntryPoint(symbol:c.POINTER[ctypes.c_char], funcPtr:c.POINTER[ctypes.c_void_p], flags:int, driverStatus:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@c.record
class textureReference(c.Struct):
  SIZE = 88
  normalized: int
  readMode: int
  filterMode: int
  addressMode: c.Array[ctypes.c_uint32, Literal[3]]
  channelDesc: hipChannelFormatDesc
  sRGB: int
  maxAnisotropy: int
  mipmapFilterMode: int
  mipmapLevelBias: float
  minMipmapLevelClamp: float
  maxMipmapLevelClamp: float
  textureObject: c.POINTER[__hip_texture]
  numChannels: int
  format: int
hipTextureReadMode: dict[int, str] = {(hipReadModeElementType:=0): 'hipReadModeElementType', (hipReadModeNormalizedFloat:=1): 'hipReadModeNormalizedFloat'}
hipTextureFilterMode: dict[int, str] = {(hipFilterModePoint:=0): 'hipFilterModePoint', (hipFilterModeLinear:=1): 'hipFilterModeLinear'}
hipTextureAddressMode: dict[int, str] = {(hipAddressModeWrap:=0): 'hipAddressModeWrap', (hipAddressModeClamp:=1): 'hipAddressModeClamp', (hipAddressModeMirror:=2): 'hipAddressModeMirror', (hipAddressModeBorder:=3): 'hipAddressModeBorder'}
class __hip_texture(c.Struct): pass
hipTextureObject_t: TypeAlias = c.POINTER[__hip_texture]
textureReference.register_fields([('normalized', ctypes.c_int32, 0), ('readMode', ctypes.c_uint32, 4), ('filterMode', ctypes.c_uint32, 8), ('addressMode', c.Array[ctypes.c_uint32, Literal[3]], 12), ('channelDesc', hipChannelFormatDesc, 24), ('sRGB', ctypes.c_int32, 44), ('maxAnisotropy', ctypes.c_uint32, 48), ('mipmapFilterMode', ctypes.c_uint32, 52), ('mipmapLevelBias', ctypes.c_float, 56), ('minMipmapLevelClamp', ctypes.c_float, 60), ('maxMipmapLevelClamp', ctypes.c_float, 64), ('textureObject', hipTextureObject_t, 72), ('numChannels', ctypes.c_int32, 80), ('format', ctypes.c_uint32, 84)])
@dll.bind(ctypes.c_uint32, c.POINTER[c.POINTER[textureReference]], hipModule_t, c.POINTER[ctypes.c_char])
def hipModuleGetTexRef(texRef:c.POINTER[c.POINTER[textureReference]], hmod:hipModule_t, name:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipModule_t], ctypes.c_void_p)
def hipModuleLoadData(module:c.POINTER[hipModule_t], image:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipModule_t], ctypes.c_void_p, ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_void_p])
def hipModuleLoadDataEx(module:c.POINTER[hipModule_t], image:ctypes.c_void_p, numOptions:int, options:c.POINTER[ctypes.c_uint32], optionValues:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLinkState_t, ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[ctypes.c_char], ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_void_p])
def hipLinkAddData(state:hipLinkState_t, type:ctypes.c_uint32, data:ctypes.c_void_p, size:size_t, name:c.POINTER[ctypes.c_char], numOptions:int, options:c.POINTER[ctypes.c_uint32], optionValues:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLinkState_t, ctypes.c_uint32, c.POINTER[ctypes.c_char], ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_void_p])
def hipLinkAddFile(state:hipLinkState_t, type:ctypes.c_uint32, path:c.POINTER[ctypes.c_char], numOptions:int, options:c.POINTER[ctypes.c_uint32], optionValues:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLinkState_t, c.POINTER[ctypes.c_void_p], c.POINTER[size_t])
def hipLinkComplete(state:hipLinkState_t, hipBinOut:c.POINTER[ctypes.c_void_p], sizeOut:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_void_p], c.POINTER[hipLinkState_t])
def hipLinkCreate(numOptions:int, options:c.POINTER[ctypes.c_uint32], optionValues:c.POINTER[ctypes.c_void_p], stateOut:c.POINTER[hipLinkState_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLinkState_t)
def hipLinkDestroy(state:hipLinkState_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, c.POINTER[ctypes.c_void_p], c.POINTER[ctypes.c_void_p])
def hipModuleLaunchKernel(f:hipFunction_t, gridDimX:int, gridDimY:int, gridDimZ:int, blockDimX:int, blockDimY:int, blockDimZ:int, sharedMemBytes:int, stream:hipStream_t, kernelParams:c.POINTER[ctypes.c_void_p], extra:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, c.POINTER[ctypes.c_void_p])
def hipModuleLaunchCooperativeKernel(f:hipFunction_t, gridDimX:int, gridDimY:int, gridDimZ:int, blockDimX:int, blockDimY:int, blockDimZ:int, sharedMemBytes:int, stream:hipStream_t, kernelParams:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipFunctionLaunchParams], ctypes.c_uint32, ctypes.c_uint32)
def hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList:c.POINTER[hipFunctionLaunchParams], numDevices:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, dim3, dim3, c.POINTER[ctypes.c_void_p], ctypes.c_uint32, hipStream_t)
def hipLaunchCooperativeKernel(f:ctypes.c_void_p, gridDim:dim3, blockDimX:dim3, kernelParams:c.POINTER[ctypes.c_void_p], sharedMemBytes:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipLaunchParams], ctypes.c_int32, ctypes.c_uint32)
def hipLaunchCooperativeKernelMultiDevice(launchParamsList:c.POINTER[hipLaunchParams], numDevices:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipLaunchParams], ctypes.c_int32, ctypes.c_uint32)
def hipExtLaunchMultiKernelMultiDevice(launchParamsList:c.POINTER[hipLaunchParams], numDevices:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipLaunchConfig_t], ctypes.c_void_p, c.POINTER[ctypes.c_void_p])
def hipLaunchKernelExC(config:c.POINTER[hipLaunchConfig_t], fPtr:ctypes.c_void_p, args:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[HIP_LAUNCH_CONFIG], hipFunction_t, c.POINTER[ctypes.c_void_p], c.POINTER[ctypes.c_void_p])
def hipDrvLaunchKernelEx(config:c.POINTER[HIP_LAUNCH_CONFIG], f:hipFunction_t, params:c.POINTER[ctypes.c_void_p], extra:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipDeviceptr_t, size_t, ctypes.c_uint32, ctypes.c_uint64)
def hipMemGetHandleForAddressRange(handle:ctypes.c_void_p, dptr:hipDeviceptr_t, size:size_t, handleType:ctypes.c_uint32, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[ctypes.c_int32], hipFunction_t, size_t, ctypes.c_int32)
def hipModuleOccupancyMaxPotentialBlockSize(gridSize:c.POINTER[ctypes.c_int32], blockSize:c.POINTER[ctypes.c_int32], f:hipFunction_t, dynSharedMemPerBlk:size_t, blockSizeLimit:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[ctypes.c_int32], hipFunction_t, size_t, ctypes.c_int32, ctypes.c_uint32)
def hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize:c.POINTER[ctypes.c_int32], blockSize:c.POINTER[ctypes.c_int32], f:hipFunction_t, dynSharedMemPerBlk:size_t, blockSizeLimit:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], hipFunction_t, ctypes.c_int32, size_t)
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:c.POINTER[ctypes.c_int32], f:hipFunction_t, blockSize:int, dynSharedMemPerBlk:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], hipFunction_t, ctypes.c_int32, size_t, ctypes.c_uint32)
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:c.POINTER[ctypes.c_int32], f:hipFunction_t, blockSize:int, dynSharedMemPerBlk:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], ctypes.c_void_p, ctypes.c_int32, size_t)
def hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:c.POINTER[ctypes.c_int32], f:ctypes.c_void_p, blockSize:int, dynSharedMemPerBlk:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], ctypes.c_void_p, ctypes.c_int32, size_t, ctypes.c_uint32)
def hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:c.POINTER[ctypes.c_int32], f:ctypes.c_void_p, blockSize:int, dynSharedMemPerBlk:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[ctypes.c_int32], ctypes.c_void_p, size_t, ctypes.c_int32)
def hipOccupancyMaxPotentialBlockSize(gridSize:c.POINTER[ctypes.c_int32], blockSize:c.POINTER[ctypes.c_int32], f:ctypes.c_void_p, dynSharedMemPerBlk:size_t, blockSizeLimit:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipProfilerStart() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipProfilerStop() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, dim3, dim3, size_t, hipStream_t)
def hipConfigureCall(gridDim:dim3, blockDim:dim3, sharedMem:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, size_t)
def hipSetupArgument(arg:ctypes.c_void_p, size:size_t, offset:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipLaunchByPtr(func:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, dim3, dim3, size_t, hipStream_t)
def __hipPushCallConfiguration(gridDim:dim3, blockDim:dim3, sharedMem:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[dim3], c.POINTER[dim3], c.POINTER[size_t], c.POINTER[hipStream_t])
def __hipPopCallConfiguration(gridDim:c.POINTER[dim3], blockDim:c.POINTER[dim3], sharedMem:c.POINTER[size_t], stream:c.POINTER[hipStream_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, dim3, dim3, c.POINTER[ctypes.c_void_p], size_t, hipStream_t)
def hipLaunchKernel(function_address:ctypes.c_void_p, numBlocks:dim3, dimBlocks:dim3, args:c.POINTER[ctypes.c_void_p], sharedMemBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, hipHostFn_t, ctypes.c_void_p)
def hipLaunchHostFunc(stream:hipStream_t, fn:hipHostFn_t, userData:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hip_Memcpy2D])
def hipDrvMemcpy2DUnaligned(pCopy:c.POINTER[hip_Memcpy2D]) -> ctypes.c_uint32: ...
@c.record
class hipResourceDesc(c.Struct):
  SIZE = 64
  resType: int
  res: hipResourceDesc_res
@c.record
class hipResourceDesc_res(c.Struct):
  SIZE = 56
  array: hipResourceDesc_res_array
  mipmap: hipResourceDesc_res_mipmap
  linear: hipResourceDesc_res_linear
  pitch2D: hipResourceDesc_res_pitch2D
@c.record
class hipResourceDesc_res_array(c.Struct):
  SIZE = 8
  array: c.POINTER[hipArray]
hipResourceDesc_res_array.register_fields([('array', hipArray_t, 0)])
@c.record
class hipResourceDesc_res_mipmap(c.Struct):
  SIZE = 8
  mipmap: c.POINTER[hipMipmappedArray]
hipResourceDesc_res_mipmap.register_fields([('mipmap', hipMipmappedArray_t, 0)])
@c.record
class hipResourceDesc_res_linear(c.Struct):
  SIZE = 40
  devPtr: ctypes.c_void_p
  desc: hipChannelFormatDesc
  sizeInBytes: int
hipResourceDesc_res_linear.register_fields([('devPtr', ctypes.c_void_p, 0), ('desc', hipChannelFormatDesc, 8), ('sizeInBytes', size_t, 32)])
@c.record
class hipResourceDesc_res_pitch2D(c.Struct):
  SIZE = 56
  devPtr: ctypes.c_void_p
  desc: hipChannelFormatDesc
  width: int
  height: int
  pitchInBytes: int
hipResourceDesc_res_pitch2D.register_fields([('devPtr', ctypes.c_void_p, 0), ('desc', hipChannelFormatDesc, 8), ('width', size_t, 32), ('height', size_t, 40), ('pitchInBytes', size_t, 48)])
hipResourceDesc_res.register_fields([('array', hipResourceDesc_res_array, 0), ('mipmap', hipResourceDesc_res_mipmap, 0), ('linear', hipResourceDesc_res_linear, 0), ('pitch2D', hipResourceDesc_res_pitch2D, 0)])
hipResourceDesc.register_fields([('resType', ctypes.c_uint32, 0), ('res', hipResourceDesc_res, 8)])
@c.record
class hipTextureDesc(c.Struct):
  SIZE = 64
  addressMode: c.Array[ctypes.c_uint32, Literal[3]]
  filterMode: int
  readMode: int
  sRGB: int
  borderColor: c.Array[ctypes.c_float, Literal[4]]
  normalizedCoords: int
  maxAnisotropy: int
  mipmapFilterMode: int
  mipmapLevelBias: float
  minMipmapLevelClamp: float
  maxMipmapLevelClamp: float
hipTextureDesc.register_fields([('addressMode', c.Array[ctypes.c_uint32, Literal[3]], 0), ('filterMode', ctypes.c_uint32, 12), ('readMode', ctypes.c_uint32, 16), ('sRGB', ctypes.c_int32, 20), ('borderColor', c.Array[ctypes.c_float, Literal[4]], 24), ('normalizedCoords', ctypes.c_int32, 40), ('maxAnisotropy', ctypes.c_uint32, 44), ('mipmapFilterMode', ctypes.c_uint32, 48), ('mipmapLevelBias', ctypes.c_float, 52), ('minMipmapLevelClamp', ctypes.c_float, 56), ('maxMipmapLevelClamp', ctypes.c_float, 60)])
@c.record
class hipResourceViewDesc(c.Struct):
  SIZE = 48
  format: int
  width: int
  height: int
  depth: int
  firstMipmapLevel: int
  lastMipmapLevel: int
  firstLayer: int
  lastLayer: int
hipResourceViewFormat: dict[int, str] = {(hipResViewFormatNone:=0): 'hipResViewFormatNone', (hipResViewFormatUnsignedChar1:=1): 'hipResViewFormatUnsignedChar1', (hipResViewFormatUnsignedChar2:=2): 'hipResViewFormatUnsignedChar2', (hipResViewFormatUnsignedChar4:=3): 'hipResViewFormatUnsignedChar4', (hipResViewFormatSignedChar1:=4): 'hipResViewFormatSignedChar1', (hipResViewFormatSignedChar2:=5): 'hipResViewFormatSignedChar2', (hipResViewFormatSignedChar4:=6): 'hipResViewFormatSignedChar4', (hipResViewFormatUnsignedShort1:=7): 'hipResViewFormatUnsignedShort1', (hipResViewFormatUnsignedShort2:=8): 'hipResViewFormatUnsignedShort2', (hipResViewFormatUnsignedShort4:=9): 'hipResViewFormatUnsignedShort4', (hipResViewFormatSignedShort1:=10): 'hipResViewFormatSignedShort1', (hipResViewFormatSignedShort2:=11): 'hipResViewFormatSignedShort2', (hipResViewFormatSignedShort4:=12): 'hipResViewFormatSignedShort4', (hipResViewFormatUnsignedInt1:=13): 'hipResViewFormatUnsignedInt1', (hipResViewFormatUnsignedInt2:=14): 'hipResViewFormatUnsignedInt2', (hipResViewFormatUnsignedInt4:=15): 'hipResViewFormatUnsignedInt4', (hipResViewFormatSignedInt1:=16): 'hipResViewFormatSignedInt1', (hipResViewFormatSignedInt2:=17): 'hipResViewFormatSignedInt2', (hipResViewFormatSignedInt4:=18): 'hipResViewFormatSignedInt4', (hipResViewFormatHalf1:=19): 'hipResViewFormatHalf1', (hipResViewFormatHalf2:=20): 'hipResViewFormatHalf2', (hipResViewFormatHalf4:=21): 'hipResViewFormatHalf4', (hipResViewFormatFloat1:=22): 'hipResViewFormatFloat1', (hipResViewFormatFloat2:=23): 'hipResViewFormatFloat2', (hipResViewFormatFloat4:=24): 'hipResViewFormatFloat4', (hipResViewFormatUnsignedBlockCompressed1:=25): 'hipResViewFormatUnsignedBlockCompressed1', (hipResViewFormatUnsignedBlockCompressed2:=26): 'hipResViewFormatUnsignedBlockCompressed2', (hipResViewFormatUnsignedBlockCompressed3:=27): 'hipResViewFormatUnsignedBlockCompressed3', (hipResViewFormatUnsignedBlockCompressed4:=28): 'hipResViewFormatUnsignedBlockCompressed4', (hipResViewFormatSignedBlockCompressed4:=29): 'hipResViewFormatSignedBlockCompressed4', (hipResViewFormatUnsignedBlockCompressed5:=30): 'hipResViewFormatUnsignedBlockCompressed5', (hipResViewFormatSignedBlockCompressed5:=31): 'hipResViewFormatSignedBlockCompressed5', (hipResViewFormatUnsignedBlockCompressed6H:=32): 'hipResViewFormatUnsignedBlockCompressed6H', (hipResViewFormatSignedBlockCompressed6H:=33): 'hipResViewFormatSignedBlockCompressed6H', (hipResViewFormatUnsignedBlockCompressed7:=34): 'hipResViewFormatUnsignedBlockCompressed7'}
hipResourceViewDesc.register_fields([('format', ctypes.c_uint32, 0), ('width', size_t, 8), ('height', size_t, 16), ('depth', size_t, 24), ('firstMipmapLevel', ctypes.c_uint32, 32), ('lastMipmapLevel', ctypes.c_uint32, 36), ('firstLayer', ctypes.c_uint32, 40), ('lastLayer', ctypes.c_uint32, 44)])
@dll.bind(ctypes.c_uint32, c.POINTER[hipTextureObject_t], c.POINTER[hipResourceDesc], c.POINTER[hipTextureDesc], c.POINTER[hipResourceViewDesc])
def hipCreateTextureObject(pTexObject:c.POINTER[hipTextureObject_t], pResDesc:c.POINTER[hipResourceDesc], pTexDesc:c.POINTER[hipTextureDesc], pResViewDesc:c.POINTER[hipResourceViewDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipTextureObject_t)
def hipDestroyTextureObject(textureObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipChannelFormatDesc], hipArray_const_t)
def hipGetChannelDesc(desc:c.POINTER[hipChannelFormatDesc], array:hipArray_const_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipResourceDesc], hipTextureObject_t)
def hipGetTextureObjectResourceDesc(pResDesc:c.POINTER[hipResourceDesc], textureObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipResourceViewDesc], hipTextureObject_t)
def hipGetTextureObjectResourceViewDesc(pResViewDesc:c.POINTER[hipResourceViewDesc], textureObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipTextureDesc], hipTextureObject_t)
def hipGetTextureObjectTextureDesc(pTexDesc:c.POINTER[hipTextureDesc], textureObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@c.record
class HIP_RESOURCE_DESC_st(c.Struct):
  SIZE = 144
  resType: int
  res: HIP_RESOURCE_DESC_st_res
  flags: int
HIP_RESOURCE_DESC: TypeAlias = HIP_RESOURCE_DESC_st
HIPresourcetype_enum: dict[int, str] = {(HIP_RESOURCE_TYPE_ARRAY:=0): 'HIP_RESOURCE_TYPE_ARRAY', (HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY:=1): 'HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY', (HIP_RESOURCE_TYPE_LINEAR:=2): 'HIP_RESOURCE_TYPE_LINEAR', (HIP_RESOURCE_TYPE_PITCH2D:=3): 'HIP_RESOURCE_TYPE_PITCH2D'}
HIPresourcetype: TypeAlias = ctypes.c_uint32
@c.record
class HIP_RESOURCE_DESC_st_res(c.Struct):
  SIZE = 128
  array: HIP_RESOURCE_DESC_st_res_array
  mipmap: HIP_RESOURCE_DESC_st_res_mipmap
  linear: HIP_RESOURCE_DESC_st_res_linear
  pitch2D: HIP_RESOURCE_DESC_st_res_pitch2D
  reserved: HIP_RESOURCE_DESC_st_res_reserved
@c.record
class HIP_RESOURCE_DESC_st_res_array(c.Struct):
  SIZE = 8
  hArray: c.POINTER[hipArray]
HIP_RESOURCE_DESC_st_res_array.register_fields([('hArray', hipArray_t, 0)])
@c.record
class HIP_RESOURCE_DESC_st_res_mipmap(c.Struct):
  SIZE = 8
  hMipmappedArray: c.POINTER[hipMipmappedArray]
HIP_RESOURCE_DESC_st_res_mipmap.register_fields([('hMipmappedArray', hipMipmappedArray_t, 0)])
@c.record
class HIP_RESOURCE_DESC_st_res_linear(c.Struct):
  SIZE = 24
  devPtr: ctypes.c_void_p
  format: int
  numChannels: int
  sizeInBytes: int
HIP_RESOURCE_DESC_st_res_linear.register_fields([('devPtr', hipDeviceptr_t, 0), ('format', ctypes.c_uint32, 8), ('numChannels', ctypes.c_uint32, 12), ('sizeInBytes', size_t, 16)])
@c.record
class HIP_RESOURCE_DESC_st_res_pitch2D(c.Struct):
  SIZE = 40
  devPtr: ctypes.c_void_p
  format: int
  numChannels: int
  width: int
  height: int
  pitchInBytes: int
HIP_RESOURCE_DESC_st_res_pitch2D.register_fields([('devPtr', hipDeviceptr_t, 0), ('format', ctypes.c_uint32, 8), ('numChannels', ctypes.c_uint32, 12), ('width', size_t, 16), ('height', size_t, 24), ('pitchInBytes', size_t, 32)])
@c.record
class HIP_RESOURCE_DESC_st_res_reserved(c.Struct):
  SIZE = 128
  reserved: c.Array[ctypes.c_int32, Literal[32]]
HIP_RESOURCE_DESC_st_res_reserved.register_fields([('reserved', c.Array[ctypes.c_int32, Literal[32]], 0)])
HIP_RESOURCE_DESC_st_res.register_fields([('array', HIP_RESOURCE_DESC_st_res_array, 0), ('mipmap', HIP_RESOURCE_DESC_st_res_mipmap, 0), ('linear', HIP_RESOURCE_DESC_st_res_linear, 0), ('pitch2D', HIP_RESOURCE_DESC_st_res_pitch2D, 0), ('reserved', HIP_RESOURCE_DESC_st_res_reserved, 0)])
HIP_RESOURCE_DESC_st.register_fields([('resType', HIPresourcetype, 0), ('res', HIP_RESOURCE_DESC_st_res, 8), ('flags', ctypes.c_uint32, 136)])
@c.record
class HIP_TEXTURE_DESC_st(c.Struct):
  SIZE = 104
  addressMode: c.Array[ctypes.c_uint32, Literal[3]]
  filterMode: int
  flags: int
  maxAnisotropy: int
  mipmapFilterMode: int
  mipmapLevelBias: float
  minMipmapLevelClamp: float
  maxMipmapLevelClamp: float
  borderColor: c.Array[ctypes.c_float, Literal[4]]
  reserved: c.Array[ctypes.c_int32, Literal[12]]
HIP_TEXTURE_DESC: TypeAlias = HIP_TEXTURE_DESC_st
HIPaddress_mode_enum: dict[int, str] = {(HIP_TR_ADDRESS_MODE_WRAP:=0): 'HIP_TR_ADDRESS_MODE_WRAP', (HIP_TR_ADDRESS_MODE_CLAMP:=1): 'HIP_TR_ADDRESS_MODE_CLAMP', (HIP_TR_ADDRESS_MODE_MIRROR:=2): 'HIP_TR_ADDRESS_MODE_MIRROR', (HIP_TR_ADDRESS_MODE_BORDER:=3): 'HIP_TR_ADDRESS_MODE_BORDER'}
HIPaddress_mode: TypeAlias = ctypes.c_uint32
HIPfilter_mode_enum: dict[int, str] = {(HIP_TR_FILTER_MODE_POINT:=0): 'HIP_TR_FILTER_MODE_POINT', (HIP_TR_FILTER_MODE_LINEAR:=1): 'HIP_TR_FILTER_MODE_LINEAR'}
HIPfilter_mode: TypeAlias = ctypes.c_uint32
HIP_TEXTURE_DESC_st.register_fields([('addressMode', c.Array[HIPaddress_mode, Literal[3]], 0), ('filterMode', HIPfilter_mode, 12), ('flags', ctypes.c_uint32, 16), ('maxAnisotropy', ctypes.c_uint32, 20), ('mipmapFilterMode', HIPfilter_mode, 24), ('mipmapLevelBias', ctypes.c_float, 28), ('minMipmapLevelClamp', ctypes.c_float, 32), ('maxMipmapLevelClamp', ctypes.c_float, 36), ('borderColor', c.Array[ctypes.c_float, Literal[4]], 40), ('reserved', c.Array[ctypes.c_int32, Literal[12]], 56)])
@c.record
class HIP_RESOURCE_VIEW_DESC_st(c.Struct):
  SIZE = 112
  format: int
  width: int
  height: int
  depth: int
  firstMipmapLevel: int
  lastMipmapLevel: int
  firstLayer: int
  lastLayer: int
  reserved: c.Array[ctypes.c_uint32, Literal[16]]
HIP_RESOURCE_VIEW_DESC: TypeAlias = HIP_RESOURCE_VIEW_DESC_st
HIPresourceViewFormat_enum: dict[int, str] = {(HIP_RES_VIEW_FORMAT_NONE:=0): 'HIP_RES_VIEW_FORMAT_NONE', (HIP_RES_VIEW_FORMAT_UINT_1X8:=1): 'HIP_RES_VIEW_FORMAT_UINT_1X8', (HIP_RES_VIEW_FORMAT_UINT_2X8:=2): 'HIP_RES_VIEW_FORMAT_UINT_2X8', (HIP_RES_VIEW_FORMAT_UINT_4X8:=3): 'HIP_RES_VIEW_FORMAT_UINT_4X8', (HIP_RES_VIEW_FORMAT_SINT_1X8:=4): 'HIP_RES_VIEW_FORMAT_SINT_1X8', (HIP_RES_VIEW_FORMAT_SINT_2X8:=5): 'HIP_RES_VIEW_FORMAT_SINT_2X8', (HIP_RES_VIEW_FORMAT_SINT_4X8:=6): 'HIP_RES_VIEW_FORMAT_SINT_4X8', (HIP_RES_VIEW_FORMAT_UINT_1X16:=7): 'HIP_RES_VIEW_FORMAT_UINT_1X16', (HIP_RES_VIEW_FORMAT_UINT_2X16:=8): 'HIP_RES_VIEW_FORMAT_UINT_2X16', (HIP_RES_VIEW_FORMAT_UINT_4X16:=9): 'HIP_RES_VIEW_FORMAT_UINT_4X16', (HIP_RES_VIEW_FORMAT_SINT_1X16:=10): 'HIP_RES_VIEW_FORMAT_SINT_1X16', (HIP_RES_VIEW_FORMAT_SINT_2X16:=11): 'HIP_RES_VIEW_FORMAT_SINT_2X16', (HIP_RES_VIEW_FORMAT_SINT_4X16:=12): 'HIP_RES_VIEW_FORMAT_SINT_4X16', (HIP_RES_VIEW_FORMAT_UINT_1X32:=13): 'HIP_RES_VIEW_FORMAT_UINT_1X32', (HIP_RES_VIEW_FORMAT_UINT_2X32:=14): 'HIP_RES_VIEW_FORMAT_UINT_2X32', (HIP_RES_VIEW_FORMAT_UINT_4X32:=15): 'HIP_RES_VIEW_FORMAT_UINT_4X32', (HIP_RES_VIEW_FORMAT_SINT_1X32:=16): 'HIP_RES_VIEW_FORMAT_SINT_1X32', (HIP_RES_VIEW_FORMAT_SINT_2X32:=17): 'HIP_RES_VIEW_FORMAT_SINT_2X32', (HIP_RES_VIEW_FORMAT_SINT_4X32:=18): 'HIP_RES_VIEW_FORMAT_SINT_4X32', (HIP_RES_VIEW_FORMAT_FLOAT_1X16:=19): 'HIP_RES_VIEW_FORMAT_FLOAT_1X16', (HIP_RES_VIEW_FORMAT_FLOAT_2X16:=20): 'HIP_RES_VIEW_FORMAT_FLOAT_2X16', (HIP_RES_VIEW_FORMAT_FLOAT_4X16:=21): 'HIP_RES_VIEW_FORMAT_FLOAT_4X16', (HIP_RES_VIEW_FORMAT_FLOAT_1X32:=22): 'HIP_RES_VIEW_FORMAT_FLOAT_1X32', (HIP_RES_VIEW_FORMAT_FLOAT_2X32:=23): 'HIP_RES_VIEW_FORMAT_FLOAT_2X32', (HIP_RES_VIEW_FORMAT_FLOAT_4X32:=24): 'HIP_RES_VIEW_FORMAT_FLOAT_4X32', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC1:=25): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC1', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC2:=26): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC2', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC3:=27): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC3', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC4:=28): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC4', (HIP_RES_VIEW_FORMAT_SIGNED_BC4:=29): 'HIP_RES_VIEW_FORMAT_SIGNED_BC4', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC5:=30): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC5', (HIP_RES_VIEW_FORMAT_SIGNED_BC5:=31): 'HIP_RES_VIEW_FORMAT_SIGNED_BC5', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H:=32): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H', (HIP_RES_VIEW_FORMAT_SIGNED_BC6H:=33): 'HIP_RES_VIEW_FORMAT_SIGNED_BC6H', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC7:=34): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC7'}
HIPresourceViewFormat: TypeAlias = ctypes.c_uint32
HIP_RESOURCE_VIEW_DESC_st.register_fields([('format', HIPresourceViewFormat, 0), ('width', size_t, 8), ('height', size_t, 16), ('depth', size_t, 24), ('firstMipmapLevel', ctypes.c_uint32, 32), ('lastMipmapLevel', ctypes.c_uint32, 36), ('firstLayer', ctypes.c_uint32, 40), ('lastLayer', ctypes.c_uint32, 44), ('reserved', c.Array[ctypes.c_uint32, Literal[16]], 48)])
@dll.bind(ctypes.c_uint32, c.POINTER[hipTextureObject_t], c.POINTER[HIP_RESOURCE_DESC], c.POINTER[HIP_TEXTURE_DESC], c.POINTER[HIP_RESOURCE_VIEW_DESC])
def hipTexObjectCreate(pTexObject:c.POINTER[hipTextureObject_t], pResDesc:c.POINTER[HIP_RESOURCE_DESC], pTexDesc:c.POINTER[HIP_TEXTURE_DESC], pResViewDesc:c.POINTER[HIP_RESOURCE_VIEW_DESC]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipTextureObject_t)
def hipTexObjectDestroy(texObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[HIP_RESOURCE_DESC], hipTextureObject_t)
def hipTexObjectGetResourceDesc(pResDesc:c.POINTER[HIP_RESOURCE_DESC], texObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[HIP_RESOURCE_VIEW_DESC], hipTextureObject_t)
def hipTexObjectGetResourceViewDesc(pResViewDesc:c.POINTER[HIP_RESOURCE_VIEW_DESC], texObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[HIP_TEXTURE_DESC], hipTextureObject_t)
def hipTexObjectGetTextureDesc(pTexDesc:c.POINTER[HIP_TEXTURE_DESC], texObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMipmappedArray_t], c.POINTER[hipChannelFormatDesc], hipExtent, ctypes.c_uint32, ctypes.c_uint32)
def hipMallocMipmappedArray(mipmappedArray:c.POINTER[hipMipmappedArray_t], desc:c.POINTER[hipChannelFormatDesc], extent:hipExtent, numLevels:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMipmappedArray_t)
def hipFreeMipmappedArray(mipmappedArray:hipMipmappedArray_t) -> ctypes.c_uint32: ...
hipMipmappedArray_const_t: TypeAlias = c.POINTER[hipMipmappedArray]
@dll.bind(ctypes.c_uint32, c.POINTER[hipArray_t], hipMipmappedArray_const_t, ctypes.c_uint32)
def hipGetMipmappedArrayLevel(levelArray:c.POINTER[hipArray_t], mipmappedArray:hipMipmappedArray_const_t, level:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMipmappedArray_t], c.POINTER[HIP_ARRAY3D_DESCRIPTOR], ctypes.c_uint32)
def hipMipmappedArrayCreate(pHandle:c.POINTER[hipMipmappedArray_t], pMipmappedArrayDesc:c.POINTER[HIP_ARRAY3D_DESCRIPTOR], numMipmapLevels:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMipmappedArray_t)
def hipMipmappedArrayDestroy(hMipmappedArray:hipMipmappedArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipArray_t], hipMipmappedArray_t, ctypes.c_uint32)
def hipMipmappedArrayGetLevel(pLevelArray:c.POINTER[hipArray_t], hMipMappedArray:hipMipmappedArray_t, level:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], hipMipmappedArray_const_t, c.POINTER[hipChannelFormatDesc])
def hipBindTextureToMipmappedArray(tex:c.POINTER[textureReference], mipmappedArray:hipMipmappedArray_const_t, desc:c.POINTER[hipChannelFormatDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[c.POINTER[textureReference]], ctypes.c_void_p)
def hipGetTextureReference(texref:c.POINTER[c.POINTER[textureReference]], symbol:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_float], c.POINTER[textureReference])
def hipTexRefGetBorderColor(pBorderColor:c.POINTER[ctypes.c_float], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipArray_t], c.POINTER[textureReference])
def hipTexRefGetArray(pArray:c.POINTER[hipArray_t], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], ctypes.c_int32, ctypes.c_uint32)
def hipTexRefSetAddressMode(texRef:c.POINTER[textureReference], dim:int, am:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], hipArray_const_t, ctypes.c_uint32)
def hipTexRefSetArray(tex:c.POINTER[textureReference], array:hipArray_const_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], ctypes.c_uint32)
def hipTexRefSetFilterMode(texRef:c.POINTER[textureReference], fm:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], ctypes.c_uint32)
def hipTexRefSetFlags(texRef:c.POINTER[textureReference], Flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], ctypes.c_uint32, ctypes.c_int32)
def hipTexRefSetFormat(texRef:c.POINTER[textureReference], fmt:ctypes.c_uint32, NumPackedComponents:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], c.POINTER[textureReference], ctypes.c_void_p, c.POINTER[hipChannelFormatDesc], size_t)
def hipBindTexture(offset:c.POINTER[size_t], tex:c.POINTER[textureReference], devPtr:ctypes.c_void_p, desc:c.POINTER[hipChannelFormatDesc], size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], c.POINTER[textureReference], ctypes.c_void_p, c.POINTER[hipChannelFormatDesc], size_t, size_t, size_t)
def hipBindTexture2D(offset:c.POINTER[size_t], tex:c.POINTER[textureReference], devPtr:ctypes.c_void_p, desc:c.POINTER[hipChannelFormatDesc], width:size_t, height:size_t, pitch:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], hipArray_const_t, c.POINTER[hipChannelFormatDesc])
def hipBindTextureToArray(tex:c.POINTER[textureReference], array:hipArray_const_t, desc:c.POINTER[hipChannelFormatDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], c.POINTER[textureReference])
def hipGetTextureAlignmentOffset(offset:c.POINTER[size_t], texref:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference])
def hipUnbindTexture(tex:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipDeviceptr_t], c.POINTER[textureReference])
def hipTexRefGetAddress(dev_ptr:c.POINTER[hipDeviceptr_t], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[textureReference], ctypes.c_int32)
def hipTexRefGetAddressMode(pam:c.POINTER[ctypes.c_uint32], texRef:c.POINTER[textureReference], dim:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[textureReference])
def hipTexRefGetFilterMode(pfm:c.POINTER[ctypes.c_uint32], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[textureReference])
def hipTexRefGetFlags(pFlags:c.POINTER[ctypes.c_uint32], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_int32], c.POINTER[textureReference])
def hipTexRefGetFormat(pFormat:c.POINTER[ctypes.c_uint32], pNumChannels:c.POINTER[ctypes.c_int32], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], c.POINTER[textureReference])
def hipTexRefGetMaxAnisotropy(pmaxAnsio:c.POINTER[ctypes.c_int32], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32], c.POINTER[textureReference])
def hipTexRefGetMipmapFilterMode(pfm:c.POINTER[ctypes.c_uint32], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_float], c.POINTER[textureReference])
def hipTexRefGetMipmapLevelBias(pbias:c.POINTER[ctypes.c_float], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_float], c.POINTER[ctypes.c_float], c.POINTER[textureReference])
def hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp:c.POINTER[ctypes.c_float], pmaxMipmapLevelClamp:c.POINTER[ctypes.c_float], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMipmappedArray_t], c.POINTER[textureReference])
def hipTexRefGetMipMappedArray(pArray:c.POINTER[hipMipmappedArray_t], texRef:c.POINTER[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], c.POINTER[textureReference], hipDeviceptr_t, size_t)
def hipTexRefSetAddress(ByteOffset:c.POINTER[size_t], texRef:c.POINTER[textureReference], dptr:hipDeviceptr_t, bytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], c.POINTER[HIP_ARRAY_DESCRIPTOR], hipDeviceptr_t, size_t)
def hipTexRefSetAddress2D(texRef:c.POINTER[textureReference], desc:c.POINTER[HIP_ARRAY_DESCRIPTOR], dptr:hipDeviceptr_t, Pitch:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], ctypes.c_uint32)
def hipTexRefSetMaxAnisotropy(texRef:c.POINTER[textureReference], maxAniso:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], c.POINTER[ctypes.c_float])
def hipTexRefSetBorderColor(texRef:c.POINTER[textureReference], pBorderColor:c.POINTER[ctypes.c_float]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], ctypes.c_uint32)
def hipTexRefSetMipmapFilterMode(texRef:c.POINTER[textureReference], fm:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], ctypes.c_float)
def hipTexRefSetMipmapLevelBias(texRef:c.POINTER[textureReference], bias:float) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], ctypes.c_float, ctypes.c_float)
def hipTexRefSetMipmapLevelClamp(texRef:c.POINTER[textureReference], minMipMapLevelClamp:float, maxMipMapLevelClamp:float) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[textureReference], c.POINTER[hipMipmappedArray], ctypes.c_uint32)
def hipTexRefSetMipmappedArray(texRef:c.POINTER[textureReference], mipmappedArray:c.POINTER[hipMipmappedArray], Flags:int) -> ctypes.c_uint32: ...
@dll.bind(c.POINTER[ctypes.c_char], uint32_t)
def hipApiName(id:uint32_t) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(c.POINTER[ctypes.c_char], hipFunction_t)
def hipKernelNameRef(f:hipFunction_t) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_void_p, hipStream_t)
def hipKernelNameRefByPtr(hostFunction:ctypes.c_void_p, stream:hipStream_t) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(ctypes.c_int32, hipStream_t)
def hipGetStreamDeviceId(stream:hipStream_t) -> int: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_uint32)
def hipStreamBeginCapture(stream:hipStream_t, mode:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, hipGraph_t, c.POINTER[hipGraphNode_t], c.POINTER[hipGraphEdgeData], size_t, ctypes.c_uint32)
def hipStreamBeginCaptureToGraph(stream:hipStream_t, graph:hipGraph_t, dependencies:c.POINTER[hipGraphNode_t], dependencyData:c.POINTER[hipGraphEdgeData], numDependencies:size_t, mode:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[hipGraph_t])
def hipStreamEndCapture(stream:hipStream_t, pGraph:c.POINTER[hipGraph_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_uint64])
def hipStreamGetCaptureInfo(stream:hipStream_t, pCaptureStatus:c.POINTER[ctypes.c_uint32], pId:c.POINTER[ctypes.c_uint64]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[ctypes.c_uint32], c.POINTER[ctypes.c_uint64], c.POINTER[hipGraph_t], c.POINTER[c.POINTER[hipGraphNode_t]], c.POINTER[size_t])
def hipStreamGetCaptureInfo_v2(stream:hipStream_t, captureStatus_out:c.POINTER[ctypes.c_uint32], id_out:c.POINTER[ctypes.c_uint64], graph_out:c.POINTER[hipGraph_t], dependencies_out:c.POINTER[c.POINTER[hipGraphNode_t]], numDependencies_out:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[ctypes.c_uint32])
def hipStreamIsCapturing(stream:hipStream_t, pCaptureStatus:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, c.POINTER[hipGraphNode_t], size_t, ctypes.c_uint32)
def hipStreamUpdateCaptureDependencies(stream:hipStream_t, dependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint32])
def hipThreadExchangeStreamCaptureMode(mode:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraph_t], ctypes.c_uint32)
def hipGraphCreate(pGraph:c.POINTER[hipGraph_t], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t)
def hipGraphDestroy(graph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, c.POINTER[hipGraphNode_t], c.POINTER[hipGraphNode_t], size_t)
def hipGraphAddDependencies(graph:hipGraph_t, _from:c.POINTER[hipGraphNode_t], to:c.POINTER[hipGraphNode_t], numDependencies:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, c.POINTER[hipGraphNode_t], c.POINTER[hipGraphNode_t], size_t)
def hipGraphRemoveDependencies(graph:hipGraph_t, _from:c.POINTER[hipGraphNode_t], to:c.POINTER[hipGraphNode_t], numDependencies:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, c.POINTER[hipGraphNode_t], c.POINTER[hipGraphNode_t], c.POINTER[size_t])
def hipGraphGetEdges(graph:hipGraph_t, _from:c.POINTER[hipGraphNode_t], to:c.POINTER[hipGraphNode_t], numEdges:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, c.POINTER[hipGraphNode_t], c.POINTER[size_t])
def hipGraphGetNodes(graph:hipGraph_t, nodes:c.POINTER[hipGraphNode_t], numNodes:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, c.POINTER[hipGraphNode_t], c.POINTER[size_t])
def hipGraphGetRootNodes(graph:hipGraph_t, pRootNodes:c.POINTER[hipGraphNode_t], pNumRootNodes:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipGraphNode_t], c.POINTER[size_t])
def hipGraphNodeGetDependencies(node:hipGraphNode_t, pDependencies:c.POINTER[hipGraphNode_t], pNumDependencies:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipGraphNode_t], c.POINTER[size_t])
def hipGraphNodeGetDependentNodes(node:hipGraphNode_t, pDependentNodes:c.POINTER[hipGraphNode_t], pNumDependentNodes:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[ctypes.c_uint32])
def hipGraphNodeGetType(node:hipGraphNode_t, pType:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t)
def hipGraphDestroyNode(node:hipGraphNode_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraph_t], hipGraph_t)
def hipGraphClone(pGraphClone:c.POINTER[hipGraph_t], originalGraph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraphNode_t, hipGraph_t)
def hipGraphNodeFindInClone(pNode:c.POINTER[hipGraphNode_t], originalNode:hipGraphNode_t, clonedGraph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphExec_t], hipGraph_t, c.POINTER[hipGraphNode_t], c.POINTER[ctypes.c_char], size_t)
def hipGraphInstantiate(pGraphExec:c.POINTER[hipGraphExec_t], graph:hipGraph_t, pErrorNode:c.POINTER[hipGraphNode_t], pLogBuffer:c.POINTER[ctypes.c_char], bufferSize:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphExec_t], hipGraph_t, ctypes.c_uint64)
def hipGraphInstantiateWithFlags(pGraphExec:c.POINTER[hipGraphExec_t], graph:hipGraph_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphExec_t], hipGraph_t, c.POINTER[hipGraphInstantiateParams])
def hipGraphInstantiateWithParams(pGraphExec:c.POINTER[hipGraphExec_t], graph:hipGraph_t, instantiateParams:c.POINTER[hipGraphInstantiateParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipStream_t)
def hipGraphLaunch(graphExec:hipGraphExec_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipStream_t)
def hipGraphUpload(graphExec:hipGraphExec_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipGraphNodeParams])
def hipGraphAddNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, nodeParams:c.POINTER[hipGraphNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, c.POINTER[ctypes.c_uint64])
def hipGraphExecGetFlags(graphExec:hipGraphExec_t, flags:c.POINTER[ctypes.c_uint64]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipGraphNodeParams])
def hipGraphNodeSetParams(node:hipGraphNode_t, nodeParams:c.POINTER[hipGraphNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipGraphNodeParams])
def hipGraphExecNodeSetParams(graphExec:hipGraphExec_t, node:hipGraphNode_t, nodeParams:c.POINTER[hipGraphNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t)
def hipGraphExecDestroy(graphExec:hipGraphExec_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraph_t, c.POINTER[hipGraphNode_t], c.POINTER[ctypes.c_uint32])
def hipGraphExecUpdate(hGraphExec:hipGraphExec_t, hGraph:hipGraph_t, hErrorNode_out:c.POINTER[hipGraphNode_t], updateResult_out:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipKernelNodeParams])
def hipGraphAddKernelNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pNodeParams:c.POINTER[hipKernelNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipKernelNodeParams])
def hipGraphKernelNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipKernelNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipKernelNodeParams])
def hipGraphKernelNodeSetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipKernelNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipKernelNodeParams])
def hipGraphExecKernelNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:c.POINTER[hipKernelNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[HIP_MEMCPY3D], hipCtx_t)
def hipDrvGraphAddMemcpyNode(phGraphNode:c.POINTER[hipGraphNode_t], hGraph:hipGraph_t, dependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, copyParams:c.POINTER[HIP_MEMCPY3D], ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipMemcpy3DParms])
def hipGraphAddMemcpyNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pCopyParams:c.POINTER[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipMemcpy3DParms])
def hipGraphMemcpyNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipMemcpy3DParms])
def hipGraphMemcpyNodeSetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_uint32, c.POINTER[hipLaunchAttributeValue])
def hipGraphKernelNodeSetAttribute(hNode:hipGraphNode_t, attr:ctypes.c_uint32, value:c.POINTER[hipLaunchAttributeValue]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_uint32, c.POINTER[hipLaunchAttributeValue])
def hipGraphKernelNodeGetAttribute(hNode:hipGraphNode_t, attr:ctypes.c_uint32, value:c.POINTER[hipLaunchAttributeValue]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipMemcpy3DParms])
def hipGraphExecMemcpyNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipGraphAddMemcpyNode1D(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, dst:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipGraphMemcpyNodeSetParams1D(node:hipGraphNode_t, dst:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipGraphExecMemcpyNodeSetParams1D(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, dst:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphAddMemcpyNodeFromSymbol(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, dst:ctypes.c_void_p, symbol:ctypes.c_void_p, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphMemcpyNodeSetParamsFromSymbol(node:hipGraphNode_t, dst:ctypes.c_void_p, symbol:ctypes.c_void_p, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, dst:ctypes.c_void_p, symbol:ctypes.c_void_p, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphAddMemcpyNodeToSymbol(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, symbol:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphMemcpyNodeSetParamsToSymbol(node:hipGraphNode_t, symbol:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, symbol:ctypes.c_void_p, src:ctypes.c_void_p, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipMemsetParams])
def hipGraphAddMemsetNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pMemsetParams:c.POINTER[hipMemsetParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipMemsetParams])
def hipGraphMemsetNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemsetParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipMemsetParams])
def hipGraphMemsetNodeSetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemsetParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipMemsetParams])
def hipGraphExecMemsetNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemsetParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipHostNodeParams])
def hipGraphAddHostNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pNodeParams:c.POINTER[hipHostNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipHostNodeParams])
def hipGraphHostNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipHostNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipHostNodeParams])
def hipGraphHostNodeSetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipHostNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipHostNodeParams])
def hipGraphExecHostNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:c.POINTER[hipHostNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, hipGraph_t)
def hipGraphAddChildGraphNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, childGraph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipGraph_t])
def hipGraphChildGraphNodeGetGraph(node:hipGraphNode_t, pGraph:c.POINTER[hipGraph_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, hipGraph_t)
def hipGraphExecChildGraphNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, childGraph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t)
def hipGraphAddEmptyNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, hipEvent_t)
def hipGraphAddEventRecordNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipEvent_t])
def hipGraphEventRecordNodeGetEvent(node:hipGraphNode_t, event_out:c.POINTER[hipEvent_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, hipEvent_t)
def hipGraphEventRecordNodeSetEvent(node:hipGraphNode_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, hipEvent_t)
def hipGraphExecEventRecordNodeSetEvent(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, hipEvent_t)
def hipGraphAddEventWaitNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipEvent_t])
def hipGraphEventWaitNodeGetEvent(node:hipGraphNode_t, event_out:c.POINTER[hipEvent_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, hipEvent_t)
def hipGraphEventWaitNodeSetEvent(node:hipGraphNode_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, hipEvent_t)
def hipGraphExecEventWaitNodeSetEvent(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipMemAllocNodeParams])
def hipGraphAddMemAllocNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, pNodeParams:c.POINTER[hipMemAllocNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipMemAllocNodeParams])
def hipGraphMemAllocNodeGetParams(node:hipGraphNode_t, pNodeParams:c.POINTER[hipMemAllocNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, ctypes.c_void_p)
def hipGraphAddMemFreeNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, dev_ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_void_p)
def hipGraphMemFreeNodeGetParams(node:hipGraphNode_t, dev_ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p)
def hipDeviceGetGraphMemAttribute(device:int, attr:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p)
def hipDeviceSetGraphMemAttribute(device:int, attr:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def hipDeviceGraphMemTrim(device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipUserObject_t], ctypes.c_void_p, hipHostFn_t, ctypes.c_uint32, ctypes.c_uint32)
def hipUserObjectCreate(object_out:c.POINTER[hipUserObject_t], ptr:ctypes.c_void_p, destroy:hipHostFn_t, initialRefcount:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipUserObject_t, ctypes.c_uint32)
def hipUserObjectRelease(object:hipUserObject_t, count:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipUserObject_t, ctypes.c_uint32)
def hipUserObjectRetain(object:hipUserObject_t, count:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, hipUserObject_t, ctypes.c_uint32, ctypes.c_uint32)
def hipGraphRetainUserObject(graph:hipGraph_t, object:hipUserObject_t, count:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, hipUserObject_t, ctypes.c_uint32)
def hipGraphReleaseUserObject(graph:hipGraph_t, object:hipUserObject_t, count:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, c.POINTER[ctypes.c_char], ctypes.c_uint32)
def hipGraphDebugDotPrint(graph:hipGraph_t, path:c.POINTER[ctypes.c_char], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, hipGraphNode_t)
def hipGraphKernelNodeCopyAttributes(hSrc:hipGraphNode_t, hDst:hipGraphNode_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.c_uint32)
def hipGraphNodeSetEnabled(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, isEnabled:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[ctypes.c_uint32])
def hipGraphNodeGetEnabled(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, isEnabled:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipExternalSemaphoreWaitNodeParams])
def hipGraphAddExternalSemaphoresWaitNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, nodeParams:c.POINTER[hipExternalSemaphoreWaitNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipExternalSemaphoreSignalNodeParams])
def hipGraphAddExternalSemaphoresSignalNode(pGraphNode:c.POINTER[hipGraphNode_t], graph:hipGraph_t, pDependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, nodeParams:c.POINTER[hipExternalSemaphoreSignalNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipExternalSemaphoreSignalNodeParams])
def hipGraphExternalSemaphoresSignalNodeSetParams(hNode:hipGraphNode_t, nodeParams:c.POINTER[hipExternalSemaphoreSignalNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipExternalSemaphoreWaitNodeParams])
def hipGraphExternalSemaphoresWaitNodeSetParams(hNode:hipGraphNode_t, nodeParams:c.POINTER[hipExternalSemaphoreWaitNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipExternalSemaphoreSignalNodeParams])
def hipGraphExternalSemaphoresSignalNodeGetParams(hNode:hipGraphNode_t, params_out:c.POINTER[hipExternalSemaphoreSignalNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[hipExternalSemaphoreWaitNodeParams])
def hipGraphExternalSemaphoresWaitNodeGetParams(hNode:hipGraphNode_t, params_out:c.POINTER[hipExternalSemaphoreWaitNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipExternalSemaphoreSignalNodeParams])
def hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:c.POINTER[hipExternalSemaphoreSignalNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipExternalSemaphoreWaitNodeParams])
def hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:c.POINTER[hipExternalSemaphoreWaitNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[HIP_MEMCPY3D])
def hipDrvGraphMemcpyNodeGetParams(hNode:hipGraphNode_t, nodeParams:c.POINTER[HIP_MEMCPY3D]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, c.POINTER[HIP_MEMCPY3D])
def hipDrvGraphMemcpyNodeSetParams(hNode:hipGraphNode_t, nodeParams:c.POINTER[HIP_MEMCPY3D]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, c.POINTER[hipMemsetParams], hipCtx_t)
def hipDrvGraphAddMemsetNode(phGraphNode:c.POINTER[hipGraphNode_t], hGraph:hipGraph_t, dependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, memsetParams:c.POINTER[hipMemsetParams], ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipGraphNode_t], hipGraph_t, c.POINTER[hipGraphNode_t], size_t, hipDeviceptr_t)
def hipDrvGraphAddMemFreeNode(phGraphNode:c.POINTER[hipGraphNode_t], hGraph:hipGraph_t, dependencies:c.POINTER[hipGraphNode_t], numDependencies:size_t, dptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[HIP_MEMCPY3D], hipCtx_t)
def hipDrvGraphExecMemcpyNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, copyParams:c.POINTER[HIP_MEMCPY3D], ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, c.POINTER[hipMemsetParams], hipCtx_t)
def hipDrvGraphExecMemsetNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, memsetParams:c.POINTER[hipMemsetParams], ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t)
def hipMemAddressFree(devPtr:ctypes.c_void_p, size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, size_t, ctypes.c_void_p, ctypes.c_uint64)
def hipMemAddressReserve(ptr:c.POINTER[ctypes.c_void_p], size:size_t, alignment:size_t, addr:ctypes.c_void_p, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemGenericAllocationHandle_t], size_t, c.POINTER[hipMemAllocationProp], ctypes.c_uint64)
def hipMemCreate(handle:c.POINTER[hipMemGenericAllocationHandle_t], size:size_t, prop:c.POINTER[hipMemAllocationProp], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipMemGenericAllocationHandle_t, ctypes.c_uint32, ctypes.c_uint64)
def hipMemExportToShareableHandle(shareableHandle:ctypes.c_void_p, handle:hipMemGenericAllocationHandle_t, handleType:ctypes.c_uint32, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_uint64], c.POINTER[hipMemLocation], ctypes.c_void_p)
def hipMemGetAccess(flags:c.POINTER[ctypes.c_uint64], location:c.POINTER[hipMemLocation], ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[size_t], c.POINTER[hipMemAllocationProp], ctypes.c_uint32)
def hipMemGetAllocationGranularity(granularity:c.POINTER[size_t], prop:c.POINTER[hipMemAllocationProp], option:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemAllocationProp], hipMemGenericAllocationHandle_t)
def hipMemGetAllocationPropertiesFromHandle(prop:c.POINTER[hipMemAllocationProp], handle:hipMemGenericAllocationHandle_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemGenericAllocationHandle_t], ctypes.c_void_p, ctypes.c_uint32)
def hipMemImportFromShareableHandle(handle:c.POINTER[hipMemGenericAllocationHandle_t], osHandle:ctypes.c_void_p, shHandleType:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, size_t, hipMemGenericAllocationHandle_t, ctypes.c_uint64)
def hipMemMap(ptr:ctypes.c_void_p, size:size_t, offset:size_t, handle:hipMemGenericAllocationHandle_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipArrayMapInfo], ctypes.c_uint32, hipStream_t)
def hipMemMapArrayAsync(mapInfoList:c.POINTER[hipArrayMapInfo], count:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemGenericAllocationHandle_t)
def hipMemRelease(handle:hipMemGenericAllocationHandle_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipMemGenericAllocationHandle_t], ctypes.c_void_p)
def hipMemRetainAllocationHandle(handle:c.POINTER[hipMemGenericAllocationHandle_t], addr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[hipMemAccessDesc], size_t)
def hipMemSetAccess(ptr:ctypes.c_void_p, size:size_t, desc:c.POINTER[hipMemAccessDesc], count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t)
def hipMemUnmap(ptr:ctypes.c_void_p, size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, c.POINTER[hipGraphicsResource_t], hipStream_t)
def hipGraphicsMapResources(count:int, resources:c.POINTER[hipGraphicsResource_t], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hipArray_t], hipGraphicsResource_t, ctypes.c_uint32, ctypes.c_uint32)
def hipGraphicsSubResourceGetMappedArray(array:c.POINTER[hipArray_t], resource:hipGraphicsResource_t, arrayIndex:int, mipLevel:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], c.POINTER[size_t], hipGraphicsResource_t)
def hipGraphicsResourceGetMappedPointer(devPtr:c.POINTER[ctypes.c_void_p], size:c.POINTER[size_t], resource:hipGraphicsResource_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, c.POINTER[hipGraphicsResource_t], hipStream_t)
def hipGraphicsUnmapResources(count:int, resources:c.POINTER[hipGraphicsResource_t], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphicsResource_t)
def hipGraphicsUnregisterResource(resource:hipGraphicsResource_t) -> ctypes.c_uint32: ...
class __hip_surface(c.Struct): pass
hipSurfaceObject_t: TypeAlias = c.POINTER[__hip_surface]
@dll.bind(ctypes.c_uint32, c.POINTER[hipSurfaceObject_t], c.POINTER[hipResourceDesc])
def hipCreateSurfaceObject(pSurfObject:c.POINTER[hipSurfaceObject_t], pResDesc:c.POINTER[hipResourceDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipSurfaceObject_t)
def hipDestroySurfaceObject(surfaceObject:hipSurfaceObject_t) -> ctypes.c_uint32: ...
hipmipmappedArray: TypeAlias = c.POINTER[hipMipmappedArray]
hipResourcetype: TypeAlias = ctypes.c_uint32
hipMemcpyFlags: dict[int, str] = {(hipMemcpyFlagDefault:=0): 'hipMemcpyFlagDefault', (hipMemcpyFlagPreferOverlapWithCompute:=1): 'hipMemcpyFlagPreferOverlapWithCompute'}
hiprtcJIT_option = hipJitOption
HIPRTC_JIT_MAX_REGISTERS = hipJitOptionMaxRegisters
HIPRTC_JIT_THREADS_PER_BLOCK = hipJitOptionThreadsPerBlock
HIPRTC_JIT_WALL_TIME = hipJitOptionWallTime
HIPRTC_JIT_INFO_LOG_BUFFER = hipJitOptionInfoLogBuffer
HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES = hipJitOptionInfoLogBufferSizeBytes
HIPRTC_JIT_ERROR_LOG_BUFFER = hipJitOptionErrorLogBuffer
HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = hipJitOptionErrorLogBufferSizeBytes
HIPRTC_JIT_OPTIMIZATION_LEVEL = hipJitOptionOptimizationLevel
HIPRTC_JIT_TARGET_FROM_HIPCONTEXT = hipJitOptionTargetFromContext
HIPRTC_JIT_TARGET = hipJitOptionTarget
HIPRTC_JIT_FALLBACK_STRATEGY = hipJitOptionFallbackStrategy
HIPRTC_JIT_GENERATE_DEBUG_INFO = hipJitOptionGenerateDebugInfo
HIPRTC_JIT_LOG_VERBOSE = hipJitOptionLogVerbose
HIPRTC_JIT_GENERATE_LINE_INFO = hipJitOptionGenerateLineInfo
HIPRTC_JIT_CACHE_MODE = hipJitOptionCacheMode
HIPRTC_JIT_NEW_SM3X_OPT = hipJitOptionSm3xOpt
HIPRTC_JIT_FAST_COMPILE = hipJitOptionFastCompile
HIPRTC_JIT_GLOBAL_SYMBOL_NAMES = hipJitOptionGlobalSymbolNames
HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS = hipJitOptionGlobalSymbolAddresses
HIPRTC_JIT_GLOBAL_SYMBOL_COUNT = hipJitOptionGlobalSymbolCount
HIPRTC_JIT_LTO = hipJitOptionLto
HIPRTC_JIT_FTZ = hipJitOptionFtz
HIPRTC_JIT_PREC_DIV = hipJitOptionPrecDiv
HIPRTC_JIT_PREC_SQRT = hipJitOptionPrecSqrt
HIPRTC_JIT_FMA = hipJitOptionFma
HIPRTC_JIT_POSITION_INDEPENDENT_CODE = hipJitOptionPositionIndependentCode
HIPRTC_JIT_MIN_CTA_PER_SM = hipJitOptionMinCTAPerSM
HIPRTC_JIT_MAX_THREADS_PER_BLOCK = hipJitOptionMaxThreadsPerBlock
HIPRTC_JIT_OVERRIDE_DIRECT_VALUES = hipJitOptionOverrideDirectiveValues
HIPRTC_JIT_NUM_OPTIONS = hipJitOptionNumOptions
HIPRTC_JIT_IR_TO_ISA_OPT_EXT = hipJitOptionIRtoISAOptExt
HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT = hipJitOptionIRtoISAOptCountExt
hiprtcJITInputType = hipJitInputType
HIPRTC_JIT_INPUT_CUBIN = hipJitInputCubin
HIPRTC_JIT_INPUT_PTX = hipJitInputPtx
HIPRTC_JIT_INPUT_FATBINARY = hipJitInputFatBinary
HIPRTC_JIT_INPUT_OBJECT = hipJitInputObject
HIPRTC_JIT_INPUT_LIBRARY = hipJitInputLibrary
HIPRTC_JIT_INPUT_NVVM = hipJitInputNvvm
HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = hipJitNumLegacyInputTypes
HIPRTC_JIT_INPUT_LLVM_BITCODE = hipJitInputLLVMBitcode
HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = hipJitInputLLVMBundledBitcode
HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = hipJitInputLLVMArchivesOfBundledBitcode
HIPRTC_JIT_INPUT_SPIRV = hipJitInputSpirv
HIPRTC_JIT_NUM_INPUT_TYPES = hipJitNumInputTypes
hipGetDeviceProperties = hipGetDevicePropertiesR0600
hipDeviceProp_t = hipDeviceProp_tR0600
hipChooseDevice = hipChooseDeviceR0600
GENERIC_GRID_LAUNCH = 1
HIP_DEPRECATED = lambda msg: __attribute__((deprecated(msg))) # type: ignore
hipIpcMemLazyEnablePeerAccess = 0x01
HIP_IPC_HANDLE_SIZE = 64
hipStreamDefault = 0x00
hipStreamNonBlocking = 0x01
hipEventDefault = 0x0
hipEventBlockingSync = 0x1
hipEventDisableTiming = 0x2
hipEventInterprocess = 0x4
hipEventRecordDefault = 0x00
hipEventRecordExternal = 0x01
hipEventWaitDefault = 0x00
hipEventWaitExternal = 0x01
hipEventDisableSystemFence = 0x20000000
hipEventReleaseToDevice = 0x40000000
hipEventReleaseToSystem = 0x80000000
hipEnableDefault = 0x0
hipEnableLegacyStream = 0x1
hipEnablePerThreadDefaultStream = 0x2
hipHostAllocDefault = 0x0
hipHostMallocDefault = 0x0
hipHostAllocPortable = 0x1
hipHostMallocPortable = 0x1
hipHostAllocMapped = 0x2
hipHostMallocMapped = 0x2
hipHostAllocWriteCombined = 0x4
hipHostMallocWriteCombined = 0x4
hipHostMallocUncached = 0x10000000
hipHostAllocUncached = hipHostMallocUncached
hipHostMallocNumaUser = 0x20000000
hipHostMallocCoherent = 0x40000000
hipHostMallocNonCoherent = 0x80000000
hipMemAttachGlobal = 0x01
hipMemAttachHost = 0x02
hipMemAttachSingle = 0x04
hipDeviceMallocDefault = 0x0
hipDeviceMallocFinegrained = 0x1
hipMallocSignalMemory = 0x2
hipDeviceMallocUncached = 0x3
hipDeviceMallocContiguous = 0x4
hipHostRegisterDefault = 0x0
hipHostRegisterPortable = 0x1
hipHostRegisterMapped = 0x2
hipHostRegisterIoMemory = 0x4
hipHostRegisterReadOnly = 0x08
hipExtHostRegisterCoarseGrained = 0x8
hipExtHostRegisterUncached = 0x80000000
hipDeviceScheduleAuto = 0x0
hipDeviceScheduleSpin = 0x1
hipDeviceScheduleYield = 0x2
hipDeviceScheduleBlockingSync = 0x4
hipDeviceScheduleMask = 0x7
hipDeviceMapHost = 0x8
hipDeviceLmemResizeToMax = 0x10
hipArrayDefault = 0x00
hipArrayLayered = 0x01
hipArraySurfaceLoadStore = 0x02
hipArrayCubemap = 0x04
hipArrayTextureGather = 0x08
hipOccupancyDefault = 0x00
hipOccupancyDisableCachingOverride = 0x01
hipCooperativeLaunchMultiDeviceNoPreSync = 0x01
hipCooperativeLaunchMultiDeviceNoPostSync = 0x02
hipExtAnyOrderLaunch = 0x01
hipStreamWaitValueGte = 0x0
hipStreamWaitValueEq = 0x1
hipStreamWaitValueAnd = 0x2
hipStreamWaitValueNor = 0x3
hipExternalMemoryDedicated = 0x1
hipStreamAttrID = hipLaunchAttributeID
hipStreamAttributeAccessPolicyWindow = hipLaunchAttributeAccessPolicyWindow
hipStreamAttributeSynchronizationPolicy = hipLaunchAttributeSynchronizationPolicy
hipStreamAttributeMemSyncDomainMap = hipLaunchAttributeMemSyncDomainMap
hipStreamAttributeMemSyncDomain = hipLaunchAttributeMemSyncDomain
hipStreamAttributePriority = hipLaunchAttributePriority
hipStreamAttrValue = hipLaunchAttributeValue
hipKernelNodeAttrID = hipLaunchAttributeID
hipKernelNodeAttributeAccessPolicyWindow = hipLaunchAttributeAccessPolicyWindow
hipKernelNodeAttributeCooperative = hipLaunchAttributeCooperative
hipKernelNodeAttributePriority = hipLaunchAttributePriority
hipKernelNodeAttrValue = hipLaunchAttributeValue
hipDrvLaunchAttributeCooperative = hipLaunchAttributeCooperative
hipDrvLaunchAttributeID = hipLaunchAttributeID
hipDrvLaunchAttributeValue = hipLaunchAttributeValue
hipDrvLaunchAttribute = hipLaunchAttribute
hipGraphKernelNodePortDefault = 0
hipGraphKernelNodePortLaunchCompletion = 2
hipGraphKernelNodePortProgrammatic = 1
HIP_TRSA_OVERRIDE_FORMAT = 0x01
HIP_TRSF_READ_AS_INTEGER = 0x01
HIP_TRSF_NORMALIZED_COORDINATES = 0x02
HIP_TRSF_SRGB = 0x10