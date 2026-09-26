# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
from tinygrad.helpers import WIN, OSX
import sysconfig, os
dll = c.DLL('webgpu', os.path.join(sysconfig.get_paths()['purelib'], 'pydawn', 'lib', 'libwebgpu_dawn.dll') if WIN else 'webgpu_dawn')
WGPUFlags: TypeAlias = ctypes.c_uint64
WGPUBool: TypeAlias = ctypes.c_uint32
class struct_WGPUAdapterImpl(c.Struct): pass
WGPUAdapter: TypeAlias = c.POINTER[struct_WGPUAdapterImpl]
class struct_WGPUBindGroupImpl(c.Struct): pass
WGPUBindGroup: TypeAlias = c.POINTER[struct_WGPUBindGroupImpl]
class struct_WGPUBindGroupLayoutImpl(c.Struct): pass
WGPUBindGroupLayout: TypeAlias = c.POINTER[struct_WGPUBindGroupLayoutImpl]
class struct_WGPUBufferImpl(c.Struct): pass
WGPUBuffer: TypeAlias = c.POINTER[struct_WGPUBufferImpl]
class struct_WGPUCommandBufferImpl(c.Struct): pass
WGPUCommandBuffer: TypeAlias = c.POINTER[struct_WGPUCommandBufferImpl]
class struct_WGPUCommandEncoderImpl(c.Struct): pass
WGPUCommandEncoder: TypeAlias = c.POINTER[struct_WGPUCommandEncoderImpl]
class struct_WGPUComputePassEncoderImpl(c.Struct): pass
WGPUComputePassEncoder: TypeAlias = c.POINTER[struct_WGPUComputePassEncoderImpl]
class struct_WGPUComputePipelineImpl(c.Struct): pass
WGPUComputePipeline: TypeAlias = c.POINTER[struct_WGPUComputePipelineImpl]
class struct_WGPUDeviceImpl(c.Struct): pass
WGPUDevice: TypeAlias = c.POINTER[struct_WGPUDeviceImpl]
class struct_WGPUExternalTextureImpl(c.Struct): pass
WGPUExternalTexture: TypeAlias = c.POINTER[struct_WGPUExternalTextureImpl]
class struct_WGPUInstanceImpl(c.Struct): pass
WGPUInstance: TypeAlias = c.POINTER[struct_WGPUInstanceImpl]
class struct_WGPUPipelineLayoutImpl(c.Struct): pass
WGPUPipelineLayout: TypeAlias = c.POINTER[struct_WGPUPipelineLayoutImpl]
class struct_WGPUQuerySetImpl(c.Struct): pass
WGPUQuerySet: TypeAlias = c.POINTER[struct_WGPUQuerySetImpl]
class struct_WGPUQueueImpl(c.Struct): pass
WGPUQueue: TypeAlias = c.POINTER[struct_WGPUQueueImpl]
class struct_WGPURenderBundleImpl(c.Struct): pass
WGPURenderBundle: TypeAlias = c.POINTER[struct_WGPURenderBundleImpl]
class struct_WGPURenderBundleEncoderImpl(c.Struct): pass
WGPURenderBundleEncoder: TypeAlias = c.POINTER[struct_WGPURenderBundleEncoderImpl]
class struct_WGPURenderPassEncoderImpl(c.Struct): pass
WGPURenderPassEncoder: TypeAlias = c.POINTER[struct_WGPURenderPassEncoderImpl]
class struct_WGPURenderPipelineImpl(c.Struct): pass
WGPURenderPipeline: TypeAlias = c.POINTER[struct_WGPURenderPipelineImpl]
class struct_WGPUSamplerImpl(c.Struct): pass
WGPUSampler: TypeAlias = c.POINTER[struct_WGPUSamplerImpl]
class struct_WGPUShaderModuleImpl(c.Struct): pass
WGPUShaderModule: TypeAlias = c.POINTER[struct_WGPUShaderModuleImpl]
class struct_WGPUSharedBufferMemoryImpl(c.Struct): pass
WGPUSharedBufferMemory: TypeAlias = c.POINTER[struct_WGPUSharedBufferMemoryImpl]
class struct_WGPUSharedFenceImpl(c.Struct): pass
WGPUSharedFence: TypeAlias = c.POINTER[struct_WGPUSharedFenceImpl]
class struct_WGPUSharedTextureMemoryImpl(c.Struct): pass
WGPUSharedTextureMemory: TypeAlias = c.POINTER[struct_WGPUSharedTextureMemoryImpl]
class struct_WGPUSurfaceImpl(c.Struct): pass
WGPUSurface: TypeAlias = c.POINTER[struct_WGPUSurfaceImpl]
class struct_WGPUTextureImpl(c.Struct): pass
WGPUTexture: TypeAlias = c.POINTER[struct_WGPUTextureImpl]
class struct_WGPUTextureViewImpl(c.Struct): pass
WGPUTextureView: TypeAlias = c.POINTER[struct_WGPUTextureViewImpl]
@c.record
class struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER(c.Struct):
  SIZE = 4
  unused: int
struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER.register_fields([('unused', WGPUBool, 0)])
@c.record
class struct_WGPUAdapterPropertiesD3D(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  shaderModel: int
@c.record
class struct_WGPUChainedStructOut(c.Struct):
  SIZE = 16
  next: c.POINTER[struct_WGPUChainedStructOut]
  sType: int
WGPUChainedStructOut: TypeAlias = struct_WGPUChainedStructOut
enum_WGPUSType: dict[int, str] = {(WGPUSType_ShaderSourceSPIRV:=1): 'WGPUSType_ShaderSourceSPIRV', (WGPUSType_ShaderSourceWGSL:=2): 'WGPUSType_ShaderSourceWGSL', (WGPUSType_RenderPassMaxDrawCount:=3): 'WGPUSType_RenderPassMaxDrawCount', (WGPUSType_SurfaceSourceMetalLayer:=4): 'WGPUSType_SurfaceSourceMetalLayer', (WGPUSType_SurfaceSourceWindowsHWND:=5): 'WGPUSType_SurfaceSourceWindowsHWND', (WGPUSType_SurfaceSourceXlibWindow:=6): 'WGPUSType_SurfaceSourceXlibWindow', (WGPUSType_SurfaceSourceWaylandSurface:=7): 'WGPUSType_SurfaceSourceWaylandSurface', (WGPUSType_SurfaceSourceAndroidNativeWindow:=8): 'WGPUSType_SurfaceSourceAndroidNativeWindow', (WGPUSType_SurfaceSourceXCBWindow:=9): 'WGPUSType_SurfaceSourceXCBWindow', (WGPUSType_AdapterPropertiesSubgroups:=10): 'WGPUSType_AdapterPropertiesSubgroups', (WGPUSType_TextureBindingViewDimensionDescriptor:=131072): 'WGPUSType_TextureBindingViewDimensionDescriptor', (WGPUSType_SurfaceSourceCanvasHTMLSelector_Emscripten:=262144): 'WGPUSType_SurfaceSourceCanvasHTMLSelector_Emscripten', (WGPUSType_SurfaceDescriptorFromWindowsCoreWindow:=327680): 'WGPUSType_SurfaceDescriptorFromWindowsCoreWindow', (WGPUSType_ExternalTextureBindingEntry:=327681): 'WGPUSType_ExternalTextureBindingEntry', (WGPUSType_ExternalTextureBindingLayout:=327682): 'WGPUSType_ExternalTextureBindingLayout', (WGPUSType_SurfaceDescriptorFromWindowsSwapChainPanel:=327683): 'WGPUSType_SurfaceDescriptorFromWindowsSwapChainPanel', (WGPUSType_DawnTextureInternalUsageDescriptor:=327684): 'WGPUSType_DawnTextureInternalUsageDescriptor', (WGPUSType_DawnEncoderInternalUsageDescriptor:=327685): 'WGPUSType_DawnEncoderInternalUsageDescriptor', (WGPUSType_DawnInstanceDescriptor:=327686): 'WGPUSType_DawnInstanceDescriptor', (WGPUSType_DawnCacheDeviceDescriptor:=327687): 'WGPUSType_DawnCacheDeviceDescriptor', (WGPUSType_DawnAdapterPropertiesPowerPreference:=327688): 'WGPUSType_DawnAdapterPropertiesPowerPreference', (WGPUSType_DawnBufferDescriptorErrorInfoFromWireClient:=327689): 'WGPUSType_DawnBufferDescriptorErrorInfoFromWireClient', (WGPUSType_DawnTogglesDescriptor:=327690): 'WGPUSType_DawnTogglesDescriptor', (WGPUSType_DawnShaderModuleSPIRVOptionsDescriptor:=327691): 'WGPUSType_DawnShaderModuleSPIRVOptionsDescriptor', (WGPUSType_RequestAdapterOptionsLUID:=327692): 'WGPUSType_RequestAdapterOptionsLUID', (WGPUSType_RequestAdapterOptionsGetGLProc:=327693): 'WGPUSType_RequestAdapterOptionsGetGLProc', (WGPUSType_RequestAdapterOptionsD3D11Device:=327694): 'WGPUSType_RequestAdapterOptionsD3D11Device', (WGPUSType_DawnRenderPassColorAttachmentRenderToSingleSampled:=327695): 'WGPUSType_DawnRenderPassColorAttachmentRenderToSingleSampled', (WGPUSType_RenderPassPixelLocalStorage:=327696): 'WGPUSType_RenderPassPixelLocalStorage', (WGPUSType_PipelineLayoutPixelLocalStorage:=327697): 'WGPUSType_PipelineLayoutPixelLocalStorage', (WGPUSType_BufferHostMappedPointer:=327698): 'WGPUSType_BufferHostMappedPointer', (WGPUSType_DawnExperimentalSubgroupLimits:=327699): 'WGPUSType_DawnExperimentalSubgroupLimits', (WGPUSType_AdapterPropertiesMemoryHeaps:=327700): 'WGPUSType_AdapterPropertiesMemoryHeaps', (WGPUSType_AdapterPropertiesD3D:=327701): 'WGPUSType_AdapterPropertiesD3D', (WGPUSType_AdapterPropertiesVk:=327702): 'WGPUSType_AdapterPropertiesVk', (WGPUSType_DawnWireWGSLControl:=327703): 'WGPUSType_DawnWireWGSLControl', (WGPUSType_DawnWGSLBlocklist:=327704): 'WGPUSType_DawnWGSLBlocklist', (WGPUSType_DrmFormatCapabilities:=327705): 'WGPUSType_DrmFormatCapabilities', (WGPUSType_ShaderModuleCompilationOptions:=327706): 'WGPUSType_ShaderModuleCompilationOptions', (WGPUSType_ColorTargetStateExpandResolveTextureDawn:=327707): 'WGPUSType_ColorTargetStateExpandResolveTextureDawn', (WGPUSType_RenderPassDescriptorExpandResolveRect:=327708): 'WGPUSType_RenderPassDescriptorExpandResolveRect', (WGPUSType_SharedTextureMemoryVkDedicatedAllocationDescriptor:=327709): 'WGPUSType_SharedTextureMemoryVkDedicatedAllocationDescriptor', (WGPUSType_SharedTextureMemoryAHardwareBufferDescriptor:=327710): 'WGPUSType_SharedTextureMemoryAHardwareBufferDescriptor', (WGPUSType_SharedTextureMemoryDmaBufDescriptor:=327711): 'WGPUSType_SharedTextureMemoryDmaBufDescriptor', (WGPUSType_SharedTextureMemoryOpaqueFDDescriptor:=327712): 'WGPUSType_SharedTextureMemoryOpaqueFDDescriptor', (WGPUSType_SharedTextureMemoryZirconHandleDescriptor:=327713): 'WGPUSType_SharedTextureMemoryZirconHandleDescriptor', (WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor:=327714): 'WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor', (WGPUSType_SharedTextureMemoryD3D11Texture2DDescriptor:=327715): 'WGPUSType_SharedTextureMemoryD3D11Texture2DDescriptor', (WGPUSType_SharedTextureMemoryIOSurfaceDescriptor:=327716): 'WGPUSType_SharedTextureMemoryIOSurfaceDescriptor', (WGPUSType_SharedTextureMemoryEGLImageDescriptor:=327717): 'WGPUSType_SharedTextureMemoryEGLImageDescriptor', (WGPUSType_SharedTextureMemoryInitializedBeginState:=327718): 'WGPUSType_SharedTextureMemoryInitializedBeginState', (WGPUSType_SharedTextureMemoryInitializedEndState:=327719): 'WGPUSType_SharedTextureMemoryInitializedEndState', (WGPUSType_SharedTextureMemoryVkImageLayoutBeginState:=327720): 'WGPUSType_SharedTextureMemoryVkImageLayoutBeginState', (WGPUSType_SharedTextureMemoryVkImageLayoutEndState:=327721): 'WGPUSType_SharedTextureMemoryVkImageLayoutEndState', (WGPUSType_SharedTextureMemoryD3DSwapchainBeginState:=327722): 'WGPUSType_SharedTextureMemoryD3DSwapchainBeginState', (WGPUSType_SharedFenceVkSemaphoreOpaqueFDDescriptor:=327723): 'WGPUSType_SharedFenceVkSemaphoreOpaqueFDDescriptor', (WGPUSType_SharedFenceVkSemaphoreOpaqueFDExportInfo:=327724): 'WGPUSType_SharedFenceVkSemaphoreOpaqueFDExportInfo', (WGPUSType_SharedFenceSyncFDDescriptor:=327725): 'WGPUSType_SharedFenceSyncFDDescriptor', (WGPUSType_SharedFenceSyncFDExportInfo:=327726): 'WGPUSType_SharedFenceSyncFDExportInfo', (WGPUSType_SharedFenceVkSemaphoreZirconHandleDescriptor:=327727): 'WGPUSType_SharedFenceVkSemaphoreZirconHandleDescriptor', (WGPUSType_SharedFenceVkSemaphoreZirconHandleExportInfo:=327728): 'WGPUSType_SharedFenceVkSemaphoreZirconHandleExportInfo', (WGPUSType_SharedFenceDXGISharedHandleDescriptor:=327729): 'WGPUSType_SharedFenceDXGISharedHandleDescriptor', (WGPUSType_SharedFenceDXGISharedHandleExportInfo:=327730): 'WGPUSType_SharedFenceDXGISharedHandleExportInfo', (WGPUSType_SharedFenceMTLSharedEventDescriptor:=327731): 'WGPUSType_SharedFenceMTLSharedEventDescriptor', (WGPUSType_SharedFenceMTLSharedEventExportInfo:=327732): 'WGPUSType_SharedFenceMTLSharedEventExportInfo', (WGPUSType_SharedBufferMemoryD3D12ResourceDescriptor:=327733): 'WGPUSType_SharedBufferMemoryD3D12ResourceDescriptor', (WGPUSType_StaticSamplerBindingLayout:=327734): 'WGPUSType_StaticSamplerBindingLayout', (WGPUSType_YCbCrVkDescriptor:=327735): 'WGPUSType_YCbCrVkDescriptor', (WGPUSType_SharedTextureMemoryAHardwareBufferProperties:=327736): 'WGPUSType_SharedTextureMemoryAHardwareBufferProperties', (WGPUSType_AHardwareBufferProperties:=327737): 'WGPUSType_AHardwareBufferProperties', (WGPUSType_DawnExperimentalImmediateDataLimits:=327738): 'WGPUSType_DawnExperimentalImmediateDataLimits', (WGPUSType_DawnTexelCopyBufferRowAlignmentLimits:=327739): 'WGPUSType_DawnTexelCopyBufferRowAlignmentLimits', (WGPUSType_Force32:=2147483647): 'WGPUSType_Force32'}
WGPUSType: TypeAlias = ctypes.c_uint32
struct_WGPUChainedStructOut.register_fields([('next', c.POINTER[struct_WGPUChainedStructOut], 0), ('sType', WGPUSType, 8)])
uint32_t: TypeAlias = ctypes.c_uint32
struct_WGPUAdapterPropertiesD3D.register_fields([('chain', WGPUChainedStructOut, 0), ('shaderModel', uint32_t, 16)])
@c.record
class struct_WGPUAdapterPropertiesSubgroups(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  subgroupMinSize: int
  subgroupMaxSize: int
struct_WGPUAdapterPropertiesSubgroups.register_fields([('chain', WGPUChainedStructOut, 0), ('subgroupMinSize', uint32_t, 16), ('subgroupMaxSize', uint32_t, 20)])
@c.record
class struct_WGPUAdapterPropertiesVk(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  driverVersion: int
struct_WGPUAdapterPropertiesVk.register_fields([('chain', WGPUChainedStructOut, 0), ('driverVersion', uint32_t, 16)])
@c.record
class struct_WGPUBindGroupEntry(c.Struct):
  SIZE = 56
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  binding: int
  buffer: c.POINTER[struct_WGPUBufferImpl]
  offset: int
  size: int
  sampler: c.POINTER[struct_WGPUSamplerImpl]
  textureView: c.POINTER[struct_WGPUTextureViewImpl]
@c.record
class struct_WGPUChainedStruct(c.Struct):
  SIZE = 16
  next: c.POINTER[struct_WGPUChainedStruct]
  sType: int
WGPUChainedStruct: TypeAlias = struct_WGPUChainedStruct
struct_WGPUChainedStruct.register_fields([('next', c.POINTER[struct_WGPUChainedStruct], 0), ('sType', WGPUSType, 8)])
uint64_t: TypeAlias = ctypes.c_uint64
struct_WGPUBindGroupEntry.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('binding', uint32_t, 8), ('buffer', WGPUBuffer, 16), ('offset', uint64_t, 24), ('size', uint64_t, 32), ('sampler', WGPUSampler, 40), ('textureView', WGPUTextureView, 48)])
@c.record
class struct_WGPUBlendComponent(c.Struct):
  SIZE = 12
  operation: int
  srcFactor: int
  dstFactor: int
enum_WGPUBlendOperation: dict[int, str] = {(WGPUBlendOperation_Undefined:=0): 'WGPUBlendOperation_Undefined', (WGPUBlendOperation_Add:=1): 'WGPUBlendOperation_Add', (WGPUBlendOperation_Subtract:=2): 'WGPUBlendOperation_Subtract', (WGPUBlendOperation_ReverseSubtract:=3): 'WGPUBlendOperation_ReverseSubtract', (WGPUBlendOperation_Min:=4): 'WGPUBlendOperation_Min', (WGPUBlendOperation_Max:=5): 'WGPUBlendOperation_Max', (WGPUBlendOperation_Force32:=2147483647): 'WGPUBlendOperation_Force32'}
WGPUBlendOperation: TypeAlias = ctypes.c_uint32
enum_WGPUBlendFactor: dict[int, str] = {(WGPUBlendFactor_Undefined:=0): 'WGPUBlendFactor_Undefined', (WGPUBlendFactor_Zero:=1): 'WGPUBlendFactor_Zero', (WGPUBlendFactor_One:=2): 'WGPUBlendFactor_One', (WGPUBlendFactor_Src:=3): 'WGPUBlendFactor_Src', (WGPUBlendFactor_OneMinusSrc:=4): 'WGPUBlendFactor_OneMinusSrc', (WGPUBlendFactor_SrcAlpha:=5): 'WGPUBlendFactor_SrcAlpha', (WGPUBlendFactor_OneMinusSrcAlpha:=6): 'WGPUBlendFactor_OneMinusSrcAlpha', (WGPUBlendFactor_Dst:=7): 'WGPUBlendFactor_Dst', (WGPUBlendFactor_OneMinusDst:=8): 'WGPUBlendFactor_OneMinusDst', (WGPUBlendFactor_DstAlpha:=9): 'WGPUBlendFactor_DstAlpha', (WGPUBlendFactor_OneMinusDstAlpha:=10): 'WGPUBlendFactor_OneMinusDstAlpha', (WGPUBlendFactor_SrcAlphaSaturated:=11): 'WGPUBlendFactor_SrcAlphaSaturated', (WGPUBlendFactor_Constant:=12): 'WGPUBlendFactor_Constant', (WGPUBlendFactor_OneMinusConstant:=13): 'WGPUBlendFactor_OneMinusConstant', (WGPUBlendFactor_Src1:=14): 'WGPUBlendFactor_Src1', (WGPUBlendFactor_OneMinusSrc1:=15): 'WGPUBlendFactor_OneMinusSrc1', (WGPUBlendFactor_Src1Alpha:=16): 'WGPUBlendFactor_Src1Alpha', (WGPUBlendFactor_OneMinusSrc1Alpha:=17): 'WGPUBlendFactor_OneMinusSrc1Alpha', (WGPUBlendFactor_Force32:=2147483647): 'WGPUBlendFactor_Force32'}
WGPUBlendFactor: TypeAlias = ctypes.c_uint32
struct_WGPUBlendComponent.register_fields([('operation', WGPUBlendOperation, 0), ('srcFactor', WGPUBlendFactor, 4), ('dstFactor', WGPUBlendFactor, 8)])
@c.record
class struct_WGPUBufferBindingLayout(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  type: int
  hasDynamicOffset: int
  minBindingSize: int
enum_WGPUBufferBindingType: dict[int, str] = {(WGPUBufferBindingType_BindingNotUsed:=0): 'WGPUBufferBindingType_BindingNotUsed', (WGPUBufferBindingType_Uniform:=1): 'WGPUBufferBindingType_Uniform', (WGPUBufferBindingType_Storage:=2): 'WGPUBufferBindingType_Storage', (WGPUBufferBindingType_ReadOnlyStorage:=3): 'WGPUBufferBindingType_ReadOnlyStorage', (WGPUBufferBindingType_Force32:=2147483647): 'WGPUBufferBindingType_Force32'}
WGPUBufferBindingType: TypeAlias = ctypes.c_uint32
struct_WGPUBufferBindingLayout.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('type', WGPUBufferBindingType, 8), ('hasDynamicOffset', WGPUBool, 12), ('minBindingSize', uint64_t, 16)])
@c.record
class struct_WGPUBufferHostMappedPointer(c.Struct):
  SIZE = 40
  chain: struct_WGPUChainedStruct
  pointer: ctypes.c_void_p
  disposeCallback: c.CFUNCTYPE[None, [ctypes.c_void_p]]
  userdata: ctypes.c_void_p
WGPUCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_void_p]]
struct_WGPUBufferHostMappedPointer.register_fields([('chain', WGPUChainedStruct, 0), ('pointer', ctypes.c_void_p, 16), ('disposeCallback', WGPUCallback, 24), ('userdata', ctypes.c_void_p, 32)])
@c.record
class struct_WGPUBufferMapCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
enum_WGPUCallbackMode: dict[int, str] = {(WGPUCallbackMode_WaitAnyOnly:=1): 'WGPUCallbackMode_WaitAnyOnly', (WGPUCallbackMode_AllowProcessEvents:=2): 'WGPUCallbackMode_AllowProcessEvents', (WGPUCallbackMode_AllowSpontaneous:=3): 'WGPUCallbackMode_AllowSpontaneous', (WGPUCallbackMode_Force32:=2147483647): 'WGPUCallbackMode_Force32'}
WGPUCallbackMode: TypeAlias = ctypes.c_uint32
enum_WGPUBufferMapAsyncStatus: dict[int, str] = {(WGPUBufferMapAsyncStatus_Success:=1): 'WGPUBufferMapAsyncStatus_Success', (WGPUBufferMapAsyncStatus_InstanceDropped:=2): 'WGPUBufferMapAsyncStatus_InstanceDropped', (WGPUBufferMapAsyncStatus_ValidationError:=3): 'WGPUBufferMapAsyncStatus_ValidationError', (WGPUBufferMapAsyncStatus_Unknown:=4): 'WGPUBufferMapAsyncStatus_Unknown', (WGPUBufferMapAsyncStatus_DeviceLost:=5): 'WGPUBufferMapAsyncStatus_DeviceLost', (WGPUBufferMapAsyncStatus_DestroyedBeforeCallback:=6): 'WGPUBufferMapAsyncStatus_DestroyedBeforeCallback', (WGPUBufferMapAsyncStatus_UnmappedBeforeCallback:=7): 'WGPUBufferMapAsyncStatus_UnmappedBeforeCallback', (WGPUBufferMapAsyncStatus_MappingAlreadyPending:=8): 'WGPUBufferMapAsyncStatus_MappingAlreadyPending', (WGPUBufferMapAsyncStatus_OffsetOutOfRange:=9): 'WGPUBufferMapAsyncStatus_OffsetOutOfRange', (WGPUBufferMapAsyncStatus_SizeOutOfRange:=10): 'WGPUBufferMapAsyncStatus_SizeOutOfRange', (WGPUBufferMapAsyncStatus_Force32:=2147483647): 'WGPUBufferMapAsyncStatus_Force32'}
WGPUBufferMapCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_void_p]]
struct_WGPUBufferMapCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUBufferMapCallback, 16), ('userdata', ctypes.c_void_p, 24)])
@c.record
class struct_WGPUColor(c.Struct):
  SIZE = 32
  r: float
  g: float
  b: float
  a: float
struct_WGPUColor.register_fields([('r', ctypes.c_double, 0), ('g', ctypes.c_double, 8), ('b', ctypes.c_double, 16), ('a', ctypes.c_double, 24)])
@c.record
class struct_WGPUColorTargetStateExpandResolveTextureDawn(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  enabled: int
struct_WGPUColorTargetStateExpandResolveTextureDawn.register_fields([('chain', WGPUChainedStruct, 0), ('enabled', WGPUBool, 16)])
@c.record
class struct_WGPUCompilationInfoCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUCompilationInfo], ctypes.c_void_p]]
  userdata: ctypes.c_void_p
enum_WGPUCompilationInfoRequestStatus: dict[int, str] = {(WGPUCompilationInfoRequestStatus_Success:=1): 'WGPUCompilationInfoRequestStatus_Success', (WGPUCompilationInfoRequestStatus_InstanceDropped:=2): 'WGPUCompilationInfoRequestStatus_InstanceDropped', (WGPUCompilationInfoRequestStatus_Error:=3): 'WGPUCompilationInfoRequestStatus_Error', (WGPUCompilationInfoRequestStatus_DeviceLost:=4): 'WGPUCompilationInfoRequestStatus_DeviceLost', (WGPUCompilationInfoRequestStatus_Unknown:=5): 'WGPUCompilationInfoRequestStatus_Unknown', (WGPUCompilationInfoRequestStatus_Force32:=2147483647): 'WGPUCompilationInfoRequestStatus_Force32'}
@c.record
class struct_WGPUCompilationInfo(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  messageCount: int
  messages: c.POINTER[struct_WGPUCompilationMessage]
size_t: TypeAlias = ctypes.c_uint64
@c.record
class struct_WGPUCompilationMessage(c.Struct):
  SIZE = 88
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  message: struct_WGPUStringView
  type: int
  lineNum: int
  linePos: int
  offset: int
  length: int
  utf16LinePos: int
  utf16Offset: int
  utf16Length: int
WGPUCompilationMessage: TypeAlias = struct_WGPUCompilationMessage
@c.record
class struct_WGPUStringView(c.Struct):
  SIZE = 16
  data: c.POINTER[ctypes.c_char]
  length: int
WGPUStringView: TypeAlias = struct_WGPUStringView
struct_WGPUStringView.register_fields([('data', c.POINTER[ctypes.c_char], 0), ('length', size_t, 8)])
enum_WGPUCompilationMessageType: dict[int, str] = {(WGPUCompilationMessageType_Error:=1): 'WGPUCompilationMessageType_Error', (WGPUCompilationMessageType_Warning:=2): 'WGPUCompilationMessageType_Warning', (WGPUCompilationMessageType_Info:=3): 'WGPUCompilationMessageType_Info', (WGPUCompilationMessageType_Force32:=2147483647): 'WGPUCompilationMessageType_Force32'}
WGPUCompilationMessageType: TypeAlias = ctypes.c_uint32
struct_WGPUCompilationMessage.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('message', WGPUStringView, 8), ('type', WGPUCompilationMessageType, 24), ('lineNum', uint64_t, 32), ('linePos', uint64_t, 40), ('offset', uint64_t, 48), ('length', uint64_t, 56), ('utf16LinePos', uint64_t, 64), ('utf16Offset', uint64_t, 72), ('utf16Length', uint64_t, 80)])
struct_WGPUCompilationInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('messageCount', size_t, 8), ('messages', c.POINTER[WGPUCompilationMessage], 16)])
WGPUCompilationInfoCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUCompilationInfo], ctypes.c_void_p]]
struct_WGPUCompilationInfoCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUCompilationInfoCallback, 16), ('userdata', ctypes.c_void_p, 24)])
@c.record
class struct_WGPUComputePassTimestampWrites(c.Struct):
  SIZE = 16
  querySet: c.POINTER[struct_WGPUQuerySetImpl]
  beginningOfPassWriteIndex: int
  endOfPassWriteIndex: int
struct_WGPUComputePassTimestampWrites.register_fields([('querySet', WGPUQuerySet, 0), ('beginningOfPassWriteIndex', uint32_t, 8), ('endOfPassWriteIndex', uint32_t, 12)])
@c.record
class struct_WGPUCopyTextureForBrowserOptions(c.Struct):
  SIZE = 56
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  flipY: int
  needsColorSpaceConversion: int
  srcAlphaMode: int
  srcTransferFunctionParameters: c.POINTER[ctypes.c_float]
  conversionMatrix: c.POINTER[ctypes.c_float]
  dstTransferFunctionParameters: c.POINTER[ctypes.c_float]
  dstAlphaMode: int
  internalUsage: int
enum_WGPUAlphaMode: dict[int, str] = {(WGPUAlphaMode_Opaque:=1): 'WGPUAlphaMode_Opaque', (WGPUAlphaMode_Premultiplied:=2): 'WGPUAlphaMode_Premultiplied', (WGPUAlphaMode_Unpremultiplied:=3): 'WGPUAlphaMode_Unpremultiplied', (WGPUAlphaMode_Force32:=2147483647): 'WGPUAlphaMode_Force32'}
WGPUAlphaMode: TypeAlias = ctypes.c_uint32
struct_WGPUCopyTextureForBrowserOptions.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('flipY', WGPUBool, 8), ('needsColorSpaceConversion', WGPUBool, 12), ('srcAlphaMode', WGPUAlphaMode, 16), ('srcTransferFunctionParameters', c.POINTER[ctypes.c_float], 24), ('conversionMatrix', c.POINTER[ctypes.c_float], 32), ('dstTransferFunctionParameters', c.POINTER[ctypes.c_float], 40), ('dstAlphaMode', WGPUAlphaMode, 48), ('internalUsage', WGPUBool, 52)])
@c.record
class struct_WGPUCreateComputePipelineAsyncCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
enum_WGPUCreatePipelineAsyncStatus: dict[int, str] = {(WGPUCreatePipelineAsyncStatus_Success:=1): 'WGPUCreatePipelineAsyncStatus_Success', (WGPUCreatePipelineAsyncStatus_InstanceDropped:=2): 'WGPUCreatePipelineAsyncStatus_InstanceDropped', (WGPUCreatePipelineAsyncStatus_ValidationError:=3): 'WGPUCreatePipelineAsyncStatus_ValidationError', (WGPUCreatePipelineAsyncStatus_InternalError:=4): 'WGPUCreatePipelineAsyncStatus_InternalError', (WGPUCreatePipelineAsyncStatus_DeviceLost:=5): 'WGPUCreatePipelineAsyncStatus_DeviceLost', (WGPUCreatePipelineAsyncStatus_DeviceDestroyed:=6): 'WGPUCreatePipelineAsyncStatus_DeviceDestroyed', (WGPUCreatePipelineAsyncStatus_Unknown:=7): 'WGPUCreatePipelineAsyncStatus_Unknown', (WGPUCreatePipelineAsyncStatus_Force32:=2147483647): 'WGPUCreatePipelineAsyncStatus_Force32'}
WGPUCreateComputePipelineAsyncCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView, ctypes.c_void_p]]
struct_WGPUCreateComputePipelineAsyncCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUCreateComputePipelineAsyncCallback, 16), ('userdata', ctypes.c_void_p, 24)])
@c.record
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
WGPUCreateRenderPipelineAsyncCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView, ctypes.c_void_p]]
struct_WGPUCreateRenderPipelineAsyncCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUCreateRenderPipelineAsyncCallback, 16), ('userdata', ctypes.c_void_p, 24)])
@c.record
class struct_WGPUDawnWGSLBlocklist(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  blocklistedFeatureCount: int
  blocklistedFeatures: c.POINTER[c.POINTER[ctypes.c_char]]
struct_WGPUDawnWGSLBlocklist.register_fields([('chain', WGPUChainedStruct, 0), ('blocklistedFeatureCount', size_t, 16), ('blocklistedFeatures', c.POINTER[c.POINTER[ctypes.c_char]], 24)])
@c.record
class struct_WGPUDawnAdapterPropertiesPowerPreference(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  powerPreference: int
enum_WGPUPowerPreference: dict[int, str] = {(WGPUPowerPreference_Undefined:=0): 'WGPUPowerPreference_Undefined', (WGPUPowerPreference_LowPower:=1): 'WGPUPowerPreference_LowPower', (WGPUPowerPreference_HighPerformance:=2): 'WGPUPowerPreference_HighPerformance', (WGPUPowerPreference_Force32:=2147483647): 'WGPUPowerPreference_Force32'}
WGPUPowerPreference: TypeAlias = ctypes.c_uint32
struct_WGPUDawnAdapterPropertiesPowerPreference.register_fields([('chain', WGPUChainedStructOut, 0), ('powerPreference', WGPUPowerPreference, 16)])
@c.record
class struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  outOfMemory: int
struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient.register_fields([('chain', WGPUChainedStruct, 0), ('outOfMemory', WGPUBool, 16)])
@c.record
class struct_WGPUDawnEncoderInternalUsageDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  useInternalUsages: int
struct_WGPUDawnEncoderInternalUsageDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('useInternalUsages', WGPUBool, 16)])
@c.record
class struct_WGPUDawnExperimentalImmediateDataLimits(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  maxImmediateDataRangeByteSize: int
struct_WGPUDawnExperimentalImmediateDataLimits.register_fields([('chain', WGPUChainedStructOut, 0), ('maxImmediateDataRangeByteSize', uint32_t, 16)])
@c.record
class struct_WGPUDawnExperimentalSubgroupLimits(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  minSubgroupSize: int
  maxSubgroupSize: int
struct_WGPUDawnExperimentalSubgroupLimits.register_fields([('chain', WGPUChainedStructOut, 0), ('minSubgroupSize', uint32_t, 16), ('maxSubgroupSize', uint32_t, 20)])
@c.record
class struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  implicitSampleCount: int
struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled.register_fields([('chain', WGPUChainedStruct, 0), ('implicitSampleCount', uint32_t, 16)])
@c.record
class struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  allowNonUniformDerivatives: int
struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('allowNonUniformDerivatives', WGPUBool, 16)])
@c.record
class struct_WGPUDawnTexelCopyBufferRowAlignmentLimits(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  minTexelCopyBufferRowAlignment: int
struct_WGPUDawnTexelCopyBufferRowAlignmentLimits.register_fields([('chain', WGPUChainedStructOut, 0), ('minTexelCopyBufferRowAlignment', uint32_t, 16)])
@c.record
class struct_WGPUDawnTextureInternalUsageDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  internalUsage: int
WGPUTextureUsage: TypeAlias = ctypes.c_uint64
struct_WGPUDawnTextureInternalUsageDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('internalUsage', WGPUTextureUsage, 16)])
@c.record
class struct_WGPUDawnTogglesDescriptor(c.Struct):
  SIZE = 48
  chain: struct_WGPUChainedStruct
  enabledToggleCount: int
  enabledToggles: c.POINTER[c.POINTER[ctypes.c_char]]
  disabledToggleCount: int
  disabledToggles: c.POINTER[c.POINTER[ctypes.c_char]]
struct_WGPUDawnTogglesDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('enabledToggleCount', size_t, 16), ('enabledToggles', c.POINTER[c.POINTER[ctypes.c_char]], 24), ('disabledToggleCount', size_t, 32), ('disabledToggles', c.POINTER[c.POINTER[ctypes.c_char]], 40)])
@c.record
class struct_WGPUDawnWireWGSLControl(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  enableExperimental: int
  enableUnsafe: int
  enableTesting: int
struct_WGPUDawnWireWGSLControl.register_fields([('chain', WGPUChainedStruct, 0), ('enableExperimental', WGPUBool, 16), ('enableUnsafe', WGPUBool, 20), ('enableTesting', WGPUBool, 24)])
@c.record
class struct_WGPUDeviceLostCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
enum_WGPUDeviceLostReason: dict[int, str] = {(WGPUDeviceLostReason_Unknown:=1): 'WGPUDeviceLostReason_Unknown', (WGPUDeviceLostReason_Destroyed:=2): 'WGPUDeviceLostReason_Destroyed', (WGPUDeviceLostReason_InstanceDropped:=3): 'WGPUDeviceLostReason_InstanceDropped', (WGPUDeviceLostReason_FailedCreation:=4): 'WGPUDeviceLostReason_FailedCreation', (WGPUDeviceLostReason_Force32:=2147483647): 'WGPUDeviceLostReason_Force32'}
WGPUDeviceLostCallbackNew: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
struct_WGPUDeviceLostCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUDeviceLostCallbackNew, 16), ('userdata', ctypes.c_void_p, 24)])
@c.record
class struct_WGPUDrmFormatProperties(c.Struct):
  SIZE = 16
  modifier: int
  modifierPlaneCount: int
struct_WGPUDrmFormatProperties.register_fields([('modifier', uint64_t, 0), ('modifierPlaneCount', uint32_t, 8)])
@c.record
class struct_WGPUExtent2D(c.Struct):
  SIZE = 8
  width: int
  height: int
struct_WGPUExtent2D.register_fields([('width', uint32_t, 0), ('height', uint32_t, 4)])
@c.record
class struct_WGPUExtent3D(c.Struct):
  SIZE = 12
  width: int
  height: int
  depthOrArrayLayers: int
struct_WGPUExtent3D.register_fields([('width', uint32_t, 0), ('height', uint32_t, 4), ('depthOrArrayLayers', uint32_t, 8)])
@c.record
class struct_WGPUExternalTextureBindingEntry(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  externalTexture: c.POINTER[struct_WGPUExternalTextureImpl]
struct_WGPUExternalTextureBindingEntry.register_fields([('chain', WGPUChainedStruct, 0), ('externalTexture', WGPUExternalTexture, 16)])
@c.record
class struct_WGPUExternalTextureBindingLayout(c.Struct):
  SIZE = 16
  chain: struct_WGPUChainedStruct
struct_WGPUExternalTextureBindingLayout.register_fields([('chain', WGPUChainedStruct, 0)])
@c.record
class struct_WGPUFormatCapabilities(c.Struct):
  SIZE = 8
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
struct_WGPUFormatCapabilities.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0)])
@c.record
class struct_WGPUFuture(c.Struct):
  SIZE = 8
  id: int
struct_WGPUFuture.register_fields([('id', uint64_t, 0)])
@c.record
class struct_WGPUInstanceFeatures(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  timedWaitAnyEnable: int
  timedWaitAnyMaxCount: int
struct_WGPUInstanceFeatures.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('timedWaitAnyEnable', WGPUBool, 8), ('timedWaitAnyMaxCount', size_t, 16)])
@c.record
class struct_WGPULimits(c.Struct):
  SIZE = 160
  maxTextureDimension1D: int
  maxTextureDimension2D: int
  maxTextureDimension3D: int
  maxTextureArrayLayers: int
  maxBindGroups: int
  maxBindGroupsPlusVertexBuffers: int
  maxBindingsPerBindGroup: int
  maxDynamicUniformBuffersPerPipelineLayout: int
  maxDynamicStorageBuffersPerPipelineLayout: int
  maxSampledTexturesPerShaderStage: int
  maxSamplersPerShaderStage: int
  maxStorageBuffersPerShaderStage: int
  maxStorageTexturesPerShaderStage: int
  maxUniformBuffersPerShaderStage: int
  maxUniformBufferBindingSize: int
  maxStorageBufferBindingSize: int
  minUniformBufferOffsetAlignment: int
  minStorageBufferOffsetAlignment: int
  maxVertexBuffers: int
  maxBufferSize: int
  maxVertexAttributes: int
  maxVertexBufferArrayStride: int
  maxInterStageShaderComponents: int
  maxInterStageShaderVariables: int
  maxColorAttachments: int
  maxColorAttachmentBytesPerSample: int
  maxComputeWorkgroupStorageSize: int
  maxComputeInvocationsPerWorkgroup: int
  maxComputeWorkgroupSizeX: int
  maxComputeWorkgroupSizeY: int
  maxComputeWorkgroupSizeZ: int
  maxComputeWorkgroupsPerDimension: int
  maxStorageBuffersInVertexStage: int
  maxStorageTexturesInVertexStage: int
  maxStorageBuffersInFragmentStage: int
  maxStorageTexturesInFragmentStage: int
struct_WGPULimits.register_fields([('maxTextureDimension1D', uint32_t, 0), ('maxTextureDimension2D', uint32_t, 4), ('maxTextureDimension3D', uint32_t, 8), ('maxTextureArrayLayers', uint32_t, 12), ('maxBindGroups', uint32_t, 16), ('maxBindGroupsPlusVertexBuffers', uint32_t, 20), ('maxBindingsPerBindGroup', uint32_t, 24), ('maxDynamicUniformBuffersPerPipelineLayout', uint32_t, 28), ('maxDynamicStorageBuffersPerPipelineLayout', uint32_t, 32), ('maxSampledTexturesPerShaderStage', uint32_t, 36), ('maxSamplersPerShaderStage', uint32_t, 40), ('maxStorageBuffersPerShaderStage', uint32_t, 44), ('maxStorageTexturesPerShaderStage', uint32_t, 48), ('maxUniformBuffersPerShaderStage', uint32_t, 52), ('maxUniformBufferBindingSize', uint64_t, 56), ('maxStorageBufferBindingSize', uint64_t, 64), ('minUniformBufferOffsetAlignment', uint32_t, 72), ('minStorageBufferOffsetAlignment', uint32_t, 76), ('maxVertexBuffers', uint32_t, 80), ('maxBufferSize', uint64_t, 88), ('maxVertexAttributes', uint32_t, 96), ('maxVertexBufferArrayStride', uint32_t, 100), ('maxInterStageShaderComponents', uint32_t, 104), ('maxInterStageShaderVariables', uint32_t, 108), ('maxColorAttachments', uint32_t, 112), ('maxColorAttachmentBytesPerSample', uint32_t, 116), ('maxComputeWorkgroupStorageSize', uint32_t, 120), ('maxComputeInvocationsPerWorkgroup', uint32_t, 124), ('maxComputeWorkgroupSizeX', uint32_t, 128), ('maxComputeWorkgroupSizeY', uint32_t, 132), ('maxComputeWorkgroupSizeZ', uint32_t, 136), ('maxComputeWorkgroupsPerDimension', uint32_t, 140), ('maxStorageBuffersInVertexStage', uint32_t, 144), ('maxStorageTexturesInVertexStage', uint32_t, 148), ('maxStorageBuffersInFragmentStage', uint32_t, 152), ('maxStorageTexturesInFragmentStage', uint32_t, 156)])
@c.record
class struct_WGPUMemoryHeapInfo(c.Struct):
  SIZE = 16
  properties: int
  size: int
WGPUHeapProperty: TypeAlias = ctypes.c_uint64
struct_WGPUMemoryHeapInfo.register_fields([('properties', WGPUHeapProperty, 0), ('size', uint64_t, 8)])
@c.record
class struct_WGPUMultisampleState(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  count: int
  mask: int
  alphaToCoverageEnabled: int
struct_WGPUMultisampleState.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('count', uint32_t, 8), ('mask', uint32_t, 12), ('alphaToCoverageEnabled', WGPUBool, 16)])
@c.record
class struct_WGPUOrigin2D(c.Struct):
  SIZE = 8
  x: int
  y: int
struct_WGPUOrigin2D.register_fields([('x', uint32_t, 0), ('y', uint32_t, 4)])
@c.record
class struct_WGPUOrigin3D(c.Struct):
  SIZE = 12
  x: int
  y: int
  z: int
struct_WGPUOrigin3D.register_fields([('x', uint32_t, 0), ('y', uint32_t, 4), ('z', uint32_t, 8)])
@c.record
class struct_WGPUPipelineLayoutStorageAttachment(c.Struct):
  SIZE = 16
  offset: int
  format: int
enum_WGPUTextureFormat: dict[int, str] = {(WGPUTextureFormat_Undefined:=0): 'WGPUTextureFormat_Undefined', (WGPUTextureFormat_R8Unorm:=1): 'WGPUTextureFormat_R8Unorm', (WGPUTextureFormat_R8Snorm:=2): 'WGPUTextureFormat_R8Snorm', (WGPUTextureFormat_R8Uint:=3): 'WGPUTextureFormat_R8Uint', (WGPUTextureFormat_R8Sint:=4): 'WGPUTextureFormat_R8Sint', (WGPUTextureFormat_R16Uint:=5): 'WGPUTextureFormat_R16Uint', (WGPUTextureFormat_R16Sint:=6): 'WGPUTextureFormat_R16Sint', (WGPUTextureFormat_R16Float:=7): 'WGPUTextureFormat_R16Float', (WGPUTextureFormat_RG8Unorm:=8): 'WGPUTextureFormat_RG8Unorm', (WGPUTextureFormat_RG8Snorm:=9): 'WGPUTextureFormat_RG8Snorm', (WGPUTextureFormat_RG8Uint:=10): 'WGPUTextureFormat_RG8Uint', (WGPUTextureFormat_RG8Sint:=11): 'WGPUTextureFormat_RG8Sint', (WGPUTextureFormat_R32Float:=12): 'WGPUTextureFormat_R32Float', (WGPUTextureFormat_R32Uint:=13): 'WGPUTextureFormat_R32Uint', (WGPUTextureFormat_R32Sint:=14): 'WGPUTextureFormat_R32Sint', (WGPUTextureFormat_RG16Uint:=15): 'WGPUTextureFormat_RG16Uint', (WGPUTextureFormat_RG16Sint:=16): 'WGPUTextureFormat_RG16Sint', (WGPUTextureFormat_RG16Float:=17): 'WGPUTextureFormat_RG16Float', (WGPUTextureFormat_RGBA8Unorm:=18): 'WGPUTextureFormat_RGBA8Unorm', (WGPUTextureFormat_RGBA8UnormSrgb:=19): 'WGPUTextureFormat_RGBA8UnormSrgb', (WGPUTextureFormat_RGBA8Snorm:=20): 'WGPUTextureFormat_RGBA8Snorm', (WGPUTextureFormat_RGBA8Uint:=21): 'WGPUTextureFormat_RGBA8Uint', (WGPUTextureFormat_RGBA8Sint:=22): 'WGPUTextureFormat_RGBA8Sint', (WGPUTextureFormat_BGRA8Unorm:=23): 'WGPUTextureFormat_BGRA8Unorm', (WGPUTextureFormat_BGRA8UnormSrgb:=24): 'WGPUTextureFormat_BGRA8UnormSrgb', (WGPUTextureFormat_RGB10A2Uint:=25): 'WGPUTextureFormat_RGB10A2Uint', (WGPUTextureFormat_RGB10A2Unorm:=26): 'WGPUTextureFormat_RGB10A2Unorm', (WGPUTextureFormat_RG11B10Ufloat:=27): 'WGPUTextureFormat_RG11B10Ufloat', (WGPUTextureFormat_RGB9E5Ufloat:=28): 'WGPUTextureFormat_RGB9E5Ufloat', (WGPUTextureFormat_RG32Float:=29): 'WGPUTextureFormat_RG32Float', (WGPUTextureFormat_RG32Uint:=30): 'WGPUTextureFormat_RG32Uint', (WGPUTextureFormat_RG32Sint:=31): 'WGPUTextureFormat_RG32Sint', (WGPUTextureFormat_RGBA16Uint:=32): 'WGPUTextureFormat_RGBA16Uint', (WGPUTextureFormat_RGBA16Sint:=33): 'WGPUTextureFormat_RGBA16Sint', (WGPUTextureFormat_RGBA16Float:=34): 'WGPUTextureFormat_RGBA16Float', (WGPUTextureFormat_RGBA32Float:=35): 'WGPUTextureFormat_RGBA32Float', (WGPUTextureFormat_RGBA32Uint:=36): 'WGPUTextureFormat_RGBA32Uint', (WGPUTextureFormat_RGBA32Sint:=37): 'WGPUTextureFormat_RGBA32Sint', (WGPUTextureFormat_Stencil8:=38): 'WGPUTextureFormat_Stencil8', (WGPUTextureFormat_Depth16Unorm:=39): 'WGPUTextureFormat_Depth16Unorm', (WGPUTextureFormat_Depth24Plus:=40): 'WGPUTextureFormat_Depth24Plus', (WGPUTextureFormat_Depth24PlusStencil8:=41): 'WGPUTextureFormat_Depth24PlusStencil8', (WGPUTextureFormat_Depth32Float:=42): 'WGPUTextureFormat_Depth32Float', (WGPUTextureFormat_Depth32FloatStencil8:=43): 'WGPUTextureFormat_Depth32FloatStencil8', (WGPUTextureFormat_BC1RGBAUnorm:=44): 'WGPUTextureFormat_BC1RGBAUnorm', (WGPUTextureFormat_BC1RGBAUnormSrgb:=45): 'WGPUTextureFormat_BC1RGBAUnormSrgb', (WGPUTextureFormat_BC2RGBAUnorm:=46): 'WGPUTextureFormat_BC2RGBAUnorm', (WGPUTextureFormat_BC2RGBAUnormSrgb:=47): 'WGPUTextureFormat_BC2RGBAUnormSrgb', (WGPUTextureFormat_BC3RGBAUnorm:=48): 'WGPUTextureFormat_BC3RGBAUnorm', (WGPUTextureFormat_BC3RGBAUnormSrgb:=49): 'WGPUTextureFormat_BC3RGBAUnormSrgb', (WGPUTextureFormat_BC4RUnorm:=50): 'WGPUTextureFormat_BC4RUnorm', (WGPUTextureFormat_BC4RSnorm:=51): 'WGPUTextureFormat_BC4RSnorm', (WGPUTextureFormat_BC5RGUnorm:=52): 'WGPUTextureFormat_BC5RGUnorm', (WGPUTextureFormat_BC5RGSnorm:=53): 'WGPUTextureFormat_BC5RGSnorm', (WGPUTextureFormat_BC6HRGBUfloat:=54): 'WGPUTextureFormat_BC6HRGBUfloat', (WGPUTextureFormat_BC6HRGBFloat:=55): 'WGPUTextureFormat_BC6HRGBFloat', (WGPUTextureFormat_BC7RGBAUnorm:=56): 'WGPUTextureFormat_BC7RGBAUnorm', (WGPUTextureFormat_BC7RGBAUnormSrgb:=57): 'WGPUTextureFormat_BC7RGBAUnormSrgb', (WGPUTextureFormat_ETC2RGB8Unorm:=58): 'WGPUTextureFormat_ETC2RGB8Unorm', (WGPUTextureFormat_ETC2RGB8UnormSrgb:=59): 'WGPUTextureFormat_ETC2RGB8UnormSrgb', (WGPUTextureFormat_ETC2RGB8A1Unorm:=60): 'WGPUTextureFormat_ETC2RGB8A1Unorm', (WGPUTextureFormat_ETC2RGB8A1UnormSrgb:=61): 'WGPUTextureFormat_ETC2RGB8A1UnormSrgb', (WGPUTextureFormat_ETC2RGBA8Unorm:=62): 'WGPUTextureFormat_ETC2RGBA8Unorm', (WGPUTextureFormat_ETC2RGBA8UnormSrgb:=63): 'WGPUTextureFormat_ETC2RGBA8UnormSrgb', (WGPUTextureFormat_EACR11Unorm:=64): 'WGPUTextureFormat_EACR11Unorm', (WGPUTextureFormat_EACR11Snorm:=65): 'WGPUTextureFormat_EACR11Snorm', (WGPUTextureFormat_EACRG11Unorm:=66): 'WGPUTextureFormat_EACRG11Unorm', (WGPUTextureFormat_EACRG11Snorm:=67): 'WGPUTextureFormat_EACRG11Snorm', (WGPUTextureFormat_ASTC4x4Unorm:=68): 'WGPUTextureFormat_ASTC4x4Unorm', (WGPUTextureFormat_ASTC4x4UnormSrgb:=69): 'WGPUTextureFormat_ASTC4x4UnormSrgb', (WGPUTextureFormat_ASTC5x4Unorm:=70): 'WGPUTextureFormat_ASTC5x4Unorm', (WGPUTextureFormat_ASTC5x4UnormSrgb:=71): 'WGPUTextureFormat_ASTC5x4UnormSrgb', (WGPUTextureFormat_ASTC5x5Unorm:=72): 'WGPUTextureFormat_ASTC5x5Unorm', (WGPUTextureFormat_ASTC5x5UnormSrgb:=73): 'WGPUTextureFormat_ASTC5x5UnormSrgb', (WGPUTextureFormat_ASTC6x5Unorm:=74): 'WGPUTextureFormat_ASTC6x5Unorm', (WGPUTextureFormat_ASTC6x5UnormSrgb:=75): 'WGPUTextureFormat_ASTC6x5UnormSrgb', (WGPUTextureFormat_ASTC6x6Unorm:=76): 'WGPUTextureFormat_ASTC6x6Unorm', (WGPUTextureFormat_ASTC6x6UnormSrgb:=77): 'WGPUTextureFormat_ASTC6x6UnormSrgb', (WGPUTextureFormat_ASTC8x5Unorm:=78): 'WGPUTextureFormat_ASTC8x5Unorm', (WGPUTextureFormat_ASTC8x5UnormSrgb:=79): 'WGPUTextureFormat_ASTC8x5UnormSrgb', (WGPUTextureFormat_ASTC8x6Unorm:=80): 'WGPUTextureFormat_ASTC8x6Unorm', (WGPUTextureFormat_ASTC8x6UnormSrgb:=81): 'WGPUTextureFormat_ASTC8x6UnormSrgb', (WGPUTextureFormat_ASTC8x8Unorm:=82): 'WGPUTextureFormat_ASTC8x8Unorm', (WGPUTextureFormat_ASTC8x8UnormSrgb:=83): 'WGPUTextureFormat_ASTC8x8UnormSrgb', (WGPUTextureFormat_ASTC10x5Unorm:=84): 'WGPUTextureFormat_ASTC10x5Unorm', (WGPUTextureFormat_ASTC10x5UnormSrgb:=85): 'WGPUTextureFormat_ASTC10x5UnormSrgb', (WGPUTextureFormat_ASTC10x6Unorm:=86): 'WGPUTextureFormat_ASTC10x6Unorm', (WGPUTextureFormat_ASTC10x6UnormSrgb:=87): 'WGPUTextureFormat_ASTC10x6UnormSrgb', (WGPUTextureFormat_ASTC10x8Unorm:=88): 'WGPUTextureFormat_ASTC10x8Unorm', (WGPUTextureFormat_ASTC10x8UnormSrgb:=89): 'WGPUTextureFormat_ASTC10x8UnormSrgb', (WGPUTextureFormat_ASTC10x10Unorm:=90): 'WGPUTextureFormat_ASTC10x10Unorm', (WGPUTextureFormat_ASTC10x10UnormSrgb:=91): 'WGPUTextureFormat_ASTC10x10UnormSrgb', (WGPUTextureFormat_ASTC12x10Unorm:=92): 'WGPUTextureFormat_ASTC12x10Unorm', (WGPUTextureFormat_ASTC12x10UnormSrgb:=93): 'WGPUTextureFormat_ASTC12x10UnormSrgb', (WGPUTextureFormat_ASTC12x12Unorm:=94): 'WGPUTextureFormat_ASTC12x12Unorm', (WGPUTextureFormat_ASTC12x12UnormSrgb:=95): 'WGPUTextureFormat_ASTC12x12UnormSrgb', (WGPUTextureFormat_R16Unorm:=327680): 'WGPUTextureFormat_R16Unorm', (WGPUTextureFormat_RG16Unorm:=327681): 'WGPUTextureFormat_RG16Unorm', (WGPUTextureFormat_RGBA16Unorm:=327682): 'WGPUTextureFormat_RGBA16Unorm', (WGPUTextureFormat_R16Snorm:=327683): 'WGPUTextureFormat_R16Snorm', (WGPUTextureFormat_RG16Snorm:=327684): 'WGPUTextureFormat_RG16Snorm', (WGPUTextureFormat_RGBA16Snorm:=327685): 'WGPUTextureFormat_RGBA16Snorm', (WGPUTextureFormat_R8BG8Biplanar420Unorm:=327686): 'WGPUTextureFormat_R8BG8Biplanar420Unorm', (WGPUTextureFormat_R10X6BG10X6Biplanar420Unorm:=327687): 'WGPUTextureFormat_R10X6BG10X6Biplanar420Unorm', (WGPUTextureFormat_R8BG8A8Triplanar420Unorm:=327688): 'WGPUTextureFormat_R8BG8A8Triplanar420Unorm', (WGPUTextureFormat_R8BG8Biplanar422Unorm:=327689): 'WGPUTextureFormat_R8BG8Biplanar422Unorm', (WGPUTextureFormat_R8BG8Biplanar444Unorm:=327690): 'WGPUTextureFormat_R8BG8Biplanar444Unorm', (WGPUTextureFormat_R10X6BG10X6Biplanar422Unorm:=327691): 'WGPUTextureFormat_R10X6BG10X6Biplanar422Unorm', (WGPUTextureFormat_R10X6BG10X6Biplanar444Unorm:=327692): 'WGPUTextureFormat_R10X6BG10X6Biplanar444Unorm', (WGPUTextureFormat_External:=327693): 'WGPUTextureFormat_External', (WGPUTextureFormat_Force32:=2147483647): 'WGPUTextureFormat_Force32'}
WGPUTextureFormat: TypeAlias = ctypes.c_uint32
struct_WGPUPipelineLayoutStorageAttachment.register_fields([('offset', uint64_t, 0), ('format', WGPUTextureFormat, 8)])
@c.record
class struct_WGPUPopErrorScopeCallbackInfo(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
  oldCallback: c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
enum_WGPUPopErrorScopeStatus: dict[int, str] = {(WGPUPopErrorScopeStatus_Success:=1): 'WGPUPopErrorScopeStatus_Success', (WGPUPopErrorScopeStatus_InstanceDropped:=2): 'WGPUPopErrorScopeStatus_InstanceDropped', (WGPUPopErrorScopeStatus_Force32:=2147483647): 'WGPUPopErrorScopeStatus_Force32'}
enum_WGPUErrorType: dict[int, str] = {(WGPUErrorType_NoError:=1): 'WGPUErrorType_NoError', (WGPUErrorType_Validation:=2): 'WGPUErrorType_Validation', (WGPUErrorType_OutOfMemory:=3): 'WGPUErrorType_OutOfMemory', (WGPUErrorType_Internal:=4): 'WGPUErrorType_Internal', (WGPUErrorType_Unknown:=5): 'WGPUErrorType_Unknown', (WGPUErrorType_DeviceLost:=6): 'WGPUErrorType_DeviceLost', (WGPUErrorType_Force32:=2147483647): 'WGPUErrorType_Force32'}
WGPUPopErrorScopeCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
WGPUErrorCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
struct_WGPUPopErrorScopeCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUPopErrorScopeCallback, 16), ('oldCallback', WGPUErrorCallback, 24), ('userdata', ctypes.c_void_p, 32)])
@c.record
class struct_WGPUPrimitiveState(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  topology: int
  stripIndexFormat: int
  frontFace: int
  cullMode: int
  unclippedDepth: int
enum_WGPUPrimitiveTopology: dict[int, str] = {(WGPUPrimitiveTopology_Undefined:=0): 'WGPUPrimitiveTopology_Undefined', (WGPUPrimitiveTopology_PointList:=1): 'WGPUPrimitiveTopology_PointList', (WGPUPrimitiveTopology_LineList:=2): 'WGPUPrimitiveTopology_LineList', (WGPUPrimitiveTopology_LineStrip:=3): 'WGPUPrimitiveTopology_LineStrip', (WGPUPrimitiveTopology_TriangleList:=4): 'WGPUPrimitiveTopology_TriangleList', (WGPUPrimitiveTopology_TriangleStrip:=5): 'WGPUPrimitiveTopology_TriangleStrip', (WGPUPrimitiveTopology_Force32:=2147483647): 'WGPUPrimitiveTopology_Force32'}
WGPUPrimitiveTopology: TypeAlias = ctypes.c_uint32
enum_WGPUIndexFormat: dict[int, str] = {(WGPUIndexFormat_Undefined:=0): 'WGPUIndexFormat_Undefined', (WGPUIndexFormat_Uint16:=1): 'WGPUIndexFormat_Uint16', (WGPUIndexFormat_Uint32:=2): 'WGPUIndexFormat_Uint32', (WGPUIndexFormat_Force32:=2147483647): 'WGPUIndexFormat_Force32'}
WGPUIndexFormat: TypeAlias = ctypes.c_uint32
enum_WGPUFrontFace: dict[int, str] = {(WGPUFrontFace_Undefined:=0): 'WGPUFrontFace_Undefined', (WGPUFrontFace_CCW:=1): 'WGPUFrontFace_CCW', (WGPUFrontFace_CW:=2): 'WGPUFrontFace_CW', (WGPUFrontFace_Force32:=2147483647): 'WGPUFrontFace_Force32'}
WGPUFrontFace: TypeAlias = ctypes.c_uint32
enum_WGPUCullMode: dict[int, str] = {(WGPUCullMode_Undefined:=0): 'WGPUCullMode_Undefined', (WGPUCullMode_None:=1): 'WGPUCullMode_None', (WGPUCullMode_Front:=2): 'WGPUCullMode_Front', (WGPUCullMode_Back:=3): 'WGPUCullMode_Back', (WGPUCullMode_Force32:=2147483647): 'WGPUCullMode_Force32'}
WGPUCullMode: TypeAlias = ctypes.c_uint32
struct_WGPUPrimitiveState.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('topology', WGPUPrimitiveTopology, 8), ('stripIndexFormat', WGPUIndexFormat, 12), ('frontFace', WGPUFrontFace, 16), ('cullMode', WGPUCullMode, 20), ('unclippedDepth', WGPUBool, 24)])
@c.record
class struct_WGPUQueueWorkDoneCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
enum_WGPUQueueWorkDoneStatus: dict[int, str] = {(WGPUQueueWorkDoneStatus_Success:=1): 'WGPUQueueWorkDoneStatus_Success', (WGPUQueueWorkDoneStatus_InstanceDropped:=2): 'WGPUQueueWorkDoneStatus_InstanceDropped', (WGPUQueueWorkDoneStatus_Error:=3): 'WGPUQueueWorkDoneStatus_Error', (WGPUQueueWorkDoneStatus_Unknown:=4): 'WGPUQueueWorkDoneStatus_Unknown', (WGPUQueueWorkDoneStatus_DeviceLost:=5): 'WGPUQueueWorkDoneStatus_DeviceLost', (WGPUQueueWorkDoneStatus_Force32:=2147483647): 'WGPUQueueWorkDoneStatus_Force32'}
WGPUQueueWorkDoneCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_void_p]]
struct_WGPUQueueWorkDoneCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUQueueWorkDoneCallback, 16), ('userdata', ctypes.c_void_p, 24)])
@c.record
class struct_WGPURenderPassDepthStencilAttachment(c.Struct):
  SIZE = 40
  view: c.POINTER[struct_WGPUTextureViewImpl]
  depthLoadOp: int
  depthStoreOp: int
  depthClearValue: float
  depthReadOnly: int
  stencilLoadOp: int
  stencilStoreOp: int
  stencilClearValue: int
  stencilReadOnly: int
enum_WGPULoadOp: dict[int, str] = {(WGPULoadOp_Undefined:=0): 'WGPULoadOp_Undefined', (WGPULoadOp_Load:=1): 'WGPULoadOp_Load', (WGPULoadOp_Clear:=2): 'WGPULoadOp_Clear', (WGPULoadOp_ExpandResolveTexture:=327683): 'WGPULoadOp_ExpandResolveTexture', (WGPULoadOp_Force32:=2147483647): 'WGPULoadOp_Force32'}
WGPULoadOp: TypeAlias = ctypes.c_uint32
enum_WGPUStoreOp: dict[int, str] = {(WGPUStoreOp_Undefined:=0): 'WGPUStoreOp_Undefined', (WGPUStoreOp_Store:=1): 'WGPUStoreOp_Store', (WGPUStoreOp_Discard:=2): 'WGPUStoreOp_Discard', (WGPUStoreOp_Force32:=2147483647): 'WGPUStoreOp_Force32'}
WGPUStoreOp: TypeAlias = ctypes.c_uint32
struct_WGPURenderPassDepthStencilAttachment.register_fields([('view', WGPUTextureView, 0), ('depthLoadOp', WGPULoadOp, 8), ('depthStoreOp', WGPUStoreOp, 12), ('depthClearValue', ctypes.c_float, 16), ('depthReadOnly', WGPUBool, 20), ('stencilLoadOp', WGPULoadOp, 24), ('stencilStoreOp', WGPUStoreOp, 28), ('stencilClearValue', uint32_t, 32), ('stencilReadOnly', WGPUBool, 36)])
@c.record
class struct_WGPURenderPassDescriptorExpandResolveRect(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  x: int
  y: int
  width: int
  height: int
struct_WGPURenderPassDescriptorExpandResolveRect.register_fields([('chain', WGPUChainedStruct, 0), ('x', uint32_t, 16), ('y', uint32_t, 20), ('width', uint32_t, 24), ('height', uint32_t, 28)])
@c.record
class struct_WGPURenderPassMaxDrawCount(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  maxDrawCount: int
struct_WGPURenderPassMaxDrawCount.register_fields([('chain', WGPUChainedStruct, 0), ('maxDrawCount', uint64_t, 16)])
@c.record
class struct_WGPURenderPassTimestampWrites(c.Struct):
  SIZE = 16
  querySet: c.POINTER[struct_WGPUQuerySetImpl]
  beginningOfPassWriteIndex: int
  endOfPassWriteIndex: int
struct_WGPURenderPassTimestampWrites.register_fields([('querySet', WGPUQuerySet, 0), ('beginningOfPassWriteIndex', uint32_t, 8), ('endOfPassWriteIndex', uint32_t, 12)])
@c.record
class struct_WGPURequestAdapterCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUAdapterImpl], struct_WGPUStringView, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
enum_WGPURequestAdapterStatus: dict[int, str] = {(WGPURequestAdapterStatus_Success:=1): 'WGPURequestAdapterStatus_Success', (WGPURequestAdapterStatus_InstanceDropped:=2): 'WGPURequestAdapterStatus_InstanceDropped', (WGPURequestAdapterStatus_Unavailable:=3): 'WGPURequestAdapterStatus_Unavailable', (WGPURequestAdapterStatus_Error:=4): 'WGPURequestAdapterStatus_Error', (WGPURequestAdapterStatus_Unknown:=5): 'WGPURequestAdapterStatus_Unknown', (WGPURequestAdapterStatus_Force32:=2147483647): 'WGPURequestAdapterStatus_Force32'}
WGPURequestAdapterCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUAdapterImpl], struct_WGPUStringView, ctypes.c_void_p]]
struct_WGPURequestAdapterCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPURequestAdapterCallback, 16), ('userdata', ctypes.c_void_p, 24)])
@c.record
class struct_WGPURequestAdapterOptions(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  compatibleSurface: c.POINTER[struct_WGPUSurfaceImpl]
  featureLevel: int
  powerPreference: int
  backendType: int
  forceFallbackAdapter: int
  compatibilityMode: int
enum_WGPUFeatureLevel: dict[int, str] = {(WGPUFeatureLevel_Undefined:=0): 'WGPUFeatureLevel_Undefined', (WGPUFeatureLevel_Compatibility:=1): 'WGPUFeatureLevel_Compatibility', (WGPUFeatureLevel_Core:=2): 'WGPUFeatureLevel_Core', (WGPUFeatureLevel_Force32:=2147483647): 'WGPUFeatureLevel_Force32'}
WGPUFeatureLevel: TypeAlias = ctypes.c_uint32
enum_WGPUBackendType: dict[int, str] = {(WGPUBackendType_Undefined:=0): 'WGPUBackendType_Undefined', (WGPUBackendType_Null:=1): 'WGPUBackendType_Null', (WGPUBackendType_WebGPU:=2): 'WGPUBackendType_WebGPU', (WGPUBackendType_D3D11:=3): 'WGPUBackendType_D3D11', (WGPUBackendType_D3D12:=4): 'WGPUBackendType_D3D12', (WGPUBackendType_Metal:=5): 'WGPUBackendType_Metal', (WGPUBackendType_Vulkan:=6): 'WGPUBackendType_Vulkan', (WGPUBackendType_OpenGL:=7): 'WGPUBackendType_OpenGL', (WGPUBackendType_OpenGLES:=8): 'WGPUBackendType_OpenGLES', (WGPUBackendType_Force32:=2147483647): 'WGPUBackendType_Force32'}
WGPUBackendType: TypeAlias = ctypes.c_uint32
struct_WGPURequestAdapterOptions.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('compatibleSurface', WGPUSurface, 8), ('featureLevel', WGPUFeatureLevel, 16), ('powerPreference', WGPUPowerPreference, 20), ('backendType', WGPUBackendType, 24), ('forceFallbackAdapter', WGPUBool, 28), ('compatibilityMode', WGPUBool, 32)])
@c.record
class struct_WGPURequestDeviceCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
enum_WGPURequestDeviceStatus: dict[int, str] = {(WGPURequestDeviceStatus_Success:=1): 'WGPURequestDeviceStatus_Success', (WGPURequestDeviceStatus_InstanceDropped:=2): 'WGPURequestDeviceStatus_InstanceDropped', (WGPURequestDeviceStatus_Error:=3): 'WGPURequestDeviceStatus_Error', (WGPURequestDeviceStatus_Unknown:=4): 'WGPURequestDeviceStatus_Unknown', (WGPURequestDeviceStatus_Force32:=2147483647): 'WGPURequestDeviceStatus_Force32'}
WGPURequestDeviceCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView, ctypes.c_void_p]]
struct_WGPURequestDeviceCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPURequestDeviceCallback, 16), ('userdata', ctypes.c_void_p, 24)])
@c.record
class struct_WGPUSamplerBindingLayout(c.Struct):
  SIZE = 16
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  type: int
enum_WGPUSamplerBindingType: dict[int, str] = {(WGPUSamplerBindingType_BindingNotUsed:=0): 'WGPUSamplerBindingType_BindingNotUsed', (WGPUSamplerBindingType_Filtering:=1): 'WGPUSamplerBindingType_Filtering', (WGPUSamplerBindingType_NonFiltering:=2): 'WGPUSamplerBindingType_NonFiltering', (WGPUSamplerBindingType_Comparison:=3): 'WGPUSamplerBindingType_Comparison', (WGPUSamplerBindingType_Force32:=2147483647): 'WGPUSamplerBindingType_Force32'}
WGPUSamplerBindingType: TypeAlias = ctypes.c_uint32
struct_WGPUSamplerBindingLayout.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('type', WGPUSamplerBindingType, 8)])
@c.record
class struct_WGPUShaderModuleCompilationOptions(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  strictMath: int
struct_WGPUShaderModuleCompilationOptions.register_fields([('chain', WGPUChainedStruct, 0), ('strictMath', WGPUBool, 16)])
@c.record
class struct_WGPUShaderSourceSPIRV(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  codeSize: int
  code: c.POINTER[ctypes.c_uint32]
struct_WGPUShaderSourceSPIRV.register_fields([('chain', WGPUChainedStruct, 0), ('codeSize', uint32_t, 16), ('code', c.POINTER[uint32_t], 24)])
@c.record
class struct_WGPUSharedBufferMemoryBeginAccessDescriptor(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  initialized: int
  fenceCount: int
  fences: c.POINTER[c.POINTER[struct_WGPUSharedFenceImpl]]
  signaledValues: c.POINTER[ctypes.c_uint64]
struct_WGPUSharedBufferMemoryBeginAccessDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('initialized', WGPUBool, 8), ('fenceCount', size_t, 16), ('fences', c.POINTER[WGPUSharedFence], 24), ('signaledValues', c.POINTER[uint64_t], 32)])
@c.record
class struct_WGPUSharedBufferMemoryEndAccessState(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
  initialized: int
  fenceCount: int
  fences: c.POINTER[c.POINTER[struct_WGPUSharedFenceImpl]]
  signaledValues: c.POINTER[ctypes.c_uint64]
struct_WGPUSharedBufferMemoryEndAccessState.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0), ('initialized', WGPUBool, 8), ('fenceCount', size_t, 16), ('fences', c.POINTER[WGPUSharedFence], 24), ('signaledValues', c.POINTER[uint64_t], 32)])
@c.record
class struct_WGPUSharedBufferMemoryProperties(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
  usage: int
  size: int
WGPUBufferUsage: TypeAlias = ctypes.c_uint64
struct_WGPUSharedBufferMemoryProperties.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0), ('usage', WGPUBufferUsage, 8), ('size', uint64_t, 16)])
@c.record
class struct_WGPUSharedFenceDXGISharedHandleDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  handle: ctypes.c_void_p
struct_WGPUSharedFenceDXGISharedHandleDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('handle', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSharedFenceDXGISharedHandleExportInfo(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  handle: ctypes.c_void_p
struct_WGPUSharedFenceDXGISharedHandleExportInfo.register_fields([('chain', WGPUChainedStructOut, 0), ('handle', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSharedFenceMTLSharedEventDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  sharedEvent: ctypes.c_void_p
struct_WGPUSharedFenceMTLSharedEventDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('sharedEvent', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSharedFenceMTLSharedEventExportInfo(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  sharedEvent: ctypes.c_void_p
struct_WGPUSharedFenceMTLSharedEventExportInfo.register_fields([('chain', WGPUChainedStructOut, 0), ('sharedEvent', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSharedFenceExportInfo(c.Struct):
  SIZE = 16
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
  type: int
enum_WGPUSharedFenceType: dict[int, str] = {(WGPUSharedFenceType_VkSemaphoreOpaqueFD:=1): 'WGPUSharedFenceType_VkSemaphoreOpaqueFD', (WGPUSharedFenceType_SyncFD:=2): 'WGPUSharedFenceType_SyncFD', (WGPUSharedFenceType_VkSemaphoreZirconHandle:=3): 'WGPUSharedFenceType_VkSemaphoreZirconHandle', (WGPUSharedFenceType_DXGISharedHandle:=4): 'WGPUSharedFenceType_DXGISharedHandle', (WGPUSharedFenceType_MTLSharedEvent:=5): 'WGPUSharedFenceType_MTLSharedEvent', (WGPUSharedFenceType_Force32:=2147483647): 'WGPUSharedFenceType_Force32'}
WGPUSharedFenceType: TypeAlias = ctypes.c_uint32
struct_WGPUSharedFenceExportInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0), ('type', WGPUSharedFenceType, 8)])
@c.record
class struct_WGPUSharedFenceSyncFDDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  handle: int
struct_WGPUSharedFenceSyncFDDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('handle', ctypes.c_int32, 16)])
@c.record
class struct_WGPUSharedFenceSyncFDExportInfo(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  handle: int
struct_WGPUSharedFenceSyncFDExportInfo.register_fields([('chain', WGPUChainedStructOut, 0), ('handle', ctypes.c_int32, 16)])
@c.record
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  handle: int
struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('handle', ctypes.c_int32, 16)])
@c.record
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  handle: int
struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo.register_fields([('chain', WGPUChainedStructOut, 0), ('handle', ctypes.c_int32, 16)])
@c.record
class struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  handle: int
struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('handle', uint32_t, 16)])
@c.record
class struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  handle: int
struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo.register_fields([('chain', WGPUChainedStructOut, 0), ('handle', uint32_t, 16)])
@c.record
class struct_WGPUSharedTextureMemoryD3DSwapchainBeginState(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  isSwapchain: int
struct_WGPUSharedTextureMemoryD3DSwapchainBeginState.register_fields([('chain', WGPUChainedStruct, 0), ('isSwapchain', WGPUBool, 16)])
@c.record
class struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  handle: ctypes.c_void_p
  useKeyedMutex: int
struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('handle', ctypes.c_void_p, 16), ('useKeyedMutex', WGPUBool, 24)])
@c.record
class struct_WGPUSharedTextureMemoryEGLImageDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  image: ctypes.c_void_p
struct_WGPUSharedTextureMemoryEGLImageDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('image', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSharedTextureMemoryIOSurfaceDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  ioSurface: ctypes.c_void_p
struct_WGPUSharedTextureMemoryIOSurfaceDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('ioSurface', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  handle: ctypes.c_void_p
  useExternalFormat: int
struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('handle', ctypes.c_void_p, 16), ('useExternalFormat', WGPUBool, 24)])
@c.record
class struct_WGPUSharedTextureMemoryBeginAccessDescriptor(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  concurrentRead: int
  initialized: int
  fenceCount: int
  fences: c.POINTER[c.POINTER[struct_WGPUSharedFenceImpl]]
  signaledValues: c.POINTER[ctypes.c_uint64]
struct_WGPUSharedTextureMemoryBeginAccessDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('concurrentRead', WGPUBool, 8), ('initialized', WGPUBool, 12), ('fenceCount', size_t, 16), ('fences', c.POINTER[WGPUSharedFence], 24), ('signaledValues', c.POINTER[uint64_t], 32)])
@c.record
class struct_WGPUSharedTextureMemoryDmaBufPlane(c.Struct):
  SIZE = 24
  fd: int
  offset: int
  stride: int
struct_WGPUSharedTextureMemoryDmaBufPlane.register_fields([('fd', ctypes.c_int32, 0), ('offset', uint64_t, 8), ('stride', uint32_t, 16)])
@c.record
class struct_WGPUSharedTextureMemoryEndAccessState(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
  initialized: int
  fenceCount: int
  fences: c.POINTER[c.POINTER[struct_WGPUSharedFenceImpl]]
  signaledValues: c.POINTER[ctypes.c_uint64]
struct_WGPUSharedTextureMemoryEndAccessState.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0), ('initialized', WGPUBool, 8), ('fenceCount', size_t, 16), ('fences', c.POINTER[WGPUSharedFence], 24), ('signaledValues', c.POINTER[uint64_t], 32)])
@c.record
class struct_WGPUSharedTextureMemoryOpaqueFDDescriptor(c.Struct):
  SIZE = 48
  chain: struct_WGPUChainedStruct
  vkImageCreateInfo: ctypes.c_void_p
  memoryFD: int
  memoryTypeIndex: int
  allocationSize: int
  dedicatedAllocation: int
struct_WGPUSharedTextureMemoryOpaqueFDDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('vkImageCreateInfo', ctypes.c_void_p, 16), ('memoryFD', ctypes.c_int32, 24), ('memoryTypeIndex', uint32_t, 28), ('allocationSize', uint64_t, 32), ('dedicatedAllocation', WGPUBool, 40)])
@c.record
class struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  dedicatedAllocation: int
struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('dedicatedAllocation', WGPUBool, 16)])
@c.record
class struct_WGPUSharedTextureMemoryVkImageLayoutBeginState(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  oldLayout: int
  newLayout: int
int32_t: TypeAlias = ctypes.c_int32
struct_WGPUSharedTextureMemoryVkImageLayoutBeginState.register_fields([('chain', WGPUChainedStruct, 0), ('oldLayout', int32_t, 16), ('newLayout', int32_t, 20)])
@c.record
class struct_WGPUSharedTextureMemoryVkImageLayoutEndState(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStructOut
  oldLayout: int
  newLayout: int
struct_WGPUSharedTextureMemoryVkImageLayoutEndState.register_fields([('chain', WGPUChainedStructOut, 0), ('oldLayout', int32_t, 16), ('newLayout', int32_t, 20)])
@c.record
class struct_WGPUSharedTextureMemoryZirconHandleDescriptor(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  memoryFD: int
  allocationSize: int
struct_WGPUSharedTextureMemoryZirconHandleDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('memoryFD', uint32_t, 16), ('allocationSize', uint64_t, 24)])
@c.record
class struct_WGPUStaticSamplerBindingLayout(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  sampler: c.POINTER[struct_WGPUSamplerImpl]
  sampledTextureBinding: int
struct_WGPUStaticSamplerBindingLayout.register_fields([('chain', WGPUChainedStruct, 0), ('sampler', WGPUSampler, 16), ('sampledTextureBinding', uint32_t, 24)])
@c.record
class struct_WGPUStencilFaceState(c.Struct):
  SIZE = 16
  compare: int
  failOp: int
  depthFailOp: int
  passOp: int
enum_WGPUCompareFunction: dict[int, str] = {(WGPUCompareFunction_Undefined:=0): 'WGPUCompareFunction_Undefined', (WGPUCompareFunction_Never:=1): 'WGPUCompareFunction_Never', (WGPUCompareFunction_Less:=2): 'WGPUCompareFunction_Less', (WGPUCompareFunction_Equal:=3): 'WGPUCompareFunction_Equal', (WGPUCompareFunction_LessEqual:=4): 'WGPUCompareFunction_LessEqual', (WGPUCompareFunction_Greater:=5): 'WGPUCompareFunction_Greater', (WGPUCompareFunction_NotEqual:=6): 'WGPUCompareFunction_NotEqual', (WGPUCompareFunction_GreaterEqual:=7): 'WGPUCompareFunction_GreaterEqual', (WGPUCompareFunction_Always:=8): 'WGPUCompareFunction_Always', (WGPUCompareFunction_Force32:=2147483647): 'WGPUCompareFunction_Force32'}
WGPUCompareFunction: TypeAlias = ctypes.c_uint32
enum_WGPUStencilOperation: dict[int, str] = {(WGPUStencilOperation_Undefined:=0): 'WGPUStencilOperation_Undefined', (WGPUStencilOperation_Keep:=1): 'WGPUStencilOperation_Keep', (WGPUStencilOperation_Zero:=2): 'WGPUStencilOperation_Zero', (WGPUStencilOperation_Replace:=3): 'WGPUStencilOperation_Replace', (WGPUStencilOperation_Invert:=4): 'WGPUStencilOperation_Invert', (WGPUStencilOperation_IncrementClamp:=5): 'WGPUStencilOperation_IncrementClamp', (WGPUStencilOperation_DecrementClamp:=6): 'WGPUStencilOperation_DecrementClamp', (WGPUStencilOperation_IncrementWrap:=7): 'WGPUStencilOperation_IncrementWrap', (WGPUStencilOperation_DecrementWrap:=8): 'WGPUStencilOperation_DecrementWrap', (WGPUStencilOperation_Force32:=2147483647): 'WGPUStencilOperation_Force32'}
WGPUStencilOperation: TypeAlias = ctypes.c_uint32
struct_WGPUStencilFaceState.register_fields([('compare', WGPUCompareFunction, 0), ('failOp', WGPUStencilOperation, 4), ('depthFailOp', WGPUStencilOperation, 8), ('passOp', WGPUStencilOperation, 12)])
@c.record
class struct_WGPUStorageTextureBindingLayout(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  access: int
  format: int
  viewDimension: int
enum_WGPUStorageTextureAccess: dict[int, str] = {(WGPUStorageTextureAccess_BindingNotUsed:=0): 'WGPUStorageTextureAccess_BindingNotUsed', (WGPUStorageTextureAccess_WriteOnly:=1): 'WGPUStorageTextureAccess_WriteOnly', (WGPUStorageTextureAccess_ReadOnly:=2): 'WGPUStorageTextureAccess_ReadOnly', (WGPUStorageTextureAccess_ReadWrite:=3): 'WGPUStorageTextureAccess_ReadWrite', (WGPUStorageTextureAccess_Force32:=2147483647): 'WGPUStorageTextureAccess_Force32'}
WGPUStorageTextureAccess: TypeAlias = ctypes.c_uint32
enum_WGPUTextureViewDimension: dict[int, str] = {(WGPUTextureViewDimension_Undefined:=0): 'WGPUTextureViewDimension_Undefined', (WGPUTextureViewDimension_1D:=1): 'WGPUTextureViewDimension_1D', (WGPUTextureViewDimension_2D:=2): 'WGPUTextureViewDimension_2D', (WGPUTextureViewDimension_2DArray:=3): 'WGPUTextureViewDimension_2DArray', (WGPUTextureViewDimension_Cube:=4): 'WGPUTextureViewDimension_Cube', (WGPUTextureViewDimension_CubeArray:=5): 'WGPUTextureViewDimension_CubeArray', (WGPUTextureViewDimension_3D:=6): 'WGPUTextureViewDimension_3D', (WGPUTextureViewDimension_Force32:=2147483647): 'WGPUTextureViewDimension_Force32'}
WGPUTextureViewDimension: TypeAlias = ctypes.c_uint32
struct_WGPUStorageTextureBindingLayout.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('access', WGPUStorageTextureAccess, 8), ('format', WGPUTextureFormat, 12), ('viewDimension', WGPUTextureViewDimension, 16)])
@c.record
class struct_WGPUSupportedFeatures(c.Struct):
  SIZE = 16
  featureCount: int
  features: c.POINTER[ctypes.c_uint32]
enum_WGPUFeatureName: dict[int, str] = {(WGPUFeatureName_DepthClipControl:=1): 'WGPUFeatureName_DepthClipControl', (WGPUFeatureName_Depth32FloatStencil8:=2): 'WGPUFeatureName_Depth32FloatStencil8', (WGPUFeatureName_TimestampQuery:=3): 'WGPUFeatureName_TimestampQuery', (WGPUFeatureName_TextureCompressionBC:=4): 'WGPUFeatureName_TextureCompressionBC', (WGPUFeatureName_TextureCompressionETC2:=5): 'WGPUFeatureName_TextureCompressionETC2', (WGPUFeatureName_TextureCompressionASTC:=6): 'WGPUFeatureName_TextureCompressionASTC', (WGPUFeatureName_IndirectFirstInstance:=7): 'WGPUFeatureName_IndirectFirstInstance', (WGPUFeatureName_ShaderF16:=8): 'WGPUFeatureName_ShaderF16', (WGPUFeatureName_RG11B10UfloatRenderable:=9): 'WGPUFeatureName_RG11B10UfloatRenderable', (WGPUFeatureName_BGRA8UnormStorage:=10): 'WGPUFeatureName_BGRA8UnormStorage', (WGPUFeatureName_Float32Filterable:=11): 'WGPUFeatureName_Float32Filterable', (WGPUFeatureName_Float32Blendable:=12): 'WGPUFeatureName_Float32Blendable', (WGPUFeatureName_Subgroups:=13): 'WGPUFeatureName_Subgroups', (WGPUFeatureName_SubgroupsF16:=14): 'WGPUFeatureName_SubgroupsF16', (WGPUFeatureName_DawnInternalUsages:=327680): 'WGPUFeatureName_DawnInternalUsages', (WGPUFeatureName_DawnMultiPlanarFormats:=327681): 'WGPUFeatureName_DawnMultiPlanarFormats', (WGPUFeatureName_DawnNative:=327682): 'WGPUFeatureName_DawnNative', (WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses:=327683): 'WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses', (WGPUFeatureName_ImplicitDeviceSynchronization:=327684): 'WGPUFeatureName_ImplicitDeviceSynchronization', (WGPUFeatureName_ChromiumExperimentalImmediateData:=327685): 'WGPUFeatureName_ChromiumExperimentalImmediateData', (WGPUFeatureName_TransientAttachments:=327686): 'WGPUFeatureName_TransientAttachments', (WGPUFeatureName_MSAARenderToSingleSampled:=327687): 'WGPUFeatureName_MSAARenderToSingleSampled', (WGPUFeatureName_DualSourceBlending:=327688): 'WGPUFeatureName_DualSourceBlending', (WGPUFeatureName_D3D11MultithreadProtected:=327689): 'WGPUFeatureName_D3D11MultithreadProtected', (WGPUFeatureName_ANGLETextureSharing:=327690): 'WGPUFeatureName_ANGLETextureSharing', (WGPUFeatureName_PixelLocalStorageCoherent:=327691): 'WGPUFeatureName_PixelLocalStorageCoherent', (WGPUFeatureName_PixelLocalStorageNonCoherent:=327692): 'WGPUFeatureName_PixelLocalStorageNonCoherent', (WGPUFeatureName_Unorm16TextureFormats:=327693): 'WGPUFeatureName_Unorm16TextureFormats', (WGPUFeatureName_Snorm16TextureFormats:=327694): 'WGPUFeatureName_Snorm16TextureFormats', (WGPUFeatureName_MultiPlanarFormatExtendedUsages:=327695): 'WGPUFeatureName_MultiPlanarFormatExtendedUsages', (WGPUFeatureName_MultiPlanarFormatP010:=327696): 'WGPUFeatureName_MultiPlanarFormatP010', (WGPUFeatureName_HostMappedPointer:=327697): 'WGPUFeatureName_HostMappedPointer', (WGPUFeatureName_MultiPlanarRenderTargets:=327698): 'WGPUFeatureName_MultiPlanarRenderTargets', (WGPUFeatureName_MultiPlanarFormatNv12a:=327699): 'WGPUFeatureName_MultiPlanarFormatNv12a', (WGPUFeatureName_FramebufferFetch:=327700): 'WGPUFeatureName_FramebufferFetch', (WGPUFeatureName_BufferMapExtendedUsages:=327701): 'WGPUFeatureName_BufferMapExtendedUsages', (WGPUFeatureName_AdapterPropertiesMemoryHeaps:=327702): 'WGPUFeatureName_AdapterPropertiesMemoryHeaps', (WGPUFeatureName_AdapterPropertiesD3D:=327703): 'WGPUFeatureName_AdapterPropertiesD3D', (WGPUFeatureName_AdapterPropertiesVk:=327704): 'WGPUFeatureName_AdapterPropertiesVk', (WGPUFeatureName_R8UnormStorage:=327705): 'WGPUFeatureName_R8UnormStorage', (WGPUFeatureName_FormatCapabilities:=327706): 'WGPUFeatureName_FormatCapabilities', (WGPUFeatureName_DrmFormatCapabilities:=327707): 'WGPUFeatureName_DrmFormatCapabilities', (WGPUFeatureName_Norm16TextureFormats:=327708): 'WGPUFeatureName_Norm16TextureFormats', (WGPUFeatureName_MultiPlanarFormatNv16:=327709): 'WGPUFeatureName_MultiPlanarFormatNv16', (WGPUFeatureName_MultiPlanarFormatNv24:=327710): 'WGPUFeatureName_MultiPlanarFormatNv24', (WGPUFeatureName_MultiPlanarFormatP210:=327711): 'WGPUFeatureName_MultiPlanarFormatP210', (WGPUFeatureName_MultiPlanarFormatP410:=327712): 'WGPUFeatureName_MultiPlanarFormatP410', (WGPUFeatureName_SharedTextureMemoryVkDedicatedAllocation:=327713): 'WGPUFeatureName_SharedTextureMemoryVkDedicatedAllocation', (WGPUFeatureName_SharedTextureMemoryAHardwareBuffer:=327714): 'WGPUFeatureName_SharedTextureMemoryAHardwareBuffer', (WGPUFeatureName_SharedTextureMemoryDmaBuf:=327715): 'WGPUFeatureName_SharedTextureMemoryDmaBuf', (WGPUFeatureName_SharedTextureMemoryOpaqueFD:=327716): 'WGPUFeatureName_SharedTextureMemoryOpaqueFD', (WGPUFeatureName_SharedTextureMemoryZirconHandle:=327717): 'WGPUFeatureName_SharedTextureMemoryZirconHandle', (WGPUFeatureName_SharedTextureMemoryDXGISharedHandle:=327718): 'WGPUFeatureName_SharedTextureMemoryDXGISharedHandle', (WGPUFeatureName_SharedTextureMemoryD3D11Texture2D:=327719): 'WGPUFeatureName_SharedTextureMemoryD3D11Texture2D', (WGPUFeatureName_SharedTextureMemoryIOSurface:=327720): 'WGPUFeatureName_SharedTextureMemoryIOSurface', (WGPUFeatureName_SharedTextureMemoryEGLImage:=327721): 'WGPUFeatureName_SharedTextureMemoryEGLImage', (WGPUFeatureName_SharedFenceVkSemaphoreOpaqueFD:=327722): 'WGPUFeatureName_SharedFenceVkSemaphoreOpaqueFD', (WGPUFeatureName_SharedFenceSyncFD:=327723): 'WGPUFeatureName_SharedFenceSyncFD', (WGPUFeatureName_SharedFenceVkSemaphoreZirconHandle:=327724): 'WGPUFeatureName_SharedFenceVkSemaphoreZirconHandle', (WGPUFeatureName_SharedFenceDXGISharedHandle:=327725): 'WGPUFeatureName_SharedFenceDXGISharedHandle', (WGPUFeatureName_SharedFenceMTLSharedEvent:=327726): 'WGPUFeatureName_SharedFenceMTLSharedEvent', (WGPUFeatureName_SharedBufferMemoryD3D12Resource:=327727): 'WGPUFeatureName_SharedBufferMemoryD3D12Resource', (WGPUFeatureName_StaticSamplers:=327728): 'WGPUFeatureName_StaticSamplers', (WGPUFeatureName_YCbCrVulkanSamplers:=327729): 'WGPUFeatureName_YCbCrVulkanSamplers', (WGPUFeatureName_ShaderModuleCompilationOptions:=327730): 'WGPUFeatureName_ShaderModuleCompilationOptions', (WGPUFeatureName_DawnLoadResolveTexture:=327731): 'WGPUFeatureName_DawnLoadResolveTexture', (WGPUFeatureName_DawnPartialLoadResolveTexture:=327732): 'WGPUFeatureName_DawnPartialLoadResolveTexture', (WGPUFeatureName_MultiDrawIndirect:=327733): 'WGPUFeatureName_MultiDrawIndirect', (WGPUFeatureName_ClipDistances:=327734): 'WGPUFeatureName_ClipDistances', (WGPUFeatureName_DawnTexelCopyBufferRowAlignment:=327735): 'WGPUFeatureName_DawnTexelCopyBufferRowAlignment', (WGPUFeatureName_FlexibleTextureViews:=327736): 'WGPUFeatureName_FlexibleTextureViews', (WGPUFeatureName_Force32:=2147483647): 'WGPUFeatureName_Force32'}
WGPUFeatureName: TypeAlias = ctypes.c_uint32
const_enum_WGPUFeatureName: dict[int, str] = {(WGPUFeatureName_DepthClipControl:=1): 'WGPUFeatureName_DepthClipControl', (WGPUFeatureName_Depth32FloatStencil8:=2): 'WGPUFeatureName_Depth32FloatStencil8', (WGPUFeatureName_TimestampQuery:=3): 'WGPUFeatureName_TimestampQuery', (WGPUFeatureName_TextureCompressionBC:=4): 'WGPUFeatureName_TextureCompressionBC', (WGPUFeatureName_TextureCompressionETC2:=5): 'WGPUFeatureName_TextureCompressionETC2', (WGPUFeatureName_TextureCompressionASTC:=6): 'WGPUFeatureName_TextureCompressionASTC', (WGPUFeatureName_IndirectFirstInstance:=7): 'WGPUFeatureName_IndirectFirstInstance', (WGPUFeatureName_ShaderF16:=8): 'WGPUFeatureName_ShaderF16', (WGPUFeatureName_RG11B10UfloatRenderable:=9): 'WGPUFeatureName_RG11B10UfloatRenderable', (WGPUFeatureName_BGRA8UnormStorage:=10): 'WGPUFeatureName_BGRA8UnormStorage', (WGPUFeatureName_Float32Filterable:=11): 'WGPUFeatureName_Float32Filterable', (WGPUFeatureName_Float32Blendable:=12): 'WGPUFeatureName_Float32Blendable', (WGPUFeatureName_Subgroups:=13): 'WGPUFeatureName_Subgroups', (WGPUFeatureName_SubgroupsF16:=14): 'WGPUFeatureName_SubgroupsF16', (WGPUFeatureName_DawnInternalUsages:=327680): 'WGPUFeatureName_DawnInternalUsages', (WGPUFeatureName_DawnMultiPlanarFormats:=327681): 'WGPUFeatureName_DawnMultiPlanarFormats', (WGPUFeatureName_DawnNative:=327682): 'WGPUFeatureName_DawnNative', (WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses:=327683): 'WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses', (WGPUFeatureName_ImplicitDeviceSynchronization:=327684): 'WGPUFeatureName_ImplicitDeviceSynchronization', (WGPUFeatureName_ChromiumExperimentalImmediateData:=327685): 'WGPUFeatureName_ChromiumExperimentalImmediateData', (WGPUFeatureName_TransientAttachments:=327686): 'WGPUFeatureName_TransientAttachments', (WGPUFeatureName_MSAARenderToSingleSampled:=327687): 'WGPUFeatureName_MSAARenderToSingleSampled', (WGPUFeatureName_DualSourceBlending:=327688): 'WGPUFeatureName_DualSourceBlending', (WGPUFeatureName_D3D11MultithreadProtected:=327689): 'WGPUFeatureName_D3D11MultithreadProtected', (WGPUFeatureName_ANGLETextureSharing:=327690): 'WGPUFeatureName_ANGLETextureSharing', (WGPUFeatureName_PixelLocalStorageCoherent:=327691): 'WGPUFeatureName_PixelLocalStorageCoherent', (WGPUFeatureName_PixelLocalStorageNonCoherent:=327692): 'WGPUFeatureName_PixelLocalStorageNonCoherent', (WGPUFeatureName_Unorm16TextureFormats:=327693): 'WGPUFeatureName_Unorm16TextureFormats', (WGPUFeatureName_Snorm16TextureFormats:=327694): 'WGPUFeatureName_Snorm16TextureFormats', (WGPUFeatureName_MultiPlanarFormatExtendedUsages:=327695): 'WGPUFeatureName_MultiPlanarFormatExtendedUsages', (WGPUFeatureName_MultiPlanarFormatP010:=327696): 'WGPUFeatureName_MultiPlanarFormatP010', (WGPUFeatureName_HostMappedPointer:=327697): 'WGPUFeatureName_HostMappedPointer', (WGPUFeatureName_MultiPlanarRenderTargets:=327698): 'WGPUFeatureName_MultiPlanarRenderTargets', (WGPUFeatureName_MultiPlanarFormatNv12a:=327699): 'WGPUFeatureName_MultiPlanarFormatNv12a', (WGPUFeatureName_FramebufferFetch:=327700): 'WGPUFeatureName_FramebufferFetch', (WGPUFeatureName_BufferMapExtendedUsages:=327701): 'WGPUFeatureName_BufferMapExtendedUsages', (WGPUFeatureName_AdapterPropertiesMemoryHeaps:=327702): 'WGPUFeatureName_AdapterPropertiesMemoryHeaps', (WGPUFeatureName_AdapterPropertiesD3D:=327703): 'WGPUFeatureName_AdapterPropertiesD3D', (WGPUFeatureName_AdapterPropertiesVk:=327704): 'WGPUFeatureName_AdapterPropertiesVk', (WGPUFeatureName_R8UnormStorage:=327705): 'WGPUFeatureName_R8UnormStorage', (WGPUFeatureName_FormatCapabilities:=327706): 'WGPUFeatureName_FormatCapabilities', (WGPUFeatureName_DrmFormatCapabilities:=327707): 'WGPUFeatureName_DrmFormatCapabilities', (WGPUFeatureName_Norm16TextureFormats:=327708): 'WGPUFeatureName_Norm16TextureFormats', (WGPUFeatureName_MultiPlanarFormatNv16:=327709): 'WGPUFeatureName_MultiPlanarFormatNv16', (WGPUFeatureName_MultiPlanarFormatNv24:=327710): 'WGPUFeatureName_MultiPlanarFormatNv24', (WGPUFeatureName_MultiPlanarFormatP210:=327711): 'WGPUFeatureName_MultiPlanarFormatP210', (WGPUFeatureName_MultiPlanarFormatP410:=327712): 'WGPUFeatureName_MultiPlanarFormatP410', (WGPUFeatureName_SharedTextureMemoryVkDedicatedAllocation:=327713): 'WGPUFeatureName_SharedTextureMemoryVkDedicatedAllocation', (WGPUFeatureName_SharedTextureMemoryAHardwareBuffer:=327714): 'WGPUFeatureName_SharedTextureMemoryAHardwareBuffer', (WGPUFeatureName_SharedTextureMemoryDmaBuf:=327715): 'WGPUFeatureName_SharedTextureMemoryDmaBuf', (WGPUFeatureName_SharedTextureMemoryOpaqueFD:=327716): 'WGPUFeatureName_SharedTextureMemoryOpaqueFD', (WGPUFeatureName_SharedTextureMemoryZirconHandle:=327717): 'WGPUFeatureName_SharedTextureMemoryZirconHandle', (WGPUFeatureName_SharedTextureMemoryDXGISharedHandle:=327718): 'WGPUFeatureName_SharedTextureMemoryDXGISharedHandle', (WGPUFeatureName_SharedTextureMemoryD3D11Texture2D:=327719): 'WGPUFeatureName_SharedTextureMemoryD3D11Texture2D', (WGPUFeatureName_SharedTextureMemoryIOSurface:=327720): 'WGPUFeatureName_SharedTextureMemoryIOSurface', (WGPUFeatureName_SharedTextureMemoryEGLImage:=327721): 'WGPUFeatureName_SharedTextureMemoryEGLImage', (WGPUFeatureName_SharedFenceVkSemaphoreOpaqueFD:=327722): 'WGPUFeatureName_SharedFenceVkSemaphoreOpaqueFD', (WGPUFeatureName_SharedFenceSyncFD:=327723): 'WGPUFeatureName_SharedFenceSyncFD', (WGPUFeatureName_SharedFenceVkSemaphoreZirconHandle:=327724): 'WGPUFeatureName_SharedFenceVkSemaphoreZirconHandle', (WGPUFeatureName_SharedFenceDXGISharedHandle:=327725): 'WGPUFeatureName_SharedFenceDXGISharedHandle', (WGPUFeatureName_SharedFenceMTLSharedEvent:=327726): 'WGPUFeatureName_SharedFenceMTLSharedEvent', (WGPUFeatureName_SharedBufferMemoryD3D12Resource:=327727): 'WGPUFeatureName_SharedBufferMemoryD3D12Resource', (WGPUFeatureName_StaticSamplers:=327728): 'WGPUFeatureName_StaticSamplers', (WGPUFeatureName_YCbCrVulkanSamplers:=327729): 'WGPUFeatureName_YCbCrVulkanSamplers', (WGPUFeatureName_ShaderModuleCompilationOptions:=327730): 'WGPUFeatureName_ShaderModuleCompilationOptions', (WGPUFeatureName_DawnLoadResolveTexture:=327731): 'WGPUFeatureName_DawnLoadResolveTexture', (WGPUFeatureName_DawnPartialLoadResolveTexture:=327732): 'WGPUFeatureName_DawnPartialLoadResolveTexture', (WGPUFeatureName_MultiDrawIndirect:=327733): 'WGPUFeatureName_MultiDrawIndirect', (WGPUFeatureName_ClipDistances:=327734): 'WGPUFeatureName_ClipDistances', (WGPUFeatureName_DawnTexelCopyBufferRowAlignment:=327735): 'WGPUFeatureName_DawnTexelCopyBufferRowAlignment', (WGPUFeatureName_FlexibleTextureViews:=327736): 'WGPUFeatureName_FlexibleTextureViews', (WGPUFeatureName_Force32:=2147483647): 'WGPUFeatureName_Force32'}
struct_WGPUSupportedFeatures.register_fields([('featureCount', size_t, 0), ('features', c.POINTER[WGPUFeatureName], 8)])
@c.record
class struct_WGPUSurfaceCapabilities(c.Struct):
  SIZE = 64
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
  usages: int
  formatCount: int
  formats: c.POINTER[ctypes.c_uint32]
  presentModeCount: int
  presentModes: c.POINTER[ctypes.c_uint32]
  alphaModeCount: int
  alphaModes: c.POINTER[ctypes.c_uint32]
enum_WGPUPresentMode: dict[int, str] = {(WGPUPresentMode_Fifo:=1): 'WGPUPresentMode_Fifo', (WGPUPresentMode_FifoRelaxed:=2): 'WGPUPresentMode_FifoRelaxed', (WGPUPresentMode_Immediate:=3): 'WGPUPresentMode_Immediate', (WGPUPresentMode_Mailbox:=4): 'WGPUPresentMode_Mailbox', (WGPUPresentMode_Force32:=2147483647): 'WGPUPresentMode_Force32'}
WGPUPresentMode: TypeAlias = ctypes.c_uint32
enum_WGPUCompositeAlphaMode: dict[int, str] = {(WGPUCompositeAlphaMode_Auto:=0): 'WGPUCompositeAlphaMode_Auto', (WGPUCompositeAlphaMode_Opaque:=1): 'WGPUCompositeAlphaMode_Opaque', (WGPUCompositeAlphaMode_Premultiplied:=2): 'WGPUCompositeAlphaMode_Premultiplied', (WGPUCompositeAlphaMode_Unpremultiplied:=3): 'WGPUCompositeAlphaMode_Unpremultiplied', (WGPUCompositeAlphaMode_Inherit:=4): 'WGPUCompositeAlphaMode_Inherit', (WGPUCompositeAlphaMode_Force32:=2147483647): 'WGPUCompositeAlphaMode_Force32'}
WGPUCompositeAlphaMode: TypeAlias = ctypes.c_uint32
const_enum_WGPUTextureFormat: dict[int, str] = {(WGPUTextureFormat_Undefined:=0): 'WGPUTextureFormat_Undefined', (WGPUTextureFormat_R8Unorm:=1): 'WGPUTextureFormat_R8Unorm', (WGPUTextureFormat_R8Snorm:=2): 'WGPUTextureFormat_R8Snorm', (WGPUTextureFormat_R8Uint:=3): 'WGPUTextureFormat_R8Uint', (WGPUTextureFormat_R8Sint:=4): 'WGPUTextureFormat_R8Sint', (WGPUTextureFormat_R16Uint:=5): 'WGPUTextureFormat_R16Uint', (WGPUTextureFormat_R16Sint:=6): 'WGPUTextureFormat_R16Sint', (WGPUTextureFormat_R16Float:=7): 'WGPUTextureFormat_R16Float', (WGPUTextureFormat_RG8Unorm:=8): 'WGPUTextureFormat_RG8Unorm', (WGPUTextureFormat_RG8Snorm:=9): 'WGPUTextureFormat_RG8Snorm', (WGPUTextureFormat_RG8Uint:=10): 'WGPUTextureFormat_RG8Uint', (WGPUTextureFormat_RG8Sint:=11): 'WGPUTextureFormat_RG8Sint', (WGPUTextureFormat_R32Float:=12): 'WGPUTextureFormat_R32Float', (WGPUTextureFormat_R32Uint:=13): 'WGPUTextureFormat_R32Uint', (WGPUTextureFormat_R32Sint:=14): 'WGPUTextureFormat_R32Sint', (WGPUTextureFormat_RG16Uint:=15): 'WGPUTextureFormat_RG16Uint', (WGPUTextureFormat_RG16Sint:=16): 'WGPUTextureFormat_RG16Sint', (WGPUTextureFormat_RG16Float:=17): 'WGPUTextureFormat_RG16Float', (WGPUTextureFormat_RGBA8Unorm:=18): 'WGPUTextureFormat_RGBA8Unorm', (WGPUTextureFormat_RGBA8UnormSrgb:=19): 'WGPUTextureFormat_RGBA8UnormSrgb', (WGPUTextureFormat_RGBA8Snorm:=20): 'WGPUTextureFormat_RGBA8Snorm', (WGPUTextureFormat_RGBA8Uint:=21): 'WGPUTextureFormat_RGBA8Uint', (WGPUTextureFormat_RGBA8Sint:=22): 'WGPUTextureFormat_RGBA8Sint', (WGPUTextureFormat_BGRA8Unorm:=23): 'WGPUTextureFormat_BGRA8Unorm', (WGPUTextureFormat_BGRA8UnormSrgb:=24): 'WGPUTextureFormat_BGRA8UnormSrgb', (WGPUTextureFormat_RGB10A2Uint:=25): 'WGPUTextureFormat_RGB10A2Uint', (WGPUTextureFormat_RGB10A2Unorm:=26): 'WGPUTextureFormat_RGB10A2Unorm', (WGPUTextureFormat_RG11B10Ufloat:=27): 'WGPUTextureFormat_RG11B10Ufloat', (WGPUTextureFormat_RGB9E5Ufloat:=28): 'WGPUTextureFormat_RGB9E5Ufloat', (WGPUTextureFormat_RG32Float:=29): 'WGPUTextureFormat_RG32Float', (WGPUTextureFormat_RG32Uint:=30): 'WGPUTextureFormat_RG32Uint', (WGPUTextureFormat_RG32Sint:=31): 'WGPUTextureFormat_RG32Sint', (WGPUTextureFormat_RGBA16Uint:=32): 'WGPUTextureFormat_RGBA16Uint', (WGPUTextureFormat_RGBA16Sint:=33): 'WGPUTextureFormat_RGBA16Sint', (WGPUTextureFormat_RGBA16Float:=34): 'WGPUTextureFormat_RGBA16Float', (WGPUTextureFormat_RGBA32Float:=35): 'WGPUTextureFormat_RGBA32Float', (WGPUTextureFormat_RGBA32Uint:=36): 'WGPUTextureFormat_RGBA32Uint', (WGPUTextureFormat_RGBA32Sint:=37): 'WGPUTextureFormat_RGBA32Sint', (WGPUTextureFormat_Stencil8:=38): 'WGPUTextureFormat_Stencil8', (WGPUTextureFormat_Depth16Unorm:=39): 'WGPUTextureFormat_Depth16Unorm', (WGPUTextureFormat_Depth24Plus:=40): 'WGPUTextureFormat_Depth24Plus', (WGPUTextureFormat_Depth24PlusStencil8:=41): 'WGPUTextureFormat_Depth24PlusStencil8', (WGPUTextureFormat_Depth32Float:=42): 'WGPUTextureFormat_Depth32Float', (WGPUTextureFormat_Depth32FloatStencil8:=43): 'WGPUTextureFormat_Depth32FloatStencil8', (WGPUTextureFormat_BC1RGBAUnorm:=44): 'WGPUTextureFormat_BC1RGBAUnorm', (WGPUTextureFormat_BC1RGBAUnormSrgb:=45): 'WGPUTextureFormat_BC1RGBAUnormSrgb', (WGPUTextureFormat_BC2RGBAUnorm:=46): 'WGPUTextureFormat_BC2RGBAUnorm', (WGPUTextureFormat_BC2RGBAUnormSrgb:=47): 'WGPUTextureFormat_BC2RGBAUnormSrgb', (WGPUTextureFormat_BC3RGBAUnorm:=48): 'WGPUTextureFormat_BC3RGBAUnorm', (WGPUTextureFormat_BC3RGBAUnormSrgb:=49): 'WGPUTextureFormat_BC3RGBAUnormSrgb', (WGPUTextureFormat_BC4RUnorm:=50): 'WGPUTextureFormat_BC4RUnorm', (WGPUTextureFormat_BC4RSnorm:=51): 'WGPUTextureFormat_BC4RSnorm', (WGPUTextureFormat_BC5RGUnorm:=52): 'WGPUTextureFormat_BC5RGUnorm', (WGPUTextureFormat_BC5RGSnorm:=53): 'WGPUTextureFormat_BC5RGSnorm', (WGPUTextureFormat_BC6HRGBUfloat:=54): 'WGPUTextureFormat_BC6HRGBUfloat', (WGPUTextureFormat_BC6HRGBFloat:=55): 'WGPUTextureFormat_BC6HRGBFloat', (WGPUTextureFormat_BC7RGBAUnorm:=56): 'WGPUTextureFormat_BC7RGBAUnorm', (WGPUTextureFormat_BC7RGBAUnormSrgb:=57): 'WGPUTextureFormat_BC7RGBAUnormSrgb', (WGPUTextureFormat_ETC2RGB8Unorm:=58): 'WGPUTextureFormat_ETC2RGB8Unorm', (WGPUTextureFormat_ETC2RGB8UnormSrgb:=59): 'WGPUTextureFormat_ETC2RGB8UnormSrgb', (WGPUTextureFormat_ETC2RGB8A1Unorm:=60): 'WGPUTextureFormat_ETC2RGB8A1Unorm', (WGPUTextureFormat_ETC2RGB8A1UnormSrgb:=61): 'WGPUTextureFormat_ETC2RGB8A1UnormSrgb', (WGPUTextureFormat_ETC2RGBA8Unorm:=62): 'WGPUTextureFormat_ETC2RGBA8Unorm', (WGPUTextureFormat_ETC2RGBA8UnormSrgb:=63): 'WGPUTextureFormat_ETC2RGBA8UnormSrgb', (WGPUTextureFormat_EACR11Unorm:=64): 'WGPUTextureFormat_EACR11Unorm', (WGPUTextureFormat_EACR11Snorm:=65): 'WGPUTextureFormat_EACR11Snorm', (WGPUTextureFormat_EACRG11Unorm:=66): 'WGPUTextureFormat_EACRG11Unorm', (WGPUTextureFormat_EACRG11Snorm:=67): 'WGPUTextureFormat_EACRG11Snorm', (WGPUTextureFormat_ASTC4x4Unorm:=68): 'WGPUTextureFormat_ASTC4x4Unorm', (WGPUTextureFormat_ASTC4x4UnormSrgb:=69): 'WGPUTextureFormat_ASTC4x4UnormSrgb', (WGPUTextureFormat_ASTC5x4Unorm:=70): 'WGPUTextureFormat_ASTC5x4Unorm', (WGPUTextureFormat_ASTC5x4UnormSrgb:=71): 'WGPUTextureFormat_ASTC5x4UnormSrgb', (WGPUTextureFormat_ASTC5x5Unorm:=72): 'WGPUTextureFormat_ASTC5x5Unorm', (WGPUTextureFormat_ASTC5x5UnormSrgb:=73): 'WGPUTextureFormat_ASTC5x5UnormSrgb', (WGPUTextureFormat_ASTC6x5Unorm:=74): 'WGPUTextureFormat_ASTC6x5Unorm', (WGPUTextureFormat_ASTC6x5UnormSrgb:=75): 'WGPUTextureFormat_ASTC6x5UnormSrgb', (WGPUTextureFormat_ASTC6x6Unorm:=76): 'WGPUTextureFormat_ASTC6x6Unorm', (WGPUTextureFormat_ASTC6x6UnormSrgb:=77): 'WGPUTextureFormat_ASTC6x6UnormSrgb', (WGPUTextureFormat_ASTC8x5Unorm:=78): 'WGPUTextureFormat_ASTC8x5Unorm', (WGPUTextureFormat_ASTC8x5UnormSrgb:=79): 'WGPUTextureFormat_ASTC8x5UnormSrgb', (WGPUTextureFormat_ASTC8x6Unorm:=80): 'WGPUTextureFormat_ASTC8x6Unorm', (WGPUTextureFormat_ASTC8x6UnormSrgb:=81): 'WGPUTextureFormat_ASTC8x6UnormSrgb', (WGPUTextureFormat_ASTC8x8Unorm:=82): 'WGPUTextureFormat_ASTC8x8Unorm', (WGPUTextureFormat_ASTC8x8UnormSrgb:=83): 'WGPUTextureFormat_ASTC8x8UnormSrgb', (WGPUTextureFormat_ASTC10x5Unorm:=84): 'WGPUTextureFormat_ASTC10x5Unorm', (WGPUTextureFormat_ASTC10x5UnormSrgb:=85): 'WGPUTextureFormat_ASTC10x5UnormSrgb', (WGPUTextureFormat_ASTC10x6Unorm:=86): 'WGPUTextureFormat_ASTC10x6Unorm', (WGPUTextureFormat_ASTC10x6UnormSrgb:=87): 'WGPUTextureFormat_ASTC10x6UnormSrgb', (WGPUTextureFormat_ASTC10x8Unorm:=88): 'WGPUTextureFormat_ASTC10x8Unorm', (WGPUTextureFormat_ASTC10x8UnormSrgb:=89): 'WGPUTextureFormat_ASTC10x8UnormSrgb', (WGPUTextureFormat_ASTC10x10Unorm:=90): 'WGPUTextureFormat_ASTC10x10Unorm', (WGPUTextureFormat_ASTC10x10UnormSrgb:=91): 'WGPUTextureFormat_ASTC10x10UnormSrgb', (WGPUTextureFormat_ASTC12x10Unorm:=92): 'WGPUTextureFormat_ASTC12x10Unorm', (WGPUTextureFormat_ASTC12x10UnormSrgb:=93): 'WGPUTextureFormat_ASTC12x10UnormSrgb', (WGPUTextureFormat_ASTC12x12Unorm:=94): 'WGPUTextureFormat_ASTC12x12Unorm', (WGPUTextureFormat_ASTC12x12UnormSrgb:=95): 'WGPUTextureFormat_ASTC12x12UnormSrgb', (WGPUTextureFormat_R16Unorm:=327680): 'WGPUTextureFormat_R16Unorm', (WGPUTextureFormat_RG16Unorm:=327681): 'WGPUTextureFormat_RG16Unorm', (WGPUTextureFormat_RGBA16Unorm:=327682): 'WGPUTextureFormat_RGBA16Unorm', (WGPUTextureFormat_R16Snorm:=327683): 'WGPUTextureFormat_R16Snorm', (WGPUTextureFormat_RG16Snorm:=327684): 'WGPUTextureFormat_RG16Snorm', (WGPUTextureFormat_RGBA16Snorm:=327685): 'WGPUTextureFormat_RGBA16Snorm', (WGPUTextureFormat_R8BG8Biplanar420Unorm:=327686): 'WGPUTextureFormat_R8BG8Biplanar420Unorm', (WGPUTextureFormat_R10X6BG10X6Biplanar420Unorm:=327687): 'WGPUTextureFormat_R10X6BG10X6Biplanar420Unorm', (WGPUTextureFormat_R8BG8A8Triplanar420Unorm:=327688): 'WGPUTextureFormat_R8BG8A8Triplanar420Unorm', (WGPUTextureFormat_R8BG8Biplanar422Unorm:=327689): 'WGPUTextureFormat_R8BG8Biplanar422Unorm', (WGPUTextureFormat_R8BG8Biplanar444Unorm:=327690): 'WGPUTextureFormat_R8BG8Biplanar444Unorm', (WGPUTextureFormat_R10X6BG10X6Biplanar422Unorm:=327691): 'WGPUTextureFormat_R10X6BG10X6Biplanar422Unorm', (WGPUTextureFormat_R10X6BG10X6Biplanar444Unorm:=327692): 'WGPUTextureFormat_R10X6BG10X6Biplanar444Unorm', (WGPUTextureFormat_External:=327693): 'WGPUTextureFormat_External', (WGPUTextureFormat_Force32:=2147483647): 'WGPUTextureFormat_Force32'}
const_enum_WGPUPresentMode: dict[int, str] = {(WGPUPresentMode_Fifo:=1): 'WGPUPresentMode_Fifo', (WGPUPresentMode_FifoRelaxed:=2): 'WGPUPresentMode_FifoRelaxed', (WGPUPresentMode_Immediate:=3): 'WGPUPresentMode_Immediate', (WGPUPresentMode_Mailbox:=4): 'WGPUPresentMode_Mailbox', (WGPUPresentMode_Force32:=2147483647): 'WGPUPresentMode_Force32'}
const_enum_WGPUCompositeAlphaMode: dict[int, str] = {(WGPUCompositeAlphaMode_Auto:=0): 'WGPUCompositeAlphaMode_Auto', (WGPUCompositeAlphaMode_Opaque:=1): 'WGPUCompositeAlphaMode_Opaque', (WGPUCompositeAlphaMode_Premultiplied:=2): 'WGPUCompositeAlphaMode_Premultiplied', (WGPUCompositeAlphaMode_Unpremultiplied:=3): 'WGPUCompositeAlphaMode_Unpremultiplied', (WGPUCompositeAlphaMode_Inherit:=4): 'WGPUCompositeAlphaMode_Inherit', (WGPUCompositeAlphaMode_Force32:=2147483647): 'WGPUCompositeAlphaMode_Force32'}
struct_WGPUSurfaceCapabilities.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0), ('usages', WGPUTextureUsage, 8), ('formatCount', size_t, 16), ('formats', c.POINTER[WGPUTextureFormat], 24), ('presentModeCount', size_t, 32), ('presentModes', c.POINTER[WGPUPresentMode], 40), ('alphaModeCount', size_t, 48), ('alphaModes', c.POINTER[WGPUCompositeAlphaMode], 56)])
@c.record
class struct_WGPUSurfaceConfiguration(c.Struct):
  SIZE = 64
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  device: c.POINTER[struct_WGPUDeviceImpl]
  format: int
  usage: int
  viewFormatCount: int
  viewFormats: c.POINTER[ctypes.c_uint32]
  alphaMode: int
  width: int
  height: int
  presentMode: int
struct_WGPUSurfaceConfiguration.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('device', WGPUDevice, 8), ('format', WGPUTextureFormat, 16), ('usage', WGPUTextureUsage, 24), ('viewFormatCount', size_t, 32), ('viewFormats', c.POINTER[WGPUTextureFormat], 40), ('alphaMode', WGPUCompositeAlphaMode, 48), ('width', uint32_t, 52), ('height', uint32_t, 56), ('presentMode', WGPUPresentMode, 60)])
@c.record
class struct_WGPUSurfaceDescriptorFromWindowsCoreWindow(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  coreWindow: ctypes.c_void_p
struct_WGPUSurfaceDescriptorFromWindowsCoreWindow.register_fields([('chain', WGPUChainedStruct, 0), ('coreWindow', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  swapChainPanel: ctypes.c_void_p
struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel.register_fields([('chain', WGPUChainedStruct, 0), ('swapChainPanel', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSurfaceSourceXCBWindow(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  connection: ctypes.c_void_p
  window: int
struct_WGPUSurfaceSourceXCBWindow.register_fields([('chain', WGPUChainedStruct, 0), ('connection', ctypes.c_void_p, 16), ('window', uint32_t, 24)])
@c.record
class struct_WGPUSurfaceSourceAndroidNativeWindow(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  window: ctypes.c_void_p
struct_WGPUSurfaceSourceAndroidNativeWindow.register_fields([('chain', WGPUChainedStruct, 0), ('window', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSurfaceSourceMetalLayer(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  layer: ctypes.c_void_p
struct_WGPUSurfaceSourceMetalLayer.register_fields([('chain', WGPUChainedStruct, 0), ('layer', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUSurfaceSourceWaylandSurface(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  display: ctypes.c_void_p
  surface: ctypes.c_void_p
struct_WGPUSurfaceSourceWaylandSurface.register_fields([('chain', WGPUChainedStruct, 0), ('display', ctypes.c_void_p, 16), ('surface', ctypes.c_void_p, 24)])
@c.record
class struct_WGPUSurfaceSourceWindowsHWND(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  hinstance: ctypes.c_void_p
  hwnd: ctypes.c_void_p
struct_WGPUSurfaceSourceWindowsHWND.register_fields([('chain', WGPUChainedStruct, 0), ('hinstance', ctypes.c_void_p, 16), ('hwnd', ctypes.c_void_p, 24)])
@c.record
class struct_WGPUSurfaceSourceXlibWindow(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  display: ctypes.c_void_p
  window: int
struct_WGPUSurfaceSourceXlibWindow.register_fields([('chain', WGPUChainedStruct, 0), ('display', ctypes.c_void_p, 16), ('window', uint64_t, 24)])
@c.record
class struct_WGPUSurfaceTexture(c.Struct):
  SIZE = 16
  texture: c.POINTER[struct_WGPUTextureImpl]
  suboptimal: int
  status: int
enum_WGPUSurfaceGetCurrentTextureStatus: dict[int, str] = {(WGPUSurfaceGetCurrentTextureStatus_Success:=1): 'WGPUSurfaceGetCurrentTextureStatus_Success', (WGPUSurfaceGetCurrentTextureStatus_Timeout:=2): 'WGPUSurfaceGetCurrentTextureStatus_Timeout', (WGPUSurfaceGetCurrentTextureStatus_Outdated:=3): 'WGPUSurfaceGetCurrentTextureStatus_Outdated', (WGPUSurfaceGetCurrentTextureStatus_Lost:=4): 'WGPUSurfaceGetCurrentTextureStatus_Lost', (WGPUSurfaceGetCurrentTextureStatus_OutOfMemory:=5): 'WGPUSurfaceGetCurrentTextureStatus_OutOfMemory', (WGPUSurfaceGetCurrentTextureStatus_DeviceLost:=6): 'WGPUSurfaceGetCurrentTextureStatus_DeviceLost', (WGPUSurfaceGetCurrentTextureStatus_Error:=7): 'WGPUSurfaceGetCurrentTextureStatus_Error', (WGPUSurfaceGetCurrentTextureStatus_Force32:=2147483647): 'WGPUSurfaceGetCurrentTextureStatus_Force32'}
WGPUSurfaceGetCurrentTextureStatus: TypeAlias = ctypes.c_uint32
struct_WGPUSurfaceTexture.register_fields([('texture', WGPUTexture, 0), ('suboptimal', WGPUBool, 8), ('status', WGPUSurfaceGetCurrentTextureStatus, 12)])
@c.record
class struct_WGPUTextureBindingLayout(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  sampleType: int
  viewDimension: int
  multisampled: int
enum_WGPUTextureSampleType: dict[int, str] = {(WGPUTextureSampleType_BindingNotUsed:=0): 'WGPUTextureSampleType_BindingNotUsed', (WGPUTextureSampleType_Float:=1): 'WGPUTextureSampleType_Float', (WGPUTextureSampleType_UnfilterableFloat:=2): 'WGPUTextureSampleType_UnfilterableFloat', (WGPUTextureSampleType_Depth:=3): 'WGPUTextureSampleType_Depth', (WGPUTextureSampleType_Sint:=4): 'WGPUTextureSampleType_Sint', (WGPUTextureSampleType_Uint:=5): 'WGPUTextureSampleType_Uint', (WGPUTextureSampleType_Force32:=2147483647): 'WGPUTextureSampleType_Force32'}
WGPUTextureSampleType: TypeAlias = ctypes.c_uint32
struct_WGPUTextureBindingLayout.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('sampleType', WGPUTextureSampleType, 8), ('viewDimension', WGPUTextureViewDimension, 12), ('multisampled', WGPUBool, 16)])
@c.record
class struct_WGPUTextureBindingViewDimensionDescriptor(c.Struct):
  SIZE = 24
  chain: struct_WGPUChainedStruct
  textureBindingViewDimension: int
struct_WGPUTextureBindingViewDimensionDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('textureBindingViewDimension', WGPUTextureViewDimension, 16)])
@c.record
class struct_WGPUTextureDataLayout(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  offset: int
  bytesPerRow: int
  rowsPerImage: int
struct_WGPUTextureDataLayout.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('offset', uint64_t, 8), ('bytesPerRow', uint32_t, 16), ('rowsPerImage', uint32_t, 20)])
@c.record
class struct_WGPUUncapturedErrorCallbackInfo(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
  userdata: ctypes.c_void_p
struct_WGPUUncapturedErrorCallbackInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('callback', WGPUErrorCallback, 8), ('userdata', ctypes.c_void_p, 16)])
@c.record
class struct_WGPUVertexAttribute(c.Struct):
  SIZE = 24
  format: int
  offset: int
  shaderLocation: int
enum_WGPUVertexFormat: dict[int, str] = {(WGPUVertexFormat_Uint8:=1): 'WGPUVertexFormat_Uint8', (WGPUVertexFormat_Uint8x2:=2): 'WGPUVertexFormat_Uint8x2', (WGPUVertexFormat_Uint8x4:=3): 'WGPUVertexFormat_Uint8x4', (WGPUVertexFormat_Sint8:=4): 'WGPUVertexFormat_Sint8', (WGPUVertexFormat_Sint8x2:=5): 'WGPUVertexFormat_Sint8x2', (WGPUVertexFormat_Sint8x4:=6): 'WGPUVertexFormat_Sint8x4', (WGPUVertexFormat_Unorm8:=7): 'WGPUVertexFormat_Unorm8', (WGPUVertexFormat_Unorm8x2:=8): 'WGPUVertexFormat_Unorm8x2', (WGPUVertexFormat_Unorm8x4:=9): 'WGPUVertexFormat_Unorm8x4', (WGPUVertexFormat_Snorm8:=10): 'WGPUVertexFormat_Snorm8', (WGPUVertexFormat_Snorm8x2:=11): 'WGPUVertexFormat_Snorm8x2', (WGPUVertexFormat_Snorm8x4:=12): 'WGPUVertexFormat_Snorm8x4', (WGPUVertexFormat_Uint16:=13): 'WGPUVertexFormat_Uint16', (WGPUVertexFormat_Uint16x2:=14): 'WGPUVertexFormat_Uint16x2', (WGPUVertexFormat_Uint16x4:=15): 'WGPUVertexFormat_Uint16x4', (WGPUVertexFormat_Sint16:=16): 'WGPUVertexFormat_Sint16', (WGPUVertexFormat_Sint16x2:=17): 'WGPUVertexFormat_Sint16x2', (WGPUVertexFormat_Sint16x4:=18): 'WGPUVertexFormat_Sint16x4', (WGPUVertexFormat_Unorm16:=19): 'WGPUVertexFormat_Unorm16', (WGPUVertexFormat_Unorm16x2:=20): 'WGPUVertexFormat_Unorm16x2', (WGPUVertexFormat_Unorm16x4:=21): 'WGPUVertexFormat_Unorm16x4', (WGPUVertexFormat_Snorm16:=22): 'WGPUVertexFormat_Snorm16', (WGPUVertexFormat_Snorm16x2:=23): 'WGPUVertexFormat_Snorm16x2', (WGPUVertexFormat_Snorm16x4:=24): 'WGPUVertexFormat_Snorm16x4', (WGPUVertexFormat_Float16:=25): 'WGPUVertexFormat_Float16', (WGPUVertexFormat_Float16x2:=26): 'WGPUVertexFormat_Float16x2', (WGPUVertexFormat_Float16x4:=27): 'WGPUVertexFormat_Float16x4', (WGPUVertexFormat_Float32:=28): 'WGPUVertexFormat_Float32', (WGPUVertexFormat_Float32x2:=29): 'WGPUVertexFormat_Float32x2', (WGPUVertexFormat_Float32x3:=30): 'WGPUVertexFormat_Float32x3', (WGPUVertexFormat_Float32x4:=31): 'WGPUVertexFormat_Float32x4', (WGPUVertexFormat_Uint32:=32): 'WGPUVertexFormat_Uint32', (WGPUVertexFormat_Uint32x2:=33): 'WGPUVertexFormat_Uint32x2', (WGPUVertexFormat_Uint32x3:=34): 'WGPUVertexFormat_Uint32x3', (WGPUVertexFormat_Uint32x4:=35): 'WGPUVertexFormat_Uint32x4', (WGPUVertexFormat_Sint32:=36): 'WGPUVertexFormat_Sint32', (WGPUVertexFormat_Sint32x2:=37): 'WGPUVertexFormat_Sint32x2', (WGPUVertexFormat_Sint32x3:=38): 'WGPUVertexFormat_Sint32x3', (WGPUVertexFormat_Sint32x4:=39): 'WGPUVertexFormat_Sint32x4', (WGPUVertexFormat_Unorm10_10_10_2:=40): 'WGPUVertexFormat_Unorm10_10_10_2', (WGPUVertexFormat_Unorm8x4BGRA:=41): 'WGPUVertexFormat_Unorm8x4BGRA', (WGPUVertexFormat_Force32:=2147483647): 'WGPUVertexFormat_Force32'}
WGPUVertexFormat: TypeAlias = ctypes.c_uint32
struct_WGPUVertexAttribute.register_fields([('format', WGPUVertexFormat, 0), ('offset', uint64_t, 8), ('shaderLocation', uint32_t, 16)])
@c.record
class struct_WGPUYCbCrVkDescriptor(c.Struct):
  SIZE = 72
  chain: struct_WGPUChainedStruct
  vkFormat: int
  vkYCbCrModel: int
  vkYCbCrRange: int
  vkComponentSwizzleRed: int
  vkComponentSwizzleGreen: int
  vkComponentSwizzleBlue: int
  vkComponentSwizzleAlpha: int
  vkXChromaOffset: int
  vkYChromaOffset: int
  vkChromaFilter: int
  forceExplicitReconstruction: int
  externalFormat: int
enum_WGPUFilterMode: dict[int, str] = {(WGPUFilterMode_Undefined:=0): 'WGPUFilterMode_Undefined', (WGPUFilterMode_Nearest:=1): 'WGPUFilterMode_Nearest', (WGPUFilterMode_Linear:=2): 'WGPUFilterMode_Linear', (WGPUFilterMode_Force32:=2147483647): 'WGPUFilterMode_Force32'}
WGPUFilterMode: TypeAlias = ctypes.c_uint32
struct_WGPUYCbCrVkDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('vkFormat', uint32_t, 16), ('vkYCbCrModel', uint32_t, 20), ('vkYCbCrRange', uint32_t, 24), ('vkComponentSwizzleRed', uint32_t, 28), ('vkComponentSwizzleGreen', uint32_t, 32), ('vkComponentSwizzleBlue', uint32_t, 36), ('vkComponentSwizzleAlpha', uint32_t, 40), ('vkXChromaOffset', uint32_t, 44), ('vkYChromaOffset', uint32_t, 48), ('vkChromaFilter', WGPUFilterMode, 52), ('forceExplicitReconstruction', WGPUBool, 56), ('externalFormat', uint64_t, 64)])
@c.record
class struct_WGPUAHardwareBufferProperties(c.Struct):
  SIZE = 72
  yCbCrInfo: struct_WGPUYCbCrVkDescriptor
WGPUYCbCrVkDescriptor: TypeAlias = struct_WGPUYCbCrVkDescriptor
struct_WGPUAHardwareBufferProperties.register_fields([('yCbCrInfo', WGPUYCbCrVkDescriptor, 0)])
@c.record
class struct_WGPUAdapterInfo(c.Struct):
  SIZE = 96
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
  vendor: struct_WGPUStringView
  architecture: struct_WGPUStringView
  device: struct_WGPUStringView
  description: struct_WGPUStringView
  backendType: int
  adapterType: int
  vendorID: int
  deviceID: int
  compatibilityMode: int
enum_WGPUAdapterType: dict[int, str] = {(WGPUAdapterType_DiscreteGPU:=1): 'WGPUAdapterType_DiscreteGPU', (WGPUAdapterType_IntegratedGPU:=2): 'WGPUAdapterType_IntegratedGPU', (WGPUAdapterType_CPU:=3): 'WGPUAdapterType_CPU', (WGPUAdapterType_Unknown:=4): 'WGPUAdapterType_Unknown', (WGPUAdapterType_Force32:=2147483647): 'WGPUAdapterType_Force32'}
WGPUAdapterType: TypeAlias = ctypes.c_uint32
struct_WGPUAdapterInfo.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0), ('vendor', WGPUStringView, 8), ('architecture', WGPUStringView, 24), ('device', WGPUStringView, 40), ('description', WGPUStringView, 56), ('backendType', WGPUBackendType, 72), ('adapterType', WGPUAdapterType, 76), ('vendorID', uint32_t, 80), ('deviceID', uint32_t, 84), ('compatibilityMode', WGPUBool, 88)])
@c.record
class struct_WGPUAdapterPropertiesMemoryHeaps(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStructOut
  heapCount: int
  heapInfo: c.POINTER[struct_WGPUMemoryHeapInfo]
WGPUMemoryHeapInfo: TypeAlias = struct_WGPUMemoryHeapInfo
struct_WGPUAdapterPropertiesMemoryHeaps.register_fields([('chain', WGPUChainedStructOut, 0), ('heapCount', size_t, 16), ('heapInfo', c.POINTER[WGPUMemoryHeapInfo], 24)])
@c.record
class struct_WGPUBindGroupDescriptor(c.Struct):
  SIZE = 48
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  layout: c.POINTER[struct_WGPUBindGroupLayoutImpl]
  entryCount: int
  entries: c.POINTER[struct_WGPUBindGroupEntry]
WGPUBindGroupEntry: TypeAlias = struct_WGPUBindGroupEntry
struct_WGPUBindGroupDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('layout', WGPUBindGroupLayout, 24), ('entryCount', size_t, 32), ('entries', c.POINTER[WGPUBindGroupEntry], 40)])
@c.record
class struct_WGPUBindGroupLayoutEntry(c.Struct):
  SIZE = 112
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  binding: int
  visibility: int
  buffer: struct_WGPUBufferBindingLayout
  sampler: struct_WGPUSamplerBindingLayout
  texture: struct_WGPUTextureBindingLayout
  storageTexture: struct_WGPUStorageTextureBindingLayout
WGPUShaderStage: TypeAlias = ctypes.c_uint64
WGPUBufferBindingLayout: TypeAlias = struct_WGPUBufferBindingLayout
WGPUSamplerBindingLayout: TypeAlias = struct_WGPUSamplerBindingLayout
WGPUTextureBindingLayout: TypeAlias = struct_WGPUTextureBindingLayout
WGPUStorageTextureBindingLayout: TypeAlias = struct_WGPUStorageTextureBindingLayout
struct_WGPUBindGroupLayoutEntry.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('binding', uint32_t, 8), ('visibility', WGPUShaderStage, 16), ('buffer', WGPUBufferBindingLayout, 24), ('sampler', WGPUSamplerBindingLayout, 48), ('texture', WGPUTextureBindingLayout, 64), ('storageTexture', WGPUStorageTextureBindingLayout, 88)])
@c.record
class struct_WGPUBlendState(c.Struct):
  SIZE = 24
  color: struct_WGPUBlendComponent
  alpha: struct_WGPUBlendComponent
WGPUBlendComponent: TypeAlias = struct_WGPUBlendComponent
struct_WGPUBlendState.register_fields([('color', WGPUBlendComponent, 0), ('alpha', WGPUBlendComponent, 12)])
@c.record
class struct_WGPUBufferDescriptor(c.Struct):
  SIZE = 48
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  usage: int
  size: int
  mappedAtCreation: int
struct_WGPUBufferDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('usage', WGPUBufferUsage, 24), ('size', uint64_t, 32), ('mappedAtCreation', WGPUBool, 40)])
@c.record
class struct_WGPUCommandBufferDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPUCommandBufferDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPUCommandEncoderDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPUCommandEncoderDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPUComputePassDescriptor(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  timestampWrites: c.POINTER[struct_WGPUComputePassTimestampWrites]
WGPUComputePassTimestampWrites: TypeAlias = struct_WGPUComputePassTimestampWrites
struct_WGPUComputePassDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('timestampWrites', c.POINTER[WGPUComputePassTimestampWrites], 24)])
@c.record
class struct_WGPUConstantEntry(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  key: struct_WGPUStringView
  value: float
struct_WGPUConstantEntry.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('key', WGPUStringView, 8), ('value', ctypes.c_double, 24)])
@c.record
class struct_WGPUDawnCacheDeviceDescriptor(c.Struct):
  SIZE = 56
  chain: struct_WGPUChainedStruct
  isolationKey: struct_WGPUStringView
  loadDataFunction: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]]
  storeDataFunction: c.CFUNCTYPE[None, [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]]
  functionUserdata: ctypes.c_void_p
WGPUDawnLoadCacheDataFunction: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]]
WGPUDawnStoreCacheDataFunction: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]]
struct_WGPUDawnCacheDeviceDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('isolationKey', WGPUStringView, 16), ('loadDataFunction', WGPUDawnLoadCacheDataFunction, 32), ('storeDataFunction', WGPUDawnStoreCacheDataFunction, 40), ('functionUserdata', ctypes.c_void_p, 48)])
@c.record
class struct_WGPUDepthStencilState(c.Struct):
  SIZE = 72
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  format: int
  depthWriteEnabled: int
  depthCompare: int
  stencilFront: struct_WGPUStencilFaceState
  stencilBack: struct_WGPUStencilFaceState
  stencilReadMask: int
  stencilWriteMask: int
  depthBias: int
  depthBiasSlopeScale: float
  depthBiasClamp: float
enum_WGPUOptionalBool: dict[int, str] = {(WGPUOptionalBool_False:=0): 'WGPUOptionalBool_False', (WGPUOptionalBool_True:=1): 'WGPUOptionalBool_True', (WGPUOptionalBool_Undefined:=2): 'WGPUOptionalBool_Undefined', (WGPUOptionalBool_Force32:=2147483647): 'WGPUOptionalBool_Force32'}
WGPUOptionalBool: TypeAlias = ctypes.c_uint32
WGPUStencilFaceState: TypeAlias = struct_WGPUStencilFaceState
struct_WGPUDepthStencilState.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('format', WGPUTextureFormat, 8), ('depthWriteEnabled', WGPUOptionalBool, 12), ('depthCompare', WGPUCompareFunction, 16), ('stencilFront', WGPUStencilFaceState, 20), ('stencilBack', WGPUStencilFaceState, 36), ('stencilReadMask', uint32_t, 52), ('stencilWriteMask', uint32_t, 56), ('depthBias', int32_t, 60), ('depthBiasSlopeScale', ctypes.c_float, 64), ('depthBiasClamp', ctypes.c_float, 68)])
@c.record
class struct_WGPUDrmFormatCapabilities(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStructOut
  propertiesCount: int
  properties: c.POINTER[struct_WGPUDrmFormatProperties]
WGPUDrmFormatProperties: TypeAlias = struct_WGPUDrmFormatProperties
struct_WGPUDrmFormatCapabilities.register_fields([('chain', WGPUChainedStructOut, 0), ('propertiesCount', size_t, 16), ('properties', c.POINTER[WGPUDrmFormatProperties], 24)])
@c.record
class struct_WGPUExternalTextureDescriptor(c.Struct):
  SIZE = 112
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  plane0: c.POINTER[struct_WGPUTextureViewImpl]
  plane1: c.POINTER[struct_WGPUTextureViewImpl]
  cropOrigin: struct_WGPUOrigin2D
  cropSize: struct_WGPUExtent2D
  apparentSize: struct_WGPUExtent2D
  doYuvToRgbConversionOnly: int
  yuvToRgbConversionMatrix: c.POINTER[ctypes.c_float]
  srcTransferFunctionParameters: c.POINTER[ctypes.c_float]
  dstTransferFunctionParameters: c.POINTER[ctypes.c_float]
  gamutConversionMatrix: c.POINTER[ctypes.c_float]
  mirrored: int
  rotation: int
WGPUOrigin2D: TypeAlias = struct_WGPUOrigin2D
WGPUExtent2D: TypeAlias = struct_WGPUExtent2D
enum_WGPUExternalTextureRotation: dict[int, str] = {(WGPUExternalTextureRotation_Rotate0Degrees:=1): 'WGPUExternalTextureRotation_Rotate0Degrees', (WGPUExternalTextureRotation_Rotate90Degrees:=2): 'WGPUExternalTextureRotation_Rotate90Degrees', (WGPUExternalTextureRotation_Rotate180Degrees:=3): 'WGPUExternalTextureRotation_Rotate180Degrees', (WGPUExternalTextureRotation_Rotate270Degrees:=4): 'WGPUExternalTextureRotation_Rotate270Degrees', (WGPUExternalTextureRotation_Force32:=2147483647): 'WGPUExternalTextureRotation_Force32'}
WGPUExternalTextureRotation: TypeAlias = ctypes.c_uint32
struct_WGPUExternalTextureDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('plane0', WGPUTextureView, 24), ('plane1', WGPUTextureView, 32), ('cropOrigin', WGPUOrigin2D, 40), ('cropSize', WGPUExtent2D, 48), ('apparentSize', WGPUExtent2D, 56), ('doYuvToRgbConversionOnly', WGPUBool, 64), ('yuvToRgbConversionMatrix', c.POINTER[ctypes.c_float], 72), ('srcTransferFunctionParameters', c.POINTER[ctypes.c_float], 80), ('dstTransferFunctionParameters', c.POINTER[ctypes.c_float], 88), ('gamutConversionMatrix', c.POINTER[ctypes.c_float], 96), ('mirrored', WGPUBool, 104), ('rotation', WGPUExternalTextureRotation, 108)])
@c.record
class struct_WGPUFutureWaitInfo(c.Struct):
  SIZE = 16
  future: struct_WGPUFuture
  completed: int
WGPUFuture: TypeAlias = struct_WGPUFuture
struct_WGPUFutureWaitInfo.register_fields([('future', WGPUFuture, 0), ('completed', WGPUBool, 8)])
@c.record
class struct_WGPUImageCopyBuffer(c.Struct):
  SIZE = 32
  layout: struct_WGPUTextureDataLayout
  buffer: c.POINTER[struct_WGPUBufferImpl]
WGPUTextureDataLayout: TypeAlias = struct_WGPUTextureDataLayout
struct_WGPUImageCopyBuffer.register_fields([('layout', WGPUTextureDataLayout, 0), ('buffer', WGPUBuffer, 24)])
@c.record
class struct_WGPUImageCopyExternalTexture(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  externalTexture: c.POINTER[struct_WGPUExternalTextureImpl]
  origin: struct_WGPUOrigin3D
  naturalSize: struct_WGPUExtent2D
WGPUOrigin3D: TypeAlias = struct_WGPUOrigin3D
struct_WGPUImageCopyExternalTexture.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('externalTexture', WGPUExternalTexture, 8), ('origin', WGPUOrigin3D, 16), ('naturalSize', WGPUExtent2D, 28)])
@c.record
class struct_WGPUImageCopyTexture(c.Struct):
  SIZE = 32
  texture: c.POINTER[struct_WGPUTextureImpl]
  mipLevel: int
  origin: struct_WGPUOrigin3D
  aspect: int
enum_WGPUTextureAspect: dict[int, str] = {(WGPUTextureAspect_Undefined:=0): 'WGPUTextureAspect_Undefined', (WGPUTextureAspect_All:=1): 'WGPUTextureAspect_All', (WGPUTextureAspect_StencilOnly:=2): 'WGPUTextureAspect_StencilOnly', (WGPUTextureAspect_DepthOnly:=3): 'WGPUTextureAspect_DepthOnly', (WGPUTextureAspect_Plane0Only:=327680): 'WGPUTextureAspect_Plane0Only', (WGPUTextureAspect_Plane1Only:=327681): 'WGPUTextureAspect_Plane1Only', (WGPUTextureAspect_Plane2Only:=327682): 'WGPUTextureAspect_Plane2Only', (WGPUTextureAspect_Force32:=2147483647): 'WGPUTextureAspect_Force32'}
WGPUTextureAspect: TypeAlias = ctypes.c_uint32
struct_WGPUImageCopyTexture.register_fields([('texture', WGPUTexture, 0), ('mipLevel', uint32_t, 8), ('origin', WGPUOrigin3D, 12), ('aspect', WGPUTextureAspect, 24)])
@c.record
class struct_WGPUInstanceDescriptor(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  features: struct_WGPUInstanceFeatures
WGPUInstanceFeatures: TypeAlias = struct_WGPUInstanceFeatures
struct_WGPUInstanceDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('features', WGPUInstanceFeatures, 8)])
@c.record
class struct_WGPUPipelineLayoutDescriptor(c.Struct):
  SIZE = 48
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  bindGroupLayoutCount: int
  bindGroupLayouts: c.POINTER[c.POINTER[struct_WGPUBindGroupLayoutImpl]]
  immediateDataRangeByteSize: int
struct_WGPUPipelineLayoutDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('bindGroupLayoutCount', size_t, 24), ('bindGroupLayouts', c.POINTER[WGPUBindGroupLayout], 32), ('immediateDataRangeByteSize', uint32_t, 40)])
@c.record
class struct_WGPUPipelineLayoutPixelLocalStorage(c.Struct):
  SIZE = 40
  chain: struct_WGPUChainedStruct
  totalPixelLocalStorageSize: int
  storageAttachmentCount: int
  storageAttachments: c.POINTER[struct_WGPUPipelineLayoutStorageAttachment]
WGPUPipelineLayoutStorageAttachment: TypeAlias = struct_WGPUPipelineLayoutStorageAttachment
struct_WGPUPipelineLayoutPixelLocalStorage.register_fields([('chain', WGPUChainedStruct, 0), ('totalPixelLocalStorageSize', uint64_t, 16), ('storageAttachmentCount', size_t, 24), ('storageAttachments', c.POINTER[WGPUPipelineLayoutStorageAttachment], 32)])
@c.record
class struct_WGPUQuerySetDescriptor(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  type: int
  count: int
enum_WGPUQueryType: dict[int, str] = {(WGPUQueryType_Occlusion:=1): 'WGPUQueryType_Occlusion', (WGPUQueryType_Timestamp:=2): 'WGPUQueryType_Timestamp', (WGPUQueryType_Force32:=2147483647): 'WGPUQueryType_Force32'}
WGPUQueryType: TypeAlias = ctypes.c_uint32
struct_WGPUQuerySetDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('type', WGPUQueryType, 24), ('count', uint32_t, 28)])
@c.record
class struct_WGPUQueueDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPUQueueDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPURenderBundleDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPURenderBundleDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPURenderBundleEncoderDescriptor(c.Struct):
  SIZE = 56
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  colorFormatCount: int
  colorFormats: c.POINTER[ctypes.c_uint32]
  depthStencilFormat: int
  sampleCount: int
  depthReadOnly: int
  stencilReadOnly: int
struct_WGPURenderBundleEncoderDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('colorFormatCount', size_t, 24), ('colorFormats', c.POINTER[WGPUTextureFormat], 32), ('depthStencilFormat', WGPUTextureFormat, 40), ('sampleCount', uint32_t, 44), ('depthReadOnly', WGPUBool, 48), ('stencilReadOnly', WGPUBool, 52)])
@c.record
class struct_WGPURenderPassColorAttachment(c.Struct):
  SIZE = 72
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  view: c.POINTER[struct_WGPUTextureViewImpl]
  depthSlice: int
  resolveTarget: c.POINTER[struct_WGPUTextureViewImpl]
  loadOp: int
  storeOp: int
  clearValue: struct_WGPUColor
WGPUColor: TypeAlias = struct_WGPUColor
struct_WGPURenderPassColorAttachment.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('view', WGPUTextureView, 8), ('depthSlice', uint32_t, 16), ('resolveTarget', WGPUTextureView, 24), ('loadOp', WGPULoadOp, 32), ('storeOp', WGPUStoreOp, 36), ('clearValue', WGPUColor, 40)])
@c.record
class struct_WGPURenderPassStorageAttachment(c.Struct):
  SIZE = 64
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  offset: int
  storage: c.POINTER[struct_WGPUTextureViewImpl]
  loadOp: int
  storeOp: int
  clearValue: struct_WGPUColor
struct_WGPURenderPassStorageAttachment.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('offset', uint64_t, 8), ('storage', WGPUTextureView, 16), ('loadOp', WGPULoadOp, 24), ('storeOp', WGPUStoreOp, 28), ('clearValue', WGPUColor, 32)])
@c.record
class struct_WGPURequiredLimits(c.Struct):
  SIZE = 168
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  limits: struct_WGPULimits
WGPULimits: TypeAlias = struct_WGPULimits
struct_WGPURequiredLimits.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('limits', WGPULimits, 8)])
@c.record
class struct_WGPUSamplerDescriptor(c.Struct):
  SIZE = 64
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  addressModeU: int
  addressModeV: int
  addressModeW: int
  magFilter: int
  minFilter: int
  mipmapFilter: int
  lodMinClamp: float
  lodMaxClamp: float
  compare: int
  maxAnisotropy: int
enum_WGPUAddressMode: dict[int, str] = {(WGPUAddressMode_Undefined:=0): 'WGPUAddressMode_Undefined', (WGPUAddressMode_ClampToEdge:=1): 'WGPUAddressMode_ClampToEdge', (WGPUAddressMode_Repeat:=2): 'WGPUAddressMode_Repeat', (WGPUAddressMode_MirrorRepeat:=3): 'WGPUAddressMode_MirrorRepeat', (WGPUAddressMode_Force32:=2147483647): 'WGPUAddressMode_Force32'}
WGPUAddressMode: TypeAlias = ctypes.c_uint32
enum_WGPUMipmapFilterMode: dict[int, str] = {(WGPUMipmapFilterMode_Undefined:=0): 'WGPUMipmapFilterMode_Undefined', (WGPUMipmapFilterMode_Nearest:=1): 'WGPUMipmapFilterMode_Nearest', (WGPUMipmapFilterMode_Linear:=2): 'WGPUMipmapFilterMode_Linear', (WGPUMipmapFilterMode_Force32:=2147483647): 'WGPUMipmapFilterMode_Force32'}
WGPUMipmapFilterMode: TypeAlias = ctypes.c_uint32
uint16_t: TypeAlias = ctypes.c_uint16
struct_WGPUSamplerDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('addressModeU', WGPUAddressMode, 24), ('addressModeV', WGPUAddressMode, 28), ('addressModeW', WGPUAddressMode, 32), ('magFilter', WGPUFilterMode, 36), ('minFilter', WGPUFilterMode, 40), ('mipmapFilter', WGPUMipmapFilterMode, 44), ('lodMinClamp', ctypes.c_float, 48), ('lodMaxClamp', ctypes.c_float, 52), ('compare', WGPUCompareFunction, 56), ('maxAnisotropy', uint16_t, 60)])
@c.record
class struct_WGPUShaderModuleDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPUShaderModuleDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPUShaderSourceWGSL(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  code: struct_WGPUStringView
struct_WGPUShaderSourceWGSL.register_fields([('chain', WGPUChainedStruct, 0), ('code', WGPUStringView, 16)])
@c.record
class struct_WGPUSharedBufferMemoryDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPUSharedBufferMemoryDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPUSharedFenceDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPUSharedFenceDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPUSharedTextureMemoryAHardwareBufferProperties(c.Struct):
  SIZE = 88
  chain: struct_WGPUChainedStructOut
  yCbCrInfo: struct_WGPUYCbCrVkDescriptor
struct_WGPUSharedTextureMemoryAHardwareBufferProperties.register_fields([('chain', WGPUChainedStructOut, 0), ('yCbCrInfo', WGPUYCbCrVkDescriptor, 16)])
@c.record
class struct_WGPUSharedTextureMemoryDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPUSharedTextureMemoryDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPUSharedTextureMemoryDmaBufDescriptor(c.Struct):
  SIZE = 56
  chain: struct_WGPUChainedStruct
  size: struct_WGPUExtent3D
  drmFormat: int
  drmModifier: int
  planeCount: int
  planes: c.POINTER[struct_WGPUSharedTextureMemoryDmaBufPlane]
WGPUExtent3D: TypeAlias = struct_WGPUExtent3D
WGPUSharedTextureMemoryDmaBufPlane: TypeAlias = struct_WGPUSharedTextureMemoryDmaBufPlane
struct_WGPUSharedTextureMemoryDmaBufDescriptor.register_fields([('chain', WGPUChainedStruct, 0), ('size', WGPUExtent3D, 16), ('drmFormat', uint32_t, 28), ('drmModifier', uint64_t, 32), ('planeCount', size_t, 40), ('planes', c.POINTER[WGPUSharedTextureMemoryDmaBufPlane], 48)])
@c.record
class struct_WGPUSharedTextureMemoryProperties(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
  usage: int
  size: struct_WGPUExtent3D
  format: int
struct_WGPUSharedTextureMemoryProperties.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0), ('usage', WGPUTextureUsage, 8), ('size', WGPUExtent3D, 16), ('format', WGPUTextureFormat, 28)])
@c.record
class struct_WGPUSupportedLimits(c.Struct):
  SIZE = 168
  nextInChain: c.POINTER[struct_WGPUChainedStructOut]
  limits: struct_WGPULimits
struct_WGPUSupportedLimits.register_fields([('nextInChain', c.POINTER[WGPUChainedStructOut], 0), ('limits', WGPULimits, 8)])
@c.record
class struct_WGPUSurfaceDescriptor(c.Struct):
  SIZE = 24
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
struct_WGPUSurfaceDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8)])
@c.record
class struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten(c.Struct):
  SIZE = 32
  chain: struct_WGPUChainedStruct
  selector: struct_WGPUStringView
struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten.register_fields([('chain', WGPUChainedStruct, 0), ('selector', WGPUStringView, 16)])
@c.record
class struct_WGPUTextureDescriptor(c.Struct):
  SIZE = 80
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  usage: int
  dimension: int
  size: struct_WGPUExtent3D
  format: int
  mipLevelCount: int
  sampleCount: int
  viewFormatCount: int
  viewFormats: c.POINTER[ctypes.c_uint32]
enum_WGPUTextureDimension: dict[int, str] = {(WGPUTextureDimension_Undefined:=0): 'WGPUTextureDimension_Undefined', (WGPUTextureDimension_1D:=1): 'WGPUTextureDimension_1D', (WGPUTextureDimension_2D:=2): 'WGPUTextureDimension_2D', (WGPUTextureDimension_3D:=3): 'WGPUTextureDimension_3D', (WGPUTextureDimension_Force32:=2147483647): 'WGPUTextureDimension_Force32'}
WGPUTextureDimension: TypeAlias = ctypes.c_uint32
struct_WGPUTextureDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('usage', WGPUTextureUsage, 24), ('dimension', WGPUTextureDimension, 32), ('size', WGPUExtent3D, 36), ('format', WGPUTextureFormat, 48), ('mipLevelCount', uint32_t, 52), ('sampleCount', uint32_t, 56), ('viewFormatCount', size_t, 64), ('viewFormats', c.POINTER[WGPUTextureFormat], 72)])
@c.record
class struct_WGPUTextureViewDescriptor(c.Struct):
  SIZE = 64
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  format: int
  dimension: int
  baseMipLevel: int
  mipLevelCount: int
  baseArrayLayer: int
  arrayLayerCount: int
  aspect: int
  usage: int
struct_WGPUTextureViewDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('format', WGPUTextureFormat, 24), ('dimension', WGPUTextureViewDimension, 28), ('baseMipLevel', uint32_t, 32), ('mipLevelCount', uint32_t, 36), ('baseArrayLayer', uint32_t, 40), ('arrayLayerCount', uint32_t, 44), ('aspect', WGPUTextureAspect, 48), ('usage', WGPUTextureUsage, 56)])
@c.record
class struct_WGPUVertexBufferLayout(c.Struct):
  SIZE = 32
  arrayStride: int
  stepMode: int
  attributeCount: int
  attributes: c.POINTER[struct_WGPUVertexAttribute]
enum_WGPUVertexStepMode: dict[int, str] = {(WGPUVertexStepMode_Undefined:=0): 'WGPUVertexStepMode_Undefined', (WGPUVertexStepMode_Vertex:=1): 'WGPUVertexStepMode_Vertex', (WGPUVertexStepMode_Instance:=2): 'WGPUVertexStepMode_Instance', (WGPUVertexStepMode_Force32:=2147483647): 'WGPUVertexStepMode_Force32'}
WGPUVertexStepMode: TypeAlias = ctypes.c_uint32
WGPUVertexAttribute: TypeAlias = struct_WGPUVertexAttribute
struct_WGPUVertexBufferLayout.register_fields([('arrayStride', uint64_t, 0), ('stepMode', WGPUVertexStepMode, 8), ('attributeCount', size_t, 16), ('attributes', c.POINTER[WGPUVertexAttribute], 24)])
@c.record
class struct_WGPUBindGroupLayoutDescriptor(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  entryCount: int
  entries: c.POINTER[struct_WGPUBindGroupLayoutEntry]
WGPUBindGroupLayoutEntry: TypeAlias = struct_WGPUBindGroupLayoutEntry
struct_WGPUBindGroupLayoutDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('entryCount', size_t, 24), ('entries', c.POINTER[WGPUBindGroupLayoutEntry], 32)])
@c.record
class struct_WGPUColorTargetState(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  format: int
  blend: c.POINTER[struct_WGPUBlendState]
  writeMask: int
WGPUBlendState: TypeAlias = struct_WGPUBlendState
WGPUColorWriteMask: TypeAlias = ctypes.c_uint64
struct_WGPUColorTargetState.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('format', WGPUTextureFormat, 8), ('blend', c.POINTER[WGPUBlendState], 16), ('writeMask', WGPUColorWriteMask, 24)])
@c.record
class struct_WGPUComputeState(c.Struct):
  SIZE = 48
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  module: c.POINTER[struct_WGPUShaderModuleImpl]
  entryPoint: struct_WGPUStringView
  constantCount: int
  constants: c.POINTER[struct_WGPUConstantEntry]
WGPUConstantEntry: TypeAlias = struct_WGPUConstantEntry
struct_WGPUComputeState.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('module', WGPUShaderModule, 8), ('entryPoint', WGPUStringView, 16), ('constantCount', size_t, 32), ('constants', c.POINTER[WGPUConstantEntry], 40)])
@c.record
class struct_WGPUDeviceDescriptor(c.Struct):
  SIZE = 144
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  requiredFeatureCount: int
  requiredFeatures: c.POINTER[ctypes.c_uint32]
  requiredLimits: c.POINTER[struct_WGPURequiredLimits]
  defaultQueue: struct_WGPUQueueDescriptor
  deviceLostCallbackInfo2: struct_WGPUDeviceLostCallbackInfo2
  uncapturedErrorCallbackInfo2: struct_WGPUUncapturedErrorCallbackInfo2
WGPURequiredLimits: TypeAlias = struct_WGPURequiredLimits
WGPUQueueDescriptor: TypeAlias = struct_WGPUQueueDescriptor
@c.record
class struct_WGPUDeviceLostCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
WGPUDeviceLostCallbackInfo2: TypeAlias = struct_WGPUDeviceLostCallbackInfo2
WGPUDeviceLostCallback2: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
struct_WGPUDeviceLostCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUDeviceLostCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
@c.record
class struct_WGPUUncapturedErrorCallbackInfo2(c.Struct):
  SIZE = 32
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  callback: c.CFUNCTYPE[None, [c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
WGPUUncapturedErrorCallbackInfo2: TypeAlias = struct_WGPUUncapturedErrorCallbackInfo2
WGPUUncapturedErrorCallback: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
struct_WGPUUncapturedErrorCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('callback', WGPUUncapturedErrorCallback, 8), ('userdata1', ctypes.c_void_p, 16), ('userdata2', ctypes.c_void_p, 24)])
struct_WGPUDeviceDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('requiredFeatureCount', size_t, 24), ('requiredFeatures', c.POINTER[WGPUFeatureName], 32), ('requiredLimits', c.POINTER[WGPURequiredLimits], 40), ('defaultQueue', WGPUQueueDescriptor, 48), ('deviceLostCallbackInfo2', WGPUDeviceLostCallbackInfo2, 72), ('uncapturedErrorCallbackInfo2', WGPUUncapturedErrorCallbackInfo2, 112)])
@c.record
class struct_WGPURenderPassDescriptor(c.Struct):
  SIZE = 64
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  colorAttachmentCount: int
  colorAttachments: c.POINTER[struct_WGPURenderPassColorAttachment]
  depthStencilAttachment: c.POINTER[struct_WGPURenderPassDepthStencilAttachment]
  occlusionQuerySet: c.POINTER[struct_WGPUQuerySetImpl]
  timestampWrites: c.POINTER[struct_WGPURenderPassTimestampWrites]
WGPURenderPassColorAttachment: TypeAlias = struct_WGPURenderPassColorAttachment
WGPURenderPassDepthStencilAttachment: TypeAlias = struct_WGPURenderPassDepthStencilAttachment
WGPURenderPassTimestampWrites: TypeAlias = struct_WGPURenderPassTimestampWrites
struct_WGPURenderPassDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('colorAttachmentCount', size_t, 24), ('colorAttachments', c.POINTER[WGPURenderPassColorAttachment], 32), ('depthStencilAttachment', c.POINTER[WGPURenderPassDepthStencilAttachment], 40), ('occlusionQuerySet', WGPUQuerySet, 48), ('timestampWrites', c.POINTER[WGPURenderPassTimestampWrites], 56)])
@c.record
class struct_WGPURenderPassPixelLocalStorage(c.Struct):
  SIZE = 40
  chain: struct_WGPUChainedStruct
  totalPixelLocalStorageSize: int
  storageAttachmentCount: int
  storageAttachments: c.POINTER[struct_WGPURenderPassStorageAttachment]
WGPURenderPassStorageAttachment: TypeAlias = struct_WGPURenderPassStorageAttachment
struct_WGPURenderPassPixelLocalStorage.register_fields([('chain', WGPUChainedStruct, 0), ('totalPixelLocalStorageSize', uint64_t, 16), ('storageAttachmentCount', size_t, 24), ('storageAttachments', c.POINTER[WGPURenderPassStorageAttachment], 32)])
@c.record
class struct_WGPUVertexState(c.Struct):
  SIZE = 64
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  module: c.POINTER[struct_WGPUShaderModuleImpl]
  entryPoint: struct_WGPUStringView
  constantCount: int
  constants: c.POINTER[struct_WGPUConstantEntry]
  bufferCount: int
  buffers: c.POINTER[struct_WGPUVertexBufferLayout]
WGPUVertexBufferLayout: TypeAlias = struct_WGPUVertexBufferLayout
struct_WGPUVertexState.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('module', WGPUShaderModule, 8), ('entryPoint', WGPUStringView, 16), ('constantCount', size_t, 32), ('constants', c.POINTER[WGPUConstantEntry], 40), ('bufferCount', size_t, 48), ('buffers', c.POINTER[WGPUVertexBufferLayout], 56)])
@c.record
class struct_WGPUComputePipelineDescriptor(c.Struct):
  SIZE = 80
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  layout: c.POINTER[struct_WGPUPipelineLayoutImpl]
  compute: struct_WGPUComputeState
WGPUComputeState: TypeAlias = struct_WGPUComputeState
struct_WGPUComputePipelineDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('layout', WGPUPipelineLayout, 24), ('compute', WGPUComputeState, 32)])
@c.record
class struct_WGPUFragmentState(c.Struct):
  SIZE = 64
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  module: c.POINTER[struct_WGPUShaderModuleImpl]
  entryPoint: struct_WGPUStringView
  constantCount: int
  constants: c.POINTER[struct_WGPUConstantEntry]
  targetCount: int
  targets: c.POINTER[struct_WGPUColorTargetState]
WGPUColorTargetState: TypeAlias = struct_WGPUColorTargetState
struct_WGPUFragmentState.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('module', WGPUShaderModule, 8), ('entryPoint', WGPUStringView, 16), ('constantCount', size_t, 32), ('constants', c.POINTER[WGPUConstantEntry], 40), ('targetCount', size_t, 48), ('targets', c.POINTER[WGPUColorTargetState], 56)])
@c.record
class struct_WGPURenderPipelineDescriptor(c.Struct):
  SIZE = 168
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  label: struct_WGPUStringView
  layout: c.POINTER[struct_WGPUPipelineLayoutImpl]
  vertex: struct_WGPUVertexState
  primitive: struct_WGPUPrimitiveState
  depthStencil: c.POINTER[struct_WGPUDepthStencilState]
  multisample: struct_WGPUMultisampleState
  fragment: c.POINTER[struct_WGPUFragmentState]
WGPUVertexState: TypeAlias = struct_WGPUVertexState
WGPUPrimitiveState: TypeAlias = struct_WGPUPrimitiveState
WGPUDepthStencilState: TypeAlias = struct_WGPUDepthStencilState
WGPUMultisampleState: TypeAlias = struct_WGPUMultisampleState
WGPUFragmentState: TypeAlias = struct_WGPUFragmentState
struct_WGPURenderPipelineDescriptor.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('label', WGPUStringView, 8), ('layout', WGPUPipelineLayout, 24), ('vertex', WGPUVertexState, 32), ('primitive', WGPUPrimitiveState, 96), ('depthStencil', c.POINTER[WGPUDepthStencilState], 128), ('multisample', WGPUMultisampleState, 136), ('fragment', c.POINTER[WGPUFragmentState], 160)])
enum_WGPUWGSLFeatureName: dict[int, str] = {(WGPUWGSLFeatureName_ReadonlyAndReadwriteStorageTextures:=1): 'WGPUWGSLFeatureName_ReadonlyAndReadwriteStorageTextures', (WGPUWGSLFeatureName_Packed4x8IntegerDotProduct:=2): 'WGPUWGSLFeatureName_Packed4x8IntegerDotProduct', (WGPUWGSLFeatureName_UnrestrictedPointerParameters:=3): 'WGPUWGSLFeatureName_UnrestrictedPointerParameters', (WGPUWGSLFeatureName_PointerCompositeAccess:=4): 'WGPUWGSLFeatureName_PointerCompositeAccess', (WGPUWGSLFeatureName_ChromiumTestingUnimplemented:=327680): 'WGPUWGSLFeatureName_ChromiumTestingUnimplemented', (WGPUWGSLFeatureName_ChromiumTestingUnsafeExperimental:=327681): 'WGPUWGSLFeatureName_ChromiumTestingUnsafeExperimental', (WGPUWGSLFeatureName_ChromiumTestingExperimental:=327682): 'WGPUWGSLFeatureName_ChromiumTestingExperimental', (WGPUWGSLFeatureName_ChromiumTestingShippedWithKillswitch:=327683): 'WGPUWGSLFeatureName_ChromiumTestingShippedWithKillswitch', (WGPUWGSLFeatureName_ChromiumTestingShipped:=327684): 'WGPUWGSLFeatureName_ChromiumTestingShipped', (WGPUWGSLFeatureName_Force32:=2147483647): 'WGPUWGSLFeatureName_Force32'}
WGPUWGSLFeatureName: TypeAlias = ctypes.c_uint32
WGPUBufferMapAsyncStatus: TypeAlias = ctypes.c_uint32
enum_WGPUBufferMapState: dict[int, str] = {(WGPUBufferMapState_Unmapped:=1): 'WGPUBufferMapState_Unmapped', (WGPUBufferMapState_Pending:=2): 'WGPUBufferMapState_Pending', (WGPUBufferMapState_Mapped:=3): 'WGPUBufferMapState_Mapped', (WGPUBufferMapState_Force32:=2147483647): 'WGPUBufferMapState_Force32'}
WGPUBufferMapState: TypeAlias = ctypes.c_uint32
WGPUCompilationInfoRequestStatus: TypeAlias = ctypes.c_uint32
WGPUCreatePipelineAsyncStatus: TypeAlias = ctypes.c_uint32
WGPUDeviceLostReason: TypeAlias = ctypes.c_uint32
enum_WGPUErrorFilter: dict[int, str] = {(WGPUErrorFilter_Validation:=1): 'WGPUErrorFilter_Validation', (WGPUErrorFilter_OutOfMemory:=2): 'WGPUErrorFilter_OutOfMemory', (WGPUErrorFilter_Internal:=3): 'WGPUErrorFilter_Internal', (WGPUErrorFilter_Force32:=2147483647): 'WGPUErrorFilter_Force32'}
WGPUErrorFilter: TypeAlias = ctypes.c_uint32
WGPUErrorType: TypeAlias = ctypes.c_uint32
enum_WGPULoggingType: dict[int, str] = {(WGPULoggingType_Verbose:=1): 'WGPULoggingType_Verbose', (WGPULoggingType_Info:=2): 'WGPULoggingType_Info', (WGPULoggingType_Warning:=3): 'WGPULoggingType_Warning', (WGPULoggingType_Error:=4): 'WGPULoggingType_Error', (WGPULoggingType_Force32:=2147483647): 'WGPULoggingType_Force32'}
WGPULoggingType: TypeAlias = ctypes.c_uint32
enum_WGPUMapAsyncStatus: dict[int, str] = {(WGPUMapAsyncStatus_Success:=1): 'WGPUMapAsyncStatus_Success', (WGPUMapAsyncStatus_InstanceDropped:=2): 'WGPUMapAsyncStatus_InstanceDropped', (WGPUMapAsyncStatus_Error:=3): 'WGPUMapAsyncStatus_Error', (WGPUMapAsyncStatus_Aborted:=4): 'WGPUMapAsyncStatus_Aborted', (WGPUMapAsyncStatus_Unknown:=5): 'WGPUMapAsyncStatus_Unknown', (WGPUMapAsyncStatus_Force32:=2147483647): 'WGPUMapAsyncStatus_Force32'}
WGPUMapAsyncStatus: TypeAlias = ctypes.c_uint32
WGPUPopErrorScopeStatus: TypeAlias = ctypes.c_uint32
WGPUQueueWorkDoneStatus: TypeAlias = ctypes.c_uint32
WGPURequestAdapterStatus: TypeAlias = ctypes.c_uint32
WGPURequestDeviceStatus: TypeAlias = ctypes.c_uint32
enum_WGPUStatus: dict[int, str] = {(WGPUStatus_Success:=1): 'WGPUStatus_Success', (WGPUStatus_Error:=2): 'WGPUStatus_Error', (WGPUStatus_Force32:=2147483647): 'WGPUStatus_Force32'}
WGPUStatus: TypeAlias = ctypes.c_uint32
enum_WGPUWaitStatus: dict[int, str] = {(WGPUWaitStatus_Success:=1): 'WGPUWaitStatus_Success', (WGPUWaitStatus_TimedOut:=2): 'WGPUWaitStatus_TimedOut', (WGPUWaitStatus_UnsupportedTimeout:=3): 'WGPUWaitStatus_UnsupportedTimeout', (WGPUWaitStatus_UnsupportedCount:=4): 'WGPUWaitStatus_UnsupportedCount', (WGPUWaitStatus_UnsupportedMixedSources:=5): 'WGPUWaitStatus_UnsupportedMixedSources', (WGPUWaitStatus_Unknown:=6): 'WGPUWaitStatus_Unknown', (WGPUWaitStatus_Force32:=2147483647): 'WGPUWaitStatus_Force32'}
WGPUWaitStatus: TypeAlias = ctypes.c_uint32
WGPUMapMode: TypeAlias = ctypes.c_uint64
WGPUDeviceLostCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
WGPULoggingCallback: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]]
WGPUProc: TypeAlias = c.CFUNCTYPE[None, []]
WGPUBufferMapCallback2: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
WGPUCompilationInfoCallback2: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUCompilationInfo], ctypes.c_void_p, ctypes.c_void_p]]
WGPUCreateComputePipelineAsyncCallback2: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
WGPUCreateRenderPipelineAsyncCallback2: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
WGPUPopErrorScopeCallback2: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
WGPUQueueWorkDoneCallback2: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]]
WGPURequestAdapterCallback2: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUAdapterImpl], struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
WGPURequestDeviceCallback2: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
@c.record
class struct_WGPUBufferMapCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
struct_WGPUBufferMapCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUBufferMapCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
WGPUBufferMapCallbackInfo2: TypeAlias = struct_WGPUBufferMapCallbackInfo2
@c.record
class struct_WGPUCompilationInfoCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUCompilationInfo], ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
struct_WGPUCompilationInfoCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUCompilationInfoCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
WGPUCompilationInfoCallbackInfo2: TypeAlias = struct_WGPUCompilationInfoCallbackInfo2
@c.record
class struct_WGPUCreateComputePipelineAsyncCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
struct_WGPUCreateComputePipelineAsyncCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUCreateComputePipelineAsyncCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
WGPUCreateComputePipelineAsyncCallbackInfo2: TypeAlias = struct_WGPUCreateComputePipelineAsyncCallbackInfo2
@c.record
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
struct_WGPUCreateRenderPipelineAsyncCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUCreateRenderPipelineAsyncCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
WGPUCreateRenderPipelineAsyncCallbackInfo2: TypeAlias = struct_WGPUCreateRenderPipelineAsyncCallbackInfo2
@c.record
class struct_WGPUPopErrorScopeCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
struct_WGPUPopErrorScopeCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUPopErrorScopeCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
WGPUPopErrorScopeCallbackInfo2: TypeAlias = struct_WGPUPopErrorScopeCallbackInfo2
@c.record
class struct_WGPUQueueWorkDoneCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
struct_WGPUQueueWorkDoneCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPUQueueWorkDoneCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
WGPUQueueWorkDoneCallbackInfo2: TypeAlias = struct_WGPUQueueWorkDoneCallbackInfo2
@c.record
class struct_WGPURequestAdapterCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUAdapterImpl], struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
struct_WGPURequestAdapterCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPURequestAdapterCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
WGPURequestAdapterCallbackInfo2: TypeAlias = struct_WGPURequestAdapterCallbackInfo2
@c.record
class struct_WGPURequestDeviceCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: c.POINTER[struct_WGPUChainedStruct]
  mode: int
  callback: c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p]]
  userdata1: ctypes.c_void_p
  userdata2: ctypes.c_void_p
struct_WGPURequestDeviceCallbackInfo2.register_fields([('nextInChain', c.POINTER[WGPUChainedStruct], 0), ('mode', WGPUCallbackMode, 8), ('callback', WGPURequestDeviceCallback2, 16), ('userdata1', ctypes.c_void_p, 24), ('userdata2', ctypes.c_void_p, 32)])
WGPURequestDeviceCallbackInfo2: TypeAlias = struct_WGPURequestDeviceCallbackInfo2
WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER: TypeAlias = struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER
WGPUAdapterPropertiesD3D: TypeAlias = struct_WGPUAdapterPropertiesD3D
WGPUAdapterPropertiesSubgroups: TypeAlias = struct_WGPUAdapterPropertiesSubgroups
WGPUAdapterPropertiesVk: TypeAlias = struct_WGPUAdapterPropertiesVk
WGPUBufferHostMappedPointer: TypeAlias = struct_WGPUBufferHostMappedPointer
WGPUBufferMapCallbackInfo: TypeAlias = struct_WGPUBufferMapCallbackInfo
WGPUColorTargetStateExpandResolveTextureDawn: TypeAlias = struct_WGPUColorTargetStateExpandResolveTextureDawn
WGPUCompilationInfoCallbackInfo: TypeAlias = struct_WGPUCompilationInfoCallbackInfo
WGPUCopyTextureForBrowserOptions: TypeAlias = struct_WGPUCopyTextureForBrowserOptions
WGPUCreateComputePipelineAsyncCallbackInfo: TypeAlias = struct_WGPUCreateComputePipelineAsyncCallbackInfo
WGPUCreateRenderPipelineAsyncCallbackInfo: TypeAlias = struct_WGPUCreateRenderPipelineAsyncCallbackInfo
WGPUDawnWGSLBlocklist: TypeAlias = struct_WGPUDawnWGSLBlocklist
WGPUDawnAdapterPropertiesPowerPreference: TypeAlias = struct_WGPUDawnAdapterPropertiesPowerPreference
WGPUDawnBufferDescriptorErrorInfoFromWireClient: TypeAlias = struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient
WGPUDawnEncoderInternalUsageDescriptor: TypeAlias = struct_WGPUDawnEncoderInternalUsageDescriptor
WGPUDawnExperimentalImmediateDataLimits: TypeAlias = struct_WGPUDawnExperimentalImmediateDataLimits
WGPUDawnExperimentalSubgroupLimits: TypeAlias = struct_WGPUDawnExperimentalSubgroupLimits
WGPUDawnRenderPassColorAttachmentRenderToSingleSampled: TypeAlias = struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled
WGPUDawnShaderModuleSPIRVOptionsDescriptor: TypeAlias = struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor
WGPUDawnTexelCopyBufferRowAlignmentLimits: TypeAlias = struct_WGPUDawnTexelCopyBufferRowAlignmentLimits
WGPUDawnTextureInternalUsageDescriptor: TypeAlias = struct_WGPUDawnTextureInternalUsageDescriptor
WGPUDawnTogglesDescriptor: TypeAlias = struct_WGPUDawnTogglesDescriptor
WGPUDawnWireWGSLControl: TypeAlias = struct_WGPUDawnWireWGSLControl
WGPUDeviceLostCallbackInfo: TypeAlias = struct_WGPUDeviceLostCallbackInfo
WGPUExternalTextureBindingEntry: TypeAlias = struct_WGPUExternalTextureBindingEntry
WGPUExternalTextureBindingLayout: TypeAlias = struct_WGPUExternalTextureBindingLayout
WGPUFormatCapabilities: TypeAlias = struct_WGPUFormatCapabilities
WGPUPopErrorScopeCallbackInfo: TypeAlias = struct_WGPUPopErrorScopeCallbackInfo
WGPUQueueWorkDoneCallbackInfo: TypeAlias = struct_WGPUQueueWorkDoneCallbackInfo
WGPURenderPassDescriptorExpandResolveRect: TypeAlias = struct_WGPURenderPassDescriptorExpandResolveRect
WGPURenderPassMaxDrawCount: TypeAlias = struct_WGPURenderPassMaxDrawCount
WGPURequestAdapterCallbackInfo: TypeAlias = struct_WGPURequestAdapterCallbackInfo
WGPURequestAdapterOptions: TypeAlias = struct_WGPURequestAdapterOptions
WGPURequestDeviceCallbackInfo: TypeAlias = struct_WGPURequestDeviceCallbackInfo
WGPUShaderModuleCompilationOptions: TypeAlias = struct_WGPUShaderModuleCompilationOptions
WGPUShaderSourceSPIRV: TypeAlias = struct_WGPUShaderSourceSPIRV
WGPUSharedBufferMemoryBeginAccessDescriptor: TypeAlias = struct_WGPUSharedBufferMemoryBeginAccessDescriptor
WGPUSharedBufferMemoryEndAccessState: TypeAlias = struct_WGPUSharedBufferMemoryEndAccessState
WGPUSharedBufferMemoryProperties: TypeAlias = struct_WGPUSharedBufferMemoryProperties
WGPUSharedFenceDXGISharedHandleDescriptor: TypeAlias = struct_WGPUSharedFenceDXGISharedHandleDescriptor
WGPUSharedFenceDXGISharedHandleExportInfo: TypeAlias = struct_WGPUSharedFenceDXGISharedHandleExportInfo
WGPUSharedFenceMTLSharedEventDescriptor: TypeAlias = struct_WGPUSharedFenceMTLSharedEventDescriptor
WGPUSharedFenceMTLSharedEventExportInfo: TypeAlias = struct_WGPUSharedFenceMTLSharedEventExportInfo
WGPUSharedFenceExportInfo: TypeAlias = struct_WGPUSharedFenceExportInfo
WGPUSharedFenceSyncFDDescriptor: TypeAlias = struct_WGPUSharedFenceSyncFDDescriptor
WGPUSharedFenceSyncFDExportInfo: TypeAlias = struct_WGPUSharedFenceSyncFDExportInfo
WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor: TypeAlias = struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor
WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo: TypeAlias = struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo
WGPUSharedFenceVkSemaphoreZirconHandleDescriptor: TypeAlias = struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor
WGPUSharedFenceVkSemaphoreZirconHandleExportInfo: TypeAlias = struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo
WGPUSharedTextureMemoryD3DSwapchainBeginState: TypeAlias = struct_WGPUSharedTextureMemoryD3DSwapchainBeginState
WGPUSharedTextureMemoryDXGISharedHandleDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor
WGPUSharedTextureMemoryEGLImageDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryEGLImageDescriptor
WGPUSharedTextureMemoryIOSurfaceDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryIOSurfaceDescriptor
WGPUSharedTextureMemoryAHardwareBufferDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor
WGPUSharedTextureMemoryBeginAccessDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryBeginAccessDescriptor
WGPUSharedTextureMemoryEndAccessState: TypeAlias = struct_WGPUSharedTextureMemoryEndAccessState
WGPUSharedTextureMemoryOpaqueFDDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryOpaqueFDDescriptor
WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor
WGPUSharedTextureMemoryVkImageLayoutBeginState: TypeAlias = struct_WGPUSharedTextureMemoryVkImageLayoutBeginState
WGPUSharedTextureMemoryVkImageLayoutEndState: TypeAlias = struct_WGPUSharedTextureMemoryVkImageLayoutEndState
WGPUSharedTextureMemoryZirconHandleDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryZirconHandleDescriptor
WGPUStaticSamplerBindingLayout: TypeAlias = struct_WGPUStaticSamplerBindingLayout
WGPUSupportedFeatures: TypeAlias = struct_WGPUSupportedFeatures
WGPUSurfaceCapabilities: TypeAlias = struct_WGPUSurfaceCapabilities
WGPUSurfaceConfiguration: TypeAlias = struct_WGPUSurfaceConfiguration
WGPUSurfaceDescriptorFromWindowsCoreWindow: TypeAlias = struct_WGPUSurfaceDescriptorFromWindowsCoreWindow
WGPUSurfaceDescriptorFromWindowsSwapChainPanel: TypeAlias = struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel
WGPUSurfaceSourceXCBWindow: TypeAlias = struct_WGPUSurfaceSourceXCBWindow
WGPUSurfaceSourceAndroidNativeWindow: TypeAlias = struct_WGPUSurfaceSourceAndroidNativeWindow
WGPUSurfaceSourceMetalLayer: TypeAlias = struct_WGPUSurfaceSourceMetalLayer
WGPUSurfaceSourceWaylandSurface: TypeAlias = struct_WGPUSurfaceSourceWaylandSurface
WGPUSurfaceSourceWindowsHWND: TypeAlias = struct_WGPUSurfaceSourceWindowsHWND
WGPUSurfaceSourceXlibWindow: TypeAlias = struct_WGPUSurfaceSourceXlibWindow
WGPUSurfaceTexture: TypeAlias = struct_WGPUSurfaceTexture
WGPUTextureBindingViewDimensionDescriptor: TypeAlias = struct_WGPUTextureBindingViewDimensionDescriptor
WGPUUncapturedErrorCallbackInfo: TypeAlias = struct_WGPUUncapturedErrorCallbackInfo
WGPUAHardwareBufferProperties: TypeAlias = struct_WGPUAHardwareBufferProperties
WGPUAdapterInfo: TypeAlias = struct_WGPUAdapterInfo
WGPUAdapterPropertiesMemoryHeaps: TypeAlias = struct_WGPUAdapterPropertiesMemoryHeaps
WGPUBindGroupDescriptor: TypeAlias = struct_WGPUBindGroupDescriptor
WGPUBufferDescriptor: TypeAlias = struct_WGPUBufferDescriptor
WGPUCommandBufferDescriptor: TypeAlias = struct_WGPUCommandBufferDescriptor
WGPUCommandEncoderDescriptor: TypeAlias = struct_WGPUCommandEncoderDescriptor
WGPUComputePassDescriptor: TypeAlias = struct_WGPUComputePassDescriptor
WGPUDawnCacheDeviceDescriptor: TypeAlias = struct_WGPUDawnCacheDeviceDescriptor
WGPUDrmFormatCapabilities: TypeAlias = struct_WGPUDrmFormatCapabilities
WGPUExternalTextureDescriptor: TypeAlias = struct_WGPUExternalTextureDescriptor
WGPUFutureWaitInfo: TypeAlias = struct_WGPUFutureWaitInfo
WGPUImageCopyBuffer: TypeAlias = struct_WGPUImageCopyBuffer
WGPUImageCopyExternalTexture: TypeAlias = struct_WGPUImageCopyExternalTexture
WGPUImageCopyTexture: TypeAlias = struct_WGPUImageCopyTexture
WGPUInstanceDescriptor: TypeAlias = struct_WGPUInstanceDescriptor
WGPUPipelineLayoutDescriptor: TypeAlias = struct_WGPUPipelineLayoutDescriptor
WGPUPipelineLayoutPixelLocalStorage: TypeAlias = struct_WGPUPipelineLayoutPixelLocalStorage
WGPUQuerySetDescriptor: TypeAlias = struct_WGPUQuerySetDescriptor
WGPURenderBundleDescriptor: TypeAlias = struct_WGPURenderBundleDescriptor
WGPURenderBundleEncoderDescriptor: TypeAlias = struct_WGPURenderBundleEncoderDescriptor
WGPUSamplerDescriptor: TypeAlias = struct_WGPUSamplerDescriptor
WGPUShaderModuleDescriptor: TypeAlias = struct_WGPUShaderModuleDescriptor
WGPUShaderSourceWGSL: TypeAlias = struct_WGPUShaderSourceWGSL
WGPUSharedBufferMemoryDescriptor: TypeAlias = struct_WGPUSharedBufferMemoryDescriptor
WGPUSharedFenceDescriptor: TypeAlias = struct_WGPUSharedFenceDescriptor
WGPUSharedTextureMemoryAHardwareBufferProperties: TypeAlias = struct_WGPUSharedTextureMemoryAHardwareBufferProperties
WGPUSharedTextureMemoryDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryDescriptor
WGPUSharedTextureMemoryDmaBufDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryDmaBufDescriptor
WGPUSharedTextureMemoryProperties: TypeAlias = struct_WGPUSharedTextureMemoryProperties
WGPUSupportedLimits: TypeAlias = struct_WGPUSupportedLimits
WGPUSurfaceDescriptor: TypeAlias = struct_WGPUSurfaceDescriptor
WGPUSurfaceSourceCanvasHTMLSelector_Emscripten: TypeAlias = struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten
WGPUTextureDescriptor: TypeAlias = struct_WGPUTextureDescriptor
WGPUTextureViewDescriptor: TypeAlias = struct_WGPUTextureViewDescriptor
WGPUBindGroupLayoutDescriptor: TypeAlias = struct_WGPUBindGroupLayoutDescriptor
WGPUCompilationInfo: TypeAlias = struct_WGPUCompilationInfo
WGPUDeviceDescriptor: TypeAlias = struct_WGPUDeviceDescriptor
WGPURenderPassDescriptor: TypeAlias = struct_WGPURenderPassDescriptor
WGPURenderPassPixelLocalStorage: TypeAlias = struct_WGPURenderPassPixelLocalStorage
WGPUComputePipelineDescriptor: TypeAlias = struct_WGPUComputePipelineDescriptor
WGPURenderPipelineDescriptor: TypeAlias = struct_WGPURenderPipelineDescriptor
WGPURenderPassDescriptorMaxDrawCount: TypeAlias = struct_WGPURenderPassMaxDrawCount
WGPUShaderModuleSPIRVDescriptor: TypeAlias = struct_WGPUShaderSourceSPIRV
WGPUShaderModuleWGSLDescriptor: TypeAlias = struct_WGPUShaderSourceWGSL
WGPUSurfaceDescriptorFromAndroidNativeWindow: TypeAlias = struct_WGPUSurfaceSourceAndroidNativeWindow
WGPUSurfaceDescriptorFromCanvasHTMLSelector: TypeAlias = struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten
WGPUSurfaceDescriptorFromMetalLayer: TypeAlias = struct_WGPUSurfaceSourceMetalLayer
WGPUSurfaceDescriptorFromWaylandSurface: TypeAlias = struct_WGPUSurfaceSourceWaylandSurface
WGPUSurfaceDescriptorFromWindowsHWND: TypeAlias = struct_WGPUSurfaceSourceWindowsHWND
WGPUSurfaceDescriptorFromXcbWindow: TypeAlias = struct_WGPUSurfaceSourceXCBWindow
WGPUSurfaceDescriptorFromXlibWindow: TypeAlias = struct_WGPUSurfaceSourceXlibWindow
WGPUProcAdapterInfoFreeMembers: TypeAlias = c.CFUNCTYPE[None, [struct_WGPUAdapterInfo]]
WGPUProcAdapterPropertiesMemoryHeapsFreeMembers: TypeAlias = c.CFUNCTYPE[None, [struct_WGPUAdapterPropertiesMemoryHeaps]]
WGPUProcCreateInstance: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUInstanceImpl], [c.POINTER[struct_WGPUInstanceDescriptor]]]
WGPUProcDrmFormatCapabilitiesFreeMembers: TypeAlias = c.CFUNCTYPE[None, [struct_WGPUDrmFormatCapabilities]]
WGPUProcGetInstanceFeatures: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUInstanceFeatures]]]
WGPUProcGetProcAddress: TypeAlias = c.CFUNCTYPE[c.CFUNCTYPE[None, []], [struct_WGPUStringView]]
WGPUProcSharedBufferMemoryEndAccessStateFreeMembers: TypeAlias = c.CFUNCTYPE[None, [struct_WGPUSharedBufferMemoryEndAccessState]]
WGPUProcSharedTextureMemoryEndAccessStateFreeMembers: TypeAlias = c.CFUNCTYPE[None, [struct_WGPUSharedTextureMemoryEndAccessState]]
WGPUProcSupportedFeaturesFreeMembers: TypeAlias = c.CFUNCTYPE[None, [struct_WGPUSupportedFeatures]]
WGPUProcSurfaceCapabilitiesFreeMembers: TypeAlias = c.CFUNCTYPE[None, [struct_WGPUSurfaceCapabilities]]
WGPUProcAdapterCreateDevice: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUDeviceImpl], [c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceDescriptor]]]
WGPUProcAdapterGetFeatures: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUSupportedFeatures]]]
WGPUProcAdapterGetFormatCapabilities: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUAdapterImpl], ctypes.c_uint32, c.POINTER[struct_WGPUFormatCapabilities]]]
WGPUProcAdapterGetInfo: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUAdapterInfo]]]
WGPUProcAdapterGetInstance: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUInstanceImpl], [c.POINTER[struct_WGPUAdapterImpl]]]
WGPUProcAdapterGetLimits: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUSupportedLimits]]]
WGPUProcAdapterHasFeature: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUAdapterImpl], ctypes.c_uint32]]
WGPUProcAdapterRequestDevice: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceDescriptor], c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView, ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcAdapterRequestDevice2: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceDescriptor], struct_WGPURequestDeviceCallbackInfo2]]
WGPUProcAdapterRequestDeviceF: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceDescriptor], struct_WGPURequestDeviceCallbackInfo]]
WGPUProcAdapterAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUAdapterImpl]]]
WGPUProcAdapterRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUAdapterImpl]]]
WGPUProcBindGroupSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBindGroupImpl], struct_WGPUStringView]]
WGPUProcBindGroupAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBindGroupImpl]]]
WGPUProcBindGroupRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBindGroupImpl]]]
WGPUProcBindGroupLayoutSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBindGroupLayoutImpl], struct_WGPUStringView]]
WGPUProcBindGroupLayoutAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBindGroupLayoutImpl]]]
WGPUProcBindGroupLayoutRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBindGroupLayoutImpl]]]
WGPUProcBufferDestroy: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBufferImpl]]]
WGPUProcBufferGetConstMappedRange: TypeAlias = c.CFUNCTYPE[ctypes.c_void_p, [c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64]]
WGPUProcBufferGetMapState: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUBufferImpl]]]
WGPUProcBufferGetMappedRange: TypeAlias = c.CFUNCTYPE[ctypes.c_void_p, [c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64]]
WGPUProcBufferGetSize: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[struct_WGPUBufferImpl]]]
WGPUProcBufferGetUsage: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[struct_WGPUBufferImpl]]]
WGPUProcBufferMapAsync: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcBufferMapAsync2: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, struct_WGPUBufferMapCallbackInfo2]]
WGPUProcBufferMapAsyncF: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, struct_WGPUBufferMapCallbackInfo]]
WGPUProcBufferSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBufferImpl], struct_WGPUStringView]]
WGPUProcBufferUnmap: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBufferImpl]]]
WGPUProcBufferAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBufferImpl]]]
WGPUProcBufferRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUBufferImpl]]]
WGPUProcCommandBufferSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandBufferImpl], struct_WGPUStringView]]
WGPUProcCommandBufferAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandBufferImpl]]]
WGPUProcCommandBufferRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandBufferImpl]]]
WGPUProcCommandEncoderBeginComputePass: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUComputePassEncoderImpl], [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUComputePassDescriptor]]]
WGPUProcCommandEncoderBeginRenderPass: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPURenderPassEncoderImpl], [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPURenderPassDescriptor]]]
WGPUProcCommandEncoderClearBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64]]
WGPUProcCommandEncoderCopyBufferToBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64]]
WGPUProcCommandEncoderCopyBufferToTexture: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUImageCopyBuffer], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUExtent3D]]]
WGPUProcCommandEncoderCopyTextureToBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUImageCopyBuffer], c.POINTER[struct_WGPUExtent3D]]]
WGPUProcCommandEncoderCopyTextureToTexture: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUExtent3D]]]
WGPUProcCommandEncoderFinish: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUCommandBufferImpl], [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUCommandBufferDescriptor]]]
WGPUProcCommandEncoderInjectValidationError: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], struct_WGPUStringView]]
WGPUProcCommandEncoderInsertDebugMarker: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], struct_WGPUStringView]]
WGPUProcCommandEncoderPopDebugGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl]]]
WGPUProcCommandEncoderPushDebugGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], struct_WGPUStringView]]
WGPUProcCommandEncoderResolveQuerySet: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUQuerySetImpl], ctypes.c_uint32, ctypes.c_uint32, c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64]]
WGPUProcCommandEncoderSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], struct_WGPUStringView]]
WGPUProcCommandEncoderWriteBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, c.POINTER[ctypes.c_ubyte], ctypes.c_uint64]]
WGPUProcCommandEncoderWriteTimestamp: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUQuerySetImpl], ctypes.c_uint32]]
WGPUProcCommandEncoderAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl]]]
WGPUProcCommandEncoderRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUCommandEncoderImpl]]]
WGPUProcComputePassEncoderDispatchWorkgroups: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]]
WGPUProcComputePassEncoderDispatchWorkgroupsIndirect: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64]]
WGPUProcComputePassEncoderEnd: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl]]]
WGPUProcComputePassEncoderInsertDebugMarker: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl], struct_WGPUStringView]]
WGPUProcComputePassEncoderPopDebugGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl]]]
WGPUProcComputePassEncoderPushDebugGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl], struct_WGPUStringView]]
WGPUProcComputePassEncoderSetBindGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl], ctypes.c_uint32, c.POINTER[struct_WGPUBindGroupImpl], ctypes.c_uint64, c.POINTER[ctypes.c_uint32]]]
WGPUProcComputePassEncoderSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl], struct_WGPUStringView]]
WGPUProcComputePassEncoderSetPipeline: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl], c.POINTER[struct_WGPUComputePipelineImpl]]]
WGPUProcComputePassEncoderWriteTimestamp: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl], c.POINTER[struct_WGPUQuerySetImpl], ctypes.c_uint32]]
WGPUProcComputePassEncoderAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl]]]
WGPUProcComputePassEncoderRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePassEncoderImpl]]]
WGPUProcComputePipelineGetBindGroupLayout: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUBindGroupLayoutImpl], [c.POINTER[struct_WGPUComputePipelineImpl], ctypes.c_uint32]]
WGPUProcComputePipelineSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView]]
WGPUProcComputePipelineAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePipelineImpl]]]
WGPUProcComputePipelineRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUComputePipelineImpl]]]
WGPUProcDeviceCreateBindGroup: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUBindGroupImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUBindGroupDescriptor]]]
WGPUProcDeviceCreateBindGroupLayout: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUBindGroupLayoutImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUBindGroupLayoutDescriptor]]]
WGPUProcDeviceCreateBuffer: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUBufferImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUBufferDescriptor]]]
WGPUProcDeviceCreateCommandEncoder: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUCommandEncoderImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUCommandEncoderDescriptor]]]
WGPUProcDeviceCreateComputePipeline: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUComputePipelineImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUComputePipelineDescriptor]]]
WGPUProcDeviceCreateComputePipelineAsync: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUComputePipelineDescriptor], c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView, ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcDeviceCreateComputePipelineAsync2: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUComputePipelineDescriptor], struct_WGPUCreateComputePipelineAsyncCallbackInfo2]]
WGPUProcDeviceCreateComputePipelineAsyncF: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUComputePipelineDescriptor], struct_WGPUCreateComputePipelineAsyncCallbackInfo]]
WGPUProcDeviceCreateErrorBuffer: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUBufferImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUBufferDescriptor]]]
WGPUProcDeviceCreateErrorExternalTexture: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUExternalTextureImpl], [c.POINTER[struct_WGPUDeviceImpl]]]
WGPUProcDeviceCreateErrorShaderModule: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUShaderModuleImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUShaderModuleDescriptor], struct_WGPUStringView]]
WGPUProcDeviceCreateErrorTexture: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUTextureImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUTextureDescriptor]]]
WGPUProcDeviceCreateExternalTexture: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUExternalTextureImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUExternalTextureDescriptor]]]
WGPUProcDeviceCreatePipelineLayout: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUPipelineLayoutImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUPipelineLayoutDescriptor]]]
WGPUProcDeviceCreateQuerySet: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUQuerySetImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUQuerySetDescriptor]]]
WGPUProcDeviceCreateRenderBundleEncoder: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPURenderBundleEncoderImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderBundleEncoderDescriptor]]]
WGPUProcDeviceCreateRenderPipeline: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPURenderPipelineImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderPipelineDescriptor]]]
WGPUProcDeviceCreateRenderPipelineAsync: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderPipelineDescriptor], c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView, ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcDeviceCreateRenderPipelineAsync2: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderPipelineDescriptor], struct_WGPUCreateRenderPipelineAsyncCallbackInfo2]]
WGPUProcDeviceCreateRenderPipelineAsyncF: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderPipelineDescriptor], struct_WGPUCreateRenderPipelineAsyncCallbackInfo]]
WGPUProcDeviceCreateSampler: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUSamplerImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSamplerDescriptor]]]
WGPUProcDeviceCreateShaderModule: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUShaderModuleImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUShaderModuleDescriptor]]]
WGPUProcDeviceCreateTexture: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUTextureImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUTextureDescriptor]]]
WGPUProcDeviceDestroy: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl]]]
WGPUProcDeviceForceLoss: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], ctypes.c_uint32, struct_WGPUStringView]]
WGPUProcDeviceGetAHardwareBufferProperties: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUDeviceImpl], ctypes.c_void_p, c.POINTER[struct_WGPUAHardwareBufferProperties]]]
WGPUProcDeviceGetAdapter: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUAdapterImpl], [c.POINTER[struct_WGPUDeviceImpl]]]
WGPUProcDeviceGetAdapterInfo: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUAdapterInfo]]]
WGPUProcDeviceGetFeatures: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSupportedFeatures]]]
WGPUProcDeviceGetLimits: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSupportedLimits]]]
WGPUProcDeviceGetLostFuture: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUDeviceImpl]]]
WGPUProcDeviceGetQueue: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUQueueImpl], [c.POINTER[struct_WGPUDeviceImpl]]]
WGPUProcDeviceHasFeature: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUDeviceImpl], ctypes.c_uint32]]
WGPUProcDeviceImportSharedBufferMemory: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUSharedBufferMemoryImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSharedBufferMemoryDescriptor]]]
WGPUProcDeviceImportSharedFence: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUSharedFenceImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSharedFenceDescriptor]]]
WGPUProcDeviceImportSharedTextureMemory: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUSharedTextureMemoryImpl], [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSharedTextureMemoryDescriptor]]]
WGPUProcDeviceInjectError: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], ctypes.c_uint32, struct_WGPUStringView]]
WGPUProcDevicePopErrorScope: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcDevicePopErrorScope2: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUDeviceImpl], struct_WGPUPopErrorScopeCallbackInfo2]]
WGPUProcDevicePopErrorScopeF: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUDeviceImpl], struct_WGPUPopErrorScopeCallbackInfo]]
WGPUProcDevicePushErrorScope: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], ctypes.c_uint32]]
WGPUProcDeviceSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView]]
WGPUProcDeviceSetLoggingCallback: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], c.CFUNCTYPE[None, [ctypes.c_uint32, struct_WGPUStringView, ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcDeviceTick: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl]]]
WGPUProcDeviceValidateTextureDescriptor: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUTextureDescriptor]]]
WGPUProcDeviceAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl]]]
WGPUProcDeviceRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUDeviceImpl]]]
WGPUProcExternalTextureDestroy: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUExternalTextureImpl]]]
WGPUProcExternalTextureExpire: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUExternalTextureImpl]]]
WGPUProcExternalTextureRefresh: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUExternalTextureImpl]]]
WGPUProcExternalTextureSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUExternalTextureImpl], struct_WGPUStringView]]
WGPUProcExternalTextureAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUExternalTextureImpl]]]
WGPUProcExternalTextureRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUExternalTextureImpl]]]
WGPUProcInstanceCreateSurface: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUSurfaceImpl], [c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPUSurfaceDescriptor]]]
WGPUProcInstanceEnumerateWGSLLanguageFeatures: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[struct_WGPUInstanceImpl], c.POINTER[ctypes.c_uint32]]]
WGPUProcInstanceHasWGSLLanguageFeature: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUInstanceImpl], ctypes.c_uint32]]
WGPUProcInstanceProcessEvents: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUInstanceImpl]]]
WGPUProcInstanceRequestAdapter: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPURequestAdapterOptions], c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUAdapterImpl], struct_WGPUStringView, ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcInstanceRequestAdapter2: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPURequestAdapterOptions], struct_WGPURequestAdapterCallbackInfo2]]
WGPUProcInstanceRequestAdapterF: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPURequestAdapterOptions], struct_WGPURequestAdapterCallbackInfo]]
WGPUProcInstanceWaitAny: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUInstanceImpl], ctypes.c_uint64, c.POINTER[struct_WGPUFutureWaitInfo], ctypes.c_uint64]]
WGPUProcInstanceAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUInstanceImpl]]]
WGPUProcInstanceRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUInstanceImpl]]]
WGPUProcPipelineLayoutSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUPipelineLayoutImpl], struct_WGPUStringView]]
WGPUProcPipelineLayoutAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUPipelineLayoutImpl]]]
WGPUProcPipelineLayoutRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUPipelineLayoutImpl]]]
WGPUProcQuerySetDestroy: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQuerySetImpl]]]
WGPUProcQuerySetGetCount: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUQuerySetImpl]]]
WGPUProcQuerySetGetType: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUQuerySetImpl]]]
WGPUProcQuerySetSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQuerySetImpl], struct_WGPUStringView]]
WGPUProcQuerySetAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQuerySetImpl]]]
WGPUProcQuerySetRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQuerySetImpl]]]
WGPUProcQueueCopyExternalTextureForBrowser: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUImageCopyExternalTexture], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUExtent3D], c.POINTER[struct_WGPUCopyTextureForBrowserOptions]]]
WGPUProcQueueCopyTextureForBrowser: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUExtent3D], c.POINTER[struct_WGPUCopyTextureForBrowserOptions]]]
WGPUProcQueueOnSubmittedWorkDone: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl], c.CFUNCTYPE[None, [ctypes.c_uint32, ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcQueueOnSubmittedWorkDone2: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUQueueImpl], struct_WGPUQueueWorkDoneCallbackInfo2]]
WGPUProcQueueOnSubmittedWorkDoneF: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUQueueImpl], struct_WGPUQueueWorkDoneCallbackInfo]]
WGPUProcQueueSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl], struct_WGPUStringView]]
WGPUProcQueueSubmit: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl], ctypes.c_uint64, c.POINTER[c.POINTER[struct_WGPUCommandBufferImpl]]]]
WGPUProcQueueWriteBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]]
WGPUProcQueueWriteTexture: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUImageCopyTexture], ctypes.c_void_p, ctypes.c_uint64, c.POINTER[struct_WGPUTextureDataLayout], c.POINTER[struct_WGPUExtent3D]]]
WGPUProcQueueAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl]]]
WGPUProcQueueRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUQueueImpl]]]
WGPUProcRenderBundleSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleImpl], struct_WGPUStringView]]
WGPUProcRenderBundleAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleImpl]]]
WGPUProcRenderBundleRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleImpl]]]
WGPUProcRenderBundleEncoderDraw: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]]
WGPUProcRenderBundleEncoderDrawIndexed: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]]
WGPUProcRenderBundleEncoderDrawIndexedIndirect: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64]]
WGPUProcRenderBundleEncoderDrawIndirect: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64]]
WGPUProcRenderBundleEncoderFinish: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPURenderBundleImpl], [c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPURenderBundleDescriptor]]]
WGPUProcRenderBundleEncoderInsertDebugMarker: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], struct_WGPUStringView]]
WGPUProcRenderBundleEncoderPopDebugGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl]]]
WGPUProcRenderBundleEncoderPushDebugGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], struct_WGPUStringView]]
WGPUProcRenderBundleEncoderSetBindGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], ctypes.c_uint32, c.POINTER[struct_WGPUBindGroupImpl], ctypes.c_uint64, c.POINTER[ctypes.c_uint32]]]
WGPUProcRenderBundleEncoderSetIndexBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64]]
WGPUProcRenderBundleEncoderSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], struct_WGPUStringView]]
WGPUProcRenderBundleEncoderSetPipeline: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPURenderPipelineImpl]]]
WGPUProcRenderBundleEncoderSetVertexBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl], ctypes.c_uint32, c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64]]
WGPUProcRenderBundleEncoderAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl]]]
WGPUProcRenderBundleEncoderRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderBundleEncoderImpl]]]
WGPUProcRenderPassEncoderBeginOcclusionQuery: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_uint32]]
WGPUProcRenderPassEncoderDraw: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]]
WGPUProcRenderPassEncoderDrawIndexed: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]]
WGPUProcRenderPassEncoderDrawIndexedIndirect: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64]]
WGPUProcRenderPassEncoderDrawIndirect: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64]]
WGPUProcRenderPassEncoderEnd: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl]]]
WGPUProcRenderPassEncoderEndOcclusionQuery: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl]]]
WGPUProcRenderPassEncoderExecuteBundles: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_uint64, c.POINTER[c.POINTER[struct_WGPURenderBundleImpl]]]]
WGPUProcRenderPassEncoderInsertDebugMarker: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], struct_WGPUStringView]]
WGPUProcRenderPassEncoderMultiDrawIndexedIndirect: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint32, c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64]]
WGPUProcRenderPassEncoderMultiDrawIndirect: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint32, c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64]]
WGPUProcRenderPassEncoderPixelLocalStorageBarrier: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl]]]
WGPUProcRenderPassEncoderPopDebugGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl]]]
WGPUProcRenderPassEncoderPushDebugGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], struct_WGPUStringView]]
WGPUProcRenderPassEncoderSetBindGroup: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_uint32, c.POINTER[struct_WGPUBindGroupImpl], ctypes.c_uint64, c.POINTER[ctypes.c_uint32]]]
WGPUProcRenderPassEncoderSetBlendConstant: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUColor]]]
WGPUProcRenderPassEncoderSetIndexBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64]]
WGPUProcRenderPassEncoderSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], struct_WGPUStringView]]
WGPUProcRenderPassEncoderSetPipeline: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPURenderPipelineImpl]]]
WGPUProcRenderPassEncoderSetScissorRect: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]]
WGPUProcRenderPassEncoderSetStencilReference: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_uint32]]
WGPUProcRenderPassEncoderSetVertexBuffer: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_uint32, c.POINTER[struct_WGPUBufferImpl], ctypes.c_uint64, ctypes.c_uint64]]
WGPUProcRenderPassEncoderSetViewport: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]]
WGPUProcRenderPassEncoderWriteTimestamp: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUQuerySetImpl], ctypes.c_uint32]]
WGPUProcRenderPassEncoderAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl]]]
WGPUProcRenderPassEncoderRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPassEncoderImpl]]]
WGPUProcRenderPipelineGetBindGroupLayout: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUBindGroupLayoutImpl], [c.POINTER[struct_WGPURenderPipelineImpl], ctypes.c_uint32]]
WGPUProcRenderPipelineSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView]]
WGPUProcRenderPipelineAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPipelineImpl]]]
WGPUProcRenderPipelineRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPURenderPipelineImpl]]]
WGPUProcSamplerSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSamplerImpl], struct_WGPUStringView]]
WGPUProcSamplerAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSamplerImpl]]]
WGPUProcSamplerRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSamplerImpl]]]
WGPUProcShaderModuleGetCompilationInfo: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUShaderModuleImpl], c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[struct_WGPUCompilationInfo], ctypes.c_void_p]], ctypes.c_void_p]]
WGPUProcShaderModuleGetCompilationInfo2: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUShaderModuleImpl], struct_WGPUCompilationInfoCallbackInfo2]]
WGPUProcShaderModuleGetCompilationInfoF: TypeAlias = c.CFUNCTYPE[struct_WGPUFuture, [c.POINTER[struct_WGPUShaderModuleImpl], struct_WGPUCompilationInfoCallbackInfo]]
WGPUProcShaderModuleSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUShaderModuleImpl], struct_WGPUStringView]]
WGPUProcShaderModuleAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUShaderModuleImpl]]]
WGPUProcShaderModuleRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUShaderModuleImpl]]]
WGPUProcSharedBufferMemoryBeginAccess: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUBufferImpl], c.POINTER[struct_WGPUSharedBufferMemoryBeginAccessDescriptor]]]
WGPUProcSharedBufferMemoryCreateBuffer: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUBufferImpl], [c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUBufferDescriptor]]]
WGPUProcSharedBufferMemoryEndAccess: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUBufferImpl], c.POINTER[struct_WGPUSharedBufferMemoryEndAccessState]]]
WGPUProcSharedBufferMemoryGetProperties: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUSharedBufferMemoryProperties]]]
WGPUProcSharedBufferMemoryIsDeviceLost: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSharedBufferMemoryImpl]]]
WGPUProcSharedBufferMemorySetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedBufferMemoryImpl], struct_WGPUStringView]]
WGPUProcSharedBufferMemoryAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedBufferMemoryImpl]]]
WGPUProcSharedBufferMemoryRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedBufferMemoryImpl]]]
WGPUProcSharedFenceExportInfo: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedFenceImpl], c.POINTER[struct_WGPUSharedFenceExportInfo]]]
WGPUProcSharedFenceAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedFenceImpl]]]
WGPUProcSharedFenceRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedFenceImpl]]]
WGPUProcSharedTextureMemoryBeginAccess: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUSharedTextureMemoryBeginAccessDescriptor]]]
WGPUProcSharedTextureMemoryCreateTexture: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUTextureImpl], [c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUTextureDescriptor]]]
WGPUProcSharedTextureMemoryEndAccess: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUSharedTextureMemoryEndAccessState]]]
WGPUProcSharedTextureMemoryGetProperties: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUSharedTextureMemoryProperties]]]
WGPUProcSharedTextureMemoryIsDeviceLost: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSharedTextureMemoryImpl]]]
WGPUProcSharedTextureMemorySetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedTextureMemoryImpl], struct_WGPUStringView]]
WGPUProcSharedTextureMemoryAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedTextureMemoryImpl]]]
WGPUProcSharedTextureMemoryRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSharedTextureMemoryImpl]]]
WGPUProcSurfaceConfigure: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSurfaceImpl], c.POINTER[struct_WGPUSurfaceConfiguration]]]
WGPUProcSurfaceGetCapabilities: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUSurfaceImpl], c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUSurfaceCapabilities]]]
WGPUProcSurfaceGetCurrentTexture: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSurfaceImpl], c.POINTER[struct_WGPUSurfaceTexture]]]
WGPUProcSurfacePresent: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSurfaceImpl]]]
WGPUProcSurfaceSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSurfaceImpl], struct_WGPUStringView]]
WGPUProcSurfaceUnconfigure: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSurfaceImpl]]]
WGPUProcSurfaceAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSurfaceImpl]]]
WGPUProcSurfaceRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUSurfaceImpl]]]
WGPUProcTextureCreateErrorView: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUTextureViewImpl], [c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUTextureViewDescriptor]]]
WGPUProcTextureCreateView: TypeAlias = c.CFUNCTYPE[c.POINTER[struct_WGPUTextureViewImpl], [c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUTextureViewDescriptor]]]
WGPUProcTextureDestroy: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureGetDepthOrArrayLayers: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureGetDimension: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureGetFormat: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureGetHeight: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureGetMipLevelCount: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureGetSampleCount: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureGetUsage: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureGetWidth: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUTextureImpl], struct_WGPUStringView]]
WGPUProcTextureAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUTextureImpl]]]
WGPUProcTextureViewSetLabel: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUTextureViewImpl], struct_WGPUStringView]]
WGPUProcTextureViewAddRef: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUTextureViewImpl]]]
WGPUProcTextureViewRelease: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_WGPUTextureViewImpl]]]
@dll.bind(None, WGPUAdapterInfo)
def wgpuAdapterInfoFreeMembers(value:WGPUAdapterInfo) -> None: ...
@dll.bind(None, WGPUAdapterPropertiesMemoryHeaps)
def wgpuAdapterPropertiesMemoryHeapsFreeMembers(value:WGPUAdapterPropertiesMemoryHeaps) -> None: ...
@dll.bind(WGPUInstance, c.POINTER[WGPUInstanceDescriptor])
def wgpuCreateInstance(descriptor:c.POINTER[WGPUInstanceDescriptor]) -> WGPUInstance: ...
@dll.bind(None, WGPUDrmFormatCapabilities)
def wgpuDrmFormatCapabilitiesFreeMembers(value:WGPUDrmFormatCapabilities) -> None: ...
@dll.bind(WGPUStatus, c.POINTER[WGPUInstanceFeatures])
def wgpuGetInstanceFeatures(features:c.POINTER[WGPUInstanceFeatures]) -> WGPUStatus: ...
@dll.bind(WGPUProc, WGPUStringView)
def wgpuGetProcAddress(procName:WGPUStringView) -> WGPUProc: ...
@dll.bind(None, WGPUSharedBufferMemoryEndAccessState)
def wgpuSharedBufferMemoryEndAccessStateFreeMembers(value:WGPUSharedBufferMemoryEndAccessState) -> None: ...
@dll.bind(None, WGPUSharedTextureMemoryEndAccessState)
def wgpuSharedTextureMemoryEndAccessStateFreeMembers(value:WGPUSharedTextureMemoryEndAccessState) -> None: ...
@dll.bind(None, WGPUSupportedFeatures)
def wgpuSupportedFeaturesFreeMembers(value:WGPUSupportedFeatures) -> None: ...
@dll.bind(None, WGPUSurfaceCapabilities)
def wgpuSurfaceCapabilitiesFreeMembers(value:WGPUSurfaceCapabilities) -> None: ...
@dll.bind(WGPUDevice, WGPUAdapter, c.POINTER[WGPUDeviceDescriptor])
def wgpuAdapterCreateDevice(adapter:WGPUAdapter, descriptor:c.POINTER[WGPUDeviceDescriptor]) -> WGPUDevice: ...
@dll.bind(None, WGPUAdapter, c.POINTER[WGPUSupportedFeatures])
def wgpuAdapterGetFeatures(adapter:WGPUAdapter, features:c.POINTER[WGPUSupportedFeatures]) -> None: ...
@dll.bind(WGPUStatus, WGPUAdapter, WGPUTextureFormat, c.POINTER[WGPUFormatCapabilities])
def wgpuAdapterGetFormatCapabilities(adapter:WGPUAdapter, format:WGPUTextureFormat, capabilities:c.POINTER[WGPUFormatCapabilities]) -> WGPUStatus: ...
@dll.bind(WGPUStatus, WGPUAdapter, c.POINTER[WGPUAdapterInfo])
def wgpuAdapterGetInfo(adapter:WGPUAdapter, info:c.POINTER[WGPUAdapterInfo]) -> WGPUStatus: ...
@dll.bind(WGPUInstance, WGPUAdapter)
def wgpuAdapterGetInstance(adapter:WGPUAdapter) -> WGPUInstance: ...
@dll.bind(WGPUStatus, WGPUAdapter, c.POINTER[WGPUSupportedLimits])
def wgpuAdapterGetLimits(adapter:WGPUAdapter, limits:c.POINTER[WGPUSupportedLimits]) -> WGPUStatus: ...
@dll.bind(WGPUBool, WGPUAdapter, WGPUFeatureName)
def wgpuAdapterHasFeature(adapter:WGPUAdapter, feature:WGPUFeatureName) -> WGPUBool: ...
@dll.bind(None, WGPUAdapter, c.POINTER[WGPUDeviceDescriptor], WGPURequestDeviceCallback, ctypes.c_void_p)
def wgpuAdapterRequestDevice(adapter:WGPUAdapter, descriptor:c.POINTER[WGPUDeviceDescriptor], callback:WGPURequestDeviceCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(WGPUFuture, WGPUAdapter, c.POINTER[WGPUDeviceDescriptor], WGPURequestDeviceCallbackInfo2)
def wgpuAdapterRequestDevice2(adapter:WGPUAdapter, options:c.POINTER[WGPUDeviceDescriptor], callbackInfo:WGPURequestDeviceCallbackInfo2) -> WGPUFuture: ...
@dll.bind(WGPUFuture, WGPUAdapter, c.POINTER[WGPUDeviceDescriptor], WGPURequestDeviceCallbackInfo)
def wgpuAdapterRequestDeviceF(adapter:WGPUAdapter, options:c.POINTER[WGPUDeviceDescriptor], callbackInfo:WGPURequestDeviceCallbackInfo) -> WGPUFuture: ...
@dll.bind(None, WGPUAdapter)
def wgpuAdapterAddRef(adapter:WGPUAdapter) -> None: ...
@dll.bind(None, WGPUAdapter)
def wgpuAdapterRelease(adapter:WGPUAdapter) -> None: ...
@dll.bind(None, WGPUBindGroup, WGPUStringView)
def wgpuBindGroupSetLabel(bindGroup:WGPUBindGroup, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUBindGroup)
def wgpuBindGroupAddRef(bindGroup:WGPUBindGroup) -> None: ...
@dll.bind(None, WGPUBindGroup)
def wgpuBindGroupRelease(bindGroup:WGPUBindGroup) -> None: ...
@dll.bind(None, WGPUBindGroupLayout, WGPUStringView)
def wgpuBindGroupLayoutSetLabel(bindGroupLayout:WGPUBindGroupLayout, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUBindGroupLayout)
def wgpuBindGroupLayoutAddRef(bindGroupLayout:WGPUBindGroupLayout) -> None: ...
@dll.bind(None, WGPUBindGroupLayout)
def wgpuBindGroupLayoutRelease(bindGroupLayout:WGPUBindGroupLayout) -> None: ...
@dll.bind(None, WGPUBuffer)
def wgpuBufferDestroy(buffer:WGPUBuffer) -> None: ...
@dll.bind(ctypes.c_void_p, WGPUBuffer, size_t, size_t)
def wgpuBufferGetConstMappedRange(buffer:WGPUBuffer, offset:size_t, size:size_t) -> ctypes.c_void_p: ...
@dll.bind(WGPUBufferMapState, WGPUBuffer)
def wgpuBufferGetMapState(buffer:WGPUBuffer) -> WGPUBufferMapState: ...
@dll.bind(ctypes.c_void_p, WGPUBuffer, size_t, size_t)
def wgpuBufferGetMappedRange(buffer:WGPUBuffer, offset:size_t, size:size_t) -> ctypes.c_void_p: ...
@dll.bind(uint64_t, WGPUBuffer)
def wgpuBufferGetSize(buffer:WGPUBuffer) -> uint64_t: ...
@dll.bind(WGPUBufferUsage, WGPUBuffer)
def wgpuBufferGetUsage(buffer:WGPUBuffer) -> WGPUBufferUsage: ...
@dll.bind(None, WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallback, ctypes.c_void_p)
def wgpuBufferMapAsync(buffer:WGPUBuffer, mode:WGPUMapMode, offset:size_t, size:size_t, callback:WGPUBufferMapCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(WGPUFuture, WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallbackInfo2)
def wgpuBufferMapAsync2(buffer:WGPUBuffer, mode:WGPUMapMode, offset:size_t, size:size_t, callbackInfo:WGPUBufferMapCallbackInfo2) -> WGPUFuture: ...
@dll.bind(WGPUFuture, WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallbackInfo)
def wgpuBufferMapAsyncF(buffer:WGPUBuffer, mode:WGPUMapMode, offset:size_t, size:size_t, callbackInfo:WGPUBufferMapCallbackInfo) -> WGPUFuture: ...
@dll.bind(None, WGPUBuffer, WGPUStringView)
def wgpuBufferSetLabel(buffer:WGPUBuffer, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUBuffer)
def wgpuBufferUnmap(buffer:WGPUBuffer) -> None: ...
@dll.bind(None, WGPUBuffer)
def wgpuBufferAddRef(buffer:WGPUBuffer) -> None: ...
@dll.bind(None, WGPUBuffer)
def wgpuBufferRelease(buffer:WGPUBuffer) -> None: ...
@dll.bind(None, WGPUCommandBuffer, WGPUStringView)
def wgpuCommandBufferSetLabel(commandBuffer:WGPUCommandBuffer, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUCommandBuffer)
def wgpuCommandBufferAddRef(commandBuffer:WGPUCommandBuffer) -> None: ...
@dll.bind(None, WGPUCommandBuffer)
def wgpuCommandBufferRelease(commandBuffer:WGPUCommandBuffer) -> None: ...
@dll.bind(WGPUComputePassEncoder, WGPUCommandEncoder, c.POINTER[WGPUComputePassDescriptor])
def wgpuCommandEncoderBeginComputePass(commandEncoder:WGPUCommandEncoder, descriptor:c.POINTER[WGPUComputePassDescriptor]) -> WGPUComputePassEncoder: ...
@dll.bind(WGPURenderPassEncoder, WGPUCommandEncoder, c.POINTER[WGPURenderPassDescriptor])
def wgpuCommandEncoderBeginRenderPass(commandEncoder:WGPUCommandEncoder, descriptor:c.POINTER[WGPURenderPassDescriptor]) -> WGPURenderPassEncoder: ...
@dll.bind(None, WGPUCommandEncoder, WGPUBuffer, uint64_t, uint64_t)
def wgpuCommandEncoderClearBuffer(commandEncoder:WGPUCommandEncoder, buffer:WGPUBuffer, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind(None, WGPUCommandEncoder, WGPUBuffer, uint64_t, WGPUBuffer, uint64_t, uint64_t)
def wgpuCommandEncoderCopyBufferToBuffer(commandEncoder:WGPUCommandEncoder, source:WGPUBuffer, sourceOffset:uint64_t, destination:WGPUBuffer, destinationOffset:uint64_t, size:uint64_t) -> None: ...
@dll.bind(None, WGPUCommandEncoder, c.POINTER[WGPUImageCopyBuffer], c.POINTER[WGPUImageCopyTexture], c.POINTER[WGPUExtent3D])
def wgpuCommandEncoderCopyBufferToTexture(commandEncoder:WGPUCommandEncoder, source:c.POINTER[WGPUImageCopyBuffer], destination:c.POINTER[WGPUImageCopyTexture], copySize:c.POINTER[WGPUExtent3D]) -> None: ...
@dll.bind(None, WGPUCommandEncoder, c.POINTER[WGPUImageCopyTexture], c.POINTER[WGPUImageCopyBuffer], c.POINTER[WGPUExtent3D])
def wgpuCommandEncoderCopyTextureToBuffer(commandEncoder:WGPUCommandEncoder, source:c.POINTER[WGPUImageCopyTexture], destination:c.POINTER[WGPUImageCopyBuffer], copySize:c.POINTER[WGPUExtent3D]) -> None: ...
@dll.bind(None, WGPUCommandEncoder, c.POINTER[WGPUImageCopyTexture], c.POINTER[WGPUImageCopyTexture], c.POINTER[WGPUExtent3D])
def wgpuCommandEncoderCopyTextureToTexture(commandEncoder:WGPUCommandEncoder, source:c.POINTER[WGPUImageCopyTexture], destination:c.POINTER[WGPUImageCopyTexture], copySize:c.POINTER[WGPUExtent3D]) -> None: ...
@dll.bind(WGPUCommandBuffer, WGPUCommandEncoder, c.POINTER[WGPUCommandBufferDescriptor])
def wgpuCommandEncoderFinish(commandEncoder:WGPUCommandEncoder, descriptor:c.POINTER[WGPUCommandBufferDescriptor]) -> WGPUCommandBuffer: ...
@dll.bind(None, WGPUCommandEncoder, WGPUStringView)
def wgpuCommandEncoderInjectValidationError(commandEncoder:WGPUCommandEncoder, message:WGPUStringView) -> None: ...
@dll.bind(None, WGPUCommandEncoder, WGPUStringView)
def wgpuCommandEncoderInsertDebugMarker(commandEncoder:WGPUCommandEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind(None, WGPUCommandEncoder)
def wgpuCommandEncoderPopDebugGroup(commandEncoder:WGPUCommandEncoder) -> None: ...
@dll.bind(None, WGPUCommandEncoder, WGPUStringView)
def wgpuCommandEncoderPushDebugGroup(commandEncoder:WGPUCommandEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind(None, WGPUCommandEncoder, WGPUQuerySet, uint32_t, uint32_t, WGPUBuffer, uint64_t)
def wgpuCommandEncoderResolveQuerySet(commandEncoder:WGPUCommandEncoder, querySet:WGPUQuerySet, firstQuery:uint32_t, queryCount:uint32_t, destination:WGPUBuffer, destinationOffset:uint64_t) -> None: ...
@dll.bind(None, WGPUCommandEncoder, WGPUStringView)
def wgpuCommandEncoderSetLabel(commandEncoder:WGPUCommandEncoder, label:WGPUStringView) -> None: ...
uint8_t: TypeAlias = ctypes.c_ubyte
@dll.bind(None, WGPUCommandEncoder, WGPUBuffer, uint64_t, c.POINTER[uint8_t], uint64_t)
def wgpuCommandEncoderWriteBuffer(commandEncoder:WGPUCommandEncoder, buffer:WGPUBuffer, bufferOffset:uint64_t, data:c.POINTER[uint8_t], size:uint64_t) -> None: ...
@dll.bind(None, WGPUCommandEncoder, WGPUQuerySet, uint32_t)
def wgpuCommandEncoderWriteTimestamp(commandEncoder:WGPUCommandEncoder, querySet:WGPUQuerySet, queryIndex:uint32_t) -> None: ...
@dll.bind(None, WGPUCommandEncoder)
def wgpuCommandEncoderAddRef(commandEncoder:WGPUCommandEncoder) -> None: ...
@dll.bind(None, WGPUCommandEncoder)
def wgpuCommandEncoderRelease(commandEncoder:WGPUCommandEncoder) -> None: ...
@dll.bind(None, WGPUComputePassEncoder, uint32_t, uint32_t, uint32_t)
def wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder:WGPUComputePassEncoder, workgroupCountX:uint32_t, workgroupCountY:uint32_t, workgroupCountZ:uint32_t) -> None: ...
@dll.bind(None, WGPUComputePassEncoder, WGPUBuffer, uint64_t)
def wgpuComputePassEncoderDispatchWorkgroupsIndirect(computePassEncoder:WGPUComputePassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind(None, WGPUComputePassEncoder)
def wgpuComputePassEncoderEnd(computePassEncoder:WGPUComputePassEncoder) -> None: ...
@dll.bind(None, WGPUComputePassEncoder, WGPUStringView)
def wgpuComputePassEncoderInsertDebugMarker(computePassEncoder:WGPUComputePassEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind(None, WGPUComputePassEncoder)
def wgpuComputePassEncoderPopDebugGroup(computePassEncoder:WGPUComputePassEncoder) -> None: ...
@dll.bind(None, WGPUComputePassEncoder, WGPUStringView)
def wgpuComputePassEncoderPushDebugGroup(computePassEncoder:WGPUComputePassEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind(None, WGPUComputePassEncoder, uint32_t, WGPUBindGroup, size_t, c.POINTER[uint32_t])
def wgpuComputePassEncoderSetBindGroup(computePassEncoder:WGPUComputePassEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:c.POINTER[uint32_t]) -> None: ...
@dll.bind(None, WGPUComputePassEncoder, WGPUStringView)
def wgpuComputePassEncoderSetLabel(computePassEncoder:WGPUComputePassEncoder, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUComputePassEncoder, WGPUComputePipeline)
def wgpuComputePassEncoderSetPipeline(computePassEncoder:WGPUComputePassEncoder, pipeline:WGPUComputePipeline) -> None: ...
@dll.bind(None, WGPUComputePassEncoder, WGPUQuerySet, uint32_t)
def wgpuComputePassEncoderWriteTimestamp(computePassEncoder:WGPUComputePassEncoder, querySet:WGPUQuerySet, queryIndex:uint32_t) -> None: ...
@dll.bind(None, WGPUComputePassEncoder)
def wgpuComputePassEncoderAddRef(computePassEncoder:WGPUComputePassEncoder) -> None: ...
@dll.bind(None, WGPUComputePassEncoder)
def wgpuComputePassEncoderRelease(computePassEncoder:WGPUComputePassEncoder) -> None: ...
@dll.bind(WGPUBindGroupLayout, WGPUComputePipeline, uint32_t)
def wgpuComputePipelineGetBindGroupLayout(computePipeline:WGPUComputePipeline, groupIndex:uint32_t) -> WGPUBindGroupLayout: ...
@dll.bind(None, WGPUComputePipeline, WGPUStringView)
def wgpuComputePipelineSetLabel(computePipeline:WGPUComputePipeline, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUComputePipeline)
def wgpuComputePipelineAddRef(computePipeline:WGPUComputePipeline) -> None: ...
@dll.bind(None, WGPUComputePipeline)
def wgpuComputePipelineRelease(computePipeline:WGPUComputePipeline) -> None: ...
@dll.bind(WGPUBindGroup, WGPUDevice, c.POINTER[WGPUBindGroupDescriptor])
def wgpuDeviceCreateBindGroup(device:WGPUDevice, descriptor:c.POINTER[WGPUBindGroupDescriptor]) -> WGPUBindGroup: ...
@dll.bind(WGPUBindGroupLayout, WGPUDevice, c.POINTER[WGPUBindGroupLayoutDescriptor])
def wgpuDeviceCreateBindGroupLayout(device:WGPUDevice, descriptor:c.POINTER[WGPUBindGroupLayoutDescriptor]) -> WGPUBindGroupLayout: ...
@dll.bind(WGPUBuffer, WGPUDevice, c.POINTER[WGPUBufferDescriptor])
def wgpuDeviceCreateBuffer(device:WGPUDevice, descriptor:c.POINTER[WGPUBufferDescriptor]) -> WGPUBuffer: ...
@dll.bind(WGPUCommandEncoder, WGPUDevice, c.POINTER[WGPUCommandEncoderDescriptor])
def wgpuDeviceCreateCommandEncoder(device:WGPUDevice, descriptor:c.POINTER[WGPUCommandEncoderDescriptor]) -> WGPUCommandEncoder: ...
@dll.bind(WGPUComputePipeline, WGPUDevice, c.POINTER[WGPUComputePipelineDescriptor])
def wgpuDeviceCreateComputePipeline(device:WGPUDevice, descriptor:c.POINTER[WGPUComputePipelineDescriptor]) -> WGPUComputePipeline: ...
@dll.bind(None, WGPUDevice, c.POINTER[WGPUComputePipelineDescriptor], WGPUCreateComputePipelineAsyncCallback, ctypes.c_void_p)
def wgpuDeviceCreateComputePipelineAsync(device:WGPUDevice, descriptor:c.POINTER[WGPUComputePipelineDescriptor], callback:WGPUCreateComputePipelineAsyncCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(WGPUFuture, WGPUDevice, c.POINTER[WGPUComputePipelineDescriptor], WGPUCreateComputePipelineAsyncCallbackInfo2)
def wgpuDeviceCreateComputePipelineAsync2(device:WGPUDevice, descriptor:c.POINTER[WGPUComputePipelineDescriptor], callbackInfo:WGPUCreateComputePipelineAsyncCallbackInfo2) -> WGPUFuture: ...
@dll.bind(WGPUFuture, WGPUDevice, c.POINTER[WGPUComputePipelineDescriptor], WGPUCreateComputePipelineAsyncCallbackInfo)
def wgpuDeviceCreateComputePipelineAsyncF(device:WGPUDevice, descriptor:c.POINTER[WGPUComputePipelineDescriptor], callbackInfo:WGPUCreateComputePipelineAsyncCallbackInfo) -> WGPUFuture: ...
@dll.bind(WGPUBuffer, WGPUDevice, c.POINTER[WGPUBufferDescriptor])
def wgpuDeviceCreateErrorBuffer(device:WGPUDevice, descriptor:c.POINTER[WGPUBufferDescriptor]) -> WGPUBuffer: ...
@dll.bind(WGPUExternalTexture, WGPUDevice)
def wgpuDeviceCreateErrorExternalTexture(device:WGPUDevice) -> WGPUExternalTexture: ...
@dll.bind(WGPUShaderModule, WGPUDevice, c.POINTER[WGPUShaderModuleDescriptor], WGPUStringView)
def wgpuDeviceCreateErrorShaderModule(device:WGPUDevice, descriptor:c.POINTER[WGPUShaderModuleDescriptor], errorMessage:WGPUStringView) -> WGPUShaderModule: ...
@dll.bind(WGPUTexture, WGPUDevice, c.POINTER[WGPUTextureDescriptor])
def wgpuDeviceCreateErrorTexture(device:WGPUDevice, descriptor:c.POINTER[WGPUTextureDescriptor]) -> WGPUTexture: ...
@dll.bind(WGPUExternalTexture, WGPUDevice, c.POINTER[WGPUExternalTextureDescriptor])
def wgpuDeviceCreateExternalTexture(device:WGPUDevice, externalTextureDescriptor:c.POINTER[WGPUExternalTextureDescriptor]) -> WGPUExternalTexture: ...
@dll.bind(WGPUPipelineLayout, WGPUDevice, c.POINTER[WGPUPipelineLayoutDescriptor])
def wgpuDeviceCreatePipelineLayout(device:WGPUDevice, descriptor:c.POINTER[WGPUPipelineLayoutDescriptor]) -> WGPUPipelineLayout: ...
@dll.bind(WGPUQuerySet, WGPUDevice, c.POINTER[WGPUQuerySetDescriptor])
def wgpuDeviceCreateQuerySet(device:WGPUDevice, descriptor:c.POINTER[WGPUQuerySetDescriptor]) -> WGPUQuerySet: ...
@dll.bind(WGPURenderBundleEncoder, WGPUDevice, c.POINTER[WGPURenderBundleEncoderDescriptor])
def wgpuDeviceCreateRenderBundleEncoder(device:WGPUDevice, descriptor:c.POINTER[WGPURenderBundleEncoderDescriptor]) -> WGPURenderBundleEncoder: ...
@dll.bind(WGPURenderPipeline, WGPUDevice, c.POINTER[WGPURenderPipelineDescriptor])
def wgpuDeviceCreateRenderPipeline(device:WGPUDevice, descriptor:c.POINTER[WGPURenderPipelineDescriptor]) -> WGPURenderPipeline: ...
@dll.bind(None, WGPUDevice, c.POINTER[WGPURenderPipelineDescriptor], WGPUCreateRenderPipelineAsyncCallback, ctypes.c_void_p)
def wgpuDeviceCreateRenderPipelineAsync(device:WGPUDevice, descriptor:c.POINTER[WGPURenderPipelineDescriptor], callback:WGPUCreateRenderPipelineAsyncCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(WGPUFuture, WGPUDevice, c.POINTER[WGPURenderPipelineDescriptor], WGPUCreateRenderPipelineAsyncCallbackInfo2)
def wgpuDeviceCreateRenderPipelineAsync2(device:WGPUDevice, descriptor:c.POINTER[WGPURenderPipelineDescriptor], callbackInfo:WGPUCreateRenderPipelineAsyncCallbackInfo2) -> WGPUFuture: ...
@dll.bind(WGPUFuture, WGPUDevice, c.POINTER[WGPURenderPipelineDescriptor], WGPUCreateRenderPipelineAsyncCallbackInfo)
def wgpuDeviceCreateRenderPipelineAsyncF(device:WGPUDevice, descriptor:c.POINTER[WGPURenderPipelineDescriptor], callbackInfo:WGPUCreateRenderPipelineAsyncCallbackInfo) -> WGPUFuture: ...
@dll.bind(WGPUSampler, WGPUDevice, c.POINTER[WGPUSamplerDescriptor])
def wgpuDeviceCreateSampler(device:WGPUDevice, descriptor:c.POINTER[WGPUSamplerDescriptor]) -> WGPUSampler: ...
@dll.bind(WGPUShaderModule, WGPUDevice, c.POINTER[WGPUShaderModuleDescriptor])
def wgpuDeviceCreateShaderModule(device:WGPUDevice, descriptor:c.POINTER[WGPUShaderModuleDescriptor]) -> WGPUShaderModule: ...
@dll.bind(WGPUTexture, WGPUDevice, c.POINTER[WGPUTextureDescriptor])
def wgpuDeviceCreateTexture(device:WGPUDevice, descriptor:c.POINTER[WGPUTextureDescriptor]) -> WGPUTexture: ...
@dll.bind(None, WGPUDevice)
def wgpuDeviceDestroy(device:WGPUDevice) -> None: ...
@dll.bind(None, WGPUDevice, WGPUDeviceLostReason, WGPUStringView)
def wgpuDeviceForceLoss(device:WGPUDevice, type:WGPUDeviceLostReason, message:WGPUStringView) -> None: ...
@dll.bind(WGPUStatus, WGPUDevice, ctypes.c_void_p, c.POINTER[WGPUAHardwareBufferProperties])
def wgpuDeviceGetAHardwareBufferProperties(device:WGPUDevice, handle:ctypes.c_void_p, properties:c.POINTER[WGPUAHardwareBufferProperties]) -> WGPUStatus: ...
@dll.bind(WGPUAdapter, WGPUDevice)
def wgpuDeviceGetAdapter(device:WGPUDevice) -> WGPUAdapter: ...
@dll.bind(WGPUStatus, WGPUDevice, c.POINTER[WGPUAdapterInfo])
def wgpuDeviceGetAdapterInfo(device:WGPUDevice, adapterInfo:c.POINTER[WGPUAdapterInfo]) -> WGPUStatus: ...
@dll.bind(None, WGPUDevice, c.POINTER[WGPUSupportedFeatures])
def wgpuDeviceGetFeatures(device:WGPUDevice, features:c.POINTER[WGPUSupportedFeatures]) -> None: ...
@dll.bind(WGPUStatus, WGPUDevice, c.POINTER[WGPUSupportedLimits])
def wgpuDeviceGetLimits(device:WGPUDevice, limits:c.POINTER[WGPUSupportedLimits]) -> WGPUStatus: ...
@dll.bind(WGPUFuture, WGPUDevice)
def wgpuDeviceGetLostFuture(device:WGPUDevice) -> WGPUFuture: ...
@dll.bind(WGPUQueue, WGPUDevice)
def wgpuDeviceGetQueue(device:WGPUDevice) -> WGPUQueue: ...
@dll.bind(WGPUBool, WGPUDevice, WGPUFeatureName)
def wgpuDeviceHasFeature(device:WGPUDevice, feature:WGPUFeatureName) -> WGPUBool: ...
@dll.bind(WGPUSharedBufferMemory, WGPUDevice, c.POINTER[WGPUSharedBufferMemoryDescriptor])
def wgpuDeviceImportSharedBufferMemory(device:WGPUDevice, descriptor:c.POINTER[WGPUSharedBufferMemoryDescriptor]) -> WGPUSharedBufferMemory: ...
@dll.bind(WGPUSharedFence, WGPUDevice, c.POINTER[WGPUSharedFenceDescriptor])
def wgpuDeviceImportSharedFence(device:WGPUDevice, descriptor:c.POINTER[WGPUSharedFenceDescriptor]) -> WGPUSharedFence: ...
@dll.bind(WGPUSharedTextureMemory, WGPUDevice, c.POINTER[WGPUSharedTextureMemoryDescriptor])
def wgpuDeviceImportSharedTextureMemory(device:WGPUDevice, descriptor:c.POINTER[WGPUSharedTextureMemoryDescriptor]) -> WGPUSharedTextureMemory: ...
@dll.bind(None, WGPUDevice, WGPUErrorType, WGPUStringView)
def wgpuDeviceInjectError(device:WGPUDevice, type:WGPUErrorType, message:WGPUStringView) -> None: ...
@dll.bind(None, WGPUDevice, WGPUErrorCallback, ctypes.c_void_p)
def wgpuDevicePopErrorScope(device:WGPUDevice, oldCallback:WGPUErrorCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(WGPUFuture, WGPUDevice, WGPUPopErrorScopeCallbackInfo2)
def wgpuDevicePopErrorScope2(device:WGPUDevice, callbackInfo:WGPUPopErrorScopeCallbackInfo2) -> WGPUFuture: ...
@dll.bind(WGPUFuture, WGPUDevice, WGPUPopErrorScopeCallbackInfo)
def wgpuDevicePopErrorScopeF(device:WGPUDevice, callbackInfo:WGPUPopErrorScopeCallbackInfo) -> WGPUFuture: ...
@dll.bind(None, WGPUDevice, WGPUErrorFilter)
def wgpuDevicePushErrorScope(device:WGPUDevice, filter:WGPUErrorFilter) -> None: ...
@dll.bind(None, WGPUDevice, WGPUStringView)
def wgpuDeviceSetLabel(device:WGPUDevice, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUDevice, WGPULoggingCallback, ctypes.c_void_p)
def wgpuDeviceSetLoggingCallback(device:WGPUDevice, callback:WGPULoggingCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(None, WGPUDevice)
def wgpuDeviceTick(device:WGPUDevice) -> None: ...
@dll.bind(None, WGPUDevice, c.POINTER[WGPUTextureDescriptor])
def wgpuDeviceValidateTextureDescriptor(device:WGPUDevice, descriptor:c.POINTER[WGPUTextureDescriptor]) -> None: ...
@dll.bind(None, WGPUDevice)
def wgpuDeviceAddRef(device:WGPUDevice) -> None: ...
@dll.bind(None, WGPUDevice)
def wgpuDeviceRelease(device:WGPUDevice) -> None: ...
@dll.bind(None, WGPUExternalTexture)
def wgpuExternalTextureDestroy(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind(None, WGPUExternalTexture)
def wgpuExternalTextureExpire(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind(None, WGPUExternalTexture)
def wgpuExternalTextureRefresh(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind(None, WGPUExternalTexture, WGPUStringView)
def wgpuExternalTextureSetLabel(externalTexture:WGPUExternalTexture, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUExternalTexture)
def wgpuExternalTextureAddRef(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind(None, WGPUExternalTexture)
def wgpuExternalTextureRelease(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind(WGPUSurface, WGPUInstance, c.POINTER[WGPUSurfaceDescriptor])
def wgpuInstanceCreateSurface(instance:WGPUInstance, descriptor:c.POINTER[WGPUSurfaceDescriptor]) -> WGPUSurface: ...
@dll.bind(size_t, WGPUInstance, c.POINTER[WGPUWGSLFeatureName])
def wgpuInstanceEnumerateWGSLLanguageFeatures(instance:WGPUInstance, features:c.POINTER[WGPUWGSLFeatureName]) -> size_t: ...
@dll.bind(WGPUBool, WGPUInstance, WGPUWGSLFeatureName)
def wgpuInstanceHasWGSLLanguageFeature(instance:WGPUInstance, feature:WGPUWGSLFeatureName) -> WGPUBool: ...
@dll.bind(None, WGPUInstance)
def wgpuInstanceProcessEvents(instance:WGPUInstance) -> None: ...
@dll.bind(None, WGPUInstance, c.POINTER[WGPURequestAdapterOptions], WGPURequestAdapterCallback, ctypes.c_void_p)
def wgpuInstanceRequestAdapter(instance:WGPUInstance, options:c.POINTER[WGPURequestAdapterOptions], callback:WGPURequestAdapterCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(WGPUFuture, WGPUInstance, c.POINTER[WGPURequestAdapterOptions], WGPURequestAdapterCallbackInfo2)
def wgpuInstanceRequestAdapter2(instance:WGPUInstance, options:c.POINTER[WGPURequestAdapterOptions], callbackInfo:WGPURequestAdapterCallbackInfo2) -> WGPUFuture: ...
@dll.bind(WGPUFuture, WGPUInstance, c.POINTER[WGPURequestAdapterOptions], WGPURequestAdapterCallbackInfo)
def wgpuInstanceRequestAdapterF(instance:WGPUInstance, options:c.POINTER[WGPURequestAdapterOptions], callbackInfo:WGPURequestAdapterCallbackInfo) -> WGPUFuture: ...
@dll.bind(WGPUWaitStatus, WGPUInstance, size_t, c.POINTER[WGPUFutureWaitInfo], uint64_t)
def wgpuInstanceWaitAny(instance:WGPUInstance, futureCount:size_t, futures:c.POINTER[WGPUFutureWaitInfo], timeoutNS:uint64_t) -> WGPUWaitStatus: ...
@dll.bind(None, WGPUInstance)
def wgpuInstanceAddRef(instance:WGPUInstance) -> None: ...
@dll.bind(None, WGPUInstance)
def wgpuInstanceRelease(instance:WGPUInstance) -> None: ...
@dll.bind(None, WGPUPipelineLayout, WGPUStringView)
def wgpuPipelineLayoutSetLabel(pipelineLayout:WGPUPipelineLayout, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUPipelineLayout)
def wgpuPipelineLayoutAddRef(pipelineLayout:WGPUPipelineLayout) -> None: ...
@dll.bind(None, WGPUPipelineLayout)
def wgpuPipelineLayoutRelease(pipelineLayout:WGPUPipelineLayout) -> None: ...
@dll.bind(None, WGPUQuerySet)
def wgpuQuerySetDestroy(querySet:WGPUQuerySet) -> None: ...
@dll.bind(uint32_t, WGPUQuerySet)
def wgpuQuerySetGetCount(querySet:WGPUQuerySet) -> uint32_t: ...
@dll.bind(WGPUQueryType, WGPUQuerySet)
def wgpuQuerySetGetType(querySet:WGPUQuerySet) -> WGPUQueryType: ...
@dll.bind(None, WGPUQuerySet, WGPUStringView)
def wgpuQuerySetSetLabel(querySet:WGPUQuerySet, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUQuerySet)
def wgpuQuerySetAddRef(querySet:WGPUQuerySet) -> None: ...
@dll.bind(None, WGPUQuerySet)
def wgpuQuerySetRelease(querySet:WGPUQuerySet) -> None: ...
@dll.bind(None, WGPUQueue, c.POINTER[WGPUImageCopyExternalTexture], c.POINTER[WGPUImageCopyTexture], c.POINTER[WGPUExtent3D], c.POINTER[WGPUCopyTextureForBrowserOptions])
def wgpuQueueCopyExternalTextureForBrowser(queue:WGPUQueue, source:c.POINTER[WGPUImageCopyExternalTexture], destination:c.POINTER[WGPUImageCopyTexture], copySize:c.POINTER[WGPUExtent3D], options:c.POINTER[WGPUCopyTextureForBrowserOptions]) -> None: ...
@dll.bind(None, WGPUQueue, c.POINTER[WGPUImageCopyTexture], c.POINTER[WGPUImageCopyTexture], c.POINTER[WGPUExtent3D], c.POINTER[WGPUCopyTextureForBrowserOptions])
def wgpuQueueCopyTextureForBrowser(queue:WGPUQueue, source:c.POINTER[WGPUImageCopyTexture], destination:c.POINTER[WGPUImageCopyTexture], copySize:c.POINTER[WGPUExtent3D], options:c.POINTER[WGPUCopyTextureForBrowserOptions]) -> None: ...
@dll.bind(None, WGPUQueue, WGPUQueueWorkDoneCallback, ctypes.c_void_p)
def wgpuQueueOnSubmittedWorkDone(queue:WGPUQueue, callback:WGPUQueueWorkDoneCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(WGPUFuture, WGPUQueue, WGPUQueueWorkDoneCallbackInfo2)
def wgpuQueueOnSubmittedWorkDone2(queue:WGPUQueue, callbackInfo:WGPUQueueWorkDoneCallbackInfo2) -> WGPUFuture: ...
@dll.bind(WGPUFuture, WGPUQueue, WGPUQueueWorkDoneCallbackInfo)
def wgpuQueueOnSubmittedWorkDoneF(queue:WGPUQueue, callbackInfo:WGPUQueueWorkDoneCallbackInfo) -> WGPUFuture: ...
@dll.bind(None, WGPUQueue, WGPUStringView)
def wgpuQueueSetLabel(queue:WGPUQueue, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUQueue, size_t, c.POINTER[WGPUCommandBuffer])
def wgpuQueueSubmit(queue:WGPUQueue, commandCount:size_t, commands:c.POINTER[WGPUCommandBuffer]) -> None: ...
@dll.bind(None, WGPUQueue, WGPUBuffer, uint64_t, ctypes.c_void_p, size_t)
def wgpuQueueWriteBuffer(queue:WGPUQueue, buffer:WGPUBuffer, bufferOffset:uint64_t, data:ctypes.c_void_p, size:size_t) -> None: ...
@dll.bind(None, WGPUQueue, c.POINTER[WGPUImageCopyTexture], ctypes.c_void_p, size_t, c.POINTER[WGPUTextureDataLayout], c.POINTER[WGPUExtent3D])
def wgpuQueueWriteTexture(queue:WGPUQueue, destination:c.POINTER[WGPUImageCopyTexture], data:ctypes.c_void_p, dataSize:size_t, dataLayout:c.POINTER[WGPUTextureDataLayout], writeSize:c.POINTER[WGPUExtent3D]) -> None: ...
@dll.bind(None, WGPUQueue)
def wgpuQueueAddRef(queue:WGPUQueue) -> None: ...
@dll.bind(None, WGPUQueue)
def wgpuQueueRelease(queue:WGPUQueue) -> None: ...
@dll.bind(None, WGPURenderBundle, WGPUStringView)
def wgpuRenderBundleSetLabel(renderBundle:WGPURenderBundle, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPURenderBundle)
def wgpuRenderBundleAddRef(renderBundle:WGPURenderBundle) -> None: ...
@dll.bind(None, WGPURenderBundle)
def wgpuRenderBundleRelease(renderBundle:WGPURenderBundle) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, uint32_t, uint32_t, uint32_t, uint32_t)
def wgpuRenderBundleEncoderDraw(renderBundleEncoder:WGPURenderBundleEncoder, vertexCount:uint32_t, instanceCount:uint32_t, firstVertex:uint32_t, firstInstance:uint32_t) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
def wgpuRenderBundleEncoderDrawIndexed(renderBundleEncoder:WGPURenderBundleEncoder, indexCount:uint32_t, instanceCount:uint32_t, firstIndex:uint32_t, baseVertex:int32_t, firstInstance:uint32_t) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, WGPUBuffer, uint64_t)
def wgpuRenderBundleEncoderDrawIndexedIndirect(renderBundleEncoder:WGPURenderBundleEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, WGPUBuffer, uint64_t)
def wgpuRenderBundleEncoderDrawIndirect(renderBundleEncoder:WGPURenderBundleEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind(WGPURenderBundle, WGPURenderBundleEncoder, c.POINTER[WGPURenderBundleDescriptor])
def wgpuRenderBundleEncoderFinish(renderBundleEncoder:WGPURenderBundleEncoder, descriptor:c.POINTER[WGPURenderBundleDescriptor]) -> WGPURenderBundle: ...
@dll.bind(None, WGPURenderBundleEncoder, WGPUStringView)
def wgpuRenderBundleEncoderInsertDebugMarker(renderBundleEncoder:WGPURenderBundleEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder)
def wgpuRenderBundleEncoderPopDebugGroup(renderBundleEncoder:WGPURenderBundleEncoder) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, WGPUStringView)
def wgpuRenderBundleEncoderPushDebugGroup(renderBundleEncoder:WGPURenderBundleEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, uint32_t, WGPUBindGroup, size_t, c.POINTER[uint32_t])
def wgpuRenderBundleEncoderSetBindGroup(renderBundleEncoder:WGPURenderBundleEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:c.POINTER[uint32_t]) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, WGPUBuffer, WGPUIndexFormat, uint64_t, uint64_t)
def wgpuRenderBundleEncoderSetIndexBuffer(renderBundleEncoder:WGPURenderBundleEncoder, buffer:WGPUBuffer, format:WGPUIndexFormat, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, WGPUStringView)
def wgpuRenderBundleEncoderSetLabel(renderBundleEncoder:WGPURenderBundleEncoder, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, WGPURenderPipeline)
def wgpuRenderBundleEncoderSetPipeline(renderBundleEncoder:WGPURenderBundleEncoder, pipeline:WGPURenderPipeline) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder, uint32_t, WGPUBuffer, uint64_t, uint64_t)
def wgpuRenderBundleEncoderSetVertexBuffer(renderBundleEncoder:WGPURenderBundleEncoder, slot:uint32_t, buffer:WGPUBuffer, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder)
def wgpuRenderBundleEncoderAddRef(renderBundleEncoder:WGPURenderBundleEncoder) -> None: ...
@dll.bind(None, WGPURenderBundleEncoder)
def wgpuRenderBundleEncoderRelease(renderBundleEncoder:WGPURenderBundleEncoder) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, uint32_t)
def wgpuRenderPassEncoderBeginOcclusionQuery(renderPassEncoder:WGPURenderPassEncoder, queryIndex:uint32_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, uint32_t)
def wgpuRenderPassEncoderDraw(renderPassEncoder:WGPURenderPassEncoder, vertexCount:uint32_t, instanceCount:uint32_t, firstVertex:uint32_t, firstInstance:uint32_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
def wgpuRenderPassEncoderDrawIndexed(renderPassEncoder:WGPURenderPassEncoder, indexCount:uint32_t, instanceCount:uint32_t, firstIndex:uint32_t, baseVertex:int32_t, firstInstance:uint32_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUBuffer, uint64_t)
def wgpuRenderPassEncoderDrawIndexedIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUBuffer, uint64_t)
def wgpuRenderPassEncoderDrawIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder)
def wgpuRenderPassEncoderEnd(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind(None, WGPURenderPassEncoder)
def wgpuRenderPassEncoderEndOcclusionQuery(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, size_t, c.POINTER[WGPURenderBundle])
def wgpuRenderPassEncoderExecuteBundles(renderPassEncoder:WGPURenderPassEncoder, bundleCount:size_t, bundles:c.POINTER[WGPURenderBundle]) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUStringView)
def wgpuRenderPassEncoderInsertDebugMarker(renderPassEncoder:WGPURenderPassEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUBuffer, uint64_t, uint32_t, WGPUBuffer, uint64_t)
def wgpuRenderPassEncoderMultiDrawIndexedIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t, maxDrawCount:uint32_t, drawCountBuffer:WGPUBuffer, drawCountBufferOffset:uint64_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUBuffer, uint64_t, uint32_t, WGPUBuffer, uint64_t)
def wgpuRenderPassEncoderMultiDrawIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t, maxDrawCount:uint32_t, drawCountBuffer:WGPUBuffer, drawCountBufferOffset:uint64_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder)
def wgpuRenderPassEncoderPixelLocalStorageBarrier(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind(None, WGPURenderPassEncoder)
def wgpuRenderPassEncoderPopDebugGroup(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUStringView)
def wgpuRenderPassEncoderPushDebugGroup(renderPassEncoder:WGPURenderPassEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, uint32_t, WGPUBindGroup, size_t, c.POINTER[uint32_t])
def wgpuRenderPassEncoderSetBindGroup(renderPassEncoder:WGPURenderPassEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:c.POINTER[uint32_t]) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, c.POINTER[WGPUColor])
def wgpuRenderPassEncoderSetBlendConstant(renderPassEncoder:WGPURenderPassEncoder, color:c.POINTER[WGPUColor]) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUBuffer, WGPUIndexFormat, uint64_t, uint64_t)
def wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder:WGPURenderPassEncoder, buffer:WGPUBuffer, format:WGPUIndexFormat, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUStringView)
def wgpuRenderPassEncoderSetLabel(renderPassEncoder:WGPURenderPassEncoder, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPURenderPipeline)
def wgpuRenderPassEncoderSetPipeline(renderPassEncoder:WGPURenderPassEncoder, pipeline:WGPURenderPipeline) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, uint32_t)
def wgpuRenderPassEncoderSetScissorRect(renderPassEncoder:WGPURenderPassEncoder, x:uint32_t, y:uint32_t, width:uint32_t, height:uint32_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, uint32_t)
def wgpuRenderPassEncoderSetStencilReference(renderPassEncoder:WGPURenderPassEncoder, reference:uint32_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, uint32_t, WGPUBuffer, uint64_t, uint64_t)
def wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder:WGPURenderPassEncoder, slot:uint32_t, buffer:WGPUBuffer, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
def wgpuRenderPassEncoderSetViewport(renderPassEncoder:WGPURenderPassEncoder, x:float, y:float, width:float, height:float, minDepth:float, maxDepth:float) -> None: ...
@dll.bind(None, WGPURenderPassEncoder, WGPUQuerySet, uint32_t)
def wgpuRenderPassEncoderWriteTimestamp(renderPassEncoder:WGPURenderPassEncoder, querySet:WGPUQuerySet, queryIndex:uint32_t) -> None: ...
@dll.bind(None, WGPURenderPassEncoder)
def wgpuRenderPassEncoderAddRef(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind(None, WGPURenderPassEncoder)
def wgpuRenderPassEncoderRelease(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind(WGPUBindGroupLayout, WGPURenderPipeline, uint32_t)
def wgpuRenderPipelineGetBindGroupLayout(renderPipeline:WGPURenderPipeline, groupIndex:uint32_t) -> WGPUBindGroupLayout: ...
@dll.bind(None, WGPURenderPipeline, WGPUStringView)
def wgpuRenderPipelineSetLabel(renderPipeline:WGPURenderPipeline, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPURenderPipeline)
def wgpuRenderPipelineAddRef(renderPipeline:WGPURenderPipeline) -> None: ...
@dll.bind(None, WGPURenderPipeline)
def wgpuRenderPipelineRelease(renderPipeline:WGPURenderPipeline) -> None: ...
@dll.bind(None, WGPUSampler, WGPUStringView)
def wgpuSamplerSetLabel(sampler:WGPUSampler, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUSampler)
def wgpuSamplerAddRef(sampler:WGPUSampler) -> None: ...
@dll.bind(None, WGPUSampler)
def wgpuSamplerRelease(sampler:WGPUSampler) -> None: ...
@dll.bind(None, WGPUShaderModule, WGPUCompilationInfoCallback, ctypes.c_void_p)
def wgpuShaderModuleGetCompilationInfo(shaderModule:WGPUShaderModule, callback:WGPUCompilationInfoCallback, userdata:ctypes.c_void_p) -> None: ...
@dll.bind(WGPUFuture, WGPUShaderModule, WGPUCompilationInfoCallbackInfo2)
def wgpuShaderModuleGetCompilationInfo2(shaderModule:WGPUShaderModule, callbackInfo:WGPUCompilationInfoCallbackInfo2) -> WGPUFuture: ...
@dll.bind(WGPUFuture, WGPUShaderModule, WGPUCompilationInfoCallbackInfo)
def wgpuShaderModuleGetCompilationInfoF(shaderModule:WGPUShaderModule, callbackInfo:WGPUCompilationInfoCallbackInfo) -> WGPUFuture: ...
@dll.bind(None, WGPUShaderModule, WGPUStringView)
def wgpuShaderModuleSetLabel(shaderModule:WGPUShaderModule, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUShaderModule)
def wgpuShaderModuleAddRef(shaderModule:WGPUShaderModule) -> None: ...
@dll.bind(None, WGPUShaderModule)
def wgpuShaderModuleRelease(shaderModule:WGPUShaderModule) -> None: ...
@dll.bind(WGPUStatus, WGPUSharedBufferMemory, WGPUBuffer, c.POINTER[WGPUSharedBufferMemoryBeginAccessDescriptor])
def wgpuSharedBufferMemoryBeginAccess(sharedBufferMemory:WGPUSharedBufferMemory, buffer:WGPUBuffer, descriptor:c.POINTER[WGPUSharedBufferMemoryBeginAccessDescriptor]) -> WGPUStatus: ...
@dll.bind(WGPUBuffer, WGPUSharedBufferMemory, c.POINTER[WGPUBufferDescriptor])
def wgpuSharedBufferMemoryCreateBuffer(sharedBufferMemory:WGPUSharedBufferMemory, descriptor:c.POINTER[WGPUBufferDescriptor]) -> WGPUBuffer: ...
@dll.bind(WGPUStatus, WGPUSharedBufferMemory, WGPUBuffer, c.POINTER[WGPUSharedBufferMemoryEndAccessState])
def wgpuSharedBufferMemoryEndAccess(sharedBufferMemory:WGPUSharedBufferMemory, buffer:WGPUBuffer, descriptor:c.POINTER[WGPUSharedBufferMemoryEndAccessState]) -> WGPUStatus: ...
@dll.bind(WGPUStatus, WGPUSharedBufferMemory, c.POINTER[WGPUSharedBufferMemoryProperties])
def wgpuSharedBufferMemoryGetProperties(sharedBufferMemory:WGPUSharedBufferMemory, properties:c.POINTER[WGPUSharedBufferMemoryProperties]) -> WGPUStatus: ...
@dll.bind(WGPUBool, WGPUSharedBufferMemory)
def wgpuSharedBufferMemoryIsDeviceLost(sharedBufferMemory:WGPUSharedBufferMemory) -> WGPUBool: ...
@dll.bind(None, WGPUSharedBufferMemory, WGPUStringView)
def wgpuSharedBufferMemorySetLabel(sharedBufferMemory:WGPUSharedBufferMemory, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUSharedBufferMemory)
def wgpuSharedBufferMemoryAddRef(sharedBufferMemory:WGPUSharedBufferMemory) -> None: ...
@dll.bind(None, WGPUSharedBufferMemory)
def wgpuSharedBufferMemoryRelease(sharedBufferMemory:WGPUSharedBufferMemory) -> None: ...
@dll.bind(None, WGPUSharedFence, c.POINTER[WGPUSharedFenceExportInfo])
def wgpuSharedFenceExportInfo(sharedFence:WGPUSharedFence, info:c.POINTER[WGPUSharedFenceExportInfo]) -> None: ...
@dll.bind(None, WGPUSharedFence)
def wgpuSharedFenceAddRef(sharedFence:WGPUSharedFence) -> None: ...
@dll.bind(None, WGPUSharedFence)
def wgpuSharedFenceRelease(sharedFence:WGPUSharedFence) -> None: ...
@dll.bind(WGPUStatus, WGPUSharedTextureMemory, WGPUTexture, c.POINTER[WGPUSharedTextureMemoryBeginAccessDescriptor])
def wgpuSharedTextureMemoryBeginAccess(sharedTextureMemory:WGPUSharedTextureMemory, texture:WGPUTexture, descriptor:c.POINTER[WGPUSharedTextureMemoryBeginAccessDescriptor]) -> WGPUStatus: ...
@dll.bind(WGPUTexture, WGPUSharedTextureMemory, c.POINTER[WGPUTextureDescriptor])
def wgpuSharedTextureMemoryCreateTexture(sharedTextureMemory:WGPUSharedTextureMemory, descriptor:c.POINTER[WGPUTextureDescriptor]) -> WGPUTexture: ...
@dll.bind(WGPUStatus, WGPUSharedTextureMemory, WGPUTexture, c.POINTER[WGPUSharedTextureMemoryEndAccessState])
def wgpuSharedTextureMemoryEndAccess(sharedTextureMemory:WGPUSharedTextureMemory, texture:WGPUTexture, descriptor:c.POINTER[WGPUSharedTextureMemoryEndAccessState]) -> WGPUStatus: ...
@dll.bind(WGPUStatus, WGPUSharedTextureMemory, c.POINTER[WGPUSharedTextureMemoryProperties])
def wgpuSharedTextureMemoryGetProperties(sharedTextureMemory:WGPUSharedTextureMemory, properties:c.POINTER[WGPUSharedTextureMemoryProperties]) -> WGPUStatus: ...
@dll.bind(WGPUBool, WGPUSharedTextureMemory)
def wgpuSharedTextureMemoryIsDeviceLost(sharedTextureMemory:WGPUSharedTextureMemory) -> WGPUBool: ...
@dll.bind(None, WGPUSharedTextureMemory, WGPUStringView)
def wgpuSharedTextureMemorySetLabel(sharedTextureMemory:WGPUSharedTextureMemory, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUSharedTextureMemory)
def wgpuSharedTextureMemoryAddRef(sharedTextureMemory:WGPUSharedTextureMemory) -> None: ...
@dll.bind(None, WGPUSharedTextureMemory)
def wgpuSharedTextureMemoryRelease(sharedTextureMemory:WGPUSharedTextureMemory) -> None: ...
@dll.bind(None, WGPUSurface, c.POINTER[WGPUSurfaceConfiguration])
def wgpuSurfaceConfigure(surface:WGPUSurface, config:c.POINTER[WGPUSurfaceConfiguration]) -> None: ...
@dll.bind(WGPUStatus, WGPUSurface, WGPUAdapter, c.POINTER[WGPUSurfaceCapabilities])
def wgpuSurfaceGetCapabilities(surface:WGPUSurface, adapter:WGPUAdapter, capabilities:c.POINTER[WGPUSurfaceCapabilities]) -> WGPUStatus: ...
@dll.bind(None, WGPUSurface, c.POINTER[WGPUSurfaceTexture])
def wgpuSurfaceGetCurrentTexture(surface:WGPUSurface, surfaceTexture:c.POINTER[WGPUSurfaceTexture]) -> None: ...
@dll.bind(None, WGPUSurface)
def wgpuSurfacePresent(surface:WGPUSurface) -> None: ...
@dll.bind(None, WGPUSurface, WGPUStringView)
def wgpuSurfaceSetLabel(surface:WGPUSurface, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUSurface)
def wgpuSurfaceUnconfigure(surface:WGPUSurface) -> None: ...
@dll.bind(None, WGPUSurface)
def wgpuSurfaceAddRef(surface:WGPUSurface) -> None: ...
@dll.bind(None, WGPUSurface)
def wgpuSurfaceRelease(surface:WGPUSurface) -> None: ...
@dll.bind(WGPUTextureView, WGPUTexture, c.POINTER[WGPUTextureViewDescriptor])
def wgpuTextureCreateErrorView(texture:WGPUTexture, descriptor:c.POINTER[WGPUTextureViewDescriptor]) -> WGPUTextureView: ...
@dll.bind(WGPUTextureView, WGPUTexture, c.POINTER[WGPUTextureViewDescriptor])
def wgpuTextureCreateView(texture:WGPUTexture, descriptor:c.POINTER[WGPUTextureViewDescriptor]) -> WGPUTextureView: ...
@dll.bind(None, WGPUTexture)
def wgpuTextureDestroy(texture:WGPUTexture) -> None: ...
@dll.bind(uint32_t, WGPUTexture)
def wgpuTextureGetDepthOrArrayLayers(texture:WGPUTexture) -> uint32_t: ...
@dll.bind(WGPUTextureDimension, WGPUTexture)
def wgpuTextureGetDimension(texture:WGPUTexture) -> WGPUTextureDimension: ...
@dll.bind(WGPUTextureFormat, WGPUTexture)
def wgpuTextureGetFormat(texture:WGPUTexture) -> WGPUTextureFormat: ...
@dll.bind(uint32_t, WGPUTexture)
def wgpuTextureGetHeight(texture:WGPUTexture) -> uint32_t: ...
@dll.bind(uint32_t, WGPUTexture)
def wgpuTextureGetMipLevelCount(texture:WGPUTexture) -> uint32_t: ...
@dll.bind(uint32_t, WGPUTexture)
def wgpuTextureGetSampleCount(texture:WGPUTexture) -> uint32_t: ...
@dll.bind(WGPUTextureUsage, WGPUTexture)
def wgpuTextureGetUsage(texture:WGPUTexture) -> WGPUTextureUsage: ...
@dll.bind(uint32_t, WGPUTexture)
def wgpuTextureGetWidth(texture:WGPUTexture) -> uint32_t: ...
@dll.bind(None, WGPUTexture, WGPUStringView)
def wgpuTextureSetLabel(texture:WGPUTexture, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUTexture)
def wgpuTextureAddRef(texture:WGPUTexture) -> None: ...
@dll.bind(None, WGPUTexture)
def wgpuTextureRelease(texture:WGPUTexture) -> None: ...
@dll.bind(None, WGPUTextureView, WGPUStringView)
def wgpuTextureViewSetLabel(textureView:WGPUTextureView, label:WGPUStringView) -> None: ...
@dll.bind(None, WGPUTextureView)
def wgpuTextureViewAddRef(textureView:WGPUTextureView) -> None: ...
@dll.bind(None, WGPUTextureView)
def wgpuTextureViewRelease(textureView:WGPUTextureView) -> None: ...
WGPUBufferUsage_None = 0x0000000000000000
WGPUBufferUsage_MapRead = 0x0000000000000001
WGPUBufferUsage_MapWrite = 0x0000000000000002
WGPUBufferUsage_CopySrc = 0x0000000000000004
WGPUBufferUsage_CopyDst = 0x0000000000000008
WGPUBufferUsage_Index = 0x0000000000000010
WGPUBufferUsage_Vertex = 0x0000000000000020
WGPUBufferUsage_Uniform = 0x0000000000000040
WGPUBufferUsage_Storage = 0x0000000000000080
WGPUBufferUsage_Indirect = 0x0000000000000100
WGPUBufferUsage_QueryResolve = 0x0000000000000200
WGPUColorWriteMask_None = 0x0000000000000000
WGPUColorWriteMask_Red = 0x0000000000000001
WGPUColorWriteMask_Green = 0x0000000000000002
WGPUColorWriteMask_Blue = 0x0000000000000004
WGPUColorWriteMask_Alpha = 0x0000000000000008
WGPUColorWriteMask_All = 0x000000000000000F
WGPUHeapProperty_DeviceLocal = 0x0000000000000001
WGPUHeapProperty_HostVisible = 0x0000000000000002
WGPUHeapProperty_HostCoherent = 0x0000000000000004
WGPUHeapProperty_HostUncached = 0x0000000000000008
WGPUHeapProperty_HostCached = 0x0000000000000010
WGPUMapMode_None = 0x0000000000000000
WGPUMapMode_Read = 0x0000000000000001
WGPUMapMode_Write = 0x0000000000000002
WGPUShaderStage_None = 0x0000000000000000
WGPUShaderStage_Vertex = 0x0000000000000001
WGPUShaderStage_Fragment = 0x0000000000000002
WGPUShaderStage_Compute = 0x0000000000000004
WGPUTextureUsage_None = 0x0000000000000000
WGPUTextureUsage_CopySrc = 0x0000000000000001
WGPUTextureUsage_CopyDst = 0x0000000000000002
WGPUTextureUsage_TextureBinding = 0x0000000000000004
WGPUTextureUsage_StorageBinding = 0x0000000000000008
WGPUTextureUsage_RenderAttachment = 0x0000000000000010
WGPUTextureUsage_TransientAttachment = 0x0000000000000020
WGPUTextureUsage_StorageAttachment = 0x0000000000000040