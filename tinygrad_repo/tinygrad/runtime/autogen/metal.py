# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
from tinygrad.runtime.support import objc
dll = c.DLL('metal', 'Metal')
@c.record
class MTLDispatchThreadgroupsIndirectArguments(c.Struct):
  SIZE = 12
  threadgroupsPerGrid: c.Array[ctypes.c_uint32, Literal[3]]
uint32_t: TypeAlias = ctypes.c_uint32
MTLDispatchThreadgroupsIndirectArguments.register_fields([('threadgroupsPerGrid', c.Array[uint32_t, Literal[3]], 0)])
@c.record
class MTLStageInRegionIndirectArguments(c.Struct):
  SIZE = 24
  stageInOrigin: c.Array[ctypes.c_uint32, Literal[3]]
  stageInSize: c.Array[ctypes.c_uint32, Literal[3]]
MTLStageInRegionIndirectArguments.register_fields([('stageInOrigin', c.Array[uint32_t, Literal[3]], 0), ('stageInSize', c.Array[uint32_t, Literal[3]], 12)])
class MTLComputeCommandEncoder(objc.Spec): pass
class MTLCommandEncoder(objc.Spec): pass
class MTLComputePipelineState(objc.Spec): pass
NSUInteger: TypeAlias = ctypes.c_uint64
class MTLBuffer(objc.Spec): pass
class MTLResource(objc.Spec): pass
@c.record
class struct__NSRange(c.Struct):
  SIZE = 16
  location: int
  length: int
NSRange: TypeAlias = struct__NSRange
struct__NSRange.register_fields([('location', NSUInteger, 0), ('length', NSUInteger, 8)])
class MTLTexture(objc.Spec): pass
class MTLTextureDescriptor(objc.Spec): pass
enum_MTLTextureType: dict[int, str] = {(MTLTextureType1D:=0): 'MTLTextureType1D', (MTLTextureType1DArray:=1): 'MTLTextureType1DArray', (MTLTextureType2D:=2): 'MTLTextureType2D', (MTLTextureType2DArray:=3): 'MTLTextureType2DArray', (MTLTextureType2DMultisample:=4): 'MTLTextureType2DMultisample', (MTLTextureTypeCube:=5): 'MTLTextureTypeCube', (MTLTextureTypeCubeArray:=6): 'MTLTextureTypeCubeArray', (MTLTextureType3D:=7): 'MTLTextureType3D', (MTLTextureType2DMultisampleArray:=8): 'MTLTextureType2DMultisampleArray', (MTLTextureTypeTextureBuffer:=9): 'MTLTextureTypeTextureBuffer'}
MTLTextureType: TypeAlias = NSUInteger
enum_MTLPixelFormat: dict[int, str] = {(MTLPixelFormatInvalid:=0): 'MTLPixelFormatInvalid', (MTLPixelFormatA8Unorm:=1): 'MTLPixelFormatA8Unorm', (MTLPixelFormatR8Unorm:=10): 'MTLPixelFormatR8Unorm', (MTLPixelFormatR8Unorm_sRGB:=11): 'MTLPixelFormatR8Unorm_sRGB', (MTLPixelFormatR8Snorm:=12): 'MTLPixelFormatR8Snorm', (MTLPixelFormatR8Uint:=13): 'MTLPixelFormatR8Uint', (MTLPixelFormatR8Sint:=14): 'MTLPixelFormatR8Sint', (MTLPixelFormatR16Unorm:=20): 'MTLPixelFormatR16Unorm', (MTLPixelFormatR16Snorm:=22): 'MTLPixelFormatR16Snorm', (MTLPixelFormatR16Uint:=23): 'MTLPixelFormatR16Uint', (MTLPixelFormatR16Sint:=24): 'MTLPixelFormatR16Sint', (MTLPixelFormatR16Float:=25): 'MTLPixelFormatR16Float', (MTLPixelFormatRG8Unorm:=30): 'MTLPixelFormatRG8Unorm', (MTLPixelFormatRG8Unorm_sRGB:=31): 'MTLPixelFormatRG8Unorm_sRGB', (MTLPixelFormatRG8Snorm:=32): 'MTLPixelFormatRG8Snorm', (MTLPixelFormatRG8Uint:=33): 'MTLPixelFormatRG8Uint', (MTLPixelFormatRG8Sint:=34): 'MTLPixelFormatRG8Sint', (MTLPixelFormatB5G6R5Unorm:=40): 'MTLPixelFormatB5G6R5Unorm', (MTLPixelFormatA1BGR5Unorm:=41): 'MTLPixelFormatA1BGR5Unorm', (MTLPixelFormatABGR4Unorm:=42): 'MTLPixelFormatABGR4Unorm', (MTLPixelFormatBGR5A1Unorm:=43): 'MTLPixelFormatBGR5A1Unorm', (MTLPixelFormatR32Uint:=53): 'MTLPixelFormatR32Uint', (MTLPixelFormatR32Sint:=54): 'MTLPixelFormatR32Sint', (MTLPixelFormatR32Float:=55): 'MTLPixelFormatR32Float', (MTLPixelFormatRG16Unorm:=60): 'MTLPixelFormatRG16Unorm', (MTLPixelFormatRG16Snorm:=62): 'MTLPixelFormatRG16Snorm', (MTLPixelFormatRG16Uint:=63): 'MTLPixelFormatRG16Uint', (MTLPixelFormatRG16Sint:=64): 'MTLPixelFormatRG16Sint', (MTLPixelFormatRG16Float:=65): 'MTLPixelFormatRG16Float', (MTLPixelFormatRGBA8Unorm:=70): 'MTLPixelFormatRGBA8Unorm', (MTLPixelFormatRGBA8Unorm_sRGB:=71): 'MTLPixelFormatRGBA8Unorm_sRGB', (MTLPixelFormatRGBA8Snorm:=72): 'MTLPixelFormatRGBA8Snorm', (MTLPixelFormatRGBA8Uint:=73): 'MTLPixelFormatRGBA8Uint', (MTLPixelFormatRGBA8Sint:=74): 'MTLPixelFormatRGBA8Sint', (MTLPixelFormatBGRA8Unorm:=80): 'MTLPixelFormatBGRA8Unorm', (MTLPixelFormatBGRA8Unorm_sRGB:=81): 'MTLPixelFormatBGRA8Unorm_sRGB', (MTLPixelFormatRGB10A2Unorm:=90): 'MTLPixelFormatRGB10A2Unorm', (MTLPixelFormatRGB10A2Uint:=91): 'MTLPixelFormatRGB10A2Uint', (MTLPixelFormatRG11B10Float:=92): 'MTLPixelFormatRG11B10Float', (MTLPixelFormatRGB9E5Float:=93): 'MTLPixelFormatRGB9E5Float', (MTLPixelFormatBGR10A2Unorm:=94): 'MTLPixelFormatBGR10A2Unorm', (MTLPixelFormatBGR10_XR:=554): 'MTLPixelFormatBGR10_XR', (MTLPixelFormatBGR10_XR_sRGB:=555): 'MTLPixelFormatBGR10_XR_sRGB', (MTLPixelFormatRG32Uint:=103): 'MTLPixelFormatRG32Uint', (MTLPixelFormatRG32Sint:=104): 'MTLPixelFormatRG32Sint', (MTLPixelFormatRG32Float:=105): 'MTLPixelFormatRG32Float', (MTLPixelFormatRGBA16Unorm:=110): 'MTLPixelFormatRGBA16Unorm', (MTLPixelFormatRGBA16Snorm:=112): 'MTLPixelFormatRGBA16Snorm', (MTLPixelFormatRGBA16Uint:=113): 'MTLPixelFormatRGBA16Uint', (MTLPixelFormatRGBA16Sint:=114): 'MTLPixelFormatRGBA16Sint', (MTLPixelFormatRGBA16Float:=115): 'MTLPixelFormatRGBA16Float', (MTLPixelFormatBGRA10_XR:=552): 'MTLPixelFormatBGRA10_XR', (MTLPixelFormatBGRA10_XR_sRGB:=553): 'MTLPixelFormatBGRA10_XR_sRGB', (MTLPixelFormatRGBA32Uint:=123): 'MTLPixelFormatRGBA32Uint', (MTLPixelFormatRGBA32Sint:=124): 'MTLPixelFormatRGBA32Sint', (MTLPixelFormatRGBA32Float:=125): 'MTLPixelFormatRGBA32Float', (MTLPixelFormatBC1_RGBA:=130): 'MTLPixelFormatBC1_RGBA', (MTLPixelFormatBC1_RGBA_sRGB:=131): 'MTLPixelFormatBC1_RGBA_sRGB', (MTLPixelFormatBC2_RGBA:=132): 'MTLPixelFormatBC2_RGBA', (MTLPixelFormatBC2_RGBA_sRGB:=133): 'MTLPixelFormatBC2_RGBA_sRGB', (MTLPixelFormatBC3_RGBA:=134): 'MTLPixelFormatBC3_RGBA', (MTLPixelFormatBC3_RGBA_sRGB:=135): 'MTLPixelFormatBC3_RGBA_sRGB', (MTLPixelFormatBC4_RUnorm:=140): 'MTLPixelFormatBC4_RUnorm', (MTLPixelFormatBC4_RSnorm:=141): 'MTLPixelFormatBC4_RSnorm', (MTLPixelFormatBC5_RGUnorm:=142): 'MTLPixelFormatBC5_RGUnorm', (MTLPixelFormatBC5_RGSnorm:=143): 'MTLPixelFormatBC5_RGSnorm', (MTLPixelFormatBC6H_RGBFloat:=150): 'MTLPixelFormatBC6H_RGBFloat', (MTLPixelFormatBC6H_RGBUfloat:=151): 'MTLPixelFormatBC6H_RGBUfloat', (MTLPixelFormatBC7_RGBAUnorm:=152): 'MTLPixelFormatBC7_RGBAUnorm', (MTLPixelFormatBC7_RGBAUnorm_sRGB:=153): 'MTLPixelFormatBC7_RGBAUnorm_sRGB', (MTLPixelFormatPVRTC_RGB_2BPP:=160): 'MTLPixelFormatPVRTC_RGB_2BPP', (MTLPixelFormatPVRTC_RGB_2BPP_sRGB:=161): 'MTLPixelFormatPVRTC_RGB_2BPP_sRGB', (MTLPixelFormatPVRTC_RGB_4BPP:=162): 'MTLPixelFormatPVRTC_RGB_4BPP', (MTLPixelFormatPVRTC_RGB_4BPP_sRGB:=163): 'MTLPixelFormatPVRTC_RGB_4BPP_sRGB', (MTLPixelFormatPVRTC_RGBA_2BPP:=164): 'MTLPixelFormatPVRTC_RGBA_2BPP', (MTLPixelFormatPVRTC_RGBA_2BPP_sRGB:=165): 'MTLPixelFormatPVRTC_RGBA_2BPP_sRGB', (MTLPixelFormatPVRTC_RGBA_4BPP:=166): 'MTLPixelFormatPVRTC_RGBA_4BPP', (MTLPixelFormatPVRTC_RGBA_4BPP_sRGB:=167): 'MTLPixelFormatPVRTC_RGBA_4BPP_sRGB', (MTLPixelFormatEAC_R11Unorm:=170): 'MTLPixelFormatEAC_R11Unorm', (MTLPixelFormatEAC_R11Snorm:=172): 'MTLPixelFormatEAC_R11Snorm', (MTLPixelFormatEAC_RG11Unorm:=174): 'MTLPixelFormatEAC_RG11Unorm', (MTLPixelFormatEAC_RG11Snorm:=176): 'MTLPixelFormatEAC_RG11Snorm', (MTLPixelFormatEAC_RGBA8:=178): 'MTLPixelFormatEAC_RGBA8', (MTLPixelFormatEAC_RGBA8_sRGB:=179): 'MTLPixelFormatEAC_RGBA8_sRGB', (MTLPixelFormatETC2_RGB8:=180): 'MTLPixelFormatETC2_RGB8', (MTLPixelFormatETC2_RGB8_sRGB:=181): 'MTLPixelFormatETC2_RGB8_sRGB', (MTLPixelFormatETC2_RGB8A1:=182): 'MTLPixelFormatETC2_RGB8A1', (MTLPixelFormatETC2_RGB8A1_sRGB:=183): 'MTLPixelFormatETC2_RGB8A1_sRGB', (MTLPixelFormatASTC_4x4_sRGB:=186): 'MTLPixelFormatASTC_4x4_sRGB', (MTLPixelFormatASTC_5x4_sRGB:=187): 'MTLPixelFormatASTC_5x4_sRGB', (MTLPixelFormatASTC_5x5_sRGB:=188): 'MTLPixelFormatASTC_5x5_sRGB', (MTLPixelFormatASTC_6x5_sRGB:=189): 'MTLPixelFormatASTC_6x5_sRGB', (MTLPixelFormatASTC_6x6_sRGB:=190): 'MTLPixelFormatASTC_6x6_sRGB', (MTLPixelFormatASTC_8x5_sRGB:=192): 'MTLPixelFormatASTC_8x5_sRGB', (MTLPixelFormatASTC_8x6_sRGB:=193): 'MTLPixelFormatASTC_8x6_sRGB', (MTLPixelFormatASTC_8x8_sRGB:=194): 'MTLPixelFormatASTC_8x8_sRGB', (MTLPixelFormatASTC_10x5_sRGB:=195): 'MTLPixelFormatASTC_10x5_sRGB', (MTLPixelFormatASTC_10x6_sRGB:=196): 'MTLPixelFormatASTC_10x6_sRGB', (MTLPixelFormatASTC_10x8_sRGB:=197): 'MTLPixelFormatASTC_10x8_sRGB', (MTLPixelFormatASTC_10x10_sRGB:=198): 'MTLPixelFormatASTC_10x10_sRGB', (MTLPixelFormatASTC_12x10_sRGB:=199): 'MTLPixelFormatASTC_12x10_sRGB', (MTLPixelFormatASTC_12x12_sRGB:=200): 'MTLPixelFormatASTC_12x12_sRGB', (MTLPixelFormatASTC_4x4_LDR:=204): 'MTLPixelFormatASTC_4x4_LDR', (MTLPixelFormatASTC_5x4_LDR:=205): 'MTLPixelFormatASTC_5x4_LDR', (MTLPixelFormatASTC_5x5_LDR:=206): 'MTLPixelFormatASTC_5x5_LDR', (MTLPixelFormatASTC_6x5_LDR:=207): 'MTLPixelFormatASTC_6x5_LDR', (MTLPixelFormatASTC_6x6_LDR:=208): 'MTLPixelFormatASTC_6x6_LDR', (MTLPixelFormatASTC_8x5_LDR:=210): 'MTLPixelFormatASTC_8x5_LDR', (MTLPixelFormatASTC_8x6_LDR:=211): 'MTLPixelFormatASTC_8x6_LDR', (MTLPixelFormatASTC_8x8_LDR:=212): 'MTLPixelFormatASTC_8x8_LDR', (MTLPixelFormatASTC_10x5_LDR:=213): 'MTLPixelFormatASTC_10x5_LDR', (MTLPixelFormatASTC_10x6_LDR:=214): 'MTLPixelFormatASTC_10x6_LDR', (MTLPixelFormatASTC_10x8_LDR:=215): 'MTLPixelFormatASTC_10x8_LDR', (MTLPixelFormatASTC_10x10_LDR:=216): 'MTLPixelFormatASTC_10x10_LDR', (MTLPixelFormatASTC_12x10_LDR:=217): 'MTLPixelFormatASTC_12x10_LDR', (MTLPixelFormatASTC_12x12_LDR:=218): 'MTLPixelFormatASTC_12x12_LDR', (MTLPixelFormatASTC_4x4_HDR:=222): 'MTLPixelFormatASTC_4x4_HDR', (MTLPixelFormatASTC_5x4_HDR:=223): 'MTLPixelFormatASTC_5x4_HDR', (MTLPixelFormatASTC_5x5_HDR:=224): 'MTLPixelFormatASTC_5x5_HDR', (MTLPixelFormatASTC_6x5_HDR:=225): 'MTLPixelFormatASTC_6x5_HDR', (MTLPixelFormatASTC_6x6_HDR:=226): 'MTLPixelFormatASTC_6x6_HDR', (MTLPixelFormatASTC_8x5_HDR:=228): 'MTLPixelFormatASTC_8x5_HDR', (MTLPixelFormatASTC_8x6_HDR:=229): 'MTLPixelFormatASTC_8x6_HDR', (MTLPixelFormatASTC_8x8_HDR:=230): 'MTLPixelFormatASTC_8x8_HDR', (MTLPixelFormatASTC_10x5_HDR:=231): 'MTLPixelFormatASTC_10x5_HDR', (MTLPixelFormatASTC_10x6_HDR:=232): 'MTLPixelFormatASTC_10x6_HDR', (MTLPixelFormatASTC_10x8_HDR:=233): 'MTLPixelFormatASTC_10x8_HDR', (MTLPixelFormatASTC_10x10_HDR:=234): 'MTLPixelFormatASTC_10x10_HDR', (MTLPixelFormatASTC_12x10_HDR:=235): 'MTLPixelFormatASTC_12x10_HDR', (MTLPixelFormatASTC_12x12_HDR:=236): 'MTLPixelFormatASTC_12x12_HDR', (MTLPixelFormatGBGR422:=240): 'MTLPixelFormatGBGR422', (MTLPixelFormatBGRG422:=241): 'MTLPixelFormatBGRG422', (MTLPixelFormatDepth16Unorm:=250): 'MTLPixelFormatDepth16Unorm', (MTLPixelFormatDepth32Float:=252): 'MTLPixelFormatDepth32Float', (MTLPixelFormatStencil8:=253): 'MTLPixelFormatStencil8', (MTLPixelFormatDepth24Unorm_Stencil8:=255): 'MTLPixelFormatDepth24Unorm_Stencil8', (MTLPixelFormatDepth32Float_Stencil8:=260): 'MTLPixelFormatDepth32Float_Stencil8', (MTLPixelFormatX32_Stencil8:=261): 'MTLPixelFormatX32_Stencil8', (MTLPixelFormatX24_Stencil8:=262): 'MTLPixelFormatX24_Stencil8'}
MTLPixelFormat: TypeAlias = NSUInteger
enum_MTLResourceOptions: dict[int, str] = {(MTLResourceCPUCacheModeDefaultCache:=0): 'MTLResourceCPUCacheModeDefaultCache', (MTLResourceCPUCacheModeWriteCombined:=1): 'MTLResourceCPUCacheModeWriteCombined', (MTLResourceStorageModeShared:=0): 'MTLResourceStorageModeShared', (MTLResourceStorageModeManaged:=16): 'MTLResourceStorageModeManaged', (MTLResourceStorageModePrivate:=32): 'MTLResourceStorageModePrivate', (MTLResourceStorageModeMemoryless:=48): 'MTLResourceStorageModeMemoryless', (MTLResourceHazardTrackingModeDefault:=0): 'MTLResourceHazardTrackingModeDefault', (MTLResourceHazardTrackingModeUntracked:=256): 'MTLResourceHazardTrackingModeUntracked', (MTLResourceHazardTrackingModeTracked:=512): 'MTLResourceHazardTrackingModeTracked', (MTLResourceOptionCPUCacheModeDefault:=0): 'MTLResourceOptionCPUCacheModeDefault', (MTLResourceOptionCPUCacheModeWriteCombined:=1): 'MTLResourceOptionCPUCacheModeWriteCombined'}
MTLResourceOptions: TypeAlias = NSUInteger
enum_MTLCPUCacheMode: dict[int, str] = {(MTLCPUCacheModeDefaultCache:=0): 'MTLCPUCacheModeDefaultCache', (MTLCPUCacheModeWriteCombined:=1): 'MTLCPUCacheModeWriteCombined'}
MTLCPUCacheMode: TypeAlias = NSUInteger
enum_MTLStorageMode: dict[int, str] = {(MTLStorageModeShared:=0): 'MTLStorageModeShared', (MTLStorageModeManaged:=1): 'MTLStorageModeManaged', (MTLStorageModePrivate:=2): 'MTLStorageModePrivate', (MTLStorageModeMemoryless:=3): 'MTLStorageModeMemoryless'}
MTLStorageMode: TypeAlias = NSUInteger
enum_MTLHazardTrackingMode: dict[int, str] = {(MTLHazardTrackingModeDefault:=0): 'MTLHazardTrackingModeDefault', (MTLHazardTrackingModeUntracked:=1): 'MTLHazardTrackingModeUntracked', (MTLHazardTrackingModeTracked:=2): 'MTLHazardTrackingModeTracked'}
MTLHazardTrackingMode: TypeAlias = NSUInteger
enum_MTLTextureUsage: dict[int, str] = {(MTLTextureUsageUnknown:=0): 'MTLTextureUsageUnknown', (MTLTextureUsageShaderRead:=1): 'MTLTextureUsageShaderRead', (MTLTextureUsageShaderWrite:=2): 'MTLTextureUsageShaderWrite', (MTLTextureUsageRenderTarget:=4): 'MTLTextureUsageRenderTarget', (MTLTextureUsagePixelFormatView:=16): 'MTLTextureUsagePixelFormatView', (MTLTextureUsageShaderAtomic:=32): 'MTLTextureUsageShaderAtomic'}
MTLTextureUsage: TypeAlias = NSUInteger
BOOL: TypeAlias = ctypes.c_int32
NSInteger: TypeAlias = ctypes.c_int64
enum_MTLTextureCompressionType: dict[int, str] = {(MTLTextureCompressionTypeLossless:=0): 'MTLTextureCompressionTypeLossless', (MTLTextureCompressionTypeLossy:=1): 'MTLTextureCompressionTypeLossy'}
MTLTextureCompressionType: TypeAlias = NSInteger
@c.record
class MTLTextureSwizzleChannels(c.Struct):
  SIZE = 4
  red: int
  green: int
  blue: int
  alpha: int
uint8_t: TypeAlias = ctypes.c_ubyte
enum_MTLTextureSwizzle: dict[int, str] = {(MTLTextureSwizzleZero:=0): 'MTLTextureSwizzleZero', (MTLTextureSwizzleOne:=1): 'MTLTextureSwizzleOne', (MTLTextureSwizzleRed:=2): 'MTLTextureSwizzleRed', (MTLTextureSwizzleGreen:=3): 'MTLTextureSwizzleGreen', (MTLTextureSwizzleBlue:=4): 'MTLTextureSwizzleBlue', (MTLTextureSwizzleAlpha:=5): 'MTLTextureSwizzleAlpha'}
MTLTextureSwizzle: TypeAlias = uint8_t
MTLTextureSwizzleChannels.register_fields([('red', MTLTextureSwizzle, 0), ('green', MTLTextureSwizzle, 1), ('blue', MTLTextureSwizzle, 2), ('alpha', MTLTextureSwizzle, 3)])
class NSObject(objc.Spec): pass
IMP: TypeAlias = c.CFUNCTYPE[None, []]
class NSInvocation(objc.Spec): pass
class NSMethodSignature(objc.Spec): pass
NSMethodSignature._bases_ = [NSObject]
NSMethodSignature._methods_ = [
  ('getArgumentTypeAtIndex:', c.POINTER[ctypes.c_char], [NSUInteger]),
  ('isOneway', BOOL, []),
  ('numberOfArguments', NSUInteger, []),
  ('frameLength', NSUInteger, []),
  ('methodReturnType', c.POINTER[ctypes.c_char], []),
  ('methodReturnLength', NSUInteger, []),
]
NSMethodSignature._classmethods_ = [
  ('signatureWithObjCTypes:', NSMethodSignature, [c.POINTER[ctypes.c_char]]),
]
NSInvocation._bases_ = [NSObject]
NSInvocation._methods_ = [
  ('retainArguments', None, []),
  ('getReturnValue:', None, [ctypes.c_void_p]),
  ('setReturnValue:', None, [ctypes.c_void_p]),
  ('getArgument:atIndex:', None, [ctypes.c_void_p, NSInteger]),
  ('setArgument:atIndex:', None, [ctypes.c_void_p, NSInteger]),
  ('invoke', None, []),
  ('invokeWithTarget:', None, [objc.id_]),
  ('invokeUsingIMP:', None, [IMP]),
  ('methodSignature', NSMethodSignature, []),
  ('argumentsRetained', BOOL, []),
  ('target', objc.id_, []),
  ('setTarget:', None, [objc.id_]),
  ('selector', objc.id_, []),
  ('setSelector:', None, [objc.id_]),
]
NSInvocation._classmethods_ = [
  ('invocationWithMethodSignature:', NSInvocation, [NSMethodSignature]),
]
class struct__NSZone(c.Struct): pass
class Protocol(objc.Spec): pass
class NSString(objc.Spec): pass
unichar: TypeAlias = ctypes.c_uint16
class NSCoder(objc.Spec): pass
class NSData(objc.Spec): pass
NSData._bases_ = [NSObject]
NSData._methods_ = [
  ('length', NSUInteger, []),
  ('bytes', ctypes.c_void_p, []),
]
NSCoder._bases_ = [NSObject]
NSCoder._methods_ = [
  ('encodeValueOfObjCType:at:', None, [c.POINTER[ctypes.c_char], ctypes.c_void_p]),
  ('encodeDataObject:', None, [NSData]),
  ('decodeDataObject', NSData, []),
  ('decodeValueOfObjCType:at:size:', None, [c.POINTER[ctypes.c_char], ctypes.c_void_p, NSUInteger]),
  ('versionForClassName:', NSInteger, [NSString]),
]
NSString._bases_ = [NSObject]
NSString._methods_ = [
  ('characterAtIndex:', unichar, [NSUInteger]),
  ('init', 'instancetype', []),
  ('initWithCoder:', 'instancetype', [NSCoder]),
  ('length', NSUInteger, []),
]
NSObject._methods_ = [
  ('init', 'instancetype', []),
  ('dealloc', None, []),
  ('finalize', None, []),
  ('copy', objc.id_, [], True),
  ('mutableCopy', objc.id_, [], True),
  ('methodForSelector:', IMP, [objc.id_]),
  ('doesNotRecognizeSelector:', None, [objc.id_]),
  ('forwardingTargetForSelector:', objc.id_, [objc.id_]),
  ('forwardInvocation:', None, [NSInvocation]),
  ('methodSignatureForSelector:', NSMethodSignature, [objc.id_]),
  ('allowsWeakReference', BOOL, []),
  ('retainWeakReference', BOOL, []),
]
NSObject._classmethods_ = [
  ('load', None, []),
  ('initialize', None, []),
  ('new', 'instancetype', [], True),
  ('allocWithZone:', 'instancetype', [c.POINTER[struct__NSZone]], True),
  ('alloc', 'instancetype', [], True),
  ('copyWithZone:', objc.id_, [c.POINTER[struct__NSZone]], True),
  ('mutableCopyWithZone:', objc.id_, [c.POINTER[struct__NSZone]], True),
  ('instancesRespondToSelector:', BOOL, [objc.id_]),
  ('conformsToProtocol:', BOOL, [Protocol]),
  ('instanceMethodForSelector:', IMP, [objc.id_]),
  ('instanceMethodSignatureForSelector:', NSMethodSignature, [objc.id_]),
  ('resolveClassMethod:', BOOL, [objc.id_]),
  ('resolveInstanceMethod:', BOOL, [objc.id_]),
  ('hash', NSUInteger, []),
  ('description', NSString, []),
  ('debugDescription', NSString, []),
]
MTLTextureDescriptor._bases_ = [NSObject]
MTLTextureDescriptor._methods_ = [
  ('textureType', MTLTextureType, []),
  ('setTextureType:', None, [MTLTextureType]),
  ('pixelFormat', MTLPixelFormat, []),
  ('setPixelFormat:', None, [MTLPixelFormat]),
  ('width', NSUInteger, []),
  ('setWidth:', None, [NSUInteger]),
  ('height', NSUInteger, []),
  ('setHeight:', None, [NSUInteger]),
  ('depth', NSUInteger, []),
  ('setDepth:', None, [NSUInteger]),
  ('mipmapLevelCount', NSUInteger, []),
  ('setMipmapLevelCount:', None, [NSUInteger]),
  ('sampleCount', NSUInteger, []),
  ('setSampleCount:', None, [NSUInteger]),
  ('arrayLength', NSUInteger, []),
  ('setArrayLength:', None, [NSUInteger]),
  ('resourceOptions', MTLResourceOptions, []),
  ('setResourceOptions:', None, [MTLResourceOptions]),
  ('cpuCacheMode', MTLCPUCacheMode, []),
  ('setCpuCacheMode:', None, [MTLCPUCacheMode]),
  ('storageMode', MTLStorageMode, []),
  ('setStorageMode:', None, [MTLStorageMode]),
  ('hazardTrackingMode', MTLHazardTrackingMode, []),
  ('setHazardTrackingMode:', None, [MTLHazardTrackingMode]),
  ('usage', MTLTextureUsage, []),
  ('setUsage:', None, [MTLTextureUsage]),
  ('allowGPUOptimizedContents', BOOL, []),
  ('setAllowGPUOptimizedContents:', None, [BOOL]),
  ('compressionType', MTLTextureCompressionType, []),
  ('setCompressionType:', None, [MTLTextureCompressionType]),
  ('swizzle', MTLTextureSwizzleChannels, []),
  ('setSwizzle:', None, [MTLTextureSwizzleChannels]),
]
MTLTextureDescriptor._classmethods_ = [
  ('texture2DDescriptorWithPixelFormat:width:height:mipmapped:', MTLTextureDescriptor, [MTLPixelFormat, NSUInteger, NSUInteger, BOOL]),
  ('textureCubeDescriptorWithPixelFormat:size:mipmapped:', MTLTextureDescriptor, [MTLPixelFormat, NSUInteger, BOOL]),
  ('textureBufferDescriptorWithPixelFormat:width:resourceOptions:usage:', MTLTextureDescriptor, [MTLPixelFormat, NSUInteger, MTLResourceOptions, MTLTextureUsage]),
]
class MTLDevice(objc.Spec): pass
uint64_t: TypeAlias = ctypes.c_uint64
MTLBuffer._bases_ = [MTLResource]
MTLBuffer._methods_ = [
  ('contents', ctypes.c_void_p, []),
  ('didModifyRange:', None, [NSRange]),
  ('newTextureWithDescriptor:offset:bytesPerRow:', MTLTexture, [MTLTextureDescriptor, NSUInteger, NSUInteger], True),
  ('addDebugMarker:range:', None, [NSString, NSRange]),
  ('removeAllDebugMarkers', None, []),
  ('newRemoteBufferViewForDevice:', MTLBuffer, [MTLDevice], True),
  ('length', NSUInteger, []),
  ('remoteStorageBuffer', MTLBuffer, []),
  ('gpuAddress', uint64_t, []),
]
class MTLVisibleFunctionTable(objc.Spec): pass
class MTLIntersectionFunctionTable(objc.Spec): pass
class MTLAccelerationStructure(objc.Spec): pass
class MTLSamplerState(objc.Spec): pass
@c.record
class MTLRegion(c.Struct):
  SIZE = 48
  origin: MTLOrigin
  size: MTLSize
@c.record
class MTLOrigin(c.Struct):
  SIZE = 24
  x: int
  y: int
  z: int
MTLOrigin.register_fields([('x', NSUInteger, 0), ('y', NSUInteger, 8), ('z', NSUInteger, 16)])
@c.record
class MTLSize(c.Struct):
  SIZE = 24
  width: int
  height: int
  depth: int
MTLSize.register_fields([('width', NSUInteger, 0), ('height', NSUInteger, 8), ('depth', NSUInteger, 16)])
MTLRegion.register_fields([('origin', MTLOrigin, 0), ('size', MTLSize, 24)])
class MTLFence(objc.Spec): pass
MTLFence._bases_ = [NSObject]
MTLFence._methods_ = [
  ('device', MTLDevice, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
]
enum_MTLPurgeableState: dict[int, str] = {(MTLPurgeableStateKeepCurrent:=1): 'MTLPurgeableStateKeepCurrent', (MTLPurgeableStateNonVolatile:=2): 'MTLPurgeableStateNonVolatile', (MTLPurgeableStateVolatile:=3): 'MTLPurgeableStateVolatile', (MTLPurgeableStateEmpty:=4): 'MTLPurgeableStateEmpty'}
MTLPurgeableState: TypeAlias = NSUInteger
kern_return_t: TypeAlias = ctypes.c_int32
task_id_token_t: TypeAlias = ctypes.c_uint32
class MTLHeap(objc.Spec): pass
MTLResource._bases_ = [NSObject]
MTLResource._methods_ = [
  ('setPurgeableState:', MTLPurgeableState, [MTLPurgeableState]),
  ('makeAliasable', None, []),
  ('isAliasable', BOOL, []),
  ('setOwnerWithIdentity:', kern_return_t, [task_id_token_t]),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
  ('cpuCacheMode', MTLCPUCacheMode, []),
  ('storageMode', MTLStorageMode, []),
  ('hazardTrackingMode', MTLHazardTrackingMode, []),
  ('resourceOptions', MTLResourceOptions, []),
  ('heap', MTLHeap, []),
  ('heapOffset', NSUInteger, []),
  ('allocatedSize', NSUInteger, [], True),
]
enum_MTLResourceUsage: dict[int, str] = {(MTLResourceUsageRead:=1): 'MTLResourceUsageRead', (MTLResourceUsageWrite:=2): 'MTLResourceUsageWrite', (MTLResourceUsageSample:=4): 'MTLResourceUsageSample'}
MTLResourceUsage: TypeAlias = NSUInteger
class MTLIndirectCommandBuffer(objc.Spec): pass
enum_MTLBarrierScope: dict[int, str] = {(MTLBarrierScopeBuffers:=1): 'MTLBarrierScopeBuffers', (MTLBarrierScopeTextures:=2): 'MTLBarrierScopeTextures', (MTLBarrierScopeRenderTargets:=4): 'MTLBarrierScopeRenderTargets'}
MTLBarrierScope: TypeAlias = NSUInteger
class MTLCounterSampleBuffer(objc.Spec): pass
MTLCounterSampleBuffer._bases_ = [NSObject]
MTLCounterSampleBuffer._methods_ = [
  ('resolveCounterRange:', NSData, [NSRange]),
  ('device', MTLDevice, []),
  ('label', NSString, []),
  ('sampleCount', NSUInteger, []),
]
enum_MTLDispatchType: dict[int, str] = {(MTLDispatchTypeSerial:=0): 'MTLDispatchTypeSerial', (MTLDispatchTypeConcurrent:=1): 'MTLDispatchTypeConcurrent'}
MTLDispatchType: TypeAlias = NSUInteger
MTLComputeCommandEncoder._bases_ = [MTLCommandEncoder]
MTLComputeCommandEncoder._methods_ = [
  ('setComputePipelineState:', None, [MTLComputePipelineState]),
  ('setBytes:length:atIndex:', None, [ctypes.c_void_p, NSUInteger, NSUInteger]),
  ('setBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setBufferOffset:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setBuffers:offsets:withRange:', None, [c.POINTER[MTLBuffer], c.POINTER[NSUInteger], NSRange]),
  ('setBuffer:offset:attributeStride:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('setBuffers:offsets:attributeStrides:withRange:', None, [c.POINTER[MTLBuffer], c.POINTER[NSUInteger], c.POINTER[NSUInteger], NSRange]),
  ('setBufferOffset:attributeStride:atIndex:', None, [NSUInteger, NSUInteger, NSUInteger]),
  ('setBytes:length:attributeStride:atIndex:', None, [ctypes.c_void_p, NSUInteger, NSUInteger, NSUInteger]),
  ('setVisibleFunctionTable:atBufferIndex:', None, [MTLVisibleFunctionTable, NSUInteger]),
  ('setVisibleFunctionTables:withBufferRange:', None, [c.POINTER[MTLVisibleFunctionTable], NSRange]),
  ('setIntersectionFunctionTable:atBufferIndex:', None, [MTLIntersectionFunctionTable, NSUInteger]),
  ('setIntersectionFunctionTables:withBufferRange:', None, [c.POINTER[MTLIntersectionFunctionTable], NSRange]),
  ('setAccelerationStructure:atBufferIndex:', None, [MTLAccelerationStructure, NSUInteger]),
  ('setTexture:atIndex:', None, [MTLTexture, NSUInteger]),
  ('setTextures:withRange:', None, [c.POINTER[MTLTexture], NSRange]),
  ('setSamplerState:atIndex:', None, [MTLSamplerState, NSUInteger]),
  ('setSamplerStates:withRange:', None, [c.POINTER[MTLSamplerState], NSRange]),
  ('setSamplerState:lodMinClamp:lodMaxClamp:atIndex:', None, [MTLSamplerState, ctypes.c_float, ctypes.c_float, NSUInteger]),
  ('setSamplerStates:lodMinClamps:lodMaxClamps:withRange:', None, [c.POINTER[MTLSamplerState], c.POINTER[ctypes.c_float], c.POINTER[ctypes.c_float], NSRange]),
  ('setThreadgroupMemoryLength:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setImageblockWidth:height:', None, [NSUInteger, NSUInteger]),
  ('setStageInRegion:', None, [MTLRegion]),
  ('setStageInRegionWithIndirectBuffer:indirectBufferOffset:', None, [MTLBuffer, NSUInteger]),
  ('dispatchThreadgroups:threadsPerThreadgroup:', None, [MTLSize, MTLSize]),
  ('dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:', None, [MTLBuffer, NSUInteger, MTLSize]),
  ('dispatchThreads:threadsPerThreadgroup:', None, [MTLSize, MTLSize]),
  ('updateFence:', None, [MTLFence]),
  ('waitForFence:', None, [MTLFence]),
  ('useResource:usage:', None, [MTLResource, MTLResourceUsage]),
  ('useResources:count:usage:', None, [c.POINTER[MTLResource], NSUInteger, MTLResourceUsage]),
  ('useHeap:', None, [MTLHeap]),
  ('useHeaps:count:', None, [c.POINTER[MTLHeap], NSUInteger]),
  ('executeCommandsInBuffer:withRange:', None, [MTLIndirectCommandBuffer, NSRange]),
  ('executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:', None, [MTLIndirectCommandBuffer, MTLBuffer, NSUInteger]),
  ('memoryBarrierWithScope:', None, [MTLBarrierScope]),
  ('memoryBarrierWithResources:count:', None, [c.POINTER[MTLResource], NSUInteger]),
  ('sampleCountersInBuffer:atSampleIndex:withBarrier:', None, [MTLCounterSampleBuffer, NSUInteger, BOOL]),
  ('dispatchType', MTLDispatchType, []),
]
class MTLComputePipelineReflection(objc.Spec): pass
MTLComputePipelineReflection._bases_ = [NSObject]
class MTLComputePipelineDescriptor(objc.Spec): pass
class MTLFunction(objc.Spec): pass
class MTLArgumentEncoder(objc.Spec): pass
class MTLArgument(objc.Spec): pass
enum_MTLArgumentType: dict[int, str] = {(MTLArgumentTypeBuffer:=0): 'MTLArgumentTypeBuffer', (MTLArgumentTypeThreadgroupMemory:=1): 'MTLArgumentTypeThreadgroupMemory', (MTLArgumentTypeTexture:=2): 'MTLArgumentTypeTexture', (MTLArgumentTypeSampler:=3): 'MTLArgumentTypeSampler', (MTLArgumentTypeImageblockData:=16): 'MTLArgumentTypeImageblockData', (MTLArgumentTypeImageblock:=17): 'MTLArgumentTypeImageblock', (MTLArgumentTypeVisibleFunctionTable:=24): 'MTLArgumentTypeVisibleFunctionTable', (MTLArgumentTypePrimitiveAccelerationStructure:=25): 'MTLArgumentTypePrimitiveAccelerationStructure', (MTLArgumentTypeInstanceAccelerationStructure:=26): 'MTLArgumentTypeInstanceAccelerationStructure', (MTLArgumentTypeIntersectionFunctionTable:=27): 'MTLArgumentTypeIntersectionFunctionTable'}
MTLArgumentType: TypeAlias = NSUInteger
enum_MTLBindingAccess: dict[int, str] = {(MTLBindingAccessReadOnly:=0): 'MTLBindingAccessReadOnly', (MTLBindingAccessReadWrite:=1): 'MTLBindingAccessReadWrite', (MTLBindingAccessWriteOnly:=2): 'MTLBindingAccessWriteOnly', (MTLArgumentAccessReadOnly:=0): 'MTLArgumentAccessReadOnly', (MTLArgumentAccessReadWrite:=1): 'MTLArgumentAccessReadWrite', (MTLArgumentAccessWriteOnly:=2): 'MTLArgumentAccessWriteOnly'}
MTLBindingAccess: TypeAlias = NSUInteger
enum_MTLDataType: dict[int, str] = {(MTLDataTypeNone:=0): 'MTLDataTypeNone', (MTLDataTypeStruct:=1): 'MTLDataTypeStruct', (MTLDataTypeArray:=2): 'MTLDataTypeArray', (MTLDataTypeFloat:=3): 'MTLDataTypeFloat', (MTLDataTypeFloat2:=4): 'MTLDataTypeFloat2', (MTLDataTypeFloat3:=5): 'MTLDataTypeFloat3', (MTLDataTypeFloat4:=6): 'MTLDataTypeFloat4', (MTLDataTypeFloat2x2:=7): 'MTLDataTypeFloat2x2', (MTLDataTypeFloat2x3:=8): 'MTLDataTypeFloat2x3', (MTLDataTypeFloat2x4:=9): 'MTLDataTypeFloat2x4', (MTLDataTypeFloat3x2:=10): 'MTLDataTypeFloat3x2', (MTLDataTypeFloat3x3:=11): 'MTLDataTypeFloat3x3', (MTLDataTypeFloat3x4:=12): 'MTLDataTypeFloat3x4', (MTLDataTypeFloat4x2:=13): 'MTLDataTypeFloat4x2', (MTLDataTypeFloat4x3:=14): 'MTLDataTypeFloat4x3', (MTLDataTypeFloat4x4:=15): 'MTLDataTypeFloat4x4', (MTLDataTypeHalf:=16): 'MTLDataTypeHalf', (MTLDataTypeHalf2:=17): 'MTLDataTypeHalf2', (MTLDataTypeHalf3:=18): 'MTLDataTypeHalf3', (MTLDataTypeHalf4:=19): 'MTLDataTypeHalf4', (MTLDataTypeHalf2x2:=20): 'MTLDataTypeHalf2x2', (MTLDataTypeHalf2x3:=21): 'MTLDataTypeHalf2x3', (MTLDataTypeHalf2x4:=22): 'MTLDataTypeHalf2x4', (MTLDataTypeHalf3x2:=23): 'MTLDataTypeHalf3x2', (MTLDataTypeHalf3x3:=24): 'MTLDataTypeHalf3x3', (MTLDataTypeHalf3x4:=25): 'MTLDataTypeHalf3x4', (MTLDataTypeHalf4x2:=26): 'MTLDataTypeHalf4x2', (MTLDataTypeHalf4x3:=27): 'MTLDataTypeHalf4x3', (MTLDataTypeHalf4x4:=28): 'MTLDataTypeHalf4x4', (MTLDataTypeInt:=29): 'MTLDataTypeInt', (MTLDataTypeInt2:=30): 'MTLDataTypeInt2', (MTLDataTypeInt3:=31): 'MTLDataTypeInt3', (MTLDataTypeInt4:=32): 'MTLDataTypeInt4', (MTLDataTypeUInt:=33): 'MTLDataTypeUInt', (MTLDataTypeUInt2:=34): 'MTLDataTypeUInt2', (MTLDataTypeUInt3:=35): 'MTLDataTypeUInt3', (MTLDataTypeUInt4:=36): 'MTLDataTypeUInt4', (MTLDataTypeShort:=37): 'MTLDataTypeShort', (MTLDataTypeShort2:=38): 'MTLDataTypeShort2', (MTLDataTypeShort3:=39): 'MTLDataTypeShort3', (MTLDataTypeShort4:=40): 'MTLDataTypeShort4', (MTLDataTypeUShort:=41): 'MTLDataTypeUShort', (MTLDataTypeUShort2:=42): 'MTLDataTypeUShort2', (MTLDataTypeUShort3:=43): 'MTLDataTypeUShort3', (MTLDataTypeUShort4:=44): 'MTLDataTypeUShort4', (MTLDataTypeChar:=45): 'MTLDataTypeChar', (MTLDataTypeChar2:=46): 'MTLDataTypeChar2', (MTLDataTypeChar3:=47): 'MTLDataTypeChar3', (MTLDataTypeChar4:=48): 'MTLDataTypeChar4', (MTLDataTypeUChar:=49): 'MTLDataTypeUChar', (MTLDataTypeUChar2:=50): 'MTLDataTypeUChar2', (MTLDataTypeUChar3:=51): 'MTLDataTypeUChar3', (MTLDataTypeUChar4:=52): 'MTLDataTypeUChar4', (MTLDataTypeBool:=53): 'MTLDataTypeBool', (MTLDataTypeBool2:=54): 'MTLDataTypeBool2', (MTLDataTypeBool3:=55): 'MTLDataTypeBool3', (MTLDataTypeBool4:=56): 'MTLDataTypeBool4', (MTLDataTypeTexture:=58): 'MTLDataTypeTexture', (MTLDataTypeSampler:=59): 'MTLDataTypeSampler', (MTLDataTypePointer:=60): 'MTLDataTypePointer', (MTLDataTypeR8Unorm:=62): 'MTLDataTypeR8Unorm', (MTLDataTypeR8Snorm:=63): 'MTLDataTypeR8Snorm', (MTLDataTypeR16Unorm:=64): 'MTLDataTypeR16Unorm', (MTLDataTypeR16Snorm:=65): 'MTLDataTypeR16Snorm', (MTLDataTypeRG8Unorm:=66): 'MTLDataTypeRG8Unorm', (MTLDataTypeRG8Snorm:=67): 'MTLDataTypeRG8Snorm', (MTLDataTypeRG16Unorm:=68): 'MTLDataTypeRG16Unorm', (MTLDataTypeRG16Snorm:=69): 'MTLDataTypeRG16Snorm', (MTLDataTypeRGBA8Unorm:=70): 'MTLDataTypeRGBA8Unorm', (MTLDataTypeRGBA8Unorm_sRGB:=71): 'MTLDataTypeRGBA8Unorm_sRGB', (MTLDataTypeRGBA8Snorm:=72): 'MTLDataTypeRGBA8Snorm', (MTLDataTypeRGBA16Unorm:=73): 'MTLDataTypeRGBA16Unorm', (MTLDataTypeRGBA16Snorm:=74): 'MTLDataTypeRGBA16Snorm', (MTLDataTypeRGB10A2Unorm:=75): 'MTLDataTypeRGB10A2Unorm', (MTLDataTypeRG11B10Float:=76): 'MTLDataTypeRG11B10Float', (MTLDataTypeRGB9E5Float:=77): 'MTLDataTypeRGB9E5Float', (MTLDataTypeRenderPipeline:=78): 'MTLDataTypeRenderPipeline', (MTLDataTypeComputePipeline:=79): 'MTLDataTypeComputePipeline', (MTLDataTypeIndirectCommandBuffer:=80): 'MTLDataTypeIndirectCommandBuffer', (MTLDataTypeLong:=81): 'MTLDataTypeLong', (MTLDataTypeLong2:=82): 'MTLDataTypeLong2', (MTLDataTypeLong3:=83): 'MTLDataTypeLong3', (MTLDataTypeLong4:=84): 'MTLDataTypeLong4', (MTLDataTypeULong:=85): 'MTLDataTypeULong', (MTLDataTypeULong2:=86): 'MTLDataTypeULong2', (MTLDataTypeULong3:=87): 'MTLDataTypeULong3', (MTLDataTypeULong4:=88): 'MTLDataTypeULong4', (MTLDataTypeVisibleFunctionTable:=115): 'MTLDataTypeVisibleFunctionTable', (MTLDataTypeIntersectionFunctionTable:=116): 'MTLDataTypeIntersectionFunctionTable', (MTLDataTypePrimitiveAccelerationStructure:=117): 'MTLDataTypePrimitiveAccelerationStructure', (MTLDataTypeInstanceAccelerationStructure:=118): 'MTLDataTypeInstanceAccelerationStructure', (MTLDataTypeBFloat:=121): 'MTLDataTypeBFloat', (MTLDataTypeBFloat2:=122): 'MTLDataTypeBFloat2', (MTLDataTypeBFloat3:=123): 'MTLDataTypeBFloat3', (MTLDataTypeBFloat4:=124): 'MTLDataTypeBFloat4'}
MTLDataType: TypeAlias = NSUInteger
class MTLStructType(objc.Spec): pass
class MTLStructMember(objc.Spec): pass
class MTLArrayType(objc.Spec): pass
class MTLTextureReferenceType(objc.Spec): pass
class MTLType(objc.Spec): pass
MTLType._bases_ = [NSObject]
MTLType._methods_ = [
  ('dataType', MTLDataType, []),
]
MTLTextureReferenceType._bases_ = [MTLType]
MTLTextureReferenceType._methods_ = [
  ('textureDataType', MTLDataType, []),
  ('textureType', MTLTextureType, []),
  ('access', MTLBindingAccess, []),
  ('isDepthTexture', BOOL, []),
]
class MTLPointerType(objc.Spec): pass
MTLPointerType._bases_ = [MTLType]
MTLPointerType._methods_ = [
  ('elementStructType', MTLStructType, []),
  ('elementArrayType', MTLArrayType, []),
  ('elementType', MTLDataType, []),
  ('access', MTLBindingAccess, []),
  ('alignment', NSUInteger, []),
  ('dataSize', NSUInteger, []),
  ('elementIsArgumentBuffer', BOOL, []),
]
MTLArrayType._bases_ = [MTLType]
MTLArrayType._methods_ = [
  ('elementStructType', MTLStructType, []),
  ('elementArrayType', MTLArrayType, []),
  ('elementTextureReferenceType', MTLTextureReferenceType, []),
  ('elementPointerType', MTLPointerType, []),
  ('elementType', MTLDataType, []),
  ('arrayLength', NSUInteger, []),
  ('stride', NSUInteger, []),
  ('argumentIndexStride', NSUInteger, []),
]
MTLStructMember._bases_ = [NSObject]
MTLStructMember._methods_ = [
  ('structType', MTLStructType, []),
  ('arrayType', MTLArrayType, []),
  ('textureReferenceType', MTLTextureReferenceType, []),
  ('pointerType', MTLPointerType, []),
  ('name', NSString, []),
  ('offset', NSUInteger, []),
  ('dataType', MTLDataType, []),
  ('argumentIndex', NSUInteger, []),
]
MTLStructType._bases_ = [MTLType]
MTLStructType._methods_ = [
  ('memberByName:', MTLStructMember, [NSString]),
]
MTLArgument._bases_ = [NSObject]
MTLArgument._methods_ = [
  ('name', NSString, []),
  ('type', MTLArgumentType, []),
  ('access', MTLBindingAccess, []),
  ('index', NSUInteger, []),
  ('isActive', BOOL, []),
  ('bufferAlignment', NSUInteger, []),
  ('bufferDataSize', NSUInteger, []),
  ('bufferDataType', MTLDataType, []),
  ('bufferStructType', MTLStructType, []),
  ('bufferPointerType', MTLPointerType, []),
  ('threadgroupMemoryAlignment', NSUInteger, []),
  ('threadgroupMemoryDataSize', NSUInteger, []),
  ('textureType', MTLTextureType, []),
  ('textureDataType', MTLDataType, []),
  ('isDepthTexture', BOOL, []),
  ('arrayLength', NSUInteger, []),
]
enum_MTLFunctionType: dict[int, str] = {(MTLFunctionTypeVertex:=1): 'MTLFunctionTypeVertex', (MTLFunctionTypeFragment:=2): 'MTLFunctionTypeFragment', (MTLFunctionTypeKernel:=3): 'MTLFunctionTypeKernel', (MTLFunctionTypeVisible:=5): 'MTLFunctionTypeVisible', (MTLFunctionTypeIntersection:=6): 'MTLFunctionTypeIntersection', (MTLFunctionTypeMesh:=7): 'MTLFunctionTypeMesh', (MTLFunctionTypeObject:=8): 'MTLFunctionTypeObject'}
MTLFunctionType: TypeAlias = NSUInteger
enum_MTLPatchType: dict[int, str] = {(MTLPatchTypeNone:=0): 'MTLPatchTypeNone', (MTLPatchTypeTriangle:=1): 'MTLPatchTypeTriangle', (MTLPatchTypeQuad:=2): 'MTLPatchTypeQuad'}
MTLPatchType: TypeAlias = NSUInteger
enum_MTLFunctionOptions: dict[int, str] = {(MTLFunctionOptionNone:=0): 'MTLFunctionOptionNone', (MTLFunctionOptionCompileToBinary:=1): 'MTLFunctionOptionCompileToBinary', (MTLFunctionOptionStoreFunctionInMetalScript:=2): 'MTLFunctionOptionStoreFunctionInMetalScript'}
MTLFunctionOptions: TypeAlias = NSUInteger
MTLFunction._bases_ = [NSObject]
MTLFunction._methods_ = [
  ('newArgumentEncoderWithBufferIndex:', MTLArgumentEncoder, [NSUInteger], True),
  ('newArgumentEncoderWithBufferIndex:reflection:', MTLArgumentEncoder, [NSUInteger, c.POINTER[MTLArgument]], True),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
  ('functionType', MTLFunctionType, []),
  ('patchType', MTLPatchType, []),
  ('patchControlPointCount', NSInteger, []),
  ('name', NSString, []),
  ('options', MTLFunctionOptions, []),
]
class MTLStageInputOutputDescriptor(objc.Spec): pass
class MTLBufferLayoutDescriptorArray(objc.Spec): pass
class MTLBufferLayoutDescriptor(objc.Spec): pass
enum_MTLStepFunction: dict[int, str] = {(MTLStepFunctionConstant:=0): 'MTLStepFunctionConstant', (MTLStepFunctionPerVertex:=1): 'MTLStepFunctionPerVertex', (MTLStepFunctionPerInstance:=2): 'MTLStepFunctionPerInstance', (MTLStepFunctionPerPatch:=3): 'MTLStepFunctionPerPatch', (MTLStepFunctionPerPatchControlPoint:=4): 'MTLStepFunctionPerPatchControlPoint', (MTLStepFunctionThreadPositionInGridX:=5): 'MTLStepFunctionThreadPositionInGridX', (MTLStepFunctionThreadPositionInGridY:=6): 'MTLStepFunctionThreadPositionInGridY', (MTLStepFunctionThreadPositionInGridXIndexed:=7): 'MTLStepFunctionThreadPositionInGridXIndexed', (MTLStepFunctionThreadPositionInGridYIndexed:=8): 'MTLStepFunctionThreadPositionInGridYIndexed'}
MTLStepFunction: TypeAlias = NSUInteger
MTLBufferLayoutDescriptor._bases_ = [NSObject]
MTLBufferLayoutDescriptor._methods_ = [
  ('stride', NSUInteger, []),
  ('setStride:', None, [NSUInteger]),
  ('stepFunction', MTLStepFunction, []),
  ('setStepFunction:', None, [MTLStepFunction]),
  ('stepRate', NSUInteger, []),
  ('setStepRate:', None, [NSUInteger]),
]
MTLBufferLayoutDescriptorArray._bases_ = [NSObject]
MTLBufferLayoutDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLBufferLayoutDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLBufferLayoutDescriptor, NSUInteger]),
]
class MTLAttributeDescriptorArray(objc.Spec): pass
class MTLAttributeDescriptor(objc.Spec): pass
enum_MTLAttributeFormat: dict[int, str] = {(MTLAttributeFormatInvalid:=0): 'MTLAttributeFormatInvalid', (MTLAttributeFormatUChar2:=1): 'MTLAttributeFormatUChar2', (MTLAttributeFormatUChar3:=2): 'MTLAttributeFormatUChar3', (MTLAttributeFormatUChar4:=3): 'MTLAttributeFormatUChar4', (MTLAttributeFormatChar2:=4): 'MTLAttributeFormatChar2', (MTLAttributeFormatChar3:=5): 'MTLAttributeFormatChar3', (MTLAttributeFormatChar4:=6): 'MTLAttributeFormatChar4', (MTLAttributeFormatUChar2Normalized:=7): 'MTLAttributeFormatUChar2Normalized', (MTLAttributeFormatUChar3Normalized:=8): 'MTLAttributeFormatUChar3Normalized', (MTLAttributeFormatUChar4Normalized:=9): 'MTLAttributeFormatUChar4Normalized', (MTLAttributeFormatChar2Normalized:=10): 'MTLAttributeFormatChar2Normalized', (MTLAttributeFormatChar3Normalized:=11): 'MTLAttributeFormatChar3Normalized', (MTLAttributeFormatChar4Normalized:=12): 'MTLAttributeFormatChar4Normalized', (MTLAttributeFormatUShort2:=13): 'MTLAttributeFormatUShort2', (MTLAttributeFormatUShort3:=14): 'MTLAttributeFormatUShort3', (MTLAttributeFormatUShort4:=15): 'MTLAttributeFormatUShort4', (MTLAttributeFormatShort2:=16): 'MTLAttributeFormatShort2', (MTLAttributeFormatShort3:=17): 'MTLAttributeFormatShort3', (MTLAttributeFormatShort4:=18): 'MTLAttributeFormatShort4', (MTLAttributeFormatUShort2Normalized:=19): 'MTLAttributeFormatUShort2Normalized', (MTLAttributeFormatUShort3Normalized:=20): 'MTLAttributeFormatUShort3Normalized', (MTLAttributeFormatUShort4Normalized:=21): 'MTLAttributeFormatUShort4Normalized', (MTLAttributeFormatShort2Normalized:=22): 'MTLAttributeFormatShort2Normalized', (MTLAttributeFormatShort3Normalized:=23): 'MTLAttributeFormatShort3Normalized', (MTLAttributeFormatShort4Normalized:=24): 'MTLAttributeFormatShort4Normalized', (MTLAttributeFormatHalf2:=25): 'MTLAttributeFormatHalf2', (MTLAttributeFormatHalf3:=26): 'MTLAttributeFormatHalf3', (MTLAttributeFormatHalf4:=27): 'MTLAttributeFormatHalf4', (MTLAttributeFormatFloat:=28): 'MTLAttributeFormatFloat', (MTLAttributeFormatFloat2:=29): 'MTLAttributeFormatFloat2', (MTLAttributeFormatFloat3:=30): 'MTLAttributeFormatFloat3', (MTLAttributeFormatFloat4:=31): 'MTLAttributeFormatFloat4', (MTLAttributeFormatInt:=32): 'MTLAttributeFormatInt', (MTLAttributeFormatInt2:=33): 'MTLAttributeFormatInt2', (MTLAttributeFormatInt3:=34): 'MTLAttributeFormatInt3', (MTLAttributeFormatInt4:=35): 'MTLAttributeFormatInt4', (MTLAttributeFormatUInt:=36): 'MTLAttributeFormatUInt', (MTLAttributeFormatUInt2:=37): 'MTLAttributeFormatUInt2', (MTLAttributeFormatUInt3:=38): 'MTLAttributeFormatUInt3', (MTLAttributeFormatUInt4:=39): 'MTLAttributeFormatUInt4', (MTLAttributeFormatInt1010102Normalized:=40): 'MTLAttributeFormatInt1010102Normalized', (MTLAttributeFormatUInt1010102Normalized:=41): 'MTLAttributeFormatUInt1010102Normalized', (MTLAttributeFormatUChar4Normalized_BGRA:=42): 'MTLAttributeFormatUChar4Normalized_BGRA', (MTLAttributeFormatUChar:=45): 'MTLAttributeFormatUChar', (MTLAttributeFormatChar:=46): 'MTLAttributeFormatChar', (MTLAttributeFormatUCharNormalized:=47): 'MTLAttributeFormatUCharNormalized', (MTLAttributeFormatCharNormalized:=48): 'MTLAttributeFormatCharNormalized', (MTLAttributeFormatUShort:=49): 'MTLAttributeFormatUShort', (MTLAttributeFormatShort:=50): 'MTLAttributeFormatShort', (MTLAttributeFormatUShortNormalized:=51): 'MTLAttributeFormatUShortNormalized', (MTLAttributeFormatShortNormalized:=52): 'MTLAttributeFormatShortNormalized', (MTLAttributeFormatHalf:=53): 'MTLAttributeFormatHalf', (MTLAttributeFormatFloatRG11B10:=54): 'MTLAttributeFormatFloatRG11B10', (MTLAttributeFormatFloatRGB9E5:=55): 'MTLAttributeFormatFloatRGB9E5'}
MTLAttributeFormat: TypeAlias = NSUInteger
MTLAttributeDescriptor._bases_ = [NSObject]
MTLAttributeDescriptor._methods_ = [
  ('format', MTLAttributeFormat, []),
  ('setFormat:', None, [MTLAttributeFormat]),
  ('offset', NSUInteger, []),
  ('setOffset:', None, [NSUInteger]),
  ('bufferIndex', NSUInteger, []),
  ('setBufferIndex:', None, [NSUInteger]),
]
MTLAttributeDescriptorArray._bases_ = [NSObject]
MTLAttributeDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLAttributeDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLAttributeDescriptor, NSUInteger]),
]
enum_MTLIndexType: dict[int, str] = {(MTLIndexTypeUInt16:=0): 'MTLIndexTypeUInt16', (MTLIndexTypeUInt32:=1): 'MTLIndexTypeUInt32'}
MTLIndexType: TypeAlias = NSUInteger
MTLStageInputOutputDescriptor._bases_ = [NSObject]
MTLStageInputOutputDescriptor._methods_ = [
  ('reset', None, []),
  ('layouts', MTLBufferLayoutDescriptorArray, []),
  ('attributes', MTLAttributeDescriptorArray, []),
  ('indexType', MTLIndexType, []),
  ('setIndexType:', None, [MTLIndexType]),
  ('indexBufferIndex', NSUInteger, []),
  ('setIndexBufferIndex:', None, [NSUInteger]),
]
MTLStageInputOutputDescriptor._classmethods_ = [
  ('stageInputOutputDescriptor', MTLStageInputOutputDescriptor, []),
]
class MTLPipelineBufferDescriptorArray(objc.Spec): pass
class MTLPipelineBufferDescriptor(objc.Spec): pass
enum_MTLMutability: dict[int, str] = {(MTLMutabilityDefault:=0): 'MTLMutabilityDefault', (MTLMutabilityMutable:=1): 'MTLMutabilityMutable', (MTLMutabilityImmutable:=2): 'MTLMutabilityImmutable'}
MTLMutability: TypeAlias = NSUInteger
MTLPipelineBufferDescriptor._bases_ = [NSObject]
MTLPipelineBufferDescriptor._methods_ = [
  ('mutability', MTLMutability, []),
  ('setMutability:', None, [MTLMutability]),
]
MTLPipelineBufferDescriptorArray._bases_ = [NSObject]
MTLPipelineBufferDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLPipelineBufferDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLPipelineBufferDescriptor, NSUInteger]),
]
class MTLLinkedFunctions(objc.Spec): pass
MTLLinkedFunctions._bases_ = [NSObject]
MTLLinkedFunctions._classmethods_ = [
  ('linkedFunctions', MTLLinkedFunctions, []),
]
MTLComputePipelineDescriptor._bases_ = [NSObject]
MTLComputePipelineDescriptor._methods_ = [
  ('reset', None, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('computeFunction', MTLFunction, []),
  ('setComputeFunction:', None, [MTLFunction]),
  ('threadGroupSizeIsMultipleOfThreadExecutionWidth', BOOL, []),
  ('setThreadGroupSizeIsMultipleOfThreadExecutionWidth:', None, [BOOL]),
  ('maxTotalThreadsPerThreadgroup', NSUInteger, []),
  ('setMaxTotalThreadsPerThreadgroup:', None, [NSUInteger]),
  ('stageInputDescriptor', MTLStageInputOutputDescriptor, []),
  ('setStageInputDescriptor:', None, [MTLStageInputOutputDescriptor]),
  ('buffers', MTLPipelineBufferDescriptorArray, []),
  ('supportIndirectCommandBuffers', BOOL, []),
  ('setSupportIndirectCommandBuffers:', None, [BOOL]),
  ('linkedFunctions', MTLLinkedFunctions, []),
  ('setLinkedFunctions:', None, [MTLLinkedFunctions]),
  ('supportAddingBinaryFunctions', BOOL, []),
  ('setSupportAddingBinaryFunctions:', None, [BOOL]),
  ('maxCallStackDepth', NSUInteger, []),
  ('setMaxCallStackDepth:', None, [NSUInteger]),
]
class MTLFunctionHandle(objc.Spec): pass
class MTLVisibleFunctionTableDescriptor(objc.Spec): pass
class MTLIntersectionFunctionTableDescriptor(objc.Spec): pass
@c.record
class struct_MTLResourceID(c.Struct):
  SIZE = 8
  _impl: int
MTLResourceID: TypeAlias = struct_MTLResourceID
struct_MTLResourceID.register_fields([('_impl', uint64_t, 0)])
MTLComputePipelineState._bases_ = [NSObject]
MTLComputePipelineState._methods_ = [
  ('imageblockMemoryLengthForDimensions:', NSUInteger, [MTLSize]),
  ('functionHandleWithFunction:', MTLFunctionHandle, [MTLFunction]),
  ('newVisibleFunctionTableWithDescriptor:', MTLVisibleFunctionTable, [MTLVisibleFunctionTableDescriptor], True),
  ('newIntersectionFunctionTableWithDescriptor:', MTLIntersectionFunctionTable, [MTLIntersectionFunctionTableDescriptor], True),
  ('label', NSString, []),
  ('device', MTLDevice, []),
  ('maxTotalThreadsPerThreadgroup', NSUInteger, []),
  ('threadExecutionWidth', NSUInteger, []),
  ('staticThreadgroupMemoryLength', NSUInteger, []),
  ('supportIndirectCommandBuffers', BOOL, []),
  ('gpuResourceID', MTLResourceID, []),
]
class MTLCommandQueue(objc.Spec): pass
class MTLCommandBuffer(objc.Spec): pass
class MTLDrawable(objc.Spec): pass
CFTimeInterval: TypeAlias = ctypes.c_double
class MTLBlitCommandEncoder(objc.Spec): pass
enum_MTLBlitOption: dict[int, str] = {(MTLBlitOptionNone:=0): 'MTLBlitOptionNone', (MTLBlitOptionDepthFromDepthStencil:=1): 'MTLBlitOptionDepthFromDepthStencil', (MTLBlitOptionStencilFromDepthStencil:=2): 'MTLBlitOptionStencilFromDepthStencil', (MTLBlitOptionRowLinearPVRTC:=4): 'MTLBlitOptionRowLinearPVRTC'}
MTLBlitOption: TypeAlias = NSUInteger
MTLBlitCommandEncoder._bases_ = [MTLCommandEncoder]
MTLBlitCommandEncoder._methods_ = [
  ('synchronizeResource:', None, [MTLResource]),
  ('synchronizeTexture:slice:level:', None, [MTLTexture, NSUInteger, NSUInteger]),
  ('copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:', None, [MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLSize, MTLTexture, NSUInteger, NSUInteger, MTLOrigin]),
  ('copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLSize, MTLTexture, NSUInteger, NSUInteger, MTLOrigin]),
  ('copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLSize, MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLBlitOption]),
  ('copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:', None, [MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLSize, MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:', None, [MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLSize, MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLBlitOption]),
  ('generateMipmapsForTexture:', None, [MTLTexture]),
  ('fillBuffer:range:value:', None, [MTLBuffer, NSRange, uint8_t]),
  ('copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:', None, [MTLTexture, NSUInteger, NSUInteger, MTLTexture, NSUInteger, NSUInteger, NSUInteger, NSUInteger]),
  ('copyFromTexture:toTexture:', None, [MTLTexture, MTLTexture]),
  ('copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:', None, [MTLBuffer, NSUInteger, MTLBuffer, NSUInteger, NSUInteger]),
  ('updateFence:', None, [MTLFence]),
  ('waitForFence:', None, [MTLFence]),
  ('getTextureAccessCounters:region:mipLevel:slice:resetCounters:countersBuffer:countersBufferOffset:', None, [MTLTexture, MTLRegion, NSUInteger, NSUInteger, BOOL, MTLBuffer, NSUInteger]),
  ('resetTextureAccessCounters:region:mipLevel:slice:', None, [MTLTexture, MTLRegion, NSUInteger, NSUInteger]),
  ('optimizeContentsForGPUAccess:', None, [MTLTexture]),
  ('optimizeContentsForGPUAccess:slice:level:', None, [MTLTexture, NSUInteger, NSUInteger]),
  ('optimizeContentsForCPUAccess:', None, [MTLTexture]),
  ('optimizeContentsForCPUAccess:slice:level:', None, [MTLTexture, NSUInteger, NSUInteger]),
  ('resetCommandsInBuffer:withRange:', None, [MTLIndirectCommandBuffer, NSRange]),
  ('copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:', None, [MTLIndirectCommandBuffer, NSRange, MTLIndirectCommandBuffer, NSUInteger]),
  ('optimizeIndirectCommandBuffer:withRange:', None, [MTLIndirectCommandBuffer, NSRange]),
  ('sampleCountersInBuffer:atSampleIndex:withBarrier:', None, [MTLCounterSampleBuffer, NSUInteger, BOOL]),
  ('resolveCounters:inRange:destinationBuffer:destinationOffset:', None, [MTLCounterSampleBuffer, NSRange, MTLBuffer, NSUInteger]),
]
class MTLRenderCommandEncoder(objc.Spec): pass
class MTLRenderPassDescriptor(objc.Spec): pass
@c.record
class MTLSamplePosition(c.Struct):
  SIZE = 8
  x: float
  y: float
MTLSamplePosition.register_fields([('x', ctypes.c_float, 0), ('y', ctypes.c_float, 4)])
class MTLRenderPassColorAttachmentDescriptorArray(objc.Spec): pass
class MTLRenderPassColorAttachmentDescriptor(objc.Spec): pass
@c.record
class MTLClearColor(c.Struct):
  SIZE = 32
  red: float
  green: float
  blue: float
  alpha: float
MTLClearColor.register_fields([('red', ctypes.c_double, 0), ('green', ctypes.c_double, 8), ('blue', ctypes.c_double, 16), ('alpha', ctypes.c_double, 24)])
class MTLRenderPassAttachmentDescriptor(objc.Spec): pass
enum_MTLLoadAction: dict[int, str] = {(MTLLoadActionDontCare:=0): 'MTLLoadActionDontCare', (MTLLoadActionLoad:=1): 'MTLLoadActionLoad', (MTLLoadActionClear:=2): 'MTLLoadActionClear'}
MTLLoadAction: TypeAlias = NSUInteger
enum_MTLStoreAction: dict[int, str] = {(MTLStoreActionDontCare:=0): 'MTLStoreActionDontCare', (MTLStoreActionStore:=1): 'MTLStoreActionStore', (MTLStoreActionMultisampleResolve:=2): 'MTLStoreActionMultisampleResolve', (MTLStoreActionStoreAndMultisampleResolve:=3): 'MTLStoreActionStoreAndMultisampleResolve', (MTLStoreActionUnknown:=4): 'MTLStoreActionUnknown', (MTLStoreActionCustomSampleDepthStore:=5): 'MTLStoreActionCustomSampleDepthStore'}
MTLStoreAction: TypeAlias = NSUInteger
enum_MTLStoreActionOptions: dict[int, str] = {(MTLStoreActionOptionNone:=0): 'MTLStoreActionOptionNone', (MTLStoreActionOptionCustomSamplePositions:=1): 'MTLStoreActionOptionCustomSamplePositions'}
MTLStoreActionOptions: TypeAlias = NSUInteger
MTLRenderPassAttachmentDescriptor._bases_ = [NSObject]
MTLRenderPassAttachmentDescriptor._methods_ = [
  ('texture', MTLTexture, []),
  ('setTexture:', None, [MTLTexture]),
  ('level', NSUInteger, []),
  ('setLevel:', None, [NSUInteger]),
  ('slice', NSUInteger, []),
  ('setSlice:', None, [NSUInteger]),
  ('depthPlane', NSUInteger, []),
  ('setDepthPlane:', None, [NSUInteger]),
  ('resolveTexture', MTLTexture, []),
  ('setResolveTexture:', None, [MTLTexture]),
  ('resolveLevel', NSUInteger, []),
  ('setResolveLevel:', None, [NSUInteger]),
  ('resolveSlice', NSUInteger, []),
  ('setResolveSlice:', None, [NSUInteger]),
  ('resolveDepthPlane', NSUInteger, []),
  ('setResolveDepthPlane:', None, [NSUInteger]),
  ('loadAction', MTLLoadAction, []),
  ('setLoadAction:', None, [MTLLoadAction]),
  ('storeAction', MTLStoreAction, []),
  ('setStoreAction:', None, [MTLStoreAction]),
  ('storeActionOptions', MTLStoreActionOptions, []),
  ('setStoreActionOptions:', None, [MTLStoreActionOptions]),
]
MTLRenderPassColorAttachmentDescriptor._bases_ = [MTLRenderPassAttachmentDescriptor]
MTLRenderPassColorAttachmentDescriptor._methods_ = [
  ('clearColor', MTLClearColor, []),
  ('setClearColor:', None, [MTLClearColor]),
]
MTLRenderPassColorAttachmentDescriptorArray._bases_ = [NSObject]
MTLRenderPassColorAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLRenderPassColorAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLRenderPassColorAttachmentDescriptor, NSUInteger]),
]
class MTLRenderPassDepthAttachmentDescriptor(objc.Spec): pass
enum_MTLMultisampleDepthResolveFilter: dict[int, str] = {(MTLMultisampleDepthResolveFilterSample0:=0): 'MTLMultisampleDepthResolveFilterSample0', (MTLMultisampleDepthResolveFilterMin:=1): 'MTLMultisampleDepthResolveFilterMin', (MTLMultisampleDepthResolveFilterMax:=2): 'MTLMultisampleDepthResolveFilterMax'}
MTLMultisampleDepthResolveFilter: TypeAlias = NSUInteger
MTLRenderPassDepthAttachmentDescriptor._bases_ = [MTLRenderPassAttachmentDescriptor]
MTLRenderPassDepthAttachmentDescriptor._methods_ = [
  ('clearDepth', ctypes.c_double, []),
  ('setClearDepth:', None, [ctypes.c_double]),
  ('depthResolveFilter', MTLMultisampleDepthResolveFilter, []),
  ('setDepthResolveFilter:', None, [MTLMultisampleDepthResolveFilter]),
]
class MTLRenderPassStencilAttachmentDescriptor(objc.Spec): pass
enum_MTLMultisampleStencilResolveFilter: dict[int, str] = {(MTLMultisampleStencilResolveFilterSample0:=0): 'MTLMultisampleStencilResolveFilterSample0', (MTLMultisampleStencilResolveFilterDepthResolvedSample:=1): 'MTLMultisampleStencilResolveFilterDepthResolvedSample'}
MTLMultisampleStencilResolveFilter: TypeAlias = NSUInteger
MTLRenderPassStencilAttachmentDescriptor._bases_ = [MTLRenderPassAttachmentDescriptor]
MTLRenderPassStencilAttachmentDescriptor._methods_ = [
  ('clearStencil', uint32_t, []),
  ('setClearStencil:', None, [uint32_t]),
  ('stencilResolveFilter', MTLMultisampleStencilResolveFilter, []),
  ('setStencilResolveFilter:', None, [MTLMultisampleStencilResolveFilter]),
]
class MTLRasterizationRateMap(objc.Spec): pass
class MTLRenderPassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLRenderPassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLRenderPassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLRenderPassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfVertexSampleIndex', NSUInteger, []),
  ('setStartOfVertexSampleIndex:', None, [NSUInteger]),
  ('endOfVertexSampleIndex', NSUInteger, []),
  ('setEndOfVertexSampleIndex:', None, [NSUInteger]),
  ('startOfFragmentSampleIndex', NSUInteger, []),
  ('setStartOfFragmentSampleIndex:', None, [NSUInteger]),
  ('endOfFragmentSampleIndex', NSUInteger, []),
  ('setEndOfFragmentSampleIndex:', None, [NSUInteger]),
]
MTLRenderPassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLRenderPassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLRenderPassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLRenderPassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLRenderPassDescriptor._bases_ = [NSObject]
MTLRenderPassDescriptor._methods_ = [
  ('setSamplePositions:count:', None, [c.POINTER[MTLSamplePosition], NSUInteger]),
  ('getSamplePositions:count:', NSUInteger, [c.POINTER[MTLSamplePosition], NSUInteger]),
  ('colorAttachments', MTLRenderPassColorAttachmentDescriptorArray, []),
  ('depthAttachment', MTLRenderPassDepthAttachmentDescriptor, []),
  ('setDepthAttachment:', None, [MTLRenderPassDepthAttachmentDescriptor]),
  ('stencilAttachment', MTLRenderPassStencilAttachmentDescriptor, []),
  ('setStencilAttachment:', None, [MTLRenderPassStencilAttachmentDescriptor]),
  ('visibilityResultBuffer', MTLBuffer, []),
  ('setVisibilityResultBuffer:', None, [MTLBuffer]),
  ('renderTargetArrayLength', NSUInteger, []),
  ('setRenderTargetArrayLength:', None, [NSUInteger]),
  ('imageblockSampleLength', NSUInteger, []),
  ('setImageblockSampleLength:', None, [NSUInteger]),
  ('threadgroupMemoryLength', NSUInteger, []),
  ('setThreadgroupMemoryLength:', None, [NSUInteger]),
  ('tileWidth', NSUInteger, []),
  ('setTileWidth:', None, [NSUInteger]),
  ('tileHeight', NSUInteger, []),
  ('setTileHeight:', None, [NSUInteger]),
  ('defaultRasterSampleCount', NSUInteger, []),
  ('setDefaultRasterSampleCount:', None, [NSUInteger]),
  ('renderTargetWidth', NSUInteger, []),
  ('setRenderTargetWidth:', None, [NSUInteger]),
  ('renderTargetHeight', NSUInteger, []),
  ('setRenderTargetHeight:', None, [NSUInteger]),
  ('rasterizationRateMap', MTLRasterizationRateMap, []),
  ('setRasterizationRateMap:', None, [MTLRasterizationRateMap]),
  ('sampleBufferAttachments', MTLRenderPassSampleBufferAttachmentDescriptorArray, []),
]
MTLRenderPassDescriptor._classmethods_ = [
  ('renderPassDescriptor', MTLRenderPassDescriptor, []),
]
class MTLComputePassDescriptor(objc.Spec): pass
class MTLComputePassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLComputePassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLComputePassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLComputePassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfEncoderSampleIndex', NSUInteger, []),
  ('setStartOfEncoderSampleIndex:', None, [NSUInteger]),
  ('endOfEncoderSampleIndex', NSUInteger, []),
  ('setEndOfEncoderSampleIndex:', None, [NSUInteger]),
]
MTLComputePassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLComputePassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLComputePassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLComputePassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLComputePassDescriptor._bases_ = [NSObject]
MTLComputePassDescriptor._methods_ = [
  ('dispatchType', MTLDispatchType, []),
  ('setDispatchType:', None, [MTLDispatchType]),
  ('sampleBufferAttachments', MTLComputePassSampleBufferAttachmentDescriptorArray, []),
]
MTLComputePassDescriptor._classmethods_ = [
  ('computePassDescriptor', MTLComputePassDescriptor, []),
]
class MTLBlitPassDescriptor(objc.Spec): pass
class MTLBlitPassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLBlitPassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLBlitPassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLBlitPassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfEncoderSampleIndex', NSUInteger, []),
  ('setStartOfEncoderSampleIndex:', None, [NSUInteger]),
  ('endOfEncoderSampleIndex', NSUInteger, []),
  ('setEndOfEncoderSampleIndex:', None, [NSUInteger]),
]
MTLBlitPassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLBlitPassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLBlitPassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLBlitPassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLBlitPassDescriptor._bases_ = [NSObject]
MTLBlitPassDescriptor._methods_ = [
  ('sampleBufferAttachments', MTLBlitPassSampleBufferAttachmentDescriptorArray, []),
]
MTLBlitPassDescriptor._classmethods_ = [
  ('blitPassDescriptor', MTLBlitPassDescriptor, []),
]
class MTLEvent(objc.Spec): pass
class MTLParallelRenderCommandEncoder(objc.Spec): pass
class MTLResourceStateCommandEncoder(objc.Spec): pass
enum_MTLSparseTextureMappingMode: dict[int, str] = {(MTLSparseTextureMappingModeMap:=0): 'MTLSparseTextureMappingModeMap', (MTLSparseTextureMappingModeUnmap:=1): 'MTLSparseTextureMappingModeUnmap'}
MTLSparseTextureMappingMode: TypeAlias = NSUInteger
MTLResourceStateCommandEncoder._bases_ = [MTLCommandEncoder]
MTLResourceStateCommandEncoder._methods_ = [
  ('updateTextureMappings:mode:regions:mipLevels:slices:numRegions:', None, [MTLTexture, MTLSparseTextureMappingMode, c.POINTER[MTLRegion], c.POINTER[NSUInteger], c.POINTER[NSUInteger], NSUInteger]),
  ('updateTextureMapping:mode:region:mipLevel:slice:', None, [MTLTexture, MTLSparseTextureMappingMode, MTLRegion, NSUInteger, NSUInteger]),
  ('updateTextureMapping:mode:indirectBuffer:indirectBufferOffset:', None, [MTLTexture, MTLSparseTextureMappingMode, MTLBuffer, NSUInteger]),
  ('updateFence:', None, [MTLFence]),
  ('waitForFence:', None, [MTLFence]),
  ('moveTextureMappingsFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:', None, [MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLSize, MTLTexture, NSUInteger, NSUInteger, MTLOrigin]),
]
class MTLResourceStatePassDescriptor(objc.Spec): pass
class MTLResourceStatePassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLResourceStatePassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLResourceStatePassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLResourceStatePassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfEncoderSampleIndex', NSUInteger, []),
  ('setStartOfEncoderSampleIndex:', None, [NSUInteger]),
  ('endOfEncoderSampleIndex', NSUInteger, []),
  ('setEndOfEncoderSampleIndex:', None, [NSUInteger]),
]
MTLResourceStatePassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLResourceStatePassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLResourceStatePassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLResourceStatePassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLResourceStatePassDescriptor._bases_ = [NSObject]
MTLResourceStatePassDescriptor._methods_ = [
  ('sampleBufferAttachments', MTLResourceStatePassSampleBufferAttachmentDescriptorArray, []),
]
MTLResourceStatePassDescriptor._classmethods_ = [
  ('resourceStatePassDescriptor', MTLResourceStatePassDescriptor, []),
]
class MTLAccelerationStructureCommandEncoder(objc.Spec): pass
class MTLAccelerationStructurePassDescriptor(objc.Spec): pass
class MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLAccelerationStructurePassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLAccelerationStructurePassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLAccelerationStructurePassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfEncoderSampleIndex', NSUInteger, []),
  ('setStartOfEncoderSampleIndex:', None, [NSUInteger]),
  ('endOfEncoderSampleIndex', NSUInteger, []),
  ('setEndOfEncoderSampleIndex:', None, [NSUInteger]),
]
MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLAccelerationStructurePassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLAccelerationStructurePassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLAccelerationStructurePassDescriptor._bases_ = [NSObject]
MTLAccelerationStructurePassDescriptor._methods_ = [
  ('sampleBufferAttachments', MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray, []),
]
MTLAccelerationStructurePassDescriptor._classmethods_ = [
  ('accelerationStructurePassDescriptor', MTLAccelerationStructurePassDescriptor, []),
]
enum_MTLCommandBufferErrorOption: dict[int, str] = {(MTLCommandBufferErrorOptionNone:=0): 'MTLCommandBufferErrorOptionNone', (MTLCommandBufferErrorOptionEncoderExecutionStatus:=1): 'MTLCommandBufferErrorOptionEncoderExecutionStatus'}
MTLCommandBufferErrorOption: TypeAlias = NSUInteger
class MTLLogContainer(objc.Spec): pass
enum_MTLCommandBufferStatus: dict[int, str] = {(MTLCommandBufferStatusNotEnqueued:=0): 'MTLCommandBufferStatusNotEnqueued', (MTLCommandBufferStatusEnqueued:=1): 'MTLCommandBufferStatusEnqueued', (MTLCommandBufferStatusCommitted:=2): 'MTLCommandBufferStatusCommitted', (MTLCommandBufferStatusScheduled:=3): 'MTLCommandBufferStatusScheduled', (MTLCommandBufferStatusCompleted:=4): 'MTLCommandBufferStatusCompleted', (MTLCommandBufferStatusError:=5): 'MTLCommandBufferStatusError'}
MTLCommandBufferStatus: TypeAlias = NSUInteger
class NSError(objc.Spec): pass
NSErrorDomain: TypeAlias = NSString
NSError._bases_ = [NSObject]
NSError._methods_ = [
  ('domain', NSErrorDomain, []),
  ('code', NSInteger, []),
  ('localizedDescription', NSString, []),
  ('localizedFailureReason', NSString, []),
  ('localizedRecoverySuggestion', NSString, []),
  ('recoveryAttempter', objc.id_, []),
  ('helpAnchor', NSString, []),
]
MTLCommandBuffer._bases_ = [NSObject]
MTLCommandBuffer._methods_ = [
  ('enqueue', None, []),
  ('commit', None, []),
  ('presentDrawable:', None, [MTLDrawable]),
  ('presentDrawable:atTime:', None, [MTLDrawable, CFTimeInterval]),
  ('presentDrawable:afterMinimumDuration:', None, [MTLDrawable, CFTimeInterval]),
  ('waitUntilScheduled', None, []),
  ('waitUntilCompleted', None, []),
  ('blitCommandEncoder', MTLBlitCommandEncoder, []),
  ('renderCommandEncoderWithDescriptor:', MTLRenderCommandEncoder, [MTLRenderPassDescriptor]),
  ('computeCommandEncoderWithDescriptor:', MTLComputeCommandEncoder, [MTLComputePassDescriptor]),
  ('blitCommandEncoderWithDescriptor:', MTLBlitCommandEncoder, [MTLBlitPassDescriptor]),
  ('computeCommandEncoder', MTLComputeCommandEncoder, []),
  ('computeCommandEncoderWithDispatchType:', MTLComputeCommandEncoder, [MTLDispatchType]),
  ('encodeWaitForEvent:value:', None, [MTLEvent, uint64_t]),
  ('encodeSignalEvent:value:', None, [MTLEvent, uint64_t]),
  ('parallelRenderCommandEncoderWithDescriptor:', MTLParallelRenderCommandEncoder, [MTLRenderPassDescriptor]),
  ('resourceStateCommandEncoder', MTLResourceStateCommandEncoder, []),
  ('resourceStateCommandEncoderWithDescriptor:', MTLResourceStateCommandEncoder, [MTLResourceStatePassDescriptor]),
  ('accelerationStructureCommandEncoder', MTLAccelerationStructureCommandEncoder, []),
  ('accelerationStructureCommandEncoderWithDescriptor:', MTLAccelerationStructureCommandEncoder, [MTLAccelerationStructurePassDescriptor]),
  ('pushDebugGroup:', None, [NSString]),
  ('popDebugGroup', None, []),
  ('device', MTLDevice, []),
  ('commandQueue', MTLCommandQueue, []),
  ('retainedReferences', BOOL, []),
  ('errorOptions', MTLCommandBufferErrorOption, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('kernelStartTime', CFTimeInterval, []),
  ('kernelEndTime', CFTimeInterval, []),
  ('logs', MTLLogContainer, []),
  ('GPUStartTime', CFTimeInterval, []),
  ('GPUEndTime', CFTimeInterval, []),
  ('status', MTLCommandBufferStatus, []),
  ('error', NSError, []),
]
class MTLCommandBufferDescriptor(objc.Spec): pass
MTLCommandBufferDescriptor._bases_ = [NSObject]
MTLCommandBufferDescriptor._methods_ = [
  ('retainedReferences', BOOL, []),
  ('setRetainedReferences:', None, [BOOL]),
  ('errorOptions', MTLCommandBufferErrorOption, []),
  ('setErrorOptions:', None, [MTLCommandBufferErrorOption]),
]
MTLCommandQueue._bases_ = [NSObject]
MTLCommandQueue._methods_ = [
  ('commandBuffer', MTLCommandBuffer, []),
  ('commandBufferWithDescriptor:', MTLCommandBuffer, [MTLCommandBufferDescriptor]),
  ('commandBufferWithUnretainedReferences', MTLCommandBuffer, []),
  ('insertDebugCaptureBoundary', None, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
]
enum_MTLIOCompressionMethod: dict[int, str] = {(MTLIOCompressionMethodZlib:=0): 'MTLIOCompressionMethodZlib', (MTLIOCompressionMethodLZFSE:=1): 'MTLIOCompressionMethodLZFSE', (MTLIOCompressionMethodLZ4:=2): 'MTLIOCompressionMethodLZ4', (MTLIOCompressionMethodLZMA:=3): 'MTLIOCompressionMethodLZMA', (MTLIOCompressionMethodLZBitmap:=4): 'MTLIOCompressionMethodLZBitmap'}
MTLIOCompressionMethod: TypeAlias = NSInteger
@dll.bind(MTLDevice)
def MTLCreateSystemDefaultDevice() -> MTLDevice: ...
MTLCreateSystemDefaultDevice = objc.returns_retained(MTLCreateSystemDefaultDevice)
MTLDeviceNotificationName: TypeAlias = NSString
try: MTLDeviceWasAddedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasAddedNotification') # type: ignore
except (ValueError,AttributeError): pass
try: MTLDeviceRemovalRequestedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceRemovalRequestedNotification') # type: ignore
except (ValueError,AttributeError): pass
try: MTLDeviceWasRemovedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasRemovedNotification') # type: ignore
except (ValueError,AttributeError): pass
@dll.bind(None, NSObject)
def MTLRemoveDeviceObserver(observer:NSObject) -> None: ...
enum_MTLFeatureSet: dict[int, str] = {(MTLFeatureSet_iOS_GPUFamily1_v1:=0): 'MTLFeatureSet_iOS_GPUFamily1_v1', (MTLFeatureSet_iOS_GPUFamily2_v1:=1): 'MTLFeatureSet_iOS_GPUFamily2_v1', (MTLFeatureSet_iOS_GPUFamily1_v2:=2): 'MTLFeatureSet_iOS_GPUFamily1_v2', (MTLFeatureSet_iOS_GPUFamily2_v2:=3): 'MTLFeatureSet_iOS_GPUFamily2_v2', (MTLFeatureSet_iOS_GPUFamily3_v1:=4): 'MTLFeatureSet_iOS_GPUFamily3_v1', (MTLFeatureSet_iOS_GPUFamily1_v3:=5): 'MTLFeatureSet_iOS_GPUFamily1_v3', (MTLFeatureSet_iOS_GPUFamily2_v3:=6): 'MTLFeatureSet_iOS_GPUFamily2_v3', (MTLFeatureSet_iOS_GPUFamily3_v2:=7): 'MTLFeatureSet_iOS_GPUFamily3_v2', (MTLFeatureSet_iOS_GPUFamily1_v4:=8): 'MTLFeatureSet_iOS_GPUFamily1_v4', (MTLFeatureSet_iOS_GPUFamily2_v4:=9): 'MTLFeatureSet_iOS_GPUFamily2_v4', (MTLFeatureSet_iOS_GPUFamily3_v3:=10): 'MTLFeatureSet_iOS_GPUFamily3_v3', (MTLFeatureSet_iOS_GPUFamily4_v1:=11): 'MTLFeatureSet_iOS_GPUFamily4_v1', (MTLFeatureSet_iOS_GPUFamily1_v5:=12): 'MTLFeatureSet_iOS_GPUFamily1_v5', (MTLFeatureSet_iOS_GPUFamily2_v5:=13): 'MTLFeatureSet_iOS_GPUFamily2_v5', (MTLFeatureSet_iOS_GPUFamily3_v4:=14): 'MTLFeatureSet_iOS_GPUFamily3_v4', (MTLFeatureSet_iOS_GPUFamily4_v2:=15): 'MTLFeatureSet_iOS_GPUFamily4_v2', (MTLFeatureSet_iOS_GPUFamily5_v1:=16): 'MTLFeatureSet_iOS_GPUFamily5_v1', (MTLFeatureSet_macOS_GPUFamily1_v1:=10000): 'MTLFeatureSet_macOS_GPUFamily1_v1', (MTLFeatureSet_OSX_GPUFamily1_v1:=10000): 'MTLFeatureSet_OSX_GPUFamily1_v1', (MTLFeatureSet_macOS_GPUFamily1_v2:=10001): 'MTLFeatureSet_macOS_GPUFamily1_v2', (MTLFeatureSet_OSX_GPUFamily1_v2:=10001): 'MTLFeatureSet_OSX_GPUFamily1_v2', (MTLFeatureSet_macOS_ReadWriteTextureTier2:=10002): 'MTLFeatureSet_macOS_ReadWriteTextureTier2', (MTLFeatureSet_OSX_ReadWriteTextureTier2:=10002): 'MTLFeatureSet_OSX_ReadWriteTextureTier2', (MTLFeatureSet_macOS_GPUFamily1_v3:=10003): 'MTLFeatureSet_macOS_GPUFamily1_v3', (MTLFeatureSet_macOS_GPUFamily1_v4:=10004): 'MTLFeatureSet_macOS_GPUFamily1_v4', (MTLFeatureSet_macOS_GPUFamily2_v1:=10005): 'MTLFeatureSet_macOS_GPUFamily2_v1', (MTLFeatureSet_tvOS_GPUFamily1_v1:=30000): 'MTLFeatureSet_tvOS_GPUFamily1_v1', (MTLFeatureSet_TVOS_GPUFamily1_v1:=30000): 'MTLFeatureSet_TVOS_GPUFamily1_v1', (MTLFeatureSet_tvOS_GPUFamily1_v2:=30001): 'MTLFeatureSet_tvOS_GPUFamily1_v2', (MTLFeatureSet_tvOS_GPUFamily1_v3:=30002): 'MTLFeatureSet_tvOS_GPUFamily1_v3', (MTLFeatureSet_tvOS_GPUFamily2_v1:=30003): 'MTLFeatureSet_tvOS_GPUFamily2_v1', (MTLFeatureSet_tvOS_GPUFamily1_v4:=30004): 'MTLFeatureSet_tvOS_GPUFamily1_v4', (MTLFeatureSet_tvOS_GPUFamily2_v2:=30005): 'MTLFeatureSet_tvOS_GPUFamily2_v2'}
MTLFeatureSet: TypeAlias = NSUInteger
enum_MTLGPUFamily: dict[int, str] = {(MTLGPUFamilyApple1:=1001): 'MTLGPUFamilyApple1', (MTLGPUFamilyApple2:=1002): 'MTLGPUFamilyApple2', (MTLGPUFamilyApple3:=1003): 'MTLGPUFamilyApple3', (MTLGPUFamilyApple4:=1004): 'MTLGPUFamilyApple4', (MTLGPUFamilyApple5:=1005): 'MTLGPUFamilyApple5', (MTLGPUFamilyApple6:=1006): 'MTLGPUFamilyApple6', (MTLGPUFamilyApple7:=1007): 'MTLGPUFamilyApple7', (MTLGPUFamilyApple8:=1008): 'MTLGPUFamilyApple8', (MTLGPUFamilyApple9:=1009): 'MTLGPUFamilyApple9', (MTLGPUFamilyMac1:=2001): 'MTLGPUFamilyMac1', (MTLGPUFamilyMac2:=2002): 'MTLGPUFamilyMac2', (MTLGPUFamilyCommon1:=3001): 'MTLGPUFamilyCommon1', (MTLGPUFamilyCommon2:=3002): 'MTLGPUFamilyCommon2', (MTLGPUFamilyCommon3:=3003): 'MTLGPUFamilyCommon3', (MTLGPUFamilyMacCatalyst1:=4001): 'MTLGPUFamilyMacCatalyst1', (MTLGPUFamilyMacCatalyst2:=4002): 'MTLGPUFamilyMacCatalyst2', (MTLGPUFamilyMetal3:=5001): 'MTLGPUFamilyMetal3'}
MTLGPUFamily: TypeAlias = NSInteger
enum_MTLDeviceLocation: dict[int, str] = {(MTLDeviceLocationBuiltIn:=0): 'MTLDeviceLocationBuiltIn', (MTLDeviceLocationSlot:=1): 'MTLDeviceLocationSlot', (MTLDeviceLocationExternal:=2): 'MTLDeviceLocationExternal', (MTLDeviceLocationUnspecified:=-1): 'MTLDeviceLocationUnspecified'}
MTLDeviceLocation: TypeAlias = NSUInteger
enum_MTLPipelineOption: dict[int, str] = {(MTLPipelineOptionNone:=0): 'MTLPipelineOptionNone', (MTLPipelineOptionArgumentInfo:=1): 'MTLPipelineOptionArgumentInfo', (MTLPipelineOptionBufferTypeInfo:=2): 'MTLPipelineOptionBufferTypeInfo', (MTLPipelineOptionFailOnBinaryArchiveMiss:=4): 'MTLPipelineOptionFailOnBinaryArchiveMiss'}
MTLPipelineOption: TypeAlias = NSUInteger
enum_MTLReadWriteTextureTier: dict[int, str] = {(MTLReadWriteTextureTierNone:=0): 'MTLReadWriteTextureTierNone', (MTLReadWriteTextureTier1:=1): 'MTLReadWriteTextureTier1', (MTLReadWriteTextureTier2:=2): 'MTLReadWriteTextureTier2'}
MTLReadWriteTextureTier: TypeAlias = NSUInteger
enum_MTLArgumentBuffersTier: dict[int, str] = {(MTLArgumentBuffersTier1:=0): 'MTLArgumentBuffersTier1', (MTLArgumentBuffersTier2:=1): 'MTLArgumentBuffersTier2'}
MTLArgumentBuffersTier: TypeAlias = NSUInteger
enum_MTLSparseTextureRegionAlignmentMode: dict[int, str] = {(MTLSparseTextureRegionAlignmentModeOutward:=0): 'MTLSparseTextureRegionAlignmentModeOutward', (MTLSparseTextureRegionAlignmentModeInward:=1): 'MTLSparseTextureRegionAlignmentModeInward'}
MTLSparseTextureRegionAlignmentMode: TypeAlias = NSUInteger
enum_MTLSparsePageSize: dict[int, str] = {(MTLSparsePageSize16:=101): 'MTLSparsePageSize16', (MTLSparsePageSize64:=102): 'MTLSparsePageSize64', (MTLSparsePageSize256:=103): 'MTLSparsePageSize256'}
MTLSparsePageSize: TypeAlias = NSInteger
@c.record
class MTLAccelerationStructureSizes(c.Struct):
  SIZE = 24
  accelerationStructureSize: int
  buildScratchBufferSize: int
  refitScratchBufferSize: int
MTLAccelerationStructureSizes.register_fields([('accelerationStructureSize', NSUInteger, 0), ('buildScratchBufferSize', NSUInteger, 8), ('refitScratchBufferSize', NSUInteger, 16)])
enum_MTLCounterSamplingPoint: dict[int, str] = {(MTLCounterSamplingPointAtStageBoundary:=0): 'MTLCounterSamplingPointAtStageBoundary', (MTLCounterSamplingPointAtDrawBoundary:=1): 'MTLCounterSamplingPointAtDrawBoundary', (MTLCounterSamplingPointAtDispatchBoundary:=2): 'MTLCounterSamplingPointAtDispatchBoundary', (MTLCounterSamplingPointAtTileDispatchBoundary:=3): 'MTLCounterSamplingPointAtTileDispatchBoundary', (MTLCounterSamplingPointAtBlitBoundary:=4): 'MTLCounterSamplingPointAtBlitBoundary'}
MTLCounterSamplingPoint: TypeAlias = NSUInteger
@c.record
class MTLSizeAndAlign(c.Struct):
  SIZE = 16
  size: int
  align: int
MTLSizeAndAlign.register_fields([('size', NSUInteger, 0), ('align', NSUInteger, 8)])
class MTLRenderPipelineReflection(objc.Spec): pass
class MTLArgumentDescriptor(objc.Spec): pass
MTLArgumentDescriptor._bases_ = [NSObject]
MTLArgumentDescriptor._methods_ = [
  ('dataType', MTLDataType, []),
  ('setDataType:', None, [MTLDataType]),
  ('index', NSUInteger, []),
  ('setIndex:', None, [NSUInteger]),
  ('arrayLength', NSUInteger, []),
  ('setArrayLength:', None, [NSUInteger]),
  ('access', MTLBindingAccess, []),
  ('setAccess:', None, [MTLBindingAccess]),
  ('textureType', MTLTextureType, []),
  ('setTextureType:', None, [MTLTextureType]),
  ('constantBlockAlignment', NSUInteger, []),
  ('setConstantBlockAlignment:', None, [NSUInteger]),
]
MTLArgumentDescriptor._classmethods_ = [
  ('argumentDescriptor', MTLArgumentDescriptor, []),
]
class MTLArchitecture(objc.Spec): pass
MTLArchitecture._bases_ = [NSObject]
MTLArchitecture._methods_ = [
  ('name', NSString, []),
]
class MTLHeapDescriptor(objc.Spec): pass
class MTLDepthStencilState(objc.Spec): pass
class MTLDepthStencilDescriptor(objc.Spec): pass
class struct___IOSurface(c.Struct): pass
IOSurfaceRef: TypeAlias = c.POINTER[struct___IOSurface]
class MTLSharedTextureHandle(objc.Spec): pass
MTLSharedTextureHandle._bases_ = [NSObject]
MTLSharedTextureHandle._methods_ = [
  ('device', MTLDevice, []),
  ('label', NSString, []),
]
class MTLSamplerDescriptor(objc.Spec): pass
class MTLLibrary(objc.Spec): pass
class MTLFunctionConstantValues(objc.Spec): pass
MTLFunctionConstantValues._bases_ = [NSObject]
MTLFunctionConstantValues._methods_ = [
  ('setConstantValue:type:atIndex:', None, [ctypes.c_void_p, MTLDataType, NSUInteger]),
  ('setConstantValues:type:withRange:', None, [ctypes.c_void_p, MTLDataType, NSRange]),
  ('setConstantValue:type:withName:', None, [ctypes.c_void_p, MTLDataType, NSString]),
  ('reset', None, []),
]
class MTLFunctionDescriptor(objc.Spec): pass
MTLFunctionDescriptor._bases_ = [NSObject]
MTLFunctionDescriptor._methods_ = [
  ('name', NSString, []),
  ('setName:', None, [NSString]),
  ('specializedName', NSString, []),
  ('setSpecializedName:', None, [NSString]),
  ('constantValues', MTLFunctionConstantValues, []),
  ('setConstantValues:', None, [MTLFunctionConstantValues]),
  ('options', MTLFunctionOptions, []),
  ('setOptions:', None, [MTLFunctionOptions]),
]
MTLFunctionDescriptor._classmethods_ = [
  ('functionDescriptor', MTLFunctionDescriptor, []),
]
class MTLIntersectionFunctionDescriptor(objc.Spec): pass
enum_MTLLibraryType: dict[int, str] = {(MTLLibraryTypeExecutable:=0): 'MTLLibraryTypeExecutable', (MTLLibraryTypeDynamic:=1): 'MTLLibraryTypeDynamic'}
MTLLibraryType: TypeAlias = NSInteger
MTLLibrary._bases_ = [NSObject]
MTLLibrary._methods_ = [
  ('newFunctionWithName:', MTLFunction, [NSString], True),
  ('newFunctionWithName:constantValues:error:', MTLFunction, [NSString, MTLFunctionConstantValues, c.POINTER[NSError]], True),
  ('newFunctionWithDescriptor:error:', MTLFunction, [MTLFunctionDescriptor, c.POINTER[NSError]], True),
  ('newIntersectionFunctionWithDescriptor:error:', MTLFunction, [MTLIntersectionFunctionDescriptor, c.POINTER[NSError]], True),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
  ('type', MTLLibraryType, []),
  ('installName', NSString, []),
]
class NSBundle(objc.Spec): pass
class NSURL(objc.Spec): pass
NSURLResourceKey: TypeAlias = NSString
enum_NSURLBookmarkCreationOptions: dict[int, str] = {(NSURLBookmarkCreationPreferFileIDResolution:=256): 'NSURLBookmarkCreationPreferFileIDResolution', (NSURLBookmarkCreationMinimalBookmark:=512): 'NSURLBookmarkCreationMinimalBookmark', (NSURLBookmarkCreationSuitableForBookmarkFile:=1024): 'NSURLBookmarkCreationSuitableForBookmarkFile', (NSURLBookmarkCreationWithSecurityScope:=2048): 'NSURLBookmarkCreationWithSecurityScope', (NSURLBookmarkCreationSecurityScopeAllowOnlyReadAccess:=4096): 'NSURLBookmarkCreationSecurityScopeAllowOnlyReadAccess', (NSURLBookmarkCreationWithoutImplicitSecurityScope:=536870912): 'NSURLBookmarkCreationWithoutImplicitSecurityScope'}
NSURLBookmarkCreationOptions: TypeAlias = NSUInteger
enum_NSURLBookmarkResolutionOptions: dict[int, str] = {(NSURLBookmarkResolutionWithoutUI:=256): 'NSURLBookmarkResolutionWithoutUI', (NSURLBookmarkResolutionWithoutMounting:=512): 'NSURLBookmarkResolutionWithoutMounting', (NSURLBookmarkResolutionWithSecurityScope:=1024): 'NSURLBookmarkResolutionWithSecurityScope', (NSURLBookmarkResolutionWithoutImplicitStartAccessing:=32768): 'NSURLBookmarkResolutionWithoutImplicitStartAccessing'}
NSURLBookmarkResolutionOptions: TypeAlias = NSUInteger
class NSNumber(objc.Spec): pass
enum_NSComparisonResult: dict[int, str] = {(NSOrderedAscending:=-1): 'NSOrderedAscending', (NSOrderedSame:=0): 'NSOrderedSame', (NSOrderedDescending:=1): 'NSOrderedDescending'}
NSComparisonResult: TypeAlias = NSInteger
class NSValue(objc.Spec): pass
NSValue._bases_ = [NSObject]
NSValue._methods_ = [
  ('getValue:size:', None, [ctypes.c_void_p, NSUInteger]),
  ('initWithBytes:objCType:', 'instancetype', [ctypes.c_void_p, c.POINTER[ctypes.c_char]]),
  ('initWithCoder:', 'instancetype', [NSCoder]),
  ('objCType', c.POINTER[ctypes.c_char], []),
]
NSNumber._bases_ = [NSValue]
NSNumber._methods_ = [
  ('initWithCoder:', 'instancetype', [NSCoder]),
  ('initWithChar:', NSNumber, [ctypes.c_char]),
  ('initWithUnsignedChar:', NSNumber, [ctypes.c_ubyte]),
  ('initWithShort:', NSNumber, [ctypes.c_int16]),
  ('initWithUnsignedShort:', NSNumber, [ctypes.c_uint16]),
  ('initWithInt:', NSNumber, [ctypes.c_int32]),
  ('initWithUnsignedInt:', NSNumber, [ctypes.c_uint32]),
  ('initWithLong:', NSNumber, [ctypes.c_int64]),
  ('initWithUnsignedLong:', NSNumber, [ctypes.c_uint64]),
  ('initWithLongLong:', NSNumber, [ctypes.c_int64]),
  ('initWithUnsignedLongLong:', NSNumber, [ctypes.c_uint64]),
  ('initWithFloat:', NSNumber, [ctypes.c_float]),
  ('initWithDouble:', NSNumber, [ctypes.c_double]),
  ('initWithBool:', NSNumber, [BOOL]),
  ('initWithInteger:', NSNumber, [NSInteger]),
  ('initWithUnsignedInteger:', NSNumber, [NSUInteger]),
  ('compare:', NSComparisonResult, [NSNumber]),
  ('isEqualToNumber:', BOOL, [NSNumber]),
  ('descriptionWithLocale:', NSString, [objc.id_]),
  ('charValue', ctypes.c_char, []),
  ('unsignedCharValue', ctypes.c_ubyte, []),
  ('shortValue', ctypes.c_int16, []),
  ('unsignedShortValue', ctypes.c_uint16, []),
  ('intValue', ctypes.c_int32, []),
  ('unsignedIntValue', ctypes.c_uint32, []),
  ('longValue', ctypes.c_int64, []),
  ('unsignedLongValue', ctypes.c_uint64, []),
  ('longLongValue', ctypes.c_int64, []),
  ('unsignedLongLongValue', ctypes.c_uint64, []),
  ('floatValue', ctypes.c_float, []),
  ('doubleValue', ctypes.c_double, []),
  ('boolValue', BOOL, []),
  ('integerValue', NSInteger, []),
  ('unsignedIntegerValue', NSUInteger, []),
  ('stringValue', NSString, []),
]
NSURLBookmarkFileCreationOptions: TypeAlias = ctypes.c_uint64
NSURL._bases_ = [NSObject]
NSURL._methods_ = [
  ('initWithScheme:host:path:', 'instancetype', [NSString, NSString, NSString]),
  ('initFileURLWithPath:isDirectory:relativeToURL:', 'instancetype', [NSString, BOOL, NSURL]),
  ('initFileURLWithPath:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('initFileURLWithPath:isDirectory:', 'instancetype', [NSString, BOOL]),
  ('initFileURLWithPath:', 'instancetype', [NSString]),
  ('initFileURLWithFileSystemRepresentation:isDirectory:relativeToURL:', 'instancetype', [c.POINTER[ctypes.c_char], BOOL, NSURL]),
  ('initWithString:', 'instancetype', [NSString]),
  ('initWithString:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('initWithString:encodingInvalidCharacters:', 'instancetype', [NSString, BOOL]),
  ('initWithDataRepresentation:relativeToURL:', 'instancetype', [NSData, NSURL]),
  ('initAbsoluteURLWithDataRepresentation:relativeToURL:', 'instancetype', [NSData, NSURL]),
  ('getFileSystemRepresentation:maxLength:', BOOL, [c.POINTER[ctypes.c_char], NSUInteger]),
  ('isFileReferenceURL', BOOL, []),
  ('fileReferenceURL', NSURL, []),
  ('getResourceValue:forKey:error:', BOOL, [c.POINTER[objc.id_], NSURLResourceKey, c.POINTER[NSError]]),
  ('setResourceValue:forKey:error:', BOOL, [objc.id_, NSURLResourceKey, c.POINTER[NSError]]),
  ('removeCachedResourceValueForKey:', None, [NSURLResourceKey]),
  ('removeAllCachedResourceValues', None, []),
  ('setTemporaryResourceValue:forKey:', None, [objc.id_, NSURLResourceKey]),
  ('initByResolvingBookmarkData:options:relativeToURL:bookmarkDataIsStale:error:', 'instancetype', [NSData, NSURLBookmarkResolutionOptions, NSURL, c.POINTER[BOOL], c.POINTER[NSError]]),
  ('startAccessingSecurityScopedResource', BOOL, []),
  ('stopAccessingSecurityScopedResource', None, []),
  ('dataRepresentation', NSData, []),
  ('absoluteString', NSString, []),
  ('relativeString', NSString, []),
  ('baseURL', NSURL, []),
  ('absoluteURL', NSURL, []),
  ('scheme', NSString, []),
  ('resourceSpecifier', NSString, []),
  ('host', NSString, []),
  ('port', NSNumber, []),
  ('user', NSString, []),
  ('password', NSString, []),
  ('path', NSString, []),
  ('fragment', NSString, []),
  ('parameterString', NSString, []),
  ('query', NSString, []),
  ('relativePath', NSString, []),
  ('hasDirectoryPath', BOOL, []),
  ('fileSystemRepresentation', c.POINTER[ctypes.c_char], []),
  ('isFileURL', BOOL, []),
  ('standardizedURL', NSURL, []),
  ('filePathURL', NSURL, []),
]
NSURL._classmethods_ = [
  ('fileURLWithPath:isDirectory:relativeToURL:', NSURL, [NSString, BOOL, NSURL]),
  ('fileURLWithPath:relativeToURL:', NSURL, [NSString, NSURL]),
  ('fileURLWithPath:isDirectory:', NSURL, [NSString, BOOL]),
  ('fileURLWithPath:', NSURL, [NSString]),
  ('fileURLWithFileSystemRepresentation:isDirectory:relativeToURL:', NSURL, [c.POINTER[ctypes.c_char], BOOL, NSURL]),
  ('URLWithString:', 'instancetype', [NSString]),
  ('URLWithString:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('URLWithString:encodingInvalidCharacters:', 'instancetype', [NSString, BOOL]),
  ('URLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('absoluteURLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('URLByResolvingBookmarkData:options:relativeToURL:bookmarkDataIsStale:error:', 'instancetype', [NSData, NSURLBookmarkResolutionOptions, NSURL, c.POINTER[BOOL], c.POINTER[NSError]]),
  ('writeBookmarkData:toURL:options:error:', BOOL, [NSData, NSURL, NSURLBookmarkFileCreationOptions, c.POINTER[NSError]]),
  ('bookmarkDataWithContentsOfURL:error:', NSData, [NSURL, c.POINTER[NSError]]),
  ('URLByResolvingAliasFileAtURL:options:error:', 'instancetype', [NSURL, NSURLBookmarkResolutionOptions, c.POINTER[NSError]]),
]
class NSAttributedString(objc.Spec): pass
NSAttributedString._bases_ = [NSObject]
NSAttributedString._methods_ = [
  ('string', NSString, []),
]
NSBundle._bases_ = [NSObject]
NSBundle._methods_ = [
  ('initWithPath:', 'instancetype', [NSString]),
  ('initWithURL:', 'instancetype', [NSURL]),
  ('load', BOOL, []),
  ('unload', BOOL, []),
  ('preflightAndReturnError:', BOOL, [c.POINTER[NSError]]),
  ('loadAndReturnError:', BOOL, [c.POINTER[NSError]]),
  ('URLForAuxiliaryExecutable:', NSURL, [NSString]),
  ('pathForAuxiliaryExecutable:', NSString, [NSString]),
  ('URLForResource:withExtension:', NSURL, [NSString, NSString]),
  ('URLForResource:withExtension:subdirectory:', NSURL, [NSString, NSString, NSString]),
  ('URLForResource:withExtension:subdirectory:localization:', NSURL, [NSString, NSString, NSString, NSString]),
  ('pathForResource:ofType:', NSString, [NSString, NSString]),
  ('pathForResource:ofType:inDirectory:', NSString, [NSString, NSString, NSString]),
  ('pathForResource:ofType:inDirectory:forLocalization:', NSString, [NSString, NSString, NSString, NSString]),
  ('localizedStringForKey:value:table:', NSString, [NSString, NSString, NSString]),
  ('localizedAttributedStringForKey:value:table:', NSAttributedString, [NSString, NSString, NSString]),
  ('objectForInfoDictionaryKey:', objc.id_, [NSString]),
  ('isLoaded', BOOL, []),
  ('bundleURL', NSURL, []),
  ('resourceURL', NSURL, []),
  ('executableURL', NSURL, []),
  ('privateFrameworksURL', NSURL, []),
  ('sharedFrameworksURL', NSURL, []),
  ('sharedSupportURL', NSURL, []),
  ('builtInPlugInsURL', NSURL, []),
  ('appStoreReceiptURL', NSURL, []),
  ('bundlePath', NSString, []),
  ('resourcePath', NSString, []),
  ('executablePath', NSString, []),
  ('privateFrameworksPath', NSString, []),
  ('sharedFrameworksPath', NSString, []),
  ('sharedSupportPath', NSString, []),
  ('builtInPlugInsPath', NSString, []),
  ('bundleIdentifier', NSString, []),
  ('developmentLocalization', NSString, []),
]
NSBundle._classmethods_ = [
  ('bundleWithPath:', 'instancetype', [NSString]),
  ('bundleWithURL:', 'instancetype', [NSURL]),
  ('bundleWithIdentifier:', NSBundle, [NSString]),
  ('URLForResource:withExtension:subdirectory:inBundleWithURL:', NSURL, [NSString, NSString, NSString, NSURL]),
  ('pathForResource:ofType:inDirectory:', NSString, [NSString, NSString, NSString]),
  ('mainBundle', NSBundle, []),
]
class MTLCompileOptions(objc.Spec): pass
enum_MTLLanguageVersion: dict[int, str] = {(MTLLanguageVersion1_0:=65536): 'MTLLanguageVersion1_0', (MTLLanguageVersion1_1:=65537): 'MTLLanguageVersion1_1', (MTLLanguageVersion1_2:=65538): 'MTLLanguageVersion1_2', (MTLLanguageVersion2_0:=131072): 'MTLLanguageVersion2_0', (MTLLanguageVersion2_1:=131073): 'MTLLanguageVersion2_1', (MTLLanguageVersion2_2:=131074): 'MTLLanguageVersion2_2', (MTLLanguageVersion2_3:=131075): 'MTLLanguageVersion2_3', (MTLLanguageVersion2_4:=131076): 'MTLLanguageVersion2_4', (MTLLanguageVersion3_0:=196608): 'MTLLanguageVersion3_0', (MTLLanguageVersion3_1:=196609): 'MTLLanguageVersion3_1'}
MTLLanguageVersion: TypeAlias = NSUInteger
enum_MTLLibraryOptimizationLevel: dict[int, str] = {(MTLLibraryOptimizationLevelDefault:=0): 'MTLLibraryOptimizationLevelDefault', (MTLLibraryOptimizationLevelSize:=1): 'MTLLibraryOptimizationLevelSize'}
MTLLibraryOptimizationLevel: TypeAlias = NSInteger
enum_MTLCompileSymbolVisibility: dict[int, str] = {(MTLCompileSymbolVisibilityDefault:=0): 'MTLCompileSymbolVisibilityDefault', (MTLCompileSymbolVisibilityHidden:=1): 'MTLCompileSymbolVisibilityHidden'}
MTLCompileSymbolVisibility: TypeAlias = NSInteger
MTLCompileOptions._bases_ = [NSObject]
MTLCompileOptions._methods_ = [
  ('fastMathEnabled', BOOL, []),
  ('setFastMathEnabled:', None, [BOOL]),
  ('languageVersion', MTLLanguageVersion, []),
  ('setLanguageVersion:', None, [MTLLanguageVersion]),
  ('libraryType', MTLLibraryType, []),
  ('setLibraryType:', None, [MTLLibraryType]),
  ('installName', NSString, []),
  ('setInstallName:', None, [NSString]),
  ('preserveInvariance', BOOL, []),
  ('setPreserveInvariance:', None, [BOOL]),
  ('optimizationLevel', MTLLibraryOptimizationLevel, []),
  ('setOptimizationLevel:', None, [MTLLibraryOptimizationLevel]),
  ('compileSymbolVisibility', MTLCompileSymbolVisibility, []),
  ('setCompileSymbolVisibility:', None, [MTLCompileSymbolVisibility]),
  ('allowReferencingUndefinedSymbols', BOOL, []),
  ('setAllowReferencingUndefinedSymbols:', None, [BOOL]),
  ('maxTotalThreadsPerThreadgroup', NSUInteger, []),
  ('setMaxTotalThreadsPerThreadgroup:', None, [NSUInteger]),
]
class MTLStitchedLibraryDescriptor(objc.Spec): pass
class MTLRenderPipelineState(objc.Spec): pass
class MTLRenderPipelineDescriptor(objc.Spec): pass
class MTLTileRenderPipelineDescriptor(objc.Spec): pass
class MTLMeshRenderPipelineDescriptor(objc.Spec): pass
class MTLRasterizationRateMapDescriptor(objc.Spec): pass
class MTLIndirectCommandBufferDescriptor(objc.Spec): pass
class MTLSharedEvent(objc.Spec): pass
class MTLSharedEventHandle(objc.Spec): pass
class MTLIOFileHandle(objc.Spec): pass
class MTLIOCommandQueue(objc.Spec): pass
class MTLIOCommandQueueDescriptor(objc.Spec): pass
class MTLCounterSampleBufferDescriptor(objc.Spec): pass
class MTLCounterSet(objc.Spec): pass
MTLCounterSet._bases_ = [NSObject]
MTLCounterSet._methods_ = [
  ('name', NSString, []),
]
MTLCounterSampleBufferDescriptor._bases_ = [NSObject]
MTLCounterSampleBufferDescriptor._methods_ = [
  ('counterSet', MTLCounterSet, []),
  ('setCounterSet:', None, [MTLCounterSet]),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('storageMode', MTLStorageMode, []),
  ('setStorageMode:', None, [MTLStorageMode]),
  ('sampleCount', NSUInteger, []),
  ('setSampleCount:', None, [NSUInteger]),
]
MTLTimestamp: TypeAlias = ctypes.c_uint64
class MTLBufferBinding(objc.Spec): pass
class MTLBinding(objc.Spec): pass
MTLBufferBinding._bases_ = [MTLBinding]
MTLBufferBinding._methods_ = [
  ('bufferAlignment', NSUInteger, []),
  ('bufferDataSize', NSUInteger, []),
  ('bufferDataType', MTLDataType, []),
  ('bufferStructType', MTLStructType, []),
  ('bufferPointerType', MTLPointerType, []),
]
class MTLDynamicLibrary(objc.Spec): pass
class MTLBinaryArchive(objc.Spec): pass
class MTLBinaryArchiveDescriptor(objc.Spec): pass
class MTLAccelerationStructureDescriptor(objc.Spec): pass
MTLDevice._bases_ = [NSObject]
MTLDevice._methods_ = [
  ('newCommandQueue', MTLCommandQueue, [], True),
  ('newCommandQueueWithMaxCommandBufferCount:', MTLCommandQueue, [NSUInteger], True),
  ('heapTextureSizeAndAlignWithDescriptor:', MTLSizeAndAlign, [MTLTextureDescriptor]),
  ('heapBufferSizeAndAlignWithLength:options:', MTLSizeAndAlign, [NSUInteger, MTLResourceOptions]),
  ('newHeapWithDescriptor:', MTLHeap, [MTLHeapDescriptor], True),
  ('newBufferWithLength:options:', MTLBuffer, [NSUInteger, MTLResourceOptions], True),
  ('newBufferWithBytes:length:options:', MTLBuffer, [ctypes.c_void_p, NSUInteger, MTLResourceOptions], True),
  ('newDepthStencilStateWithDescriptor:', MTLDepthStencilState, [MTLDepthStencilDescriptor], True),
  ('newTextureWithDescriptor:', MTLTexture, [MTLTextureDescriptor], True),
  ('newTextureWithDescriptor:iosurface:plane:', MTLTexture, [MTLTextureDescriptor, IOSurfaceRef, NSUInteger], True),
  ('newSharedTextureWithDescriptor:', MTLTexture, [MTLTextureDescriptor], True),
  ('newSharedTextureWithHandle:', MTLTexture, [MTLSharedTextureHandle], True),
  ('newSamplerStateWithDescriptor:', MTLSamplerState, [MTLSamplerDescriptor], True),
  ('newDefaultLibrary', MTLLibrary, [], True),
  ('newDefaultLibraryWithBundle:error:', MTLLibrary, [NSBundle, c.POINTER[NSError]], True),
  ('newLibraryWithFile:error:', MTLLibrary, [NSString, c.POINTER[NSError]], True),
  ('newLibraryWithURL:error:', MTLLibrary, [NSURL, c.POINTER[NSError]], True),
  ('newLibraryWithData:error:', MTLLibrary, [objc.id_, c.POINTER[NSError]], True),
  ('newLibraryWithSource:options:error:', MTLLibrary, [NSString, MTLCompileOptions, c.POINTER[NSError]], True),
  ('newLibraryWithStitchedDescriptor:error:', MTLLibrary, [MTLStitchedLibraryDescriptor, c.POINTER[NSError]], True),
  ('newRenderPipelineStateWithDescriptor:error:', MTLRenderPipelineState, [MTLRenderPipelineDescriptor, c.POINTER[NSError]], True),
  ('newRenderPipelineStateWithDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLRenderPipelineDescriptor, MTLPipelineOption, c.POINTER[MTLRenderPipelineReflection], c.POINTER[NSError]], True),
  ('newComputePipelineStateWithFunction:error:', MTLComputePipelineState, [MTLFunction, c.POINTER[NSError]], True),
  ('newComputePipelineStateWithFunction:options:reflection:error:', MTLComputePipelineState, [MTLFunction, MTLPipelineOption, c.POINTER[MTLComputePipelineReflection], c.POINTER[NSError]], True),
  ('newComputePipelineStateWithDescriptor:options:reflection:error:', MTLComputePipelineState, [MTLComputePipelineDescriptor, MTLPipelineOption, c.POINTER[MTLComputePipelineReflection], c.POINTER[NSError]], True),
  ('newFence', MTLFence, [], True),
  ('supportsFeatureSet:', BOOL, [MTLFeatureSet]),
  ('supportsFamily:', BOOL, [MTLGPUFamily]),
  ('supportsTextureSampleCount:', BOOL, [NSUInteger]),
  ('minimumLinearTextureAlignmentForPixelFormat:', NSUInteger, [MTLPixelFormat]),
  ('minimumTextureBufferAlignmentForPixelFormat:', NSUInteger, [MTLPixelFormat]),
  ('newRenderPipelineStateWithTileDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLTileRenderPipelineDescriptor, MTLPipelineOption, c.POINTER[MTLRenderPipelineReflection], c.POINTER[NSError]], True),
  ('newRenderPipelineStateWithMeshDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLMeshRenderPipelineDescriptor, MTLPipelineOption, c.POINTER[MTLRenderPipelineReflection], c.POINTER[NSError]], True),
  ('getDefaultSamplePositions:count:', None, [c.POINTER[MTLSamplePosition], NSUInteger]),
  ('supportsRasterizationRateMapWithLayerCount:', BOOL, [NSUInteger]),
  ('newRasterizationRateMapWithDescriptor:', MTLRasterizationRateMap, [MTLRasterizationRateMapDescriptor], True),
  ('newIndirectCommandBufferWithDescriptor:maxCommandCount:options:', MTLIndirectCommandBuffer, [MTLIndirectCommandBufferDescriptor, NSUInteger, MTLResourceOptions], True),
  ('newEvent', MTLEvent, [], True),
  ('newSharedEvent', MTLSharedEvent, [], True),
  ('newSharedEventWithHandle:', MTLSharedEvent, [MTLSharedEventHandle], True),
  ('newIOHandleWithURL:error:', MTLIOFileHandle, [NSURL, c.POINTER[NSError]], True),
  ('newIOCommandQueueWithDescriptor:error:', MTLIOCommandQueue, [MTLIOCommandQueueDescriptor, c.POINTER[NSError]], True),
  ('newIOHandleWithURL:compressionMethod:error:', MTLIOFileHandle, [NSURL, MTLIOCompressionMethod, c.POINTER[NSError]], True),
  ('newIOFileHandleWithURL:error:', MTLIOFileHandle, [NSURL, c.POINTER[NSError]], True),
  ('newIOFileHandleWithURL:compressionMethod:error:', MTLIOFileHandle, [NSURL, MTLIOCompressionMethod, c.POINTER[NSError]], True),
  ('sparseTileSizeWithTextureType:pixelFormat:sampleCount:', MTLSize, [MTLTextureType, MTLPixelFormat, NSUInteger]),
  ('convertSparsePixelRegions:toTileRegions:withTileSize:alignmentMode:numRegions:', None, [c.POINTER[MTLRegion], c.POINTER[MTLRegion], MTLSize, MTLSparseTextureRegionAlignmentMode, NSUInteger]),
  ('convertSparseTileRegions:toPixelRegions:withTileSize:numRegions:', None, [c.POINTER[MTLRegion], c.POINTER[MTLRegion], MTLSize, NSUInteger]),
  ('sparseTileSizeInBytesForSparsePageSize:', NSUInteger, [MTLSparsePageSize]),
  ('sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:', MTLSize, [MTLTextureType, MTLPixelFormat, NSUInteger, MTLSparsePageSize]),
  ('newCounterSampleBufferWithDescriptor:error:', MTLCounterSampleBuffer, [MTLCounterSampleBufferDescriptor, c.POINTER[NSError]], True),
  ('sampleTimestamps:gpuTimestamp:', None, [c.POINTER[MTLTimestamp], c.POINTER[MTLTimestamp]]),
  ('newArgumentEncoderWithBufferBinding:', MTLArgumentEncoder, [MTLBufferBinding], True),
  ('supportsCounterSampling:', BOOL, [MTLCounterSamplingPoint]),
  ('supportsVertexAmplificationCount:', BOOL, [NSUInteger]),
  ('newDynamicLibrary:error:', MTLDynamicLibrary, [MTLLibrary, c.POINTER[NSError]], True),
  ('newDynamicLibraryWithURL:error:', MTLDynamicLibrary, [NSURL, c.POINTER[NSError]], True),
  ('newBinaryArchiveWithDescriptor:error:', MTLBinaryArchive, [MTLBinaryArchiveDescriptor, c.POINTER[NSError]], True),
  ('accelerationStructureSizesWithDescriptor:', MTLAccelerationStructureSizes, [MTLAccelerationStructureDescriptor]),
  ('newAccelerationStructureWithSize:', MTLAccelerationStructure, [NSUInteger], True),
  ('newAccelerationStructureWithDescriptor:', MTLAccelerationStructure, [MTLAccelerationStructureDescriptor], True),
  ('heapAccelerationStructureSizeAndAlignWithSize:', MTLSizeAndAlign, [NSUInteger]),
  ('heapAccelerationStructureSizeAndAlignWithDescriptor:', MTLSizeAndAlign, [MTLAccelerationStructureDescriptor]),
  ('name', NSString, []),
  ('registryID', uint64_t, []),
  ('architecture', MTLArchitecture, []),
  ('maxThreadsPerThreadgroup', MTLSize, []),
  ('isLowPower', BOOL, []),
  ('isHeadless', BOOL, []),
  ('isRemovable', BOOL, []),
  ('hasUnifiedMemory', BOOL, []),
  ('recommendedMaxWorkingSetSize', uint64_t, []),
  ('location', MTLDeviceLocation, []),
  ('locationNumber', NSUInteger, []),
  ('maxTransferRate', uint64_t, []),
  ('isDepth24Stencil8PixelFormatSupported', BOOL, []),
  ('readWriteTextureSupport', MTLReadWriteTextureTier, []),
  ('argumentBuffersSupport', MTLArgumentBuffersTier, []),
  ('areRasterOrderGroupsSupported', BOOL, []),
  ('supports32BitFloatFiltering', BOOL, []),
  ('supports32BitMSAA', BOOL, []),
  ('supportsQueryTextureLOD', BOOL, []),
  ('supportsBCTextureCompression', BOOL, []),
  ('supportsPullModelInterpolation', BOOL, []),
  ('areBarycentricCoordsSupported', BOOL, []),
  ('supportsShaderBarycentricCoordinates', BOOL, []),
  ('currentAllocatedSize', NSUInteger, []),
  ('maxThreadgroupMemoryLength', NSUInteger, []),
  ('maxArgumentBufferSamplerCount', NSUInteger, []),
  ('areProgrammableSamplePositionsSupported', BOOL, []),
  ('peerGroupID', uint64_t, []),
  ('peerIndex', uint32_t, []),
  ('peerCount', uint32_t, []),
  ('sparseTileSizeInBytes', NSUInteger, []),
  ('maxBufferLength', NSUInteger, []),
  ('supportsDynamicLibraries', BOOL, []),
  ('supportsRenderDynamicLibraries', BOOL, []),
  ('supportsRaytracing', BOOL, []),
  ('supportsFunctionPointers', BOOL, []),
  ('supportsFunctionPointersFromRender', BOOL, []),
  ('supportsRaytracingFromRender', BOOL, []),
  ('supportsPrimitiveMotionBlur', BOOL, []),
  ('shouldMaximizeConcurrentCompilation', BOOL, []),
  ('setShouldMaximizeConcurrentCompilation:', None, [BOOL]),
  ('maximumConcurrentCompilationTaskCount', NSUInteger, []),
]
enum_MTLIndirectCommandType: dict[int, str] = {(MTLIndirectCommandTypeDraw:=1): 'MTLIndirectCommandTypeDraw', (MTLIndirectCommandTypeDrawIndexed:=2): 'MTLIndirectCommandTypeDrawIndexed', (MTLIndirectCommandTypeDrawPatches:=4): 'MTLIndirectCommandTypeDrawPatches', (MTLIndirectCommandTypeDrawIndexedPatches:=8): 'MTLIndirectCommandTypeDrawIndexedPatches', (MTLIndirectCommandTypeConcurrentDispatch:=32): 'MTLIndirectCommandTypeConcurrentDispatch', (MTLIndirectCommandTypeConcurrentDispatchThreads:=64): 'MTLIndirectCommandTypeConcurrentDispatchThreads', (MTLIndirectCommandTypeDrawMeshThreadgroups:=128): 'MTLIndirectCommandTypeDrawMeshThreadgroups', (MTLIndirectCommandTypeDrawMeshThreads:=256): 'MTLIndirectCommandTypeDrawMeshThreads'}
MTLIndirectCommandType: TypeAlias = NSUInteger
@c.record
class MTLIndirectCommandBufferExecutionRange(c.Struct):
  SIZE = 8
  location: int
  length: int
MTLIndirectCommandBufferExecutionRange.register_fields([('location', uint32_t, 0), ('length', uint32_t, 4)])
MTLIndirectCommandBufferDescriptor._bases_ = [NSObject]
MTLIndirectCommandBufferDescriptor._methods_ = [
  ('commandTypes', MTLIndirectCommandType, []),
  ('setCommandTypes:', None, [MTLIndirectCommandType]),
  ('inheritPipelineState', BOOL, []),
  ('setInheritPipelineState:', None, [BOOL]),
  ('inheritBuffers', BOOL, []),
  ('setInheritBuffers:', None, [BOOL]),
  ('maxVertexBufferBindCount', NSUInteger, []),
  ('setMaxVertexBufferBindCount:', None, [NSUInteger]),
  ('maxFragmentBufferBindCount', NSUInteger, []),
  ('setMaxFragmentBufferBindCount:', None, [NSUInteger]),
  ('maxKernelBufferBindCount', NSUInteger, []),
  ('setMaxKernelBufferBindCount:', None, [NSUInteger]),
  ('maxKernelThreadgroupMemoryBindCount', NSUInteger, []),
  ('setMaxKernelThreadgroupMemoryBindCount:', None, [NSUInteger]),
  ('maxObjectBufferBindCount', NSUInteger, []),
  ('setMaxObjectBufferBindCount:', None, [NSUInteger]),
  ('maxMeshBufferBindCount', NSUInteger, []),
  ('setMaxMeshBufferBindCount:', None, [NSUInteger]),
  ('maxObjectThreadgroupMemoryBindCount', NSUInteger, []),
  ('setMaxObjectThreadgroupMemoryBindCount:', None, [NSUInteger]),
  ('supportRayTracing', BOOL, []),
  ('setSupportRayTracing:', None, [BOOL]),
  ('supportDynamicAttributeStride', BOOL, []),
  ('setSupportDynamicAttributeStride:', None, [BOOL]),
]
class MTLIndirectRenderCommand(objc.Spec): pass
enum_MTLPrimitiveType: dict[int, str] = {(MTLPrimitiveTypePoint:=0): 'MTLPrimitiveTypePoint', (MTLPrimitiveTypeLine:=1): 'MTLPrimitiveTypeLine', (MTLPrimitiveTypeLineStrip:=2): 'MTLPrimitiveTypeLineStrip', (MTLPrimitiveTypeTriangle:=3): 'MTLPrimitiveTypeTriangle', (MTLPrimitiveTypeTriangleStrip:=4): 'MTLPrimitiveTypeTriangleStrip'}
MTLPrimitiveType: TypeAlias = NSUInteger
MTLIndirectRenderCommand._bases_ = [NSObject]
MTLIndirectRenderCommand._methods_ = [
  ('setRenderPipelineState:', None, [MTLRenderPipelineState]),
  ('setVertexBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setFragmentBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setVertexBuffer:offset:attributeStride:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:', None, [NSUInteger, NSUInteger, NSUInteger, MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLBuffer, NSUInteger, NSUInteger]),
  ('drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:', None, [NSUInteger, NSUInteger, NSUInteger, MTLBuffer, NSUInteger, MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLBuffer, NSUInteger, NSUInteger]),
  ('drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:', None, [MTLPrimitiveType, NSUInteger, NSUInteger, NSUInteger, NSUInteger]),
  ('drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:', None, [MTLPrimitiveType, NSUInteger, MTLIndexType, MTLBuffer, NSUInteger, NSUInteger, NSInteger, NSUInteger]),
  ('setObjectThreadgroupMemoryLength:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setObjectBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setMeshBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:', None, [MTLSize, MTLSize, MTLSize]),
  ('drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:', None, [MTLSize, MTLSize, MTLSize]),
  ('setBarrier', None, []),
  ('clearBarrier', None, []),
  ('reset', None, []),
]
class MTLIndirectComputeCommand(objc.Spec): pass
MTLIndirectComputeCommand._bases_ = [NSObject]
MTLIndirectComputeCommand._methods_ = [
  ('setComputePipelineState:', None, [MTLComputePipelineState]),
  ('setKernelBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setKernelBuffer:offset:attributeStride:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('concurrentDispatchThreadgroups:threadsPerThreadgroup:', None, [MTLSize, MTLSize]),
  ('concurrentDispatchThreads:threadsPerThreadgroup:', None, [MTLSize, MTLSize]),
  ('setBarrier', None, []),
  ('clearBarrier', None, []),
  ('setImageblockWidth:height:', None, [NSUInteger, NSUInteger]),
  ('reset', None, []),
  ('setThreadgroupMemoryLength:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setStageInRegion:', None, [MTLRegion]),
]
MTLIndirectCommandBuffer._bases_ = [MTLResource]
MTLIndirectCommandBuffer._methods_ = [
  ('resetWithRange:', None, [NSRange]),
  ('indirectRenderCommandAtIndex:', MTLIndirectRenderCommand, [NSUInteger]),
  ('indirectComputeCommandAtIndex:', MTLIndirectComputeCommand, [NSUInteger]),
  ('size', NSUInteger, []),
  ('gpuResourceID', MTLResourceID, []),
]
MTLCommandEncoder._bases_ = [NSObject]
MTLCommandEncoder._methods_ = [
  ('endEncoding', None, []),
  ('insertDebugSignpost:', None, [NSString]),
  ('pushDebugGroup:', None, [NSString]),
  ('popDebugGroup', None, []),
  ('device', MTLDevice, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
]
MTLResourceCPUCacheModeShift = 0
MTLResourceCPUCacheModeMask = (0xf << MTLResourceCPUCacheModeShift)
MTLResourceStorageModeShift = 4
MTLResourceStorageModeMask = (0xf << MTLResourceStorageModeShift)
MTLResourceHazardTrackingModeShift = 8
MTLResourceHazardTrackingModeMask = (0x3 << MTLResourceHazardTrackingModeShift)