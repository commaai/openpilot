# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import objc
dll = DLL('metal', 'Metal')
class MTLDispatchThreadgroupsIndirectArguments(Struct): pass
uint32_t = ctypes.c_uint32
MTLDispatchThreadgroupsIndirectArguments._fields_ = [
  ('threadgroupsPerGrid', (uint32_t * 3)),
]
class MTLStageInRegionIndirectArguments(Struct): pass
MTLStageInRegionIndirectArguments._fields_ = [
  ('stageInOrigin', (uint32_t * 3)),
  ('stageInSize', (uint32_t * 3)),
]
class MTLComputeCommandEncoder(objc.Spec): pass
class MTLCommandEncoder(objc.Spec): pass
class MTLComputePipelineState(objc.Spec): pass
NSUInteger = ctypes.c_uint64
class MTLBuffer(objc.Spec): pass
class MTLResource(objc.Spec): pass
class struct__NSRange(Struct): pass
NSRange = struct__NSRange
struct__NSRange._fields_ = [
  ('location', NSUInteger),
  ('length', NSUInteger),
]
class MTLTexture(objc.Spec): pass
class MTLTextureDescriptor(objc.Spec): pass
enum_MTLTextureType = CEnum(NSUInteger)
MTLTextureType1D = enum_MTLTextureType.define('MTLTextureType1D', 0)
MTLTextureType1DArray = enum_MTLTextureType.define('MTLTextureType1DArray', 1)
MTLTextureType2D = enum_MTLTextureType.define('MTLTextureType2D', 2)
MTLTextureType2DArray = enum_MTLTextureType.define('MTLTextureType2DArray', 3)
MTLTextureType2DMultisample = enum_MTLTextureType.define('MTLTextureType2DMultisample', 4)
MTLTextureTypeCube = enum_MTLTextureType.define('MTLTextureTypeCube', 5)
MTLTextureTypeCubeArray = enum_MTLTextureType.define('MTLTextureTypeCubeArray', 6)
MTLTextureType3D = enum_MTLTextureType.define('MTLTextureType3D', 7)
MTLTextureType2DMultisampleArray = enum_MTLTextureType.define('MTLTextureType2DMultisampleArray', 8)
MTLTextureTypeTextureBuffer = enum_MTLTextureType.define('MTLTextureTypeTextureBuffer', 9)

MTLTextureType = enum_MTLTextureType
enum_MTLPixelFormat = CEnum(NSUInteger)
MTLPixelFormatInvalid = enum_MTLPixelFormat.define('MTLPixelFormatInvalid', 0)
MTLPixelFormatA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatA8Unorm', 1)
MTLPixelFormatR8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatR8Unorm', 10)
MTLPixelFormatR8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatR8Unorm_sRGB', 11)
MTLPixelFormatR8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatR8Snorm', 12)
MTLPixelFormatR8Uint = enum_MTLPixelFormat.define('MTLPixelFormatR8Uint', 13)
MTLPixelFormatR8Sint = enum_MTLPixelFormat.define('MTLPixelFormatR8Sint', 14)
MTLPixelFormatR16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatR16Unorm', 20)
MTLPixelFormatR16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatR16Snorm', 22)
MTLPixelFormatR16Uint = enum_MTLPixelFormat.define('MTLPixelFormatR16Uint', 23)
MTLPixelFormatR16Sint = enum_MTLPixelFormat.define('MTLPixelFormatR16Sint', 24)
MTLPixelFormatR16Float = enum_MTLPixelFormat.define('MTLPixelFormatR16Float', 25)
MTLPixelFormatRG8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRG8Unorm', 30)
MTLPixelFormatRG8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatRG8Unorm_sRGB', 31)
MTLPixelFormatRG8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRG8Snorm', 32)
MTLPixelFormatRG8Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG8Uint', 33)
MTLPixelFormatRG8Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG8Sint', 34)
MTLPixelFormatB5G6R5Unorm = enum_MTLPixelFormat.define('MTLPixelFormatB5G6R5Unorm', 40)
MTLPixelFormatA1BGR5Unorm = enum_MTLPixelFormat.define('MTLPixelFormatA1BGR5Unorm', 41)
MTLPixelFormatABGR4Unorm = enum_MTLPixelFormat.define('MTLPixelFormatABGR4Unorm', 42)
MTLPixelFormatBGR5A1Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGR5A1Unorm', 43)
MTLPixelFormatR32Uint = enum_MTLPixelFormat.define('MTLPixelFormatR32Uint', 53)
MTLPixelFormatR32Sint = enum_MTLPixelFormat.define('MTLPixelFormatR32Sint', 54)
MTLPixelFormatR32Float = enum_MTLPixelFormat.define('MTLPixelFormatR32Float', 55)
MTLPixelFormatRG16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRG16Unorm', 60)
MTLPixelFormatRG16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRG16Snorm', 62)
MTLPixelFormatRG16Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG16Uint', 63)
MTLPixelFormatRG16Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG16Sint', 64)
MTLPixelFormatRG16Float = enum_MTLPixelFormat.define('MTLPixelFormatRG16Float', 65)
MTLPixelFormatRGBA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Unorm', 70)
MTLPixelFormatRGBA8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Unorm_sRGB', 71)
MTLPixelFormatRGBA8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Snorm', 72)
MTLPixelFormatRGBA8Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Uint', 73)
MTLPixelFormatRGBA8Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Sint', 74)
MTLPixelFormatBGRA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGRA8Unorm', 80)
MTLPixelFormatBGRA8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGRA8Unorm_sRGB', 81)
MTLPixelFormatRGB10A2Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGB10A2Unorm', 90)
MTLPixelFormatRGB10A2Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGB10A2Uint', 91)
MTLPixelFormatRG11B10Float = enum_MTLPixelFormat.define('MTLPixelFormatRG11B10Float', 92)
MTLPixelFormatRGB9E5Float = enum_MTLPixelFormat.define('MTLPixelFormatRGB9E5Float', 93)
MTLPixelFormatBGR10A2Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGR10A2Unorm', 94)
MTLPixelFormatBGR10_XR = enum_MTLPixelFormat.define('MTLPixelFormatBGR10_XR', 554)
MTLPixelFormatBGR10_XR_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGR10_XR_sRGB', 555)
MTLPixelFormatRG32Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG32Uint', 103)
MTLPixelFormatRG32Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG32Sint', 104)
MTLPixelFormatRG32Float = enum_MTLPixelFormat.define('MTLPixelFormatRG32Float', 105)
MTLPixelFormatRGBA16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Unorm', 110)
MTLPixelFormatRGBA16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Snorm', 112)
MTLPixelFormatRGBA16Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Uint', 113)
MTLPixelFormatRGBA16Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Sint', 114)
MTLPixelFormatRGBA16Float = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Float', 115)
MTLPixelFormatBGRA10_XR = enum_MTLPixelFormat.define('MTLPixelFormatBGRA10_XR', 552)
MTLPixelFormatBGRA10_XR_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGRA10_XR_sRGB', 553)
MTLPixelFormatRGBA32Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Uint', 123)
MTLPixelFormatRGBA32Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Sint', 124)
MTLPixelFormatRGBA32Float = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Float', 125)
MTLPixelFormatBC1_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC1_RGBA', 130)
MTLPixelFormatBC1_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC1_RGBA_sRGB', 131)
MTLPixelFormatBC2_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC2_RGBA', 132)
MTLPixelFormatBC2_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC2_RGBA_sRGB', 133)
MTLPixelFormatBC3_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC3_RGBA', 134)
MTLPixelFormatBC3_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC3_RGBA_sRGB', 135)
MTLPixelFormatBC4_RUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC4_RUnorm', 140)
MTLPixelFormatBC4_RSnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC4_RSnorm', 141)
MTLPixelFormatBC5_RGUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC5_RGUnorm', 142)
MTLPixelFormatBC5_RGSnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC5_RGSnorm', 143)
MTLPixelFormatBC6H_RGBFloat = enum_MTLPixelFormat.define('MTLPixelFormatBC6H_RGBFloat', 150)
MTLPixelFormatBC6H_RGBUfloat = enum_MTLPixelFormat.define('MTLPixelFormatBC6H_RGBUfloat', 151)
MTLPixelFormatBC7_RGBAUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC7_RGBAUnorm', 152)
MTLPixelFormatBC7_RGBAUnorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC7_RGBAUnorm_sRGB', 153)
MTLPixelFormatPVRTC_RGB_2BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_2BPP', 160)
MTLPixelFormatPVRTC_RGB_2BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_2BPP_sRGB', 161)
MTLPixelFormatPVRTC_RGB_4BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_4BPP', 162)
MTLPixelFormatPVRTC_RGB_4BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_4BPP_sRGB', 163)
MTLPixelFormatPVRTC_RGBA_2BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_2BPP', 164)
MTLPixelFormatPVRTC_RGBA_2BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_2BPP_sRGB', 165)
MTLPixelFormatPVRTC_RGBA_4BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_4BPP', 166)
MTLPixelFormatPVRTC_RGBA_4BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_4BPP_sRGB', 167)
MTLPixelFormatEAC_R11Unorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_R11Unorm', 170)
MTLPixelFormatEAC_R11Snorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_R11Snorm', 172)
MTLPixelFormatEAC_RG11Unorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RG11Unorm', 174)
MTLPixelFormatEAC_RG11Snorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RG11Snorm', 176)
MTLPixelFormatEAC_RGBA8 = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RGBA8', 178)
MTLPixelFormatEAC_RGBA8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RGBA8_sRGB', 179)
MTLPixelFormatETC2_RGB8 = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8', 180)
MTLPixelFormatETC2_RGB8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8_sRGB', 181)
MTLPixelFormatETC2_RGB8A1 = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8A1', 182)
MTLPixelFormatETC2_RGB8A1_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8A1_sRGB', 183)
MTLPixelFormatASTC_4x4_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_sRGB', 186)
MTLPixelFormatASTC_5x4_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_sRGB', 187)
MTLPixelFormatASTC_5x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_sRGB', 188)
MTLPixelFormatASTC_6x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_sRGB', 189)
MTLPixelFormatASTC_6x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_sRGB', 190)
MTLPixelFormatASTC_8x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_sRGB', 192)
MTLPixelFormatASTC_8x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_sRGB', 193)
MTLPixelFormatASTC_8x8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_sRGB', 194)
MTLPixelFormatASTC_10x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_sRGB', 195)
MTLPixelFormatASTC_10x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_sRGB', 196)
MTLPixelFormatASTC_10x8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_sRGB', 197)
MTLPixelFormatASTC_10x10_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_sRGB', 198)
MTLPixelFormatASTC_12x10_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_sRGB', 199)
MTLPixelFormatASTC_12x12_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_sRGB', 200)
MTLPixelFormatASTC_4x4_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_LDR', 204)
MTLPixelFormatASTC_5x4_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_LDR', 205)
MTLPixelFormatASTC_5x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_LDR', 206)
MTLPixelFormatASTC_6x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_LDR', 207)
MTLPixelFormatASTC_6x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_LDR', 208)
MTLPixelFormatASTC_8x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_LDR', 210)
MTLPixelFormatASTC_8x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_LDR', 211)
MTLPixelFormatASTC_8x8_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_LDR', 212)
MTLPixelFormatASTC_10x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_LDR', 213)
MTLPixelFormatASTC_10x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_LDR', 214)
MTLPixelFormatASTC_10x8_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_LDR', 215)
MTLPixelFormatASTC_10x10_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_LDR', 216)
MTLPixelFormatASTC_12x10_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_LDR', 217)
MTLPixelFormatASTC_12x12_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_LDR', 218)
MTLPixelFormatASTC_4x4_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_HDR', 222)
MTLPixelFormatASTC_5x4_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_HDR', 223)
MTLPixelFormatASTC_5x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_HDR', 224)
MTLPixelFormatASTC_6x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_HDR', 225)
MTLPixelFormatASTC_6x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_HDR', 226)
MTLPixelFormatASTC_8x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_HDR', 228)
MTLPixelFormatASTC_8x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_HDR', 229)
MTLPixelFormatASTC_8x8_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_HDR', 230)
MTLPixelFormatASTC_10x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_HDR', 231)
MTLPixelFormatASTC_10x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_HDR', 232)
MTLPixelFormatASTC_10x8_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_HDR', 233)
MTLPixelFormatASTC_10x10_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_HDR', 234)
MTLPixelFormatASTC_12x10_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_HDR', 235)
MTLPixelFormatASTC_12x12_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_HDR', 236)
MTLPixelFormatGBGR422 = enum_MTLPixelFormat.define('MTLPixelFormatGBGR422', 240)
MTLPixelFormatBGRG422 = enum_MTLPixelFormat.define('MTLPixelFormatBGRG422', 241)
MTLPixelFormatDepth16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatDepth16Unorm', 250)
MTLPixelFormatDepth32Float = enum_MTLPixelFormat.define('MTLPixelFormatDepth32Float', 252)
MTLPixelFormatStencil8 = enum_MTLPixelFormat.define('MTLPixelFormatStencil8', 253)
MTLPixelFormatDepth24Unorm_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatDepth24Unorm_Stencil8', 255)
MTLPixelFormatDepth32Float_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatDepth32Float_Stencil8', 260)
MTLPixelFormatX32_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatX32_Stencil8', 261)
MTLPixelFormatX24_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatX24_Stencil8', 262)

MTLPixelFormat = enum_MTLPixelFormat
enum_MTLResourceOptions = CEnum(NSUInteger)
MTLResourceCPUCacheModeDefaultCache = enum_MTLResourceOptions.define('MTLResourceCPUCacheModeDefaultCache', 0)
MTLResourceCPUCacheModeWriteCombined = enum_MTLResourceOptions.define('MTLResourceCPUCacheModeWriteCombined', 1)
MTLResourceStorageModeShared = enum_MTLResourceOptions.define('MTLResourceStorageModeShared', 0)
MTLResourceStorageModeManaged = enum_MTLResourceOptions.define('MTLResourceStorageModeManaged', 16)
MTLResourceStorageModePrivate = enum_MTLResourceOptions.define('MTLResourceStorageModePrivate', 32)
MTLResourceStorageModeMemoryless = enum_MTLResourceOptions.define('MTLResourceStorageModeMemoryless', 48)
MTLResourceHazardTrackingModeDefault = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeDefault', 0)
MTLResourceHazardTrackingModeUntracked = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeUntracked', 256)
MTLResourceHazardTrackingModeTracked = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeTracked', 512)
MTLResourceOptionCPUCacheModeDefault = enum_MTLResourceOptions.define('MTLResourceOptionCPUCacheModeDefault', 0)
MTLResourceOptionCPUCacheModeWriteCombined = enum_MTLResourceOptions.define('MTLResourceOptionCPUCacheModeWriteCombined', 1)

MTLResourceOptions = enum_MTLResourceOptions
enum_MTLCPUCacheMode = CEnum(NSUInteger)
MTLCPUCacheModeDefaultCache = enum_MTLCPUCacheMode.define('MTLCPUCacheModeDefaultCache', 0)
MTLCPUCacheModeWriteCombined = enum_MTLCPUCacheMode.define('MTLCPUCacheModeWriteCombined', 1)

MTLCPUCacheMode = enum_MTLCPUCacheMode
enum_MTLStorageMode = CEnum(NSUInteger)
MTLStorageModeShared = enum_MTLStorageMode.define('MTLStorageModeShared', 0)
MTLStorageModeManaged = enum_MTLStorageMode.define('MTLStorageModeManaged', 1)
MTLStorageModePrivate = enum_MTLStorageMode.define('MTLStorageModePrivate', 2)
MTLStorageModeMemoryless = enum_MTLStorageMode.define('MTLStorageModeMemoryless', 3)

MTLStorageMode = enum_MTLStorageMode
enum_MTLHazardTrackingMode = CEnum(NSUInteger)
MTLHazardTrackingModeDefault = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeDefault', 0)
MTLHazardTrackingModeUntracked = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeUntracked', 1)
MTLHazardTrackingModeTracked = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeTracked', 2)

MTLHazardTrackingMode = enum_MTLHazardTrackingMode
enum_MTLTextureUsage = CEnum(NSUInteger)
MTLTextureUsageUnknown = enum_MTLTextureUsage.define('MTLTextureUsageUnknown', 0)
MTLTextureUsageShaderRead = enum_MTLTextureUsage.define('MTLTextureUsageShaderRead', 1)
MTLTextureUsageShaderWrite = enum_MTLTextureUsage.define('MTLTextureUsageShaderWrite', 2)
MTLTextureUsageRenderTarget = enum_MTLTextureUsage.define('MTLTextureUsageRenderTarget', 4)
MTLTextureUsagePixelFormatView = enum_MTLTextureUsage.define('MTLTextureUsagePixelFormatView', 16)
MTLTextureUsageShaderAtomic = enum_MTLTextureUsage.define('MTLTextureUsageShaderAtomic', 32)

MTLTextureUsage = enum_MTLTextureUsage
BOOL = ctypes.c_int32
NSInteger = ctypes.c_int64
enum_MTLTextureCompressionType = CEnum(NSInteger)
MTLTextureCompressionTypeLossless = enum_MTLTextureCompressionType.define('MTLTextureCompressionTypeLossless', 0)
MTLTextureCompressionTypeLossy = enum_MTLTextureCompressionType.define('MTLTextureCompressionTypeLossy', 1)

MTLTextureCompressionType = enum_MTLTextureCompressionType
class MTLTextureSwizzleChannels(Struct): pass
uint8_t = ctypes.c_ubyte
enum_MTLTextureSwizzle = CEnum(uint8_t)
MTLTextureSwizzleZero = enum_MTLTextureSwizzle.define('MTLTextureSwizzleZero', 0)
MTLTextureSwizzleOne = enum_MTLTextureSwizzle.define('MTLTextureSwizzleOne', 1)
MTLTextureSwizzleRed = enum_MTLTextureSwizzle.define('MTLTextureSwizzleRed', 2)
MTLTextureSwizzleGreen = enum_MTLTextureSwizzle.define('MTLTextureSwizzleGreen', 3)
MTLTextureSwizzleBlue = enum_MTLTextureSwizzle.define('MTLTextureSwizzleBlue', 4)
MTLTextureSwizzleAlpha = enum_MTLTextureSwizzle.define('MTLTextureSwizzleAlpha', 5)

MTLTextureSwizzle = enum_MTLTextureSwizzle
MTLTextureSwizzleChannels._fields_ = [
  ('red', MTLTextureSwizzle),
  ('green', MTLTextureSwizzle),
  ('blue', MTLTextureSwizzle),
  ('alpha', MTLTextureSwizzle),
]
class NSObject(objc.Spec): pass
IMP = ctypes.CFUNCTYPE(None, )
class NSInvocation(objc.Spec): pass
class NSMethodSignature(objc.Spec): pass
NSMethodSignature._bases_ = [NSObject]
NSMethodSignature._methods_ = [
  ('getArgumentTypeAtIndex:', ctypes.POINTER(ctypes.c_char), [NSUInteger]),
  ('isOneway', BOOL, []),
  ('numberOfArguments', NSUInteger, []),
  ('frameLength', NSUInteger, []),
  ('methodReturnType', ctypes.POINTER(ctypes.c_char), []),
  ('methodReturnLength', NSUInteger, []),
]
NSMethodSignature._classmethods_ = [
  ('signatureWithObjCTypes:', NSMethodSignature, [ctypes.POINTER(ctypes.c_char)]),
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
class struct__NSZone(Struct): pass
class Protocol(objc.Spec): pass
class NSString(objc.Spec): pass
unichar = ctypes.c_uint16
class NSCoder(objc.Spec): pass
class NSData(objc.Spec): pass
NSData._bases_ = [NSObject]
NSData._methods_ = [
  ('length', NSUInteger, []),
  ('bytes', ctypes.c_void_p, []),
]
NSCoder._bases_ = [NSObject]
NSCoder._methods_ = [
  ('encodeValueOfObjCType:at:', None, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]),
  ('encodeDataObject:', None, [NSData]),
  ('decodeDataObject', NSData, []),
  ('decodeValueOfObjCType:at:size:', None, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, NSUInteger]),
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
  ('allocWithZone:', 'instancetype', [ctypes.POINTER(struct__NSZone)], True),
  ('alloc', 'instancetype', [], True),
  ('copyWithZone:', objc.id_, [ctypes.POINTER(struct__NSZone)], True),
  ('mutableCopyWithZone:', objc.id_, [ctypes.POINTER(struct__NSZone)], True),
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
uint64_t = ctypes.c_uint64
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
class MTLRegion(Struct): pass
class MTLOrigin(Struct): pass
MTLOrigin._fields_ = [
  ('x', NSUInteger),
  ('y', NSUInteger),
  ('z', NSUInteger),
]
class MTLSize(Struct): pass
MTLSize._fields_ = [
  ('width', NSUInteger),
  ('height', NSUInteger),
  ('depth', NSUInteger),
]
MTLRegion._fields_ = [
  ('origin', MTLOrigin),
  ('size', MTLSize),
]
class MTLFence(objc.Spec): pass
MTLFence._bases_ = [NSObject]
MTLFence._methods_ = [
  ('device', MTLDevice, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
]
enum_MTLPurgeableState = CEnum(NSUInteger)
MTLPurgeableStateKeepCurrent = enum_MTLPurgeableState.define('MTLPurgeableStateKeepCurrent', 1)
MTLPurgeableStateNonVolatile = enum_MTLPurgeableState.define('MTLPurgeableStateNonVolatile', 2)
MTLPurgeableStateVolatile = enum_MTLPurgeableState.define('MTLPurgeableStateVolatile', 3)
MTLPurgeableStateEmpty = enum_MTLPurgeableState.define('MTLPurgeableStateEmpty', 4)

MTLPurgeableState = enum_MTLPurgeableState
kern_return_t = ctypes.c_int32
task_id_token_t = ctypes.c_uint32
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
enum_MTLResourceUsage = CEnum(NSUInteger)
MTLResourceUsageRead = enum_MTLResourceUsage.define('MTLResourceUsageRead', 1)
MTLResourceUsageWrite = enum_MTLResourceUsage.define('MTLResourceUsageWrite', 2)
MTLResourceUsageSample = enum_MTLResourceUsage.define('MTLResourceUsageSample', 4)

MTLResourceUsage = enum_MTLResourceUsage
class MTLIndirectCommandBuffer(objc.Spec): pass
enum_MTLBarrierScope = CEnum(NSUInteger)
MTLBarrierScopeBuffers = enum_MTLBarrierScope.define('MTLBarrierScopeBuffers', 1)
MTLBarrierScopeTextures = enum_MTLBarrierScope.define('MTLBarrierScopeTextures', 2)
MTLBarrierScopeRenderTargets = enum_MTLBarrierScope.define('MTLBarrierScopeRenderTargets', 4)

MTLBarrierScope = enum_MTLBarrierScope
class MTLCounterSampleBuffer(objc.Spec): pass
MTLCounterSampleBuffer._bases_ = [NSObject]
MTLCounterSampleBuffer._methods_ = [
  ('resolveCounterRange:', NSData, [NSRange]),
  ('device', MTLDevice, []),
  ('label', NSString, []),
  ('sampleCount', NSUInteger, []),
]
enum_MTLDispatchType = CEnum(NSUInteger)
MTLDispatchTypeSerial = enum_MTLDispatchType.define('MTLDispatchTypeSerial', 0)
MTLDispatchTypeConcurrent = enum_MTLDispatchType.define('MTLDispatchTypeConcurrent', 1)

MTLDispatchType = enum_MTLDispatchType
MTLComputeCommandEncoder._bases_ = [MTLCommandEncoder]
MTLComputeCommandEncoder._methods_ = [
  ('setComputePipelineState:', None, [MTLComputePipelineState]),
  ('setBytes:length:atIndex:', None, [ctypes.c_void_p, NSUInteger, NSUInteger]),
  ('setBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setBufferOffset:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setBuffers:offsets:withRange:', None, [ctypes.POINTER(MTLBuffer), ctypes.POINTER(NSUInteger), NSRange]),
  ('setBuffer:offset:attributeStride:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('setBuffers:offsets:attributeStrides:withRange:', None, [ctypes.POINTER(MTLBuffer), ctypes.POINTER(NSUInteger), ctypes.POINTER(NSUInteger), NSRange]),
  ('setBufferOffset:attributeStride:atIndex:', None, [NSUInteger, NSUInteger, NSUInteger]),
  ('setBytes:length:attributeStride:atIndex:', None, [ctypes.c_void_p, NSUInteger, NSUInteger, NSUInteger]),
  ('setVisibleFunctionTable:atBufferIndex:', None, [MTLVisibleFunctionTable, NSUInteger]),
  ('setVisibleFunctionTables:withBufferRange:', None, [ctypes.POINTER(MTLVisibleFunctionTable), NSRange]),
  ('setIntersectionFunctionTable:atBufferIndex:', None, [MTLIntersectionFunctionTable, NSUInteger]),
  ('setIntersectionFunctionTables:withBufferRange:', None, [ctypes.POINTER(MTLIntersectionFunctionTable), NSRange]),
  ('setAccelerationStructure:atBufferIndex:', None, [MTLAccelerationStructure, NSUInteger]),
  ('setTexture:atIndex:', None, [MTLTexture, NSUInteger]),
  ('setTextures:withRange:', None, [ctypes.POINTER(MTLTexture), NSRange]),
  ('setSamplerState:atIndex:', None, [MTLSamplerState, NSUInteger]),
  ('setSamplerStates:withRange:', None, [ctypes.POINTER(MTLSamplerState), NSRange]),
  ('setSamplerState:lodMinClamp:lodMaxClamp:atIndex:', None, [MTLSamplerState, ctypes.c_float, ctypes.c_float, NSUInteger]),
  ('setSamplerStates:lodMinClamps:lodMaxClamps:withRange:', None, [ctypes.POINTER(MTLSamplerState), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), NSRange]),
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
  ('useResources:count:usage:', None, [ctypes.POINTER(MTLResource), NSUInteger, MTLResourceUsage]),
  ('useHeap:', None, [MTLHeap]),
  ('useHeaps:count:', None, [ctypes.POINTER(MTLHeap), NSUInteger]),
  ('executeCommandsInBuffer:withRange:', None, [MTLIndirectCommandBuffer, NSRange]),
  ('executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:', None, [MTLIndirectCommandBuffer, MTLBuffer, NSUInteger]),
  ('memoryBarrierWithScope:', None, [MTLBarrierScope]),
  ('memoryBarrierWithResources:count:', None, [ctypes.POINTER(MTLResource), NSUInteger]),
  ('sampleCountersInBuffer:atSampleIndex:withBarrier:', None, [MTLCounterSampleBuffer, NSUInteger, BOOL]),
  ('dispatchType', MTLDispatchType, []),
]
class MTLComputePipelineReflection(objc.Spec): pass
MTLComputePipelineReflection._bases_ = [NSObject]
class MTLComputePipelineDescriptor(objc.Spec): pass
class MTLFunction(objc.Spec): pass
class MTLArgumentEncoder(objc.Spec): pass
class MTLArgument(objc.Spec): pass
enum_MTLArgumentType = CEnum(NSUInteger)
MTLArgumentTypeBuffer = enum_MTLArgumentType.define('MTLArgumentTypeBuffer', 0)
MTLArgumentTypeThreadgroupMemory = enum_MTLArgumentType.define('MTLArgumentTypeThreadgroupMemory', 1)
MTLArgumentTypeTexture = enum_MTLArgumentType.define('MTLArgumentTypeTexture', 2)
MTLArgumentTypeSampler = enum_MTLArgumentType.define('MTLArgumentTypeSampler', 3)
MTLArgumentTypeImageblockData = enum_MTLArgumentType.define('MTLArgumentTypeImageblockData', 16)
MTLArgumentTypeImageblock = enum_MTLArgumentType.define('MTLArgumentTypeImageblock', 17)
MTLArgumentTypeVisibleFunctionTable = enum_MTLArgumentType.define('MTLArgumentTypeVisibleFunctionTable', 24)
MTLArgumentTypePrimitiveAccelerationStructure = enum_MTLArgumentType.define('MTLArgumentTypePrimitiveAccelerationStructure', 25)
MTLArgumentTypeInstanceAccelerationStructure = enum_MTLArgumentType.define('MTLArgumentTypeInstanceAccelerationStructure', 26)
MTLArgumentTypeIntersectionFunctionTable = enum_MTLArgumentType.define('MTLArgumentTypeIntersectionFunctionTable', 27)

MTLArgumentType = enum_MTLArgumentType
enum_MTLBindingAccess = CEnum(NSUInteger)
MTLBindingAccessReadOnly = enum_MTLBindingAccess.define('MTLBindingAccessReadOnly', 0)
MTLBindingAccessReadWrite = enum_MTLBindingAccess.define('MTLBindingAccessReadWrite', 1)
MTLBindingAccessWriteOnly = enum_MTLBindingAccess.define('MTLBindingAccessWriteOnly', 2)
MTLArgumentAccessReadOnly = enum_MTLBindingAccess.define('MTLArgumentAccessReadOnly', 0)
MTLArgumentAccessReadWrite = enum_MTLBindingAccess.define('MTLArgumentAccessReadWrite', 1)
MTLArgumentAccessWriteOnly = enum_MTLBindingAccess.define('MTLArgumentAccessWriteOnly', 2)

MTLBindingAccess = enum_MTLBindingAccess
enum_MTLDataType = CEnum(NSUInteger)
MTLDataTypeNone = enum_MTLDataType.define('MTLDataTypeNone', 0)
MTLDataTypeStruct = enum_MTLDataType.define('MTLDataTypeStruct', 1)
MTLDataTypeArray = enum_MTLDataType.define('MTLDataTypeArray', 2)
MTLDataTypeFloat = enum_MTLDataType.define('MTLDataTypeFloat', 3)
MTLDataTypeFloat2 = enum_MTLDataType.define('MTLDataTypeFloat2', 4)
MTLDataTypeFloat3 = enum_MTLDataType.define('MTLDataTypeFloat3', 5)
MTLDataTypeFloat4 = enum_MTLDataType.define('MTLDataTypeFloat4', 6)
MTLDataTypeFloat2x2 = enum_MTLDataType.define('MTLDataTypeFloat2x2', 7)
MTLDataTypeFloat2x3 = enum_MTLDataType.define('MTLDataTypeFloat2x3', 8)
MTLDataTypeFloat2x4 = enum_MTLDataType.define('MTLDataTypeFloat2x4', 9)
MTLDataTypeFloat3x2 = enum_MTLDataType.define('MTLDataTypeFloat3x2', 10)
MTLDataTypeFloat3x3 = enum_MTLDataType.define('MTLDataTypeFloat3x3', 11)
MTLDataTypeFloat3x4 = enum_MTLDataType.define('MTLDataTypeFloat3x4', 12)
MTLDataTypeFloat4x2 = enum_MTLDataType.define('MTLDataTypeFloat4x2', 13)
MTLDataTypeFloat4x3 = enum_MTLDataType.define('MTLDataTypeFloat4x3', 14)
MTLDataTypeFloat4x4 = enum_MTLDataType.define('MTLDataTypeFloat4x4', 15)
MTLDataTypeHalf = enum_MTLDataType.define('MTLDataTypeHalf', 16)
MTLDataTypeHalf2 = enum_MTLDataType.define('MTLDataTypeHalf2', 17)
MTLDataTypeHalf3 = enum_MTLDataType.define('MTLDataTypeHalf3', 18)
MTLDataTypeHalf4 = enum_MTLDataType.define('MTLDataTypeHalf4', 19)
MTLDataTypeHalf2x2 = enum_MTLDataType.define('MTLDataTypeHalf2x2', 20)
MTLDataTypeHalf2x3 = enum_MTLDataType.define('MTLDataTypeHalf2x3', 21)
MTLDataTypeHalf2x4 = enum_MTLDataType.define('MTLDataTypeHalf2x4', 22)
MTLDataTypeHalf3x2 = enum_MTLDataType.define('MTLDataTypeHalf3x2', 23)
MTLDataTypeHalf3x3 = enum_MTLDataType.define('MTLDataTypeHalf3x3', 24)
MTLDataTypeHalf3x4 = enum_MTLDataType.define('MTLDataTypeHalf3x4', 25)
MTLDataTypeHalf4x2 = enum_MTLDataType.define('MTLDataTypeHalf4x2', 26)
MTLDataTypeHalf4x3 = enum_MTLDataType.define('MTLDataTypeHalf4x3', 27)
MTLDataTypeHalf4x4 = enum_MTLDataType.define('MTLDataTypeHalf4x4', 28)
MTLDataTypeInt = enum_MTLDataType.define('MTLDataTypeInt', 29)
MTLDataTypeInt2 = enum_MTLDataType.define('MTLDataTypeInt2', 30)
MTLDataTypeInt3 = enum_MTLDataType.define('MTLDataTypeInt3', 31)
MTLDataTypeInt4 = enum_MTLDataType.define('MTLDataTypeInt4', 32)
MTLDataTypeUInt = enum_MTLDataType.define('MTLDataTypeUInt', 33)
MTLDataTypeUInt2 = enum_MTLDataType.define('MTLDataTypeUInt2', 34)
MTLDataTypeUInt3 = enum_MTLDataType.define('MTLDataTypeUInt3', 35)
MTLDataTypeUInt4 = enum_MTLDataType.define('MTLDataTypeUInt4', 36)
MTLDataTypeShort = enum_MTLDataType.define('MTLDataTypeShort', 37)
MTLDataTypeShort2 = enum_MTLDataType.define('MTLDataTypeShort2', 38)
MTLDataTypeShort3 = enum_MTLDataType.define('MTLDataTypeShort3', 39)
MTLDataTypeShort4 = enum_MTLDataType.define('MTLDataTypeShort4', 40)
MTLDataTypeUShort = enum_MTLDataType.define('MTLDataTypeUShort', 41)
MTLDataTypeUShort2 = enum_MTLDataType.define('MTLDataTypeUShort2', 42)
MTLDataTypeUShort3 = enum_MTLDataType.define('MTLDataTypeUShort3', 43)
MTLDataTypeUShort4 = enum_MTLDataType.define('MTLDataTypeUShort4', 44)
MTLDataTypeChar = enum_MTLDataType.define('MTLDataTypeChar', 45)
MTLDataTypeChar2 = enum_MTLDataType.define('MTLDataTypeChar2', 46)
MTLDataTypeChar3 = enum_MTLDataType.define('MTLDataTypeChar3', 47)
MTLDataTypeChar4 = enum_MTLDataType.define('MTLDataTypeChar4', 48)
MTLDataTypeUChar = enum_MTLDataType.define('MTLDataTypeUChar', 49)
MTLDataTypeUChar2 = enum_MTLDataType.define('MTLDataTypeUChar2', 50)
MTLDataTypeUChar3 = enum_MTLDataType.define('MTLDataTypeUChar3', 51)
MTLDataTypeUChar4 = enum_MTLDataType.define('MTLDataTypeUChar4', 52)
MTLDataTypeBool = enum_MTLDataType.define('MTLDataTypeBool', 53)
MTLDataTypeBool2 = enum_MTLDataType.define('MTLDataTypeBool2', 54)
MTLDataTypeBool3 = enum_MTLDataType.define('MTLDataTypeBool3', 55)
MTLDataTypeBool4 = enum_MTLDataType.define('MTLDataTypeBool4', 56)
MTLDataTypeTexture = enum_MTLDataType.define('MTLDataTypeTexture', 58)
MTLDataTypeSampler = enum_MTLDataType.define('MTLDataTypeSampler', 59)
MTLDataTypePointer = enum_MTLDataType.define('MTLDataTypePointer', 60)
MTLDataTypeR8Unorm = enum_MTLDataType.define('MTLDataTypeR8Unorm', 62)
MTLDataTypeR8Snorm = enum_MTLDataType.define('MTLDataTypeR8Snorm', 63)
MTLDataTypeR16Unorm = enum_MTLDataType.define('MTLDataTypeR16Unorm', 64)
MTLDataTypeR16Snorm = enum_MTLDataType.define('MTLDataTypeR16Snorm', 65)
MTLDataTypeRG8Unorm = enum_MTLDataType.define('MTLDataTypeRG8Unorm', 66)
MTLDataTypeRG8Snorm = enum_MTLDataType.define('MTLDataTypeRG8Snorm', 67)
MTLDataTypeRG16Unorm = enum_MTLDataType.define('MTLDataTypeRG16Unorm', 68)
MTLDataTypeRG16Snorm = enum_MTLDataType.define('MTLDataTypeRG16Snorm', 69)
MTLDataTypeRGBA8Unorm = enum_MTLDataType.define('MTLDataTypeRGBA8Unorm', 70)
MTLDataTypeRGBA8Unorm_sRGB = enum_MTLDataType.define('MTLDataTypeRGBA8Unorm_sRGB', 71)
MTLDataTypeRGBA8Snorm = enum_MTLDataType.define('MTLDataTypeRGBA8Snorm', 72)
MTLDataTypeRGBA16Unorm = enum_MTLDataType.define('MTLDataTypeRGBA16Unorm', 73)
MTLDataTypeRGBA16Snorm = enum_MTLDataType.define('MTLDataTypeRGBA16Snorm', 74)
MTLDataTypeRGB10A2Unorm = enum_MTLDataType.define('MTLDataTypeRGB10A2Unorm', 75)
MTLDataTypeRG11B10Float = enum_MTLDataType.define('MTLDataTypeRG11B10Float', 76)
MTLDataTypeRGB9E5Float = enum_MTLDataType.define('MTLDataTypeRGB9E5Float', 77)
MTLDataTypeRenderPipeline = enum_MTLDataType.define('MTLDataTypeRenderPipeline', 78)
MTLDataTypeComputePipeline = enum_MTLDataType.define('MTLDataTypeComputePipeline', 79)
MTLDataTypeIndirectCommandBuffer = enum_MTLDataType.define('MTLDataTypeIndirectCommandBuffer', 80)
MTLDataTypeLong = enum_MTLDataType.define('MTLDataTypeLong', 81)
MTLDataTypeLong2 = enum_MTLDataType.define('MTLDataTypeLong2', 82)
MTLDataTypeLong3 = enum_MTLDataType.define('MTLDataTypeLong3', 83)
MTLDataTypeLong4 = enum_MTLDataType.define('MTLDataTypeLong4', 84)
MTLDataTypeULong = enum_MTLDataType.define('MTLDataTypeULong', 85)
MTLDataTypeULong2 = enum_MTLDataType.define('MTLDataTypeULong2', 86)
MTLDataTypeULong3 = enum_MTLDataType.define('MTLDataTypeULong3', 87)
MTLDataTypeULong4 = enum_MTLDataType.define('MTLDataTypeULong4', 88)
MTLDataTypeVisibleFunctionTable = enum_MTLDataType.define('MTLDataTypeVisibleFunctionTable', 115)
MTLDataTypeIntersectionFunctionTable = enum_MTLDataType.define('MTLDataTypeIntersectionFunctionTable', 116)
MTLDataTypePrimitiveAccelerationStructure = enum_MTLDataType.define('MTLDataTypePrimitiveAccelerationStructure', 117)
MTLDataTypeInstanceAccelerationStructure = enum_MTLDataType.define('MTLDataTypeInstanceAccelerationStructure', 118)
MTLDataTypeBFloat = enum_MTLDataType.define('MTLDataTypeBFloat', 121)
MTLDataTypeBFloat2 = enum_MTLDataType.define('MTLDataTypeBFloat2', 122)
MTLDataTypeBFloat3 = enum_MTLDataType.define('MTLDataTypeBFloat3', 123)
MTLDataTypeBFloat4 = enum_MTLDataType.define('MTLDataTypeBFloat4', 124)

MTLDataType = enum_MTLDataType
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
enum_MTLFunctionType = CEnum(NSUInteger)
MTLFunctionTypeVertex = enum_MTLFunctionType.define('MTLFunctionTypeVertex', 1)
MTLFunctionTypeFragment = enum_MTLFunctionType.define('MTLFunctionTypeFragment', 2)
MTLFunctionTypeKernel = enum_MTLFunctionType.define('MTLFunctionTypeKernel', 3)
MTLFunctionTypeVisible = enum_MTLFunctionType.define('MTLFunctionTypeVisible', 5)
MTLFunctionTypeIntersection = enum_MTLFunctionType.define('MTLFunctionTypeIntersection', 6)
MTLFunctionTypeMesh = enum_MTLFunctionType.define('MTLFunctionTypeMesh', 7)
MTLFunctionTypeObject = enum_MTLFunctionType.define('MTLFunctionTypeObject', 8)

MTLFunctionType = enum_MTLFunctionType
enum_MTLPatchType = CEnum(NSUInteger)
MTLPatchTypeNone = enum_MTLPatchType.define('MTLPatchTypeNone', 0)
MTLPatchTypeTriangle = enum_MTLPatchType.define('MTLPatchTypeTriangle', 1)
MTLPatchTypeQuad = enum_MTLPatchType.define('MTLPatchTypeQuad', 2)

MTLPatchType = enum_MTLPatchType
enum_MTLFunctionOptions = CEnum(NSUInteger)
MTLFunctionOptionNone = enum_MTLFunctionOptions.define('MTLFunctionOptionNone', 0)
MTLFunctionOptionCompileToBinary = enum_MTLFunctionOptions.define('MTLFunctionOptionCompileToBinary', 1)
MTLFunctionOptionStoreFunctionInMetalScript = enum_MTLFunctionOptions.define('MTLFunctionOptionStoreFunctionInMetalScript', 2)

MTLFunctionOptions = enum_MTLFunctionOptions
MTLFunction._bases_ = [NSObject]
MTLFunction._methods_ = [
  ('newArgumentEncoderWithBufferIndex:', MTLArgumentEncoder, [NSUInteger], True),
  ('newArgumentEncoderWithBufferIndex:reflection:', MTLArgumentEncoder, [NSUInteger, ctypes.POINTER(MTLArgument)], True),
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
enum_MTLStepFunction = CEnum(NSUInteger)
MTLStepFunctionConstant = enum_MTLStepFunction.define('MTLStepFunctionConstant', 0)
MTLStepFunctionPerVertex = enum_MTLStepFunction.define('MTLStepFunctionPerVertex', 1)
MTLStepFunctionPerInstance = enum_MTLStepFunction.define('MTLStepFunctionPerInstance', 2)
MTLStepFunctionPerPatch = enum_MTLStepFunction.define('MTLStepFunctionPerPatch', 3)
MTLStepFunctionPerPatchControlPoint = enum_MTLStepFunction.define('MTLStepFunctionPerPatchControlPoint', 4)
MTLStepFunctionThreadPositionInGridX = enum_MTLStepFunction.define('MTLStepFunctionThreadPositionInGridX', 5)
MTLStepFunctionThreadPositionInGridY = enum_MTLStepFunction.define('MTLStepFunctionThreadPositionInGridY', 6)
MTLStepFunctionThreadPositionInGridXIndexed = enum_MTLStepFunction.define('MTLStepFunctionThreadPositionInGridXIndexed', 7)
MTLStepFunctionThreadPositionInGridYIndexed = enum_MTLStepFunction.define('MTLStepFunctionThreadPositionInGridYIndexed', 8)

MTLStepFunction = enum_MTLStepFunction
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
enum_MTLAttributeFormat = CEnum(NSUInteger)
MTLAttributeFormatInvalid = enum_MTLAttributeFormat.define('MTLAttributeFormatInvalid', 0)
MTLAttributeFormatUChar2 = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar2', 1)
MTLAttributeFormatUChar3 = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar3', 2)
MTLAttributeFormatUChar4 = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar4', 3)
MTLAttributeFormatChar2 = enum_MTLAttributeFormat.define('MTLAttributeFormatChar2', 4)
MTLAttributeFormatChar3 = enum_MTLAttributeFormat.define('MTLAttributeFormatChar3', 5)
MTLAttributeFormatChar4 = enum_MTLAttributeFormat.define('MTLAttributeFormatChar4', 6)
MTLAttributeFormatUChar2Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar2Normalized', 7)
MTLAttributeFormatUChar3Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar3Normalized', 8)
MTLAttributeFormatUChar4Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar4Normalized', 9)
MTLAttributeFormatChar2Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatChar2Normalized', 10)
MTLAttributeFormatChar3Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatChar3Normalized', 11)
MTLAttributeFormatChar4Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatChar4Normalized', 12)
MTLAttributeFormatUShort2 = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort2', 13)
MTLAttributeFormatUShort3 = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort3', 14)
MTLAttributeFormatUShort4 = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort4', 15)
MTLAttributeFormatShort2 = enum_MTLAttributeFormat.define('MTLAttributeFormatShort2', 16)
MTLAttributeFormatShort3 = enum_MTLAttributeFormat.define('MTLAttributeFormatShort3', 17)
MTLAttributeFormatShort4 = enum_MTLAttributeFormat.define('MTLAttributeFormatShort4', 18)
MTLAttributeFormatUShort2Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort2Normalized', 19)
MTLAttributeFormatUShort3Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort3Normalized', 20)
MTLAttributeFormatUShort4Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort4Normalized', 21)
MTLAttributeFormatShort2Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatShort2Normalized', 22)
MTLAttributeFormatShort3Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatShort3Normalized', 23)
MTLAttributeFormatShort4Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatShort4Normalized', 24)
MTLAttributeFormatHalf2 = enum_MTLAttributeFormat.define('MTLAttributeFormatHalf2', 25)
MTLAttributeFormatHalf3 = enum_MTLAttributeFormat.define('MTLAttributeFormatHalf3', 26)
MTLAttributeFormatHalf4 = enum_MTLAttributeFormat.define('MTLAttributeFormatHalf4', 27)
MTLAttributeFormatFloat = enum_MTLAttributeFormat.define('MTLAttributeFormatFloat', 28)
MTLAttributeFormatFloat2 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloat2', 29)
MTLAttributeFormatFloat3 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloat3', 30)
MTLAttributeFormatFloat4 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloat4', 31)
MTLAttributeFormatInt = enum_MTLAttributeFormat.define('MTLAttributeFormatInt', 32)
MTLAttributeFormatInt2 = enum_MTLAttributeFormat.define('MTLAttributeFormatInt2', 33)
MTLAttributeFormatInt3 = enum_MTLAttributeFormat.define('MTLAttributeFormatInt3', 34)
MTLAttributeFormatInt4 = enum_MTLAttributeFormat.define('MTLAttributeFormatInt4', 35)
MTLAttributeFormatUInt = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt', 36)
MTLAttributeFormatUInt2 = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt2', 37)
MTLAttributeFormatUInt3 = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt3', 38)
MTLAttributeFormatUInt4 = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt4', 39)
MTLAttributeFormatInt1010102Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatInt1010102Normalized', 40)
MTLAttributeFormatUInt1010102Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt1010102Normalized', 41)
MTLAttributeFormatUChar4Normalized_BGRA = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar4Normalized_BGRA', 42)
MTLAttributeFormatUChar = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar', 45)
MTLAttributeFormatChar = enum_MTLAttributeFormat.define('MTLAttributeFormatChar', 46)
MTLAttributeFormatUCharNormalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUCharNormalized', 47)
MTLAttributeFormatCharNormalized = enum_MTLAttributeFormat.define('MTLAttributeFormatCharNormalized', 48)
MTLAttributeFormatUShort = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort', 49)
MTLAttributeFormatShort = enum_MTLAttributeFormat.define('MTLAttributeFormatShort', 50)
MTLAttributeFormatUShortNormalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUShortNormalized', 51)
MTLAttributeFormatShortNormalized = enum_MTLAttributeFormat.define('MTLAttributeFormatShortNormalized', 52)
MTLAttributeFormatHalf = enum_MTLAttributeFormat.define('MTLAttributeFormatHalf', 53)
MTLAttributeFormatFloatRG11B10 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloatRG11B10', 54)
MTLAttributeFormatFloatRGB9E5 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloatRGB9E5', 55)

MTLAttributeFormat = enum_MTLAttributeFormat
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
enum_MTLIndexType = CEnum(NSUInteger)
MTLIndexTypeUInt16 = enum_MTLIndexType.define('MTLIndexTypeUInt16', 0)
MTLIndexTypeUInt32 = enum_MTLIndexType.define('MTLIndexTypeUInt32', 1)

MTLIndexType = enum_MTLIndexType
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
enum_MTLMutability = CEnum(NSUInteger)
MTLMutabilityDefault = enum_MTLMutability.define('MTLMutabilityDefault', 0)
MTLMutabilityMutable = enum_MTLMutability.define('MTLMutabilityMutable', 1)
MTLMutabilityImmutable = enum_MTLMutability.define('MTLMutabilityImmutable', 2)

MTLMutability = enum_MTLMutability
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
class struct_MTLResourceID(Struct): pass
MTLResourceID = struct_MTLResourceID
struct_MTLResourceID._fields_ = [
  ('_impl', uint64_t),
]
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
CFTimeInterval = ctypes.c_double
class MTLBlitCommandEncoder(objc.Spec): pass
enum_MTLBlitOption = CEnum(NSUInteger)
MTLBlitOptionNone = enum_MTLBlitOption.define('MTLBlitOptionNone', 0)
MTLBlitOptionDepthFromDepthStencil = enum_MTLBlitOption.define('MTLBlitOptionDepthFromDepthStencil', 1)
MTLBlitOptionStencilFromDepthStencil = enum_MTLBlitOption.define('MTLBlitOptionStencilFromDepthStencil', 2)
MTLBlitOptionRowLinearPVRTC = enum_MTLBlitOption.define('MTLBlitOptionRowLinearPVRTC', 4)

MTLBlitOption = enum_MTLBlitOption
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
class MTLSamplePosition(Struct): pass
MTLSamplePosition._fields_ = [
  ('x', ctypes.c_float),
  ('y', ctypes.c_float),
]
class MTLRenderPassColorAttachmentDescriptorArray(objc.Spec): pass
class MTLRenderPassColorAttachmentDescriptor(objc.Spec): pass
class MTLClearColor(Struct): pass
MTLClearColor._fields_ = [
  ('red', ctypes.c_double),
  ('green', ctypes.c_double),
  ('blue', ctypes.c_double),
  ('alpha', ctypes.c_double),
]
class MTLRenderPassAttachmentDescriptor(objc.Spec): pass
enum_MTLLoadAction = CEnum(NSUInteger)
MTLLoadActionDontCare = enum_MTLLoadAction.define('MTLLoadActionDontCare', 0)
MTLLoadActionLoad = enum_MTLLoadAction.define('MTLLoadActionLoad', 1)
MTLLoadActionClear = enum_MTLLoadAction.define('MTLLoadActionClear', 2)

MTLLoadAction = enum_MTLLoadAction
enum_MTLStoreAction = CEnum(NSUInteger)
MTLStoreActionDontCare = enum_MTLStoreAction.define('MTLStoreActionDontCare', 0)
MTLStoreActionStore = enum_MTLStoreAction.define('MTLStoreActionStore', 1)
MTLStoreActionMultisampleResolve = enum_MTLStoreAction.define('MTLStoreActionMultisampleResolve', 2)
MTLStoreActionStoreAndMultisampleResolve = enum_MTLStoreAction.define('MTLStoreActionStoreAndMultisampleResolve', 3)
MTLStoreActionUnknown = enum_MTLStoreAction.define('MTLStoreActionUnknown', 4)
MTLStoreActionCustomSampleDepthStore = enum_MTLStoreAction.define('MTLStoreActionCustomSampleDepthStore', 5)

MTLStoreAction = enum_MTLStoreAction
enum_MTLStoreActionOptions = CEnum(NSUInteger)
MTLStoreActionOptionNone = enum_MTLStoreActionOptions.define('MTLStoreActionOptionNone', 0)
MTLStoreActionOptionCustomSamplePositions = enum_MTLStoreActionOptions.define('MTLStoreActionOptionCustomSamplePositions', 1)

MTLStoreActionOptions = enum_MTLStoreActionOptions
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
enum_MTLMultisampleDepthResolveFilter = CEnum(NSUInteger)
MTLMultisampleDepthResolveFilterSample0 = enum_MTLMultisampleDepthResolveFilter.define('MTLMultisampleDepthResolveFilterSample0', 0)
MTLMultisampleDepthResolveFilterMin = enum_MTLMultisampleDepthResolveFilter.define('MTLMultisampleDepthResolveFilterMin', 1)
MTLMultisampleDepthResolveFilterMax = enum_MTLMultisampleDepthResolveFilter.define('MTLMultisampleDepthResolveFilterMax', 2)

MTLMultisampleDepthResolveFilter = enum_MTLMultisampleDepthResolveFilter
MTLRenderPassDepthAttachmentDescriptor._bases_ = [MTLRenderPassAttachmentDescriptor]
MTLRenderPassDepthAttachmentDescriptor._methods_ = [
  ('clearDepth', ctypes.c_double, []),
  ('setClearDepth:', None, [ctypes.c_double]),
  ('depthResolveFilter', MTLMultisampleDepthResolveFilter, []),
  ('setDepthResolveFilter:', None, [MTLMultisampleDepthResolveFilter]),
]
class MTLRenderPassStencilAttachmentDescriptor(objc.Spec): pass
enum_MTLMultisampleStencilResolveFilter = CEnum(NSUInteger)
MTLMultisampleStencilResolveFilterSample0 = enum_MTLMultisampleStencilResolveFilter.define('MTLMultisampleStencilResolveFilterSample0', 0)
MTLMultisampleStencilResolveFilterDepthResolvedSample = enum_MTLMultisampleStencilResolveFilter.define('MTLMultisampleStencilResolveFilterDepthResolvedSample', 1)

MTLMultisampleStencilResolveFilter = enum_MTLMultisampleStencilResolveFilter
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
  ('setSamplePositions:count:', None, [ctypes.POINTER(MTLSamplePosition), NSUInteger]),
  ('getSamplePositions:count:', NSUInteger, [ctypes.POINTER(MTLSamplePosition), NSUInteger]),
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
enum_MTLSparseTextureMappingMode = CEnum(NSUInteger)
MTLSparseTextureMappingModeMap = enum_MTLSparseTextureMappingMode.define('MTLSparseTextureMappingModeMap', 0)
MTLSparseTextureMappingModeUnmap = enum_MTLSparseTextureMappingMode.define('MTLSparseTextureMappingModeUnmap', 1)

MTLSparseTextureMappingMode = enum_MTLSparseTextureMappingMode
MTLResourceStateCommandEncoder._bases_ = [MTLCommandEncoder]
MTLResourceStateCommandEncoder._methods_ = [
  ('updateTextureMappings:mode:regions:mipLevels:slices:numRegions:', None, [MTLTexture, MTLSparseTextureMappingMode, ctypes.POINTER(MTLRegion), ctypes.POINTER(NSUInteger), ctypes.POINTER(NSUInteger), NSUInteger]),
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
enum_MTLCommandBufferErrorOption = CEnum(NSUInteger)
MTLCommandBufferErrorOptionNone = enum_MTLCommandBufferErrorOption.define('MTLCommandBufferErrorOptionNone', 0)
MTLCommandBufferErrorOptionEncoderExecutionStatus = enum_MTLCommandBufferErrorOption.define('MTLCommandBufferErrorOptionEncoderExecutionStatus', 1)

MTLCommandBufferErrorOption = enum_MTLCommandBufferErrorOption
class MTLLogContainer(objc.Spec): pass
enum_MTLCommandBufferStatus = CEnum(NSUInteger)
MTLCommandBufferStatusNotEnqueued = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusNotEnqueued', 0)
MTLCommandBufferStatusEnqueued = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusEnqueued', 1)
MTLCommandBufferStatusCommitted = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusCommitted', 2)
MTLCommandBufferStatusScheduled = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusScheduled', 3)
MTLCommandBufferStatusCompleted = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusCompleted', 4)
MTLCommandBufferStatusError = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusError', 5)

MTLCommandBufferStatus = enum_MTLCommandBufferStatus
class NSError(objc.Spec): pass
NSErrorDomain = NSString
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
enum_MTLIOCompressionMethod = CEnum(NSInteger)
MTLIOCompressionMethodZlib = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodZlib', 0)
MTLIOCompressionMethodLZFSE = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZFSE', 1)
MTLIOCompressionMethodLZ4 = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZ4', 2)
MTLIOCompressionMethodLZMA = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZMA', 3)
MTLIOCompressionMethodLZBitmap = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZBitmap', 4)

MTLIOCompressionMethod = enum_MTLIOCompressionMethod
try: (MTLCreateSystemDefaultDevice:=dll.MTLCreateSystemDefaultDevice).restype, MTLCreateSystemDefaultDevice.argtypes = MTLDevice, []
except AttributeError: pass

MTLCreateSystemDefaultDevice = objc.returns_retained(MTLCreateSystemDefaultDevice)
MTLDeviceNotificationName = NSString
try: MTLDeviceWasAddedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasAddedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceRemovalRequestedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceRemovalRequestedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceWasRemovedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasRemovedNotification')
except (ValueError,AttributeError): pass
try: (MTLRemoveDeviceObserver:=dll.MTLRemoveDeviceObserver).restype, MTLRemoveDeviceObserver.argtypes = None, [NSObject]
except AttributeError: pass

enum_MTLFeatureSet = CEnum(NSUInteger)
MTLFeatureSet_iOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v1', 0)
MTLFeatureSet_iOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v1', 1)
MTLFeatureSet_iOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v2', 2)
MTLFeatureSet_iOS_GPUFamily2_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v2', 3)
MTLFeatureSet_iOS_GPUFamily3_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v1', 4)
MTLFeatureSet_iOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v3', 5)
MTLFeatureSet_iOS_GPUFamily2_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v3', 6)
MTLFeatureSet_iOS_GPUFamily3_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v2', 7)
MTLFeatureSet_iOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v4', 8)
MTLFeatureSet_iOS_GPUFamily2_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v4', 9)
MTLFeatureSet_iOS_GPUFamily3_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v3', 10)
MTLFeatureSet_iOS_GPUFamily4_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily4_v1', 11)
MTLFeatureSet_iOS_GPUFamily1_v5 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v5', 12)
MTLFeatureSet_iOS_GPUFamily2_v5 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v5', 13)
MTLFeatureSet_iOS_GPUFamily3_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v4', 14)
MTLFeatureSet_iOS_GPUFamily4_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily4_v2', 15)
MTLFeatureSet_iOS_GPUFamily5_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily5_v1', 16)
MTLFeatureSet_macOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v1', 10000)
MTLFeatureSet_OSX_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_GPUFamily1_v1', 10000)
MTLFeatureSet_macOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v2', 10001)
MTLFeatureSet_OSX_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_GPUFamily1_v2', 10001)
MTLFeatureSet_macOS_ReadWriteTextureTier2 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_ReadWriteTextureTier2', 10002)
MTLFeatureSet_OSX_ReadWriteTextureTier2 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_ReadWriteTextureTier2', 10002)
MTLFeatureSet_macOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v3', 10003)
MTLFeatureSet_macOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v4', 10004)
MTLFeatureSet_macOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily2_v1', 10005)
MTLFeatureSet_tvOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v1', 30000)
MTLFeatureSet_TVOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_TVOS_GPUFamily1_v1', 30000)
MTLFeatureSet_tvOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v2', 30001)
MTLFeatureSet_tvOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v3', 30002)
MTLFeatureSet_tvOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily2_v1', 30003)
MTLFeatureSet_tvOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v4', 30004)
MTLFeatureSet_tvOS_GPUFamily2_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily2_v2', 30005)

MTLFeatureSet = enum_MTLFeatureSet
enum_MTLGPUFamily = CEnum(NSInteger)
MTLGPUFamilyApple1 = enum_MTLGPUFamily.define('MTLGPUFamilyApple1', 1001)
MTLGPUFamilyApple2 = enum_MTLGPUFamily.define('MTLGPUFamilyApple2', 1002)
MTLGPUFamilyApple3 = enum_MTLGPUFamily.define('MTLGPUFamilyApple3', 1003)
MTLGPUFamilyApple4 = enum_MTLGPUFamily.define('MTLGPUFamilyApple4', 1004)
MTLGPUFamilyApple5 = enum_MTLGPUFamily.define('MTLGPUFamilyApple5', 1005)
MTLGPUFamilyApple6 = enum_MTLGPUFamily.define('MTLGPUFamilyApple6', 1006)
MTLGPUFamilyApple7 = enum_MTLGPUFamily.define('MTLGPUFamilyApple7', 1007)
MTLGPUFamilyApple8 = enum_MTLGPUFamily.define('MTLGPUFamilyApple8', 1008)
MTLGPUFamilyApple9 = enum_MTLGPUFamily.define('MTLGPUFamilyApple9', 1009)
MTLGPUFamilyMac1 = enum_MTLGPUFamily.define('MTLGPUFamilyMac1', 2001)
MTLGPUFamilyMac2 = enum_MTLGPUFamily.define('MTLGPUFamilyMac2', 2002)
MTLGPUFamilyCommon1 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon1', 3001)
MTLGPUFamilyCommon2 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon2', 3002)
MTLGPUFamilyCommon3 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon3', 3003)
MTLGPUFamilyMacCatalyst1 = enum_MTLGPUFamily.define('MTLGPUFamilyMacCatalyst1', 4001)
MTLGPUFamilyMacCatalyst2 = enum_MTLGPUFamily.define('MTLGPUFamilyMacCatalyst2', 4002)
MTLGPUFamilyMetal3 = enum_MTLGPUFamily.define('MTLGPUFamilyMetal3', 5001)

MTLGPUFamily = enum_MTLGPUFamily
enum_MTLDeviceLocation = CEnum(NSUInteger)
MTLDeviceLocationBuiltIn = enum_MTLDeviceLocation.define('MTLDeviceLocationBuiltIn', 0)
MTLDeviceLocationSlot = enum_MTLDeviceLocation.define('MTLDeviceLocationSlot', 1)
MTLDeviceLocationExternal = enum_MTLDeviceLocation.define('MTLDeviceLocationExternal', 2)
MTLDeviceLocationUnspecified = enum_MTLDeviceLocation.define('MTLDeviceLocationUnspecified', -1)

MTLDeviceLocation = enum_MTLDeviceLocation
enum_MTLPipelineOption = CEnum(NSUInteger)
MTLPipelineOptionNone = enum_MTLPipelineOption.define('MTLPipelineOptionNone', 0)
MTLPipelineOptionArgumentInfo = enum_MTLPipelineOption.define('MTLPipelineOptionArgumentInfo', 1)
MTLPipelineOptionBufferTypeInfo = enum_MTLPipelineOption.define('MTLPipelineOptionBufferTypeInfo', 2)
MTLPipelineOptionFailOnBinaryArchiveMiss = enum_MTLPipelineOption.define('MTLPipelineOptionFailOnBinaryArchiveMiss', 4)

MTLPipelineOption = enum_MTLPipelineOption
enum_MTLReadWriteTextureTier = CEnum(NSUInteger)
MTLReadWriteTextureTierNone = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTierNone', 0)
MTLReadWriteTextureTier1 = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTier1', 1)
MTLReadWriteTextureTier2 = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTier2', 2)

MTLReadWriteTextureTier = enum_MTLReadWriteTextureTier
enum_MTLArgumentBuffersTier = CEnum(NSUInteger)
MTLArgumentBuffersTier1 = enum_MTLArgumentBuffersTier.define('MTLArgumentBuffersTier1', 0)
MTLArgumentBuffersTier2 = enum_MTLArgumentBuffersTier.define('MTLArgumentBuffersTier2', 1)

MTLArgumentBuffersTier = enum_MTLArgumentBuffersTier
enum_MTLSparseTextureRegionAlignmentMode = CEnum(NSUInteger)
MTLSparseTextureRegionAlignmentModeOutward = enum_MTLSparseTextureRegionAlignmentMode.define('MTLSparseTextureRegionAlignmentModeOutward', 0)
MTLSparseTextureRegionAlignmentModeInward = enum_MTLSparseTextureRegionAlignmentMode.define('MTLSparseTextureRegionAlignmentModeInward', 1)

MTLSparseTextureRegionAlignmentMode = enum_MTLSparseTextureRegionAlignmentMode
enum_MTLSparsePageSize = CEnum(NSInteger)
MTLSparsePageSize16 = enum_MTLSparsePageSize.define('MTLSparsePageSize16', 101)
MTLSparsePageSize64 = enum_MTLSparsePageSize.define('MTLSparsePageSize64', 102)
MTLSparsePageSize256 = enum_MTLSparsePageSize.define('MTLSparsePageSize256', 103)

MTLSparsePageSize = enum_MTLSparsePageSize
class MTLAccelerationStructureSizes(Struct): pass
MTLAccelerationStructureSizes._fields_ = [
  ('accelerationStructureSize', NSUInteger),
  ('buildScratchBufferSize', NSUInteger),
  ('refitScratchBufferSize', NSUInteger),
]
enum_MTLCounterSamplingPoint = CEnum(NSUInteger)
MTLCounterSamplingPointAtStageBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtStageBoundary', 0)
MTLCounterSamplingPointAtDrawBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtDrawBoundary', 1)
MTLCounterSamplingPointAtDispatchBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtDispatchBoundary', 2)
MTLCounterSamplingPointAtTileDispatchBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtTileDispatchBoundary', 3)
MTLCounterSamplingPointAtBlitBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtBlitBoundary', 4)

MTLCounterSamplingPoint = enum_MTLCounterSamplingPoint
class MTLSizeAndAlign(Struct): pass
MTLSizeAndAlign._fields_ = [
  ('size', NSUInteger),
  ('align', NSUInteger),
]
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
class struct___IOSurface(Struct): pass
IOSurfaceRef = ctypes.POINTER(struct___IOSurface)
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
enum_MTLLibraryType = CEnum(NSInteger)
MTLLibraryTypeExecutable = enum_MTLLibraryType.define('MTLLibraryTypeExecutable', 0)
MTLLibraryTypeDynamic = enum_MTLLibraryType.define('MTLLibraryTypeDynamic', 1)

MTLLibraryType = enum_MTLLibraryType
MTLLibrary._bases_ = [NSObject]
MTLLibrary._methods_ = [
  ('newFunctionWithName:', MTLFunction, [NSString], True),
  ('newFunctionWithName:constantValues:error:', MTLFunction, [NSString, MTLFunctionConstantValues, ctypes.POINTER(NSError)], True),
  ('newFunctionWithDescriptor:error:', MTLFunction, [MTLFunctionDescriptor, ctypes.POINTER(NSError)], True),
  ('newIntersectionFunctionWithDescriptor:error:', MTLFunction, [MTLIntersectionFunctionDescriptor, ctypes.POINTER(NSError)], True),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
  ('type', MTLLibraryType, []),
  ('installName', NSString, []),
]
class NSBundle(objc.Spec): pass
class NSURL(objc.Spec): pass
NSURLResourceKey = NSString
enum_NSURLBookmarkCreationOptions = CEnum(NSUInteger)
NSURLBookmarkCreationPreferFileIDResolution = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationPreferFileIDResolution', 256)
NSURLBookmarkCreationMinimalBookmark = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationMinimalBookmark', 512)
NSURLBookmarkCreationSuitableForBookmarkFile = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationSuitableForBookmarkFile', 1024)
NSURLBookmarkCreationWithSecurityScope = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationWithSecurityScope', 2048)
NSURLBookmarkCreationSecurityScopeAllowOnlyReadAccess = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationSecurityScopeAllowOnlyReadAccess', 4096)
NSURLBookmarkCreationWithoutImplicitSecurityScope = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationWithoutImplicitSecurityScope', 536870912)

NSURLBookmarkCreationOptions = enum_NSURLBookmarkCreationOptions
enum_NSURLBookmarkResolutionOptions = CEnum(NSUInteger)
NSURLBookmarkResolutionWithoutUI = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutUI', 256)
NSURLBookmarkResolutionWithoutMounting = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutMounting', 512)
NSURLBookmarkResolutionWithSecurityScope = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithSecurityScope', 1024)
NSURLBookmarkResolutionWithoutImplicitStartAccessing = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutImplicitStartAccessing', 32768)

NSURLBookmarkResolutionOptions = enum_NSURLBookmarkResolutionOptions
class NSNumber(objc.Spec): pass
enum_NSComparisonResult = CEnum(NSInteger)
NSOrderedAscending = enum_NSComparisonResult.define('NSOrderedAscending', -1)
NSOrderedSame = enum_NSComparisonResult.define('NSOrderedSame', 0)
NSOrderedDescending = enum_NSComparisonResult.define('NSOrderedDescending', 1)

NSComparisonResult = enum_NSComparisonResult
class NSValue(objc.Spec): pass
NSValue._bases_ = [NSObject]
NSValue._methods_ = [
  ('getValue:size:', None, [ctypes.c_void_p, NSUInteger]),
  ('initWithBytes:objCType:', 'instancetype', [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]),
  ('initWithCoder:', 'instancetype', [NSCoder]),
  ('objCType', ctypes.POINTER(ctypes.c_char), []),
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
NSURLBookmarkFileCreationOptions = ctypes.c_uint64
NSURL._bases_ = [NSObject]
NSURL._methods_ = [
  ('initWithScheme:host:path:', 'instancetype', [NSString, NSString, NSString]),
  ('initFileURLWithPath:isDirectory:relativeToURL:', 'instancetype', [NSString, BOOL, NSURL]),
  ('initFileURLWithPath:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('initFileURLWithPath:isDirectory:', 'instancetype', [NSString, BOOL]),
  ('initFileURLWithPath:', 'instancetype', [NSString]),
  ('initFileURLWithFileSystemRepresentation:isDirectory:relativeToURL:', 'instancetype', [ctypes.POINTER(ctypes.c_char), BOOL, NSURL]),
  ('initWithString:', 'instancetype', [NSString]),
  ('initWithString:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('initWithString:encodingInvalidCharacters:', 'instancetype', [NSString, BOOL]),
  ('initWithDataRepresentation:relativeToURL:', 'instancetype', [NSData, NSURL]),
  ('initAbsoluteURLWithDataRepresentation:relativeToURL:', 'instancetype', [NSData, NSURL]),
  ('getFileSystemRepresentation:maxLength:', BOOL, [ctypes.POINTER(ctypes.c_char), NSUInteger]),
  ('isFileReferenceURL', BOOL, []),
  ('fileReferenceURL', NSURL, []),
  ('getResourceValue:forKey:error:', BOOL, [ctypes.POINTER(objc.id_), NSURLResourceKey, ctypes.POINTER(NSError)]),
  ('setResourceValue:forKey:error:', BOOL, [objc.id_, NSURLResourceKey, ctypes.POINTER(NSError)]),
  ('removeCachedResourceValueForKey:', None, [NSURLResourceKey]),
  ('removeAllCachedResourceValues', None, []),
  ('setTemporaryResourceValue:forKey:', None, [objc.id_, NSURLResourceKey]),
  ('initByResolvingBookmarkData:options:relativeToURL:bookmarkDataIsStale:error:', 'instancetype', [NSData, NSURLBookmarkResolutionOptions, NSURL, ctypes.POINTER(BOOL), ctypes.POINTER(NSError)]),
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
  ('fileSystemRepresentation', ctypes.POINTER(ctypes.c_char), []),
  ('isFileURL', BOOL, []),
  ('standardizedURL', NSURL, []),
  ('filePathURL', NSURL, []),
]
NSURL._classmethods_ = [
  ('fileURLWithPath:isDirectory:relativeToURL:', NSURL, [NSString, BOOL, NSURL]),
  ('fileURLWithPath:relativeToURL:', NSURL, [NSString, NSURL]),
  ('fileURLWithPath:isDirectory:', NSURL, [NSString, BOOL]),
  ('fileURLWithPath:', NSURL, [NSString]),
  ('fileURLWithFileSystemRepresentation:isDirectory:relativeToURL:', NSURL, [ctypes.POINTER(ctypes.c_char), BOOL, NSURL]),
  ('URLWithString:', 'instancetype', [NSString]),
  ('URLWithString:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('URLWithString:encodingInvalidCharacters:', 'instancetype', [NSString, BOOL]),
  ('URLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('absoluteURLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('URLByResolvingBookmarkData:options:relativeToURL:bookmarkDataIsStale:error:', 'instancetype', [NSData, NSURLBookmarkResolutionOptions, NSURL, ctypes.POINTER(BOOL), ctypes.POINTER(NSError)]),
  ('writeBookmarkData:toURL:options:error:', BOOL, [NSData, NSURL, NSURLBookmarkFileCreationOptions, ctypes.POINTER(NSError)]),
  ('bookmarkDataWithContentsOfURL:error:', NSData, [NSURL, ctypes.POINTER(NSError)]),
  ('URLByResolvingAliasFileAtURL:options:error:', 'instancetype', [NSURL, NSURLBookmarkResolutionOptions, ctypes.POINTER(NSError)]),
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
  ('preflightAndReturnError:', BOOL, [ctypes.POINTER(NSError)]),
  ('loadAndReturnError:', BOOL, [ctypes.POINTER(NSError)]),
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
enum_MTLLanguageVersion = CEnum(NSUInteger)
MTLLanguageVersion1_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_0', 65536)
MTLLanguageVersion1_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_1', 65537)
MTLLanguageVersion1_2 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_2', 65538)
MTLLanguageVersion2_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_0', 131072)
MTLLanguageVersion2_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_1', 131073)
MTLLanguageVersion2_2 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_2', 131074)
MTLLanguageVersion2_3 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_3', 131075)
MTLLanguageVersion2_4 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_4', 131076)
MTLLanguageVersion3_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion3_0', 196608)
MTLLanguageVersion3_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion3_1', 196609)

MTLLanguageVersion = enum_MTLLanguageVersion
enum_MTLLibraryOptimizationLevel = CEnum(NSInteger)
MTLLibraryOptimizationLevelDefault = enum_MTLLibraryOptimizationLevel.define('MTLLibraryOptimizationLevelDefault', 0)
MTLLibraryOptimizationLevelSize = enum_MTLLibraryOptimizationLevel.define('MTLLibraryOptimizationLevelSize', 1)

MTLLibraryOptimizationLevel = enum_MTLLibraryOptimizationLevel
enum_MTLCompileSymbolVisibility = CEnum(NSInteger)
MTLCompileSymbolVisibilityDefault = enum_MTLCompileSymbolVisibility.define('MTLCompileSymbolVisibilityDefault', 0)
MTLCompileSymbolVisibilityHidden = enum_MTLCompileSymbolVisibility.define('MTLCompileSymbolVisibilityHidden', 1)

MTLCompileSymbolVisibility = enum_MTLCompileSymbolVisibility
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
MTLTimestamp = ctypes.c_uint64
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
  ('newDefaultLibraryWithBundle:error:', MTLLibrary, [NSBundle, ctypes.POINTER(NSError)], True),
  ('newLibraryWithFile:error:', MTLLibrary, [NSString, ctypes.POINTER(NSError)], True),
  ('newLibraryWithURL:error:', MTLLibrary, [NSURL, ctypes.POINTER(NSError)], True),
  ('newLibraryWithData:error:', MTLLibrary, [objc.id_, ctypes.POINTER(NSError)], True),
  ('newLibraryWithSource:options:error:', MTLLibrary, [NSString, MTLCompileOptions, ctypes.POINTER(NSError)], True),
  ('newLibraryWithStitchedDescriptor:error:', MTLLibrary, [MTLStitchedLibraryDescriptor, ctypes.POINTER(NSError)], True),
  ('newRenderPipelineStateWithDescriptor:error:', MTLRenderPipelineState, [MTLRenderPipelineDescriptor, ctypes.POINTER(NSError)], True),
  ('newRenderPipelineStateWithDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLRenderPipelineDescriptor, MTLPipelineOption, ctypes.POINTER(MTLRenderPipelineReflection), ctypes.POINTER(NSError)], True),
  ('newComputePipelineStateWithFunction:error:', MTLComputePipelineState, [MTLFunction, ctypes.POINTER(NSError)], True),
  ('newComputePipelineStateWithFunction:options:reflection:error:', MTLComputePipelineState, [MTLFunction, MTLPipelineOption, ctypes.POINTER(MTLComputePipelineReflection), ctypes.POINTER(NSError)], True),
  ('newComputePipelineStateWithDescriptor:options:reflection:error:', MTLComputePipelineState, [MTLComputePipelineDescriptor, MTLPipelineOption, ctypes.POINTER(MTLComputePipelineReflection), ctypes.POINTER(NSError)], True),
  ('newFence', MTLFence, [], True),
  ('supportsFeatureSet:', BOOL, [MTLFeatureSet]),
  ('supportsFamily:', BOOL, [MTLGPUFamily]),
  ('supportsTextureSampleCount:', BOOL, [NSUInteger]),
  ('minimumLinearTextureAlignmentForPixelFormat:', NSUInteger, [MTLPixelFormat]),
  ('minimumTextureBufferAlignmentForPixelFormat:', NSUInteger, [MTLPixelFormat]),
  ('newRenderPipelineStateWithTileDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLTileRenderPipelineDescriptor, MTLPipelineOption, ctypes.POINTER(MTLRenderPipelineReflection), ctypes.POINTER(NSError)], True),
  ('newRenderPipelineStateWithMeshDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLMeshRenderPipelineDescriptor, MTLPipelineOption, ctypes.POINTER(MTLRenderPipelineReflection), ctypes.POINTER(NSError)], True),
  ('getDefaultSamplePositions:count:', None, [ctypes.POINTER(MTLSamplePosition), NSUInteger]),
  ('supportsRasterizationRateMapWithLayerCount:', BOOL, [NSUInteger]),
  ('newRasterizationRateMapWithDescriptor:', MTLRasterizationRateMap, [MTLRasterizationRateMapDescriptor], True),
  ('newIndirectCommandBufferWithDescriptor:maxCommandCount:options:', MTLIndirectCommandBuffer, [MTLIndirectCommandBufferDescriptor, NSUInteger, MTLResourceOptions], True),
  ('newEvent', MTLEvent, [], True),
  ('newSharedEvent', MTLSharedEvent, [], True),
  ('newSharedEventWithHandle:', MTLSharedEvent, [MTLSharedEventHandle], True),
  ('newIOHandleWithURL:error:', MTLIOFileHandle, [NSURL, ctypes.POINTER(NSError)], True),
  ('newIOCommandQueueWithDescriptor:error:', MTLIOCommandQueue, [MTLIOCommandQueueDescriptor, ctypes.POINTER(NSError)], True),
  ('newIOHandleWithURL:compressionMethod:error:', MTLIOFileHandle, [NSURL, MTLIOCompressionMethod, ctypes.POINTER(NSError)], True),
  ('newIOFileHandleWithURL:error:', MTLIOFileHandle, [NSURL, ctypes.POINTER(NSError)], True),
  ('newIOFileHandleWithURL:compressionMethod:error:', MTLIOFileHandle, [NSURL, MTLIOCompressionMethod, ctypes.POINTER(NSError)], True),
  ('sparseTileSizeWithTextureType:pixelFormat:sampleCount:', MTLSize, [MTLTextureType, MTLPixelFormat, NSUInteger]),
  ('convertSparsePixelRegions:toTileRegions:withTileSize:alignmentMode:numRegions:', None, [ctypes.POINTER(MTLRegion), ctypes.POINTER(MTLRegion), MTLSize, MTLSparseTextureRegionAlignmentMode, NSUInteger]),
  ('convertSparseTileRegions:toPixelRegions:withTileSize:numRegions:', None, [ctypes.POINTER(MTLRegion), ctypes.POINTER(MTLRegion), MTLSize, NSUInteger]),
  ('sparseTileSizeInBytesForSparsePageSize:', NSUInteger, [MTLSparsePageSize]),
  ('sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:', MTLSize, [MTLTextureType, MTLPixelFormat, NSUInteger, MTLSparsePageSize]),
  ('newCounterSampleBufferWithDescriptor:error:', MTLCounterSampleBuffer, [MTLCounterSampleBufferDescriptor, ctypes.POINTER(NSError)], True),
  ('sampleTimestamps:gpuTimestamp:', None, [ctypes.POINTER(MTLTimestamp), ctypes.POINTER(MTLTimestamp)]),
  ('newArgumentEncoderWithBufferBinding:', MTLArgumentEncoder, [MTLBufferBinding], True),
  ('supportsCounterSampling:', BOOL, [MTLCounterSamplingPoint]),
  ('supportsVertexAmplificationCount:', BOOL, [NSUInteger]),
  ('newDynamicLibrary:error:', MTLDynamicLibrary, [MTLLibrary, ctypes.POINTER(NSError)], True),
  ('newDynamicLibraryWithURL:error:', MTLDynamicLibrary, [NSURL, ctypes.POINTER(NSError)], True),
  ('newBinaryArchiveWithDescriptor:error:', MTLBinaryArchive, [MTLBinaryArchiveDescriptor, ctypes.POINTER(NSError)], True),
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
enum_MTLIndirectCommandType = CEnum(NSUInteger)
MTLIndirectCommandTypeDraw = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDraw', 1)
MTLIndirectCommandTypeDrawIndexed = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawIndexed', 2)
MTLIndirectCommandTypeDrawPatches = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawPatches', 4)
MTLIndirectCommandTypeDrawIndexedPatches = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawIndexedPatches', 8)
MTLIndirectCommandTypeConcurrentDispatch = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeConcurrentDispatch', 32)
MTLIndirectCommandTypeConcurrentDispatchThreads = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeConcurrentDispatchThreads', 64)
MTLIndirectCommandTypeDrawMeshThreadgroups = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawMeshThreadgroups', 128)
MTLIndirectCommandTypeDrawMeshThreads = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawMeshThreads', 256)

MTLIndirectCommandType = enum_MTLIndirectCommandType
class MTLIndirectCommandBufferExecutionRange(Struct): pass
MTLIndirectCommandBufferExecutionRange._fields_ = [
  ('location', uint32_t),
  ('length', uint32_t),
]
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
enum_MTLPrimitiveType = CEnum(NSUInteger)
MTLPrimitiveTypePoint = enum_MTLPrimitiveType.define('MTLPrimitiveTypePoint', 0)
MTLPrimitiveTypeLine = enum_MTLPrimitiveType.define('MTLPrimitiveTypeLine', 1)
MTLPrimitiveTypeLineStrip = enum_MTLPrimitiveType.define('MTLPrimitiveTypeLineStrip', 2)
MTLPrimitiveTypeTriangle = enum_MTLPrimitiveType.define('MTLPrimitiveTypeTriangle', 3)
MTLPrimitiveTypeTriangleStrip = enum_MTLPrimitiveType.define('MTLPrimitiveTypeTriangleStrip', 4)

MTLPrimitiveType = enum_MTLPrimitiveType
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