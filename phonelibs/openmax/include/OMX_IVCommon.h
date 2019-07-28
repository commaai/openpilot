/**
 * Copyright (c) 2008 The Khronos Group Inc. 
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions: 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software. 
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
 *
 */

/** 
 * @file OMX_IVCommon.h - OpenMax IL version 1.1.2
 *  The structures needed by Video and Image components to exchange
 *  parameters and configuration data with the components.
 */
#ifndef OMX_IVCommon_h
#define OMX_IVCommon_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * Each OMX header must include all required header files to allow the header
 * to compile without errors.  The includes below are required for this header
 * file to compile successfully 
 */

#include <OMX_Core.h>

/** @defgroup iv OpenMAX IL Imaging and Video Domain
 * Common structures for OpenMAX IL Imaging and Video domains
 * @{
 */


/** 
 * Enumeration defining possible uncompressed image/video formats. 
 *
 * ENUMS:
 *  Unused                 : Placeholder value when format is N/A
 *  Monochrome             : black and white
 *  8bitRGB332             : Red 7:5, Green 4:2, Blue 1:0
 *  12bitRGB444            : Red 11:8, Green 7:4, Blue 3:0
 *  16bitARGB4444          : Alpha 15:12, Red 11:8, Green 7:4, Blue 3:0
 *  16bitARGB1555          : Alpha 15, Red 14:10, Green 9:5, Blue 4:0
 *  16bitRGB565            : Red 15:11, Green 10:5, Blue 4:0
 *  16bitBGR565            : Blue 15:11, Green 10:5, Red 4:0
 *  18bitRGB666            : Red 17:12, Green 11:6, Blue 5:0
 *  18bitARGB1665          : Alpha 17, Red 16:11, Green 10:5, Blue 4:0
 *  19bitARGB1666          : Alpha 18, Red 17:12, Green 11:6, Blue 5:0
 *  24bitRGB888            : Red 24:16, Green 15:8, Blue 7:0
 *  24bitBGR888            : Blue 24:16, Green 15:8, Red 7:0
 *  24bitARGB1887          : Alpha 23, Red 22:15, Green 14:7, Blue 6:0
 *  25bitARGB1888          : Alpha 24, Red 23:16, Green 15:8, Blue 7:0
 *  32bitBGRA8888          : Blue 31:24, Green 23:16, Red 15:8, Alpha 7:0
 *  32bitARGB8888          : Alpha 31:24, Red 23:16, Green 15:8, Blue 7:0
 *  YUV411Planar           : U,Y are subsampled by a factor of 4 horizontally
 *  YUV411PackedPlanar     : packed per payload in planar slices
 *  YUV420Planar           : Three arrays Y,U,V.
 *  YUV420PackedPlanar     : packed per payload in planar slices
 *  YUV420SemiPlanar       : Two arrays, one is all Y, the other is U and V
 *  YUV422Planar           : Three arrays Y,U,V.
 *  YUV422PackedPlanar     : packed per payload in planar slices
 *  YUV422SemiPlanar       : Two arrays, one is all Y, the other is U and V
 *  YCbYCr                 : Organized as 16bit YUYV (i.e. YCbYCr)
 *  YCrYCb                 : Organized as 16bit YVYU (i.e. YCrYCb)
 *  CbYCrY                 : Organized as 16bit UYVY (i.e. CbYCrY)
 *  CrYCbY                 : Organized as 16bit VYUY (i.e. CrYCbY)
 *  YUV444Interleaved      : Each pixel contains equal parts YUV
 *  RawBayer8bit           : SMIA camera output format
 *  RawBayer10bit          : SMIA camera output format
 *  RawBayer8bitcompressed : SMIA camera output format
 */
typedef enum OMX_COLOR_FORMATTYPE {
    OMX_COLOR_FormatUnused,
    OMX_COLOR_FormatMonochrome,
    OMX_COLOR_Format8bitRGB332,
    OMX_COLOR_Format12bitRGB444,
    OMX_COLOR_Format16bitARGB4444,
    OMX_COLOR_Format16bitARGB1555,
    OMX_COLOR_Format16bitRGB565,
    OMX_COLOR_Format16bitBGR565,
    OMX_COLOR_Format18bitRGB666,
    OMX_COLOR_Format18bitARGB1665,
    OMX_COLOR_Format19bitARGB1666, 
    OMX_COLOR_Format24bitRGB888,
    OMX_COLOR_Format24bitBGR888,
    OMX_COLOR_Format24bitARGB1887,
    OMX_COLOR_Format25bitARGB1888,
    OMX_COLOR_Format32bitBGRA8888,
    OMX_COLOR_Format32bitARGB8888,
    OMX_COLOR_FormatYUV411Planar,
    OMX_COLOR_FormatYUV411PackedPlanar,
    OMX_COLOR_FormatYUV420Planar,
    OMX_COLOR_FormatYUV420PackedPlanar,
    OMX_COLOR_FormatYUV420SemiPlanar,
    OMX_COLOR_FormatYUV422Planar,
    OMX_COLOR_FormatYUV422PackedPlanar,
    OMX_COLOR_FormatYUV422SemiPlanar,
    OMX_COLOR_FormatYCbYCr,
    OMX_COLOR_FormatYCrYCb,
    OMX_COLOR_FormatCbYCrY,
    OMX_COLOR_FormatCrYCbY,
    OMX_COLOR_FormatYUV444Interleaved,
    OMX_COLOR_FormatRawBayer8bit,
    OMX_COLOR_FormatRawBayer10bit,
    OMX_COLOR_FormatRawBayer8bitcompressed,
    OMX_COLOR_FormatL2, 
    OMX_COLOR_FormatL4, 
    OMX_COLOR_FormatL8, 
    OMX_COLOR_FormatL16, 
    OMX_COLOR_FormatL24, 
    OMX_COLOR_FormatL32,
    OMX_COLOR_FormatYUV420PackedSemiPlanar,
    OMX_COLOR_FormatYUV422PackedSemiPlanar,
    OMX_COLOR_Format18BitBGR666,
    OMX_COLOR_Format24BitARGB6666,
    OMX_COLOR_Format24BitABGR6666,
    OMX_COLOR_FormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_COLOR_FormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    /**<Reserved android opaque colorformat. Tells the encoder that
     * the actual colorformat will be  relayed by the
     * Gralloc Buffers.
     * FIXME: In the process of reserving some enum values for
     * Android-specific OMX IL colorformats. Change this enum to
     * an acceptable range once that is done.
     * */
    OMX_COLOR_FormatAndroidOpaque = 0x7F000789,
    OMX_TI_COLOR_FormatYUV420PackedSemiPlanar = 0x7F000100,
    OMX_QCOM_COLOR_FormatYVU420SemiPlanar = 0x7FA30C00,
    OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka = 0x7FA30C03,
    OMX_SEC_COLOR_FormatNV12Tiled = 0x7FC00002,
    OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar32m = 0x7FA30C04,
    OMX_COLOR_FormatMax = 0x7FFFFFFF
} OMX_COLOR_FORMATTYPE;


/** 
 * Defines the matrix for conversion from RGB to YUV or vice versa.
 * iColorMatrix should be initialized with the fixed point values 
 * used in converting between formats.
 */
typedef struct OMX_CONFIG_COLORCONVERSIONTYPE {
    OMX_U32 nSize;              /**< Size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version info */ 
    OMX_U32 nPortIndex;         /**< Port that this struct applies to */
    OMX_S32 xColorMatrix[3][3]; /**< Stored in signed Q16 format */
    OMX_S32 xColorOffset[4];    /**< Stored in signed Q16 format */
}OMX_CONFIG_COLORCONVERSIONTYPE;


/** 
 * Structure defining percent to scale each frame dimension.  For example:  
 * To make the width 50% larger, use fWidth = 1.5 and to make the width
 * 1/2 the original size, use fWidth = 0.5
 */
typedef struct OMX_CONFIG_SCALEFACTORTYPE {
    OMX_U32 nSize;            /**< Size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version info */ 
    OMX_U32 nPortIndex;       /**< Port that this struct applies to */
    OMX_S32 xWidth;           /**< Fixed point value stored as Q16 */
    OMX_S32 xHeight;          /**< Fixed point value stored as Q16 */
}OMX_CONFIG_SCALEFACTORTYPE;


/** 
 * Enumeration of possible image filter types 
 */
typedef enum OMX_IMAGEFILTERTYPE {
    OMX_ImageFilterNone,
    OMX_ImageFilterNoise,
    OMX_ImageFilterEmboss,
    OMX_ImageFilterNegative,
    OMX_ImageFilterSketch,
    OMX_ImageFilterOilPaint,
    OMX_ImageFilterHatch,
    OMX_ImageFilterGpen,
    OMX_ImageFilterAntialias, 
    OMX_ImageFilterDeRing,       
    OMX_ImageFilterSolarize,
    OMX_ImageFilterKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_ImageFilterVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_ImageFilterMax = 0x7FFFFFFF
} OMX_IMAGEFILTERTYPE;


/** 
 * Image filter configuration 
 *
 * STRUCT MEMBERS:
 *  nSize        : Size of the structure in bytes       
 *  nVersion     : OMX specification version information
 *  nPortIndex   : Port that this structure applies to 
 *  eImageFilter : Image filter type enumeration      
 */
typedef struct OMX_CONFIG_IMAGEFILTERTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_IMAGEFILTERTYPE eImageFilter;
} OMX_CONFIG_IMAGEFILTERTYPE;


/** 
 * Customized U and V for color enhancement 
 *
 * STRUCT MEMBERS:
 *  nSize             : Size of the structure in bytes
 *  nVersion          : OMX specification version information 
 *  nPortIndex        : Port that this structure applies to
 *  bColorEnhancement : Enable/disable color enhancement
 *  nCustomizedU      : Practical values: 16-240, range: 0-255, value set for 
 *                      U component
 *  nCustomizedV      : Practical values: 16-240, range: 0-255, value set for 
 *                      V component
 */
typedef struct OMX_CONFIG_COLORENHANCEMENTTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion; 
    OMX_U32 nPortIndex;
    OMX_BOOL bColorEnhancement;
    OMX_U8 nCustomizedU;
    OMX_U8 nCustomizedV;
} OMX_CONFIG_COLORENHANCEMENTTYPE;


/** 
 * Define color key and color key mask 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information 
 *  nPortIndex : Port that this structure applies to
 *  nARGBColor : 32bit Alpha, Red, Green, Blue Color
 *  nARGBMask  : 32bit Mask for Alpha, Red, Green, Blue channels
 */
typedef struct OMX_CONFIG_COLORKEYTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nARGBColor;
    OMX_U32 nARGBMask;
} OMX_CONFIG_COLORKEYTYPE;


/** 
 * List of color blend types for pre/post processing 
 *
 * ENUMS:
 *  None          : No color blending present
 *  AlphaConstant : Function is (alpha_constant * src) + 
 *                  (1 - alpha_constant) * dst)
 *  AlphaPerPixel : Function is (alpha * src) + (1 - alpha) * dst)
 *  Alternate     : Function is alternating pixels from src and dst
 *  And           : Function is (src & dst)
 *  Or            : Function is (src | dst)
 *  Invert        : Function is ~src
 */
typedef enum OMX_COLORBLENDTYPE {
    OMX_ColorBlendNone,
    OMX_ColorBlendAlphaConstant,
    OMX_ColorBlendAlphaPerPixel,
    OMX_ColorBlendAlternate,
    OMX_ColorBlendAnd,
    OMX_ColorBlendOr,
    OMX_ColorBlendInvert,
    OMX_ColorBlendKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_ColorBlendVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_ColorBlendMax = 0x7FFFFFFF
} OMX_COLORBLENDTYPE;


/** 
 * Color blend configuration 
 *
 * STRUCT MEMBERS:
 *  nSize             : Size of the structure in bytes                        
 *  nVersion          : OMX specification version information                
 *  nPortIndex        : Port that this structure applies to                   
 *  nRGBAlphaConstant : Constant global alpha values when global alpha is used
 *  eColorBlend       : Color blend type enumeration                         
 */
typedef struct OMX_CONFIG_COLORBLENDTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nRGBAlphaConstant;
    OMX_COLORBLENDTYPE  eColorBlend;
} OMX_CONFIG_COLORBLENDTYPE;


/** 
 * Hold frame dimension
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes      
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to     
 *  nWidth     : Frame width in pixels                 
 *  nHeight    : Frame height in pixels                
 */
typedef struct OMX_FRAMESIZETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nWidth;
    OMX_U32 nHeight;
} OMX_FRAMESIZETYPE;


/**
 * Rotation configuration 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes             
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  nRotation  : +/- integer rotation value               
 */
typedef struct OMX_CONFIG_ROTATIONTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_S32 nRotation; 
} OMX_CONFIG_ROTATIONTYPE;


/** 
 * Possible mirroring directions for pre/post processing 
 *
 * ENUMS:
 *  None       : No mirroring                         
 *  Vertical   : Vertical mirroring, flip on X axis   
 *  Horizontal : Horizontal mirroring, flip on Y axis  
 *  Both       : Both vertical and horizontal mirroring
 */
typedef enum OMX_MIRRORTYPE {
    OMX_MirrorNone = 0,
    OMX_MirrorVertical,
    OMX_MirrorHorizontal,
    OMX_MirrorBoth, 
    OMX_MirrorKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_MirrorVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_MirrorMax = 0x7FFFFFFF   
} OMX_MIRRORTYPE;


/** 
 * Mirroring configuration 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes      
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to  
 *  eMirror    : Mirror type enumeration              
 */
typedef struct OMX_CONFIG_MIRRORTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion; 
    OMX_U32 nPortIndex;
    OMX_MIRRORTYPE  eMirror;
} OMX_CONFIG_MIRRORTYPE;


/** 
 * Position information only 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes               
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  nX         : X coordinate for the point                     
 *  nY         : Y coordinate for the point 
 */                      
typedef struct OMX_CONFIG_POINTTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_S32 nX;
    OMX_S32 nY;
} OMX_CONFIG_POINTTYPE;


/** 
 * Frame size plus position 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes                    
 *  nVersion   : OMX specification version information      
 *  nPortIndex : Port that this structure applies to    
 *  nLeft      : X Coordinate of the top left corner of the rectangle
 *  nTop       : Y Coordinate of the top left corner of the rectangle
 *  nWidth     : Width of the rectangle                              
 *  nHeight    : Height of the rectangle                             
 */
typedef struct OMX_CONFIG_RECTTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;  
    OMX_U32 nPortIndex; 
    OMX_S32 nLeft; 
    OMX_S32 nTop;
    OMX_U32 nWidth;
    OMX_U32 nHeight;
} OMX_CONFIG_RECTTYPE;


/** 
 * Deblocking state; it is required to be set up before starting the codec 
 *
 * STRUCT MEMBERS:
 *  nSize       : Size of the structure in bytes      
 *  nVersion    : OMX specification version information 
 *  nPortIndex  : Port that this structure applies to
 *  bDeblocking : Enable/disable deblocking mode    
 */
typedef struct OMX_PARAM_DEBLOCKINGTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL bDeblocking;
} OMX_PARAM_DEBLOCKINGTYPE;


/** 
 * Stabilization state 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes          
 *  nVersion   : OMX specification version information    
 *  nPortIndex : Port that this structure applies to   
 *  bStab      : Enable/disable frame stabilization state
 */
typedef struct OMX_CONFIG_FRAMESTABTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL bStab;
} OMX_CONFIG_FRAMESTABTYPE;


/** 
 * White Balance control type 
 *
 * STRUCT MEMBERS:
 *  SunLight : Referenced in JSR-234
 *  Flash    : Optimal for device's integrated flash
 */
typedef enum OMX_WHITEBALCONTROLTYPE {
    OMX_WhiteBalControlOff = 0,
    OMX_WhiteBalControlAuto,
    OMX_WhiteBalControlSunLight,
    OMX_WhiteBalControlCloudy,
    OMX_WhiteBalControlShade,
    OMX_WhiteBalControlTungsten,
    OMX_WhiteBalControlFluorescent,
    OMX_WhiteBalControlIncandescent,
    OMX_WhiteBalControlFlash,
    OMX_WhiteBalControlHorizon,
    OMX_WhiteBalControlKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_WhiteBalControlVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_WhiteBalControlMax = 0x7FFFFFFF
} OMX_WHITEBALCONTROLTYPE;


/** 
 * White Balance control configuration 
 *
 * STRUCT MEMBERS:
 *  nSize            : Size of the structure in bytes       
 *  nVersion         : OMX specification version information
 *  nPortIndex       : Port that this structure applies to                 
 *  eWhiteBalControl : White balance enumeration            
 */
typedef struct OMX_CONFIG_WHITEBALCONTROLTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_WHITEBALCONTROLTYPE eWhiteBalControl;
} OMX_CONFIG_WHITEBALCONTROLTYPE;


/** 
 * Exposure control type 
 */
typedef enum OMX_EXPOSURECONTROLTYPE {
    OMX_ExposureControlOff = 0,
    OMX_ExposureControlAuto,
    OMX_ExposureControlNight,
    OMX_ExposureControlBackLight,
    OMX_ExposureControlSpotLight,
    OMX_ExposureControlSports,
    OMX_ExposureControlSnow,
    OMX_ExposureControlBeach,
    OMX_ExposureControlLargeAperture,
    OMX_ExposureControlSmallApperture,
    OMX_ExposureControlKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_ExposureControlVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_ExposureControlMax = 0x7FFFFFFF
} OMX_EXPOSURECONTROLTYPE;


/** 
 * White Balance control configuration 
 *
 * STRUCT MEMBERS:
 *  nSize            : Size of the structure in bytes      
 *  nVersion         : OMX specification version information
 *  nPortIndex       : Port that this structure applies to                
 *  eExposureControl : Exposure control enumeration         
 */
typedef struct OMX_CONFIG_EXPOSURECONTROLTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_EXPOSURECONTROLTYPE eExposureControl;
} OMX_CONFIG_EXPOSURECONTROLTYPE;


/** 
 * Defines sensor supported mode. 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes           
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to 
 *  nFrameRate : Single shot mode is indicated by a 0     
 *  bOneShot   : Enable for single shot, disable for streaming
 *  sFrameSize : Framesize                                          
 */
typedef struct OMX_PARAM_SENSORMODETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nFrameRate;
    OMX_BOOL bOneShot;
    OMX_FRAMESIZETYPE sFrameSize;
} OMX_PARAM_SENSORMODETYPE;


/** 
 * Defines contrast level 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes                              
 *  nVersion   : OMX specification version information                
 *  nPortIndex : Port that this structure applies to                 
 *  nContrast  : Values allowed for contrast -100 to 100, zero means no change
 */
typedef struct OMX_CONFIG_CONTRASTTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_S32 nContrast;
} OMX_CONFIG_CONTRASTTYPE;


/** 
 * Defines brightness level 
 *
 * STRUCT MEMBERS:
 *  nSize       : Size of the structure in bytes          
 *  nVersion    : OMX specification version information 
 *  nPortIndex  : Port that this structure applies to 
 *  nBrightness : 0-100%        
 */
typedef struct OMX_CONFIG_BRIGHTNESSTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nBrightness;
} OMX_CONFIG_BRIGHTNESSTYPE;


/** 
 * Defines backlight level configuration for a video sink, e.g. LCD panel 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information 
 *  nPortIndex : Port that this structure applies to
 *  nBacklight : Values allowed for backlight 0-100%
 *  nTimeout   : Number of milliseconds before backlight automatically turns 
 *               off.  A value of 0x0 disables backight timeout 
 */
typedef struct OMX_CONFIG_BACKLIGHTTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nBacklight;
    OMX_U32 nTimeout;
} OMX_CONFIG_BACKLIGHTTYPE;


/** 
 * Defines setting for Gamma 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information 
 *  nPortIndex : Port that this structure applies to
 *  nGamma     : Values allowed for gamma -100 to 100, zero means no change
 */
typedef struct OMX_CONFIG_GAMMATYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_S32 nGamma;
} OMX_CONFIG_GAMMATYPE;


/** 
 * Define for setting saturation 
 * 
 * STRUCT MEMBERS:
 *  nSize       : Size of the structure in bytes
 *  nVersion    : OMX specification version information
 *  nPortIndex  : Port that this structure applies to
 *  nSaturation : Values allowed for saturation -100 to 100, zero means 
 *                no change
 */
typedef struct OMX_CONFIG_SATURATIONTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_S32 nSaturation;
} OMX_CONFIG_SATURATIONTYPE;


/** 
 * Define for setting Lightness 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  nLightness : Values allowed for lightness -100 to 100, zero means no 
 *               change
 */
typedef struct OMX_CONFIG_LIGHTNESSTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_S32 nLightness;
} OMX_CONFIG_LIGHTNESSTYPE;


/** 
 * Plane blend configuration 
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes 
 *  nVersion   : OMX specification version information
 *  nPortIndex : Index of input port associated with the plane.
 *  nDepth     : Depth of the plane in relation to the screen. Higher 
 *               numbered depths are "behind" lower number depths.  
 *               This number defaults to the Port Index number.
 *  nAlpha     : Transparency blending component for the entire plane.  
 *               See blending modes for more detail.
 */
typedef struct OMX_CONFIG_PLANEBLENDTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nDepth;
    OMX_U32 nAlpha;
} OMX_CONFIG_PLANEBLENDTYPE;


/** 
 * Define interlace type
 *
 * STRUCT MEMBERS:
 *  nSize                 : Size of the structure in bytes 
 *  nVersion              : OMX specification version information 
 *  nPortIndex            : Port that this structure applies to
 *  bEnable               : Enable control variable for this functionality 
 *                          (see below)
 *  nInterleavePortIndex  : Index of input or output port associated with  
 *                          the interleaved plane. 
 *  pPlanarPortIndexes[4] : Index of input or output planar ports.
 */
typedef struct OMX_PARAM_INTERLEAVETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL bEnable;
    OMX_U32 nInterleavePortIndex;
} OMX_PARAM_INTERLEAVETYPE;


/** 
 * Defines the picture effect used for an input picture 
 */
typedef enum OMX_TRANSITIONEFFECTTYPE {
    OMX_EffectNone,
    OMX_EffectFadeFromBlack,
    OMX_EffectFadeToBlack,
    OMX_EffectUnspecifiedThroughConstantColor,
    OMX_EffectDissolve,
    OMX_EffectWipe,
    OMX_EffectUnspecifiedMixOfTwoScenes,
    OMX_EffectKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_EffectVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_EffectMax = 0x7FFFFFFF
} OMX_TRANSITIONEFFECTTYPE;


/** 
 * Structure used to configure current transition effect 
 *
 * STRUCT MEMBERS:
 * nSize      : Size of the structure in bytes
 * nVersion   : OMX specification version information 
 * nPortIndex : Port that this structure applies to
 * eEffect    : Effect to enable
 */
typedef struct OMX_CONFIG_TRANSITIONEFFECTTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_TRANSITIONEFFECTTYPE eEffect;
} OMX_CONFIG_TRANSITIONEFFECTTYPE;


/** 
 * Defines possible data unit types for encoded video data. The data unit 
 * types are used both for encoded video input for playback as well as
 * encoded video output from recording. 
 */
typedef enum OMX_DATAUNITTYPE {
    OMX_DataUnitCodedPicture,
    OMX_DataUnitVideoSegment,
    OMX_DataUnitSeveralSegments,
    OMX_DataUnitArbitraryStreamSection,
    OMX_DataUnitKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_DataUnitVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_DataUnitMax = 0x7FFFFFFF
} OMX_DATAUNITTYPE;


/** 
 * Defines possible encapsulation types for coded video data unit. The 
 * encapsulation information is used both for encoded video input for 
 * playback as well as encoded video output from recording. 
 */
typedef enum OMX_DATAUNITENCAPSULATIONTYPE {
    OMX_DataEncapsulationElementaryStream,
    OMX_DataEncapsulationGenericPayload,
    OMX_DataEncapsulationRtpPayload,
    OMX_DataEncapsulationKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_DataEncapsulationVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_DataEncapsulationMax = 0x7FFFFFFF
} OMX_DATAUNITENCAPSULATIONTYPE;


/** 
 * Structure used to configure the type of being decoded/encoded 
 */
typedef struct OMX_PARAM_DATAUNITTYPE {
    OMX_U32 nSize;            /**< Size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */ 
    OMX_U32 nPortIndex;       /**< Port that this structure applies to */
    OMX_DATAUNITTYPE eUnitType;
    OMX_DATAUNITENCAPSULATIONTYPE eEncapsulationType;
} OMX_PARAM_DATAUNITTYPE;


/**
 * Defines dither types 
 */
typedef enum OMX_DITHERTYPE {
    OMX_DitherNone,
    OMX_DitherOrdered,
    OMX_DitherErrorDiffusion,
    OMX_DitherOther,
    OMX_DitherKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_DitherVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_DitherMax = 0x7FFFFFFF
} OMX_DITHERTYPE;


/** 
 * Structure used to configure current type of dithering 
 */
typedef struct OMX_CONFIG_DITHERTYPE {
    OMX_U32 nSize;            /**< Size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */ 
    OMX_U32 nPortIndex;       /**< Port that this structure applies to */
    OMX_DITHERTYPE eDither;   /**< Type of dithering to use */
} OMX_CONFIG_DITHERTYPE;

typedef struct OMX_CONFIG_CAPTUREMODETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;     /**< Port that this structure applies to */
    OMX_BOOL bContinuous;   /**< If true then ignore frame rate and emit capture 
                             *   data as fast as possible (otherwise obey port's frame rate). */
    OMX_BOOL bFrameLimited; /**< If true then terminate capture after the port emits the 
                             *   specified number of frames (otherwise the port does not 
                             *   terminate the capture until instructed to do so by the client). 
                             *   Even if set, the client may manually terminate the capture prior 
                             *   to reaching the limit. */
    OMX_U32 nFrameLimit;      /**< Limit on number of frames emitted during a capture (only
                               *   valid if bFrameLimited is set). */
} OMX_CONFIG_CAPTUREMODETYPE;

typedef enum OMX_METERINGTYPE {
 
    OMX_MeteringModeAverage,     /**< Center-weighted average metering. */
    OMX_MeteringModeSpot,  	      /**< Spot (partial) metering. */
    OMX_MeteringModeMatrix,      /**< Matrix or evaluative metering. */
 
    OMX_MeteringKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_MeteringVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_EVModeMax = 0x7fffffff
} OMX_METERINGTYPE;
 
typedef struct OMX_CONFIG_EXPOSUREVALUETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_METERINGTYPE eMetering;
    OMX_S32 xEVCompensation;      /**< Fixed point value stored as Q16 */
    OMX_U32 nApertureFNumber;     /**< e.g. nApertureFNumber = 2 implies "f/2" - Q16 format */
    OMX_BOOL bAutoAperture;		/**< Whether aperture number is defined automatically */
    OMX_U32 nShutterSpeedMsec;    /**< Shutterspeed in milliseconds */ 
    OMX_BOOL bAutoShutterSpeed;	/**< Whether shutter speed is defined automatically */ 
    OMX_U32 nSensitivity;         /**< e.g. nSensitivity = 100 implies "ISO 100" */
    OMX_BOOL bAutoSensitivity;	/**< Whether sensitivity is defined automatically */
} OMX_CONFIG_EXPOSUREVALUETYPE;

/** 
 * Focus region configuration 
 *
 * STRUCT MEMBERS:
 *  nSize           : Size of the structure in bytes
 *  nVersion        : OMX specification version information
 *  nPortIndex      : Port that this structure applies to
 *  bCenter         : Use center region as focus region of interest
 *  bLeft           : Use left region as focus region of interest
 *  bRight          : Use right region as focus region of interest
 *  bTop            : Use top region as focus region of interest
 *  bBottom         : Use bottom region as focus region of interest
 *  bTopLeft        : Use top left region as focus region of interest
 *  bTopRight       : Use top right region as focus region of interest
 *  bBottomLeft     : Use bottom left region as focus region of interest
 *  bBottomRight    : Use bottom right region as focus region of interest
 */
typedef struct OMX_CONFIG_FOCUSREGIONTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL bCenter;
    OMX_BOOL bLeft;
    OMX_BOOL bRight;
    OMX_BOOL bTop;
    OMX_BOOL bBottom;
    OMX_BOOL bTopLeft;
    OMX_BOOL bTopRight;
    OMX_BOOL bBottomLeft;
    OMX_BOOL bBottomRight;
} OMX_CONFIG_FOCUSREGIONTYPE;

/** 
 * Focus Status type 
 */
typedef enum OMX_FOCUSSTATUSTYPE {
    OMX_FocusStatusOff = 0,
    OMX_FocusStatusRequest,
    OMX_FocusStatusReached,
    OMX_FocusStatusUnableToReach,
    OMX_FocusStatusLost,
    OMX_FocusStatusKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */ 
    OMX_FocusStatusVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_FocusStatusMax = 0x7FFFFFFF
} OMX_FOCUSSTATUSTYPE;

/** 
 * Focus status configuration 
 *
 * STRUCT MEMBERS:
 *  nSize               : Size of the structure in bytes
 *  nVersion            : OMX specification version information
 *  nPortIndex          : Port that this structure applies to
 *  eFocusStatus        : Specifies the focus status
 *  bCenterStatus       : Use center region as focus region of interest
 *  bLeftStatus         : Use left region as focus region of interest
 *  bRightStatus        : Use right region as focus region of interest
 *  bTopStatus          : Use top region as focus region of interest
 *  bBottomStatus       : Use bottom region as focus region of interest
 *  bTopLeftStatus      : Use top left region as focus region of interest
 *  bTopRightStatus     : Use top right region as focus region of interest
 *  bBottomLeftStatus   : Use bottom left region as focus region of interest
 *  bBottomRightStatus  : Use bottom right region as focus region of interest
 */
typedef struct OMX_PARAM_FOCUSSTATUSTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_FOCUSSTATUSTYPE eFocusStatus;
    OMX_BOOL bCenterStatus;
    OMX_BOOL bLeftStatus;
    OMX_BOOL bRightStatus;
    OMX_BOOL bTopStatus;
    OMX_BOOL bBottomStatus;
    OMX_BOOL bTopLeftStatus;
    OMX_BOOL bTopRightStatus;
    OMX_BOOL bBottomLeftStatus;
    OMX_BOOL bBottomRightStatus;
} OMX_PARAM_FOCUSSTATUSTYPE;

/** @} */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */
