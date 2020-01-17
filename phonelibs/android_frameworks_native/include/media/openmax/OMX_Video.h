/* ------------------------------------------------------------------
 * Copyright (C) 1998-2009 PacketVideo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * -------------------------------------------------------------------
 */
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
 *  @file OMX_Video.h - OpenMax IL version 1.1.2
 *  The structures is needed by Video components to exchange parameters
 *  and configuration data with OMX components.
 */
#ifndef OMX_Video_h
#define OMX_Video_h

/** @defgroup video OpenMAX IL Video Domain
 * @ingroup iv
 * Structures for OpenMAX IL Video domain
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/**
 * Each OMX header must include all required header files to allow the
 * header to compile without errors.  The includes below are required
 * for this header file to compile successfully
 */

#include <OMX_IVCommon.h>


/**
 * Enumeration used to define the possible video compression codings.
 * NOTE:  This essentially refers to file extensions. If the coding is
 *        being used to specify the ENCODE type, then additional work
 *        must be done to configure the exact flavor of the compression
 *        to be used.  For decode cases where the user application can
 *        not differentiate between MPEG-4 and H.264 bit streams, it is
 *        up to the codec to handle this.
 */
typedef enum OMX_VIDEO_CODINGTYPE {
    OMX_VIDEO_CodingUnused,     /**< Value when coding is N/A */
    OMX_VIDEO_CodingAutoDetect, /**< Autodetection of coding type */
    OMX_VIDEO_CodingMPEG2,      /**< AKA: H.262 */
    OMX_VIDEO_CodingH263,       /**< H.263 */
    OMX_VIDEO_CodingMPEG4,      /**< MPEG-4 */
    OMX_VIDEO_CodingWMV,        /**< all versions of Windows Media Video */
    OMX_VIDEO_CodingRV,         /**< all versions of Real Video */
    OMX_VIDEO_CodingAVC,        /**< H.264/AVC */
    OMX_VIDEO_CodingMJPEG,      /**< Motion JPEG */
    OMX_VIDEO_CodingVP8,        /**< Google VP8, formerly known as On2 VP8 */
    OMX_VIDEO_CodingVP9,        /**< Google VP9 */
    OMX_VIDEO_CodingHEVC,       /**< ITU H.265/HEVC */
    OMX_VIDEO_CodingKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_CodingVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_CodingMax = 0x7FFFFFFF
} OMX_VIDEO_CODINGTYPE;


/**
 * Data structure used to define a video path.  The number of Video paths for
 * input and output will vary by type of the Video component.
 *
 *    Input (aka Source) : zero Inputs, one Output,
 *    Splitter           : one Input, 2 or more Outputs,
 *    Processing Element : one Input, one output,
 *    Mixer              : 2 or more inputs, one output,
 *    Output (aka Sink)  : one Input, zero outputs.
 *
 * The PortDefinition structure is used to define all of the parameters
 * necessary for the compliant component to setup an input or an output video
 * path.  If additional vendor specific data is required, it should be
 * transmitted to the component using the CustomCommand function.  Compliant
 * components will prepopulate this structure with optimal values during the
 * GetDefaultInitParams command.
 *
 * STRUCT MEMBERS:
 *  cMIMEType             : MIME type of data for the port
 *  pNativeRender         : Platform specific reference for a display if a
 *                          sync, otherwise this field is 0
 *  nFrameWidth           : Width of frame to be used on channel if
 *                          uncompressed format is used.  Use 0 for unknown,
 *                          don't care or variable
 *  nFrameHeight          : Height of frame to be used on channel if
 *                          uncompressed format is used. Use 0 for unknown,
 *                          don't care or variable
 *  nStride               : Number of bytes per span of an image
 *                          (i.e. indicates the number of bytes to get
 *                          from span N to span N+1, where negative stride
 *                          indicates the image is bottom up
 *  nSliceHeight          : Height used when encoding in slices
 *  nBitrate              : Bit rate of frame to be used on channel if
 *                          compressed format is used. Use 0 for unknown,
 *                          don't care or variable
 *  xFramerate            : Frame rate to be used on channel if uncompressed
 *                          format is used. Use 0 for unknown, don't care or
 *                          variable.  Units are Q16 frames per second.
 *  bFlagErrorConcealment : Turns on error concealment if it is supported by
 *                          the OMX component
 *  eCompressionFormat    : Compression format used in this instance of the
 *                          component. When OMX_VIDEO_CodingUnused is
 *                          specified, eColorFormat is used
 *  eColorFormat : Decompressed format used by this component
 *  pNativeWindow : Platform specific reference for a window object if a
 *                          display sink , otherwise this field is 0x0.
 */
typedef struct OMX_VIDEO_PORTDEFINITIONTYPE {
    OMX_STRING cMIMEType;
    OMX_NATIVE_DEVICETYPE pNativeRender;
    OMX_U32 nFrameWidth;
    OMX_U32 nFrameHeight;
    OMX_S32 nStride;
    OMX_U32 nSliceHeight;
    OMX_U32 nBitrate;
    OMX_U32 xFramerate;
    OMX_BOOL bFlagErrorConcealment;
    OMX_VIDEO_CODINGTYPE eCompressionFormat;
    OMX_COLOR_FORMATTYPE eColorFormat;
    OMX_NATIVE_WINDOWTYPE pNativeWindow;
} OMX_VIDEO_PORTDEFINITIONTYPE;

/**
 * Port format parameter.  This structure is used to enumerate the various
 * data input/output format supported by the port.
 *
 * STRUCT MEMBERS:
 *  nSize              : Size of the structure in bytes
 *  nVersion           : OMX specification version information
 *  nPortIndex         : Indicates which port to set
 *  nIndex             : Indicates the enumeration index for the format from
 *                       0x0 to N-1
 *  eCompressionFormat : Compression format used in this instance of the
 *                       component. When OMX_VIDEO_CodingUnused is specified,
 *                       eColorFormat is used
 *  eColorFormat       : Decompressed format used by this component
 *  xFrameRate         : Indicates the video frame rate in Q16 format
 */
typedef struct OMX_VIDEO_PARAM_PORTFORMATTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nIndex;
    OMX_VIDEO_CODINGTYPE eCompressionFormat;
    OMX_COLOR_FORMATTYPE eColorFormat;
    OMX_U32 xFramerate;
} OMX_VIDEO_PARAM_PORTFORMATTYPE;


/**
 * This is a structure for configuring video compression quantization
 * parameter values.  Codecs may support different QP values for different
 * frame types.
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version info
 *  nPortIndex : Port that this structure applies to
 *  nQpI       : QP value to use for index frames
 *  nQpP       : QP value to use for P frames
 *  nQpB       : QP values to use for bidirectional frames
 */
typedef struct OMX_VIDEO_PARAM_QUANTIZATIONTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nQpI;
    OMX_U32 nQpP;
    OMX_U32 nQpB;
} OMX_VIDEO_PARAM_QUANTIZATIONTYPE;


/**
 * Structure for configuration of video fast update parameters.
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version info
 *  nPortIndex : Port that this structure applies to
 *  bEnableVFU : Enable/Disable video fast update
 *  nFirstGOB  : Specifies the number of the first macroblock row
 *  nFirstMB   : specifies the first MB relative to the specified first GOB
 *  nNumMBs    : Specifies the number of MBs to be refreshed from nFirstGOB
 *               and nFirstMB
 */
typedef struct OMX_VIDEO_PARAM_VIDEOFASTUPDATETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL bEnableVFU;
    OMX_U32 nFirstGOB;
    OMX_U32 nFirstMB;
    OMX_U32 nNumMBs;
} OMX_VIDEO_PARAM_VIDEOFASTUPDATETYPE;


/**
 * Enumeration of possible bitrate control types
 */
typedef enum OMX_VIDEO_CONTROLRATETYPE {
    OMX_Video_ControlRateDisable,
    OMX_Video_ControlRateVariable,
    OMX_Video_ControlRateConstant,
    OMX_Video_ControlRateVariableSkipFrames,
    OMX_Video_ControlRateConstantSkipFrames,
    OMX_Video_ControlRateKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_Video_ControlRateVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_Video_ControlRateMax = 0x7FFFFFFF
} OMX_VIDEO_CONTROLRATETYPE;


/**
 * Structure for configuring bitrate mode of a codec.
 *
 * STRUCT MEMBERS:
 *  nSize          : Size of the struct in bytes
 *  nVersion       : OMX spec version info
 *  nPortIndex     : Port that this struct applies to
 *  eControlRate   : Control rate type enum
 *  nTargetBitrate : Target bitrate to encode with
 */
typedef struct OMX_VIDEO_PARAM_BITRATETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_VIDEO_CONTROLRATETYPE eControlRate;
    OMX_U32 nTargetBitrate;
} OMX_VIDEO_PARAM_BITRATETYPE;


/**
 * Enumeration of possible motion vector (MV) types
 */
typedef enum OMX_VIDEO_MOTIONVECTORTYPE {
    OMX_Video_MotionVectorPixel,
    OMX_Video_MotionVectorHalfPel,
    OMX_Video_MotionVectorQuarterPel,
    OMX_Video_MotionVectorEighthPel,
    OMX_Video_MotionVectorKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_Video_MotionVectorVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_Video_MotionVectorMax = 0x7FFFFFFF
} OMX_VIDEO_MOTIONVECTORTYPE;


/**
 * Structure for configuring the number of motion vectors used as well
 * as their accuracy.
 *
 * STRUCT MEMBERS:
 *  nSize            : Size of the struct in bytes
 *  nVersion         : OMX spec version info
 *  nPortIndex       : port that this structure applies to
 *  eAccuracy        : Enumerated MV accuracy
 *  bUnrestrictedMVs : Allow unrestricted MVs
 *  bFourMV          : Allow use of 4 MVs
 *  sXSearchRange    : Search range in horizontal direction for MVs
 *  sYSearchRange    : Search range in vertical direction for MVs
 */
typedef struct OMX_VIDEO_PARAM_MOTIONVECTORTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_VIDEO_MOTIONVECTORTYPE eAccuracy;
    OMX_BOOL bUnrestrictedMVs;
    OMX_BOOL bFourMV;
    OMX_S32 sXSearchRange;
    OMX_S32 sYSearchRange;
} OMX_VIDEO_PARAM_MOTIONVECTORTYPE;


/**
 * Enumeration of possible methods to use for Intra Refresh
 */
typedef enum OMX_VIDEO_INTRAREFRESHTYPE {
    OMX_VIDEO_IntraRefreshCyclic,
    OMX_VIDEO_IntraRefreshAdaptive,
    OMX_VIDEO_IntraRefreshBoth,
    OMX_VIDEO_IntraRefreshKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_IntraRefreshVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_IntraRefreshMax = 0x7FFFFFFF
} OMX_VIDEO_INTRAREFRESHTYPE;


/**
 * Structure for configuring intra refresh mode
 *
 * STRUCT MEMBERS:
 *  nSize        : Size of the structure in bytes
 *  nVersion     : OMX specification version information
 *  nPortIndex   : Port that this structure applies to
 *  eRefreshMode : Cyclic, Adaptive, or Both
 *  nAirMBs      : Number of intra macroblocks to refresh in a frame when
 *                 AIR is enabled
 *  nAirRef      : Number of times a motion marked macroblock has to be
 *                 intra coded
 *  nCirMBs      : Number of consecutive macroblocks to be coded as "intra"
 *                 when CIR is enabled
 */
typedef struct OMX_VIDEO_PARAM_INTRAREFRESHTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_VIDEO_INTRAREFRESHTYPE eRefreshMode;
    OMX_U32 nAirMBs;
    OMX_U32 nAirRef;
    OMX_U32 nCirMBs;
} OMX_VIDEO_PARAM_INTRAREFRESHTYPE;


/**
 * Structure for enabling various error correction methods for video
 * compression.
 *
 * STRUCT MEMBERS:
 *  nSize                   : Size of the structure in bytes
 *  nVersion                : OMX specification version information
 *  nPortIndex              : Port that this structure applies to
 *  bEnableHEC              : Enable/disable header extension codes (HEC)
 *  bEnableResync           : Enable/disable resynchronization markers
 *  nResynchMarkerSpacing   : Resynch markers interval (in bits) to be
 *                            applied in the stream
 *  bEnableDataPartitioning : Enable/disable data partitioning
 *  bEnableRVLC             : Enable/disable reversible variable length
 *                            coding
 */
typedef struct OMX_VIDEO_PARAM_ERRORCORRECTIONTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL bEnableHEC;
    OMX_BOOL bEnableResync;
    OMX_U32  nResynchMarkerSpacing;
    OMX_BOOL bEnableDataPartitioning;
    OMX_BOOL bEnableRVLC;
} OMX_VIDEO_PARAM_ERRORCORRECTIONTYPE;


/**
 * Configuration of variable block-size motion compensation (VBSMC)
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  b16x16     : Enable inter block search 16x16
 *  b16x8      : Enable inter block search 16x8
 *  b8x16      : Enable inter block search 8x16
 *  b8x8       : Enable inter block search 8x8
 *  b8x4       : Enable inter block search 8x4
 *  b4x8       : Enable inter block search 4x8
 *  b4x4       : Enable inter block search 4x4
 */
typedef struct OMX_VIDEO_PARAM_VBSMCTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL b16x16;
    OMX_BOOL b16x8;
    OMX_BOOL b8x16;
    OMX_BOOL b8x8;
    OMX_BOOL b8x4;
    OMX_BOOL b4x8;
    OMX_BOOL b4x4;
} OMX_VIDEO_PARAM_VBSMCTYPE;


/**
 * H.263 profile types, each profile indicates support for various
 * performance bounds and different annexes.
 *
 * ENUMS:
 *  Baseline           : Baseline Profile: H.263 (V1), no optional modes
 *  H320 Coding        : H.320 Coding Efficiency Backward Compatibility
 *                       Profile: H.263+ (V2), includes annexes I, J, L.4
 *                       and T
 *  BackwardCompatible : Backward Compatibility Profile: H.263 (V1),
 *                       includes annex F
 *  ISWV2              : Interactive Streaming Wireless Profile: H.263+
 *                       (V2), includes annexes I, J, K and T
 *  ISWV3              : Interactive Streaming Wireless Profile: H.263++
 *                       (V3), includes profile 3 and annexes V and W.6.3.8
 *  HighCompression    : Conversational High Compression Profile: H.263++
 *                       (V3), includes profiles 1 & 2 and annexes D and U
 *  Internet           : Conversational Internet Profile: H.263++ (V3),
 *                       includes profile 5 and annex K
 *  Interlace          : Conversational Interlace Profile: H.263++ (V3),
 *                       includes profile 5 and annex W.6.3.11
 *  HighLatency        : High Latency Profile: H.263++ (V3), includes
 *                       profile 6 and annexes O.1 and P.5
 */
typedef enum OMX_VIDEO_H263PROFILETYPE {
    OMX_VIDEO_H263ProfileBaseline            = 0x01,
    OMX_VIDEO_H263ProfileH320Coding          = 0x02,
    OMX_VIDEO_H263ProfileBackwardCompatible  = 0x04,
    OMX_VIDEO_H263ProfileISWV2               = 0x08,
    OMX_VIDEO_H263ProfileISWV3               = 0x10,
    OMX_VIDEO_H263ProfileHighCompression     = 0x20,
    OMX_VIDEO_H263ProfileInternet            = 0x40,
    OMX_VIDEO_H263ProfileInterlace           = 0x80,
    OMX_VIDEO_H263ProfileHighLatency         = 0x100,
    OMX_VIDEO_H263ProfileKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_H263ProfileVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_H263ProfileMax                 = 0x7FFFFFFF
} OMX_VIDEO_H263PROFILETYPE;


/**
 * H.263 level types, each level indicates support for various frame sizes,
 * bit rates, decoder frame rates.
 */
typedef enum OMX_VIDEO_H263LEVELTYPE {
    OMX_VIDEO_H263Level10  = 0x01,
    OMX_VIDEO_H263Level20  = 0x02,
    OMX_VIDEO_H263Level30  = 0x04,
    OMX_VIDEO_H263Level40  = 0x08,
    OMX_VIDEO_H263Level45  = 0x10,
    OMX_VIDEO_H263Level50  = 0x20,
    OMX_VIDEO_H263Level60  = 0x40,
    OMX_VIDEO_H263Level70  = 0x80,
    OMX_VIDEO_H263LevelKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_H263LevelVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_H263LevelMax = 0x7FFFFFFF
} OMX_VIDEO_H263LEVELTYPE;


/**
 * Specifies the picture type. These values should be OR'd to signal all
 * pictures types which are allowed.
 *
 * ENUMS:
 *  Generic Picture Types:          I, P and B
 *  H.263 Specific Picture Types:   SI and SP
 *  H.264 Specific Picture Types:   EI and EP
 *  MPEG-4 Specific Picture Types:  S
 */
typedef enum OMX_VIDEO_PICTURETYPE {
    OMX_VIDEO_PictureTypeI   = 0x01,
    OMX_VIDEO_PictureTypeP   = 0x02,
    OMX_VIDEO_PictureTypeB   = 0x04,
    OMX_VIDEO_PictureTypeSI  = 0x08,
    OMX_VIDEO_PictureTypeSP  = 0x10,
    OMX_VIDEO_PictureTypeEI  = 0x11,
    OMX_VIDEO_PictureTypeEP  = 0x12,
    OMX_VIDEO_PictureTypeS   = 0x14,
    OMX_VIDEO_PictureTypeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_PictureTypeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_PictureTypeMax = 0x7FFFFFFF
} OMX_VIDEO_PICTURETYPE;


/**
 * H.263 Params
 *
 * STRUCT MEMBERS:
 *  nSize                    : Size of the structure in bytes
 *  nVersion                 : OMX specification version information
 *  nPortIndex               : Port that this structure applies to
 *  nPFrames                 : Number of P frames between each I frame
 *  nBFrames                 : Number of B frames between each I frame
 *  eProfile                 : H.263 profile(s) to use
 *  eLevel                   : H.263 level(s) to use
 *  bPLUSPTYPEAllowed        : Indicating that it is allowed to use PLUSPTYPE
 *                             (specified in the 1998 version of H.263) to
 *                             indicate custom picture sizes or clock
 *                             frequencies
 *  nAllowedPictureTypes     : Specifies the picture types allowed in the
 *                             bitstream
 *  bForceRoundingTypeToZero : value of the RTYPE bit (bit 6 of MPPTYPE) is
 *                             not constrained. It is recommended to change
 *                             the value of the RTYPE bit for each reference
 *                             picture in error-free communication
 *  nPictureHeaderRepetition : Specifies the frequency of picture header
 *                             repetition
 *  nGOBHeaderInterval       : Specifies the interval of non-empty GOB
 *                             headers in units of GOBs
 */
typedef struct OMX_VIDEO_PARAM_H263TYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nPFrames;
    OMX_U32 nBFrames;
    OMX_VIDEO_H263PROFILETYPE eProfile;
    OMX_VIDEO_H263LEVELTYPE eLevel;
    OMX_BOOL bPLUSPTYPEAllowed;
    OMX_U32 nAllowedPictureTypes;
    OMX_BOOL bForceRoundingTypeToZero;
    OMX_U32 nPictureHeaderRepetition;
    OMX_U32 nGOBHeaderInterval;
} OMX_VIDEO_PARAM_H263TYPE;


/**
 * MPEG-2 profile types, each profile indicates support for various
 * performance bounds and different annexes.
 */
typedef enum OMX_VIDEO_MPEG2PROFILETYPE {
    OMX_VIDEO_MPEG2ProfileSimple = 0,  /**< Simple Profile */
    OMX_VIDEO_MPEG2ProfileMain,        /**< Main Profile */
    OMX_VIDEO_MPEG2Profile422,         /**< 4:2:2 Profile */
    OMX_VIDEO_MPEG2ProfileSNR,         /**< SNR Profile */
    OMX_VIDEO_MPEG2ProfileSpatial,     /**< Spatial Profile */
    OMX_VIDEO_MPEG2ProfileHigh,        /**< High Profile */
    OMX_VIDEO_MPEG2ProfileKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_MPEG2ProfileVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_MPEG2ProfileMax = 0x7FFFFFFF
} OMX_VIDEO_MPEG2PROFILETYPE;


/**
 * MPEG-2 level types, each level indicates support for various frame
 * sizes, bit rates, decoder frame rates.  No need
 */
typedef enum OMX_VIDEO_MPEG2LEVELTYPE {
    OMX_VIDEO_MPEG2LevelLL = 0,  /**< Low Level */
    OMX_VIDEO_MPEG2LevelML,      /**< Main Level */
    OMX_VIDEO_MPEG2LevelH14,     /**< High 1440 */
    OMX_VIDEO_MPEG2LevelHL,      /**< High Level */
    OMX_VIDEO_MPEG2LevelKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_MPEG2LevelVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_MPEG2LevelMax = 0x7FFFFFFF
} OMX_VIDEO_MPEG2LEVELTYPE;


/**
 * MPEG-2 params
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  nPFrames   : Number of P frames between each I frame
 *  nBFrames   : Number of B frames between each I frame
 *  eProfile   : MPEG-2 profile(s) to use
 *  eLevel     : MPEG-2 levels(s) to use
 */
typedef struct OMX_VIDEO_PARAM_MPEG2TYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nPFrames;
    OMX_U32 nBFrames;
    OMX_VIDEO_MPEG2PROFILETYPE eProfile;
    OMX_VIDEO_MPEG2LEVELTYPE eLevel;
} OMX_VIDEO_PARAM_MPEG2TYPE;


/**
 * MPEG-4 profile types, each profile indicates support for various
 * performance bounds and different annexes.
 *
 * ENUMS:
 *  - Simple Profile, Levels 1-3
 *  - Simple Scalable Profile, Levels 1-2
 *  - Core Profile, Levels 1-2
 *  - Main Profile, Levels 2-4
 *  - N-bit Profile, Level 2
 *  - Scalable Texture Profile, Level 1
 *  - Simple Face Animation Profile, Levels 1-2
 *  - Simple Face and Body Animation (FBA) Profile, Levels 1-2
 *  - Basic Animated Texture Profile, Levels 1-2
 *  - Hybrid Profile, Levels 1-2
 *  - Advanced Real Time Simple Profiles, Levels 1-4
 *  - Core Scalable Profile, Levels 1-3
 *  - Advanced Coding Efficiency Profile, Levels 1-4
 *  - Advanced Core Profile, Levels 1-2
 *  - Advanced Scalable Texture, Levels 2-3
 */
typedef enum OMX_VIDEO_MPEG4PROFILETYPE {
    OMX_VIDEO_MPEG4ProfileSimple           = 0x01,
    OMX_VIDEO_MPEG4ProfileSimpleScalable   = 0x02,
    OMX_VIDEO_MPEG4ProfileCore             = 0x04,
    OMX_VIDEO_MPEG4ProfileMain             = 0x08,
    OMX_VIDEO_MPEG4ProfileNbit             = 0x10,
    OMX_VIDEO_MPEG4ProfileScalableTexture  = 0x20,
    OMX_VIDEO_MPEG4ProfileSimpleFace       = 0x40,
    OMX_VIDEO_MPEG4ProfileSimpleFBA        = 0x80,
    OMX_VIDEO_MPEG4ProfileBasicAnimated    = 0x100,
    OMX_VIDEO_MPEG4ProfileHybrid           = 0x200,
    OMX_VIDEO_MPEG4ProfileAdvancedRealTime = 0x400,
    OMX_VIDEO_MPEG4ProfileCoreScalable     = 0x800,
    OMX_VIDEO_MPEG4ProfileAdvancedCoding   = 0x1000,
    OMX_VIDEO_MPEG4ProfileAdvancedCore     = 0x2000,
    OMX_VIDEO_MPEG4ProfileAdvancedScalable = 0x4000,
    OMX_VIDEO_MPEG4ProfileAdvancedSimple   = 0x8000,
    OMX_VIDEO_MPEG4ProfileKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_MPEG4ProfileVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_MPEG4ProfileMax              = 0x7FFFFFFF
} OMX_VIDEO_MPEG4PROFILETYPE;


/**
 * MPEG-4 level types, each level indicates support for various frame
 * sizes, bit rates, decoder frame rates.  No need
 */
typedef enum OMX_VIDEO_MPEG4LEVELTYPE {
    OMX_VIDEO_MPEG4Level0  = 0x01,   /**< Level 0 */
    OMX_VIDEO_MPEG4Level0b = 0x02,   /**< Level 0b */
    OMX_VIDEO_MPEG4Level1  = 0x04,   /**< Level 1 */
    OMX_VIDEO_MPEG4Level2  = 0x08,   /**< Level 2 */
    OMX_VIDEO_MPEG4Level3  = 0x10,   /**< Level 3 */
    OMX_VIDEO_MPEG4Level4  = 0x20,   /**< Level 4 */
    OMX_VIDEO_MPEG4Level4a = 0x40,   /**< Level 4a */
    OMX_VIDEO_MPEG4Level5  = 0x80,   /**< Level 5 */
    OMX_VIDEO_MPEG4LevelKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_MPEG4LevelVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_MPEG4LevelMax = 0x7FFFFFFF
} OMX_VIDEO_MPEG4LEVELTYPE;


/**
 * MPEG-4 configuration.  This structure handles configuration options
 * which are specific to MPEG4 algorithms
 *
 * STRUCT MEMBERS:
 *  nSize                : Size of the structure in bytes
 *  nVersion             : OMX specification version information
 *  nPortIndex           : Port that this structure applies to
 *  nSliceHeaderSpacing  : Number of macroblocks between slice header (H263+
 *                         Annex K). Put zero if not used
 *  bSVH                 : Enable Short Video Header mode
 *  bGov                 : Flag to enable GOV
 *  nPFrames             : Number of P frames between each I frame (also called
 *                         GOV period)
 *  nBFrames             : Number of B frames between each I frame
 *  nIDCVLCThreshold     : Value of intra DC VLC threshold
 *  bACPred              : Flag to use ac prediction
 *  nMaxPacketSize       : Maximum size of packet in bytes.
 *  nTimeIncRes          : Used to pass VOP time increment resolution for MPEG4.
 *                         Interpreted as described in MPEG4 standard.
 *  eProfile             : MPEG-4 profile(s) to use.
 *  eLevel               : MPEG-4 level(s) to use.
 *  nAllowedPictureTypes : Specifies the picture types allowed in the bitstream
 *  nHeaderExtension     : Specifies the number of consecutive video packet
 *                         headers within a VOP
 *  bReversibleVLC       : Specifies whether reversible variable length coding
 *                         is in use
 */
typedef struct OMX_VIDEO_PARAM_MPEG4TYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nSliceHeaderSpacing;
    OMX_BOOL bSVH;
    OMX_BOOL bGov;
    OMX_U32 nPFrames;
    OMX_U32 nBFrames;
    OMX_U32 nIDCVLCThreshold;
    OMX_BOOL bACPred;
    OMX_U32 nMaxPacketSize;
    OMX_U32 nTimeIncRes;
    OMX_VIDEO_MPEG4PROFILETYPE eProfile;
    OMX_VIDEO_MPEG4LEVELTYPE eLevel;
    OMX_U32 nAllowedPictureTypes;
    OMX_U32 nHeaderExtension;
    OMX_BOOL bReversibleVLC;
} OMX_VIDEO_PARAM_MPEG4TYPE;


/**
 * WMV Versions
 */
typedef enum OMX_VIDEO_WMVFORMATTYPE {
    OMX_VIDEO_WMVFormatUnused = 0x01,   /**< Format unused or unknown */
    OMX_VIDEO_WMVFormat7      = 0x02,   /**< Windows Media Video format 7 */
    OMX_VIDEO_WMVFormat8      = 0x04,   /**< Windows Media Video format 8 */
    OMX_VIDEO_WMVFormat9      = 0x08,   /**< Windows Media Video format 9 */
    OMX_VIDEO_WMFFormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_WMFFormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_WMVFormatMax    = 0x7FFFFFFF
} OMX_VIDEO_WMVFORMATTYPE;


/**
 * WMV Params
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  eFormat    : Version of WMV stream / data
 */
typedef struct OMX_VIDEO_PARAM_WMVTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_VIDEO_WMVFORMATTYPE eFormat;
} OMX_VIDEO_PARAM_WMVTYPE;


/**
 * Real Video Version
 */
typedef enum OMX_VIDEO_RVFORMATTYPE {
    OMX_VIDEO_RVFormatUnused = 0, /**< Format unused or unknown */
    OMX_VIDEO_RVFormat8,          /**< Real Video format 8 */
    OMX_VIDEO_RVFormat9,          /**< Real Video format 9 */
    OMX_VIDEO_RVFormatG2,         /**< Real Video Format G2 */
    OMX_VIDEO_RVFormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_RVFormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_RVFormatMax = 0x7FFFFFFF
} OMX_VIDEO_RVFORMATTYPE;


/**
 * Real Video Params
 *
 * STUCT MEMBERS:
 *  nSize              : Size of the structure in bytes
 *  nVersion           : OMX specification version information
 *  nPortIndex         : Port that this structure applies to
 *  eFormat            : Version of RV stream / data
 *  nBitsPerPixel      : Bits per pixel coded in the frame
 *  nPaddedWidth       : Padded width in pixel of a video frame
 *  nPaddedHeight      : Padded Height in pixels of a video frame
 *  nFrameRate         : Rate of video in frames per second
 *  nBitstreamFlags    : Flags which internal information about the bitstream
 *  nBitstreamVersion  : Bitstream version
 *  nMaxEncodeFrameSize: Max encoded frame size
 *  bEnablePostFilter  : Turn on/off post filter
 *  bEnableTemporalInterpolation : Turn on/off temporal interpolation
 *  bEnableLatencyMode : When enabled, the decoder does not display a decoded
 *                       frame until it has detected that no enhancement layer
 *                       frames or dependent B frames will be coming. This
 *                       detection usually occurs when a subsequent non-B
 *                       frame is encountered
 */
typedef struct OMX_VIDEO_PARAM_RVTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_VIDEO_RVFORMATTYPE eFormat;
    OMX_U16 nBitsPerPixel;
    OMX_U16 nPaddedWidth;
    OMX_U16 nPaddedHeight;
    OMX_U32 nFrameRate;
    OMX_U32 nBitstreamFlags;
    OMX_U32 nBitstreamVersion;
    OMX_U32 nMaxEncodeFrameSize;
    OMX_BOOL bEnablePostFilter;
    OMX_BOOL bEnableTemporalInterpolation;
    OMX_BOOL bEnableLatencyMode;
} OMX_VIDEO_PARAM_RVTYPE;


/**
 * AVC profile types, each profile indicates support for various
 * performance bounds and different annexes.
 */
typedef enum OMX_VIDEO_AVCPROFILETYPE {
    OMX_VIDEO_AVCProfileBaseline = 0x01,   /**< Baseline profile */
    OMX_VIDEO_AVCProfileMain     = 0x02,   /**< Main profile */
    OMX_VIDEO_AVCProfileExtended = 0x04,   /**< Extended profile */
    OMX_VIDEO_AVCProfileHigh     = 0x08,   /**< High profile */
    OMX_VIDEO_AVCProfileHigh10   = 0x10,   /**< High 10 profile */
    OMX_VIDEO_AVCProfileHigh422  = 0x20,   /**< High 4:2:2 profile */
    OMX_VIDEO_AVCProfileHigh444  = 0x40,   /**< High 4:4:4 profile */
    OMX_VIDEO_AVCProfileKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_AVCProfileVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_AVCProfileMax      = 0x7FFFFFFF
} OMX_VIDEO_AVCPROFILETYPE;


/**
 * AVC level types, each level indicates support for various frame sizes,
 * bit rates, decoder frame rates.  No need
 */
typedef enum OMX_VIDEO_AVCLEVELTYPE {
    OMX_VIDEO_AVCLevel1   = 0x01,     /**< Level 1 */
    OMX_VIDEO_AVCLevel1b  = 0x02,     /**< Level 1b */
    OMX_VIDEO_AVCLevel11  = 0x04,     /**< Level 1.1 */
    OMX_VIDEO_AVCLevel12  = 0x08,     /**< Level 1.2 */
    OMX_VIDEO_AVCLevel13  = 0x10,     /**< Level 1.3 */
    OMX_VIDEO_AVCLevel2   = 0x20,     /**< Level 2 */
    OMX_VIDEO_AVCLevel21  = 0x40,     /**< Level 2.1 */
    OMX_VIDEO_AVCLevel22  = 0x80,     /**< Level 2.2 */
    OMX_VIDEO_AVCLevel3   = 0x100,    /**< Level 3 */
    OMX_VIDEO_AVCLevel31  = 0x200,    /**< Level 3.1 */
    OMX_VIDEO_AVCLevel32  = 0x400,    /**< Level 3.2 */
    OMX_VIDEO_AVCLevel4   = 0x800,    /**< Level 4 */
    OMX_VIDEO_AVCLevel41  = 0x1000,   /**< Level 4.1 */
    OMX_VIDEO_AVCLevel42  = 0x2000,   /**< Level 4.2 */
    OMX_VIDEO_AVCLevel5   = 0x4000,   /**< Level 5 */
    OMX_VIDEO_AVCLevel51  = 0x8000,   /**< Level 5.1 */
    OMX_VIDEO_AVCLevel52  = 0x10000,  /**< Level 5.2 */
    OMX_VIDEO_AVCLevelKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_AVCLevelVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_AVCLevelMax = 0x7FFFFFFF
} OMX_VIDEO_AVCLEVELTYPE;


/**
 * AVC loop filter modes
 *
 * OMX_VIDEO_AVCLoopFilterEnable               : Enable
 * OMX_VIDEO_AVCLoopFilterDisable              : Disable
 * OMX_VIDEO_AVCLoopFilterDisableSliceBoundary : Disabled on slice boundaries
 */
typedef enum OMX_VIDEO_AVCLOOPFILTERTYPE {
    OMX_VIDEO_AVCLoopFilterEnable = 0,
    OMX_VIDEO_AVCLoopFilterDisable,
    OMX_VIDEO_AVCLoopFilterDisableSliceBoundary,
    OMX_VIDEO_AVCLoopFilterKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_AVCLoopFilterVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_AVCLoopFilterMax = 0x7FFFFFFF
} OMX_VIDEO_AVCLOOPFILTERTYPE;


/**
 * AVC params
 *
 * STRUCT MEMBERS:
 *  nSize                     : Size of the structure in bytes
 *  nVersion                  : OMX specification version information
 *  nPortIndex                : Port that this structure applies to
 *  nSliceHeaderSpacing       : Number of macroblocks between slice header, put
 *                              zero if not used
 *  nPFrames                  : Number of P frames between each I frame
 *  nBFrames                  : Number of B frames between each I frame
 *  bUseHadamard              : Enable/disable Hadamard transform
 *  nRefFrames                : Max number of reference frames to use for inter
 *                              motion search (1-16)
 *  nRefIdxTrailing           : Pic param set ref frame index (index into ref
 *                              frame buffer of trailing frames list), B frame
 *                              support
 *  nRefIdxForward            : Pic param set ref frame index (index into ref
 *                              frame buffer of forward frames list), B frame
 *                              support
 *  bEnableUEP                : Enable/disable unequal error protection. This
 *                              is only valid of data partitioning is enabled.
 *  bEnableFMO                : Enable/disable flexible macroblock ordering
 *  bEnableASO                : Enable/disable arbitrary slice ordering
 *  bEnableRS                 : Enable/disable sending of redundant slices
 *  eProfile                  : AVC profile(s) to use
 *  eLevel                    : AVC level(s) to use
 *  nAllowedPictureTypes      : Specifies the picture types allowed in the
 *                              bitstream
 *  bFrameMBsOnly             : specifies that every coded picture of the
 *                              coded video sequence is a coded frame
 *                              containing only frame macroblocks
 *  bMBAFF                    : Enable/disable switching between frame and
 *                              field macroblocks within a picture
 *  bEntropyCodingCABAC       : Entropy decoding method to be applied for the
 *                              syntax elements for which two descriptors appear
 *                              in the syntax tables
 *  bWeightedPPrediction      : Enable/disable weighted prediction shall not
 *                              be applied to P and SP slices
 *  nWeightedBipredicitonMode : Default weighted prediction is applied to B
 *                              slices
 *  bconstIpred               : Enable/disable intra prediction
 *  bDirect8x8Inference       : Specifies the method used in the derivation
 *                              process for luma motion vectors for B_Skip,
 *                              B_Direct_16x16 and B_Direct_8x8 as specified
 *                              in subclause 8.4.1.2 of the AVC spec
 *  bDirectSpatialTemporal    : Flag indicating spatial or temporal direct
 *                              mode used in B slice coding (related to
 *                              bDirect8x8Inference) . Spatial direct mode is
 *                              more common and should be the default.
 *  nCabacInitIdx             : Index used to init CABAC contexts
 *  eLoopFilterMode           : Enable/disable loop filter
 */
typedef struct OMX_VIDEO_PARAM_AVCTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nSliceHeaderSpacing;
    OMX_U32 nPFrames;
    OMX_U32 nBFrames;
    OMX_BOOL bUseHadamard;
    OMX_U32 nRefFrames;
    OMX_U32 nRefIdx10ActiveMinus1;
    OMX_U32 nRefIdx11ActiveMinus1;
    OMX_BOOL bEnableUEP;
    OMX_BOOL bEnableFMO;
    OMX_BOOL bEnableASO;
    OMX_BOOL bEnableRS;
    OMX_VIDEO_AVCPROFILETYPE eProfile;
    OMX_VIDEO_AVCLEVELTYPE eLevel;
    OMX_U32 nAllowedPictureTypes;
    OMX_BOOL bFrameMBsOnly;
    OMX_BOOL bMBAFF;
    OMX_BOOL bEntropyCodingCABAC;
    OMX_BOOL bWeightedPPrediction;
    OMX_U32 nWeightedBipredicitonMode;
    OMX_BOOL bconstIpred ;
    OMX_BOOL bDirect8x8Inference;
    OMX_BOOL bDirectSpatialTemporal;
    OMX_U32 nCabacInitIdc;
    OMX_VIDEO_AVCLOOPFILTERTYPE eLoopFilterMode;
} OMX_VIDEO_PARAM_AVCTYPE;

typedef struct OMX_VIDEO_PARAM_PROFILELEVELTYPE {
   OMX_U32 nSize;
   OMX_VERSIONTYPE nVersion;
   OMX_U32 nPortIndex;
   OMX_U32 eProfile;      /**< type is OMX_VIDEO_AVCPROFILETYPE, OMX_VIDEO_H263PROFILETYPE,
                                 or OMX_VIDEO_MPEG4PROFILETYPE depending on context */
   OMX_U32 eLevel;        /**< type is OMX_VIDEO_AVCLEVELTYPE, OMX_VIDEO_H263LEVELTYPE,
                                 or OMX_VIDEO_MPEG4PROFILETYPE depending on context */
   OMX_U32 nProfileIndex; /**< Used to query for individual profile support information,
                               This parameter is valid only for
                               OMX_IndexParamVideoProfileLevelQuerySupported index,
                               For all other indices this parameter is to be ignored. */
} OMX_VIDEO_PARAM_PROFILELEVELTYPE;

/**
 * Structure for dynamically configuring bitrate mode of a codec.
 *
 * STRUCT MEMBERS:
 *  nSize          : Size of the struct in bytes
 *  nVersion       : OMX spec version info
 *  nPortIndex     : Port that this struct applies to
 *  nEncodeBitrate : Target average bitrate to be generated in bps
 */
typedef struct OMX_VIDEO_CONFIG_BITRATETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nEncodeBitrate;
} OMX_VIDEO_CONFIG_BITRATETYPE;

/**
 * Defines Encoder Frame Rate setting
 *
 * STRUCT MEMBERS:
 *  nSize            : Size of the structure in bytes
 *  nVersion         : OMX specification version information
 *  nPortIndex       : Port that this structure applies to
 *  xEncodeFramerate : Encoding framerate represented in Q16 format
 */
typedef struct OMX_CONFIG_FRAMERATETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 xEncodeFramerate; /* Q16 format */
} OMX_CONFIG_FRAMERATETYPE;

typedef struct OMX_CONFIG_INTRAREFRESHVOPTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL IntraRefreshVOP;
} OMX_CONFIG_INTRAREFRESHVOPTYPE;

typedef struct OMX_CONFIG_MACROBLOCKERRORMAPTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nErrMapSize;           /* Size of the Error Map in bytes */
    OMX_U8  ErrMap[1];             /* Error map hint */
} OMX_CONFIG_MACROBLOCKERRORMAPTYPE;

typedef struct OMX_CONFIG_MBERRORREPORTINGTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_BOOL bEnabled;
} OMX_CONFIG_MBERRORREPORTINGTYPE;

typedef struct OMX_PARAM_MACROBLOCKSTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nMacroblocks;
} OMX_PARAM_MACROBLOCKSTYPE;

/**
 * AVC Slice Mode modes
 *
 * OMX_VIDEO_SLICEMODE_AVCDefault   : Normal frame encoding, one slice per frame
 * OMX_VIDEO_SLICEMODE_AVCMBSlice   : NAL mode, number of MBs per frame
 * OMX_VIDEO_SLICEMODE_AVCByteSlice : NAL mode, number of bytes per frame
 */
typedef enum OMX_VIDEO_AVCSLICEMODETYPE {
    OMX_VIDEO_SLICEMODE_AVCDefault = 0,
    OMX_VIDEO_SLICEMODE_AVCMBSlice,
    OMX_VIDEO_SLICEMODE_AVCByteSlice,
    OMX_VIDEO_SLICEMODE_AVCKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_VIDEO_SLICEMODE_AVCVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_SLICEMODE_AVCLevelMax = 0x7FFFFFFF
} OMX_VIDEO_AVCSLICEMODETYPE;

/**
 * AVC FMO Slice Mode Params
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  nNumSliceGroups : Specifies the number of slice groups
 *  nSliceGroupMapType : Specifies the type of slice groups
 *  eSliceMode : Specifies the type of slice
 */
typedef struct OMX_VIDEO_PARAM_AVCSLICEFMO {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U8 nNumSliceGroups;
    OMX_U8 nSliceGroupMapType;
    OMX_VIDEO_AVCSLICEMODETYPE eSliceMode;
} OMX_VIDEO_PARAM_AVCSLICEFMO;

/**
 * AVC IDR Period Configs
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  nIDRPeriod : Specifies periodicity of IDR frames
 *  nPFrames : Specifies internal of coding Intra frames
 */
typedef struct OMX_VIDEO_CONFIG_AVCINTRAPERIOD {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nIDRPeriod;
    OMX_U32 nPFrames;
} OMX_VIDEO_CONFIG_AVCINTRAPERIOD;

/**
 * AVC NAL Size Configs
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  nNaluBytes : Specifies the NAL unit size
 */
typedef struct OMX_VIDEO_CONFIG_NALSIZE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nNaluBytes;
} OMX_VIDEO_CONFIG_NALSIZE;

/** @} */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */

