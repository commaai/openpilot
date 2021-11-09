/*@@@+++@@@@******************************************************************

 Microsoft Skype Engineering
 Copyright (C) 2014 Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

*@@@---@@@@******************************************************************/


#ifndef __OMX_SKYPE_VIDEOEXTENSIONS_H__
#define __OMX_SKYPE_VIDEOEXTENSIONS_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <OMX_Core.h>

#pragma pack(push, 1)


typedef enum OMX_SKYPE_VIDEO_SliceControlMode
{
    OMX_SKYPE_VIDEO_SliceControlModeNone        = 0,
    OMX_SKYPE_VIDEO_SliceControlModeMB          = 1,
    OMX_SKYPE_VIDEO_SliceControlModeByte        = 2,
    OMX_SKYPE_VIDEO_SliceControlModMBRow        = 3,
} OMX_SKYPE_VIDEO_SliceControlMode;


typedef enum OMX_SKYPE_VIDEO_HierarType
{
    OMX_SKYPE_VIDEO_HierarType_P                = 0x01,
    OMX_SKYPE_VIDEO_HierarType_B                = 0x02,
} OMX_SKYPE_VIDEO_HIERAR_HierarType;

typedef enum OMX_VIDEO_EXTENSION_AVCPROFILETYPE
{
    OMX_VIDEO_EXT_AVCProfileConstrainedBaseline = 0x01,
    OMX_VIDEO_EXT_AVCProfileConstrainedHigh     = 0x02,
} OMX_VIDEO_EXTENSION_AVCPROFILETYPE;

typedef struct OMX_SKYPE_VIDEO_ENCODERPARAMS {
    OMX_BOOL bLowLatency;
    OMX_BOOL bUseExtendedProfile;
    OMX_BOOL bSequenceHeaderWithIDR;
    OMX_VIDEO_EXTENSION_AVCPROFILETYPE eProfile;
    OMX_U32 nLTRFrames;
    OMX_SKYPE_VIDEO_HierarType eHierarType;
    OMX_U32 nMaxTemporalLayerCount;
    OMX_SKYPE_VIDEO_SliceControlMode eSliceControlMode;
    OMX_U32 nSarIndex;
    OMX_U32 nSarWidth;
    OMX_U32 nSarHeight;
} OMX_SKYPE_VIDEO_ENCODERPARAMS;

typedef struct OMX_SKYPE_VIDEO_PARAM_ENCODERSETTING {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_SKYPE_VIDEO_ENCODERPARAMS stEncParam;
} OMX_SKYPE_VIDEO_PARAM_ENCODESETTING;

typedef struct OMX_SKYPE_VIDEO_ENCODERCAP {
    OMX_BOOL bLowLatency;
    OMX_U32 nMaxFrameWidth;
    OMX_U32 nMaxFrameHeight;
    OMX_U32 nMaxInstances;
    OMX_U32 nMaxTemporaLayerCount;
    OMX_U32 nMaxRefFrames;
    OMX_U32 nMaxLTRFrames;
    OMX_VIDEO_AVCLEVELTYPE nMaxLevel;
    OMX_U32 nSliceControlModesBM;
    OMX_U32 nMaxMacroblockProcessingRate;
    OMX_U32 xMinScaleFactor;
} OMX_SKYPE_VIDEO_ENCODERCAP;

typedef struct OMX_SKYPE_VIDEO_PARAM_ENCODERCAP {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_SKYPE_VIDEO_ENCODERCAP stEncCap;
} OMX_SKYPE_VIDEO_PARAM_ENCODERCAP;

typedef struct OMX_SKYPE_VIDEO_DECODERCAP {
    OMX_BOOL bLowLatency;
    OMX_U32 nMaxFrameWidth;
    OMX_U32 nMaxFrameHeight;
    OMX_U32 nMaxInstances;
    OMX_VIDEO_AVCLEVELTYPE nMaxLevel;
    OMX_U32 nMaxMacroblockProcessingRate;
} OMX_SKYPE_VIDEO_DECODERCAP;

typedef struct OMX_SKYPE_VIDEO_PARAM_DECODERCAP {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_SKYPE_VIDEO_DECODERCAP stDecoderCap;
} OMX_SKYPE_VIDEO_PARAM_DECODERCAP;

typedef struct OMX_SKYPE_VIDEO_CONFIG_QP {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nQP;
} OMX_SKYPE_VIDEO_CONFIG_QP;

typedef struct OMX_SKYPE_VIDEO_CONFIG_BASELAYERPID{
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nPID;
} OMX_SKYPE_VIDEO_CONFIG_BASELAYERPID;

typedef struct OMX_SKYPE_VIDEO_PARAM_DRIVERVER {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U64 nDriverVersion;
} OMX_SKYPE_VIDEO_PARAM_DRIVERVER;

typedef enum OMX_SKYPE_VIDEO_DownScaleFactor
{
    OMX_SKYPE_VIDEO_DownScaleFactor_1_1         = 0,
    OMX_SKYPE_VIDEO_DownScaleFactor_Equal_AR    = 1,
    OMX_SKYPE_VIDEO_DownScaleFactor_Any         = 2,
} OMX_SKYPE_VIDEO_DownScaleFactor;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif
