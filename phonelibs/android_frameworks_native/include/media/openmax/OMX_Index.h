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
/*
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

/** @file OMX_Index.h - OpenMax IL version 1.1.2
 *  The OMX_Index header file contains the definitions for both applications
 *  and components .
 */


#ifndef OMX_Index_h
#define OMX_Index_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Each OMX header must include all required header files to allow the
 *  header to compile without errors.  The includes below are required
 *  for this header file to compile successfully
 */
#include <OMX_Types.h>

/** The OMX_INDEXTYPE enumeration is used to select a structure when either
 *  getting or setting parameters and/or configuration data.  Each entry in
 *  this enumeration maps to an OMX specified structure.  When the
 *  OMX_GetParameter, OMX_SetParameter, OMX_GetConfig or OMX_SetConfig methods
 *  are used, the second parameter will always be an entry from this enumeration
 *  and the third entry will be the structure shown in the comments for the entry.
 *  For example, if the application is initializing a cropping function, the
 *  OMX_SetConfig command would have OMX_IndexConfigCommonInputCrop as the second parameter
 *  and would send a pointer to an initialized OMX_RECTTYPE structure as the
 *  third parameter.
 *
 *  The enumeration entries named with the OMX_Config prefix are sent using
 *  the OMX_SetConfig command and the enumeration entries named with the
 *  OMX_PARAM_ prefix are sent using the OMX_SetParameter command.
 */
typedef enum OMX_INDEXTYPE {

    OMX_IndexComponentStartUnused = 0x01000000,
    OMX_IndexParamPriorityMgmt,             /**< reference: OMX_PRIORITYMGMTTYPE */
    OMX_IndexParamAudioInit,                /**< reference: OMX_PORT_PARAM_TYPE */
    OMX_IndexParamImageInit,                /**< reference: OMX_PORT_PARAM_TYPE */
    OMX_IndexParamVideoInit,                /**< reference: OMX_PORT_PARAM_TYPE */
    OMX_IndexParamOtherInit,                /**< reference: OMX_PORT_PARAM_TYPE */
    OMX_IndexParamNumAvailableStreams,      /**< reference: OMX_PARAM_U32TYPE */
    OMX_IndexParamActiveStream,             /**< reference: OMX_PARAM_U32TYPE */
    OMX_IndexParamSuspensionPolicy,         /**< reference: OMX_PARAM_SUSPENSIONPOLICYTYPE */
    OMX_IndexParamComponentSuspended,       /**< reference: OMX_PARAM_SUSPENSIONTYPE */
    OMX_IndexConfigCapturing,               /**< reference: OMX_CONFIG_BOOLEANTYPE */
    OMX_IndexConfigCaptureMode,             /**< reference: OMX_CONFIG_CAPTUREMODETYPE */
    OMX_IndexAutoPauseAfterCapture,         /**< reference: OMX_CONFIG_BOOLEANTYPE */
    OMX_IndexParamContentURI,               /**< reference: OMX_PARAM_CONTENTURITYPE */
    OMX_IndexParamCustomContentPipe,        /**< reference: OMX_PARAM_CONTENTPIPETYPE */
    OMX_IndexParamDisableResourceConcealment, /**< reference: OMX_RESOURCECONCEALMENTTYPE */
    OMX_IndexConfigMetadataItemCount,       /**< reference: OMX_CONFIG_METADATAITEMCOUNTTYPE */
    OMX_IndexConfigContainerNodeCount,      /**< reference: OMX_CONFIG_CONTAINERNODECOUNTTYPE */
    OMX_IndexConfigMetadataItem,            /**< reference: OMX_CONFIG_METADATAITEMTYPE */
    OMX_IndexConfigCounterNodeID,           /**< reference: OMX_CONFIG_CONTAINERNODEIDTYPE */
    OMX_IndexParamMetadataFilterType,       /**< reference: OMX_PARAM_METADATAFILTERTYPE */
    OMX_IndexParamMetadataKeyFilter,        /**< reference: OMX_PARAM_METADATAFILTERTYPE */
    OMX_IndexConfigPriorityMgmt,            /**< reference: OMX_PRIORITYMGMTTYPE */
    OMX_IndexParamStandardComponentRole,    /**< reference: OMX_PARAM_COMPONENTROLETYPE */

    OMX_IndexPortStartUnused = 0x02000000,
    OMX_IndexParamPortDefinition,           /**< reference: OMX_PARAM_PORTDEFINITIONTYPE */
    OMX_IndexParamCompBufferSupplier,       /**< reference: OMX_PARAM_BUFFERSUPPLIERTYPE */
    OMX_IndexReservedStartUnused = 0x03000000,

    /* Audio parameters and configurations */
    OMX_IndexAudioStartUnused = 0x04000000,
    OMX_IndexParamAudioPortFormat,          /**< reference: OMX_AUDIO_PARAM_PORTFORMATTYPE */
    OMX_IndexParamAudioPcm,                 /**< reference: OMX_AUDIO_PARAM_PCMMODETYPE */
    OMX_IndexParamAudioAac,                 /**< reference: OMX_AUDIO_PARAM_AACPROFILETYPE */
    OMX_IndexParamAudioRa,                  /**< reference: OMX_AUDIO_PARAM_RATYPE */
    OMX_IndexParamAudioMp3,                 /**< reference: OMX_AUDIO_PARAM_MP3TYPE */
    OMX_IndexParamAudioAdpcm,               /**< reference: OMX_AUDIO_PARAM_ADPCMTYPE */
    OMX_IndexParamAudioG723,                /**< reference: OMX_AUDIO_PARAM_G723TYPE */
    OMX_IndexParamAudioG729,                /**< reference: OMX_AUDIO_PARAM_G729TYPE */
    OMX_IndexParamAudioAmr,                 /**< reference: OMX_AUDIO_PARAM_AMRTYPE */
    OMX_IndexParamAudioWma,                 /**< reference: OMX_AUDIO_PARAM_WMATYPE */
    OMX_IndexParamAudioSbc,                 /**< reference: OMX_AUDIO_PARAM_SBCTYPE */
    OMX_IndexParamAudioMidi,                /**< reference: OMX_AUDIO_PARAM_MIDITYPE */
    OMX_IndexParamAudioGsm_FR,              /**< reference: OMX_AUDIO_PARAM_GSMFRTYPE */
    OMX_IndexParamAudioMidiLoadUserSound,   /**< reference: OMX_AUDIO_PARAM_MIDILOADUSERSOUNDTYPE */
    OMX_IndexParamAudioG726,                /**< reference: OMX_AUDIO_PARAM_G726TYPE */
    OMX_IndexParamAudioGsm_EFR,             /**< reference: OMX_AUDIO_PARAM_GSMEFRTYPE */
    OMX_IndexParamAudioGsm_HR,              /**< reference: OMX_AUDIO_PARAM_GSMHRTYPE */
    OMX_IndexParamAudioPdc_FR,              /**< reference: OMX_AUDIO_PARAM_PDCFRTYPE */
    OMX_IndexParamAudioPdc_EFR,             /**< reference: OMX_AUDIO_PARAM_PDCEFRTYPE */
    OMX_IndexParamAudioPdc_HR,              /**< reference: OMX_AUDIO_PARAM_PDCHRTYPE */
    OMX_IndexParamAudioTdma_FR,             /**< reference: OMX_AUDIO_PARAM_TDMAFRTYPE */
    OMX_IndexParamAudioTdma_EFR,            /**< reference: OMX_AUDIO_PARAM_TDMAEFRTYPE */
    OMX_IndexParamAudioQcelp8,              /**< reference: OMX_AUDIO_PARAM_QCELP8TYPE */
    OMX_IndexParamAudioQcelp13,             /**< reference: OMX_AUDIO_PARAM_QCELP13TYPE */
    OMX_IndexParamAudioEvrc,                /**< reference: OMX_AUDIO_PARAM_EVRCTYPE */
    OMX_IndexParamAudioSmv,                 /**< reference: OMX_AUDIO_PARAM_SMVTYPE */
    OMX_IndexParamAudioVorbis,              /**< reference: OMX_AUDIO_PARAM_VORBISTYPE */
    OMX_IndexParamAudioFlac,                /**< reference: OMX_AUDIO_PARAM_FLACTYPE */

    OMX_IndexConfigAudioMidiImmediateEvent, /**< reference: OMX_AUDIO_CONFIG_MIDIIMMEDIATEEVENTTYPE */
    OMX_IndexConfigAudioMidiControl,        /**< reference: OMX_AUDIO_CONFIG_MIDICONTROLTYPE */
    OMX_IndexConfigAudioMidiSoundBankProgram, /**< reference: OMX_AUDIO_CONFIG_MIDISOUNDBANKPROGRAMTYPE */
    OMX_IndexConfigAudioMidiStatus,         /**< reference: OMX_AUDIO_CONFIG_MIDISTATUSTYPE */
    OMX_IndexConfigAudioMidiMetaEvent,      /**< reference: OMX_AUDIO_CONFIG_MIDIMETAEVENTTYPE */
    OMX_IndexConfigAudioMidiMetaEventData,  /**< reference: OMX_AUDIO_CONFIG_MIDIMETAEVENTDATATYPE */
    OMX_IndexConfigAudioVolume,             /**< reference: OMX_AUDIO_CONFIG_VOLUMETYPE */
    OMX_IndexConfigAudioBalance,            /**< reference: OMX_AUDIO_CONFIG_BALANCETYPE */
    OMX_IndexConfigAudioChannelMute,        /**< reference: OMX_AUDIO_CONFIG_CHANNELMUTETYPE */
    OMX_IndexConfigAudioMute,               /**< reference: OMX_AUDIO_CONFIG_MUTETYPE */
    OMX_IndexConfigAudioLoudness,           /**< reference: OMX_AUDIO_CONFIG_LOUDNESSTYPE */
    OMX_IndexConfigAudioEchoCancelation,    /**< reference: OMX_AUDIO_CONFIG_ECHOCANCELATIONTYPE */
    OMX_IndexConfigAudioNoiseReduction,     /**< reference: OMX_AUDIO_CONFIG_NOISEREDUCTIONTYPE */
    OMX_IndexConfigAudioBass,               /**< reference: OMX_AUDIO_CONFIG_BASSTYPE */
    OMX_IndexConfigAudioTreble,             /**< reference: OMX_AUDIO_CONFIG_TREBLETYPE */
    OMX_IndexConfigAudioStereoWidening,     /**< reference: OMX_AUDIO_CONFIG_STEREOWIDENINGTYPE */
    OMX_IndexConfigAudioChorus,             /**< reference: OMX_AUDIO_CONFIG_CHORUSTYPE */
    OMX_IndexConfigAudioEqualizer,          /**< reference: OMX_AUDIO_CONFIG_EQUALIZERTYPE */
    OMX_IndexConfigAudioReverberation,      /**< reference: OMX_AUDIO_CONFIG_REVERBERATIONTYPE */
    OMX_IndexConfigAudioChannelVolume,      /**< reference: OMX_AUDIO_CONFIG_CHANNELVOLUMETYPE */

    /* Image specific parameters and configurations */
    OMX_IndexImageStartUnused = 0x05000000,
    OMX_IndexParamImagePortFormat,          /**< reference: OMX_IMAGE_PARAM_PORTFORMATTYPE */
    OMX_IndexParamFlashControl,             /**< reference: OMX_IMAGE_PARAM_FLASHCONTROLTYPE */
    OMX_IndexConfigFocusControl,            /**< reference: OMX_IMAGE_CONFIG_FOCUSCONTROLTYPE */
    OMX_IndexParamQFactor,                  /**< reference: OMX_IMAGE_PARAM_QFACTORTYPE */
    OMX_IndexParamQuantizationTable,        /**< reference: OMX_IMAGE_PARAM_QUANTIZATIONTABLETYPE */
    OMX_IndexParamHuffmanTable,             /**< reference: OMX_IMAGE_PARAM_HUFFMANTTABLETYPE */
    OMX_IndexConfigFlashControl,            /**< reference: OMX_IMAGE_PARAM_FLASHCONTROLTYPE */

    /* Video specific parameters and configurations */
    OMX_IndexVideoStartUnused = 0x06000000,
    OMX_IndexParamVideoPortFormat,          /**< reference: OMX_VIDEO_PARAM_PORTFORMATTYPE */
    OMX_IndexParamVideoQuantization,        /**< reference: OMX_VIDEO_PARAM_QUANTIZATIONTYPE */
    OMX_IndexParamVideoFastUpdate,          /**< reference: OMX_VIDEO_PARAM_VIDEOFASTUPDATETYPE */
    OMX_IndexParamVideoBitrate,             /**< reference: OMX_VIDEO_PARAM_BITRATETYPE */
    OMX_IndexParamVideoMotionVector,        /**< reference: OMX_VIDEO_PARAM_MOTIONVECTORTYPE */
    OMX_IndexParamVideoIntraRefresh,        /**< reference: OMX_VIDEO_PARAM_INTRAREFRESHTYPE */
    OMX_IndexParamVideoErrorCorrection,     /**< reference: OMX_VIDEO_PARAM_ERRORCORRECTIONTYPE */
    OMX_IndexParamVideoVBSMC,               /**< reference: OMX_VIDEO_PARAM_VBSMCTYPE */
    OMX_IndexParamVideoMpeg2,               /**< reference: OMX_VIDEO_PARAM_MPEG2TYPE */
    OMX_IndexParamVideoMpeg4,               /**< reference: OMX_VIDEO_PARAM_MPEG4TYPE */
    OMX_IndexParamVideoWmv,                 /**< reference: OMX_VIDEO_PARAM_WMVTYPE */
    OMX_IndexParamVideoRv,                  /**< reference: OMX_VIDEO_PARAM_RVTYPE */
    OMX_IndexParamVideoAvc,                 /**< reference: OMX_VIDEO_PARAM_AVCTYPE */
    OMX_IndexParamVideoH263,                /**< reference: OMX_VIDEO_PARAM_H263TYPE */
    OMX_IndexParamVideoProfileLevelQuerySupported, /**< reference: OMX_VIDEO_PARAM_PROFILELEVELTYPE */
    OMX_IndexParamVideoProfileLevelCurrent, /**< reference: OMX_VIDEO_PARAM_PROFILELEVELTYPE */
    OMX_IndexConfigVideoBitrate,            /**< reference: OMX_VIDEO_CONFIG_BITRATETYPE */
    OMX_IndexConfigVideoFramerate,          /**< reference: OMX_CONFIG_FRAMERATETYPE */
    OMX_IndexConfigVideoIntraVOPRefresh,    /**< reference: OMX_CONFIG_INTRAREFRESHVOPTYPE */
    OMX_IndexConfigVideoIntraMBRefresh,     /**< reference: OMX_CONFIG_MACROBLOCKERRORMAPTYPE */
    OMX_IndexConfigVideoMBErrorReporting,   /**< reference: OMX_CONFIG_MBERRORREPORTINGTYPE */
    OMX_IndexParamVideoMacroblocksPerFrame, /**< reference: OMX_PARAM_MACROBLOCKSTYPE */
    OMX_IndexConfigVideoMacroBlockErrorMap, /**< reference: OMX_CONFIG_MACROBLOCKERRORMAPTYPE */
    OMX_IndexParamVideoSliceFMO,            /**< reference: OMX_VIDEO_PARAM_AVCSLICEFMO */
    OMX_IndexConfigVideoAVCIntraPeriod,     /**< reference: OMX_VIDEO_CONFIG_AVCINTRAPERIOD */
    OMX_IndexConfigVideoNalSize,            /**< reference: OMX_VIDEO_CONFIG_NALSIZE */

    /* Image & Video common Configurations */
    OMX_IndexCommonStartUnused = 0x07000000,
    OMX_IndexParamCommonDeblocking,         /**< reference: OMX_PARAM_DEBLOCKINGTYPE */
    OMX_IndexParamCommonSensorMode,         /**< reference: OMX_PARAM_SENSORMODETYPE */
    OMX_IndexParamCommonInterleave,         /**< reference: OMX_PARAM_INTERLEAVETYPE */
    OMX_IndexConfigCommonColorFormatConversion, /**< reference: OMX_CONFIG_COLORCONVERSIONTYPE */
    OMX_IndexConfigCommonScale,             /**< reference: OMX_CONFIG_SCALEFACTORTYPE */
    OMX_IndexConfigCommonImageFilter,       /**< reference: OMX_CONFIG_IMAGEFILTERTYPE */
    OMX_IndexConfigCommonColorEnhancement,  /**< reference: OMX_CONFIG_COLORENHANCEMENTTYPE */
    OMX_IndexConfigCommonColorKey,          /**< reference: OMX_CONFIG_COLORKEYTYPE */
    OMX_IndexConfigCommonColorBlend,        /**< reference: OMX_CONFIG_COLORBLENDTYPE */
    OMX_IndexConfigCommonFrameStabilisation,/**< reference: OMX_CONFIG_FRAMESTABTYPE */
    OMX_IndexConfigCommonRotate,            /**< reference: OMX_CONFIG_ROTATIONTYPE */
    OMX_IndexConfigCommonMirror,            /**< reference: OMX_CONFIG_MIRRORTYPE */
    OMX_IndexConfigCommonOutputPosition,    /**< reference: OMX_CONFIG_POINTTYPE */
    OMX_IndexConfigCommonInputCrop,         /**< reference: OMX_CONFIG_RECTTYPE */
    OMX_IndexConfigCommonOutputCrop,        /**< reference: OMX_CONFIG_RECTTYPE */
    OMX_IndexConfigCommonDigitalZoom,       /**< reference: OMX_CONFIG_SCALEFACTORTYPE */
    OMX_IndexConfigCommonOpticalZoom,       /**< reference: OMX_CONFIG_SCALEFACTORTYPE*/
    OMX_IndexConfigCommonWhiteBalance,      /**< reference: OMX_CONFIG_WHITEBALCONTROLTYPE */
    OMX_IndexConfigCommonExposure,          /**< reference: OMX_CONFIG_EXPOSURECONTROLTYPE */
    OMX_IndexConfigCommonContrast,          /**< reference: OMX_CONFIG_CONTRASTTYPE */
    OMX_IndexConfigCommonBrightness,        /**< reference: OMX_CONFIG_BRIGHTNESSTYPE */
    OMX_IndexConfigCommonBacklight,         /**< reference: OMX_CONFIG_BACKLIGHTTYPE */
    OMX_IndexConfigCommonGamma,             /**< reference: OMX_CONFIG_GAMMATYPE */
    OMX_IndexConfigCommonSaturation,        /**< reference: OMX_CONFIG_SATURATIONTYPE */
    OMX_IndexConfigCommonLightness,         /**< reference: OMX_CONFIG_LIGHTNESSTYPE */
    OMX_IndexConfigCommonExclusionRect,     /**< reference: OMX_CONFIG_RECTTYPE */
    OMX_IndexConfigCommonDithering,         /**< reference: OMX_CONFIG_DITHERTYPE */
    OMX_IndexConfigCommonPlaneBlend,        /**< reference: OMX_CONFIG_PLANEBLENDTYPE */
    OMX_IndexConfigCommonExposureValue,     /**< reference: OMX_CONFIG_EXPOSUREVALUETYPE */
    OMX_IndexConfigCommonOutputSize,        /**< reference: OMX_FRAMESIZETYPE */
    OMX_IndexParamCommonExtraQuantData,     /**< reference: OMX_OTHER_EXTRADATATYPE */
    OMX_IndexConfigCommonFocusRegion,       /**< reference: OMX_CONFIG_FOCUSREGIONTYPE */
    OMX_IndexConfigCommonFocusStatus,       /**< reference: OMX_PARAM_FOCUSSTATUSTYPE */
    OMX_IndexConfigCommonTransitionEffect,  /**< reference: OMX_CONFIG_TRANSITIONEFFECTTYPE */

    /* Reserved Configuration range */
    OMX_IndexOtherStartUnused = 0x08000000,
    OMX_IndexParamOtherPortFormat,          /**< reference: OMX_OTHER_PARAM_PORTFORMATTYPE */
    OMX_IndexConfigOtherPower,              /**< reference: OMX_OTHER_CONFIG_POWERTYPE */
    OMX_IndexConfigOtherStats,              /**< reference: OMX_OTHER_CONFIG_STATSTYPE */


    /* Reserved Time range */
    OMX_IndexTimeStartUnused = 0x09000000,
    OMX_IndexConfigTimeScale,               /**< reference: OMX_TIME_CONFIG_SCALETYPE */
    OMX_IndexConfigTimeClockState,          /**< reference: OMX_TIME_CONFIG_CLOCKSTATETYPE */
    OMX_IndexConfigTimeActiveRefClock,      /**< reference: OMX_TIME_CONFIG_ACTIVEREFCLOCKTYPE */
    OMX_IndexConfigTimeCurrentMediaTime,    /**< reference: OMX_TIME_CONFIG_TIMESTAMPTYPE (read only) */
    OMX_IndexConfigTimeCurrentWallTime,     /**< reference: OMX_TIME_CONFIG_TIMESTAMPTYPE (read only) */
    OMX_IndexConfigTimeCurrentAudioReference, /**< reference: OMX_TIME_CONFIG_TIMESTAMPTYPE (write only) */
    OMX_IndexConfigTimeCurrentVideoReference, /**< reference: OMX_TIME_CONFIG_TIMESTAMPTYPE (write only) */
    OMX_IndexConfigTimeMediaTimeRequest,    /**< reference: OMX_TIME_CONFIG_MEDIATIMEREQUESTTYPE (write only) */
    OMX_IndexConfigTimeClientStartTime,     /**<reference:  OMX_TIME_CONFIG_TIMESTAMPTYPE (write only) */
    OMX_IndexConfigTimePosition,            /**< reference: OMX_TIME_CONFIG_TIMESTAMPTYPE */
    OMX_IndexConfigTimeSeekMode,            /**< reference: OMX_TIME_CONFIG_SEEKMODETYPE */


    OMX_IndexKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    /* Vendor specific area */
    OMX_IndexVendorStartUnused = 0x7F000000,
    /* Vendor specific structures should be in the range of 0x7F000000
       to 0x7FFFFFFE.  This range is not broken out by vendor, so
       private indexes are not guaranteed unique and therefore should
       only be sent to the appropriate component. */

    OMX_IndexMax = 0x7FFFFFFF

} OMX_INDEXTYPE;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */
