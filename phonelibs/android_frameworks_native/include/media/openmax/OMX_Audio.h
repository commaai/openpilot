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

/** @file OMX_Audio.h - OpenMax IL version 1.1.2
 *  The structures needed by Audio components to exchange
 *  parameters and configuration data with the componenmilts.
 */

#ifndef OMX_Audio_h
#define OMX_Audio_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Each OMX header must include all required header files to allow the
 *  header to compile without errors.  The includes below are required
 *  for this header file to compile successfully
 */

#include <OMX_Core.h>

/** @defgroup midi MIDI
 * @ingroup audio
 */

/** @defgroup effects Audio effects
 * @ingroup audio
 */

/** @defgroup audio OpenMAX IL Audio Domain
 * Structures for OpenMAX IL Audio domain
 * @{
 */

/** Enumeration used to define the possible audio codings.
 *  If "OMX_AUDIO_CodingUnused" is selected, the coding selection must
 *  be done in a vendor specific way.  Since this is for an audio
 *  processing element this enum is relevant.  However, for another
 *  type of component other enums would be in this area.
 */
typedef enum OMX_AUDIO_CODINGTYPE {
    OMX_AUDIO_CodingUnused = 0,  /**< Placeholder value when coding is N/A  */
    OMX_AUDIO_CodingAutoDetect,  /**< auto detection of audio format */
    OMX_AUDIO_CodingPCM,         /**< Any variant of PCM coding */
    OMX_AUDIO_CodingADPCM,       /**< Any variant of ADPCM encoded data */
    OMX_AUDIO_CodingAMR,         /**< Any variant of AMR encoded data */
    OMX_AUDIO_CodingGSMFR,       /**< Any variant of GSM fullrate (i.e. GSM610) */
    OMX_AUDIO_CodingGSMEFR,      /**< Any variant of GSM Enhanced Fullrate encoded data*/
    OMX_AUDIO_CodingGSMHR,       /**< Any variant of GSM Halfrate encoded data */
    OMX_AUDIO_CodingPDCFR,       /**< Any variant of PDC Fullrate encoded data */
    OMX_AUDIO_CodingPDCEFR,      /**< Any variant of PDC Enhanced Fullrate encoded data */
    OMX_AUDIO_CodingPDCHR,       /**< Any variant of PDC Halfrate encoded data */
    OMX_AUDIO_CodingTDMAFR,      /**< Any variant of TDMA Fullrate encoded data (TIA/EIA-136-420) */
    OMX_AUDIO_CodingTDMAEFR,     /**< Any variant of TDMA Enhanced Fullrate encoded data (TIA/EIA-136-410) */
    OMX_AUDIO_CodingQCELP8,      /**< Any variant of QCELP 8kbps encoded data */
    OMX_AUDIO_CodingQCELP13,     /**< Any variant of QCELP 13kbps encoded data */
    OMX_AUDIO_CodingEVRC,        /**< Any variant of EVRC encoded data */
    OMX_AUDIO_CodingSMV,         /**< Any variant of SMV encoded data */
    OMX_AUDIO_CodingG711,        /**< Any variant of G.711 encoded data */
    OMX_AUDIO_CodingG723,        /**< Any variant of G.723 dot 1 encoded data */
    OMX_AUDIO_CodingG726,        /**< Any variant of G.726 encoded data */
    OMX_AUDIO_CodingG729,        /**< Any variant of G.729 encoded data */
    OMX_AUDIO_CodingAAC,         /**< Any variant of AAC encoded data */
    OMX_AUDIO_CodingMP3,         /**< Any variant of MP3 encoded data */
    OMX_AUDIO_CodingSBC,         /**< Any variant of SBC encoded data */
    OMX_AUDIO_CodingVORBIS,      /**< Any variant of VORBIS encoded data */
    OMX_AUDIO_CodingWMA,         /**< Any variant of WMA encoded data */
    OMX_AUDIO_CodingRA,          /**< Any variant of RA encoded data */
    OMX_AUDIO_CodingMIDI,        /**< Any variant of MIDI encoded data */
    OMX_AUDIO_CodingFLAC,        /**< Any variant of FLAC encoded data */
    OMX_AUDIO_CodingKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_CodingVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_CodingMax = 0x7FFFFFFF
} OMX_AUDIO_CODINGTYPE;


/** The PortDefinition structure is used to define all of the parameters
 *  necessary for the compliant component to setup an input or an output audio
 *  path.  If additional information is needed to define the parameters of the
 *  port (such as frequency), additional structures must be sent such as the
 *  OMX_AUDIO_PARAM_PCMMODETYPE structure to supply the extra parameters for the port.
 */
typedef struct OMX_AUDIO_PORTDEFINITIONTYPE {
    OMX_STRING cMIMEType;            /**< MIME type of data for the port */
    OMX_NATIVE_DEVICETYPE pNativeRender; /** < platform specific reference
                                               for an output device,
                                               otherwise this field is 0 */
    OMX_BOOL bFlagErrorConcealment;  /**< Turns on error concealment if it is
                                          supported by the OMX component */
    OMX_AUDIO_CODINGTYPE eEncoding;  /**< Type of data expected for this
                                          port (e.g. PCM, AMR, MP3, etc) */
} OMX_AUDIO_PORTDEFINITIONTYPE;


/**  Port format parameter.  This structure is used to enumerate
  *  the various data input/output format supported by the port.
  */
typedef struct OMX_AUDIO_PARAM_PORTFORMATTYPE {
    OMX_U32 nSize;                  /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;       /**< OMX specification version information */
    OMX_U32 nPortIndex;             /**< Indicates which port to set */
    OMX_U32 nIndex;                 /**< Indicates the enumeration index for the format from 0x0 to N-1 */
    OMX_AUDIO_CODINGTYPE eEncoding; /**< Type of data expected for this port (e.g. PCM, AMR, MP3, etc) */
} OMX_AUDIO_PARAM_PORTFORMATTYPE;


/** PCM mode type  */
typedef enum OMX_AUDIO_PCMMODETYPE {
    OMX_AUDIO_PCMModeLinear = 0,  /**< Linear PCM encoded data */
    OMX_AUDIO_PCMModeALaw,        /**< A law PCM encoded data (G.711) */
    OMX_AUDIO_PCMModeMULaw,       /**< Mu law PCM encoded data (G.711)  */
    OMX_AUDIO_PCMModeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_PCMModeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_PCMModeMax = 0x7FFFFFFF
} OMX_AUDIO_PCMMODETYPE;


typedef enum OMX_AUDIO_CHANNELTYPE {
    OMX_AUDIO_ChannelNone = 0x0,    /**< Unused or empty */
    OMX_AUDIO_ChannelLF   = 0x1,    /**< Left front */
    OMX_AUDIO_ChannelRF   = 0x2,    /**< Right front */
    OMX_AUDIO_ChannelCF   = 0x3,    /**< Center front */
    OMX_AUDIO_ChannelLS   = 0x4,    /**< Left surround */
    OMX_AUDIO_ChannelRS   = 0x5,    /**< Right surround */
    OMX_AUDIO_ChannelLFE  = 0x6,    /**< Low frequency effects */
    OMX_AUDIO_ChannelCS   = 0x7,    /**< Back surround */
    OMX_AUDIO_ChannelLR   = 0x8,    /**< Left rear. */
    OMX_AUDIO_ChannelRR   = 0x9,    /**< Right rear. */
    OMX_AUDIO_ChannelKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_ChannelVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_ChannelMax  = 0x7FFFFFFF
} OMX_AUDIO_CHANNELTYPE;

#define OMX_AUDIO_MAXCHANNELS 16  /**< maximum number distinct audio channels that a buffer may contain */
#define OMX_MIN_PCMPAYLOAD_MSEC 5 /**< Minimum audio buffer payload size for uncompressed (PCM) audio */

/** PCM format description */
typedef struct OMX_AUDIO_PARAM_PCMMODETYPE {
    OMX_U32 nSize;                    /**< Size of this structure, in Bytes */
    OMX_VERSIONTYPE nVersion;         /**< OMX specification version information */
    OMX_U32 nPortIndex;               /**< port that this structure applies to */
    OMX_U32 nChannels;                /**< Number of channels (e.g. 2 for stereo) */
    OMX_NUMERICALDATATYPE eNumData;   /**< indicates PCM data as signed or unsigned */
    OMX_ENDIANTYPE eEndian;           /**< indicates PCM data as little or big endian */
    OMX_BOOL bInterleaved;            /**< True for normal interleaved data; false for
                                           non-interleaved data (e.g. block data) */
    OMX_U32 nBitPerSample;            /**< Bit per sample */
    OMX_U32 nSamplingRate;            /**< Sampling rate of the source data.  Use 0 for
                                           variable or unknown sampling rate. */
    OMX_AUDIO_PCMMODETYPE ePCMMode;   /**< PCM mode enumeration */
    OMX_AUDIO_CHANNELTYPE eChannelMapping[OMX_AUDIO_MAXCHANNELS]; /**< Slot i contains channel defined by eChannelMap[i] */

} OMX_AUDIO_PARAM_PCMMODETYPE;


/** Audio channel mode.  This is used by both AAC and MP3, although the names are more appropriate
 * for the MP3.  For example, JointStereo for MP3 is CouplingChannels for AAC.
 */
typedef enum OMX_AUDIO_CHANNELMODETYPE {
    OMX_AUDIO_ChannelModeStereo = 0,  /**< 2 channels, the bitrate allocation between those
                                          two channels changes accordingly to each channel information */
    OMX_AUDIO_ChannelModeJointStereo, /**< mode that takes advantage of what is common between
                                           2 channels for higher compression gain */
    OMX_AUDIO_ChannelModeDual,        /**< 2 mono-channels, each channel is encoded with half
                                           the bitrate of the overall bitrate */
    OMX_AUDIO_ChannelModeMono,        /**< Mono channel mode */
    OMX_AUDIO_ChannelModeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_ChannelModeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_ChannelModeMax = 0x7FFFFFFF
} OMX_AUDIO_CHANNELMODETYPE;


typedef enum OMX_AUDIO_MP3STREAMFORMATTYPE {
    OMX_AUDIO_MP3StreamFormatMP1Layer3 = 0, /**< MP3 Audio MPEG 1 Layer 3 Stream format */
    OMX_AUDIO_MP3StreamFormatMP2Layer3,     /**< MP3 Audio MPEG 2 Layer 3 Stream format */
    OMX_AUDIO_MP3StreamFormatMP2_5Layer3,   /**< MP3 Audio MPEG2.5 Layer 3 Stream format */
    OMX_AUDIO_MP3StreamFormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_MP3StreamFormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_MP3StreamFormatMax = 0x7FFFFFFF
} OMX_AUDIO_MP3STREAMFORMATTYPE;

/** MP3 params */
typedef struct OMX_AUDIO_PARAM_MP3TYPE {
    OMX_U32 nSize;                 /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;      /**< OMX specification version information */
    OMX_U32 nPortIndex;            /**< port that this structure applies to */
    OMX_U32 nChannels;             /**< Number of channels */
    OMX_U32 nBitRate;              /**< Bit rate of the input data.  Use 0 for variable
                                        rate or unknown bit rates */
    OMX_U32 nSampleRate;           /**< Sampling rate of the source data.  Use 0 for
                                        variable or unknown sampling rate. */
    OMX_U32 nAudioBandWidth;       /**< Audio band width (in Hz) to which an encoder should
                                        limit the audio signal. Use 0 to let encoder decide */
    OMX_AUDIO_CHANNELMODETYPE eChannelMode;   /**< Channel mode enumeration */
    OMX_AUDIO_MP3STREAMFORMATTYPE eFormat;  /**< MP3 stream format */
} OMX_AUDIO_PARAM_MP3TYPE;


typedef enum OMX_AUDIO_AACSTREAMFORMATTYPE {
    OMX_AUDIO_AACStreamFormatMP2ADTS = 0, /**< AAC Audio Data Transport Stream 2 format */
    OMX_AUDIO_AACStreamFormatMP4ADTS,     /**< AAC Audio Data Transport Stream 4 format */
    OMX_AUDIO_AACStreamFormatMP4LOAS,     /**< AAC Low Overhead Audio Stream format */
    OMX_AUDIO_AACStreamFormatMP4LATM,     /**< AAC Low overhead Audio Transport Multiplex */
    OMX_AUDIO_AACStreamFormatADIF,        /**< AAC Audio Data Interchange Format */
    OMX_AUDIO_AACStreamFormatMP4FF,       /**< AAC inside MPEG-4/ISO File Format */
    OMX_AUDIO_AACStreamFormatRAW,         /**< AAC Raw Format */
    OMX_AUDIO_AACStreamFormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_AACStreamFormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_AACStreamFormatMax = 0x7FFFFFFF
} OMX_AUDIO_AACSTREAMFORMATTYPE;


/** AAC mode type.  Note that the term profile is used with the MPEG-2
 * standard and the term object type and profile is used with MPEG-4 */
typedef enum OMX_AUDIO_AACPROFILETYPE{
  OMX_AUDIO_AACObjectNull = 0,      /**< Null, not used */
  OMX_AUDIO_AACObjectMain = 1,      /**< AAC Main object */
  OMX_AUDIO_AACObjectLC,            /**< AAC Low Complexity object (AAC profile) */
  OMX_AUDIO_AACObjectSSR,           /**< AAC Scalable Sample Rate object */
  OMX_AUDIO_AACObjectLTP,           /**< AAC Long Term Prediction object */
  OMX_AUDIO_AACObjectHE,            /**< AAC High Efficiency (object type SBR, HE-AAC profile) */
  OMX_AUDIO_AACObjectScalable,      /**< AAC Scalable object */
  OMX_AUDIO_AACObjectERLC = 17,     /**< ER AAC Low Complexity object (Error Resilient AAC-LC) */
  OMX_AUDIO_AACObjectLD = 23,       /**< AAC Low Delay object (Error Resilient) */
  OMX_AUDIO_AACObjectHE_PS = 29,    /**< AAC High Efficiency with Parametric Stereo coding (HE-AAC v2, object type PS) */
  OMX_AUDIO_AACObjectELD = 39,      /** AAC Enhanced Low Delay. NOTE: Pending Khronos standardization **/
  OMX_AUDIO_AACObjectKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
  OMX_AUDIO_AACObjectVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
  OMX_AUDIO_AACObjectMax = 0x7FFFFFFF
} OMX_AUDIO_AACPROFILETYPE;


/** AAC tool usage (for nAACtools in OMX_AUDIO_PARAM_AACPROFILETYPE).
 * Required for encoder configuration and optional as decoder info output.
 * For MP3, OMX_AUDIO_CHANNELMODETYPE is sufficient. */
#define OMX_AUDIO_AACToolNone 0x00000000 /**< no AAC tools allowed (encoder config) or active (decoder info output) */
#define OMX_AUDIO_AACToolMS   0x00000001 /**< MS: Mid/side joint coding tool allowed or active */
#define OMX_AUDIO_AACToolIS   0x00000002 /**< IS: Intensity stereo tool allowed or active */
#define OMX_AUDIO_AACToolTNS  0x00000004 /**< TNS: Temporal Noise Shaping tool allowed or active */
#define OMX_AUDIO_AACToolPNS  0x00000008 /**< PNS: MPEG-4 Perceptual Noise substitution tool allowed or active */
#define OMX_AUDIO_AACToolLTP  0x00000010 /**< LTP: MPEG-4 Long Term Prediction tool allowed or active */
#define OMX_AUDIO_AACToolVendor 0x00010000 /**< NOT A KHRONOS VALUE, offset for vendor-specific additions */
#define OMX_AUDIO_AACToolAll  0x7FFFFFFF /**< all AAC tools allowed or active (*/

/** MPEG-4 AAC error resilience (ER) tool usage (for nAACERtools in OMX_AUDIO_PARAM_AACPROFILETYPE).
 * Required for ER encoder configuration and optional as decoder info output */
#define OMX_AUDIO_AACERNone  0x00000000  /**< no AAC ER tools allowed/used */
#define OMX_AUDIO_AACERVCB11 0x00000001  /**< VCB11: Virtual Code Books for AAC section data */
#define OMX_AUDIO_AACERRVLC  0x00000002  /**< RVLC: Reversible Variable Length Coding */
#define OMX_AUDIO_AACERHCR   0x00000004  /**< HCR: Huffman Codeword Reordering */
#define OMX_AUDIO_AACERAll   0x7FFFFFFF  /**< all AAC ER tools allowed/used */


/** AAC params */
typedef struct OMX_AUDIO_PARAM_AACPROFILETYPE {
    OMX_U32 nSize;                 /**< Size of this structure, in Bytes */
    OMX_VERSIONTYPE nVersion;      /**< OMX specification version information */
    OMX_U32 nPortIndex;            /**< Port that this structure applies to */
    OMX_U32 nChannels;             /**< Number of channels */
    OMX_U32 nSampleRate;           /**< Sampling rate of the source data.  Use 0 for
                                        variable or unknown sampling rate. */
    OMX_U32 nBitRate;              /**< Bit rate of the input data.  Use 0 for variable
                                        rate or unknown bit rates */
    OMX_U32 nAudioBandWidth;       /**< Audio band width (in Hz) to which an encoder should
                                        limit the audio signal. Use 0 to let encoder decide */
    OMX_U32 nFrameLength;          /**< Frame length (in audio samples per channel) of the codec.
                                        Can be 1024 or 960 (AAC-LC), 2048 (HE-AAC), 480 or 512 (AAC-LD).
                                        Use 0 to let encoder decide */
    OMX_U32 nAACtools;             /**< AAC tool usage */
    OMX_U32 nAACERtools;           /**< MPEG-4 AAC error resilience tool usage */
    OMX_AUDIO_AACPROFILETYPE eAACProfile;   /**< AAC profile enumeration */
    OMX_AUDIO_AACSTREAMFORMATTYPE eAACStreamFormat; /**< AAC stream format enumeration */
    OMX_AUDIO_CHANNELMODETYPE eChannelMode;   /**< Channel mode enumeration */
} OMX_AUDIO_PARAM_AACPROFILETYPE;


/** VORBIS params */
typedef struct OMX_AUDIO_PARAM_VORBISTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_U32 nChannels;        /**< Number of channels */
    OMX_U32 nBitRate;         /**< Bit rate of the encoded data data.  Use 0 for variable
                                   rate or unknown bit rates. Encoding is set to the
                                   bitrate closest to specified  value (in bps) */
    OMX_U32 nMinBitRate;      /**< Sets minimum bitrate (in bps). */
    OMX_U32 nMaxBitRate;      /**< Sets maximum bitrate (in bps). */

    OMX_U32 nSampleRate;      /**< Sampling rate of the source data.  Use 0 for
                                   variable or unknown sampling rate. */
    OMX_U32 nAudioBandWidth;  /**< Audio band width (in Hz) to which an encoder should
                                   limit the audio signal. Use 0 to let encoder decide */
    OMX_S32 nQuality;         /**< Sets encoding quality to n, between -1 (low) and 10 (high).
                                   In the default mode of operation, teh quality level is 3.
                                   Normal quality range is 0 - 10. */
    OMX_BOOL bManaged;        /**< Set  bitrate  management  mode. This turns off the
                                   normal VBR encoding, but allows hard or soft bitrate
                                   constraints to be enforced by the encoder. This mode can
                                   be slower, and may also be lower quality. It is
                                   primarily useful for streaming. */
    OMX_BOOL bDownmix;        /**< Downmix input from stereo to mono (has no effect on
                                   non-stereo streams). Useful for lower-bitrate encoding. */
} OMX_AUDIO_PARAM_VORBISTYPE;


/** FLAC params */
typedef struct OMX_AUDIO_PARAM_FLACTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_U32 nChannels;        /**< Number of channels */
    OMX_U32 nSampleRate;      /**< Sampling rate of the source data.  Use 0 for
                                   unknown sampling rate. */
    OMX_U32 nCompressionLevel;/**< FLAC compression level, from 0 (fastest compression)
                                   to 8 (highest compression */
} OMX_AUDIO_PARAM_FLACTYPE;


/** WMA Version */
typedef enum OMX_AUDIO_WMAFORMATTYPE {
  OMX_AUDIO_WMAFormatUnused = 0, /**< format unused or unknown */
  OMX_AUDIO_WMAFormat7,          /**< Windows Media Audio format 7 */
  OMX_AUDIO_WMAFormat8,          /**< Windows Media Audio format 8 */
  OMX_AUDIO_WMAFormat9,          /**< Windows Media Audio format 9 */
  OMX_AUDIO_WMAFormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
  OMX_AUDIO_WMAFormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
  OMX_AUDIO_WMAFormatMax = 0x7FFFFFFF
} OMX_AUDIO_WMAFORMATTYPE;


/** WMA Profile */
typedef enum OMX_AUDIO_WMAPROFILETYPE {
  OMX_AUDIO_WMAProfileUnused = 0,  /**< profile unused or unknown */
  OMX_AUDIO_WMAProfileL1,          /**< Windows Media audio version 9 profile L1 */
  OMX_AUDIO_WMAProfileL2,          /**< Windows Media audio version 9 profile L2 */
  OMX_AUDIO_WMAProfileL3,          /**< Windows Media audio version 9 profile L3 */
  OMX_AUDIO_WMAProfileKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
  OMX_AUDIO_WMAProfileVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
  OMX_AUDIO_WMAProfileMax = 0x7FFFFFFF
} OMX_AUDIO_WMAPROFILETYPE;


/** WMA params */
typedef struct OMX_AUDIO_PARAM_WMATYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_U16 nChannels;        /**< Number of channels */
    OMX_U32 nBitRate;         /**< Bit rate of the input data.  Use 0 for variable
                                   rate or unknown bit rates */
    OMX_AUDIO_WMAFORMATTYPE eFormat; /**< Version of WMA stream / data */
    OMX_AUDIO_WMAPROFILETYPE eProfile;  /**< Profile of WMA stream / data */
    OMX_U32 nSamplingRate;    /**< Sampling rate of the source data */
    OMX_U16 nBlockAlign;      /**< is the block alignment, or block size, in bytes of the audio codec */
    OMX_U16 nEncodeOptions;   /**< WMA Type-specific data */
    OMX_U32 nSuperBlockAlign; /**< WMA Type-specific data */
} OMX_AUDIO_PARAM_WMATYPE;

/**
 * RealAudio format
 */
typedef enum OMX_AUDIO_RAFORMATTYPE {
    OMX_AUDIO_RAFormatUnused = 0, /**< Format unused or unknown */
    OMX_AUDIO_RA8,                /**< RealAudio 8 codec */
    OMX_AUDIO_RA9,                /**< RealAudio 9 codec */
    OMX_AUDIO_RA10_AAC,           /**< MPEG-4 AAC codec for bitrates of more than 128kbps */
    OMX_AUDIO_RA10_CODEC,         /**< RealAudio codec for bitrates less than 128 kbps */
    OMX_AUDIO_RA10_LOSSLESS,      /**< RealAudio Lossless */
    OMX_AUDIO_RA10_MULTICHANNEL,  /**< RealAudio Multichannel */
    OMX_AUDIO_RA10_VOICE,         /**< RealAudio Voice for bitrates below 15 kbps */
    OMX_AUDIO_RAFormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_RAFormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_VIDEO_RAFormatMax = 0x7FFFFFFF
} OMX_AUDIO_RAFORMATTYPE;

/** RA (Real Audio) params */
typedef struct OMX_AUDIO_PARAM_RATYPE {
    OMX_U32 nSize;              /**< Size of this structure, in Bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< Port that this structure applies to */
    OMX_U32 nChannels;          /**< Number of channels */
    OMX_U32 nSamplingRate;      /**< is the sampling rate of the source data */
    OMX_U32 nBitsPerFrame;      /**< is the value for bits per frame  */
    OMX_U32 nSamplePerFrame;    /**< is the value for samples per frame */
    OMX_U32 nCouplingQuantBits; /**< is the number of coupling quantization bits in the stream */
    OMX_U32 nCouplingStartRegion;   /**< is the coupling start region in the stream  */
    OMX_U32 nNumRegions;        /**< is the number of regions value */
    OMX_AUDIO_RAFORMATTYPE eFormat; /**< is the RealAudio audio format */
} OMX_AUDIO_PARAM_RATYPE;


/** SBC Allocation Method Type */
typedef enum OMX_AUDIO_SBCALLOCMETHODTYPE {
  OMX_AUDIO_SBCAllocMethodLoudness, /**< Loudness allocation method */
  OMX_AUDIO_SBCAllocMethodSNR,      /**< SNR allocation method */
  OMX_AUDIO_SBCAllocMethodKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
  OMX_AUDIO_SBCAllocMethodVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
  OMX_AUDIO_SBCAllocMethodMax = 0x7FFFFFFF
} OMX_AUDIO_SBCALLOCMETHODTYPE;


/** SBC params */
typedef struct OMX_AUDIO_PARAM_SBCTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_U32 nChannels;         /**< Number of channels */
    OMX_U32 nBitRate;          /**< Bit rate of the input data.  Use 0 for variable
                                    rate or unknown bit rates */
    OMX_U32 nSampleRate;       /**< Sampling rate of the source data.  Use 0 for
                                    variable or unknown sampling rate. */
    OMX_U32 nBlocks;           /**< Number of blocks */
    OMX_U32 nSubbands;         /**< Number of subbands */
    OMX_U32 nBitPool;          /**< Bitpool value */
    OMX_BOOL bEnableBitrate;   /**< Use bitrate value instead of bitpool */
    OMX_AUDIO_CHANNELMODETYPE eChannelMode; /**< Channel mode enumeration */
    OMX_AUDIO_SBCALLOCMETHODTYPE eSBCAllocType;   /**< SBC Allocation method type */
} OMX_AUDIO_PARAM_SBCTYPE;


/** ADPCM stream format parameters */
typedef struct OMX_AUDIO_PARAM_ADPCMTYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< port that this structure applies to */
    OMX_U32 nChannels;          /**< Number of channels in the data stream (not
                                     necessarily the same as the number of channels
                                     to be rendered. */
    OMX_U32 nBitsPerSample;     /**< Number of bits in each sample */
    OMX_U32 nSampleRate;        /**< Sampling rate of the source data.  Use 0 for
                                    variable or unknown sampling rate. */
} OMX_AUDIO_PARAM_ADPCMTYPE;


/** G723 rate */
typedef enum OMX_AUDIO_G723RATE {
    OMX_AUDIO_G723ModeUnused = 0,  /**< AMRNB Mode unused / unknown */
    OMX_AUDIO_G723ModeLow,         /**< 5300 bps */
    OMX_AUDIO_G723ModeHigh,        /**< 6300 bps */
    OMX_AUDIO_G723ModeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_G723ModeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_G723ModeMax = 0x7FFFFFFF
} OMX_AUDIO_G723RATE;


/** G723 - Sample rate must be 8 KHz */
typedef struct OMX_AUDIO_PARAM_G723TYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_BOOL bDTX;                /**< Enable Discontinuous Transmisssion */
    OMX_AUDIO_G723RATE eBitRate;  /**< todo: Should this be moved to a config? */
    OMX_BOOL bHiPassFilter;       /**< Enable High Pass Filter */
    OMX_BOOL bPostFilter;         /**< Enable Post Filter */
} OMX_AUDIO_PARAM_G723TYPE;


/** ITU G726 (ADPCM) rate */
typedef enum OMX_AUDIO_G726MODE {
    OMX_AUDIO_G726ModeUnused = 0,  /**< G726 Mode unused / unknown */
    OMX_AUDIO_G726Mode16,          /**< 16 kbps */
    OMX_AUDIO_G726Mode24,          /**< 24 kbps */
    OMX_AUDIO_G726Mode32,          /**< 32 kbps, most common rate, also G721 */
    OMX_AUDIO_G726Mode40,          /**< 40 kbps */
    OMX_AUDIO_G726ModeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_G726ModeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_G726ModeMax = 0x7FFFFFFF
} OMX_AUDIO_G726MODE;


/** G.726 stream format parameters - must be at 8KHz */
typedef struct OMX_AUDIO_PARAM_G726TYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< port that this structure applies to */
    OMX_U32 nChannels;          /**< Number of channels in the data stream (not
                                     necessarily the same as the number of channels
                                     to be rendered. */
     OMX_AUDIO_G726MODE eG726Mode;
} OMX_AUDIO_PARAM_G726TYPE;


/** G729 coder type */
typedef enum OMX_AUDIO_G729TYPE {
    OMX_AUDIO_G729 = 0,           /**< ITU G.729  encoded data */
    OMX_AUDIO_G729A,              /**< ITU G.729 annex A  encoded data */
    OMX_AUDIO_G729B,              /**< ITU G.729 with annex B encoded data */
    OMX_AUDIO_G729AB,             /**< ITU G.729 annexes A and B encoded data */
    OMX_AUDIO_G729KhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_G729VendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_G729Max = 0x7FFFFFFF
} OMX_AUDIO_G729TYPE;


/** G729 stream format parameters - fixed 6KHz sample rate */
typedef struct OMX_AUDIO_PARAM_G729TYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_U32 nChannels;        /**< Number of channels in the data stream (not
                                   necessarily the same as the number of channels
                                   to be rendered. */
    OMX_BOOL bDTX;            /**< Enable Discontinuous Transmisssion */
    OMX_AUDIO_G729TYPE eBitType;
} OMX_AUDIO_PARAM_G729TYPE;


/** AMR Frame format */
typedef enum OMX_AUDIO_AMRFRAMEFORMATTYPE {
    OMX_AUDIO_AMRFrameFormatConformance = 0,  /**< Frame Format is AMR Conformance
                                                   (Standard) Format */
    OMX_AUDIO_AMRFrameFormatIF1,              /**< Frame Format is AMR Interface
                                                   Format 1 */
    OMX_AUDIO_AMRFrameFormatIF2,              /**< Frame Format is AMR Interface
                                                   Format 2*/
    OMX_AUDIO_AMRFrameFormatFSF,              /**< Frame Format is AMR File Storage
                                                   Format */
    OMX_AUDIO_AMRFrameFormatRTPPayload,       /**< Frame Format is AMR Real-Time
                                                   Transport Protocol Payload Format */
    OMX_AUDIO_AMRFrameFormatITU,              /**< Frame Format is ITU Format (added at Motorola request) */
    OMX_AUDIO_AMRFrameFormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_AMRFrameFormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_AMRFrameFormatMax = 0x7FFFFFFF
} OMX_AUDIO_AMRFRAMEFORMATTYPE;


/** AMR band mode */
typedef enum OMX_AUDIO_AMRBANDMODETYPE {
    OMX_AUDIO_AMRBandModeUnused = 0,          /**< AMRNB Mode unused / unknown */
    OMX_AUDIO_AMRBandModeNB0,                 /**< AMRNB Mode 0 =  4750 bps */
    OMX_AUDIO_AMRBandModeNB1,                 /**< AMRNB Mode 1 =  5150 bps */
    OMX_AUDIO_AMRBandModeNB2,                 /**< AMRNB Mode 2 =  5900 bps */
    OMX_AUDIO_AMRBandModeNB3,                 /**< AMRNB Mode 3 =  6700 bps */
    OMX_AUDIO_AMRBandModeNB4,                 /**< AMRNB Mode 4 =  7400 bps */
    OMX_AUDIO_AMRBandModeNB5,                 /**< AMRNB Mode 5 =  7950 bps */
    OMX_AUDIO_AMRBandModeNB6,                 /**< AMRNB Mode 6 = 10200 bps */
    OMX_AUDIO_AMRBandModeNB7,                 /**< AMRNB Mode 7 = 12200 bps */
    OMX_AUDIO_AMRBandModeWB0,                 /**< AMRWB Mode 0 =  6600 bps */
    OMX_AUDIO_AMRBandModeWB1,                 /**< AMRWB Mode 1 =  8850 bps */
    OMX_AUDIO_AMRBandModeWB2,                 /**< AMRWB Mode 2 = 12650 bps */
    OMX_AUDIO_AMRBandModeWB3,                 /**< AMRWB Mode 3 = 14250 bps */
    OMX_AUDIO_AMRBandModeWB4,                 /**< AMRWB Mode 4 = 15850 bps */
    OMX_AUDIO_AMRBandModeWB5,                 /**< AMRWB Mode 5 = 18250 bps */
    OMX_AUDIO_AMRBandModeWB6,                 /**< AMRWB Mode 6 = 19850 bps */
    OMX_AUDIO_AMRBandModeWB7,                 /**< AMRWB Mode 7 = 23050 bps */
    OMX_AUDIO_AMRBandModeWB8,                 /**< AMRWB Mode 8 = 23850 bps */
    OMX_AUDIO_AMRBandModeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_AMRBandModeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_AMRBandModeMax = 0x7FFFFFFF
} OMX_AUDIO_AMRBANDMODETYPE;


/** AMR Discontinuous Transmission mode */
typedef enum OMX_AUDIO_AMRDTXMODETYPE {
    OMX_AUDIO_AMRDTXModeOff = 0,        /**< AMR Discontinuous Transmission Mode is disabled */
    OMX_AUDIO_AMRDTXModeOnVAD1,         /**< AMR Discontinuous Transmission Mode using
                                             Voice Activity Detector 1 (VAD1) is enabled */
    OMX_AUDIO_AMRDTXModeOnVAD2,         /**< AMR Discontinuous Transmission Mode using
                                             Voice Activity Detector 2 (VAD2) is enabled */
    OMX_AUDIO_AMRDTXModeOnAuto,         /**< The codec will automatically select between
                                             Off, VAD1 or VAD2 modes */

    OMX_AUDIO_AMRDTXasEFR,             /**< DTX as EFR instead of AMR standard (3GPP 26.101, frame type =8,9,10) */

    OMX_AUDIO_AMRDTXModeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_AMRDTXModeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_AMRDTXModeMax = 0x7FFFFFFF
} OMX_AUDIO_AMRDTXMODETYPE;


/** AMR params */
typedef struct OMX_AUDIO_PARAM_AMRTYPE {
    OMX_U32 nSize;                          /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;               /**< OMX specification version information */
    OMX_U32 nPortIndex;                     /**< port that this structure applies to */
    OMX_U32 nChannels;                      /**< Number of channels */
    OMX_U32 nBitRate;                       /**< Bit rate read only field */
    OMX_AUDIO_AMRBANDMODETYPE eAMRBandMode; /**< AMR Band Mode enumeration */
    OMX_AUDIO_AMRDTXMODETYPE  eAMRDTXMode;  /**< AMR DTX Mode enumeration */
    OMX_AUDIO_AMRFRAMEFORMATTYPE eAMRFrameFormat; /**< AMR frame format enumeration */
} OMX_AUDIO_PARAM_AMRTYPE;


/** GSM_FR (ETSI 06.10, 3GPP 46.010) stream format parameters */
typedef struct OMX_AUDIO_PARAM_GSMFRTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_BOOL bDTX;            /**< Enable Discontinuous Transmisssion */
    OMX_BOOL bHiPassFilter;   /**< Enable High Pass Filter */
} OMX_AUDIO_PARAM_GSMFRTYPE;


/** GSM-HR (ETSI 06.20, 3GPP 46.020) stream format parameters */
typedef struct OMX_AUDIO_PARAM_GSMHRTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_BOOL bDTX;            /**< Enable Discontinuous Transmisssion */
    OMX_BOOL bHiPassFilter;   /**< Enable High Pass Filter */
} OMX_AUDIO_PARAM_GSMHRTYPE;


/** GSM-EFR (ETSI 06.60, 3GPP 46.060) stream format parameters */
typedef struct OMX_AUDIO_PARAM_GSMEFRTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_BOOL bDTX;            /**< Enable Discontinuous Transmisssion */
    OMX_BOOL bHiPassFilter;   /**< Enable High Pass Filter */
} OMX_AUDIO_PARAM_GSMEFRTYPE;


/** TDMA FR (TIA/EIA-136-420, VSELP 7.95kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_TDMAFRTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_BOOL bDTX;                /**< Enable Discontinuous Transmisssion */
    OMX_BOOL bHiPassFilter;       /**< Enable High Pass Filter */
} OMX_AUDIO_PARAM_TDMAFRTYPE;


/** TDMA EFR (TIA/EIA-136-410, ACELP 7.4kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_TDMAEFRTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_BOOL bDTX;                /**< Enable Discontinuous Transmisssion */
    OMX_BOOL bHiPassFilter;       /**< Enable High Pass Filter */
} OMX_AUDIO_PARAM_TDMAEFRTYPE;


/** PDC FR ( RCR-27, VSELP 6.7kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_PDCFRTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_BOOL bDTX;                /**< Enable Discontinuous Transmisssion */
    OMX_BOOL bHiPassFilter;       /**< Enable High Pass Filter */
} OMX_AUDIO_PARAM_PDCFRTYPE;


/** PDC EFR ( RCR-27, ACELP 6.7kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_PDCEFRTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_BOOL bDTX;                /**< Enable Discontinuous Transmisssion */
    OMX_BOOL bHiPassFilter;       /**< Enable High Pass Filter */
} OMX_AUDIO_PARAM_PDCEFRTYPE;

/** PDC HR ( RCR-27, PSI-CELP 3.45kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_PDCHRTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_BOOL bDTX;                /**< Enable Discontinuous Transmisssion */
    OMX_BOOL bHiPassFilter;       /**< Enable High Pass Filter */
} OMX_AUDIO_PARAM_PDCHRTYPE;


/** CDMA Rate types */
typedef enum OMX_AUDIO_CDMARATETYPE {
    OMX_AUDIO_CDMARateBlank = 0,          /**< CDMA encoded frame is blank */
    OMX_AUDIO_CDMARateFull,               /**< CDMA encoded frame in full rate */
    OMX_AUDIO_CDMARateHalf,               /**< CDMA encoded frame in half rate */
    OMX_AUDIO_CDMARateQuarter,            /**< CDMA encoded frame in quarter rate */
    OMX_AUDIO_CDMARateEighth,             /**< CDMA encoded frame in eighth rate (DTX)*/
    OMX_AUDIO_CDMARateErasure,            /**< CDMA erasure frame */
    OMX_AUDIO_CDMARateKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_CDMARateVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_CDMARateMax = 0x7FFFFFFF
} OMX_AUDIO_CDMARATETYPE;


/** QCELP8 (TIA/EIA-96, up to 8kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_QCELP8TYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_U32 nBitRate;             /**< Bit rate of the input data.  Use 0 for variable
                                       rate or unknown bit rates */
    OMX_AUDIO_CDMARATETYPE eCDMARate; /**< Frame rate */
    OMX_U32 nMinBitRate;          /**< minmal rate for the encoder = 1,2,3,4, default = 1 */
    OMX_U32 nMaxBitRate;          /**< maximal rate for the encoder = 1,2,3,4, default = 4 */
} OMX_AUDIO_PARAM_QCELP8TYPE;


/** QCELP13 ( CDMA, EIA/TIA-733, 13.3kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_QCELP13TYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_AUDIO_CDMARATETYPE eCDMARate; /**< Frame rate */
    OMX_U32 nMinBitRate;          /**< minmal rate for the encoder = 1,2,3,4, default = 1 */
    OMX_U32 nMaxBitRate;          /**< maximal rate for the encoder = 1,2,3,4, default = 4 */
} OMX_AUDIO_PARAM_QCELP13TYPE;


/** EVRC ( CDMA, EIA/TIA-127, RCELP up to 8.55kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_EVRCTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_AUDIO_CDMARATETYPE eCDMARate; /**< actual Frame rate */
    OMX_BOOL bRATE_REDUCon;       /**< RATE_REDUCtion is requested for this frame */
    OMX_U32 nMinBitRate;          /**< minmal rate for the encoder = 1,2,3,4, default = 1 */
    OMX_U32 nMaxBitRate;          /**< maximal rate for the encoder = 1,2,3,4, default = 4 */
    OMX_BOOL bHiPassFilter;       /**< Enable encoder's High Pass Filter */
    OMX_BOOL bNoiseSuppressor;    /**< Enable encoder's noise suppressor pre-processing */
    OMX_BOOL bPostFilter;         /**< Enable decoder's post Filter */
} OMX_AUDIO_PARAM_EVRCTYPE;


/** SMV ( up to 8.55kbps coder) stream format parameters */
typedef struct OMX_AUDIO_PARAM_SMVTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_U32 nChannels;            /**< Number of channels in the data stream (not
                                       necessarily the same as the number of channels
                                       to be rendered. */
    OMX_AUDIO_CDMARATETYPE eCDMARate; /**< Frame rate */
    OMX_BOOL bRATE_REDUCon;           /**< RATE_REDUCtion is requested for this frame */
    OMX_U32 nMinBitRate;          /**< minmal rate for the encoder = 1,2,3,4, default = 1 ??*/
    OMX_U32 nMaxBitRate;          /**< maximal rate for the encoder = 1,2,3,4, default = 4 ??*/
    OMX_BOOL bHiPassFilter;       /**< Enable encoder's High Pass Filter ??*/
    OMX_BOOL bNoiseSuppressor;    /**< Enable encoder's noise suppressor pre-processing */
    OMX_BOOL bPostFilter;         /**< Enable decoder's post Filter ??*/
} OMX_AUDIO_PARAM_SMVTYPE;


/** MIDI Format
 * @ingroup midi
 */
typedef enum OMX_AUDIO_MIDIFORMATTYPE
{
    OMX_AUDIO_MIDIFormatUnknown = 0, /**< MIDI Format unknown or don't care */
    OMX_AUDIO_MIDIFormatSMF0,        /**< Standard MIDI File Type 0 */
    OMX_AUDIO_MIDIFormatSMF1,        /**< Standard MIDI File Type 1 */
    OMX_AUDIO_MIDIFormatSMF2,        /**< Standard MIDI File Type 2 */
    OMX_AUDIO_MIDIFormatSPMIDI,      /**< SP-MIDI */
    OMX_AUDIO_MIDIFormatXMF0,        /**< eXtensible Music Format type 0 */
    OMX_AUDIO_MIDIFormatXMF1,        /**< eXtensible Music Format type 1 */
    OMX_AUDIO_MIDIFormatMobileXMF,   /**< Mobile XMF (eXtensible Music Format type 2) */
    OMX_AUDIO_MIDIFormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_MIDIFormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_MIDIFormatMax = 0x7FFFFFFF
} OMX_AUDIO_MIDIFORMATTYPE;


/** MIDI params
 * @ingroup midi
 */
typedef struct OMX_AUDIO_PARAM_MIDITYPE {
    OMX_U32 nSize;                 /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;      /**< OMX specification version information */
    OMX_U32 nPortIndex;            /**< port that this structure applies to */
    OMX_U32 nFileSize;             /**< size of the MIDI file in bytes, where the entire
                                        MIDI file passed in, otherwise if 0x0, the MIDI data
                                        is merged and streamed (instead of passed as an
                                        entire MIDI file) */
    OMX_BU32 sMaxPolyphony;        /**< Specifies the maximum simultaneous polyphonic
                                        voices. A value of zero indicates that the default
                                        polyphony of the device is used  */
    OMX_BOOL bLoadDefaultSound;    /**< Whether to load default sound
                                        bank at initialization */
    OMX_AUDIO_MIDIFORMATTYPE eMidiFormat; /**< Version of the MIDI file */
} OMX_AUDIO_PARAM_MIDITYPE;


/** Type of the MIDI sound bank
 * @ingroup midi
 */
typedef enum OMX_AUDIO_MIDISOUNDBANKTYPE {
    OMX_AUDIO_MIDISoundBankUnused = 0,           /**< unused/unknown soundbank type */
    OMX_AUDIO_MIDISoundBankDLS1,                 /**< DLS version 1 */
    OMX_AUDIO_MIDISoundBankDLS2,                 /**< DLS version 2 */
    OMX_AUDIO_MIDISoundBankMobileDLSBase,        /**< Mobile DLS, using the base functionality */
    OMX_AUDIO_MIDISoundBankMobileDLSPlusOptions, /**< Mobile DLS, using the specification-defined optional feature set */
    OMX_AUDIO_MIDISoundBankKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_MIDISoundBankVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_MIDISoundBankMax = 0x7FFFFFFF
} OMX_AUDIO_MIDISOUNDBANKTYPE;


/** Bank Layout describes how bank MSB & LSB are used in the DLS instrument definitions sound bank
 * @ingroup midi
 */
typedef enum OMX_AUDIO_MIDISOUNDBANKLAYOUTTYPE {
   OMX_AUDIO_MIDISoundBankLayoutUnused = 0,   /**< unused/unknown soundbank type */
   OMX_AUDIO_MIDISoundBankLayoutGM,           /**< GS layout (based on bank MSB 0x00) */
   OMX_AUDIO_MIDISoundBankLayoutGM2,          /**< General MIDI 2 layout (using MSB 0x78/0x79, LSB 0x00) */
   OMX_AUDIO_MIDISoundBankLayoutUser,         /**< Does not conform to any bank numbering standards */
   OMX_AUDIO_MIDISoundBankLayoutKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
   OMX_AUDIO_MIDISoundBankLayoutVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
   OMX_AUDIO_MIDISoundBankLayoutMax = 0x7FFFFFFF
} OMX_AUDIO_MIDISOUNDBANKLAYOUTTYPE;


/** MIDI params to load/unload user soundbank
 * @ingroup midi
 */
typedef struct OMX_AUDIO_PARAM_MIDILOADUSERSOUNDTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_U32 nDLSIndex;        /**< DLS file index to be loaded */
    OMX_U32 nDLSSize;         /**< Size in bytes */
    OMX_PTR pDLSData;         /**< Pointer to DLS file data */
    OMX_AUDIO_MIDISOUNDBANKTYPE eMidiSoundBank;   /**< Midi sound bank type enumeration */
    OMX_AUDIO_MIDISOUNDBANKLAYOUTTYPE eMidiSoundBankLayout; /**< Midi sound bank layout enumeration */
} OMX_AUDIO_PARAM_MIDILOADUSERSOUNDTYPE;


/** Structure for Live MIDI events and MIP messages.
 * (MIP = Maximum Instantaneous Polyphony; part of the SP-MIDI standard.)
 * @ingroup midi
 */
typedef struct OMX_AUDIO_CONFIG_MIDIIMMEDIATEEVENTTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< Port that this structure applies to */
    OMX_U32 nMidiEventSize;   /**< Size of immediate MIDI events or MIP message in bytes  */
    OMX_U8 nMidiEvents[1];    /**< MIDI event array to be rendered immediately, or an
                                   array for the MIP message buffer, where the size is
                                   indicated by nMidiEventSize */
} OMX_AUDIO_CONFIG_MIDIIMMEDIATEEVENTTYPE;


/** MIDI sound bank/ program pair in a given channel
 * @ingroup midi
 */
typedef struct OMX_AUDIO_CONFIG_MIDISOUNDBANKPROGRAMTYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< Port that this structure applies to */
    OMX_U32 nChannel;           /**< Valid channel values range from 1 to 16 */
    OMX_U16 nIDProgram;         /**< Valid program ID range is 1 to 128 */
    OMX_U16 nIDSoundBank;       /**< Sound bank ID */
    OMX_U32 nUserSoundBankIndex;/**< User soundbank index, easier to access soundbanks
                                     by index if multiple banks are present */
} OMX_AUDIO_CONFIG_MIDISOUNDBANKPROGRAMTYPE;


/** MIDI control
 * @ingroup midi
 */
typedef struct OMX_AUDIO_CONFIG_MIDICONTROLTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_BS32 sPitchTransposition; /**< Pitch transposition in semitones, stored as Q22.10
                                       format based on JAVA MMAPI (JSR-135) requirement */
    OMX_BU32 sPlayBackRate;       /**< Relative playback rate, stored as Q14.17 fixed-point
                                       number based on JSR-135 requirement */
    OMX_BU32 sTempo ;             /**< Tempo in beats per minute (BPM), stored as Q22.10
                                       fixed-point number based on JSR-135 requirement */
    OMX_U32 nMaxPolyphony;        /**< Specifies the maximum simultaneous polyphonic
                                       voices. A value of zero indicates that the default
                                       polyphony of the device is used  */
    OMX_U32 nNumRepeat;           /**< Number of times to repeat playback */
    OMX_U32 nStopTime;            /**< Time in milliseconds to indicate when playback
                                       will stop automatically.  Set to zero if not used */
    OMX_U16 nChannelMuteMask;     /**< 16 bit mask for channel mute status */
    OMX_U16 nChannelSoloMask;     /**< 16 bit mask for channel solo status */
    OMX_U32 nTrack0031MuteMask;   /**< 32 bit mask for track mute status. Note: This is for tracks 0-31 */
    OMX_U32 nTrack3263MuteMask;   /**< 32 bit mask for track mute status. Note: This is for tracks 32-63 */
    OMX_U32 nTrack0031SoloMask;   /**< 32 bit mask for track solo status. Note: This is for tracks 0-31 */
    OMX_U32 nTrack3263SoloMask;   /**< 32 bit mask for track solo status. Note: This is for tracks 32-63 */

} OMX_AUDIO_CONFIG_MIDICONTROLTYPE;


/** MIDI Playback States
 * @ingroup midi
 */
typedef enum OMX_AUDIO_MIDIPLAYBACKSTATETYPE {
  OMX_AUDIO_MIDIPlayBackStateUnknown = 0,      /**< Unknown state or state does not map to
                                                    other defined states */
  OMX_AUDIO_MIDIPlayBackStateClosedEngaged,    /**< No MIDI resource is currently open.
                                                    The MIDI engine is currently processing
                                                    MIDI events. */
  OMX_AUDIO_MIDIPlayBackStateParsing,          /**< A MIDI resource is open and is being
                                                    primed. The MIDI engine is currently
                                                    processing MIDI events. */
  OMX_AUDIO_MIDIPlayBackStateOpenEngaged,      /**< A MIDI resource is open and primed but
                                                    not playing. The MIDI engine is currently
                                                    processing MIDI events. The transition to
                                                    this state is only possible from the
                                                    OMX_AUDIO_MIDIPlayBackStatePlaying state,
                                                    when the 'playback head' reaches the end
                                                    of media data or the playback stops due
                                                    to stop time set.*/
  OMX_AUDIO_MIDIPlayBackStatePlaying,          /**< A MIDI resource is open and currently
                                                    playing. The MIDI engine is currently
                                                    processing MIDI events.*/
  OMX_AUDIO_MIDIPlayBackStatePlayingPartially, /**< Best-effort playback due to SP-MIDI/DLS
                                                    resource constraints */
  OMX_AUDIO_MIDIPlayBackStatePlayingSilently,  /**< Due to system resource constraints and
                                                    SP-MIDI content constraints, there is
                                                    no audible MIDI content during playback
                                                    currently. The situation may change if
                                                    resources are freed later.*/
  OMX_AUDIO_MIDIPlayBackStateKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
  OMX_AUDIO_MIDIPlayBackStateVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
  OMX_AUDIO_MIDIPlayBackStateMax = 0x7FFFFFFF
} OMX_AUDIO_MIDIPLAYBACKSTATETYPE;


/** MIDI status
 * @ingroup midi
 */
typedef struct OMX_AUDIO_CONFIG_MIDISTATUSTYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< port that this structure applies to */
    OMX_U16 nNumTracks;         /**< Number of MIDI tracks in the file, read only field.
                                     NOTE: May not return a meaningful value until the entire
                                     file is parsed and buffered.  */
    OMX_U32 nDuration;          /**< The length of the currently open MIDI resource
                                     in milliseconds. NOTE: May not return a meaningful value
                                     until the entire file is parsed and buffered.  */
    OMX_U32 nPosition;          /**< Current Position of the MIDI resource being played
                                     in milliseconds */
    OMX_BOOL bVibra;            /**< Does Vibra track exist? NOTE: May not return a meaningful
                                     value until the entire file is parsed and buffered. */
    OMX_U32 nNumMetaEvents;     /**< Total number of MIDI Meta Events in the currently
                                     open MIDI resource. NOTE: May not return a meaningful value
                                     until the entire file is parsed and buffered.  */
    OMX_U32 nNumActiveVoices;   /**< Number of active voices in the currently playing
                                     MIDI resource. NOTE: May not return a meaningful value until
                                     the entire file is parsed and buffered. */
    OMX_AUDIO_MIDIPLAYBACKSTATETYPE eMIDIPlayBackState;  /**< MIDI playback state enumeration, read only field */
} OMX_AUDIO_CONFIG_MIDISTATUSTYPE;


/** MIDI Meta Event structure one per Meta Event.
 *  MIDI Meta Events are like audio metadata, except that they are interspersed
 *  with the MIDI content throughout the file and are not localized in the header.
 *  As such, it is necessary to retrieve information about these Meta Events from
 *  the engine, as it encounters these Meta Events within the MIDI content.
 *  For example, SMF files can have up to 14 types of MIDI Meta Events (copyright,
 *  author, default tempo, etc.) scattered throughout the file.
 *  @ingroup midi
 */
typedef struct OMX_AUDIO_CONFIG_MIDIMETAEVENTTYPE{
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_U32 nIndex;           /**< Index of Meta Event */
    OMX_U8 nMetaEventType;    /**< Meta Event Type, 7bits (i.e. 0 - 127) */
    OMX_U32 nMetaEventSize;   /**< size of the Meta Event in bytes */
    OMX_U32 nTrack;           /**< track number for the meta event */
    OMX_U32 nPosition;        /**< Position of the meta-event in milliseconds */
} OMX_AUDIO_CONFIG_MIDIMETAEVENTTYPE;


/** MIDI Meta Event Data structure - one per Meta Event.
 * @ingroup midi
 */
typedef struct OMX_AUDIO_CONFIG_MIDIMETAEVENTDATATYPE{
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex;       /**< port that this structure applies to */
    OMX_U32 nIndex;           /**< Index of Meta Event */
    OMX_U32 nMetaEventSize;   /**< size of the Meta Event in bytes */
    OMX_U8 nData[1];          /**< array of one or more bytes of meta data
                                   as indicated by the nMetaEventSize field */
} OMX_AUDIO_CONFIG__MIDIMETAEVENTDATATYPE;


/** Audio Volume adjustment for a port */
typedef struct OMX_AUDIO_CONFIG_VOLUMETYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< Port index indicating which port to
                                     set.  Select the input port to set
                                     just that port's volume.  Select the
                                     output port to adjust the master
                                     volume. */
    OMX_BOOL bLinear;           /**< Is the volume to be set in linear (0.100)
                                     or logarithmic scale (mB) */
    OMX_BS32 sVolume;           /**< Volume linear setting in the 0..100 range, OR
                                     Volume logarithmic setting for this port.  The values
                                     for volume are in mB (millibels = 1/100 dB) relative
                                     to a gain of 1 (e.g. the output is the same as the
                                     input level).  Values are in mB from nMax
                                     (maximum volume) to nMin mB (typically negative).
                                     Since the volume is "voltage"
                                     and not a "power", it takes a setting of
                                     -600 mB to decrease the volume by 1/2.  If
                                     a component cannot accurately set the
                                     volume to the requested value, it must
                                     set the volume to the closest value BELOW
                                     the requested value.  When getting the
                                     volume setting, the current actual volume
                                     must be returned. */
} OMX_AUDIO_CONFIG_VOLUMETYPE;


/** Audio Volume adjustment for a channel */
typedef struct OMX_AUDIO_CONFIG_CHANNELVOLUMETYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< Port index indicating which port to
                                     set.  Select the input port to set
                                     just that port's volume.  Select the
                                     output port to adjust the master
                                     volume. */
    OMX_U32 nChannel;           /**< channel to select from 0 to N-1,
                                     using OMX_ALL to apply volume settings
                                     to all channels */
    OMX_BOOL bLinear;           /**< Is the volume to be set in linear (0.100) or
                                     logarithmic scale (mB) */
    OMX_BS32 sVolume;           /**< Volume linear setting in the 0..100 range, OR
                                     Volume logarithmic setting for this port.
                                     The values for volume are in mB
                                     (millibels = 1/100 dB) relative to a gain
                                     of 1 (e.g. the output is the same as the
                                     input level).  Values are in mB from nMax
                                     (maximum volume) to nMin mB (typically negative).
                                     Since the volume is "voltage"
                                     and not a "power", it takes a setting of
                                     -600 mB to decrease the volume by 1/2.  If
                                     a component cannot accurately set the
                                     volume to the requested value, it must
                                     set the volume to the closest value BELOW
                                     the requested value.  When getting the
                                     volume setting, the current actual volume
                                     must be returned. */
    OMX_BOOL bIsMIDI;           /**< TRUE if nChannel refers to a MIDI channel,
                                     FALSE otherwise */
} OMX_AUDIO_CONFIG_CHANNELVOLUMETYPE;


/** Audio balance setting */
typedef struct OMX_AUDIO_CONFIG_BALANCETYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< Port index indicating which port to
                                     set.  Select the input port to set
                                     just that port's balance.  Select the
                                     output port to adjust the master
                                     balance. */
    OMX_S32 nBalance;           /**< balance setting for this port
                                     (-100 to 100, where -100 indicates
                                     all left, and no right */
} OMX_AUDIO_CONFIG_BALANCETYPE;


/** Audio Port mute */
typedef struct OMX_AUDIO_CONFIG_MUTETYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< Port index indicating which port to
                                     set.  Select the input port to set
                                     just that port's mute.  Select the
                                     output port to adjust the master
                                     mute. */
    OMX_BOOL bMute;             /**< Mute setting for this port */
} OMX_AUDIO_CONFIG_MUTETYPE;


/** Audio Channel mute */
typedef struct OMX_AUDIO_CONFIG_CHANNELMUTETYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< port that this structure applies to */
    OMX_U32 nChannel;           /**< channel to select from 0 to N-1,
                                     using OMX_ALL to apply mute settings
                                     to all channels */
    OMX_BOOL bMute;             /**< Mute setting for this channel */
    OMX_BOOL bIsMIDI;           /**< TRUE if nChannel refers to a MIDI channel,
                                     FALSE otherwise */
} OMX_AUDIO_CONFIG_CHANNELMUTETYPE;



/** Enable / Disable for loudness control, which boosts bass and to a
 *  smaller extent high end frequencies to compensate for hearing
 *  ability at the extreme ends of the audio spectrum
 */
typedef struct OMX_AUDIO_CONFIG_LOUDNESSTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_BOOL bLoudness;        /**< Enable/disable for loudness */
} OMX_AUDIO_CONFIG_LOUDNESSTYPE;


/** Enable / Disable for bass, which controls low frequencies
 */
typedef struct OMX_AUDIO_CONFIG_BASSTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_BOOL bEnable;          /**< Enable/disable for bass control */
    OMX_S32 nBass;             /**< bass setting for the port, as a
                                    continuous value from -100 to 100
                                    (0 means no change in bass level)*/
} OMX_AUDIO_CONFIG_BASSTYPE;


/** Enable / Disable for treble, which controls high frequencies tones
 */
typedef struct OMX_AUDIO_CONFIG_TREBLETYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_BOOL bEnable;          /**< Enable/disable for treble control */
    OMX_S32  nTreble;          /**< treble setting for the port, as a
                                    continuous value from -100 to 100
                                    (0 means no change in treble level) */
} OMX_AUDIO_CONFIG_TREBLETYPE;


/** An equalizer is typically used for two reasons: to compensate for an
 *  sub-optimal frequency response of a system to make it sound more natural
 *  or to create intentionally some unnatural coloring to the sound to create
 *  an effect.
 *  @ingroup effects
 */
typedef struct OMX_AUDIO_CONFIG_EQUALIZERTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_BOOL bEnable;          /**< Enable/disable for equalizer */
    OMX_BU32 sBandIndex;       /**< Band number to be set.  Upper Limit is
                                    N-1, where N is the number of bands, lower limit is 0 */
    OMX_BU32 sCenterFreq;      /**< Center frequecies in Hz.  This is a
                                    read only element and is used to determine
                                    the lower, center and upper frequency of
                                    this band.  */
    OMX_BS32 sBandLevel;       /**< band level in millibels */
} OMX_AUDIO_CONFIG_EQUALIZERTYPE;


/** Stereo widening mode type
 * @ingroup effects
 */
typedef enum OMX_AUDIO_STEREOWIDENINGTYPE {
    OMX_AUDIO_StereoWideningHeadphones,    /**< Stereo widening for loudspeakers */
    OMX_AUDIO_StereoWideningLoudspeakers,  /**< Stereo widening for closely spaced loudspeakers */
    OMX_AUDIO_StereoWideningKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_AUDIO_StereoWideningVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_AUDIO_StereoWideningMax = 0x7FFFFFFF
} OMX_AUDIO_STEREOWIDENINGTYPE;


/** Control for stereo widening, which is a special 2-channel
 *  case of the audio virtualizer effect. For example, for 5.1-channel
 *  output, it translates to virtual surround sound.
 * @ingroup effects
 */
typedef struct OMX_AUDIO_CONFIG_STEREOWIDENINGTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_BOOL bEnable;          /**< Enable/disable for stereo widening control */
    OMX_AUDIO_STEREOWIDENINGTYPE eWideningType; /**< Stereo widening algorithm type */
    OMX_U32  nStereoWidening;  /**< stereo widening setting for the port,
                                    as a continuous value from 0 to 100  */
} OMX_AUDIO_CONFIG_STEREOWIDENINGTYPE;


/** The chorus effect (or ``choralizer'') is any signal processor which makes
 *  one sound source (such as a voice) sound like many such sources singing
 *  (or playing) in unison. Since performance in unison is never exact, chorus
 *  effects simulate this by making independently modified copies of the input
 *  signal. Modifications may include (1) delay, (2) frequency shift, and
 *  (3) amplitude modulation.
 * @ingroup effects
 */
typedef struct OMX_AUDIO_CONFIG_CHORUSTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_BOOL bEnable;          /**< Enable/disable for chorus */
    OMX_BU32 sDelay;           /**< average delay in milliseconds */
    OMX_BU32 sModulationRate;  /**< rate of modulation in millihertz */
    OMX_U32 nModulationDepth;  /**< depth of modulation as a percentage of
                                    delay (i.e. 0 to 100) */
    OMX_BU32 nFeedback;        /**< Feedback from chorus output to input in percentage */
} OMX_AUDIO_CONFIG_CHORUSTYPE;


/** Reverberation is part of the reflected sound that follows the early
 *  reflections. In a typical room, this consists of a dense succession of
 *  echoes whose energy decays exponentially. The reverberation effect structure
 *  as defined here includes both (early) reflections as well as (late) reverberations.
 * @ingroup effects
 */
typedef struct OMX_AUDIO_CONFIG_REVERBERATIONTYPE {
    OMX_U32 nSize;                /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;     /**< OMX specification version information */
    OMX_U32 nPortIndex;           /**< port that this structure applies to */
    OMX_BOOL bEnable;             /**< Enable/disable for reverberation control */
    OMX_BS32 sRoomLevel;          /**< Intensity level for the whole room effect
                                       (i.e. both early reflections and late
                                       reverberation) in millibels */
    OMX_BS32 sRoomHighFreqLevel;  /**< Attenuation at high frequencies
                                       relative to the intensity at low
                                       frequencies in millibels */
    OMX_BS32 sReflectionsLevel;   /**< Intensity level of early reflections
                                       (relative to room value), in millibels */
    OMX_BU32 sReflectionsDelay;   /**< Delay time of the first reflection relative
                                       to the direct path, in milliseconds */
    OMX_BS32 sReverbLevel;        /**< Intensity level of late reverberation
                                       relative to room level, in millibels */
    OMX_BU32 sReverbDelay;        /**< Time delay from the first early reflection
                                       to the beginning of the late reverberation
                                       section, in milliseconds */
    OMX_BU32 sDecayTime;          /**< Late reverberation decay time at low
                                       frequencies, in milliseconds */
    OMX_BU32 nDecayHighFreqRatio; /**< Ratio of high frequency decay time relative
                                       to low frequency decay time in percent  */
    OMX_U32 nDensity;             /**< Modal density in the late reverberation decay,
                                       in percent (i.e. 0 - 100) */
    OMX_U32 nDiffusion;           /**< Echo density in the late reverberation decay,
                                       in percent (i.e. 0 - 100) */
    OMX_BU32 sReferenceHighFreq;  /**< Reference high frequency in Hertz. This is
                                       the frequency used as the reference for all
                                       the high-frequency settings above */

} OMX_AUDIO_CONFIG_REVERBERATIONTYPE;


/** Possible settings for the Echo Cancelation structure to use
 * @ingroup effects
 */
typedef enum OMX_AUDIO_ECHOCANTYPE {
   OMX_AUDIO_EchoCanOff = 0,    /**< Echo Cancellation is disabled */
   OMX_AUDIO_EchoCanNormal,     /**< Echo Cancellation normal operation -
                                     echo from plastics and face */
   OMX_AUDIO_EchoCanHFree,      /**< Echo Cancellation optimized for
                                     Hands Free operation */
   OMX_AUDIO_EchoCanCarKit,    /**< Echo Cancellation optimized for
                                     Car Kit (longer echo) */
   OMX_AUDIO_EchoCanKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
   OMX_AUDIO_EchoCanVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
   OMX_AUDIO_EchoCanMax = 0x7FFFFFFF
} OMX_AUDIO_ECHOCANTYPE;


/** Enable / Disable for echo cancelation, which removes undesired echo's
 *  from the audio
 * @ingroup effects
 */
typedef struct OMX_AUDIO_CONFIG_ECHOCANCELATIONTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_AUDIO_ECHOCANTYPE eEchoCancelation; /**< Echo cancelation settings */
} OMX_AUDIO_CONFIG_ECHOCANCELATIONTYPE;


/** Enable / Disable for noise reduction, which undesired noise from
 * the audio
 * @ingroup effects
 */
typedef struct OMX_AUDIO_CONFIG_NOISEREDUCTIONTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_U32 nPortIndex;        /**< port that this structure applies to */
    OMX_BOOL bNoiseReduction;  /**< Enable/disable for noise reduction */
} OMX_AUDIO_CONFIG_NOISEREDUCTIONTYPE;

/** @} */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */
