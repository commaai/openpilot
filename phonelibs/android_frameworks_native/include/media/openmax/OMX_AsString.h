/*
 * Copyright 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* NOTE: This file contains several sections for individual OMX include files.
   Each section has its own include guard.  This file should be included AFTER
   the OMX include files. */

#ifdef OMX_Audio_h
/* asString definitions if media/openmax/OMX_Audio.h was included */

#ifndef AS_STRING_FOR_OMX_AUDIO_H
#define AS_STRING_FOR_OMX_AUDIO_H

inline static const char *asString(OMX_AUDIO_CODINGTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_AUDIO_CodingUnused:     return "Unused";      // unused
        case OMX_AUDIO_CodingAutoDetect: return "AutoDetect";  // unused
        case OMX_AUDIO_CodingPCM:        return "PCM";
        case OMX_AUDIO_CodingADPCM:      return "ADPCM";       // unused
        case OMX_AUDIO_CodingAMR:        return "AMR";
        case OMX_AUDIO_CodingGSMFR:      return "GSMFR";
        case OMX_AUDIO_CodingGSMEFR:     return "GSMEFR";      // unused
        case OMX_AUDIO_CodingGSMHR:      return "GSMHR";       // unused
        case OMX_AUDIO_CodingPDCFR:      return "PDCFR";       // unused
        case OMX_AUDIO_CodingPDCEFR:     return "PDCEFR";      // unused
        case OMX_AUDIO_CodingPDCHR:      return "PDCHR";       // unused
        case OMX_AUDIO_CodingTDMAFR:     return "TDMAFR";      // unused
        case OMX_AUDIO_CodingTDMAEFR:    return "TDMAEFR";     // unused
        case OMX_AUDIO_CodingQCELP8:     return "QCELP8";      // unused
        case OMX_AUDIO_CodingQCELP13:    return "QCELP13";     // unused
        case OMX_AUDIO_CodingEVRC:       return "EVRC";        // unused
        case OMX_AUDIO_CodingSMV:        return "SMV";         // unused
        case OMX_AUDIO_CodingG711:       return "G711";
        case OMX_AUDIO_CodingG723:       return "G723";        // unused
        case OMX_AUDIO_CodingG726:       return "G726";        // unused
        case OMX_AUDIO_CodingG729:       return "G729";        // unused
        case OMX_AUDIO_CodingAAC:        return "AAC";
        case OMX_AUDIO_CodingMP3:        return "MP3";
        case OMX_AUDIO_CodingSBC:        return "SBC";         // unused
        case OMX_AUDIO_CodingVORBIS:     return "VORBIS";
        case OMX_AUDIO_CodingWMA:        return "WMA";         // unused
        case OMX_AUDIO_CodingRA:         return "RA";          // unused
        case OMX_AUDIO_CodingMIDI:       return "MIDI";        // unused
        case OMX_AUDIO_CodingFLAC:       return "FLAC";
        default:                         return def;
    }
}

inline static const char *asString(OMX_AUDIO_PCMMODETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_AUDIO_PCMModeLinear: return "Linear";
        case OMX_AUDIO_PCMModeALaw:   return "ALaw";
        case OMX_AUDIO_PCMModeMULaw:  return "MULaw";
        default:                      return def;
    }
}

inline static const char *asString(OMX_AUDIO_CHANNELTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_AUDIO_ChannelNone: return "None";  // unused
        case OMX_AUDIO_ChannelLF:   return "LF";
        case OMX_AUDIO_ChannelRF:   return "RF";
        case OMX_AUDIO_ChannelCF:   return "CF";
        case OMX_AUDIO_ChannelLS:   return "LS";
        case OMX_AUDIO_ChannelRS:   return "RS";
        case OMX_AUDIO_ChannelLFE:  return "LFE";
        case OMX_AUDIO_ChannelCS:   return "CS";
        case OMX_AUDIO_ChannelLR:   return "LR";
        case OMX_AUDIO_ChannelRR:   return "RR";
        default:                    return def;
    }
}

inline static const char *asString(OMX_AUDIO_CHANNELMODETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_AUDIO_ChannelModeStereo:      return "Stereo";
//      case OMX_AUDIO_ChannelModeJointStereo: return "JointStereo";
//      case OMX_AUDIO_ChannelModeDual:        return "Dual";
        case OMX_AUDIO_ChannelModeMono:        return "Mono";
        default:                               return def;
    }
}

inline static const char *asString(OMX_AUDIO_AACSTREAMFORMATTYPE i, const char *def = "??") {
    switch (i) {
//      case OMX_AUDIO_AACStreamFormatMP2ADTS: return "MP2ADTS";
        case OMX_AUDIO_AACStreamFormatMP4ADTS: return "MP4ADTS";
//      case OMX_AUDIO_AACStreamFormatMP4LOAS: return "MP4LOAS";
//      case OMX_AUDIO_AACStreamFormatMP4LATM: return "MP4LATM";
//      case OMX_AUDIO_AACStreamFormatADIF:    return "ADIF";
        case OMX_AUDIO_AACStreamFormatMP4FF:   return "MP4FF";
//      case OMX_AUDIO_AACStreamFormatRAW:     return "RAW";
        default:                               return def;
    }
}

inline static const char *asString(OMX_AUDIO_AMRFRAMEFORMATTYPE i, const char *def = "??") {
    switch (i) {
//      case OMX_AUDIO_AMRFrameFormatConformance: return "Conformance";
//      case OMX_AUDIO_AMRFrameFormatIF1:         return "IF1";
//      case OMX_AUDIO_AMRFrameFormatIF2:         return "IF2";
        case OMX_AUDIO_AMRFrameFormatFSF:         return "FSF";
//      case OMX_AUDIO_AMRFrameFormatRTPPayload:  return "RTPPayload";
//      case OMX_AUDIO_AMRFrameFormatITU:         return "ITU";
        default:                                  return def;
    }
}

inline static const char *asString(OMX_AUDIO_AMRBANDMODETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_AUDIO_AMRBandModeUnused: return "Unused";
        case OMX_AUDIO_AMRBandModeNB0:    return "NB0";
        case OMX_AUDIO_AMRBandModeNB1:    return "NB1";
        case OMX_AUDIO_AMRBandModeNB2:    return "NB2";
        case OMX_AUDIO_AMRBandModeNB3:    return "NB3";
        case OMX_AUDIO_AMRBandModeNB4:    return "NB4";
        case OMX_AUDIO_AMRBandModeNB5:    return "NB5";
        case OMX_AUDIO_AMRBandModeNB6:    return "NB6";
        case OMX_AUDIO_AMRBandModeNB7:    return "NB7";
        case OMX_AUDIO_AMRBandModeWB0:    return "WB0";
        case OMX_AUDIO_AMRBandModeWB1:    return "WB1";
        case OMX_AUDIO_AMRBandModeWB2:    return "WB2";
        case OMX_AUDIO_AMRBandModeWB3:    return "WB3";
        case OMX_AUDIO_AMRBandModeWB4:    return "WB4";
        case OMX_AUDIO_AMRBandModeWB5:    return "WB5";
        case OMX_AUDIO_AMRBandModeWB6:    return "WB6";
        case OMX_AUDIO_AMRBandModeWB7:    return "WB7";
        case OMX_AUDIO_AMRBandModeWB8:    return "WB8";
        default:                          return def;
    }
}

inline static const char *asString(OMX_AUDIO_AMRDTXMODETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_AUDIO_AMRDTXModeOff:    return "ModeOff";
//      case OMX_AUDIO_AMRDTXModeOnVAD1: return "ModeOnVAD1";
//      case OMX_AUDIO_AMRDTXModeOnVAD2: return "ModeOnVAD2";
//      case OMX_AUDIO_AMRDTXModeOnAuto: return "ModeOnAuto";
//      case OMX_AUDIO_AMRDTXasEFR:      return "asEFR";
        default:                         return def;
    }
}

#endif // AS_STRING_FOR_OMX_AUDIO_H

#endif // OMX_Audio_h

#ifdef OMX_AudioExt_h
/* asString definitions if media/openmax/OMX_AudioExt.h was included */

#ifndef AS_STRING_FOR_OMX_AUDIOEXT_H
#define AS_STRING_FOR_OMX_AUDIOEXT_H

inline static const char *asString(OMX_AUDIO_CODINGEXTTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_AUDIO_CodingAndroidAC3:  return "AndroidAC3";
        case OMX_AUDIO_CodingAndroidOPUS: return "AndroidOPUS";
        default:                          return asString((OMX_AUDIO_CODINGTYPE)i, def);
    }
}

#endif // AS_STRING_FOR_OMX_AUDIOEXT_H

#endif // OMX_AudioExt_h

#ifdef OMX_Component_h
/* asString definitions if media/openmax/OMX_Component.h was included */

#ifndef AS_STRING_FOR_OMX_COMPONENT_H
#define AS_STRING_FOR_OMX_COMPONENT_H

inline static const char *asString(OMX_PORTDOMAINTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_PortDomainAudio: return "Audio";
        case OMX_PortDomainVideo: return "Video";
        case OMX_PortDomainImage: return "Image";
//      case OMX_PortDomainOther: return "Other";
        default:                  return def;
    }
}

#endif // AS_STRING_FOR_OMX_COMPONENT_H

#endif // OMX_Component_h

#ifdef OMX_Core_h
/* asString definitions if media/openmax/OMX_Core.h was included */

#ifndef AS_STRING_FOR_OMX_CORE_H
#define AS_STRING_FOR_OMX_CORE_H

inline static const char *asString(OMX_COMMANDTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_CommandStateSet:    return "StateSet";
        case OMX_CommandFlush:       return "Flush";
        case OMX_CommandPortDisable: return "PortDisable";
        case OMX_CommandPortEnable:  return "PortEnable";
//      case OMX_CommandMarkBuffer:  return "MarkBuffer";
        default:                     return def;
    }
}

inline static const char *asString(OMX_STATETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_StateInvalid:          return "Invalid";
        case OMX_StateLoaded:           return "Loaded";
        case OMX_StateIdle:             return "Idle";
        case OMX_StateExecuting:        return "Executing";
//      case OMX_StatePause:            return "Pause";
//      case OMX_StateWaitForResources: return "WaitForResources";
        default:                        return def;
    }
}

inline static const char *asString(OMX_ERRORTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_ErrorNone:                               return "None";
        case OMX_ErrorInsufficientResources:              return "InsufficientResources";
        case OMX_ErrorUndefined:                          return "Undefined";
        case OMX_ErrorInvalidComponentName:               return "InvalidComponentName";
        case OMX_ErrorComponentNotFound:                  return "ComponentNotFound";
        case OMX_ErrorInvalidComponent:                   return "InvalidComponent";       // unused
        case OMX_ErrorBadParameter:                       return "BadParameter";
        case OMX_ErrorNotImplemented:                     return "NotImplemented";
        case OMX_ErrorUnderflow:                          return "Underflow";              // unused
        case OMX_ErrorOverflow:                           return "Overflow";               // unused
        case OMX_ErrorHardware:                           return "Hardware";               // unused
        case OMX_ErrorInvalidState:                       return "InvalidState";
        case OMX_ErrorStreamCorrupt:                      return "StreamCorrupt";
        case OMX_ErrorPortsNotCompatible:                 return "PortsNotCompatible";     // unused
        case OMX_ErrorResourcesLost:                      return "ResourcesLost";
        case OMX_ErrorNoMore:                             return "NoMore";
        case OMX_ErrorVersionMismatch:                    return "VersionMismatch";        // unused
        case OMX_ErrorNotReady:                           return "NotReady";               // unused
        case OMX_ErrorTimeout:                            return "Timeout";                // unused
        case OMX_ErrorSameState:                          return "SameState";              // unused
        case OMX_ErrorResourcesPreempted:                 return "ResourcesPreempted";     // unused
        case OMX_ErrorPortUnresponsiveDuringAllocation:
            return "PortUnresponsiveDuringAllocation";    // unused
        case OMX_ErrorPortUnresponsiveDuringDeallocation:
            return "PortUnresponsiveDuringDeallocation";  // unused
        case OMX_ErrorPortUnresponsiveDuringStop:
            return "PortUnresponsiveDuringStop";          // unused
        case OMX_ErrorIncorrectStateTransition:
            return "IncorrectStateTransition";            // unused
        case OMX_ErrorIncorrectStateOperation:
            return "IncorrectStateOperation";             // unused
        case OMX_ErrorUnsupportedSetting:                 return "UnsupportedSetting";
        case OMX_ErrorUnsupportedIndex:                   return "UnsupportedIndex";
        case OMX_ErrorBadPortIndex:                       return "BadPortIndex";
        case OMX_ErrorPortUnpopulated:                    return "PortUnpopulated";        // unused
        case OMX_ErrorComponentSuspended:                 return "ComponentSuspended";     // unused
        case OMX_ErrorDynamicResourcesUnavailable:
            return "DynamicResourcesUnavailable";         // unused
        case OMX_ErrorMbErrorsInFrame:                    return "MbErrorsInFrame";        // unused
        case OMX_ErrorFormatNotDetected:                  return "FormatNotDetected";      // unused
        case OMX_ErrorContentPipeOpenFailed:              return "ContentPipeOpenFailed";  // unused
        case OMX_ErrorContentPipeCreationFailed:
            return "ContentPipeCreationFailed";           // unused
        case OMX_ErrorSeperateTablesUsed:                 return "SeperateTablesUsed";     // unused
        case OMX_ErrorTunnelingUnsupported:               return "TunnelingUnsupported";   // unused
        default:                                          return def;
    }
}

inline static const char *asString(OMX_EVENTTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_EventCmdComplete:               return "CmdComplete";
        case OMX_EventError:                     return "Error";
//      case OMX_EventMark:                      return "Mark";
        case OMX_EventPortSettingsChanged:       return "PortSettingsChanged";
        case OMX_EventBufferFlag:                return "BufferFlag";
//      case OMX_EventResourcesAcquired:         return "ResourcesAcquired";
//      case OMX_EventComponentResumed:          return "ComponentResumed";
//      case OMX_EventDynamicResourcesAvailable: return "DynamicResourcesAvailable";
//      case OMX_EventPortFormatDetected:        return "PortFormatDetected";
        case OMX_EventOutputRendered:            return "OutputRendered";
        default:                                 return def;
    }
}

#endif // AS_STRING_FOR_OMX_CORE_H

#endif // OMX_Core_h

#ifdef OMX_Image_h
/* asString definitions if media/openmax/OMX_Image.h was included */

#ifndef AS_STRING_FOR_OMX_IMAGE_H
#define AS_STRING_FOR_OMX_IMAGE_H

inline static const char *asString(OMX_IMAGE_CODINGTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_IMAGE_CodingUnused:     return "Unused";
        case OMX_IMAGE_CodingAutoDetect: return "AutoDetect";  // unused
        case OMX_IMAGE_CodingJPEG:       return "JPEG";
        case OMX_IMAGE_CodingJPEG2K:     return "JPEG2K";      // unused
        case OMX_IMAGE_CodingEXIF:       return "EXIF";        // unused
        case OMX_IMAGE_CodingTIFF:       return "TIFF";        // unused
        case OMX_IMAGE_CodingGIF:        return "GIF";         // unused
        case OMX_IMAGE_CodingPNG:        return "PNG";         // unused
        case OMX_IMAGE_CodingLZW:        return "LZW";         // unused
        case OMX_IMAGE_CodingBMP:        return "BMP";         // unused
        default:                         return def;
    }
}

#endif // AS_STRING_FOR_OMX_IMAGE_H

#endif // OMX_Image_h

#ifdef OMX_Index_h
/* asString definitions if media/openmax/OMX_Index.h was included */

#ifndef AS_STRING_FOR_OMX_INDEX_H
#define AS_STRING_FOR_OMX_INDEX_H

inline static const char *asString(OMX_INDEXTYPE i, const char *def = "??") {
    switch (i) {
//      case OMX_IndexParamPriorityMgmt:                    return "ParamPriorityMgmt";
//      case OMX_IndexParamAudioInit:                       return "ParamAudioInit";
//      case OMX_IndexParamImageInit:                       return "ParamImageInit";
//      case OMX_IndexParamVideoInit:                       return "ParamVideoInit";
//      case OMX_IndexParamOtherInit:                       return "ParamOtherInit";
//      case OMX_IndexParamNumAvailableStreams:             return "ParamNumAvailableStreams";
//      case OMX_IndexParamActiveStream:                    return "ParamActiveStream";
//      case OMX_IndexParamSuspensionPolicy:                return "ParamSuspensionPolicy";
//      case OMX_IndexParamComponentSuspended:              return "ParamComponentSuspended";
//      case OMX_IndexConfigCapturing:                      return "ConfigCapturing";
//      case OMX_IndexConfigCaptureMode:                    return "ConfigCaptureMode";
//      case OMX_IndexAutoPauseAfterCapture:                return "AutoPauseAfterCapture";
//      case OMX_IndexParamContentURI:                      return "ParamContentURI";
//      case OMX_IndexParamCustomContentPipe:               return "ParamCustomContentPipe";
//      case OMX_IndexParamDisableResourceConcealment:
//          return "ParamDisableResourceConcealment";
//      case OMX_IndexConfigMetadataItemCount:              return "ConfigMetadataItemCount";
//      case OMX_IndexConfigContainerNodeCount:             return "ConfigContainerNodeCount";
//      case OMX_IndexConfigMetadataItem:                   return "ConfigMetadataItem";
//      case OMX_IndexConfigCounterNodeID:                  return "ConfigCounterNodeID";
//      case OMX_IndexParamMetadataFilterType:              return "ParamMetadataFilterType";
//      case OMX_IndexParamMetadataKeyFilter:               return "ParamMetadataKeyFilter";
//      case OMX_IndexConfigPriorityMgmt:                   return "ConfigPriorityMgmt";
        case OMX_IndexParamStandardComponentRole:           return "ParamStandardComponentRole";
        case OMX_IndexParamPortDefinition:                  return "ParamPortDefinition";
//      case OMX_IndexParamCompBufferSupplier:              return "ParamCompBufferSupplier";
        case OMX_IndexParamAudioPortFormat:                 return "ParamAudioPortFormat";
        case OMX_IndexParamAudioPcm:                        return "ParamAudioPcm";
        case OMX_IndexParamAudioAac:                        return "ParamAudioAac";
//      case OMX_IndexParamAudioRa:                         return "ParamAudioRa";
        case OMX_IndexParamAudioMp3:                        return "ParamAudioMp3";
//      case OMX_IndexParamAudioAdpcm:                      return "ParamAudioAdpcm";
//      case OMX_IndexParamAudioG723:                       return "ParamAudioG723";
//      case OMX_IndexParamAudioG729:                       return "ParamAudioG729";
        case OMX_IndexParamAudioAmr:                        return "ParamAudioAmr";
//      case OMX_IndexParamAudioWma:                        return "ParamAudioWma";
//      case OMX_IndexParamAudioSbc:                        return "ParamAudioSbc";
//      case OMX_IndexParamAudioMidi:                       return "ParamAudioMidi";
//      case OMX_IndexParamAudioGsm_FR:                     return "ParamAudioGsm_FR";
//      case OMX_IndexParamAudioMidiLoadUserSound:          return "ParamAudioMidiLoadUserSound";
//      case OMX_IndexParamAudioG726:                       return "ParamAudioG726";
//      case OMX_IndexParamAudioGsm_EFR:                    return "ParamAudioGsm_EFR";
//      case OMX_IndexParamAudioGsm_HR:                     return "ParamAudioGsm_HR";
//      case OMX_IndexParamAudioPdc_FR:                     return "ParamAudioPdc_FR";
//      case OMX_IndexParamAudioPdc_EFR:                    return "ParamAudioPdc_EFR";
//      case OMX_IndexParamAudioPdc_HR:                     return "ParamAudioPdc_HR";
//      case OMX_IndexParamAudioTdma_FR:                    return "ParamAudioTdma_FR";
//      case OMX_IndexParamAudioTdma_EFR:                   return "ParamAudioTdma_EFR";
//      case OMX_IndexParamAudioQcelp8:                     return "ParamAudioQcelp8";
//      case OMX_IndexParamAudioQcelp13:                    return "ParamAudioQcelp13";
//      case OMX_IndexParamAudioEvrc:                       return "ParamAudioEvrc";
//      case OMX_IndexParamAudioSmv:                        return "ParamAudioSmv";
        case OMX_IndexParamAudioVorbis:                     return "ParamAudioVorbis";
        case OMX_IndexParamAudioFlac:                       return "ParamAudioFlac";
//      case OMX_IndexConfigAudioMidiImmediateEvent:        return "ConfigAudioMidiImmediateEvent";
//      case OMX_IndexConfigAudioMidiControl:               return "ConfigAudioMidiControl";
//      case OMX_IndexConfigAudioMidiSoundBankProgram:
//          return "ConfigAudioMidiSoundBankProgram";
//      case OMX_IndexConfigAudioMidiStatus:                return "ConfigAudioMidiStatus";
//      case OMX_IndexConfigAudioMidiMetaEvent:             return "ConfigAudioMidiMetaEvent";
//      case OMX_IndexConfigAudioMidiMetaEventData:         return "ConfigAudioMidiMetaEventData";
//      case OMX_IndexConfigAudioVolume:                    return "ConfigAudioVolume";
//      case OMX_IndexConfigAudioBalance:                   return "ConfigAudioBalance";
//      case OMX_IndexConfigAudioChannelMute:               return "ConfigAudioChannelMute";
//      case OMX_IndexConfigAudioMute:                      return "ConfigAudioMute";
//      case OMX_IndexConfigAudioLoudness:                  return "ConfigAudioLoudness";
//      case OMX_IndexConfigAudioEchoCancelation:           return "ConfigAudioEchoCancelation";
//      case OMX_IndexConfigAudioNoiseReduction:            return "ConfigAudioNoiseReduction";
//      case OMX_IndexConfigAudioBass:                      return "ConfigAudioBass";
//      case OMX_IndexConfigAudioTreble:                    return "ConfigAudioTreble";
//      case OMX_IndexConfigAudioStereoWidening:            return "ConfigAudioStereoWidening";
//      case OMX_IndexConfigAudioChorus:                    return "ConfigAudioChorus";
//      case OMX_IndexConfigAudioEqualizer:                 return "ConfigAudioEqualizer";
//      case OMX_IndexConfigAudioReverberation:             return "ConfigAudioReverberation";
//      case OMX_IndexConfigAudioChannelVolume:             return "ConfigAudioChannelVolume";
//      case OMX_IndexParamImagePortFormat:                 return "ParamImagePortFormat";
//      case OMX_IndexParamFlashControl:                    return "ParamFlashControl";
//      case OMX_IndexConfigFocusControl:                   return "ConfigFocusControl";
//      case OMX_IndexParamQFactor:                         return "ParamQFactor";
//      case OMX_IndexParamQuantizationTable:               return "ParamQuantizationTable";
//      case OMX_IndexParamHuffmanTable:                    return "ParamHuffmanTable";
//      case OMX_IndexConfigFlashControl:                   return "ConfigFlashControl";
        case OMX_IndexParamVideoPortFormat:                 return "ParamVideoPortFormat";
//      case OMX_IndexParamVideoQuantization:               return "ParamVideoQuantization";
//      case OMX_IndexParamVideoFastUpdate:                 return "ParamVideoFastUpdate";
        case OMX_IndexParamVideoBitrate:                    return "ParamVideoBitrate";
//      case OMX_IndexParamVideoMotionVector:               return "ParamVideoMotionVector";
        case OMX_IndexParamVideoIntraRefresh:               return "ParamVideoIntraRefresh";
        case OMX_IndexParamVideoErrorCorrection:            return "ParamVideoErrorCorrection";
//      case OMX_IndexParamVideoVBSMC:                      return "ParamVideoVBSMC";
//      case OMX_IndexParamVideoMpeg2:                      return "ParamVideoMpeg2";
        case OMX_IndexParamVideoMpeg4:                      return "ParamVideoMpeg4";
//      case OMX_IndexParamVideoWmv:                        return "ParamVideoWmv";
//      case OMX_IndexParamVideoRv:                         return "ParamVideoRv";
        case OMX_IndexParamVideoAvc:                        return "ParamVideoAvc";
        case OMX_IndexParamVideoH263:                       return "ParamVideoH263";
        case OMX_IndexParamVideoProfileLevelQuerySupported:
            return "ParamVideoProfileLevelQuerySupported";
        case OMX_IndexParamVideoProfileLevelCurrent:        return "ParamVideoProfileLevelCurrent";
        case OMX_IndexConfigVideoBitrate:                   return "ConfigVideoBitrate";
//      case OMX_IndexConfigVideoFramerate:                 return "ConfigVideoFramerate";
        case OMX_IndexConfigVideoIntraVOPRefresh:           return "ConfigVideoIntraVOPRefresh";
//      case OMX_IndexConfigVideoIntraMBRefresh:            return "ConfigVideoIntraMBRefresh";
//      case OMX_IndexConfigVideoMBErrorReporting:          return "ConfigVideoMBErrorReporting";
//      case OMX_IndexParamVideoMacroblocksPerFrame:        return "ParamVideoMacroblocksPerFrame";
//      case OMX_IndexConfigVideoMacroBlockErrorMap:        return "ConfigVideoMacroBlockErrorMap";
//      case OMX_IndexParamVideoSliceFMO:                   return "ParamVideoSliceFMO";
//      case OMX_IndexConfigVideoAVCIntraPeriod:            return "ConfigVideoAVCIntraPeriod";
//      case OMX_IndexConfigVideoNalSize:                   return "ConfigVideoNalSize";
//      case OMX_IndexParamCommonDeblocking:                return "ParamCommonDeblocking";
//      case OMX_IndexParamCommonSensorMode:                return "ParamCommonSensorMode";
//      case OMX_IndexParamCommonInterleave:                return "ParamCommonInterleave";
//      case OMX_IndexConfigCommonColorFormatConversion:
//          return "ConfigCommonColorFormatConversion";
        case OMX_IndexConfigCommonScale:                    return "ConfigCommonScale";
//      case OMX_IndexConfigCommonImageFilter:              return "ConfigCommonImageFilter";
//      case OMX_IndexConfigCommonColorEnhancement:         return "ConfigCommonColorEnhancement";
//      case OMX_IndexConfigCommonColorKey:                 return "ConfigCommonColorKey";
//      case OMX_IndexConfigCommonColorBlend:               return "ConfigCommonColorBlend";
//      case OMX_IndexConfigCommonFrameStabilisation:       return "ConfigCommonFrameStabilisation";
//      case OMX_IndexConfigCommonRotate:                   return "ConfigCommonRotate";
//      case OMX_IndexConfigCommonMirror:                   return "ConfigCommonMirror";
//      case OMX_IndexConfigCommonOutputPosition:           return "ConfigCommonOutputPosition";
        case OMX_IndexConfigCommonInputCrop:                return "ConfigCommonInputCrop";
        case OMX_IndexConfigCommonOutputCrop:               return "ConfigCommonOutputCrop";
//      case OMX_IndexConfigCommonDigitalZoom:              return "ConfigCommonDigitalZoom";
//      case OMX_IndexConfigCommonOpticalZoom:              return "ConfigCommonOpticalZoom";
//      case OMX_IndexConfigCommonWhiteBalance:             return "ConfigCommonWhiteBalance";
//      case OMX_IndexConfigCommonExposure:                 return "ConfigCommonExposure";
//      case OMX_IndexConfigCommonContrast:                 return "ConfigCommonContrast";
//      case OMX_IndexConfigCommonBrightness:               return "ConfigCommonBrightness";
//      case OMX_IndexConfigCommonBacklight:                return "ConfigCommonBacklight";
//      case OMX_IndexConfigCommonGamma:                    return "ConfigCommonGamma";
//      case OMX_IndexConfigCommonSaturation:               return "ConfigCommonSaturation";
//      case OMX_IndexConfigCommonLightness:                return "ConfigCommonLightness";
//      case OMX_IndexConfigCommonExclusionRect:            return "ConfigCommonExclusionRect";
//      case OMX_IndexConfigCommonDithering:                return "ConfigCommonDithering";
//      case OMX_IndexConfigCommonPlaneBlend:               return "ConfigCommonPlaneBlend";
//      case OMX_IndexConfigCommonExposureValue:            return "ConfigCommonExposureValue";
//      case OMX_IndexConfigCommonOutputSize:               return "ConfigCommonOutputSize";
//      case OMX_IndexParamCommonExtraQuantData:            return "ParamCommonExtraQuantData";
//      case OMX_IndexConfigCommonFocusRegion:              return "ConfigCommonFocusRegion";
//      case OMX_IndexConfigCommonFocusStatus:              return "ConfigCommonFocusStatus";
//      case OMX_IndexConfigCommonTransitionEffect:         return "ConfigCommonTransitionEffect";
//      case OMX_IndexParamOtherPortFormat:                 return "ParamOtherPortFormat";
//      case OMX_IndexConfigOtherPower:                     return "ConfigOtherPower";
//      case OMX_IndexConfigOtherStats:                     return "ConfigOtherStats";
//      case OMX_IndexConfigTimeScale:                      return "ConfigTimeScale";
//      case OMX_IndexConfigTimeClockState:                 return "ConfigTimeClockState";
//      case OMX_IndexConfigTimeActiveRefClock:             return "ConfigTimeActiveRefClock";
//      case OMX_IndexConfigTimeCurrentMediaTime:           return "ConfigTimeCurrentMediaTime";
//      case OMX_IndexConfigTimeCurrentWallTime:            return "ConfigTimeCurrentWallTime";
//      case OMX_IndexConfigTimeCurrentAudioReference:
//          return "ConfigTimeCurrentAudioReference";
//      case OMX_IndexConfigTimeCurrentVideoReference:
//          return "ConfigTimeCurrentVideoReference";
//      case OMX_IndexConfigTimeMediaTimeRequest:           return "ConfigTimeMediaTimeRequest";
//      case OMX_IndexConfigTimeClientStartTime:            return "ConfigTimeClientStartTime";
//      case OMX_IndexConfigTimePosition:                   return "ConfigTimePosition";
//      case OMX_IndexConfigTimeSeekMode:                   return "ConfigTimeSeekMode";
        default:                                            return def;
    }
}

#endif // AS_STRING_FOR_OMX_INDEX_H

#endif // OMX_Index_h

#ifdef OMX_IndexExt_h
/* asString definitions if media/openmax/OMX_IndexExt.h was included */

#ifndef AS_STRING_FOR_OMX_INDEXEXT_H
#define AS_STRING_FOR_OMX_INDEXEXT_H

inline static const char *asString(OMX_INDEXEXTTYPE i, const char *def = "??") {
    switch (i) {
//      case OMX_IndexConfigCallbackRequest:            return "ConfigCallbackRequest";
//      case OMX_IndexConfigCommitMode:                 return "ConfigCommitMode";
//      case OMX_IndexConfigCommit:                     return "ConfigCommit";
        case OMX_IndexParamAudioAndroidAc3:             return "ParamAudioAndroidAc3";
        case OMX_IndexParamAudioAndroidOpus:            return "ParamAudioAndroidOpus";
        case OMX_IndexParamAudioAndroidAacPresentation: return "ParamAudioAndroidAacPresentation";
//      case OMX_IndexParamNalStreamFormatSupported:    return "ParamNalStreamFormatSupported";
//      case OMX_IndexParamNalStreamFormat:             return "ParamNalStreamFormat";
//      case OMX_IndexParamNalStreamFormatSelect:       return "ParamNalStreamFormatSelect";
        case OMX_IndexParamVideoVp8:                    return "ParamVideoVp8";
//      case OMX_IndexConfigVideoVp8ReferenceFrame:     return "ConfigVideoVp8ReferenceFrame";
//      case OMX_IndexConfigVideoVp8ReferenceFrameType: return "ConfigVideoVp8ReferenceFrameType";
        case OMX_IndexParamVideoAndroidVp8Encoder:      return "ParamVideoAndroidVp8Encoder";
        case OMX_IndexParamVideoHevc:                   return "ParamVideoHevc";
//      case OMX_IndexParamSliceSegments:               return "ParamSliceSegments";
        case OMX_IndexConfigAutoFramerateConversion:    return "ConfigAutoFramerateConversion";
        case OMX_IndexConfigPriority:                   return "ConfigPriority";
        case OMX_IndexConfigOperatingRate:              return "ConfigOperatingRate";
        case OMX_IndexParamConsumerUsageBits:           return "ParamConsumerUsageBits";
        default:                                        return asString((OMX_INDEXTYPE)i, def);
    }
}

#endif // AS_STRING_FOR_OMX_INDEXEXT_H

#endif // OMX_IndexExt_h

#ifdef OMX_IVCommon_h
/* asString definitions if media/openmax/OMX_IVCommon.h was included */

#ifndef AS_STRING_FOR_OMX_IVCOMMON_H
#define AS_STRING_FOR_OMX_IVCOMMON_H

inline static const char *asString(OMX_COLOR_FORMATTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_COLOR_FormatUnused:
            return "COLOR_FormatUnused";
        case OMX_COLOR_FormatMonochrome:
            return "COLOR_FormatMonochrome";
        case OMX_COLOR_Format8bitRGB332:
            return "COLOR_Format8bitRGB332";
        case OMX_COLOR_Format12bitRGB444:
            return "COLOR_Format12bitRGB444";
        case OMX_COLOR_Format16bitARGB4444:
            return "COLOR_Format16bitARGB4444";
        case OMX_COLOR_Format16bitARGB1555:
            return "COLOR_Format16bitARGB1555";
        case OMX_COLOR_Format16bitRGB565:
            return "COLOR_Format16bitRGB565";
        case OMX_COLOR_Format16bitBGR565:
            return "COLOR_Format16bitBGR565";
        case OMX_COLOR_Format18bitRGB666:
            return "COLOR_Format18bitRGB666";
        case OMX_COLOR_Format18bitARGB1665:
            return "COLOR_Format18bitARGB1665";
        case OMX_COLOR_Format19bitARGB1666:
            return "COLOR_Format19bitARGB1666";
        case OMX_COLOR_Format24bitRGB888:
            return "COLOR_Format24bitRGB888";
        case OMX_COLOR_Format24bitBGR888:
            return "COLOR_Format24bitBGR888";
        case OMX_COLOR_Format24bitARGB1887:
            return "COLOR_Format24bitARGB1887";
        case OMX_COLOR_Format25bitARGB1888:
            return "COLOR_Format25bitARGB1888";
        case OMX_COLOR_Format32bitBGRA8888:
            return "COLOR_Format32bitBGRA8888";
        case OMX_COLOR_Format32bitARGB8888:
            return "COLOR_Format32bitARGB8888";
        case OMX_COLOR_FormatYUV411Planar:
            return "COLOR_FormatYUV411Planar";
        case OMX_COLOR_FormatYUV411PackedPlanar:
            return "COLOR_FormatYUV411PackedPlanar";
        case OMX_COLOR_FormatYUV420Planar:
            return "COLOR_FormatYUV420Planar";
        case OMX_COLOR_FormatYUV420PackedPlanar:
            return "COLOR_FormatYUV420PackedPlanar";
        case OMX_COLOR_FormatYUV420SemiPlanar:
            return "COLOR_FormatYUV420SemiPlanar";
        case OMX_COLOR_FormatYUV422Planar:
            return "COLOR_FormatYUV422Planar";
        case OMX_COLOR_FormatYUV422PackedPlanar:
            return "COLOR_FormatYUV422PackedPlanar";
        case OMX_COLOR_FormatYUV422SemiPlanar:
            return "COLOR_FormatYUV422SemiPlanar";
        case OMX_COLOR_FormatYCbYCr:
            return "COLOR_FormatYCbYCr";
        case OMX_COLOR_FormatYCrYCb:
            return "COLOR_FormatYCrYCb";
        case OMX_COLOR_FormatCbYCrY:
            return "COLOR_FormatCbYCrY";
        case OMX_COLOR_FormatCrYCbY:
            return "COLOR_FormatCrYCbY";
        case OMX_COLOR_FormatYUV444Interleaved:
            return "COLOR_FormatYUV444Interleaved";
        case OMX_COLOR_FormatRawBayer8bit:
            return "COLOR_FormatRawBayer8bit";
        case OMX_COLOR_FormatRawBayer10bit:
            return "COLOR_FormatRawBayer10bit";
        case OMX_COLOR_FormatRawBayer8bitcompressed:
            return "COLOR_FormatRawBayer8bitcompressed";
        case OMX_COLOR_FormatL2:
            return "COLOR_FormatL2";
        case OMX_COLOR_FormatL4:
            return "COLOR_FormatL4";
        case OMX_COLOR_FormatL8:
            return "COLOR_FormatL8";
        case OMX_COLOR_FormatL16:
            return "COLOR_FormatL16";
        case OMX_COLOR_FormatL24:
            return "COLOR_FormatL24";
        case OMX_COLOR_FormatL32:
            return "COLOR_FormatL32";
        case OMX_COLOR_FormatYUV420PackedSemiPlanar:
            return "COLOR_FormatYUV420PackedSemiPlanar";
        case OMX_COLOR_FormatYUV422PackedSemiPlanar:
            return "COLOR_FormatYUV422PackedSemiPlanar";
        case OMX_COLOR_Format18BitBGR666:
            return "COLOR_Format18BitBGR666";
        case OMX_COLOR_Format24BitARGB6666:
            return "COLOR_Format24BitARGB6666";
        case OMX_COLOR_Format24BitABGR6666:
            return "COLOR_Format24BitABGR6666";
        case OMX_COLOR_FormatAndroidOpaque:
            return "COLOR_FormatAndroidOpaque";
        case OMX_COLOR_FormatYUV420Flexible:
            return "COLOR_FormatYUV420Flexible";
        case OMX_TI_COLOR_FormatYUV420PackedSemiPlanar:
            return "TI_COLOR_FormatYUV420PackedSemiPlanar";
        case OMX_QCOM_COLOR_FormatYVU420SemiPlanar:
            return "QCOM_COLOR_FormatYVU420SemiPlanar";
//      case OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka:
//          return "QCOM_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka";
//      case OMX_SEC_COLOR_FormatNV12Tiled:
//          return "SEC_COLOR_FormatNV12Tiled";
//      case OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar32m:
//          return "QCOM_COLOR_FormatYUV420PackedSemiPlanar32m";
        default:
            return def;
    }
}

#endif // AS_STRING_FOR_OMX_IVCOMMON_H

#endif // OMX_IVCommon_h

#ifdef OMX_Types_h
/* asString definitions if media/openmax/OMX_Types.h was included */

#ifndef AS_STRING_FOR_OMX_TYPES_H
#define AS_STRING_FOR_OMX_TYPES_H

inline static const char *asString(OMX_BOOL i, const char *def = "??") {
    switch (i) {
        case OMX_FALSE: return "FALSE";
        case OMX_TRUE:  return "TRUE";
        default:        return def;
    }
}

inline static const char *asString(OMX_DIRTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_DirInput:  return "Input";
        case OMX_DirOutput: return "Output";
        default:            return def;
    }
}

inline static const char *asString(OMX_ENDIANTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_EndianBig:    return "Big";
//      case OMX_EndianLittle: return "Little";
        default:               return def;
    }
}

inline static const char *asString(OMX_NUMERICALDATATYPE i, const char *def = "??") {
    switch (i) {
        case OMX_NumericalDataSigned:   return "Signed";
//      case OMX_NumericalDataUnsigned: return "Unsigned";
        default:                        return def;
    }
}

#endif // AS_STRING_FOR_OMX_TYPES_H

#endif // OMX_Types_h

#ifdef OMX_Video_h
/* asString definitions if media/openmax/OMX_Video.h was included */

#ifndef AS_STRING_FOR_OMX_VIDEO_H
#define AS_STRING_FOR_OMX_VIDEO_H

inline static const char *asString(OMX_VIDEO_CODINGTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_CodingUnused:     return "Unused";
        case OMX_VIDEO_CodingAutoDetect: return "AutoDetect";  // unused
        case OMX_VIDEO_CodingMPEG2:      return "MPEG2";
        case OMX_VIDEO_CodingH263:       return "H263";
        case OMX_VIDEO_CodingMPEG4:      return "MPEG4";
        case OMX_VIDEO_CodingWMV:        return "WMV";         // unused
        case OMX_VIDEO_CodingRV:         return "RV";          // unused
        case OMX_VIDEO_CodingAVC:        return "AVC";
        case OMX_VIDEO_CodingMJPEG:      return "MJPEG";       // unused
        case OMX_VIDEO_CodingVP8:        return "VP8";
        case OMX_VIDEO_CodingVP9:        return "VP9";
        case OMX_VIDEO_CodingHEVC:       return "HEVC";
        default:                         return def;
    }
}

inline static const char *asString(OMX_VIDEO_CONTROLRATETYPE i, const char *def = "??") {
    switch (i) {
//      case OMX_Video_ControlRateDisable:            return "Disable";
        case OMX_Video_ControlRateVariable:           return "Variable";
        case OMX_Video_ControlRateConstant:           return "Constant";
//      case OMX_Video_ControlRateVariableSkipFrames: return "VariableSkipFrames";
//      case OMX_Video_ControlRateConstantSkipFrames: return "ConstantSkipFrames";
        default:                                      return def;
    }
}

inline static const char *asString(OMX_VIDEO_INTRAREFRESHTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_IntraRefreshCyclic:   return "Cyclic";
        case OMX_VIDEO_IntraRefreshAdaptive: return "Adaptive";
        case OMX_VIDEO_IntraRefreshBoth:     return "Both";
        default:                             return def;
    }
}

inline static const char *asString(OMX_VIDEO_H263PROFILETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_H263ProfileBaseline:           return "Baseline";
        case OMX_VIDEO_H263ProfileH320Coding:         return "H320Coding";
        case OMX_VIDEO_H263ProfileBackwardCompatible: return "BackwardCompatible";
        case OMX_VIDEO_H263ProfileISWV2:              return "ISWV2";
        case OMX_VIDEO_H263ProfileISWV3:              return "ISWV3";
        case OMX_VIDEO_H263ProfileHighCompression:    return "HighCompression";
        case OMX_VIDEO_H263ProfileInternet:           return "Internet";
        case OMX_VIDEO_H263ProfileInterlace:          return "Interlace";
        case OMX_VIDEO_H263ProfileHighLatency:        return "HighLatency";
        default:                                      return def;
    }
}

inline static const char *asString(OMX_VIDEO_H263LEVELTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_H263Level10: return "Level10";
        case OMX_VIDEO_H263Level20: return "Level20";
        case OMX_VIDEO_H263Level30: return "Level30";
        case OMX_VIDEO_H263Level40: return "Level40";
        case OMX_VIDEO_H263Level45: return "Level45";
        case OMX_VIDEO_H263Level50: return "Level50";
        case OMX_VIDEO_H263Level60: return "Level60";
        case OMX_VIDEO_H263Level70: return "Level70";
        default:                    return def;
    }
}

inline static const char *asString(OMX_VIDEO_PICTURETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_PictureTypeI:  return "I";
        case OMX_VIDEO_PictureTypeP:  return "P";
        case OMX_VIDEO_PictureTypeB:  return "B";
//      case OMX_VIDEO_PictureTypeSI: return "SI";
//      case OMX_VIDEO_PictureTypeSP: return "SP";
//      case OMX_VIDEO_PictureTypeEI: return "EI";
//      case OMX_VIDEO_PictureTypeEP: return "EP";
//      case OMX_VIDEO_PictureTypeS:  return "S";
        default:                      return def;
    }
}

inline static const char *asString(OMX_VIDEO_MPEG4PROFILETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_MPEG4ProfileSimple:           return "Simple";
        case OMX_VIDEO_MPEG4ProfileSimpleScalable:   return "SimpleScalable";
        case OMX_VIDEO_MPEG4ProfileCore:             return "Core";
        case OMX_VIDEO_MPEG4ProfileMain:             return "Main";
        case OMX_VIDEO_MPEG4ProfileNbit:             return "Nbit";
        case OMX_VIDEO_MPEG4ProfileScalableTexture:  return "ScalableTexture";
        case OMX_VIDEO_MPEG4ProfileSimpleFace:       return "SimpleFace";
        case OMX_VIDEO_MPEG4ProfileSimpleFBA:        return "SimpleFBA";
        case OMX_VIDEO_MPEG4ProfileBasicAnimated:    return "BasicAnimated";
        case OMX_VIDEO_MPEG4ProfileHybrid:           return "Hybrid";
        case OMX_VIDEO_MPEG4ProfileAdvancedRealTime: return "AdvancedRealTime";
        case OMX_VIDEO_MPEG4ProfileCoreScalable:     return "CoreScalable";
        case OMX_VIDEO_MPEG4ProfileAdvancedCoding:   return "AdvancedCoding";
        case OMX_VIDEO_MPEG4ProfileAdvancedCore:     return "AdvancedCore";
        case OMX_VIDEO_MPEG4ProfileAdvancedScalable: return "AdvancedScalable";
        case OMX_VIDEO_MPEG4ProfileAdvancedSimple:   return "AdvancedSimple";
        default:                                     return def;
    }
}

inline static const char *asString(OMX_VIDEO_MPEG4LEVELTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_MPEG4Level0:  return "Level0";
        case OMX_VIDEO_MPEG4Level0b: return "Level0b";
        case OMX_VIDEO_MPEG4Level1:  return "Level1";
        case OMX_VIDEO_MPEG4Level2:  return "Level2";
        case OMX_VIDEO_MPEG4Level3:  return "Level3";
        case OMX_VIDEO_MPEG4Level4:  return "Level4";
        case OMX_VIDEO_MPEG4Level4a: return "Level4a";
        case OMX_VIDEO_MPEG4Level5:  return "Level5";
        default:                     return def;
    }
}

inline static const char *asString(OMX_VIDEO_AVCPROFILETYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_AVCProfileBaseline: return "Baseline";
        case OMX_VIDEO_AVCProfileMain:     return "Main";
        case OMX_VIDEO_AVCProfileExtended: return "Extended";
        case OMX_VIDEO_AVCProfileHigh:     return "High";
        case OMX_VIDEO_AVCProfileHigh10:   return "High10";
        case OMX_VIDEO_AVCProfileHigh422:  return "High422";
        case OMX_VIDEO_AVCProfileHigh444:  return "High444";
        default:                           return def;
    }
}

inline static const char *asString(OMX_VIDEO_AVCLEVELTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_AVCLevel1:  return "Level1";
        case OMX_VIDEO_AVCLevel1b: return "Level1b";
        case OMX_VIDEO_AVCLevel11: return "Level11";
        case OMX_VIDEO_AVCLevel12: return "Level12";
        case OMX_VIDEO_AVCLevel13: return "Level13";
        case OMX_VIDEO_AVCLevel2:  return "Level2";
        case OMX_VIDEO_AVCLevel21: return "Level21";
        case OMX_VIDEO_AVCLevel22: return "Level22";
        case OMX_VIDEO_AVCLevel3:  return "Level3";
        case OMX_VIDEO_AVCLevel31: return "Level31";
        case OMX_VIDEO_AVCLevel32: return "Level32";
        case OMX_VIDEO_AVCLevel4:  return "Level4";
        case OMX_VIDEO_AVCLevel41: return "Level41";
        case OMX_VIDEO_AVCLevel42: return "Level42";
        case OMX_VIDEO_AVCLevel5:  return "Level5";
        case OMX_VIDEO_AVCLevel51: return "Level51";
        case OMX_VIDEO_AVCLevel52: return "Level52";
        default:                   return def;
    }
}

inline static const char *asString(OMX_VIDEO_AVCLOOPFILTERTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_AVCLoopFilterEnable:               return "Enable";
//      case OMX_VIDEO_AVCLoopFilterDisable:              return "Disable";
//      case OMX_VIDEO_AVCLoopFilterDisableSliceBoundary: return "DisableSliceBoundary";
        default:                                          return def;
    }
}

#endif // AS_STRING_FOR_OMX_VIDEO_H

#endif // OMX_Video_h

#ifdef OMX_VideoExt_h
/* asString definitions if media/openmax/OMX_VideoExt.h was included */

#ifndef AS_STRING_FOR_OMX_VIDEOEXT_H
#define AS_STRING_FOR_OMX_VIDEOEXT_H

inline static const char *asString(OMX_VIDEO_VP8PROFILETYPE i, const char *def = "!!") {
    switch (i) {
        case OMX_VIDEO_VP8ProfileMain:    return "Main";
        case OMX_VIDEO_VP8ProfileUnknown: return "Unknown";  // unused
        default:                          return def;
    }
}

inline static const char *asString(OMX_VIDEO_VP8LEVELTYPE i, const char *def = "!!") {
    switch (i) {
        case OMX_VIDEO_VP8Level_Version0: return "_Version0";
        case OMX_VIDEO_VP8Level_Version1: return "_Version1";
        case OMX_VIDEO_VP8Level_Version2: return "_Version2";
        case OMX_VIDEO_VP8Level_Version3: return "_Version3";
        case OMX_VIDEO_VP8LevelUnknown:   return "Unknown";    // unused
        default:                          return def;
    }
}

inline static const char *asString(
        OMX_VIDEO_ANDROID_VPXTEMPORALLAYERPATTERNTYPE i, const char *def = "??") {
    switch (i) {
        case OMX_VIDEO_VPXTemporalLayerPatternNone:   return "VPXTemporalLayerPatternNone";
        case OMX_VIDEO_VPXTemporalLayerPatternWebRTC: return "VPXTemporalLayerPatternWebRTC";
        default:                                      return def;
    }
}

inline static const char *asString(OMX_VIDEO_HEVCPROFILETYPE i, const char *def = "!!") {
    switch (i) {
        case OMX_VIDEO_HEVCProfileUnknown: return "Unknown";  // unused
        case OMX_VIDEO_HEVCProfileMain:    return "Main";
        case OMX_VIDEO_HEVCProfileMain10:  return "Main10";
        default:                           return def;
    }
}

inline static const char *asString(OMX_VIDEO_HEVCLEVELTYPE i, const char *def = "!!") {
    switch (i) {
        case OMX_VIDEO_HEVCLevelUnknown:    return "LevelUnknown";     // unused
        case OMX_VIDEO_HEVCMainTierLevel1:  return "MainTierLevel1";
        case OMX_VIDEO_HEVCHighTierLevel1:  return "HighTierLevel1";
        case OMX_VIDEO_HEVCMainTierLevel2:  return "MainTierLevel2";
        case OMX_VIDEO_HEVCHighTierLevel2:  return "HighTierLevel2";
        case OMX_VIDEO_HEVCMainTierLevel21: return "MainTierLevel21";
        case OMX_VIDEO_HEVCHighTierLevel21: return "HighTierLevel21";
        case OMX_VIDEO_HEVCMainTierLevel3:  return "MainTierLevel3";
        case OMX_VIDEO_HEVCHighTierLevel3:  return "HighTierLevel3";
        case OMX_VIDEO_HEVCMainTierLevel31: return "MainTierLevel31";
        case OMX_VIDEO_HEVCHighTierLevel31: return "HighTierLevel31";
        case OMX_VIDEO_HEVCMainTierLevel4:  return "MainTierLevel4";
        case OMX_VIDEO_HEVCHighTierLevel4:  return "HighTierLevel4";
        case OMX_VIDEO_HEVCMainTierLevel41: return "MainTierLevel41";
        case OMX_VIDEO_HEVCHighTierLevel41: return "HighTierLevel41";
        case OMX_VIDEO_HEVCMainTierLevel5:  return "MainTierLevel5";
        case OMX_VIDEO_HEVCHighTierLevel5:  return "HighTierLevel5";
        case OMX_VIDEO_HEVCMainTierLevel51: return "MainTierLevel51";
        case OMX_VIDEO_HEVCHighTierLevel51: return "HighTierLevel51";
        case OMX_VIDEO_HEVCMainTierLevel52: return "MainTierLevel52";
        case OMX_VIDEO_HEVCHighTierLevel52: return "HighTierLevel52";
        case OMX_VIDEO_HEVCMainTierLevel6:  return "MainTierLevel6";
        case OMX_VIDEO_HEVCHighTierLevel6:  return "HighTierLevel6";
        case OMX_VIDEO_HEVCMainTierLevel61: return "MainTierLevel61";
        case OMX_VIDEO_HEVCHighTierLevel61: return "HighTierLevel61";
        case OMX_VIDEO_HEVCMainTierLevel62: return "MainTierLevel62";
        case OMX_VIDEO_HEVCHighTierLevel62: return "HighTierLevel62";
        default:                            return def;
    }
}

#endif // AS_STRING_FOR_OMX_VIDEOEXT_H

#endif // OMX_VideoExt_h
