/*
 * Copyright (C) 2014 The Android Open Source Project
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

/* This file contains shared utility functions to handle the tinyalsa
 * implementation for Android internal audio, generally in the hardware layer.
 * Some routines may log a fatal error on failure, as noted.
 */

#ifndef ANDROID_AUDIO_ALSAOPS_H
#define ANDROID_AUDIO_ALSAOPS_H

#include <log/log.h>

#include <system/audio.h>
#include <tinyalsa/asoundlib.h>

__BEGIN_DECLS

/* Converts audio_format to pcm_format.
 * Parameters:
 *  format  the audio_format_t to convert
 *
 * Logs a fatal error if format is not a valid convertible audio_format_t.
 */
static inline enum pcm_format pcm_format_from_audio_format(audio_format_t format)
{
    switch (format) {
#if HAVE_BIG_ENDIAN
    case AUDIO_FORMAT_PCM_16_BIT:
        return PCM_FORMAT_S16_BE;
    case AUDIO_FORMAT_PCM_24_BIT_PACKED:
        return PCM_FORMAT_S24_3BE;
    case AUDIO_FORMAT_PCM_32_BIT:
        return PCM_FORMAT_S32_BE;
    case AUDIO_FORMAT_PCM_8_24_BIT:
        return PCM_FORMAT_S24_BE;
#else
    case AUDIO_FORMAT_PCM_16_BIT:
        return PCM_FORMAT_S16_LE;
    case AUDIO_FORMAT_PCM_24_BIT_PACKED:
        return PCM_FORMAT_S24_3LE;
    case AUDIO_FORMAT_PCM_32_BIT:
        return PCM_FORMAT_S32_LE;
    case AUDIO_FORMAT_PCM_8_24_BIT:
        return PCM_FORMAT_S24_LE;
#endif
    case AUDIO_FORMAT_PCM_FLOAT:  /* there is no equivalent for float */
    default:
        LOG_ALWAYS_FATAL("pcm_format_from_audio_format: invalid audio format %#x", format);
        return 0;
    }
}

/* Converts pcm_format to audio_format.
 * Parameters:
 *  format  the pcm_format to convert
 *
 * Logs a fatal error if format is not a valid convertible pcm_format.
 */
static inline audio_format_t audio_format_from_pcm_format(enum pcm_format format)
{
    switch (format) {
#if HAVE_BIG_ENDIAN
    case PCM_FORMAT_S16_BE:
        return AUDIO_FORMAT_PCM_16_BIT;
    case PCM_FORMAT_S24_3BE:
        return AUDIO_FORMAT_PCM_24_BIT_PACKED;
    case PCM_FORMAT_S24_BE:
        return AUDIO_FORMAT_PCM_8_24_BIT;
    case PCM_FORMAT_S32_BE:
        return AUDIO_FORMAT_PCM_32_BIT;
#else
    case PCM_FORMAT_S16_LE:
        return AUDIO_FORMAT_PCM_16_BIT;
    case PCM_FORMAT_S24_3LE:
        return AUDIO_FORMAT_PCM_24_BIT_PACKED;
    case PCM_FORMAT_S24_LE:
        return AUDIO_FORMAT_PCM_8_24_BIT;
    case PCM_FORMAT_S32_LE:
        return AUDIO_FORMAT_PCM_32_BIT;
#endif
    default:
        LOG_ALWAYS_FATAL("audio_format_from_pcm_format: invalid pcm format %#x", format);
        return 0;
    }
}

__END_DECLS

#endif /* ANDROID_AUDIO_ALSAOPS_H */
