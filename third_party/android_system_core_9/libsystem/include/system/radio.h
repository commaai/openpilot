/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef ANDROID_RADIO_H
#define ANDROID_RADIO_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/cdefs.h>
#include <sys/types.h>


#define RADIO_NUM_BANDS_MAX     16
#define RADIO_NUM_SPACINGS_MAX  16
#define RADIO_STRING_LEN_MAX    128

/*
 * Radio hardware module class. A given radio hardware module HAL is of one class
 * only. The platform can not have more than one hardware module of each class.
 * Current version of the framework only supports RADIO_CLASS_AM_FM.
 */
typedef enum {
    RADIO_CLASS_AM_FM = 0,  /* FM (including HD radio) and AM */
    RADIO_CLASS_SAT   = 1,  /* Satellite Radio */
    RADIO_CLASS_DT    = 2,  /* Digital Radio (DAB) */
} radio_class_t;

/* value for field "type" of radio band described in struct radio_hal_band_config */
typedef enum {
    RADIO_BAND_AM     = 0,  /* Amplitude Modulation band: LW, MW, SW */
    RADIO_BAND_FM     = 1,  /* Frequency Modulation band: FM */
    RADIO_BAND_FM_HD  = 2,  /* FM HD Radio / DRM (IBOC) */
    RADIO_BAND_AM_HD  = 3,  /* AM HD Radio / DRM (IBOC) */
} radio_band_t;

/* RDS variant implemented. A struct radio_hal_fm_band_config can list none or several. */
enum {
    RADIO_RDS_NONE   = 0x0,
    RADIO_RDS_WORLD  = 0x01,
    RADIO_RDS_US     = 0x02,
};
typedef unsigned int radio_rds_t;

/* FM deemphasis variant implemented. A struct radio_hal_fm_band_config can list one or more. */
enum {
    RADIO_DEEMPHASIS_50   = 0x1,
    RADIO_DEEMPHASIS_75   = 0x2,
};
typedef unsigned int radio_deemphasis_t;

/* Region a particular radio band configuration corresponds to. Not used at the HAL.
 * Derived by the framework when converting the band descriptors retrieved from the HAL to
 * individual band descriptors for each supported region. */
typedef enum {
    RADIO_REGION_NONE  = -1,
    RADIO_REGION_ITU_1 = 0,
    RADIO_REGION_ITU_2 = 1,
    RADIO_REGION_OIRT  = 2,
    RADIO_REGION_JAPAN = 3,
    RADIO_REGION_KOREA = 4,
} radio_region_t;

/* scanning direction for scan() and step() tuner APIs */
typedef enum {
    RADIO_DIRECTION_UP,
    RADIO_DIRECTION_DOWN
} radio_direction_t;

/* unique handle allocated to a radio module */
typedef uint32_t radio_handle_t;

/* Opaque meta data structure used by radio meta data API (see system/radio_metadata.h) */
typedef struct radio_metadata radio_metadata_t;


/* Additional attributes for an FM band configuration */
typedef struct radio_hal_fm_band_config {
    radio_deemphasis_t  deemphasis; /* deemphasis variant */
    bool                stereo;     /* stereo supported */
    radio_rds_t         rds;        /* RDS variants supported */
    bool                ta;         /* Traffic Announcement supported */
    bool                af;         /* Alternate Frequency supported */
    bool                ea;         /* Emergency announcements supported */
} radio_hal_fm_band_config_t;

/* Additional attributes for an AM band configuration */
typedef struct radio_hal_am_band_config {
    bool                stereo;     /* stereo supported */
} radio_hal_am_band_config_t;

/* Radio band configuration. Describes a given band supported by the radio module.
 * The HAL can expose only one band per type with the the maximum range supported and all options.
 * THe framework will derive the actual regions were this module can operate and expose separate
 * band configurations for applications to chose from. */
typedef struct radio_hal_band_config {
    radio_band_t type;
    bool         antenna_connected;
    uint32_t     lower_limit;
    uint32_t     upper_limit;
    uint32_t     num_spacings;
    uint32_t     spacings[RADIO_NUM_SPACINGS_MAX];
    union {
        radio_hal_fm_band_config_t fm;
        radio_hal_am_band_config_t am;
    };
} radio_hal_band_config_t;

/* Used internally by the framework to represent a band for s specific region */
typedef struct radio_band_config {
    radio_region_t  region;
    radio_hal_band_config_t band;
} radio_band_config_t;


/* Exposes properties of a given hardware radio module.
 * NOTE: current framework implementation supports only one audio source (num_audio_sources = 1).
 * The source corresponds to AUDIO_DEVICE_IN_FM_TUNER.
 * If more than one tuner is supported (num_tuners > 1), only one can be connected to the audio
 * source. */
typedef struct radio_hal_properties {
    radio_class_t   class_id;   /* Class of this module. E.g RADIO_CLASS_AM_FM */
    char            implementor[RADIO_STRING_LEN_MAX];  /* implementor name */
    char            product[RADIO_STRING_LEN_MAX];  /* product name */
    char            version[RADIO_STRING_LEN_MAX];  /* product version */
    char            serial[RADIO_STRING_LEN_MAX];  /* serial number (for subscription services) */
    uint32_t        num_tuners;     /* number of tuners controllable independently */
    uint32_t        num_audio_sources; /* number of audio sources driven simultaneously */
    bool            supports_capture; /* the hardware supports capture of audio source audio HAL */
    uint32_t        num_bands;      /* number of band descriptors */
    radio_hal_band_config_t bands[RADIO_NUM_BANDS_MAX]; /* band descriptors */
} radio_hal_properties_t;

/* Used internally by the framework. Same information as in struct radio_hal_properties plus a
 * unique handle and one band configuration per region. */
typedef struct radio_properties {
    radio_handle_t      handle;
    radio_class_t       class_id;
    char                implementor[RADIO_STRING_LEN_MAX];
    char                product[RADIO_STRING_LEN_MAX];
    char                version[RADIO_STRING_LEN_MAX];
    char                serial[RADIO_STRING_LEN_MAX];
    uint32_t            num_tuners;
    uint32_t            num_audio_sources;
    bool                supports_capture;
    uint32_t            num_bands;
    radio_band_config_t bands[RADIO_NUM_BANDS_MAX];
} radio_properties_t;

/* Radio program information. Returned by the HAL with event RADIO_EVENT_TUNED.
 * Contains information on currently tuned channel.
 */
typedef struct radio_program_info {
    uint32_t         channel;   /* current channel. (e.g kHz for band type RADIO_BAND_FM) */
    uint32_t         sub_channel; /* current sub channel. (used for RADIO_BAND_FM_HD) */
    bool             tuned;     /* tuned to a program or not */
    bool             stereo;    /* program is stereo or not */
    bool             digital;   /* digital program or not (e.g HD Radio program) */
    uint32_t         signal_strength; /* signal strength from 0 to 100 */
                                /* meta data (e.g PTY, song title ...), must not be NULL */
    __attribute__((aligned(8))) radio_metadata_t *metadata;
} radio_program_info_t;


/* Events sent to the framework via the HAL callback. An event can notify the completion of an
 * asynchronous command (configuration, tune, scan ...) or a spontaneous change (antenna connection,
 * failure, AF switching, meta data reception... */
enum {
    RADIO_EVENT_HW_FAILURE  = 0,  /* hardware module failure. Requires reopening the tuner */
    RADIO_EVENT_CONFIG      = 1,  /* configuration change completed */
    RADIO_EVENT_ANTENNA     = 2,  /* Antenna connected, disconnected */
    RADIO_EVENT_TUNED       = 3,  /* tune, step, scan completed */
    RADIO_EVENT_METADATA    = 4,  /* New meta data received */
    RADIO_EVENT_TA          = 5,  /* Traffic announcement start or stop */
    RADIO_EVENT_AF_SWITCH   = 6,  /* Switch to Alternate Frequency */
    RADIO_EVENT_EA          = 7,  /* Emergency announcement start or stop */
    // begin framework only events
    RADIO_EVENT_CONTROL     = 100, /* loss/gain of tuner control */
    RADIO_EVENT_SERVER_DIED = 101, /* radio service died */
};
typedef unsigned int radio_event_type_t;

/* Event passed to the framework by the HAL callback */
typedef struct radio_hal_event {
    radio_event_type_t  type;       /* event type */
    int32_t             status;     /* used by RADIO_EVENT_CONFIG, RADIO_EVENT_TUNED */
    union {
        /* RADIO_EVENT_ANTENNA, RADIO_EVENT_TA, RADIO_EVENT_EA */
        bool                    on;
        radio_hal_band_config_t config; /* RADIO_EVENT_CONFIG */
        radio_program_info_t    info;   /* RADIO_EVENT_TUNED, RADIO_EVENT_AF_SWITCH */
        radio_metadata_t        *metadata; /* RADIO_EVENT_METADATA */
    };
} radio_hal_event_t;

/* Used internally by the framework. Same information as in struct radio_hal_event */
typedef struct radio_event {
    radio_event_type_t  type;
    int32_t             status;
    union {
        bool                    on;
        radio_band_config_t     config;
        radio_program_info_t    info;
                                /* meta data (e.g PTY, song title ...), must not be NULL */
        __attribute__((aligned(8))) radio_metadata_t *metadata;
    };
} radio_event_t;


static inline
radio_rds_t radio_rds_for_region(bool rds, radio_region_t region) {
    if (!rds)
        return RADIO_RDS_NONE;
    switch(region) {
        case RADIO_REGION_ITU_1:
        case RADIO_REGION_OIRT:
        case RADIO_REGION_JAPAN:
        case RADIO_REGION_KOREA:
            return RADIO_RDS_WORLD;
        case RADIO_REGION_ITU_2:
            return RADIO_RDS_US;
        default:
            return RADIO_REGION_NONE;
    }
}

static inline
radio_deemphasis_t radio_demephasis_for_region(radio_region_t region) {
    switch(region) {
        case RADIO_REGION_KOREA:
        case RADIO_REGION_ITU_2:
            return RADIO_DEEMPHASIS_75;
        case RADIO_REGION_ITU_1:
        case RADIO_REGION_OIRT:
        case RADIO_REGION_JAPAN:
        default:
            return RADIO_DEEMPHASIS_50;
    }
}

#endif  // ANDROID_RADIO_H
