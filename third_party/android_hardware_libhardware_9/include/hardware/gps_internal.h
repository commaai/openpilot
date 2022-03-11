/*
 * Copyright (C) 2016 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_HARDWARE_GPS_INTERNAL_H
#define ANDROID_INCLUDE_HARDWARE_GPS_INTERNAL_H

#include "hardware/gps.h"

/****************************************************************************
 * This file contains legacy structs that are deprecated/retired from gps.h *
 ****************************************************************************/

__BEGIN_DECLS

/**
 * Legacy GPS callback structure.
 * Deprecated, to be removed in the next Android release.
 * Use GpsCallbacks instead.
 */
typedef struct {
    /** set to sizeof(GpsCallbacks_v1) */
    size_t      size;
    gps_location_callback location_cb;
    gps_status_callback status_cb;
    gps_sv_status_callback sv_status_cb;
    gps_nmea_callback nmea_cb;
    gps_set_capabilities set_capabilities_cb;
    gps_acquire_wakelock acquire_wakelock_cb;
    gps_release_wakelock release_wakelock_cb;
    gps_create_thread create_thread_cb;
    gps_request_utc_time request_utc_time_cb;
} GpsCallbacks_v1;

#pragma pack(push,4)
// We need to keep the alignment of this data structure to 4-bytes, to ensure that in 64-bit
// environments the size of this legacy definition does not collide with _v2. Implementations should
// be using _v2 and _v3, so it's OK to pay the 'unaligned' penalty in 64-bit if an old
// implementation is still in use.

/**
 * Legacy struct to represent the status of AGPS.
 */
typedef struct {
    /** set to sizeof(AGpsStatus_v1) */
    size_t          size;
    AGpsType        type;
    AGpsStatusValue status;
} AGpsStatus_v1;

#pragma pack(pop)

/**
 * Legacy struct to represent the status of AGPS augmented with a IPv4 address
 * field.
 */
typedef struct {
    /** set to sizeof(AGpsStatus_v2) */
    size_t          size;
    AGpsType        type;
    AGpsStatusValue status;

    /*-------------------- New fields in _v2 --------------------*/

    uint32_t        ipaddr;
} AGpsStatus_v2;

/**
 * Legacy extended interface for AGPS support.
 * See AGpsInterface_v2 for more information.
 */
typedef struct {
    /** set to sizeof(AGpsInterface_v1) */
    size_t          size;
    void  (*init)( AGpsCallbacks* callbacks );
    int  (*data_conn_open)( const char* apn );
    int  (*data_conn_closed)();
    int  (*data_conn_failed)();
    int  (*set_server)( AGpsType type, const char* hostname, int port );
} AGpsInterface_v1;

__END_DECLS

#endif /* ANDROID_INCLUDE_HARDWARE_GPS_INTERNAL_H */
