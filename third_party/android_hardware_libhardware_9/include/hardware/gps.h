/*
 * Copyright (C) 2010 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_HARDWARE_GPS_H
#define ANDROID_INCLUDE_HARDWARE_GPS_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <pthread.h>
#include <sys/socket.h>
#include <stdbool.h>

#include <hardware/hardware.h>

#include "gnss-base.h"

__BEGIN_DECLS

/*
 * Enums defined in HIDL in hardware/interfaces are auto-generated and present
 * in gnss-base.h.
 */

/* for compatibility */

/** Maximum number of SVs for gps_sv_status_callback(). */
#define GNSS_MAX_SVS GNSS_MAX_SVS_COUNT
/** Maximum number of Measurements in gnss_measurement_callback(). */
#define GNSS_MAX_MEASUREMENT GNSS_MAX_SVS_COUNT

#define GPS_REQUEST_AGPS_DATA_CONN GNSS_REQUEST_AGNSS_DATA_CONN
#define GPS_RELEASE_AGPS_DATA_CONN GNSS_RELEASE_AGNSS_DATA_CONN
#define GPS_AGPS_DATA_CONNECTED GNSS_AGNSS_DATA_CONNECTED
#define GPS_AGPS_DATA_CONN_DONE GNSS_AGNSS_DATA_CONN_DONE
#define GPS_AGPS_DATA_CONN_FAILED GNSS_AGNSS_DATA_CONN_FAILED
#define AGPS_RIL_NETWORK_TYPE_MOBILE_MMS AGPS_RIL_NETWORK_TYPE_MMS
#define AGPS_RIL_NETWORK_TYPE_MOBILE_SUPL AGPS_RIL_NETWORK_TYPE_SUPL
#define AGPS_RIL_NETWORK_TTYPE_MOBILE_DUN AGPS_RIL_NETWORK_TYPE_DUN
#define AGPS_RIL_NETWORK_TTYPE_MOBILE_HIPRI AGPS_RIL_NETWORK_TYPE_HIPRI
#define AGPS_RIL_NETWORK_TTYPE_WIMAX AGPS_RIL_NETWORK_TYPE_WIMAX
#define GNSS_MULTIPATH_INDICATOR_NOT_PRESENT GNSS_MULTIPATH_INDICATIOR_NOT_PRESENT
#define AGPS_SETID_TYPE_MSISDN AGPS_SETID_TYPE_MSISDM
#define GPS_MEASUREMENT_OPERATION_SUCCESS GPS_MEASUREMENT_SUCCESS
#define GPS_NAVIGATION_MESSAGE_OPERATION_SUCCESS GPS_NAVIGATION_MESSAGE_SUCCESS
#define GNSS_NAVIGATION_MESSAGE_TYPE_GPS_L1CA GNSS_NAVIGATION_MESSAGE_TYPE_GNSS_L1CA
#define GNSS_NAVIGATION_MESSAGE_TYPE_GPS_L2CNAV GNSS_NAVIGATION_MESSAGE_TYPE_GNSS_L2CNAV
#define GNSS_NAVIGATION_MESSAGE_TYPE_GPS_L5CNAV GNSS_NAVIGATION_MESSAGE_TYPE_GNSS_L5CNAV
#define GNSS_NAVIGATION_MESSAGE_TYPE_GPS_CNAV2 GNSS_NAVIGATION_MESSAGE_TYPE_GNSS_CNAV2
#define GPS_LOCATION_HAS_ACCURACY GPS_LOCATION_HAS_HORIZONTAL_ACCURACY

/**
 * The id of this module
 */
#define GPS_HARDWARE_MODULE_ID "gps"


/** Milliseconds since January 1, 1970 */
typedef int64_t GpsUtcTime;

/** Maximum number of SVs for gps_sv_status_callback(). */
#define GPS_MAX_SVS 32

/** Maximum number of Measurements in gps_measurement_callback(). */
#define GPS_MAX_MEASUREMENT   32

/** Requested operational mode for GPS operation. */
typedef uint32_t GpsPositionMode;

/** Requested recurrence mode for GPS operation. */
typedef uint32_t GpsPositionRecurrence;

/** GPS status event values. */
typedef uint16_t GpsStatusValue;

/** Flags to indicate which values are valid in a GpsLocation. */
typedef uint16_t GpsLocationFlags;

/**
 * Flags used to specify which aiding data to delete when calling
 * delete_aiding_data().
 */
typedef uint16_t GpsAidingData;

/** AGPS type */
typedef uint16_t AGpsType;

typedef uint16_t AGpsSetIDType;

typedef uint16_t ApnIpType;

/**
 * String length constants
 */
#define GPS_NI_SHORT_STRING_MAXLEN      256
#define GPS_NI_LONG_STRING_MAXLEN       2048

/**
 * GpsNiType constants
 */
typedef uint32_t GpsNiType;

/**
 * GpsNiNotifyFlags constants
 */
typedef uint32_t GpsNiNotifyFlags;

/**
 * GPS NI responses, used to define the response in
 * NI structures
 */
typedef int GpsUserResponseType;

/**
 * NI data encoding scheme
 */
typedef int GpsNiEncodingType;

/** AGPS status event values. */
typedef uint16_t AGpsStatusValue;

typedef uint16_t AGpsRefLocationType;

/* Deprecated, to be removed in the next Android release. */
#define AGPS_REG_LOCATION_TYPE_MAC          3

/* The following typedef together with its constants below are deprecated, and
 * will be removed in the next release. */
typedef uint16_t GpsClockFlags;
#define GPS_CLOCK_HAS_LEAP_SECOND               (1<<0)
#define GPS_CLOCK_HAS_TIME_UNCERTAINTY          (1<<1)
#define GPS_CLOCK_HAS_FULL_BIAS                 (1<<2)
#define GPS_CLOCK_HAS_BIAS                      (1<<3)
#define GPS_CLOCK_HAS_BIAS_UNCERTAINTY          (1<<4)
#define GPS_CLOCK_HAS_DRIFT                     (1<<5)
#define GPS_CLOCK_HAS_DRIFT_UNCERTAINTY         (1<<6)

/**
 * Flags to indicate what fields in GnssClock are valid.
 */
typedef uint16_t GnssClockFlags;

/* The following typedef together with its constants below are deprecated, and
 * will be removed in the next release. */
typedef uint8_t GpsClockType;
#define GPS_CLOCK_TYPE_UNKNOWN                  0
#define GPS_CLOCK_TYPE_LOCAL_HW_TIME            1
#define GPS_CLOCK_TYPE_GPS_TIME                 2

/* The following typedef together with its constants below are deprecated, and
 * will be removed in the next release. */
typedef uint32_t GpsMeasurementFlags;
#define GPS_MEASUREMENT_HAS_SNR                               (1<<0)
#define GPS_MEASUREMENT_HAS_ELEVATION                         (1<<1)
#define GPS_MEASUREMENT_HAS_ELEVATION_UNCERTAINTY             (1<<2)
#define GPS_MEASUREMENT_HAS_AZIMUTH                           (1<<3)
#define GPS_MEASUREMENT_HAS_AZIMUTH_UNCERTAINTY               (1<<4)
#define GPS_MEASUREMENT_HAS_PSEUDORANGE                       (1<<5)
#define GPS_MEASUREMENT_HAS_PSEUDORANGE_UNCERTAINTY           (1<<6)
#define GPS_MEASUREMENT_HAS_CODE_PHASE                        (1<<7)
#define GPS_MEASUREMENT_HAS_CODE_PHASE_UNCERTAINTY            (1<<8)
#define GPS_MEASUREMENT_HAS_CARRIER_FREQUENCY                 (1<<9)
#define GPS_MEASUREMENT_HAS_CARRIER_CYCLES                    (1<<10)
#define GPS_MEASUREMENT_HAS_CARRIER_PHASE                     (1<<11)
#define GPS_MEASUREMENT_HAS_CARRIER_PHASE_UNCERTAINTY         (1<<12)
#define GPS_MEASUREMENT_HAS_BIT_NUMBER                        (1<<13)
#define GPS_MEASUREMENT_HAS_TIME_FROM_LAST_BIT                (1<<14)
#define GPS_MEASUREMENT_HAS_DOPPLER_SHIFT                     (1<<15)
#define GPS_MEASUREMENT_HAS_DOPPLER_SHIFT_UNCERTAINTY         (1<<16)
#define GPS_MEASUREMENT_HAS_USED_IN_FIX                       (1<<17)
#define GPS_MEASUREMENT_HAS_UNCORRECTED_PSEUDORANGE_RATE      (1<<18)

/**
 * Flags to indicate what fields in GnssMeasurement are valid.
 */
typedef uint32_t GnssMeasurementFlags;

/* The following typedef together with its constants below are deprecated, and
 * will be removed in the next release. */
typedef uint8_t GpsLossOfLock;
#define GPS_LOSS_OF_LOCK_UNKNOWN                            0
#define GPS_LOSS_OF_LOCK_OK                                 1
#define GPS_LOSS_OF_LOCK_CYCLE_SLIP                         2

/* The following typedef together with its constants below are deprecated, and
 * will be removed in the next release. Use GnssMultipathIndicator instead.
 */
typedef uint8_t GpsMultipathIndicator;
#define GPS_MULTIPATH_INDICATOR_UNKNOWN                 0
#define GPS_MULTIPATH_INDICATOR_DETECTED                1
#define GPS_MULTIPATH_INDICATOR_NOT_USED                2

/**
 * Enumeration of available values for the GNSS Measurement's multipath
 * indicator.
 */
typedef uint8_t GnssMultipathIndicator;

/* The following typedef together with its constants below are deprecated, and
 * will be removed in the next release. */
typedef uint16_t GpsMeasurementState;
#define GPS_MEASUREMENT_STATE_UNKNOWN                   0
#define GPS_MEASUREMENT_STATE_CODE_LOCK             (1<<0)
#define GPS_MEASUREMENT_STATE_BIT_SYNC              (1<<1)
#define GPS_MEASUREMENT_STATE_SUBFRAME_SYNC         (1<<2)
#define GPS_MEASUREMENT_STATE_TOW_DECODED           (1<<3)
#define GPS_MEASUREMENT_STATE_MSEC_AMBIGUOUS        (1<<4)

/**
 * Flags indicating the GNSS measurement state.
 *
 * The expected behavior here is for GPS HAL to set all the flags that applies.
 * For example, if the state for a satellite is only C/A code locked and bit
 * synchronized, and there is still millisecond ambiguity, the state should be
 * set as:
 *
 * GNSS_MEASUREMENT_STATE_CODE_LOCK | GNSS_MEASUREMENT_STATE_BIT_SYNC |
 *         GNSS_MEASUREMENT_STATE_MSEC_AMBIGUOUS
 *
 * If GNSS is still searching for a satellite, the corresponding state should be
 * set to GNSS_MEASUREMENT_STATE_UNKNOWN(0).
 */
typedef uint32_t GnssMeasurementState;

/* The following typedef together with its constants below are deprecated, and
 * will be removed in the next release. */
typedef uint16_t GpsAccumulatedDeltaRangeState;
#define GPS_ADR_STATE_UNKNOWN                       0
#define GPS_ADR_STATE_VALID                     (1<<0)
#define GPS_ADR_STATE_RESET                     (1<<1)
#define GPS_ADR_STATE_CYCLE_SLIP                (1<<2)

/**
 * Flags indicating the Accumulated Delta Range's states.
 */
typedef uint16_t GnssAccumulatedDeltaRangeState;

/* The following typedef together with its constants below are deprecated, and
 * will be removed in the next release. */
typedef uint8_t GpsNavigationMessageType;
#define GPS_NAVIGATION_MESSAGE_TYPE_UNKNOWN         0
#define GPS_NAVIGATION_MESSAGE_TYPE_L1CA            1
#define GPS_NAVIGATION_MESSAGE_TYPE_L2CNAV          2
#define GPS_NAVIGATION_MESSAGE_TYPE_L5CNAV          3
#define GPS_NAVIGATION_MESSAGE_TYPE_CNAV2           4

/**
 * Enumeration of available values to indicate the GNSS Navigation message
 * types.
 *
 * For convenience, first byte is the GnssConstellationType on which that signal
 * is typically transmitted
 */
typedef int16_t GnssNavigationMessageType;

/**
 * Status of Navigation Message
 * When a message is received properly without any parity error in its navigation words, the
 * status should be set to NAV_MESSAGE_STATUS_PARITY_PASSED. But if a message is received
 * with words that failed parity check, but GPS is able to correct those words, the status
 * should be set to NAV_MESSAGE_STATUS_PARITY_REBUILT.
 * No need to send any navigation message that contains words with parity error and cannot be
 * corrected.
 */
typedef uint16_t NavigationMessageStatus;

/* This constant is deprecated, and will be removed in the next release. */
#define NAV_MESSAGE_STATUS_UNKONW              0

/**
 * Flags that indicate information about the satellite
 */
typedef uint8_t                                 GnssSvFlags;

/**
 * Constellation type of GnssSvInfo
 */
typedef uint8_t                         GnssConstellationType;

/**
 * Name for the GPS XTRA interface.
 */
#define GPS_XTRA_INTERFACE      "gps-xtra"

/**
 * Name for the GPS DEBUG interface.
 */
#define GPS_DEBUG_INTERFACE      "gps-debug"

/**
 * Name for the AGPS interface.
 */
#define AGPS_INTERFACE      "agps"

/**
 * Name of the Supl Certificate interface.
 */
#define SUPL_CERTIFICATE_INTERFACE  "supl-certificate"

/**
 * Name for NI interface
 */
#define GPS_NI_INTERFACE "gps-ni"

/**
 * Name for the AGPS-RIL interface.
 */
#define AGPS_RIL_INTERFACE      "agps_ril"

/**
 * Name for the GPS_Geofencing interface.
 */
#define GPS_GEOFENCING_INTERFACE   "gps_geofencing"

/**
 * Name of the GPS Measurements interface.
 */
#define GPS_MEASUREMENT_INTERFACE   "gps_measurement"

/**
 * Name of the GPS navigation message interface.
 */
#define GPS_NAVIGATION_MESSAGE_INTERFACE     "gps_navigation_message"

/**
 * Name of the GNSS/GPS configuration interface.
 */
#define GNSS_CONFIGURATION_INTERFACE     "gnss_configuration"

/** Represents a location. */
typedef struct {
    /** set to sizeof(GpsLocation) */
    size_t          size;
    /** Contains GpsLocationFlags bits. */
    uint16_t        flags;
    /** Represents latitude in degrees. */
    double          latitude;
    /** Represents longitude in degrees. */
    double          longitude;
    /**
     * Represents altitude in meters above the WGS 84 reference ellipsoid.
     */
    double          altitude;
    /** Represents speed in meters per second. */
    float           speed;
    /** Represents heading in degrees. */
    float           bearing;
    /** Represents expected accuracy in meters. */
    float           accuracy;
    /** Timestamp for the location fix. */
    GpsUtcTime      timestamp;
} GpsLocation;

/** Represents the status. */
typedef struct {
    /** set to sizeof(GpsStatus) */
    size_t          size;
    GpsStatusValue status;
} GpsStatus;

/**
 * Legacy struct to represents SV information.
 * Deprecated, to be removed in the next Android release.
 * Use GnssSvInfo instead.
 */
typedef struct {
    /** set to sizeof(GpsSvInfo) */
    size_t          size;
    /** Pseudo-random number for the SV. */
    int     prn;
    /** Signal to noise ratio. */
    float   snr;
    /** Elevation of SV in degrees. */
    float   elevation;
    /** Azimuth of SV in degrees. */
    float   azimuth;
} GpsSvInfo;

typedef struct {
    /** set to sizeof(GnssSvInfo) */
    size_t size;

    /**
     * Pseudo-random number for the SV, or FCN/OSN number for Glonass. The
     * distinction is made by looking at constellation field. Values should be
     * in the range of:
     *
     * - GPS:     1-32
     * - SBAS:    120-151, 183-192
     * - GLONASS: 1-24, the orbital slot number (OSN), if known.  Or, if not:
     *            93-106, the frequency channel number (FCN) (-7 to +6) offset by + 100
     *            i.e. report an FCN of -7 as 93, FCN of 0 as 100, and FCN of +6 as 106.
     * - QZSS:    193-200
     * - Galileo: 1-36
     * - Beidou:  1-37
     */
    int16_t svid;

    /**
     * Defines the constellation of the given SV. Value should be one of those
     * GNSS_CONSTELLATION_* constants
     */
    GnssConstellationType constellation;

    /**
     * Carrier-to-noise density in dB-Hz, typically in the range [0, 63].
     * It contains the measured C/N0 value for the signal at the antenna port.
     *
     * This is a mandatory value.
     */
    float c_n0_dbhz;

    /** Elevation of SV in degrees. */
    float elevation;

    /** Azimuth of SV in degrees. */
    float azimuth;

    /**
     * Contains additional data about the given SV. Value should be one of those
     * GNSS_SV_FLAGS_* constants
     */
    GnssSvFlags flags;

} GnssSvInfo;

/**
 * Legacy struct to represents SV status.
 * Deprecated, to be removed in the next Android release.
 * Use GnssSvStatus instead.
 */
typedef struct {
    /** set to sizeof(GpsSvStatus) */
    size_t size;
    int num_svs;
    GpsSvInfo sv_list[GPS_MAX_SVS];
    uint32_t ephemeris_mask;
    uint32_t almanac_mask;
    uint32_t used_in_fix_mask;
} GpsSvStatus;

/**
 * Represents SV status.
 */
typedef struct {
    /** set to sizeof(GnssSvStatus) */
    size_t size;

    /** Number of GPS SVs currently visible, refers to the SVs stored in sv_list */
    int num_svs;
    /**
     * Pointer to an array of SVs information for all GNSS constellations,
     * except GPS, which is reported using sv_list
     */
    GnssSvInfo gnss_sv_list[GNSS_MAX_SVS];

} GnssSvStatus;

/* CellID for 2G, 3G and LTE, used in AGPS. */
typedef struct {
    AGpsRefLocationType type;
    /** Mobile Country Code. */
    uint16_t mcc;
    /** Mobile Network Code .*/
    uint16_t mnc;
    /** Location Area Code in 2G, 3G and LTE. In 3G lac is discarded. In LTE,
     * lac is populated with tac, to ensure that we don't break old clients that
     * might rely in the old (wrong) behavior.
     */
    uint16_t lac;
    /** Cell id in 2G. Utran Cell id in 3G. Cell Global Id EUTRA in LTE. */
    uint32_t cid;
    /** Tracking Area Code in LTE. */
    uint16_t tac;
    /** Physical Cell id in LTE (not used in 2G and 3G) */
    uint16_t pcid;
} AGpsRefLocationCellID;

typedef struct {
    uint8_t mac[6];
} AGpsRefLocationMac;

/** Represents ref locations */
typedef struct {
    AGpsRefLocationType type;
    union {
        AGpsRefLocationCellID   cellID;
        AGpsRefLocationMac      mac;
    } u;
} AGpsRefLocation;

/**
 * Callback with location information. Can only be called from a thread created
 * by create_thread_cb.
 */
typedef void (* gps_location_callback)(GpsLocation* location);

/**
 * Callback with status information. Can only be called from a thread created by
 * create_thread_cb.
 */
typedef void (* gps_status_callback)(GpsStatus* status);

/**
 * Legacy callback with SV status information.
 * Can only be called from a thread created by create_thread_cb.
 *
 * This callback is deprecated, and will be removed in the next release. Use
 * gnss_sv_status_callback() instead.
 */
typedef void (* gps_sv_status_callback)(GpsSvStatus* sv_info);

/**
 * Callback with SV status information.
 * Can only be called from a thread created by create_thread_cb.
 */
typedef void (* gnss_sv_status_callback)(GnssSvStatus* sv_info);

/**
 * Callback for reporting NMEA sentences. Can only be called from a thread
 * created by create_thread_cb.
 */
typedef void (* gps_nmea_callback)(GpsUtcTime timestamp, const char* nmea, int length);

/**
 * Callback to inform framework of the GPS engine's capabilities. Capability
 * parameter is a bit field of GPS_CAPABILITY_* flags.
 */
typedef void (* gps_set_capabilities)(uint32_t capabilities);

/**
 * Callback utility for acquiring the GPS wakelock. This can be used to prevent
 * the CPU from suspending while handling GPS events.
 */
typedef void (* gps_acquire_wakelock)();

/** Callback utility for releasing the GPS wakelock. */
typedef void (* gps_release_wakelock)();

/** Callback for requesting NTP time */
typedef void (* gps_request_utc_time)();

/**
 * Callback for creating a thread that can call into the Java framework code.
 * This must be used to create any threads that report events up to the
 * framework.
 */
typedef pthread_t (* gps_create_thread)(const char* name, void (*start)(void *), void* arg);

/**
 * Provides information about how new the underlying GPS/GNSS hardware and
 * software is.
 *
 * This information will be available for Android Test Applications. If a GPS
 * HAL does not provide this information, it will be considered "2015 or
 * earlier".
 *
 * If a GPS HAL does provide this information, then newer years will need to
 * meet newer CTS standards. E.g. if the date are 2016 or above, then N+ level
 * GpsMeasurement support will be verified.
 */
typedef struct {
    /** Set to sizeof(GnssSystemInfo) */
    size_t   size;
    /* year in which the last update was made to the underlying hardware/firmware
     * used to capture GNSS signals, e.g. 2016 */
    uint16_t year_of_hw;
} GnssSystemInfo;

/**
 * Callback to inform framework of the engine's hardware version information.
 */
typedef void (*gnss_set_system_info)(const GnssSystemInfo* info);

/** New GPS callback structure. */
typedef struct {
    /** set to sizeof(GpsCallbacks) */
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

    gnss_set_system_info set_system_info_cb;
    gnss_sv_status_callback gnss_sv_status_cb;
} GpsCallbacks;

/** Represents the standard GPS interface. */
typedef struct {
    /** set to sizeof(GpsInterface) */
    size_t          size;
    /**
     * Opens the interface and provides the callback routines
     * to the implementation of this interface.
     */
    int   (*init)( GpsCallbacks* callbacks );

    /** Starts navigating. */
    int   (*start)( void );

    /** Stops navigating. */
    int   (*stop)( void );

    /** Closes the interface. */
    void  (*cleanup)( void );

    /** Injects the current time. */
    int   (*inject_time)(GpsUtcTime time, int64_t timeReference,
                         int uncertainty);

    /**
     * Injects current location from another location provider (typically cell
     * ID). Latitude and longitude are measured in degrees expected accuracy is
     * measured in meters
     */
    int  (*inject_location)(double latitude, double longitude, float accuracy);

    /**
     * Specifies that the next call to start will not use the
     * information defined in the flags. GPS_DELETE_ALL is passed for
     * a cold start.
     */
    void  (*delete_aiding_data)(GpsAidingData flags);

    /**
     * min_interval represents the time between fixes in milliseconds.
     * preferred_accuracy represents the requested fix accuracy in meters.
     * preferred_time represents the requested time to first fix in milliseconds.
     *
     * 'mode' parameter should be one of GPS_POSITION_MODE_MS_BASED
     * or GPS_POSITION_MODE_STANDALONE.
     * It is allowed by the platform (and it is recommended) to fallback to
     * GPS_POSITION_MODE_MS_BASED if GPS_POSITION_MODE_MS_ASSISTED is passed in, and
     * GPS_POSITION_MODE_MS_BASED is supported.
     */
    int   (*set_position_mode)(GpsPositionMode mode, GpsPositionRecurrence recurrence,
            uint32_t min_interval, uint32_t preferred_accuracy, uint32_t preferred_time);

    /** Get a pointer to extension information. */
    const void* (*get_extension)(const char* name);
} GpsInterface;

/**
 * Callback to request the client to download XTRA data. The client should
 * download XTRA data and inject it by calling inject_xtra_data(). Can only be
 * called from a thread created by create_thread_cb.
 */
typedef void (* gps_xtra_download_request)();

/** Callback structure for the XTRA interface. */
typedef struct {
    gps_xtra_download_request download_request_cb;
    gps_create_thread create_thread_cb;
} GpsXtraCallbacks;

/** Extended interface for XTRA support. */
typedef struct {
    /** set to sizeof(GpsXtraInterface) */
    size_t          size;
    /**
     * Opens the XTRA interface and provides the callback routines
     * to the implementation of this interface.
     */
    int  (*init)( GpsXtraCallbacks* callbacks );
    /** Injects XTRA data into the GPS. */
    int  (*inject_xtra_data)( char* data, int length );
} GpsXtraInterface;

/** Extended interface for DEBUG support. */
typedef struct {
    /** set to sizeof(GpsDebugInterface) */
    size_t          size;

    /**
     * This function should return any information that the native
     * implementation wishes to include in a bugreport.
     */
    size_t (*get_internal_state)(char* buffer, size_t bufferSize);
} GpsDebugInterface;

/*
 * Represents the status of AGPS augmented to support IPv4 and IPv6.
 */
typedef struct {
    /** set to sizeof(AGpsStatus) */
    size_t                  size;

    AGpsType                type;
    AGpsStatusValue         status;

    /**
     * Must be set to a valid IPv4 address if the field 'addr' contains an IPv4
     * address, or set to INADDR_NONE otherwise.
     */
    uint32_t                ipaddr;

    /**
     * Must contain the IPv4 (AF_INET) or IPv6 (AF_INET6) address to report.
     * Any other value of addr.ss_family will be rejected.
     */
    struct sockaddr_storage addr;
} AGpsStatus;

/**
 * Callback with AGPS status information. Can only be called from a thread
 * created by create_thread_cb.
 */
typedef void (* agps_status_callback)(AGpsStatus* status);

/** Callback structure for the AGPS interface. */
typedef struct {
    agps_status_callback status_cb;
    gps_create_thread create_thread_cb;
} AGpsCallbacks;

/**
 * Extended interface for AGPS support, it is augmented to enable to pass
 * extra APN data.
 */
typedef struct {
    /** set to sizeof(AGpsInterface) */
    size_t size;

    /**
     * Opens the AGPS interface and provides the callback routines to the
     * implementation of this interface.
     */
    void (*init)(AGpsCallbacks* callbacks);
    /**
     * Deprecated.
     * If the HAL supports AGpsInterface_v2 this API will not be used, see
     * data_conn_open_with_apn_ip_type for more information.
     */
    int (*data_conn_open)(const char* apn);
    /**
     * Notifies that the AGPS data connection has been closed.
     */
    int (*data_conn_closed)();
    /**
     * Notifies that a data connection is not available for AGPS.
     */
    int (*data_conn_failed)();
    /**
     * Sets the hostname and port for the AGPS server.
     */
    int (*set_server)(AGpsType type, const char* hostname, int port);

    /**
     * Notifies that a data connection is available and sets the name of the
     * APN, and its IP type, to be used for SUPL connections.
     */
    int (*data_conn_open_with_apn_ip_type)(
            const char* apn,
            ApnIpType apnIpType);
} AGpsInterface;

/** Error codes associated with certificate operations */
#define AGPS_CERTIFICATE_OPERATION_SUCCESS               0
#define AGPS_CERTIFICATE_ERROR_GENERIC                -100
#define AGPS_CERTIFICATE_ERROR_TOO_MANY_CERTIFICATES  -101

/** A data structure that represents an X.509 certificate using DER encoding */
typedef struct {
    size_t  length;
    u_char* data;
} DerEncodedCertificate;

/**
 * A type definition for SHA1 Fingerprints used to identify X.509 Certificates
 * The Fingerprint is a digest of the DER Certificate that uniquely identifies it.
 */
typedef struct {
    u_char data[20];
} Sha1CertificateFingerprint;

/** AGPS Interface to handle SUPL certificate operations */
typedef struct {
    /** set to sizeof(SuplCertificateInterface) */
    size_t size;

    /**
     * Installs a set of Certificates used for SUPL connections to the AGPS server.
     * If needed the HAL should find out internally any certificates that need to be removed to
     * accommodate the certificates to install.
     * The certificates installed represent a full set of valid certificates needed to connect to
     * AGPS SUPL servers.
     * The list of certificates is required, and all must be available at the same time, when trying
     * to establish a connection with the AGPS Server.
     *
     * Parameters:
     *      certificates - A pointer to an array of DER encoded certificates that are need to be
     *                     installed in the HAL.
     *      length - The number of certificates to install.
     * Returns:
     *      AGPS_CERTIFICATE_OPERATION_SUCCESS if the operation is completed successfully
     *      AGPS_CERTIFICATE_ERROR_TOO_MANY_CERTIFICATES if the HAL cannot store the number of
     *          certificates attempted to be installed, the state of the certificates stored should
     *          remain the same as before on this error case.
     *
     * IMPORTANT:
     *      If needed the HAL should find out internally the set of certificates that need to be
     *      removed to accommodate the certificates to install.
     */
    int  (*install_certificates) ( const DerEncodedCertificate* certificates, size_t length );

    /**
     * Notifies the HAL that a list of certificates used for SUPL connections are revoked. It is
     * expected that the given set of certificates is removed from the internal store of the HAL.
     *
     * Parameters:
     *      fingerprints - A pointer to an array of SHA1 Fingerprints to identify the set of
     *                     certificates to revoke.
     *      length - The number of fingerprints provided.
     * Returns:
     *      AGPS_CERTIFICATE_OPERATION_SUCCESS if the operation is completed successfully.
     *
     * IMPORTANT:
     *      If any of the certificates provided (through its fingerprint) is not known by the HAL,
     *      it should be ignored and continue revoking/deleting the rest of them.
     */
    int  (*revoke_certificates) ( const Sha1CertificateFingerprint* fingerprints, size_t length );
} SuplCertificateInterface;

/** Represents an NI request */
typedef struct {
    /** set to sizeof(GpsNiNotification) */
    size_t          size;

    /**
     * An ID generated by HAL to associate NI notifications and UI
     * responses
     */
    int             notification_id;

    /**
     * An NI type used to distinguish different categories of NI
     * events, such as GPS_NI_TYPE_VOICE, GPS_NI_TYPE_UMTS_SUPL, ...
     */
    GpsNiType       ni_type;

    /**
     * Notification/verification options, combinations of GpsNiNotifyFlags constants
     */
    GpsNiNotifyFlags notify_flags;

    /**
     * Timeout period to wait for user response.
     * Set to 0 for no time out limit.
     */
    int             timeout;

    /**
     * Default response when time out.
     */
    GpsUserResponseType default_response;

    /**
     * Requestor ID
     */
    char            requestor_id[GPS_NI_SHORT_STRING_MAXLEN];

    /**
     * Notification message. It can also be used to store client_id in some cases
     */
    char            text[GPS_NI_LONG_STRING_MAXLEN];

    /**
     * Client name decoding scheme
     */
    GpsNiEncodingType requestor_id_encoding;

    /**
     * Client name decoding scheme
     */
    GpsNiEncodingType text_encoding;

    /**
     * A pointer to extra data. Format:
     * key_1 = value_1
     * key_2 = value_2
     */
    char           extras[GPS_NI_LONG_STRING_MAXLEN];

} GpsNiNotification;

/**
 * Callback with NI notification. Can only be called from a thread created by
 * create_thread_cb.
 */
typedef void (*gps_ni_notify_callback)(GpsNiNotification *notification);

/** GPS NI callback structure. */
typedef struct
{
    /**
     * Sends the notification request from HAL to GPSLocationProvider.
     */
    gps_ni_notify_callback notify_cb;
    gps_create_thread create_thread_cb;
} GpsNiCallbacks;

/**
 * Extended interface for Network-initiated (NI) support.
 */
typedef struct
{
    /** set to sizeof(GpsNiInterface) */
    size_t          size;

   /** Registers the callbacks for HAL to use. */
   void (*init) (GpsNiCallbacks *callbacks);

   /** Sends a response to HAL. */
   void (*respond) (int notif_id, GpsUserResponseType user_response);
} GpsNiInterface;

struct gps_device_t {
    struct hw_device_t common;

    /**
     * Set the provided lights to the provided values.
     *
     * Returns: 0 on succes, error code on failure.
     */
    const GpsInterface* (*get_gps_interface)(struct gps_device_t* dev);
};

#define AGPS_RIL_REQUEST_REFLOC_CELLID  (1<<0L)
#define AGPS_RIL_REQUEST_REFLOC_MAC     (1<<1L)

typedef void (*agps_ril_request_set_id)(uint32_t flags);
typedef void (*agps_ril_request_ref_loc)(uint32_t flags);

typedef struct {
    agps_ril_request_set_id request_setid;
    agps_ril_request_ref_loc request_refloc;
    gps_create_thread create_thread_cb;
} AGpsRilCallbacks;

/** Extended interface for AGPS_RIL support. */
typedef struct {
    /** set to sizeof(AGpsRilInterface) */
    size_t          size;
    /**
     * Opens the AGPS interface and provides the callback routines
     * to the implementation of this interface.
     */
    void  (*init)( AGpsRilCallbacks* callbacks );

    /**
     * Sets the reference location.
     */
    void (*set_ref_location) (const AGpsRefLocation *agps_reflocation, size_t sz_struct);
    /**
     * Sets the set ID.
     */
    void (*set_set_id) (AGpsSetIDType type, const char* setid);

    /**
     * Send network initiated message.
     */
    void (*ni_message) (uint8_t *msg, size_t len);

    /**
     * Notify GPS of network status changes.
     * These parameters match values in the android.net.NetworkInfo class.
     */
    void (*update_network_state) (int connected, int type, int roaming, const char* extra_info);

    /**
     * Notify GPS of network status changes.
     * These parameters match values in the android.net.NetworkInfo class.
     */
    void (*update_network_availability) (int avaiable, const char* apn);
} AGpsRilInterface;

/**
 * GPS Geofence.
 *      There are 3 states associated with a Geofence: Inside, Outside, Unknown.
 * There are 3 transitions: ENTERED, EXITED, UNCERTAIN.
 *
 * An example state diagram with confidence level: 95% and Unknown time limit
 * set as 30 secs is shown below. (confidence level and Unknown time limit are
 * explained latter)
 *                         ____________________________
 *                        |       Unknown (30 secs)   |
 *                         """"""""""""""""""""""""""""
 *                            ^ |                  |  ^
 *                   UNCERTAIN| |ENTERED     EXITED|  |UNCERTAIN
 *                            | v                  v  |
 *                        ________    EXITED     _________
 *                       | Inside | -----------> | Outside |
 *                       |        | <----------- |         |
 *                        """"""""    ENTERED    """""""""
 *
 * Inside state: We are 95% confident that the user is inside the geofence.
 * Outside state: We are 95% confident that the user is outside the geofence
 * Unknown state: Rest of the time.
 *
 * The Unknown state is better explained with an example:
 *
 *                            __________
 *                           |         c|
 *                           |  ___     |    _______
 *                           |  |a|     |   |   b   |
 *                           |  """     |    """""""
 *                           |          |
 *                            """"""""""
 * In the diagram above, "a" and "b" are 2 geofences and "c" is the accuracy
 * circle reported by the GPS subsystem. Now with regard to "b", the system is
 * confident that the user is outside. But with regard to "a" is not confident
 * whether it is inside or outside the geofence. If the accuracy remains the
 * same for a sufficient period of time, the UNCERTAIN transition would be
 * triggered with the state set to Unknown. If the accuracy improves later, an
 * appropriate transition should be triggered.  This "sufficient period of time"
 * is defined by the parameter in the add_geofence_area API.
 *     In other words, Unknown state can be interpreted as a state in which the
 * GPS subsystem isn't confident enough that the user is either inside or
 * outside the Geofence. It moves to Unknown state only after the expiry of the
 * timeout.
 *
 * The geofence callback needs to be triggered for the ENTERED and EXITED
 * transitions, when the GPS system is confident that the user has entered
 * (Inside state) or exited (Outside state) the Geofence. An implementation
 * which uses a value of 95% as the confidence is recommended. The callback
 * should be triggered only for the transitions requested by the
 * add_geofence_area call.
 *
 * Even though the diagram and explanation talks about states and transitions,
 * the callee is only interested in the transistions. The states are mentioned
 * here for illustrative purposes.
 *
 * Startup Scenario: When the device boots up, if an application adds geofences,
 * and then we get an accurate GPS location fix, it needs to trigger the
 * appropriate (ENTERED or EXITED) transition for every Geofence it knows about.
 * By default, all the Geofences will be in the Unknown state.
 *
 * When the GPS system is unavailable, gps_geofence_status_callback should be
 * called to inform the upper layers of the same. Similarly, when it becomes
 * available the callback should be called. This is a global state while the
 * UNKNOWN transition described above is per geofence.
 *
 * An important aspect to note is that users of this API (framework), will use
 * other subsystems like wifi, sensors, cell to handle Unknown case and
 * hopefully provide a definitive state transition to the third party
 * application. GPS Geofence will just be a signal indicating what the GPS
 * subsystem knows about the Geofence.
 *
 */

/**
 * The callback associated with the geofence.
 * Parameters:
 *      geofence_id - The id associated with the add_geofence_area.
 *      location    - The current GPS location.
 *      transition  - Can be one of GPS_GEOFENCE_ENTERED, GPS_GEOFENCE_EXITED,
 *                    GPS_GEOFENCE_UNCERTAIN.
 *      timestamp   - Timestamp when the transition was detected.
 *
 * The callback should only be called when the caller is interested in that
 * particular transition. For instance, if the caller is interested only in
 * ENTERED transition, then the callback should NOT be called with the EXITED
 * transition.
 *
 * IMPORTANT: If a transition is triggered resulting in this callback, the GPS
 * subsystem will wake up the application processor, if its in suspend state.
 */
typedef void (*gps_geofence_transition_callback) (int32_t geofence_id,  GpsLocation* location,
        int32_t transition, GpsUtcTime timestamp);

/**
 * The callback associated with the availability of the GPS system for geofencing
 * monitoring. If the GPS system determines that it cannot monitor geofences
 * because of lack of reliability or unavailability of the GPS signals, it will
 * call this callback with GPS_GEOFENCE_UNAVAILABLE parameter.
 *
 * Parameters:
 *  status - GPS_GEOFENCE_UNAVAILABLE or GPS_GEOFENCE_AVAILABLE.
 *  last_location - Last known location.
 */
typedef void (*gps_geofence_status_callback) (int32_t status, GpsLocation* last_location);

/**
 * The callback associated with the add_geofence call.
 *
 * Parameter:
 * geofence_id - Id of the geofence.
 * status - GPS_GEOFENCE_OPERATION_SUCCESS
 *          GPS_GEOFENCE_ERROR_TOO_MANY_GEOFENCES  - geofence limit has been reached.
 *          GPS_GEOFENCE_ERROR_ID_EXISTS  - geofence with id already exists
 *          GPS_GEOFENCE_ERROR_INVALID_TRANSITION - the monitorTransition contains an
 *              invalid transition
 *          GPS_GEOFENCE_ERROR_GENERIC - for other errors.
 */
typedef void (*gps_geofence_add_callback) (int32_t geofence_id, int32_t status);

/**
 * The callback associated with the remove_geofence call.
 *
 * Parameter:
 * geofence_id - Id of the geofence.
 * status - GPS_GEOFENCE_OPERATION_SUCCESS
 *          GPS_GEOFENCE_ERROR_ID_UNKNOWN - for invalid id
 *          GPS_GEOFENCE_ERROR_GENERIC for others.
 */
typedef void (*gps_geofence_remove_callback) (int32_t geofence_id, int32_t status);


/**
 * The callback associated with the pause_geofence call.
 *
 * Parameter:
 * geofence_id - Id of the geofence.
 * status - GPS_GEOFENCE_OPERATION_SUCCESS
 *          GPS_GEOFENCE_ERROR_ID_UNKNOWN - for invalid id
 *          GPS_GEOFENCE_ERROR_INVALID_TRANSITION -
 *                    when monitor_transitions is invalid
 *          GPS_GEOFENCE_ERROR_GENERIC for others.
 */
typedef void (*gps_geofence_pause_callback) (int32_t geofence_id, int32_t status);

/**
 * The callback associated with the resume_geofence call.
 *
 * Parameter:
 * geofence_id - Id of the geofence.
 * status - GPS_GEOFENCE_OPERATION_SUCCESS
 *          GPS_GEOFENCE_ERROR_ID_UNKNOWN - for invalid id
 *          GPS_GEOFENCE_ERROR_GENERIC for others.
 */
typedef void (*gps_geofence_resume_callback) (int32_t geofence_id, int32_t status);

typedef struct {
    gps_geofence_transition_callback geofence_transition_callback;
    gps_geofence_status_callback geofence_status_callback;
    gps_geofence_add_callback geofence_add_callback;
    gps_geofence_remove_callback geofence_remove_callback;
    gps_geofence_pause_callback geofence_pause_callback;
    gps_geofence_resume_callback geofence_resume_callback;
    gps_create_thread create_thread_cb;
} GpsGeofenceCallbacks;

/** Extended interface for GPS_Geofencing support */
typedef struct {
   /** set to sizeof(GpsGeofencingInterface) */
   size_t          size;

   /**
    * Opens the geofence interface and provides the callback routines
    * to the implementation of this interface.
    */
   void  (*init)( GpsGeofenceCallbacks* callbacks );

   /**
    * Add a geofence area. This api currently supports circular geofences.
    * Parameters:
    *    geofence_id - The id for the geofence. If a geofence with this id
    *       already exists, an error value (GPS_GEOFENCE_ERROR_ID_EXISTS)
    *       should be returned.
    *    latitude, longtitude, radius_meters - The lat, long and radius
    *       (in meters) for the geofence
    *    last_transition - The current state of the geofence. For example, if
    *       the system already knows that the user is inside the geofence,
    *       this will be set to GPS_GEOFENCE_ENTERED. In most cases, it
    *       will be GPS_GEOFENCE_UNCERTAIN.
    *    monitor_transition - Which transitions to monitor. Bitwise OR of
    *       GPS_GEOFENCE_ENTERED, GPS_GEOFENCE_EXITED and
    *       GPS_GEOFENCE_UNCERTAIN.
    *    notification_responsiveness_ms - Defines the best-effort description
    *       of how soon should the callback be called when the transition
    *       associated with the Geofence is triggered. For instance, if set
    *       to 1000 millseconds with GPS_GEOFENCE_ENTERED, the callback
    *       should be called 1000 milliseconds within entering the geofence.
    *       This parameter is defined in milliseconds.
    *       NOTE: This is not to be confused with the rate that the GPS is
    *       polled at. It is acceptable to dynamically vary the rate of
    *       sampling the GPS for power-saving reasons; thus the rate of
    *       sampling may be faster or slower than this.
    *    unknown_timer_ms - The time limit after which the UNCERTAIN transition
    *       should be triggered. This parameter is defined in milliseconds.
    *       See above for a detailed explanation.
    */
   void (*add_geofence_area) (int32_t geofence_id, double latitude, double longitude,
       double radius_meters, int last_transition, int monitor_transitions,
       int notification_responsiveness_ms, int unknown_timer_ms);

   /**
    * Pause monitoring a particular geofence.
    * Parameters:
    *   geofence_id - The id for the geofence.
    */
   void (*pause_geofence) (int32_t geofence_id);

   /**
    * Resume monitoring a particular geofence.
    * Parameters:
    *   geofence_id - The id for the geofence.
    *   monitor_transitions - Which transitions to monitor. Bitwise OR of
    *       GPS_GEOFENCE_ENTERED, GPS_GEOFENCE_EXITED and
    *       GPS_GEOFENCE_UNCERTAIN.
    *       This supersedes the value associated provided in the
    *       add_geofence_area call.
    */
   void (*resume_geofence) (int32_t geofence_id, int monitor_transitions);

   /**
    * Remove a geofence area. After the function returns, no notifications
    * should be sent.
    * Parameter:
    *   geofence_id - The id for the geofence.
    */
   void (*remove_geofence_area) (int32_t geofence_id);
} GpsGeofencingInterface;

/**
 * Legacy struct to represent an estimate of the GPS clock time.
 * Deprecated, to be removed in the next Android release.
 * Use GnssClock instead.
 */
typedef struct {
    /** set to sizeof(GpsClock) */
    size_t size;
    GpsClockFlags flags;
    int16_t leap_second;
    GpsClockType type;
    int64_t time_ns;
    double time_uncertainty_ns;
    int64_t full_bias_ns;
    double bias_ns;
    double bias_uncertainty_ns;
    double drift_nsps;
    double drift_uncertainty_nsps;
} GpsClock;

/**
 * Represents an estimate of the GPS clock time.
 */
typedef struct {
    /** set to sizeof(GnssClock) */
    size_t size;

    /**
     * A set of flags indicating the validity of the fields in this data
     * structure.
     */
    GnssClockFlags flags;

    /**
     * Leap second data.
     * The sign of the value is defined by the following equation:
     *      utc_time_ns = time_ns - (full_bias_ns + bias_ns) - leap_second *
     *      1,000,000,000
     *
     * If the data is available 'flags' must contain GNSS_CLOCK_HAS_LEAP_SECOND.
     */
    int16_t leap_second;

    /**
     * The GNSS receiver internal clock value. This is the local hardware clock
     * value.
     *
     * For local hardware clock, this value is expected to be monotonically
     * increasing while the hardware clock remains power on. (For the case of a
     * HW clock that is not continuously on, see the
     * hw_clock_discontinuity_count field). The receiver's estimate of GPS time
     * can be derived by substracting the sum of full_bias_ns and bias_ns (when
     * available) from this value.
     *
     * This GPS time is expected to be the best estimate of current GPS time
     * that GNSS receiver can achieve.
     *
     * Sub-nanosecond accuracy can be provided by means of the 'bias_ns' field.
     * The value contains the 'time uncertainty' in it.
     *
     * This field is mandatory.
     */
    int64_t time_ns;

    /**
     * 1-Sigma uncertainty associated with the clock's time in nanoseconds.
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * If the data is available, 'flags' must contain
     * GNSS_CLOCK_HAS_TIME_UNCERTAINTY. This value is effectively zero (it is
     * the reference local clock, by which all other times and time
     * uncertainties are measured.)  (And thus this field can be not provided,
     * per GNSS_CLOCK_HAS_TIME_UNCERTAINTY flag, or provided & set to 0.)
     */
    double time_uncertainty_ns;

    /**
     * The difference between hardware clock ('time' field) inside GPS receiver
     * and the true GPS time since 0000Z, January 6, 1980, in nanoseconds.
     *
     * The sign of the value is defined by the following equation:
     *      local estimate of GPS time = time_ns - (full_bias_ns + bias_ns)
     *
     * This value is mandatory if the receiver has estimated GPS time. If the
     * computed time is for a non-GPS constellation, the time offset of that
     * constellation to GPS has to be applied to fill this value. The error
     * estimate for the sum of this and the bias_ns is the bias_uncertainty_ns,
     * and the caller is responsible for using this uncertainty (it can be very
     * large before the GPS time has been solved for.) If the data is available
     * 'flags' must contain GNSS_CLOCK_HAS_FULL_BIAS.
     */
    int64_t full_bias_ns;

    /**
     * Sub-nanosecond bias.
     * The error estimate for the sum of this and the full_bias_ns is the
     * bias_uncertainty_ns
     *
     * If the data is available 'flags' must contain GNSS_CLOCK_HAS_BIAS. If GPS
     * has computed a position fix. This value is mandatory if the receiver has
     * estimated GPS time.
     */
    double bias_ns;

    /**
     * 1-Sigma uncertainty associated with the local estimate of GPS time (clock
     * bias) in nanoseconds. The uncertainty is represented as an absolute
     * (single sided) value.
     *
     * If the data is available 'flags' must contain
     * GNSS_CLOCK_HAS_BIAS_UNCERTAINTY. This value is mandatory if the receiver
     * has estimated GPS time.
     */
    double bias_uncertainty_ns;

    /**
     * The clock's drift in nanoseconds (per second).
     *
     * A positive value means that the frequency is higher than the nominal
     * frequency, and that the (full_bias_ns + bias_ns) is growing more positive
     * over time.
     *
     * The value contains the 'drift uncertainty' in it.
     * If the data is available 'flags' must contain GNSS_CLOCK_HAS_DRIFT.
     *
     * This value is mandatory if the receiver has estimated GNSS time
     */
    double drift_nsps;

    /**
     * 1-Sigma uncertainty associated with the clock's drift in nanoseconds (per second).
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * If the data is available 'flags' must contain
     * GNSS_CLOCK_HAS_DRIFT_UNCERTAINTY. If GPS has computed a position fix this
     * field is mandatory and must be populated.
     */
    double drift_uncertainty_nsps;

    /**
     * When there are any discontinuities in the HW clock, this field is
     * mandatory.
     *
     * A "discontinuity" is meant to cover the case of a switch from one source
     * of clock to another.  A single free-running crystal oscillator (XO)
     * should generally not have any discontinuities, and this can be set and
     * left at 0.
     *
     * If, however, the time_ns value (HW clock) is derived from a composite of
     * sources, that is not as smooth as a typical XO, or is otherwise stopped &
     * restarted, then this value shall be incremented each time a discontinuity
     * occurs.  (E.g. this value may start at zero at device boot-up and
     * increment each time there is a change in clock continuity. In the
     * unlikely event that this value reaches full scale, rollover (not
     * clamping) is required, such that this value continues to change, during
     * subsequent discontinuity events.)
     *
     * While this number stays the same, between GnssClock reports, it can be
     * safely assumed that the time_ns value has been running continuously, e.g.
     * derived from a single, high quality clock (XO like, or better, that's
     * typically used during continuous GNSS signal sampling.)
     *
     * It is expected, esp. during periods where there are few GNSS signals
     * available, that the HW clock be discontinuity-free as long as possible,
     * as this avoids the need to use (waste) a GNSS measurement to fully
     * re-solve for the GPS clock bias and drift, when using the accompanying
     * measurements, from consecutive GnssData reports.
     */
    uint32_t hw_clock_discontinuity_count;

} GnssClock;

/**
 * Legacy struct to represent a GPS Measurement, it contains raw and computed
 * information.
 * Deprecated, to be removed in the next Android release.
 * Use GnssMeasurement instead.
 */
typedef struct {
    /** set to sizeof(GpsMeasurement) */
    size_t size;
    GpsMeasurementFlags flags;
    int8_t prn;
    double time_offset_ns;
    GpsMeasurementState state;
    int64_t received_gps_tow_ns;
    int64_t received_gps_tow_uncertainty_ns;
    double c_n0_dbhz;
    double pseudorange_rate_mps;
    double pseudorange_rate_uncertainty_mps;
    GpsAccumulatedDeltaRangeState accumulated_delta_range_state;
    double accumulated_delta_range_m;
    double accumulated_delta_range_uncertainty_m;
    double pseudorange_m;
    double pseudorange_uncertainty_m;
    double code_phase_chips;
    double code_phase_uncertainty_chips;
    float carrier_frequency_hz;
    int64_t carrier_cycles;
    double carrier_phase;
    double carrier_phase_uncertainty;
    GpsLossOfLock loss_of_lock;
    int32_t bit_number;
    int16_t time_from_last_bit_ms;
    double doppler_shift_hz;
    double doppler_shift_uncertainty_hz;
    GpsMultipathIndicator multipath_indicator;
    double snr_db;
    double elevation_deg;
    double elevation_uncertainty_deg;
    double azimuth_deg;
    double azimuth_uncertainty_deg;
    bool used_in_fix;
} GpsMeasurement;

/**
 * Represents a GNSS Measurement, it contains raw and computed information.
 *
 * Independence - All signal measurement information (e.g. sv_time,
 * pseudorange_rate, multipath_indicator) reported in this struct should be
 * based on GNSS signal measurements only. You may not synthesize measurements
 * by calculating or reporting expected measurements based on known or estimated
 * position, velocity, or time.
 */
typedef struct {
    /** set to sizeof(GpsMeasurement) */
    size_t size;

    /** A set of flags indicating the validity of the fields in this data structure. */
    GnssMeasurementFlags flags;

    /**
     * Satellite vehicle ID number, as defined in GnssSvInfo::svid
     * This is a mandatory value.
     */
    int16_t svid;

    /**
     * Defines the constellation of the given SV. Value should be one of those
     * GNSS_CONSTELLATION_* constants
     */
    GnssConstellationType constellation;

    /**
     * Time offset at which the measurement was taken in nanoseconds.
     * The reference receiver's time is specified by GpsData::clock::time_ns and should be
     * interpreted in the same way as indicated by GpsClock::type.
     *
     * The sign of time_offset_ns is given by the following equation:
     *      measurement time = GpsClock::time_ns + time_offset_ns
     *
     * It provides an individual time-stamp for the measurement, and allows sub-nanosecond accuracy.
     * This is a mandatory value.
     */
    double time_offset_ns;

    /**
     * Per satellite sync state. It represents the current sync state for the associated satellite.
     * Based on the sync state, the 'received GPS tow' field should be interpreted accordingly.
     *
     * This is a mandatory value.
     */
    GnssMeasurementState state;

    /**
     * The received GNSS Time-of-Week at the measurement time, in nanoseconds.
     * Ensure that this field is independent (see comment at top of
     * GnssMeasurement struct.)
     *
     * For GPS & QZSS, this is:
     *   Received GPS Time-of-Week at the measurement time, in nanoseconds.
     *   The value is relative to the beginning of the current GPS week.
     *
     *   Given the highest sync state that can be achieved, per each satellite, valid range
     *   for this field can be:
     *     Searching       : [ 0       ]   : GNSS_MEASUREMENT_STATE_UNKNOWN
     *     C/A code lock   : [ 0   1ms ]   : GNSS_MEASUREMENT_STATE_CODE_LOCK is set
     *     Bit sync        : [ 0  20ms ]   : GNSS_MEASUREMENT_STATE_BIT_SYNC is set
     *     Subframe sync   : [ 0    6s ]   : GNSS_MEASUREMENT_STATE_SUBFRAME_SYNC is set
     *     TOW decoded     : [ 0 1week ]   : GNSS_MEASUREMENT_STATE_TOW_DECODED is set
     *
     *   Note well: if there is any ambiguity in integer millisecond,
     *   GNSS_MEASUREMENT_STATE_MSEC_AMBIGUOUS should be set accordingly, in the 'state' field.
     *
     *   This value must be populated if 'state' != GNSS_MEASUREMENT_STATE_UNKNOWN.
     *
     * For Glonass, this is:
     *   Received Glonass time of day, at the measurement time in nanoseconds.
     *
     *   Given the highest sync state that can be achieved, per each satellite, valid range for
     *   this field can be:
     *     Searching       : [ 0       ]   : GNSS_MEASUREMENT_STATE_UNKNOWN
     *     C/A code lock   : [ 0   1ms ]   : GNSS_MEASUREMENT_STATE_CODE_LOCK is set
     *     Symbol sync     : [ 0  10ms ]   : GNSS_MEASUREMENT_STATE_SYMBOL_SYNC is set
     *     Bit sync        : [ 0  20ms ]   : GNSS_MEASUREMENT_STATE_BIT_SYNC is set
     *     String sync     : [ 0    2s ]   : GNSS_MEASUREMENT_STATE_GLO_STRING_SYNC is set
     *     Time of day     : [ 0  1day ]   : GNSS_MEASUREMENT_STATE_GLO_TOD_DECODED is set
     *
     * For Beidou, this is:
     *   Received Beidou time of week, at the measurement time in nanoseconds.
     *
     *   Given the highest sync state that can be achieved, per each satellite, valid range for
     *   this field can be:
     *     Searching    : [ 0       ] : GNSS_MEASUREMENT_STATE_UNKNOWN
     *     C/A code lock: [ 0   1ms ] : GNSS_MEASUREMENT_STATE_CODE_LOCK is set
     *     Bit sync (D2): [ 0   2ms ] : GNSS_MEASUREMENT_STATE_BDS_D2_BIT_SYNC is set
     *     Bit sync (D1): [ 0  20ms ] : GNSS_MEASUREMENT_STATE_BIT_SYNC is set
     *     Subframe (D2): [ 0  0.6s ] : GNSS_MEASUREMENT_STATE_BDS_D2_SUBFRAME_SYNC is set
     *     Subframe (D1): [ 0    6s ] : GNSS_MEASUREMENT_STATE_SUBFRAME_SYNC is set
     *     Time of week : [ 0 1week ] : GNSS_MEASUREMENT_STATE_TOW_DECODED is set
     *
     * For Galileo, this is:
     *   Received Galileo time of week, at the measurement time in nanoseconds.
     *
     *     E1BC code lock   : [ 0   4ms ]   : GNSS_MEASUREMENT_STATE_GAL_E1BC_CODE_LOCK is set
     *     E1C 2nd code lock: [ 0 100ms ]   :
     *     GNSS_MEASUREMENT_STATE_GAL_E1C_2ND_CODE_LOCK is set
     *
     *     E1B page    : [ 0    2s ] : GNSS_MEASUREMENT_STATE_GAL_E1B_PAGE_SYNC is set
     *     Time of week: [ 0 1week ] : GNSS_MEASUREMENT_STATE_TOW_DECODED is set
     *
     * For SBAS, this is:
     *   Received SBAS time, at the measurement time in nanoseconds.
     *
     *   Given the highest sync state that can be achieved, per each satellite,
     *   valid range for this field can be:
     *     Searching    : [ 0     ] : GNSS_MEASUREMENT_STATE_UNKNOWN
     *     C/A code lock: [ 0 1ms ] : GNSS_MEASUREMENT_STATE_CODE_LOCK is set
     *     Symbol sync  : [ 0 2ms ] : GNSS_MEASUREMENT_STATE_SYMBOL_SYNC is set
     *     Message      : [ 0  1s ] : GNSS_MEASUREMENT_STATE_SBAS_SYNC is set
    */
    int64_t received_sv_time_in_ns;

    /**
     * 1-Sigma uncertainty of the Received GPS Time-of-Week in nanoseconds.
     *
     * This value must be populated if 'state' != GPS_MEASUREMENT_STATE_UNKNOWN.
     */
    int64_t received_sv_time_uncertainty_in_ns;

    /**
     * Carrier-to-noise density in dB-Hz, typically in the range [0, 63].
     * It contains the measured C/N0 value for the signal at the antenna port.
     *
     * This is a mandatory value.
     */
    double c_n0_dbhz;

    /**
     * Pseudorange rate at the timestamp in m/s. The correction of a given
     * Pseudorange Rate value includes corrections for receiver and satellite
     * clock frequency errors. Ensure that this field is independent (see
     * comment at top of GnssMeasurement struct.)
     *
     * It is mandatory to provide the 'uncorrected' 'pseudorange rate', and provide GpsClock's
     * 'drift' field as well (When providing the uncorrected pseudorange rate, do not apply the
     * corrections described above.)
     *
     * The value includes the 'pseudorange rate uncertainty' in it.
     * A positive 'uncorrected' value indicates that the SV is moving away from the receiver.
     *
     * The sign of the 'uncorrected' 'pseudorange rate' and its relation to the sign of 'doppler
     * shift' is given by the equation:
     *      pseudorange rate = -k * doppler shift   (where k is a constant)
     *
     * This should be the most accurate pseudorange rate available, based on
     * fresh signal measurements from this channel.
     *
     * It is mandatory that this value be provided at typical carrier phase PRR
     * quality (few cm/sec per second of uncertainty, or better) - when signals
     * are sufficiently strong & stable, e.g. signals from a GPS simulator at >=
     * 35 dB-Hz.
     */
    double pseudorange_rate_mps;

    /**
     * 1-Sigma uncertainty of the pseudorange_rate_mps.
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * This is a mandatory value.
     */
    double pseudorange_rate_uncertainty_mps;

    /**
     * Accumulated delta range's state. It indicates whether ADR is reset or there is a cycle slip
     * (indicating loss of lock).
     *
     * This is a mandatory value.
     */
    GnssAccumulatedDeltaRangeState accumulated_delta_range_state;

    /**
     * Accumulated delta range since the last channel reset in meters.
     * A positive value indicates that the SV is moving away from the receiver.
     *
     * The sign of the 'accumulated delta range' and its relation to the sign of 'carrier phase'
     * is given by the equation:
     *          accumulated delta range = -k * carrier phase    (where k is a constant)
     *
     * This value must be populated if 'accumulated delta range state' != GPS_ADR_STATE_UNKNOWN.
     * However, it is expected that the data is only accurate when:
     *      'accumulated delta range state' == GPS_ADR_STATE_VALID.
     */
    double accumulated_delta_range_m;

    /**
     * 1-Sigma uncertainty of the accumulated delta range in meters.
     * This value must be populated if 'accumulated delta range state' != GPS_ADR_STATE_UNKNOWN.
     */
    double accumulated_delta_range_uncertainty_m;

    /**
     * Carrier frequency at which codes and messages are modulated, it can be L1 or L2.
     * If the field is not set, the carrier frequency is assumed to be L1.
     *
     * If the data is available, 'flags' must contain
     * GNSS_MEASUREMENT_HAS_CARRIER_FREQUENCY.
     */
    float carrier_frequency_hz;

    /**
     * The number of full carrier cycles between the satellite and the receiver.
     * The reference frequency is given by the field 'carrier_frequency_hz'.
     * Indications of possible cycle slips and resets in the accumulation of
     * this value can be inferred from the accumulated_delta_range_state flags.
     *
     * If the data is available, 'flags' must contain
     * GNSS_MEASUREMENT_HAS_CARRIER_CYCLES.
     */
    int64_t carrier_cycles;

    /**
     * The RF phase detected by the receiver, in the range [0.0, 1.0].
     * This is usually the fractional part of the complete carrier phase measurement.
     *
     * The reference frequency is given by the field 'carrier_frequency_hz'.
     * The value contains the 'carrier-phase uncertainty' in it.
     *
     * If the data is available, 'flags' must contain
     * GNSS_MEASUREMENT_HAS_CARRIER_PHASE.
     */
    double carrier_phase;

    /**
     * 1-Sigma uncertainty of the carrier-phase.
     * If the data is available, 'flags' must contain
     * GNSS_MEASUREMENT_HAS_CARRIER_PHASE_UNCERTAINTY.
     */
    double carrier_phase_uncertainty;

    /**
     * An enumeration that indicates the 'multipath' state of the event.
     *
     * The multipath Indicator is intended to report the presence of overlapping
     * signals that manifest as distorted correlation peaks.
     *
     * - if there is a distorted correlation peak shape, report that multipath
     *   is GNSS_MULTIPATH_INDICATOR_PRESENT.
     * - if there is not a distorted correlation peak shape, report
     *   GNSS_MULTIPATH_INDICATOR_NOT_PRESENT
     * - if signals are too weak to discern this information, report
     *   GNSS_MULTIPATH_INDICATOR_UNKNOWN
     *
     * Example: when doing the standardized overlapping Multipath Performance
     * test (3GPP TS 34.171) the Multipath indicator should report
     * GNSS_MULTIPATH_INDICATOR_PRESENT for those signals that are tracked, and
     * contain multipath, and GNSS_MULTIPATH_INDICATOR_NOT_PRESENT for those
     * signals that are tracked and do not contain multipath.
     */
    GnssMultipathIndicator multipath_indicator;

    /**
     * Signal-to-noise ratio at correlator output in dB.
     * If the data is available, 'flags' must contain GNSS_MEASUREMENT_HAS_SNR.
     * This is the power ratio of the "correlation peak height above the
     * observed noise floor" to "the noise RMS".
     */
    double snr_db;
} GnssMeasurement;

/**
 * Legacy struct to represents a reading of GPS measurements.
 * Deprecated, to be removed in the next Android release.
 * Use GnssData instead.
 */
typedef struct {
    /** set to sizeof(GpsData) */
    size_t size;
    size_t measurement_count;
    GpsMeasurement measurements[GPS_MAX_MEASUREMENT];

    /** The GPS clock time reading. */
    GpsClock clock;
} GpsData;

/**
 * Represents a reading of GNSS measurements. For devices where GnssSystemInfo's
 * year_of_hw is set to 2016+, it is mandatory that these be provided, on
 * request, when the GNSS receiver is searching/tracking signals.
 *
 * - Reporting of GPS constellation measurements is mandatory.
 * - Reporting of all tracked constellations are encouraged.
 */
typedef struct {
    /** set to sizeof(GnssData) */
    size_t size;

    /** Number of measurements. */
    size_t measurement_count;

    /** The array of measurements. */
    GnssMeasurement measurements[GNSS_MAX_MEASUREMENT];

    /** The GPS clock time reading. */
    GnssClock clock;
} GnssData;

/**
 * The legacy callback for to report measurements from the HAL.
 *
 * This callback is deprecated, and will be removed in the next release. Use
 * gnss_measurement_callback() instead.
 *
 * Parameters:
 *    data - A data structure containing the measurements.
 */
typedef void (*gps_measurement_callback) (GpsData* data);

/**
 * The callback for to report measurements from the HAL.
 *
 * Parameters:
 *    data - A data structure containing the measurements.
 */
typedef void (*gnss_measurement_callback) (GnssData* data);

typedef struct {
    /** set to sizeof(GpsMeasurementCallbacks) */
    size_t size;
    gps_measurement_callback measurement_callback;
    gnss_measurement_callback gnss_measurement_callback;
} GpsMeasurementCallbacks;

/**
 * Extended interface for GPS Measurements support.
 */
typedef struct {
    /** Set to sizeof(GpsMeasurementInterface) */
    size_t size;

    /**
     * Initializes the interface and registers the callback routines with the HAL.
     * After a successful call to 'init' the HAL must begin to provide updates at its own phase.
     *
     * Status:
     *    GPS_MEASUREMENT_OPERATION_SUCCESS
     *    GPS_MEASUREMENT_ERROR_ALREADY_INIT - if a callback has already been registered without a
     *              corresponding call to 'close'
     *    GPS_MEASUREMENT_ERROR_GENERIC - if any other error occurred, it is expected that the HAL
     *              will not generate any updates upon returning this error code.
     */
    int (*init) (GpsMeasurementCallbacks* callbacks);

    /**
     * Stops updates from the HAL, and unregisters the callback routines.
     * After a call to stop, the previously registered callbacks must be considered invalid by the
     * HAL.
     * If stop is invoked without a previous 'init', this function should perform no work.
     */
    void (*close) ();

} GpsMeasurementInterface;

/**
 * Legacy struct to represents a GPS navigation message (or a fragment of it).
 * Deprecated, to be removed in the next Android release.
 * Use GnssNavigationMessage instead.
 */
typedef struct {
    /** set to sizeof(GpsNavigationMessage) */
    size_t size;
    int8_t prn;
    GpsNavigationMessageType type;
    NavigationMessageStatus status;
    int16_t message_id;
    int16_t submessage_id;
    size_t data_length;
    uint8_t* data;
} GpsNavigationMessage;

/** Represents a GPS navigation message (or a fragment of it). */
typedef struct {
    /** set to sizeof(GnssNavigationMessage) */
    size_t size;

    /**
     * Satellite vehicle ID number, as defined in GnssSvInfo::svid
     * This is a mandatory value.
     */
    int16_t svid;

    /**
     * The type of message contained in the structure.
     * This is a mandatory value.
     */
    GnssNavigationMessageType type;

    /**
     * The status of the received navigation message.
     * No need to send any navigation message that contains words with parity error and cannot be
     * corrected.
     */
    NavigationMessageStatus status;

    /**
     * Message identifier. It provides an index so the complete Navigation
     * Message can be assembled.
     *
     * - For GPS L1 C/A subframe 4 and 5, this value corresponds to the 'frame
     *   id' of the navigation message, in the range of 1-25 (Subframe 1, 2, 3
     *   does not contain a 'frame id' and this value can be set to -1.)
     *
     * - For Glonass L1 C/A, this refers to the frame ID, in the range of 1-5.
     *
     * - For BeiDou D1, this refers to the frame number in the range of 1-24
     *
     * - For Beidou D2, this refers to the frame number, in the range of 1-120
     *
     * - For Galileo F/NAV nominal frame structure, this refers to the subframe
     *   number, in the range of 1-12
     *
     * - For Galileo I/NAV nominal frame structure, this refers to the subframe
     *   number in the range of 1-24
     */
    int16_t message_id;

    /**
     * Sub-message identifier. If required by the message 'type', this value
     * contains a sub-index within the current message (or frame) that is being
     * transmitted.
     *
     * - For GPS L1 C/A, BeiDou D1 & BeiDou D2, the submessage id corresponds to
     *   the subframe number of the navigation message, in the range of 1-5.
     *
     * - For Glonass L1 C/A, this refers to the String number, in the range from
     *   1-15
     *
     * - For Galileo F/NAV, this refers to the page type in the range 1-6
     *
     * - For Galileo I/NAV, this refers to the word type in the range 1-10+
     */
    int16_t submessage_id;

    /**
     * The length of the data (in bytes) contained in the current message.
     * If this value is different from zero, 'data' must point to an array of the same size.
     * e.g. for L1 C/A the size of the sub-frame will be 40 bytes (10 words, 30 bits/word).
     *
     * This is a mandatory value.
     */
    size_t data_length;

    /**
     * The data of the reported GPS message. The bytes (or words) specified
     * using big endian format (MSB first).
     *
     * - For GPS L1 C/A, Beidou D1 & Beidou D2, each subframe contains 10 30-bit
     *   words. Each word (30 bits) should be fit into the last 30 bits in a
     *   4-byte word (skip B31 and B32), with MSB first, for a total of 40
     *   bytes, covering a time period of 6, 6, and 0.6 seconds, respectively.
     *
     * - For Glonass L1 C/A, each string contains 85 data bits, including the
     *   checksum.  These bits should be fit into 11 bytes, with MSB first (skip
     *   B86-B88), covering a time period of 2 seconds.
     *
     * - For Galileo F/NAV, each word consists of 238-bit (sync & tail symbols
     *   excluded). Each word should be fit into 30-bytes, with MSB first (skip
     *   B239, B240), covering a time period of 10 seconds.
     *
     * - For Galileo I/NAV, each page contains 2 page parts, even and odd, with
     *   a total of 2x114 = 228 bits, (sync & tail excluded) that should be fit
     *   into 29 bytes, with MSB first (skip B229-B232).
     */
    uint8_t* data;

} GnssNavigationMessage;

/**
 * The legacy callback to report an available fragment of a GPS navigation
 * messages from the HAL.
 *
 * This callback is deprecated, and will be removed in the next release. Use
 * gnss_navigation_message_callback() instead.
 *
 * Parameters:
 *      message - The GPS navigation submessage/subframe representation.
 */
typedef void (*gps_navigation_message_callback) (GpsNavigationMessage* message);

/**
 * The callback to report an available fragment of a GPS navigation messages from the HAL.
 *
 * Parameters:
 *      message - The GPS navigation submessage/subframe representation.
 */
typedef void (*gnss_navigation_message_callback) (GnssNavigationMessage* message);

typedef struct {
    /** set to sizeof(GpsNavigationMessageCallbacks) */
    size_t size;
    gps_navigation_message_callback navigation_message_callback;
    gnss_navigation_message_callback gnss_navigation_message_callback;
} GpsNavigationMessageCallbacks;

/**
 * Extended interface for GPS navigation message reporting support.
 */
typedef struct {
    /** Set to sizeof(GpsNavigationMessageInterface) */
    size_t size;

    /**
     * Initializes the interface and registers the callback routines with the HAL.
     * After a successful call to 'init' the HAL must begin to provide updates as they become
     * available.
     *
     * Status:
     *      GPS_NAVIGATION_MESSAGE_OPERATION_SUCCESS
     *      GPS_NAVIGATION_MESSAGE_ERROR_ALREADY_INIT - if a callback has already been registered
     *              without a corresponding call to 'close'.
     *      GPS_NAVIGATION_MESSAGE_ERROR_GENERIC - if any other error occurred, it is expected that
     *              the HAL will not generate any updates upon returning this error code.
     */
    int (*init) (GpsNavigationMessageCallbacks* callbacks);

    /**
     * Stops updates from the HAL, and unregisters the callback routines.
     * After a call to stop, the previously registered callbacks must be considered invalid by the
     * HAL.
     * If stop is invoked without a previous 'init', this function should perform no work.
     */
    void (*close) ();

} GpsNavigationMessageInterface;

/**
 * Interface for passing GNSS configuration contents from platform to HAL.
 */
typedef struct {
    /** Set to sizeof(GnssConfigurationInterface) */
    size_t size;

    /**
     * Deliver GNSS configuration contents to HAL.
     * Parameters:
     *     config_data - a pointer to a char array which holds what usually is expected from
                         file(/etc/gps.conf), i.e., a sequence of UTF8 strings separated by '\n'.
     *     length - total number of UTF8 characters in configuraiton data.
     *
     * IMPORTANT:
     *      GPS HAL should expect this function can be called multiple times. And it may be
     *      called even when GpsLocationProvider is already constructed and enabled. GPS HAL
     *      should maintain the existing requests for various callback regardless the change
     *      in configuration data.
     */
    void (*configuration_update) (const char* config_data, int32_t length);
} GnssConfigurationInterface;

__END_DECLS

#endif /* ANDROID_INCLUDE_HARDWARE_GPS_H */

