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

__BEGIN_DECLS

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
// IMPORTANT: Note that the following values must match
// constants in GpsLocationProvider.java.
/** Mode for running GPS standalone (no assistance). */
#define GPS_POSITION_MODE_STANDALONE    0
/** AGPS MS-Based mode. */
#define GPS_POSITION_MODE_MS_BASED      1
/**
 * AGPS MS-Assisted mode. This mode is not maintained by the platform anymore.
 * It is strongly recommended to use GPS_POSITION_MODE_MS_BASE instead.
 */
#define GPS_POSITION_MODE_MS_ASSISTED   2

/** Requested recurrence mode for GPS operation. */
typedef uint32_t GpsPositionRecurrence;
// IMPORTANT: Note that the following values must match
// constants in GpsLocationProvider.java.
/** Receive GPS fixes on a recurring basis at a specified period. */
#define GPS_POSITION_RECURRENCE_PERIODIC    0
/** Request a single shot GPS fix. */
#define GPS_POSITION_RECURRENCE_SINGLE      1

/** GPS status event values. */
typedef uint16_t GpsStatusValue;
// IMPORTANT: Note that the following values must match
// constants in GpsLocationProvider.java.
/** GPS status unknown. */
#define GPS_STATUS_NONE             0
/** GPS has begun navigating. */
#define GPS_STATUS_SESSION_BEGIN    1
/** GPS has stopped navigating. */
#define GPS_STATUS_SESSION_END      2
/** GPS has powered on but is not navigating. */
#define GPS_STATUS_ENGINE_ON        3
/** GPS is powered off. */
#define GPS_STATUS_ENGINE_OFF       4

/** Flags to indicate which values are valid in a GpsLocation. */
typedef uint16_t GpsLocationFlags;
// IMPORTANT: Note that the following values must match
// constants in GpsLocationProvider.java.
/** GpsLocation has valid latitude and longitude. */
#define GPS_LOCATION_HAS_LAT_LONG   0x0001
/** GpsLocation has valid altitude. */
#define GPS_LOCATION_HAS_ALTITUDE   0x0002
/** GpsLocation has valid speed. */
#define GPS_LOCATION_HAS_SPEED      0x0004
/** GpsLocation has valid bearing. */
#define GPS_LOCATION_HAS_BEARING    0x0008
/** GpsLocation has valid accuracy. */
#define GPS_LOCATION_HAS_ACCURACY   0x0010

/** Flags for the gps_set_capabilities callback. */

/** GPS HAL schedules fixes for GPS_POSITION_RECURRENCE_PERIODIC mode.
    If this is not set, then the framework will use 1000ms for min_interval
    and will start and call start() and stop() to schedule the GPS.
 */
#define GPS_CAPABILITY_SCHEDULING       0x0000001
/** GPS supports MS-Based AGPS mode */
#define GPS_CAPABILITY_MSB              0x0000002
/** GPS supports MS-Assisted AGPS mode */
#define GPS_CAPABILITY_MSA              0x0000004
/** GPS supports single-shot fixes */
#define GPS_CAPABILITY_SINGLE_SHOT      0x0000008
/** GPS supports on demand time injection */
#define GPS_CAPABILITY_ON_DEMAND_TIME   0x0000010
/** GPS supports Geofencing  */
#define GPS_CAPABILITY_GEOFENCING       0x0000020
/** GPS supports Measurements */
#define GPS_CAPABILITY_MEASUREMENTS     0x0000040
/** GPS supports Navigation Messages */
#define GPS_CAPABILITY_NAV_MESSAGES     0x0000080

/** Flags used to specify which aiding data to delete
    when calling delete_aiding_data(). */
typedef uint16_t GpsAidingData;
// IMPORTANT: Note that the following values must match
// constants in GpsLocationProvider.java.
#define GPS_DELETE_EPHEMERIS        0x0001
#define GPS_DELETE_ALMANAC          0x0002
#define GPS_DELETE_POSITION         0x0004
#define GPS_DELETE_TIME             0x0008
#define GPS_DELETE_IONO             0x0010
#define GPS_DELETE_UTC              0x0020
#define GPS_DELETE_HEALTH           0x0040
#define GPS_DELETE_SVDIR            0x0080
#define GPS_DELETE_SVSTEER          0x0100
#define GPS_DELETE_SADATA           0x0200
#define GPS_DELETE_RTI              0x0400
#define GPS_DELETE_CELLDB_INFO      0x8000
#define GPS_DELETE_ALL              0xFFFF

/** AGPS type */
typedef uint16_t AGpsType;
#define AGPS_TYPE_SUPL          1
#define AGPS_TYPE_C2K           2

typedef uint16_t AGpsSetIDType;
#define AGPS_SETID_TYPE_NONE    0
#define AGPS_SETID_TYPE_IMSI    1
#define AGPS_SETID_TYPE_MSISDN  2

typedef uint16_t ApnIpType;
#define APN_IP_INVALID          0
#define APN_IP_IPV4             1
#define APN_IP_IPV6             2
#define APN_IP_IPV4V6           3

/**
 * String length constants
 */
#define GPS_NI_SHORT_STRING_MAXLEN      256
#define GPS_NI_LONG_STRING_MAXLEN       2048

/**
 * GpsNiType constants
 */
typedef uint32_t GpsNiType;
#define GPS_NI_TYPE_VOICE              1
#define GPS_NI_TYPE_UMTS_SUPL          2
#define GPS_NI_TYPE_UMTS_CTRL_PLANE    3

/**
 * GpsNiNotifyFlags constants
 */
typedef uint32_t GpsNiNotifyFlags;
/** NI requires notification */
#define GPS_NI_NEED_NOTIFY          0x0001
/** NI requires verification */
#define GPS_NI_NEED_VERIFY          0x0002
/** NI requires privacy override, no notification/minimal trace */
#define GPS_NI_PRIVACY_OVERRIDE     0x0004

/**
 * GPS NI responses, used to define the response in
 * NI structures
 */
typedef int GpsUserResponseType;
#define GPS_NI_RESPONSE_ACCEPT         1
#define GPS_NI_RESPONSE_DENY           2
#define GPS_NI_RESPONSE_NORESP         3

/**
 * NI data encoding scheme
 */
typedef int GpsNiEncodingType;
#define GPS_ENC_NONE                   0
#define GPS_ENC_SUPL_GSM_DEFAULT       1
#define GPS_ENC_SUPL_UTF8              2
#define GPS_ENC_SUPL_UCS2              3
#define GPS_ENC_UNKNOWN                -1

/** AGPS status event values. */
typedef uint16_t AGpsStatusValue;
/** GPS requests data connection for AGPS. */
#define GPS_REQUEST_AGPS_DATA_CONN  1
/** GPS releases the AGPS data connection. */
#define GPS_RELEASE_AGPS_DATA_CONN  2
/** AGPS data connection initiated */
#define GPS_AGPS_DATA_CONNECTED     3
/** AGPS data connection completed */
#define GPS_AGPS_DATA_CONN_DONE     4
/** AGPS data connection failed */
#define GPS_AGPS_DATA_CONN_FAILED   5

#define AGPS_REF_LOCATION_TYPE_GSM_CELLID   1
#define AGPS_REF_LOCATION_TYPE_UMTS_CELLID  2
#define AGPS_REG_LOCATION_TYPE_MAC          3

/** Network types for update_network_state "type" parameter */
#define AGPS_RIL_NETWORK_TYPE_MOBILE        0
#define AGPS_RIL_NETWORK_TYPE_WIFI          1
#define AGPS_RIL_NETWORK_TYPE_MOBILE_MMS    2
#define AGPS_RIL_NETWORK_TYPE_MOBILE_SUPL   3
#define AGPS_RIL_NETWORK_TTYPE_MOBILE_DUN   4
#define AGPS_RIL_NETWORK_TTYPE_MOBILE_HIPRI 5
#define AGPS_RIL_NETWORK_TTYPE_WIMAX        6

/**
 * Flags to indicate what fields in GpsClock are valid.
 */
typedef uint16_t GpsClockFlags;
/** A valid 'leap second' is stored in the data structure. */
#define GPS_CLOCK_HAS_LEAP_SECOND               (1<<0)
/** A valid 'time uncertainty' is stored in the data structure. */
#define GPS_CLOCK_HAS_TIME_UNCERTAINTY          (1<<1)
/** A valid 'full bias' is stored in the data structure. */
#define GPS_CLOCK_HAS_FULL_BIAS                 (1<<2)
/** A valid 'bias' is stored in the data structure. */
#define GPS_CLOCK_HAS_BIAS                      (1<<3)
/** A valid 'bias uncertainty' is stored in the data structure. */
#define GPS_CLOCK_HAS_BIAS_UNCERTAINTY          (1<<4)
/** A valid 'drift' is stored in the data structure. */
#define GPS_CLOCK_HAS_DRIFT                     (1<<5)
/** A valid 'drift uncertainty' is stored in the data structure. */
#define GPS_CLOCK_HAS_DRIFT_UNCERTAINTY         (1<<6)

/**
 * Enumeration of the available values for the GPS Clock type.
 */
typedef uint8_t GpsClockType;
/** The type is not available ot it is unknown. */
#define GPS_CLOCK_TYPE_UNKNOWN                  0
/** The source of the time value reported by GPS clock is the local hardware clock. */
#define GPS_CLOCK_TYPE_LOCAL_HW_TIME            1
/**
 * The source of the time value reported by GPS clock is the GPS time derived from satellites
 * (epoch = Jan 6, 1980)
 */
#define GPS_CLOCK_TYPE_GPS_TIME                 2

/**
 * Flags to indicate what fields in GpsMeasurement are valid.
 */
typedef uint32_t GpsMeasurementFlags;
/** A valid 'snr' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_SNR                               (1<<0)
/** A valid 'elevation' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_ELEVATION                         (1<<1)
/** A valid 'elevation uncertainty' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_ELEVATION_UNCERTAINTY             (1<<2)
/** A valid 'azimuth' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_AZIMUTH                           (1<<3)
/** A valid 'azimuth uncertainty' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_AZIMUTH_UNCERTAINTY               (1<<4)
/** A valid 'pseudorange' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_PSEUDORANGE                       (1<<5)
/** A valid 'pseudorange uncertainty' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_PSEUDORANGE_UNCERTAINTY           (1<<6)
/** A valid 'code phase' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_CODE_PHASE                        (1<<7)
/** A valid 'code phase uncertainty' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_CODE_PHASE_UNCERTAINTY            (1<<8)
/** A valid 'carrier frequency' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_CARRIER_FREQUENCY                 (1<<9)
/** A valid 'carrier cycles' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_CARRIER_CYCLES                    (1<<10)
/** A valid 'carrier phase' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_CARRIER_PHASE                     (1<<11)
/** A valid 'carrier phase uncertainty' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_CARRIER_PHASE_UNCERTAINTY         (1<<12)
/** A valid 'bit number' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_BIT_NUMBER                        (1<<13)
/** A valid 'time from last bit' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_TIME_FROM_LAST_BIT                (1<<14)
/** A valid 'doppler shift' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_DOPPLER_SHIFT                     (1<<15)
/** A valid 'doppler shift uncertainty' is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_DOPPLER_SHIFT_UNCERTAINTY         (1<<16)
/** A valid 'used in fix' flag is stored in the data structure. */
#define GPS_MEASUREMENT_HAS_USED_IN_FIX                       (1<<17)
/** The value of 'pseudorange rate' is uncorrected. */
#define GPS_MEASUREMENT_HAS_UNCORRECTED_PSEUDORANGE_RATE      (1<<18)

/**
 * Enumeration of the available values for the GPS Measurement's loss of lock.
 */
typedef uint8_t GpsLossOfLock;
/** The indicator is not available or it is unknown. */
#define GPS_LOSS_OF_LOCK_UNKNOWN                            0
/** The measurement does not present any indication of loss of lock. */
#define GPS_LOSS_OF_LOCK_OK                                 1
/** Loss of lock between previous and current observation: cycle slip possible. */
#define GPS_LOSS_OF_LOCK_CYCLE_SLIP                         2

/**
 * Enumeration of available values for the GPS Measurement's multipath indicator.
 */
typedef uint8_t GpsMultipathIndicator;
/** The indicator is not available or unknown. */
#define GPS_MULTIPATH_INDICATOR_UNKNOWN                 0
/** The measurement has been indicated to use multipath. */
#define GPS_MULTIPATH_INDICATOR_DETECTED                1
/** The measurement has been indicated Not to use multipath. */
#define GPS_MULTIPATH_INDICATOR_NOT_USED                2

/**
 * Flags indicating the GPS measurement state.
 * The expected behavior here is for GPS HAL to set all the flags that applies. For
 * example, if the state for a satellite is only C/A code locked and bit synchronized,
 * and there is still millisecond ambiguity, the state should be set as:
 * GPS_MEASUREMENT_STATE_CODE_LOCK|GPS_MEASUREMENT_STATE_BIT_SYNC|GPS_MEASUREMENT_STATE_MSEC_AMBIGUOUS
 * If GPS is still searching for a satellite, the corresponding state should be set to
 * GPS_MEASUREMENT_STATE_UNKNOWN(0).
 */
typedef uint16_t GpsMeasurementState;
#define GPS_MEASUREMENT_STATE_UNKNOWN                   0
#define GPS_MEASUREMENT_STATE_CODE_LOCK             (1<<0)
#define GPS_MEASUREMENT_STATE_BIT_SYNC              (1<<1)
#define GPS_MEASUREMENT_STATE_SUBFRAME_SYNC         (1<<2)
#define GPS_MEASUREMENT_STATE_TOW_DECODED           (1<<3)
#define GPS_MEASUREMENT_STATE_MSEC_AMBIGUOUS        (1<<4)

/**
 * Flags indicating the Accumulated Delta Range's states.
 */
typedef uint16_t GpsAccumulatedDeltaRangeState;
#define GPS_ADR_STATE_UNKNOWN                       0
#define GPS_ADR_STATE_VALID                     (1<<0)
#define GPS_ADR_STATE_RESET                     (1<<1)
#define GPS_ADR_STATE_CYCLE_SLIP                (1<<2)

/**
 * Enumeration of available values to indicate the available GPS Navigation message types.
 */
typedef uint8_t GpsNavigationMessageType;
/** The message type is unknown. */
#define GPS_NAVIGATION_MESSAGE_TYPE_UNKNOWN         0
/** L1 C/A message contained in the structure.  */
#define GPS_NAVIGATION_MESSAGE_TYPE_L1CA            1
/** L2-CNAV message contained in the structure. */
#define GPS_NAVIGATION_MESSAGE_TYPE_L2CNAV          2
/** L5-CNAV message contained in the structure. */
#define GPS_NAVIGATION_MESSAGE_TYPE_L5CNAV          3
/** CNAV-2 message contained in the structure. */
#define GPS_NAVIGATION_MESSAGE_TYPE_CNAV2           4

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
#define NAV_MESSAGE_STATUS_UNKONW              0
#define NAV_MESSAGE_STATUS_PARITY_PASSED   (1<<0)
#define NAV_MESSAGE_STATUS_PARITY_REBUILT  (1<<1)

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
    /** Represents altitude in meters above the WGS 84 reference
     * ellipsoid. */
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

/** Represents SV information. */
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

/** Represents SV status. */
typedef struct {
    /** set to sizeof(GpsSvStatus) */
    size_t          size;

    /** Number of SVs currently visible. */
    int         num_svs;

    /** Contains an array of SV information. */
    GpsSvInfo   sv_list[GPS_MAX_SVS];

    /** Represents a bit mask indicating which SVs
     * have ephemeris data.
     */
    uint32_t    ephemeris_mask;

    /** Represents a bit mask indicating which SVs
     * have almanac data.
     */
    uint32_t    almanac_mask;

    /**
     * Represents a bit mask indicating which SVs
     * were used for computing the most recent position fix.
     */
    uint32_t    used_in_fix_mask;
} GpsSvStatus;


/* 2G and 3G */
/* In 3G lac is discarded */
typedef struct {
    uint16_t type;
    uint16_t mcc;
    uint16_t mnc;
    uint16_t lac;
    uint32_t cid;
} AGpsRefLocationCellID;

typedef struct {
    uint8_t mac[6];
} AGpsRefLocationMac;

/** Represents ref locations */
typedef struct {
    uint16_t type;
    union {
        AGpsRefLocationCellID   cellID;
        AGpsRefLocationMac      mac;
    } u;
} AGpsRefLocation;

/** Callback with location information.
 *  Can only be called from a thread created by create_thread_cb.
 */
typedef void (* gps_location_callback)(GpsLocation* location);

/** Callback with status information.
 *  Can only be called from a thread created by create_thread_cb.
 */
typedef void (* gps_status_callback)(GpsStatus* status);

/**
 * Callback with SV status information.
 * Can only be called from a thread created by create_thread_cb.
 */
typedef void (* gps_sv_status_callback)(GpsSvStatus* sv_info);

/** Callback for reporting NMEA sentences.
 *  Can only be called from a thread created by create_thread_cb.
 */
typedef void (* gps_nmea_callback)(GpsUtcTime timestamp, const char* nmea, int length);

/** Callback to inform framework of the GPS engine's capabilities.
 *  Capability parameter is a bit field of GPS_CAPABILITY_* flags.
 */
typedef void (* gps_set_capabilities)(uint32_t capabilities);

/** Callback utility for acquiring the GPS wakelock.
 *  This can be used to prevent the CPU from suspending while handling GPS events.
 */
typedef void (* gps_acquire_wakelock)();

/** Callback utility for releasing the GPS wakelock. */
typedef void (* gps_release_wakelock)();

/** Callback for requesting NTP time */
typedef void (* gps_request_utc_time)();

/** Callback for creating a thread that can call into the Java framework code.
 *  This must be used to create any threads that report events up to the framework.
 */
typedef pthread_t (* gps_create_thread)(const char* name, void (*start)(void *), void* arg);

/** GPS callback structure. */
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

    /** Injects current location from another location provider
     *  (typically cell ID).
     *  latitude and longitude are measured in degrees
     *  expected accuracy is measured in meters
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
     * 'mode' parameter should be one of GPS_POSITION_MODE_MS_BASE
     * or GPS_POSITION_MODE_STANDALONE.
     * It is allowed by the platform (and it is recommended) to fallback to
     * GPS_POSITION_MODE_MS_BASE if GPS_POSITION_MODE_MS_ASSISTED is passed in, and
     * GPS_POSITION_MODE_MS_BASED is supported.
     */
    int   (*set_position_mode)(GpsPositionMode mode, GpsPositionRecurrence recurrence,
            uint32_t min_interval, uint32_t preferred_accuracy, uint32_t preferred_time);

    /** Get a pointer to extension information. */
    const void* (*get_extension)(const char* name);
} GpsInterface;

/** Callback to request the client to download XTRA data.
 *  The client should download XTRA data and inject it by calling inject_xtra_data().
 *  Can only be called from a thread created by create_thread_cb.
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

#pragma pack(push,4)
// We need to keep the alignment of this data structure to 4-bytes, to ensure that in 64-bit
// environments the size of this legacy definition does not collide with _v2. Implementations should
// be using _v2 and _v3, so it's OK to pay the 'unaligned' penalty in 64-bit if an old
// implementation is still in use.

/** Represents the status of AGPS. */
typedef struct {
    /** set to sizeof(AGpsStatus_v1) */
    size_t          size;

    AGpsType        type;
    AGpsStatusValue status;
} AGpsStatus_v1;

#pragma pack(pop)

/** Represents the status of AGPS augmented with a IPv4 address field. */
typedef struct {
    /** set to sizeof(AGpsStatus_v2) */
    size_t          size;

    AGpsType        type;
    AGpsStatusValue status;
    uint32_t        ipaddr;
} AGpsStatus_v2;

/* Represents the status of AGPS augmented to support IPv4 and IPv6. */
typedef struct {
    /** set to sizeof(AGpsStatus_v3) */
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
     * */
    struct sockaddr_storage addr;
} AGpsStatus_v3;

typedef AGpsStatus_v3     AGpsStatus;

/** Callback with AGPS status information.
 *  Can only be called from a thread created by create_thread_cb.
 */
typedef void (* agps_status_callback)(AGpsStatus* status);

/** Callback structure for the AGPS interface. */
typedef struct {
    agps_status_callback status_cb;
    gps_create_thread create_thread_cb;
} AGpsCallbacks;


/** Extended interface for AGPS support. */
typedef struct {
    /** set to sizeof(AGpsInterface_v1) */
    size_t          size;

    /**
     * Opens the AGPS interface and provides the callback routines
     * to the implementation of this interface.
     */
    void  (*init)( AGpsCallbacks* callbacks );
    /**
     * Notifies that a data connection is available and sets
     * the name of the APN to be used for SUPL.
     */
    int  (*data_conn_open)( const char* apn );
    /**
     * Notifies that the AGPS data connection has been closed.
     */
    int  (*data_conn_closed)();
    /**
     * Notifies that a data connection is not available for AGPS.
     */
    int  (*data_conn_failed)();
    /**
     * Sets the hostname and port for the AGPS server.
     */
    int  (*set_server)( AGpsType type, const char* hostname, int port );
} AGpsInterface_v1;

/**
 * Extended interface for AGPS support, it is augmented to enable to pass
 * extra APN data.
 */
typedef struct {
    /** set to sizeof(AGpsInterface_v2) */
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
} AGpsInterface_v2;

typedef AGpsInterface_v2    AGpsInterface;

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

/** Callback with NI notification.
 *  Can only be called from a thread created by create_thread_cb.
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

#define AGPS_RIL_REQUEST_SETID_IMSI     (1<<0L)
#define AGPS_RIL_REQUEST_SETID_MSISDN   (1<<1L)

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
#define GPS_GEOFENCE_ENTERED     (1<<0L)
#define GPS_GEOFENCE_EXITED      (1<<1L)
#define GPS_GEOFENCE_UNCERTAIN   (1<<2L)

#define GPS_GEOFENCE_UNAVAILABLE (1<<0L)
#define GPS_GEOFENCE_AVAILABLE   (1<<1L)

#define GPS_GEOFENCE_OPERATION_SUCCESS           0
#define GPS_GEOFENCE_ERROR_TOO_MANY_GEOFENCES -100
#define GPS_GEOFENCE_ERROR_ID_EXISTS          -101
#define GPS_GEOFENCE_ERROR_ID_UNKNOWN         -102
#define GPS_GEOFENCE_ERROR_INVALID_TRANSITION -103
#define GPS_GEOFENCE_ERROR_GENERIC            -149

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
 * Represents an estimate of the GPS clock time.
 */
typedef struct {
    /** set to sizeof(GpsClock) */
    size_t size;

    /** A set of flags indicating the validity of the fields in this data structure. */
    GpsClockFlags flags;

    /**
     * Leap second data.
     * The sign of the value is defined by the following equation:
     *      utc_time_ns = time_ns + (full_bias_ns + bias_ns) - leap_second * 1,000,000,000
     *
     * If the data is available 'flags' must contain GPS_CLOCK_HAS_LEAP_SECOND.
     */
    int16_t leap_second;

    /**
     * Indicates the type of time reported by the 'time_ns' field.
     * This is a Mandatory field.
     */
    GpsClockType type;

    /**
     * The GPS receiver internal clock value. This can be either the local hardware clock value
     * (GPS_CLOCK_TYPE_LOCAL_HW_TIME), or the current GPS time derived inside GPS receiver
     * (GPS_CLOCK_TYPE_GPS_TIME). The field 'type' defines the time reported.
     *
     * For local hardware clock, this value is expected to be monotonically increasing during
     * the reporting session. The real GPS time can be derived by compensating the 'full bias'
     * (when it is available) from this value.
     *
     * For GPS time, this value is expected to be the best estimation of current GPS time that GPS
     * receiver can achieve. Set the 'time uncertainty' appropriately when GPS time is specified.
     *
     * Sub-nanosecond accuracy can be provided by means of the 'bias' field.
     * The value contains the 'time uncertainty' in it.
     *
     * This is a Mandatory field.
     */
    int64_t time_ns;

    /**
     * 1-Sigma uncertainty associated with the clock's time in nanoseconds.
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * This value should be set if GPS_CLOCK_TYPE_GPS_TIME is set.
     * If the data is available 'flags' must contain GPS_CLOCK_HAS_TIME_UNCERTAINTY.
     */
    double time_uncertainty_ns;

    /**
     * The difference between hardware clock ('time' field) inside GPS receiver and the true GPS
     * time since 0000Z, January 6, 1980, in nanoseconds.
     * This value is used if and only if GPS_CLOCK_TYPE_LOCAL_HW_TIME is set, and GPS receiver
     * has solved the clock for GPS time.
     * The caller is responsible for using the 'bias uncertainty' field for quality check.
     *
     * The sign of the value is defined by the following equation:
     *      true time (GPS time) = time_ns + (full_bias_ns + bias_ns)
     *
     * This value contains the 'bias uncertainty' in it.
     * If the data is available 'flags' must contain GPS_CLOCK_HAS_FULL_BIAS.

     */
    int64_t full_bias_ns;

    /**
     * Sub-nanosecond bias.
     * The value contains the 'bias uncertainty' in it.
     *
     * If the data is available 'flags' must contain GPS_CLOCK_HAS_BIAS.
     */
    double bias_ns;

    /**
     * 1-Sigma uncertainty associated with the clock's bias in nanoseconds.
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * If the data is available 'flags' must contain GPS_CLOCK_HAS_BIAS_UNCERTAINTY.
     */
    double bias_uncertainty_ns;

    /**
     * The clock's drift in nanoseconds (per second).
     * A positive value means that the frequency is higher than the nominal frequency.
     *
     * The value contains the 'drift uncertainty' in it.
     * If the data is available 'flags' must contain GPS_CLOCK_HAS_DRIFT.
     *
     * If GpsMeasurement's 'flags' field contains GPS_MEASUREMENT_HAS_UNCORRECTED_PSEUDORANGE_RATE,
     * it is encouraged that this field is also provided.
     */
    double drift_nsps;

    /**
     * 1-Sigma uncertainty associated with the clock's drift in nanoseconds (per second).
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * If the data is available 'flags' must contain GPS_CLOCK_HAS_DRIFT_UNCERTAINTY.
     */
    double drift_uncertainty_nsps;
} GpsClock;

/**
 * Represents a GPS Measurement, it contains raw and computed information.
 */
typedef struct {
    /** set to sizeof(GpsMeasurement) */
    size_t size;

    /** A set of flags indicating the validity of the fields in this data structure. */
    GpsMeasurementFlags flags;

    /**
     * Pseudo-random number in the range of [1, 32]
     * This is a Mandatory value.
     */
    int8_t prn;

    /**
     * Time offset at which the measurement was taken in nanoseconds.
     * The reference receiver's time is specified by GpsData::clock::time_ns and should be
     * interpreted in the same way as indicated by GpsClock::type.
     *
     * The sign of time_offset_ns is given by the following equation:
     *      measurement time = GpsClock::time_ns + time_offset_ns
     *
     * It provides an individual time-stamp for the measurement, and allows sub-nanosecond accuracy.
     * This is a Mandatory value.
     */
    double time_offset_ns;

    /**
     * Per satellite sync state. It represents the current sync state for the associated satellite.
     * Based on the sync state, the 'received GPS tow' field should be interpreted accordingly.
     *
     * This is a Mandatory value.
     */
    GpsMeasurementState state;

    /**
     * Received GPS Time-of-Week at the measurement time, in nanoseconds.
     * The value is relative to the beginning of the current GPS week.
     *
     * Given the highest sync state that can be achieved, per each satellite, valid range for
     * this field can be:
     *     Searching       : [ 0       ]   : GPS_MEASUREMENT_STATE_UNKNOWN
     *     C/A code lock   : [ 0   1ms ]   : GPS_MEASUREMENT_STATE_CODE_LOCK is set
     *     Bit sync        : [ 0  20ms ]   : GPS_MEASUREMENT_STATE_BIT_SYNC is set
     *     Subframe sync   : [ 0    6s ]   : GPS_MEASUREMENT_STATE_SUBFRAME_SYNC is set
     *     TOW decoded     : [ 0 1week ]   : GPS_MEASUREMENT_STATE_TOW_DECODED is set
     *
     * However, if there is any ambiguity in integer millisecond,
     * GPS_MEASUREMENT_STATE_MSEC_AMBIGUOUS should be set accordingly, in the 'state' field.
     *
     * This value must be populated if 'state' != GPS_MEASUREMENT_STATE_UNKNOWN.
     */
    int64_t received_gps_tow_ns;

    /**
     * 1-Sigma uncertainty of the Received GPS Time-of-Week in nanoseconds.
     *
     * This value must be populated if 'state' != GPS_MEASUREMENT_STATE_UNKNOWN.
     */
    int64_t received_gps_tow_uncertainty_ns;

    /**
     * Carrier-to-noise density in dB-Hz, in the range [0, 63].
     * It contains the measured C/N0 value for the signal at the antenna input.
     *
     * This is a Mandatory value.
     */
    double c_n0_dbhz;

    /**
     * Pseudorange rate at the timestamp in m/s.
     * The correction of a given Pseudorange Rate value includes corrections for receiver and
     * satellite clock frequency errors.
     *
     * If GPS_MEASUREMENT_HAS_UNCORRECTED_PSEUDORANGE_RATE is set in 'flags' field, this field must
     * be populated with the 'uncorrected' reading.
     * If GPS_MEASUREMENT_HAS_UNCORRECTED_PSEUDORANGE_RATE is not set in 'flags' field, this field
     * must be populated with the 'corrected' reading. This is the default behavior.
     *
     * It is encouraged to provide the 'uncorrected' 'pseudorange rate', and provide GpsClock's
     * 'drift' field as well.
     *
     * The value includes the 'pseudorange rate uncertainty' in it.
     * A positive 'uncorrected' value indicates that the SV is moving away from the receiver.
     *
     * The sign of the 'uncorrected' 'pseudorange rate' and its relation to the sign of 'doppler
     * shift' is given by the equation:
     *      pseudorange rate = -k * doppler shift   (where k is a constant)
     *
     * This is a Mandatory value.
     */
    double pseudorange_rate_mps;

    /**
     * 1-Sigma uncertainty of the pseudurange rate in m/s.
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * This is a Mandatory value.
     */
    double pseudorange_rate_uncertainty_mps;

    /**
     * Accumulated delta range's state. It indicates whether ADR is reset or there is a cycle slip
     * (indicating loss of lock).
     *
     * This is a Mandatory value.
     */
    GpsAccumulatedDeltaRangeState accumulated_delta_range_state;

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
     * Best derived Pseudorange by the chip-set, in meters.
     * The value contains the 'pseudorange uncertainty' in it.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_PSEUDORANGE.
     */
    double pseudorange_m;

    /**
     * 1-Sigma uncertainty of the pseudorange in meters.
     * The value contains the 'pseudorange' and 'clock' uncertainty in it.
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_PSEUDORANGE_UNCERTAINTY.
     */
    double pseudorange_uncertainty_m;

    /**
     * A fraction of the current C/A code cycle, in the range [0.0, 1023.0]
     * This value contains the time (in Chip units) since the last C/A code cycle (GPS Msec epoch).
     *
     * The reference frequency is given by the field 'carrier_frequency_hz'.
     * The value contains the 'code-phase uncertainty' in it.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_CODE_PHASE.
     */
    double code_phase_chips;

    /**
     * 1-Sigma uncertainty of the code-phase, in a fraction of chips.
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_CODE_PHASE_UNCERTAINTY.
     */
    double code_phase_uncertainty_chips;

    /**
     * Carrier frequency at which codes and messages are modulated, it can be L1 or L2.
     * If the field is not set, the carrier frequency is assumed to be L1.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_CARRIER_FREQUENCY.
     */
    float carrier_frequency_hz;

    /**
     * The number of full carrier cycles between the satellite and the receiver.
     * The reference frequency is given by the field 'carrier_frequency_hz'.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_CARRIER_CYCLES.
     */
    int64_t carrier_cycles;

    /**
     * The RF phase detected by the receiver, in the range [0.0, 1.0].
     * This is usually the fractional part of the complete carrier phase measurement.
     *
     * The reference frequency is given by the field 'carrier_frequency_hz'.
     * The value contains the 'carrier-phase uncertainty' in it.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_CARRIER_PHASE.
     */
    double carrier_phase;

    /**
     * 1-Sigma uncertainty of the carrier-phase.
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_CARRIER_PHASE_UNCERTAINTY.
     */
    double carrier_phase_uncertainty;

    /**
     * An enumeration that indicates the 'loss of lock' state of the event.
     */
    GpsLossOfLock loss_of_lock;

    /**
     * The number of GPS bits transmitted since Sat-Sun midnight (GPS week).
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_BIT_NUMBER.
     */
    int32_t bit_number;

    /**
     * The elapsed time since the last received bit in milliseconds, in the range [0, 20]
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_TIME_FROM_LAST_BIT.
     */
    int16_t time_from_last_bit_ms;

    /**
     * Doppler shift in Hz.
     * A positive value indicates that the SV is moving toward the receiver.
     *
     * The reference frequency is given by the field 'carrier_frequency_hz'.
     * The value contains the 'doppler shift uncertainty' in it.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_DOPPLER_SHIFT.
     */
    double doppler_shift_hz;

    /**
     * 1-Sigma uncertainty of the doppler shift in Hz.
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_DOPPLER_SHIFT_UNCERTAINTY.
     */
    double doppler_shift_uncertainty_hz;

    /**
     * An enumeration that indicates the 'multipath' state of the event.
     */
    GpsMultipathIndicator multipath_indicator;

    /**
     * Signal-to-noise ratio in dB.
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_SNR.
     */
    double snr_db;

    /**
     * Elevation in degrees, the valid range is [-90, 90].
     * The value contains the 'elevation uncertainty' in it.
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_ELEVATION.
     */
    double elevation_deg;

    /**
     * 1-Sigma uncertainty of the elevation in degrees, the valid range is [0, 90].
     * The uncertainty is represented as the absolute (single sided) value.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_ELEVATION_UNCERTAINTY.
     */
    double elevation_uncertainty_deg;

    /**
     * Azimuth in degrees, in the range [0, 360).
     * The value contains the 'azimuth uncertainty' in it.
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_AZIMUTH.
     *  */
    double azimuth_deg;

    /**
     * 1-Sigma uncertainty of the azimuth in degrees, the valid range is [0, 180].
     * The uncertainty is represented as an absolute (single sided) value.
     *
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_AZIMUTH_UNCERTAINTY.
     */
    double azimuth_uncertainty_deg;

    /**
     * Whether the GPS represented by the measurement was used for computing the most recent fix.
     * If the data is available, 'flags' must contain GPS_MEASUREMENT_HAS_USED_IN_FIX.
     */
    bool used_in_fix;
} GpsMeasurement;

/** Represents a reading of GPS measurements. */
typedef struct {
    /** set to sizeof(GpsData) */
    size_t size;

    /** Number of measurements. */
    size_t measurement_count;

    /** The array of measurements. */
    GpsMeasurement measurements[GPS_MAX_MEASUREMENT];

    /** The GPS clock time reading. */
    GpsClock clock;
} GpsData;

/**
 * The callback for to report measurements from the HAL.
 *
 * Parameters:
 *    data - A data structure containing the measurements.
 */
typedef void (*gps_measurement_callback) (GpsData* data);

typedef struct {
    /** set to sizeof(GpsMeasurementCallbacks) */
    size_t size;
    gps_measurement_callback measurement_callback;
} GpsMeasurementCallbacks;

#define GPS_MEASUREMENT_OPERATION_SUCCESS          0
#define GPS_MEASUREMENT_ERROR_ALREADY_INIT      -100
#define GPS_MEASUREMENT_ERROR_GENERIC           -101

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


/** Represents a GPS navigation message (or a fragment of it). */
typedef struct {
    /** set to sizeof(GpsNavigationMessage) */
    size_t size;

    /**
     * Pseudo-random number in the range of [1, 32]
     * This is a Mandatory value.
     */
    int8_t prn;

    /**
     * The type of message contained in the structure.
     * This is a Mandatory value.
     */
    GpsNavigationMessageType type;

    /**
     * The status of the received navigation message.
     * No need to send any navigation message that contains words with parity error and cannot be
     * corrected.
     */
    NavigationMessageStatus status;

    /**
     * Message identifier.
     * It provides an index so the complete Navigation Message can be assembled. i.e. fo L1 C/A
     * subframe 4 and 5, this value corresponds to the 'frame id' of the navigation message.
     * Subframe 1, 2, 3 does not contain a 'frame id' and this value can be set to -1.
     */
    int16_t message_id;

    /**
     * Sub-message identifier.
     * If required by the message 'type', this value contains a sub-index within the current
     * message (or frame) that is being transmitted.
     * i.e. for L1 C/A the submessage id corresponds to the sub-frame id of the navigation message.
     */
    int16_t submessage_id;

    /**
     * The length of the data (in bytes) contained in the current message.
     * If this value is different from zero, 'data' must point to an array of the same size.
     * e.g. for L1 C/A the size of the sub-frame will be 40 bytes (10 words, 30 bits/word).
     *
     * This is a Mandatory value.
     */
    size_t data_length;

    /**
     * The data of the reported GPS message.
     * The bytes (or words) specified using big endian format (MSB first).
     *
     * For L1 C/A, each subframe contains 10 30-bit GPS words. Each GPS word (30 bits) should be
     * fitted into the last 30 bits in a 4-byte word (skip B31 and B32), with MSB first.
     */
    uint8_t* data;

} GpsNavigationMessage;

/**
 * The callback to report an available fragment of a GPS navigation messages from the HAL.
 *
 * Parameters:
 *      message - The GPS navigation submessage/subframe representation.
 */
typedef void (*gps_navigation_message_callback) (GpsNavigationMessage* message);

typedef struct {
    /** set to sizeof(GpsNavigationMessageCallbacks) */
    size_t size;
    gps_navigation_message_callback navigation_message_callback;
} GpsNavigationMessageCallbacks;

#define GPS_NAVIGATION_MESSAGE_OPERATION_SUCCESS             0
#define GPS_NAVIGATION_MESSAGE_ERROR_ALREADY_INIT         -100
#define GPS_NAVIGATION_MESSAGE_ERROR_GENERIC              -101

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

