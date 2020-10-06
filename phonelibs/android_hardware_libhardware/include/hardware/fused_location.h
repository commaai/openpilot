/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_HARDWARE_FUSED_LOCATION_H
#define ANDROID_INCLUDE_HARDWARE_FUSED_LOCATION_H

#include <hardware/hardware.h>


/**
 * This header file defines the interface of the Fused Location Provider.
 * Fused Location Provider is designed to fuse data from various sources
 * like GPS, Wifi, Cell, Sensors, Bluetooth etc to provide a fused location to the
 * upper layers. The advantage of doing fusion in hardware is power savings.
 * The goal is to do this without waking up the AP to get additional data.
 * The software implementation of FLP will decide when to use
 * the hardware fused location. Other location features like geofencing will
 * also be implemented using fusion in hardware.
 */
__BEGIN_DECLS

#define FLP_HEADER_VERSION          1
#define FLP_MODULE_API_VERSION_0_1  HARDWARE_MODULE_API_VERSION(0, 1)
#define FLP_DEVICE_API_VERSION_0_1  HARDWARE_DEVICE_API_VERSION_2(0, 1, FLP_HEADER_VERSION)

/**
 * The id of this module
 */
#define FUSED_LOCATION_HARDWARE_MODULE_ID "flp"

/**
 * Name for the FLP location interface
 */
#define FLP_LOCATION_INTERFACE     "flp_location"

/**
 * Name for the FLP location interface
 */
#define FLP_DIAGNOSTIC_INTERFACE     "flp_diagnostic"

/**
 * Name for the FLP_Geofencing interface.
 */
#define FLP_GEOFENCING_INTERFACE   "flp_geofencing"

/**
 * Name for the FLP_device context interface.
 */
#define FLP_DEVICE_CONTEXT_INTERFACE   "flp_device_context"

/**
 * Constants to indicate the various subsystems
 * that will be used.
 */
#define FLP_TECH_MASK_GNSS      (1U<<0)
#define FLP_TECH_MASK_WIFI      (1U<<1)
#define FLP_TECH_MASK_SENSORS   (1U<<2)
#define FLP_TECH_MASK_CELL      (1U<<3)
#define FLP_TECH_MASK_BLUETOOTH (1U<<4)

/**
 * Set when your implementation can produce GNNS-derived locations,
 * for use with flp_capabilities_callback.
 *
 * GNNS is a required capability for a particular feature to be used
 * (batching or geofencing).  If not supported that particular feature
 * won't be used by the upper layer.
 */
#define CAPABILITY_GNSS         (1U<<0)
/**
 * Set when your implementation can produce WiFi-derived locations, for
 * use with flp_capabilities_callback.
 */
#define CAPABILITY_WIFI         (1U<<1)
/**
 * Set when your implementation can produce cell-derived locations, for
 * use with flp_capabilities_callback.
 */
#define CAPABILITY_CELL         (1U<<3)

/**
 * Status to return in flp_status_callback when your implementation transitions
 * from being unsuccessful in determining location to being successful.
 */
#define FLP_STATUS_LOCATION_AVAILABLE         0
/**
 * Status to return in flp_status_callback when your implementation transitions
 * from being successful in determining location to being unsuccessful.
 */
#define FLP_STATUS_LOCATION_UNAVAILABLE       1

/**
 * This constant is used with the batched locations
 * APIs. Batching is mandatory when FLP implementation
 * is supported. If the flag is set, the hardware implementation
 * will wake up the application processor when the FIFO is full,
 * If the flag is not set, the hardware implementation will drop
 * the oldest data when the FIFO is full.
 */
#define FLP_BATCH_WAKEUP_ON_FIFO_FULL        0x0000001

/**
 * While batching, the implementation should not call the
 * flp_location_callback on every location fix. However,
 * sometimes in high power mode, the system might need
 * a location callback every single time the location
 * fix has been obtained. This flag controls that option.
 * Its the responsibility of the upper layers (caller) to switch
 * it off, if it knows that the AP might go to sleep.
 * When this bit is on amidst a batching session, batching should
 * continue while location fixes are reported in real time.
 */
#define FLP_BATCH_CALLBACK_ON_LOCATION_FIX   0x0000002

/** Flags to indicate which values are valid in a FlpLocation. */
typedef uint16_t FlpLocationFlags;

// IMPORTANT: Note that the following values must match
// constants in the corresponding java file.

/** FlpLocation has valid latitude and longitude. */
#define FLP_LOCATION_HAS_LAT_LONG   (1U<<0)
/** FlpLocation has valid altitude. */
#define FLP_LOCATION_HAS_ALTITUDE   (1U<<1)
/** FlpLocation has valid speed. */
#define FLP_LOCATION_HAS_SPEED      (1U<<2)
/** FlpLocation has valid bearing. */
#define FLP_LOCATION_HAS_BEARING    (1U<<4)
/** FlpLocation has valid accuracy. */
#define FLP_LOCATION_HAS_ACCURACY   (1U<<8)


typedef int64_t FlpUtcTime;

/** Represents a location. */
typedef struct {
    /** set to sizeof(FlpLocation) */
    size_t          size;

    /** Flags associated with the location object. */
    FlpLocationFlags flags;

    /** Represents latitude in degrees. */
    double          latitude;

    /** Represents longitude in degrees. */
    double          longitude;

    /**
     * Represents altitude in meters above the WGS 84 reference
     * ellipsoid. */
    double          altitude;

    /** Represents speed in meters per second. */
    float           speed;

    /** Represents heading in degrees. */
    float           bearing;

    /** Represents expected accuracy in meters. */
    float           accuracy;

    /** Timestamp for the location fix. */
    FlpUtcTime      timestamp;

    /** Sources used, will be Bitwise OR of the FLP_TECH_MASK bits. */
    uint32_t         sources_used;
} FlpLocation;

typedef enum {
    ASSOCIATE_JVM,
    DISASSOCIATE_JVM,
} ThreadEvent;

/**
 *  Callback with location information.
 *  Can only be called from a thread associated to JVM using set_thread_event_cb.
 *  Parameters:
 *     num_locations is the number of batched locations available.
 *     location is the pointer to an array of pointers to location objects.
 */
typedef void (*flp_location_callback)(int32_t num_locations, FlpLocation** location);

/**
 * Callback utility for acquiring a wakelock.
 * This can be used to prevent the CPU from suspending while handling FLP events.
 */
typedef void (*flp_acquire_wakelock)();

/**
 * Callback utility for releasing the FLP wakelock.
 */
typedef void (*flp_release_wakelock)();

/**
 * Callback for associating a thread that can call into the Java framework code.
 * This must be used to initialize any threads that report events up to the framework.
 * Return value:
 *      FLP_RESULT_SUCCESS on success.
 *      FLP_RESULT_ERROR if the association failed in the current thread.
 */
typedef int (*flp_set_thread_event)(ThreadEvent event);

/**
 * Callback for technologies supported by this implementation.
 *
 * Parameters: capabilities is a bitmask of FLP_CAPABILITY_* values describing
 * which features your implementation supports.  You should support
 * CAPABILITY_GNSS at a minimum for your implementation to be utilized.  You can
 * return 0 in FlpGeofenceCallbacks to indicate you don't support geofencing,
 * or 0 in FlpCallbacks to indicate you don't support location batching.
 */
typedef void (*flp_capabilities_callback)(int capabilities);

/**
 * Callback with status information on the ability to compute location.
 * To avoid waking up the application processor you should only send
 * changes in status (you shouldn't call this method twice in a row
 * with the same status value).  As a guideline you should not call this
 * more frequently then the requested batch period set with period_ns
 * in FlpBatchOptions.  For example if period_ns is set to 5 minutes and
 * the status changes many times in that interval, you should only report
 * one status change every 5 minutes.
 *
 * Parameters:
 *     status is one of FLP_STATUS_LOCATION_AVAILABLE
 *     or FLP_STATUS_LOCATION_UNAVAILABLE.
 */
typedef void (*flp_status_callback)(int32_t status);

/** FLP callback structure. */
typedef struct {
    /** set to sizeof(FlpCallbacks) */
    size_t      size;
    flp_location_callback location_cb;
    flp_acquire_wakelock acquire_wakelock_cb;
    flp_release_wakelock release_wakelock_cb;
    flp_set_thread_event set_thread_event_cb;
    flp_capabilities_callback flp_capabilities_cb;
    flp_status_callback flp_status_cb;
} FlpCallbacks;


/** Options with the batching FLP APIs */
typedef struct {
    /**
     * Maximum power in mW that the underlying implementation
     * can use for this batching call.
     * If max_power_allocation_mW is 0, only fixes that are generated
     * at no additional cost of power shall be reported.
     */
    double max_power_allocation_mW;

    /** Bitwise OR of the FLP_TECH_MASKS to use */
    uint32_t sources_to_use;

    /**
     * FLP_BATCH_WAKEUP_ON_FIFO_FULL - If set the hardware
     * will wake up the AP when the buffer is full. If not set, the
     * hardware will drop the oldest location object.
     *
     * FLP_BATCH_CALLBACK_ON_LOCATION_FIX - If set the location
     * callback will be called every time there is a location fix.
     * Its the responsibility of the upper layers (caller) to switch
     * it off, if it knows that the AP might go to sleep. When this
     * bit is on amidst a batching session, batching should continue
     * while location fixes are reported in real time.
     *
     * Other flags to be bitwised ORed in the future.
     */
    uint32_t flags;

    /**
     * Frequency with which location needs to be batched in nano
     * seconds.
     */
    int64_t period_ns;

    /**
     * The smallest displacement between reported locations in meters.
     *
     * If set to 0, then you should report locations at the requested
     * interval even if the device is stationary.  If positive, you
     * can use this parameter as a hint to save power (e.g. throttling
     * location period if the user hasn't traveled close to the displacement
     * threshold).  Even small positive values can be interpreted to mean
     * that you don't have to compute location when the device is stationary.
     *
     * There is no need to filter location delivery based on this parameter.
     * Locations can be delivered even if they have a displacement smaller than
     * requested. This parameter can safely be ignored at the cost of potential
     * power savings.
     */
    float smallest_displacement_meters;
} FlpBatchOptions;

#define FLP_RESULT_SUCCESS                       0
#define FLP_RESULT_ERROR                        -1
#define FLP_RESULT_INSUFFICIENT_MEMORY          -2
#define FLP_RESULT_TOO_MANY_GEOFENCES           -3
#define FLP_RESULT_ID_EXISTS                    -4
#define FLP_RESULT_ID_UNKNOWN                   -5
#define FLP_RESULT_INVALID_GEOFENCE_TRANSITION  -6

/**
 * Represents the standard FLP interface.
 */
typedef struct {
    /**
     * set to sizeof(FlpLocationInterface)
     */
    size_t size;

    /**
     * Opens the interface and provides the callback routines
     * to the implementation of this interface.  Once called you should respond
     * by calling the flp_capabilities_callback in FlpCallbacks to
     * specify the capabilities that your implementation supports.
     */
    int (*init)(FlpCallbacks* callbacks );

    /**
     * Return the batch size (in number of FlpLocation objects)
     * available in the hardware.  Note, different HW implementations
     * may have different sample sizes.  This shall return number
     * of samples defined in the format of FlpLocation.
     * This will be used by the upper layer, to decide on the batching
     * interval and whether the AP should be woken up or not.
     */
    int (*get_batch_size)();

    /**
     * Start batching locations. This API is primarily used when the AP is
     * asleep and the device can batch locations in the hardware.
     *   flp_location_callback is used to return the locations. When the buffer
     * is full and FLP_BATCH_WAKEUP_ON_FIFO_FULL is used, the AP is woken up.
     * When the buffer is full and FLP_BATCH_WAKEUP_ON_FIFO_FULL is not set,
     * the oldest location object is dropped. In this case the  AP will not be
     * woken up. The upper layer will use get_batched_location
     * API to explicitly ask for the location.
     *   If FLP_BATCH_CALLBACK_ON_LOCATION_FIX is set, the implementation
     * will call the flp_location_callback every single time there is a location
     * fix. This overrides FLP_BATCH_WAKEUP_ON_FIFO_FULL flag setting.
     * It's the responsibility of the upper layers (caller) to switch
     * it off, if it knows that the AP might go to sleep. This is useful
     * for nagivational applications when the system is in high power mode.
     * Parameters:
     *    id - Id for the request.
     *    options - See FlpBatchOptions struct definition.
     * Return value:
     *    FLP_RESULT_SUCCESS on success, FLP_RESULT_INSUFFICIENT_MEMORY,
     *    FLP_RESULT_ID_EXISTS, FLP_RESULT_ERROR on failure.
     */
    int (*start_batching)(int id, FlpBatchOptions* options);

    /**
     * Update FlpBatchOptions associated with a batching request.
     * When a batching operation is in progress and a batching option
     * such as FLP_BATCH_WAKEUP_ON_FIFO_FULL needs to be updated, this API
     * will be used. For instance, this can happen when the AP is awake and
     * the maps application is being used.
     * Parameters:
     *    id - Id of an existing batch request.
     *    new_options - Updated FlpBatchOptions
     * Return value:
     *    FLP_RESULT_SUCCESS on success, FLP_RESULT_ID_UNKNOWN,
     *    FLP_RESULT_ERROR on error.
     */
    int (*update_batching_options)(int id, FlpBatchOptions* new_options);

    /**
     * Stop batching.
     * Parameters:
     *    id - Id for the request.
     * Return Value:
     *    FLP_RESULT_SUCCESS on success, FLP_RESULT_ID_UNKNOWN or
     *    FLP_RESULT_ERROR on failure.
     */
    int (*stop_batching)(int id);

    /**
     * Closes the interface. If any batch operations are in progress,
     * they should be stopped.
     */
    void (*cleanup)();

    /**
     * Get the fused location that was batched.
     *   flp_location_callback is used to return the location. The location object
     * is dropped from the buffer only when the buffer is full. Do not remove it
     * from the buffer just because it has been returned using the callback.
     * In other words, when there is no new location object, two calls to
     * get_batched_location(1) should return the same location object.
     * Parameters:
     *      last_n_locations - Number of locations to get. This can be one or many.
     *      If the last_n_locations is 1, you get the latest location known to the
     *      hardware.
     */
    void (*get_batched_location)(int last_n_locations);

    /**
     * Injects current location from another location provider
     * latitude and longitude are measured in degrees
     * expected accuracy is measured in meters
     * Parameters:
     *      location - The location object being injected.
     * Return value: FLP_RESULT_SUCCESS or FLP_RESULT_ERROR.
     */
    int  (*inject_location)(FlpLocation* location);

    /**
     * Get a pointer to extension information.
     */
    const void* (*get_extension)(const char* name);

    /**
     * Retrieve all batched locations currently stored and clear the buffer.
     * flp_location_callback MUST be called in response, even if there are
     * no locations to flush (in which case num_locations should be 0).
     * Subsequent calls to get_batched_location or flush_batched_locations
     * should not return any of the locations returned in this call.
     */
    void (*flush_batched_locations)();
} FlpLocationInterface;

struct flp_device_t {
    struct hw_device_t common;

    /**
     * Get a handle to the FLP Interface.
     */
    const FlpLocationInterface* (*get_flp_interface)(struct flp_device_t* dev);
};

/**
 * Callback for reports diagnostic data into the Java framework code.
*/
typedef void (*report_data)(char* data, int length);

/**
 * FLP diagnostic callback structure.
 * Currently, not used - but this for future extension.
 */
typedef struct {
    /** set to sizeof(FlpDiagnosticCallbacks) */
    size_t      size;

    flp_set_thread_event set_thread_event_cb;

    /** reports diagnostic data into the Java framework code */
    report_data data_cb;
} FlpDiagnosticCallbacks;

/** Extended interface for diagnostic support. */
typedef struct {
    /** set to sizeof(FlpDiagnosticInterface) */
    size_t          size;

    /**
     * Opens the diagnostic interface and provides the callback routines
     * to the implemenation of this interface.
     */
    void  (*init)(FlpDiagnosticCallbacks* callbacks);

    /**
     * Injects diagnostic data into the FLP subsystem.
     * Return 0 on success, -1 on error.
     **/
    int  (*inject_data)(char* data, int length );
} FlpDiagnosticInterface;

/**
 * Context setting information.
 * All these settings shall be injected to FLP HAL at FLP init time.
 * Following that, only the changed setting need to be re-injected
 * upon changes.
 */

#define FLP_DEVICE_CONTEXT_GPS_ENABLED                     (1U<<0)
#define FLP_DEVICE_CONTEXT_AGPS_ENABLED                    (1U<<1)
#define FLP_DEVICE_CONTEXT_NETWORK_POSITIONING_ENABLED     (1U<<2)
#define FLP_DEVICE_CONTEXT_WIFI_CONNECTIVITY_ENABLED       (1U<<3)
#define FLP_DEVICE_CONTEXT_WIFI_POSITIONING_ENABLED        (1U<<4)
#define FLP_DEVICE_CONTEXT_HW_NETWORK_POSITIONING_ENABLED  (1U<<5)
#define FLP_DEVICE_CONTEXT_AIRPLANE_MODE_ON                (1U<<6)
#define FLP_DEVICE_CONTEXT_DATA_ENABLED                    (1U<<7)
#define FLP_DEVICE_CONTEXT_ROAMING_ENABLED                 (1U<<8)
#define FLP_DEVICE_CONTEXT_CURRENTLY_ROAMING               (1U<<9)
#define FLP_DEVICE_CONTEXT_SENSOR_ENABLED                  (1U<<10)
#define FLP_DEVICE_CONTEXT_BLUETOOTH_ENABLED               (1U<<11)
#define FLP_DEVICE_CONTEXT_CHARGER_ON                      (1U<<12)

/** Extended interface for device context support. */
typedef struct {
    /** set to sizeof(FlpDeviceContextInterface) */
    size_t          size;

    /**
     * Injects debug data into the FLP subsystem.
     * Return 0 on success, -1 on error.
     **/
    int  (*inject_device_context)(uint32_t enabledMask);
} FlpDeviceContextInterface;


/**
 * There are 3 states associated with a Geofence: Inside, Outside, Unknown.
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
 * circle reported by the FLP subsystem. Now with regard to "b", the system is
 * confident that the user is outside. But with regard to "a" is not confident
 * whether it is inside or outside the geofence. If the accuracy remains the
 * same for a sufficient period of time, the UNCERTAIN transition would be
 * triggered with the state set to Unknown. If the accuracy improves later, an
 * appropriate transition should be triggered.  This "sufficient period of time"
 * is defined by the parameter in the add_geofence_area API.
 *     In other words, Unknown state can be interpreted as a state in which the
 * FLP subsystem isn't confident enough that the user is either inside or
 * outside the Geofence. It moves to Unknown state only after the expiry of the
 * timeout.
 *
 * The geofence callback needs to be triggered for the ENTERED and EXITED
 * transitions, when the FLP system is confident that the user has entered
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
 * and then we get an accurate FLP location fix, it needs to trigger the
 * appropriate (ENTERED or EXITED) transition for every Geofence it knows about.
 * By default, all the Geofences will be in the Unknown state.
 *
 * When the FLP system is unavailable, flp_geofence_status_callback should be
 * called to inform the upper layers of the same. Similarly, when it becomes
 * available the callback should be called. This is a global state while the
 * UNKNOWN transition described above is per geofence.
 *
 */
#define FLP_GEOFENCE_TRANSITION_ENTERED     (1L<<0)
#define FLP_GEOFENCE_TRANSITION_EXITED      (1L<<1)
#define FLP_GEOFENCE_TRANSITION_UNCERTAIN   (1L<<2)

#define FLP_GEOFENCE_MONITOR_STATUS_UNAVAILABLE (1L<<0)
#define FLP_GEOFENCE_MONITOR_STATUS_AVAILABLE   (1L<<1)

/**
 * The callback associated with the geofence.
 * Parameters:
 *      geofence_id - The id associated with the add_geofence_area.
 *      location    - The current location as determined by the FLP subsystem.
 *      transition  - Can be one of FLP_GEOFENCE_TRANSITION_ENTERED, FLP_GEOFENCE_TRANSITION_EXITED,
 *                    FLP_GEOFENCE_TRANSITION_UNCERTAIN.
 *      timestamp   - Timestamp when the transition was detected; -1 if not available.
 *      sources_used - Bitwise OR of FLP_TECH_MASK flags indicating which
 *                     subsystems were used.
 *
 * The callback should only be called when the caller is interested in that
 * particular transition. For instance, if the caller is interested only in
 * ENTERED transition, then the callback should NOT be called with the EXITED
 * transition.
 *
 * IMPORTANT: If a transition is triggered resulting in this callback, the
 * subsystem will wake up the application processor, if its in suspend state.
 */
typedef void (*flp_geofence_transition_callback) (int32_t geofence_id,  FlpLocation* location,
        int32_t transition, FlpUtcTime timestamp, uint32_t sources_used);

/**
 * The callback associated with the availablity of one the sources used for geofence
 * monitoring by the FLP sub-system For example, if the GPS system determines that it cannot
 * monitor geofences because of lack of reliability or unavailability of the GPS signals,
 * it will call this callback with FLP_GEOFENCE_MONITOR_STATUS_UNAVAILABLE parameter and the
 * source set to FLP_TECH_MASK_GNSS.
 *
 * Parameters:
 *  status - FLP_GEOFENCE_MONITOR_STATUS_UNAVAILABLE or FLP_GEOFENCE_MONITOR_STATUS_AVAILABLE.
 *  source - One of the FLP_TECH_MASKS
 *  last_location - Last known location.
 */
typedef void (*flp_geofence_monitor_status_callback) (int32_t status, uint32_t source,
                                                      FlpLocation* last_location);

/**
 * The callback associated with the add_geofence call.
 *
 * Parameter:
 * geofence_id - Id of the geofence.
 * result - FLP_RESULT_SUCCESS
 *          FLP_RESULT_ERROR_TOO_MANY_GEOFENCES  - geofence limit has been reached.
 *          FLP_RESULT_ID_EXISTS  - geofence with id already exists
 *          FLP_RESULT_INVALID_GEOFENCE_TRANSITION - the monitorTransition contains an
 *              invalid transition
 *          FLP_RESULT_ERROR - for other errors.
 */
typedef void (*flp_geofence_add_callback) (int32_t geofence_id, int32_t result);

/**
 * The callback associated with the remove_geofence call.
 *
 * Parameter:
 * geofence_id - Id of the geofence.
 * result - FLP_RESULT_SUCCESS
 *          FLP_RESULT_ID_UNKNOWN - for invalid id
 *          FLP_RESULT_ERROR for others.
 */
typedef void (*flp_geofence_remove_callback) (int32_t geofence_id, int32_t result);


/**
 * The callback associated with the pause_geofence call.
 *
 * Parameter:
 * geofence_id - Id of the geofence.
 * result - FLP_RESULT_SUCCESS
 *          FLP_RESULT__ID_UNKNOWN - for invalid id
 *          FLP_RESULT_INVALID_TRANSITION -
 *                    when monitor_transitions is invalid
 *          FLP_RESULT_ERROR for others.
 */
typedef void (*flp_geofence_pause_callback) (int32_t geofence_id, int32_t result);

/**
 * The callback associated with the resume_geofence call.
 *
 * Parameter:
 * geofence_id - Id of the geofence.
 * result - FLP_RESULT_SUCCESS
 *          FLP_RESULT_ID_UNKNOWN - for invalid id
 *          FLP_RESULT_ERROR for others.
 */
typedef void (*flp_geofence_resume_callback) (int32_t geofence_id, int32_t result);

typedef struct {
    /** set to sizeof(FlpGeofenceCallbacks) */
    size_t size;
    flp_geofence_transition_callback geofence_transition_callback;
    flp_geofence_monitor_status_callback geofence_status_callback;
    flp_geofence_add_callback geofence_add_callback;
    flp_geofence_remove_callback geofence_remove_callback;
    flp_geofence_pause_callback geofence_pause_callback;
    flp_geofence_resume_callback geofence_resume_callback;
    flp_set_thread_event set_thread_event_cb;
    flp_capabilities_callback flp_capabilities_cb;
} FlpGeofenceCallbacks;


/** Type of geofence */
typedef enum {
    TYPE_CIRCLE = 0,
} GeofenceType;

/** Circular geofence is represented by lat / long / radius */
typedef struct {
    double latitude;
    double longitude;
    double radius_m;
} GeofenceCircle;

/** Represents the type of geofence and data */
typedef struct {
    GeofenceType type;
    union {
        GeofenceCircle circle;
    } geofence;
} GeofenceData;

/** Geofence Options */
typedef struct {
   /**
    * The current state of the geofence. For example, if
    * the system already knows that the user is inside the geofence,
    * this will be set to FLP_GEOFENCE_TRANSITION_ENTERED. In most cases, it
    * will be FLP_GEOFENCE_TRANSITION_UNCERTAIN. */
    int last_transition;

   /**
    * Transitions to monitor. Bitwise OR of
    * FLP_GEOFENCE_TRANSITION_ENTERED, FLP_GEOFENCE_TRANSITION_EXITED and
    * FLP_GEOFENCE_TRANSITION_UNCERTAIN.
    */
    int monitor_transitions;

   /**
    * Defines the best-effort description
    * of how soon should the callback be called when the transition
    * associated with the Geofence is triggered. For instance, if set
    * to 1000 millseconds with FLP_GEOFENCE_TRANSITION_ENTERED, the callback
    * should be called 1000 milliseconds within entering the geofence.
    * This parameter is defined in milliseconds.
    * NOTE: This is not to be confused with the rate that the GPS is
    * polled at. It is acceptable to dynamically vary the rate of
    * sampling the GPS for power-saving reasons; thus the rate of
    * sampling may be faster or slower than this.
    */
    int notification_responsivenes_ms;

   /**
    * The time limit after which the UNCERTAIN transition
    * should be triggered. This paramter is defined in milliseconds.
    */
    int unknown_timer_ms;

    /**
     * The sources to use for monitoring geofences. Its a BITWISE-OR
     * of FLP_TECH_MASK flags.
     */
    uint32_t sources_to_use;
} GeofenceOptions;

/** Geofence struct */
typedef struct {
    int32_t geofence_id;
    GeofenceData* data;
    GeofenceOptions* options;
} Geofence;

/** Extended interface for FLP_Geofencing support */
typedef struct {
   /** set to sizeof(FlpGeofencingInterface) */
   size_t          size;

   /**
    * Opens the geofence interface and provides the callback routines
    * to the implemenation of this interface.  Once called you should respond
    * by calling the flp_capabilities_callback in FlpGeofenceCallbacks to
    * specify the capabilities that your implementation supports.
    */
   void  (*init)( FlpGeofenceCallbacks* callbacks );

   /**
    * Add a list of geofences.
    * Parameters:
    *     number_of_geofences - The number of geofences that needed to be added.
    *     geofences - Pointer to array of pointers to Geofence structure.
    */
   void (*add_geofences) (int32_t number_of_geofences, Geofence** geofences);

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
    *       FLP_GEOFENCE_TRANSITION_ENTERED, FLP_GEOFENCE_TRANSITION_EXITED and
    *       FLP_GEOFENCE_TRANSITION_UNCERTAIN.
    *       This supersedes the value associated provided in the
    *       add_geofence_area call.
    */
   void (*resume_geofence) (int32_t geofence_id, int monitor_transitions);

   /**
    * Modify a particular geofence option.
    * Parameters:
    *    geofence_id - The id for the geofence.
    *    options - Various options associated with the geofence. See
    *        GeofenceOptions structure for details.
    */
   void (*modify_geofence_option) (int32_t geofence_id, GeofenceOptions* options);

   /**
    * Remove a list of geofences. After the function returns, no notifications
    * should be sent.
    * Parameter:
    *     number_of_geofences - The number of geofences that needed to be added.
    *     geofence_id - Pointer to array of geofence_ids to be removed.
    */
   void (*remove_geofences) (int32_t number_of_geofences, int32_t* geofence_id);
} FlpGeofencingInterface;

__END_DECLS

#endif /* ANDROID_INCLUDE_HARDWARE_FLP_H */

