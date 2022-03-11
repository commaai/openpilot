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

/**
 * @addtogroup Sensor
 * @{
 */

/**
 * @file sensor.h
 */

#ifndef ANDROID_SENSOR_H
#define ANDROID_SENSOR_H

/******************************************************************
 *
 * IMPORTANT NOTICE:
 *
 *   This file is part of Android's set of stable system headers
 *   exposed by the Android NDK (Native Development Kit).
 *
 *   Third-party source AND binary code relies on the definitions
 *   here to be FROZEN ON ALL UPCOMING PLATFORM RELEASES.
 *
 *   - DO NOT MODIFY ENUMS (EXCEPT IF YOU ADD NEW 32-BIT VALUES)
 *   - DO NOT MODIFY CONSTANTS OR FUNCTIONAL MACROS
 *   - DO NOT CHANGE THE SIGNATURE OF FUNCTIONS IN ANY WAY
 *   - DO NOT CHANGE THE LAYOUT OR SIZE OF STRUCTURES
 */

/**
 * Structures and functions to receive and process sensor events in
 * native code.
 *
 */

#include <android/looper.h>

#include <stdbool.h>
#include <sys/types.h>
#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AHardwareBuffer AHardwareBuffer;

#define ASENSOR_RESOLUTION_INVALID     (nanf(""))
#define ASENSOR_FIFO_COUNT_INVALID     (-1)
#define ASENSOR_DELAY_INVALID          INT32_MIN

/* (Keep in sync with hardware/sensors-base.h and Sensor.java.) */

/**
 * Sensor types.
 *
 * See
 * [android.hardware.SensorEvent#values](https://developer.android.com/reference/android/hardware/SensorEvent.html#values)
 * for detailed explanations of the data returned for each of these types.
 */
enum {
    /**
     * Invalid sensor type. Returned by {@link ASensor_getType} as error value.
     */
    ASENSOR_TYPE_INVALID = -1,
    /**
     * {@link ASENSOR_TYPE_ACCELEROMETER}
     * reporting-mode: continuous
     *
     *  All values are in SI units (m/s^2) and measure the acceleration of the
     *  device minus the force of gravity.
     */
    ASENSOR_TYPE_ACCELEROMETER       = 1,
    /**
     * {@link ASENSOR_TYPE_MAGNETIC_FIELD}
     * reporting-mode: continuous
     *
     *  All values are in micro-Tesla (uT) and measure the geomagnetic
     *  field in the X, Y and Z axis.
     */
    ASENSOR_TYPE_MAGNETIC_FIELD      = 2,
    /**
     * {@link ASENSOR_TYPE_GYROSCOPE}
     * reporting-mode: continuous
     *
     *  All values are in radians/second and measure the rate of rotation
     *  around the X, Y and Z axis.
     */
    ASENSOR_TYPE_GYROSCOPE           = 4,
    /**
     * {@link ASENSOR_TYPE_LIGHT}
     * reporting-mode: on-change
     *
     * The light sensor value is returned in SI lux units.
     */
    ASENSOR_TYPE_LIGHT               = 5,
    /**
     * {@link ASENSOR_TYPE_PRESSURE}
     *
     * The pressure sensor value is returned in hPa (millibar).
     */
    ASENSOR_TYPE_PRESSURE            = 6,
    /**
     * {@link ASENSOR_TYPE_PROXIMITY}
     * reporting-mode: on-change
     *
     * The proximity sensor which turns the screen off and back on during calls is the
     * wake-up proximity sensor. Implement wake-up proximity sensor before implementing
     * a non wake-up proximity sensor. For the wake-up proximity sensor set the flag
     * SENSOR_FLAG_WAKE_UP.
     * The value corresponds to the distance to the nearest object in centimeters.
     */
    ASENSOR_TYPE_PROXIMITY           = 8,
    /**
     * {@link ASENSOR_TYPE_GRAVITY}
     *
     * All values are in SI units (m/s^2) and measure the direction and
     * magnitude of gravity. When the device is at rest, the output of
     * the gravity sensor should be identical to that of the accelerometer.
     */
    ASENSOR_TYPE_GRAVITY             = 9,
    /**
     * {@link ASENSOR_TYPE_LINEAR_ACCELERATION}
     * reporting-mode: continuous
     *
     *  All values are in SI units (m/s^2) and measure the acceleration of the
     *  device not including the force of gravity.
     */
    ASENSOR_TYPE_LINEAR_ACCELERATION = 10,
    /**
     * {@link ASENSOR_TYPE_ROTATION_VECTOR}
     */
    ASENSOR_TYPE_ROTATION_VECTOR     = 11,
    /**
     * {@link ASENSOR_TYPE_RELATIVE_HUMIDITY}
     *
     * The relative humidity sensor value is returned in percent.
     */
    ASENSOR_TYPE_RELATIVE_HUMIDITY   = 12,
    /**
     * {@link ASENSOR_TYPE_AMBIENT_TEMPERATURE}
     *
     * The ambient temperature sensor value is returned in Celcius.
     */
    ASENSOR_TYPE_AMBIENT_TEMPERATURE = 13,
    /**
     * {@link ASENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED}
     */
    ASENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED = 14,
    /**
     * {@link ASENSOR_TYPE_GAME_ROTATION_VECTOR}
     */
    ASENSOR_TYPE_GAME_ROTATION_VECTOR = 15,
    /**
     * {@link ASENSOR_TYPE_GYROSCOPE_UNCALIBRATED}
     */
    ASENSOR_TYPE_GYROSCOPE_UNCALIBRATED = 16,
    /**
     * {@link ASENSOR_TYPE_SIGNIFICANT_MOTION}
     */
    ASENSOR_TYPE_SIGNIFICANT_MOTION = 17,
    /**
     * {@link ASENSOR_TYPE_STEP_DETECTOR}
     */
    ASENSOR_TYPE_STEP_DETECTOR = 18,
    /**
     * {@link ASENSOR_TYPE_STEP_COUNTER}
     */
    ASENSOR_TYPE_STEP_COUNTER = 19,
    /**
     * {@link ASENSOR_TYPE_GEOMAGNETIC_ROTATION_VECTOR}
     */
    ASENSOR_TYPE_GEOMAGNETIC_ROTATION_VECTOR = 20,
    /**
     * {@link ASENSOR_TYPE_HEART_RATE}
     */
    ASENSOR_TYPE_HEART_RATE = 21,
    /**
     * {@link ASENSOR_TYPE_POSE_6DOF}
     */
    ASENSOR_TYPE_POSE_6DOF = 28,
    /**
     * {@link ASENSOR_TYPE_STATIONARY_DETECT}
     */
    ASENSOR_TYPE_STATIONARY_DETECT = 29,
    /**
     * {@link ASENSOR_TYPE_MOTION_DETECT}
     */
    ASENSOR_TYPE_MOTION_DETECT = 30,
    /**
     * {@link ASENSOR_TYPE_HEART_BEAT}
     */
    ASENSOR_TYPE_HEART_BEAT = 31,
    /**
     * {@link ASENSOR_TYPE_LOW_LATENCY_OFFBODY_DETECT}
     */
    ASENSOR_TYPE_LOW_LATENCY_OFFBODY_DETECT = 34,
    /**
     * {@link ASENSOR_TYPE_ACCELEROMETER_UNCALIBRATED}
     */
    ASENSOR_TYPE_ACCELEROMETER_UNCALIBRATED = 35,
};

/**
 * Sensor accuracy measure.
 */
enum {
    /** no contact */
    ASENSOR_STATUS_NO_CONTACT       = -1,
    /** unreliable */
    ASENSOR_STATUS_UNRELIABLE       = 0,
    /** low accuracy */
    ASENSOR_STATUS_ACCURACY_LOW     = 1,
    /** medium accuracy */
    ASENSOR_STATUS_ACCURACY_MEDIUM  = 2,
    /** high accuracy */
    ASENSOR_STATUS_ACCURACY_HIGH    = 3
};

/**
 * Sensor Reporting Modes.
 */
enum {
    /** invalid reporting mode */
    AREPORTING_MODE_INVALID = -1,
    /** continuous reporting */
    AREPORTING_MODE_CONTINUOUS = 0,
    /** reporting on change */
    AREPORTING_MODE_ON_CHANGE = 1,
    /** on shot reporting */
    AREPORTING_MODE_ONE_SHOT = 2,
    /** special trigger reporting */
    AREPORTING_MODE_SPECIAL_TRIGGER = 3
};

/**
 * Sensor Direct Report Rates.
 */
enum {
    /** stopped */
    ASENSOR_DIRECT_RATE_STOP = 0,
    /** nominal 50Hz */
    ASENSOR_DIRECT_RATE_NORMAL = 1,
    /** nominal 200Hz */
    ASENSOR_DIRECT_RATE_FAST = 2,
    /** nominal 800Hz */
    ASENSOR_DIRECT_RATE_VERY_FAST = 3
};

/**
 * Sensor Direct Channel Type.
 */
enum {
    /** shared memory created by ASharedMemory_create */
    ASENSOR_DIRECT_CHANNEL_TYPE_SHARED_MEMORY = 1,
    /** AHardwareBuffer */
    ASENSOR_DIRECT_CHANNEL_TYPE_HARDWARE_BUFFER = 2
};

/*
 * A few useful constants
 */

/** Earth's gravity in m/s^2 */
#define ASENSOR_STANDARD_GRAVITY            (9.80665f)
/** Maximum magnetic field on Earth's surface in uT */
#define ASENSOR_MAGNETIC_FIELD_EARTH_MAX    (60.0f)
/** Minimum magnetic field on Earth's surface in uT*/
#define ASENSOR_MAGNETIC_FIELD_EARTH_MIN    (30.0f)

/**
 * A sensor event.
 */

/* NOTE: changes to these structs have to be backward compatible */
typedef struct ASensorVector {
    union {
        float v[3];
        struct {
            float x;
            float y;
            float z;
        };
        struct {
            float azimuth;
            float pitch;
            float roll;
        };
    };
    int8_t status;
    uint8_t reserved[3];
} ASensorVector;

typedef struct AMetaDataEvent {
    int32_t what;
    int32_t sensor;
} AMetaDataEvent;

typedef struct AUncalibratedEvent {
    union {
        float uncalib[3];
        struct {
            float x_uncalib;
            float y_uncalib;
            float z_uncalib;
        };
    };
    union {
        float bias[3];
        struct {
            float x_bias;
            float y_bias;
            float z_bias;
        };
    };
} AUncalibratedEvent;

typedef struct AHeartRateEvent {
    float bpm;
    int8_t status;
} AHeartRateEvent;

typedef struct ADynamicSensorEvent {
    int32_t  connected;
    int32_t  handle;
} ADynamicSensorEvent;

typedef struct {
    int32_t type;
    int32_t serial;
    union {
        int32_t data_int32[14];
        float   data_float[14];
    };
} AAdditionalInfoEvent;

/* NOTE: changes to this struct has to be backward compatible */
typedef struct ASensorEvent {
    int32_t version; /* sizeof(struct ASensorEvent) */
    int32_t sensor;
    int32_t type;
    int32_t reserved0;
    int64_t timestamp;
    union {
        union {
            float           data[16];
            ASensorVector   vector;
            ASensorVector   acceleration;
            ASensorVector   magnetic;
            float           temperature;
            float           distance;
            float           light;
            float           pressure;
            float           relative_humidity;
            AUncalibratedEvent uncalibrated_gyro;
            AUncalibratedEvent uncalibrated_magnetic;
            AMetaDataEvent meta_data;
            AHeartRateEvent heart_rate;
            ADynamicSensorEvent dynamic_sensor_meta;
            AAdditionalInfoEvent additional_info;
        };
        union {
            uint64_t        data[8];
            uint64_t        step_counter;
        } u64;
    };

    uint32_t flags;
    int32_t reserved1[3];
} ASensorEvent;

struct ASensorManager;
/**
 * {@link ASensorManager} is an opaque type to manage sensors and
 * events queues.
 *
 * {@link ASensorManager} is a singleton that can be obtained using
 * ASensorManager_getInstance().
 *
 * This file provides a set of functions that uses {@link
 * ASensorManager} to access and list hardware sensors, and
 * create and destroy event queues:
 * - ASensorManager_getSensorList()
 * - ASensorManager_getDefaultSensor()
 * - ASensorManager_getDefaultSensorEx()
 * - ASensorManager_createEventQueue()
 * - ASensorManager_destroyEventQueue()
 */
typedef struct ASensorManager ASensorManager;


struct ASensorEventQueue;
/**
 * {@link ASensorEventQueue} is an opaque type that provides access to
 * {@link ASensorEvent} from hardware sensors.
 *
 * A new {@link ASensorEventQueue} can be obtained using ASensorManager_createEventQueue().
 *
 * This file provides a set of functions to enable and disable
 * sensors, check and get events, and set event rates on a {@link
 * ASensorEventQueue}.
 * - ASensorEventQueue_enableSensor()
 * - ASensorEventQueue_disableSensor()
 * - ASensorEventQueue_hasEvents()
 * - ASensorEventQueue_getEvents()
 * - ASensorEventQueue_setEventRate()
 */
typedef struct ASensorEventQueue ASensorEventQueue;

struct ASensor;
/**
 * {@link ASensor} is an opaque type that provides information about
 * an hardware sensors.
 *
 * A {@link ASensor} pointer can be obtained using
 * ASensorManager_getDefaultSensor(),
 * ASensorManager_getDefaultSensorEx() or from a {@link ASensorList}.
 *
 * This file provides a set of functions to access properties of a
 * {@link ASensor}:
 * - ASensor_getName()
 * - ASensor_getVendor()
 * - ASensor_getType()
 * - ASensor_getResolution()
 * - ASensor_getMinDelay()
 * - ASensor_getFifoMaxEventCount()
 * - ASensor_getFifoReservedEventCount()
 * - ASensor_getStringType()
 * - ASensor_getReportingMode()
 * - ASensor_isWakeUpSensor()
 */
typedef struct ASensor ASensor;
/**
 * {@link ASensorRef} is a type for constant pointers to {@link ASensor}.
 *
 * This is used to define entry in {@link ASensorList} arrays.
 */
typedef ASensor const* ASensorRef;
/**
 * {@link ASensorList} is an array of reference to {@link ASensor}.
 *
 * A {@link ASensorList} can be initialized using ASensorManager_getSensorList().
 */
typedef ASensorRef const* ASensorList;

/*****************************************************************************/

/**
 * Get a reference to the sensor manager. ASensorManager is a singleton
 * per package as different packages may have access to different sensors.
 *
 * Deprecated: Use ASensorManager_getInstanceForPackage(const char*) instead.
 *
 * Example:
 *
 *     ASensorManager* sensorManager = ASensorManager_getInstance();
 *
 */
#if __ANDROID_API__ >= __ANDROID_API_O__
__attribute__ ((deprecated)) ASensorManager* ASensorManager_getInstance();
#else
ASensorManager* ASensorManager_getInstance();
#endif

#if __ANDROID_API__ >= __ANDROID_API_O__
/**
 * Get a reference to the sensor manager. ASensorManager is a singleton
 * per package as different packages may have access to different sensors.
 *
 * Example:
 *
 *     ASensorManager* sensorManager = ASensorManager_getInstanceForPackage("foo.bar.baz");
 *
 */
ASensorManager* ASensorManager_getInstanceForPackage(const char* packageName);
#endif

/**
 * Returns the list of available sensors.
 */
int ASensorManager_getSensorList(ASensorManager* manager, ASensorList* list);

/**
 * Returns the default sensor for the given type, or NULL if no sensor
 * of that type exists.
 */
ASensor const* ASensorManager_getDefaultSensor(ASensorManager* manager, int type);

#if __ANDROID_API__ >= 21
/**
 * Returns the default sensor with the given type and wakeUp properties or NULL if no sensor
 * of this type and wakeUp properties exists.
 */
ASensor const* ASensorManager_getDefaultSensorEx(ASensorManager* manager, int type, bool wakeUp);
#endif

/**
 * Creates a new sensor event queue and associate it with a looper.
 *
 * "ident" is a identifier for the events that will be returned when
 * calling ALooper_pollOnce(). The identifier must be >= 0, or
 * ALOOPER_POLL_CALLBACK if providing a non-NULL callback.
 */
ASensorEventQueue* ASensorManager_createEventQueue(ASensorManager* manager,
        ALooper* looper, int ident, ALooper_callbackFunc callback, void* data);

/**
 * Destroys the event queue and free all resources associated to it.
 */
int ASensorManager_destroyEventQueue(ASensorManager* manager, ASensorEventQueue* queue);

#if __ANDROID_API__ >= __ANDROID_API_O__
/**
 * Create direct channel based on shared memory
 *
 * Create a direct channel of {@link ASENSOR_DIRECT_CHANNEL_TYPE_SHARED_MEMORY} to be used
 * for configuring sensor direct report.
 *
 * \param manager the {@link ASensorManager} instance obtained from
 *                {@link ASensorManager_getInstanceForPackage}.
 * \param fd      file descriptor representing a shared memory created by
 *                {@link ASharedMemory_create}
 * \param size    size to be used, must be less or equal to size of shared memory.
 *
 * \return a positive integer as a channel id to be used in
 *         {@link ASensorManager_destroyDirectChannel} and
 *         {@link ASensorManager_configureDirectReport}, or value less or equal to 0 for failures.
 */
int ASensorManager_createSharedMemoryDirectChannel(ASensorManager* manager, int fd, size_t size);

/**
 * Create direct channel based on AHardwareBuffer
 *
 * Create a direct channel of {@link ASENSOR_DIRECT_CHANNEL_TYPE_HARDWARE_BUFFER} type to be used
 * for configuring sensor direct report.
 *
 * \param manager the {@link ASensorManager} instance obtained from
 *                {@link ASensorManager_getInstanceForPackage}.
 * \param buffer  {@link AHardwareBuffer} instance created by {@link AHardwareBuffer_allocate}.
 * \param size    the intended size to be used, must be less or equal to size of buffer.
 *
 * \return a positive integer as a channel id to be used in
 *         {@link ASensorManager_destroyDirectChannel} and
 *         {@link ASensorManager_configureDirectReport}, or value less or equal to 0 for failures.
 */
int ASensorManager_createHardwareBufferDirectChannel(
        ASensorManager* manager, AHardwareBuffer const * buffer, size_t size);

/**
 * Destroy a direct channel
 *
 * Destroy a direct channel previously created using {@link ASensorManager_createDirectChannel}.
 * The buffer used for creating direct channel does not get destroyed with
 * {@link ASensorManager_destroy} and has to be close or released separately.
 *
 * \param manager the {@link ASensorManager} instance obtained from
 *                {@link ASensorManager_getInstanceForPackage}.
 * \param channelId channel id (a positive integer) returned from
 *                  {@link ASensorManager_createSharedMemoryDirectChannel} or
 *                  {@link ASensorManager_createHardwareBufferDirectChannel}.
 */
void ASensorManager_destroyDirectChannel(ASensorManager* manager, int channelId);

/**
 * Configure direct report on channel
 *
 * Configure sensor direct report on a direct channel: set rate to value other than
 * {@link ASENSOR_DIRECT_RATE_STOP} so that sensor event can be directly
 * written into the shared memory region used for creating the buffer. It returns a positive token
 * which can be used for identify sensor events from different sensors on success. Calling with rate
 * {@link ASENSOR_DIRECT_RATE_STOP} will stop direct report of the sensor specified in the channel.
 *
 * To stop all active sensor direct report configured to a channel, set sensor to NULL and rate to
 * {@link ASENSOR_DIRECT_RATE_STOP}.
 *
 * In order to successfully configure a direct report, the sensor has to support the specified rate
 * and the channel type, which can be checked by {@link ASensor_getHighestDirectReportRateLevel} and
 * {@link ASensor_isDirectChannelTypeSupported}, respectively.
 *
 * Example:
 *
 *     ASensorManager *manager = ...;
 *     ASensor *sensor = ...;
 *     int channelId = ...;
 *
 *     ASensorManager_configureDirectReport(manager, sensor, channel_id, ASENSOR_DIRECT_RATE_FAST);
 *
 * \param manager   the {@link ASensorManager} instance obtained from
 *                  {@link ASensorManager_getInstanceForPackage}.
 * \param sensor    a {@link ASensor} to denote which sensor to be operate. It can be NULL if rate
 *                  is {@link ASENSOR_DIRECT_RATE_STOP}, denoting stopping of all active sensor
 *                  direct report.
 * \param channelId channel id (a positive integer) returned from
 *                  {@link ASensorManager_createSharedMemoryDirectChannel} or
 *                  {@link ASensorManager_createHardwareBufferDirectChannel}.
 *
 * \return positive token for success or negative error code.
 */
int ASensorManager_configureDirectReport(
        ASensorManager* manager, ASensor const* sensor, int channelId, int rate);
#endif

/*****************************************************************************/

/**
 * Enable the selected sensor with sampling and report parameters
 *
 * Enable the selected sensor at a specified sampling period and max batch report latency.
 * To disable  sensor, use {@link ASensorEventQueue_disableSensor}.
 *
 * \param queue {@link ASensorEventQueue} for sensor event to be report to.
 * \param sensor {@link ASensor} to be enabled.
 * \param samplingPeriodUs sampling period of sensor in microseconds.
 * \param maxBatchReportLatencyus maximum time interval between two batch of sensor events are
 *                                delievered in microseconds. For sensor streaming, set to 0.
 * \return 0 on success or a negative error code on failure.
 */
int ASensorEventQueue_registerSensor(ASensorEventQueue* queue, ASensor const* sensor,
        int32_t samplingPeriodUs, int64_t maxBatchReportLatencyUs);

/**
 * Enable the selected sensor at default sampling rate.
 *
 * Start event reports of a sensor to specified sensor event queue at a default rate.
 *
 * \param queue {@link ASensorEventQueue} for sensor event to be report to.
 * \param sensor {@link ASensor} to be enabled.
 *
 * \return 0 on success or a negative error code on failure.
 */
int ASensorEventQueue_enableSensor(ASensorEventQueue* queue, ASensor const* sensor);

/**
 * Disable the selected sensor.
 *
 * Stop event reports from the sensor to specified sensor event queue.
 *
 * \param queue {@link ASensorEventQueue} to be changed
 * \param sensor {@link ASensor} to be disabled
 * \return 0 on success or a negative error code on failure.
 */
int ASensorEventQueue_disableSensor(ASensorEventQueue* queue, ASensor const* sensor);

/**
 * Sets the delivery rate of events in microseconds for the given sensor.
 *
 * This function has to be called after {@link ASensorEventQueue_enableSensor}.
 * Note that this is a hint only, generally event will arrive at a higher
 * rate. It is an error to set a rate inferior to the value returned by
 * ASensor_getMinDelay().
 *
 * \param queue {@link ASensorEventQueue} to which sensor event is delivered.
 * \param sensor {@link ASensor} of which sampling rate to be updated.
 * \param usec sensor sampling period (1/sampling rate) in microseconds
 * \return 0 on sucess or a negative error code on failure.
 */
int ASensorEventQueue_setEventRate(ASensorEventQueue* queue, ASensor const* sensor, int32_t usec);

/**
 * Determine if a sensor event queue has pending event to be processed.
 *
 * \param queue {@link ASensorEventQueue} to be queried
 * \return 1 if the queue has events; 0 if it does not have events;
 *         or a negative value if there is an error.
 */
int ASensorEventQueue_hasEvents(ASensorEventQueue* queue);

/**
 * Retrieve pending events in sensor event queue
 *
 * Retrieve next available events from the queue to a specified event array.
 *
 * \param queue {@link ASensorEventQueue} to get events from
 * \param events pointer to an array of {@link ASensorEvents}.
 * \param count max number of event that can be filled into array event.
 * \return number of events returned on success; negative error code when
 *         no events are pending or an error has occurred.
 *
 * Examples:
 *
 *     ASensorEvent event;
 *     ssize_t numEvent = ASensorEventQueue_getEvents(queue, &event, 1);
 *
 *     ASensorEvent eventBuffer[8];
 *     ssize_t numEvent = ASensorEventQueue_getEvents(queue, eventBuffer, 8);
 *
 */
ssize_t ASensorEventQueue_getEvents(ASensorEventQueue* queue, ASensorEvent* events, size_t count);


/*****************************************************************************/

/**
 * Returns this sensor's name (non localized)
 */
const char* ASensor_getName(ASensor const* sensor);

/**
 * Returns this sensor's vendor's name (non localized)
 */
const char* ASensor_getVendor(ASensor const* sensor);

/**
 * Return this sensor's type
 */
int ASensor_getType(ASensor const* sensor);

/**
 * Returns this sensors's resolution
 */
float ASensor_getResolution(ASensor const* sensor);

/**
 * Returns the minimum delay allowed between events in microseconds.
 * A value of zero means that this sensor doesn't report events at a
 * constant rate, but rather only when a new data is available.
 */
int ASensor_getMinDelay(ASensor const* sensor);

#if __ANDROID_API__ >= 21
/**
 * Returns the maximum size of batches for this sensor. Batches will often be
 * smaller, as the hardware fifo might be used for other sensors.
 */
int ASensor_getFifoMaxEventCount(ASensor const* sensor);

/**
 * Returns the hardware batch fifo size reserved to this sensor.
 */
int ASensor_getFifoReservedEventCount(ASensor const* sensor);

/**
 * Returns this sensor's string type.
 */
const char* ASensor_getStringType(ASensor const* sensor);

/**
 * Returns the reporting mode for this sensor. One of AREPORTING_MODE_* constants.
 */
int ASensor_getReportingMode(ASensor const* sensor);

/**
 * Returns true if this is a wake up sensor, false otherwise.
 */
bool ASensor_isWakeUpSensor(ASensor const* sensor);
#endif /* __ANDROID_API__ >= 21 */

#if __ANDROID_API__ >= __ANDROID_API_O__
/**
 * Test if sensor supports a certain type of direct channel.
 *
 * \param sensor  a {@link ASensor} to denote the sensor to be checked.
 * \param channelType  Channel type constant, either
 *                     {@ASENSOR_DIRECT_CHANNEL_TYPE_SHARED_MEMORY}
 *                     or {@link ASENSOR_DIRECT_CHANNEL_TYPE_HARDWARE_BUFFER}.
 * \returns true if sensor supports the specified direct channel type.
 */
bool ASensor_isDirectChannelTypeSupported(ASensor const* sensor, int channelType);
/**
 * Get the highest direct rate level that a sensor support.
 *
 * \param sensor  a {@link ASensor} to denote the sensor to be checked.
 *
 * \return a ASENSOR_DIRECT_RATE_... enum denoting the highest rate level supported by the sensor.
 *         If return value is {@link ASENSOR_DIRECT_RATE_STOP}, it means the sensor
 *         does not support direct report.
 */
int ASensor_getHighestDirectReportRateLevel(ASensor const* sensor);
#endif

#ifdef __cplusplus
};
#endif

#endif // ANDROID_SENSOR_H

/** @} */
