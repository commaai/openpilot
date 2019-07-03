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

#include <sys/types.h>

#include <android/looper.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Sensor types.
 * (keep in sync with hardware/sensor.h)
 */
enum {
    /**
     * {@link ASENSOR_TYPE_ACCELEROMETER}
     * reporting-mode: continuous
     *
     *  All values are in SI units (m/s^2) and measure the acceleration of the
     *  device minus the force of gravity.
     */
    ASENSOR_TYPE_ACCELEROMETER      = 1,
    /**
     * {@link ASENSOR_TYPE_MAGNETIC_FIELD}
     * reporting-mode: continuous
     *
     *  All values are in micro-Tesla (uT) and measure the geomagnetic
     *  field in the X, Y and Z axis.
     */
    ASENSOR_TYPE_MAGNETIC_FIELD     = 2,
    /**
     * {@link ASENSOR_TYPE_GYROSCOPE}
     * reporting-mode: continuous
     *
     *  All values are in radians/second and measure the rate of rotation
     *  around the X, Y and Z axis.
     */
    ASENSOR_TYPE_GYROSCOPE          = 4,
    /**
     * {@link ASENSOR_TYPE_LIGHT}
     * reporting-mode: on-change
     *
     * The light sensor value is returned in SI lux units.
     */
    ASENSOR_TYPE_LIGHT              = 5,
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
    ASENSOR_TYPE_PROXIMITY          = 8
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
    /** continuous reporting */
    AREPORTING_MODE_CONTINUOUS = 0,
    /** reporting on change */
    AREPORTING_MODE_ON_CHANGE = 1,
    /** on shot reporting */
    AREPORTING_MODE_ONE_SHOT = 2,
    /** special trigger reporting */
    AREPORTING_MODE_SPECIAL_TRIGGER = 3
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

/* NOTE: Must match hardware/sensors.h */
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

/* NOTE: Must match hardware/sensors.h */
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
__attribute__ ((deprecated)) ASensorManager* ASensorManager_getInstance();

/*
 * Get a reference to the sensor manager. ASensorManager is a singleton
 * per package as different packages may have access to different sensors.
 *
 * Example:
 *
 *    ASensorManager* sensorManager = ASensorManager_getInstanceForPackage("foo.bar.baz");
 *
 */
ASensorManager* ASensorManager_getInstanceForPackage(const char* packageName);

/**
 * Returns the list of available sensors.
 */
int ASensorManager_getSensorList(ASensorManager* manager, ASensorList* list);

/**
 * Returns the default sensor for the given type, or NULL if no sensor
 * of that type exists.
 */
ASensor const* ASensorManager_getDefaultSensor(ASensorManager* manager, int type);

/**
 * Returns the default sensor with the given type and wakeUp properties or NULL if no sensor
 * of this type and wakeUp properties exists.
 */
ASensor const* ASensorManager_getDefaultSensorEx(ASensorManager* manager, int type,
        bool wakeUp);

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


/*****************************************************************************/

/**
 * Enable the selected sensor. Returns a negative error code on failure.
 */
int ASensorEventQueue_enableSensor(ASensorEventQueue* queue, ASensor const* sensor);

/**
 * Disable the selected sensor. Returns a negative error code on failure.
 */
int ASensorEventQueue_disableSensor(ASensorEventQueue* queue, ASensor const* sensor);

/**
 * Sets the delivery rate of events in microseconds for the given sensor.
 * Note that this is a hint only, generally event will arrive at a higher
 * rate. It is an error to set a rate inferior to the value returned by
 * ASensor_getMinDelay().
 * Returns a negative error code on failure.
 */
int ASensorEventQueue_setEventRate(ASensorEventQueue* queue, ASensor const* sensor, int32_t usec);

/**
 * Returns true if there are one or more events available in the
 * sensor queue.  Returns 1 if the queue has events; 0 if
 * it does not have events; and a negative value if there is an error.
 */
int ASensorEventQueue_hasEvents(ASensorEventQueue* queue);

/**
 * Returns the next available events from the queue.  Returns a negative
 * value if no events are available or an error has occurred, otherwise
 * the number of events returned.
 *
 * Examples:
 *   ASensorEvent event;
 *   ssize_t numEvent = ASensorEventQueue_getEvents(queue, &event, 1);
 *
 *   ASensorEvent eventBuffer[8];
 *   ssize_t numEvent = ASensorEventQueue_getEvents(queue, eventBuffer, 8);
 *
 */
ssize_t ASensorEventQueue_getEvents(ASensorEventQueue* queue,
                ASensorEvent* events, size_t count);


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

#ifdef __cplusplus
};
#endif

#endif // ANDROID_SENSOR_H

/** @} */
