/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef ANDROID_SENSORS_INTERFACE_H
#define ANDROID_SENSORS_INTERFACE_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>
#include <cutils/native_handle.h>

#include "sensors-base.h"

__BEGIN_DECLS

/*****************************************************************************/

#define SENSORS_HEADER_VERSION          1
#define SENSORS_MODULE_API_VERSION_0_1  HARDWARE_MODULE_API_VERSION(0, 1)
#define SENSORS_DEVICE_API_VERSION_0_1  HARDWARE_DEVICE_API_VERSION_2(0, 1, SENSORS_HEADER_VERSION)
#define SENSORS_DEVICE_API_VERSION_1_0  HARDWARE_DEVICE_API_VERSION_2(1, 0, SENSORS_HEADER_VERSION)
#define SENSORS_DEVICE_API_VERSION_1_1  HARDWARE_DEVICE_API_VERSION_2(1, 1, SENSORS_HEADER_VERSION)
#define SENSORS_DEVICE_API_VERSION_1_2  HARDWARE_DEVICE_API_VERSION_2(1, 2, SENSORS_HEADER_VERSION)
#define SENSORS_DEVICE_API_VERSION_1_3  HARDWARE_DEVICE_API_VERSION_2(1, 3, SENSORS_HEADER_VERSION)
#define SENSORS_DEVICE_API_VERSION_1_4  HARDWARE_DEVICE_API_VERSION_2(1, 4, SENSORS_HEADER_VERSION)

/**
 * Please see the Sensors section of source.android.com for an
 * introduction to and detailed descriptions of Android sensor types:
 * http://source.android.com/devices/sensors/index.html
 */

/**
 * The id of this module
 */
#define SENSORS_HARDWARE_MODULE_ID "sensors"

/**
 * Name of the sensors device to open
 */
#define SENSORS_HARDWARE_POLL       "poll"

/**
 * Sensor handle is greater than 0 and less than INT32_MAX.
 *
 * **** Deprecated ****
 * Defined values below are kept for code compatibility. Note sensor handle can be as large as
 * INT32_MAX.
 */
#define SENSORS_HANDLE_BASE             0
#define SENSORS_HANDLE_BITS             31
#define SENSORS_HANDLE_COUNT            (1ull<<SENSORS_HANDLE_BITS)


/*
 * **** Deprecated *****
 * flags for (*batch)()
 * Availability: SENSORS_DEVICE_API_VERSION_1_0
 * see (*batch)() documentation for details.
 * Deprecated as of  SENSORS_DEVICE_API_VERSION_1_3.
 * WAKE_UP_* sensors replace WAKE_UPON_FIFO_FULL concept.
 */
enum {
    SENSORS_BATCH_DRY_RUN               = 0x00000001,
    SENSORS_BATCH_WAKE_UPON_FIFO_FULL   = 0x00000002
};

/*
 * what field for meta_data_event_t
 */
enum {
    /* a previous flush operation has completed */
    // META_DATA_FLUSH_COMPLETE = 1,
    META_DATA_VERSION   /* always last, leave auto-assigned */
};

/*
 * The permission to use for body sensors (like heart rate monitors).
 * See sensor types for more details on what sensors should require this
 * permission.
 */
#define SENSOR_PERMISSION_BODY_SENSORS "android.permission.BODY_SENSORS"

/*
 * sensor flags legacy names
 *
 * please use SENSOR_FLAG_* directly for new implementation.
 * @see sensor_t
 */

#define SENSOR_FLAG_MASK(nbit, shift)   (((1<<(nbit))-1)<<(shift))
#define SENSOR_FLAG_MASK_1(shift)       SENSOR_FLAG_MASK(1, shift)

/*
 * Mask and shift for reporting mode sensor flags defined above.
 */
#define REPORTING_MODE_SHIFT            SENSOR_FLAG_SHIFT_REPORTING_MODE
#define REPORTING_MODE_NBIT             (3)
#define REPORTING_MODE_MASK             SENSOR_FLAG_MASK_REPORTING_MODE

/*
 * Mask and shift for data_injection mode sensor flags defined above.
 */
#define DATA_INJECTION_SHIFT            SENSOR_FLAG_SHIFT_DATA_INJECTION
#define DATA_INJECTION_MASK             SENSOR_FLAG_DATA_INJECTION

/*
 * Mask and shift for dynamic sensor flag.
 */
#define DYNAMIC_SENSOR_SHIFT            SENSOR_FLAG_SHIFT_DYNAMIC_SENSOR
#define DYNAMIC_SENSOR_MASK             SENSOR_FLAG_DYNAMIC_SENSOR

/*
 * Mask and shift for sensor additional information support.
 */
#define ADDITIONAL_INFO_SHIFT           SENSOR_FLAG_SHIFT_ADDITIONAL_INFO
#define ADDITIONAL_INFO_MASK            SENSOR_FLAG_ADDITIONAL_INFO

/*
 * Legacy alias of SENSOR_TYPE_MAGNETIC_FIELD.
 *
 * Previously, the type of a sensor measuring local magnetic field is named
 * SENSOR_TYPE_GEOMAGNETIC_FIELD and SENSOR_TYPE_MAGNETIC_FIELD is its alias.
 * SENSOR_TYPE_MAGNETIC_FIELD is redefined as primary name to avoid confusion.
 * SENSOR_TYPE_GEOMAGNETIC_FIELD is the alias and is deprecating. New implementation must not use
 * SENSOR_TYPE_GEOMAGNETIC_FIELD.
 */
#define SENSOR_TYPE_GEOMAGNETIC_FIELD   SENSOR_TYPE_MAGNETIC_FIELD

/*
 * Sensor string types for Android defined sensor types.
 *
 * For Android defined sensor types, string type will be override in sensor service and thus no
 * longer needed to be added to sensor_t data structure.
 *
 * These definitions are going to be removed soon.
 */
#define SENSOR_STRING_TYPE_ACCELEROMETER                "android.sensor.accelerometer"
#define SENSOR_STRING_TYPE_MAGNETIC_FIELD               "android.sensor.magnetic_field"
#define SENSOR_STRING_TYPE_ORIENTATION                  "android.sensor.orientation"
#define SENSOR_STRING_TYPE_GYROSCOPE                    "android.sensor.gyroscope"
#define SENSOR_STRING_TYPE_LIGHT                        "android.sensor.light"
#define SENSOR_STRING_TYPE_PRESSURE                     "android.sensor.pressure"
#define SENSOR_STRING_TYPE_TEMPERATURE                  "android.sensor.temperature"
#define SENSOR_STRING_TYPE_PROXIMITY                    "android.sensor.proximity"
#define SENSOR_STRING_TYPE_GRAVITY                      "android.sensor.gravity"
#define SENSOR_STRING_TYPE_LINEAR_ACCELERATION          "android.sensor.linear_acceleration"
#define SENSOR_STRING_TYPE_ROTATION_VECTOR              "android.sensor.rotation_vector"
#define SENSOR_STRING_TYPE_RELATIVE_HUMIDITY            "android.sensor.relative_humidity"
#define SENSOR_STRING_TYPE_AMBIENT_TEMPERATURE          "android.sensor.ambient_temperature"
#define SENSOR_STRING_TYPE_MAGNETIC_FIELD_UNCALIBRATED  "android.sensor.magnetic_field_uncalibrated"
#define SENSOR_STRING_TYPE_GAME_ROTATION_VECTOR         "android.sensor.game_rotation_vector"
#define SENSOR_STRING_TYPE_GYROSCOPE_UNCALIBRATED       "android.sensor.gyroscope_uncalibrated"
#define SENSOR_STRING_TYPE_SIGNIFICANT_MOTION           "android.sensor.significant_motion"
#define SENSOR_STRING_TYPE_STEP_DETECTOR                "android.sensor.step_detector"
#define SENSOR_STRING_TYPE_STEP_COUNTER                 "android.sensor.step_counter"
#define SENSOR_STRING_TYPE_GEOMAGNETIC_ROTATION_VECTOR  "android.sensor.geomagnetic_rotation_vector"
#define SENSOR_STRING_TYPE_HEART_RATE                   "android.sensor.heart_rate"
#define SENSOR_STRING_TYPE_TILT_DETECTOR                "android.sensor.tilt_detector"
#define SENSOR_STRING_TYPE_WAKE_GESTURE                 "android.sensor.wake_gesture"
#define SENSOR_STRING_TYPE_GLANCE_GESTURE               "android.sensor.glance_gesture"
#define SENSOR_STRING_TYPE_PICK_UP_GESTURE              "android.sensor.pick_up_gesture"
#define SENSOR_STRING_TYPE_WRIST_TILT_GESTURE           "android.sensor.wrist_tilt_gesture"
#define SENSOR_STRING_TYPE_DEVICE_ORIENTATION           "android.sensor.device_orientation"
#define SENSOR_STRING_TYPE_POSE_6DOF                    "android.sensor.pose_6dof"
#define SENSOR_STRING_TYPE_STATIONARY_DETECT            "android.sensor.stationary_detect"
#define SENSOR_STRING_TYPE_MOTION_DETECT                "android.sensor.motion_detect"
#define SENSOR_STRING_TYPE_HEART_BEAT                   "android.sensor.heart_beat"
#define SENSOR_STRING_TYPE_DYNAMIC_SENSOR_META          "android.sensor.dynamic_sensor_meta"
#define SENSOR_STRING_TYPE_ADDITIONAL_INFO              "android.sensor.additional_info"
#define SENSOR_STRING_TYPE_LOW_LATENCY_OFFBODY_DETECT   "android.sensor.low_latency_offbody_detect"
#define SENSOR_STRING_TYPE_ACCELEROMETER_UNCALIBRATED   "android.sensor.accelerometer_uncalibrated"

/**
 * Values returned by the accelerometer in various locations in the universe.
 * all values are in SI units (m/s^2)
 */
#define GRAVITY_SUN             (275.0f)
#define GRAVITY_EARTH           (9.80665f)

/** Maximum magnetic field on Earth's surface */
#define MAGNETIC_FIELD_EARTH_MAX    (60.0f)

/** Minimum magnetic field on Earth's surface */
#define MAGNETIC_FIELD_EARTH_MIN    (30.0f)

struct sensor_t;

/**
 * sensor event data
 */
typedef struct {
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
} sensors_vec_t;

/**
 * uncalibrated accelerometer, gyroscope and magnetometer event data
 */
typedef struct {
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
} uncalibrated_event_t;

/**
 * Meta data event data
 */
typedef struct meta_data_event {
    int32_t what;
    int32_t sensor;
} meta_data_event_t;

/**
 * Dynamic sensor meta event. See the description of SENSOR_TYPE_DYNAMIC_SENSOR_META type for
 * details.
 */
typedef struct dynamic_sensor_meta_event {
    int32_t  connected;
    int32_t  handle;
    const struct sensor_t * sensor; // should be NULL if connected == false
    uint8_t uuid[16];               // UUID of a dynamic sensor (using RFC 4122 byte order)
                                    // For UUID 12345678-90AB-CDEF-1122-334455667788 the uuid field
                                    // should be initialized as:
                                    // {0x12, 0x34, 0x56, 0x78, 0x90, 0xAB, 0xCD, 0xEF, 0x11, ...}
} dynamic_sensor_meta_event_t;

/**
 * Heart rate event data
 */
typedef struct {
  // Heart rate in beats per minute.
  // Set to 0 when status is SENSOR_STATUS_UNRELIABLE or ..._NO_CONTACT
  float bpm;
  // Status of the sensor for this reading. Set to one SENSOR_STATUS_...
  // Note that this value should only be set for sensors that explicitly define
  // the meaning of this field. This field is not piped through the framework
  // for other sensors.
  int8_t status;
} heart_rate_event_t;

typedef struct {
    int32_t type;                           // type of payload data, see additional_info_type_t
    int32_t serial;                         // sequence number of this frame for this type
    union {
        // for each frame, a single data type, either int32_t or float, should be used.
        int32_t data_int32[14];
        float   data_float[14];
    };
} additional_info_event_t;

/**
 * Union of the various types of sensor data
 * that can be returned.
 */
typedef struct sensors_event_t {
    /* must be sizeof(struct sensors_event_t) */
    int32_t version;

    /* sensor identifier */
    int32_t sensor;

    /* sensor type */
    int32_t type;

    /* reserved */
    int32_t reserved0;

    /* time is in nanosecond */
    int64_t timestamp;

    union {
        union {
            float           data[16];

            /* acceleration values are in meter per second per second (m/s^2) */
            sensors_vec_t   acceleration;

            /* magnetic vector values are in micro-Tesla (uT) */
            sensors_vec_t   magnetic;

            /* orientation values are in degrees */
            sensors_vec_t   orientation;

            /* gyroscope values are in rad/s */
            sensors_vec_t   gyro;

            /* temperature is in degrees centigrade (Celsius) */
            float           temperature;

            /* distance in centimeters */
            float           distance;

            /* light in SI lux units */
            float           light;

            /* pressure in hectopascal (hPa) */
            float           pressure;

            /* relative humidity in percent */
            float           relative_humidity;

            /* uncalibrated gyroscope values are in rad/s */
            uncalibrated_event_t uncalibrated_gyro;

            /* uncalibrated magnetometer values are in micro-Teslas */
            uncalibrated_event_t uncalibrated_magnetic;

            /* uncalibrated accelerometer values are in  meter per second per second (m/s^2) */
            uncalibrated_event_t uncalibrated_accelerometer;

            /* heart rate data containing value in bpm and status */
            heart_rate_event_t heart_rate;

            /* this is a special event. see SENSOR_TYPE_META_DATA above.
             * sensors_meta_data_event_t events are all reported with a type of
             * SENSOR_TYPE_META_DATA. The handle is ignored and must be zero.
             */
            meta_data_event_t meta_data;

            /* dynamic sensor meta event. See SENSOR_TYPE_DYNAMIC_SENSOR_META type for details */
            dynamic_sensor_meta_event_t dynamic_sensor_meta;

            /*
             * special additional sensor information frame, see
             * SENSOR_TYPE_ADDITIONAL_INFO for details.
             */
            additional_info_event_t additional_info;
        };

        union {
            uint64_t        data[8];

            /* step-counter */
            uint64_t        step_counter;
        } u64;
    };

    /* Reserved flags for internal use. Set to zero. */
    uint32_t flags;

    uint32_t reserved1[3];
} sensors_event_t;


/* see SENSOR_TYPE_META_DATA */
typedef sensors_event_t sensors_meta_data_event_t;


/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
struct sensors_module_t {
    struct hw_module_t common;

    /**
     * Enumerate all available sensors. The list is returned in "list".
     * return number of sensors in the list
     */
    int (*get_sensors_list)(struct sensors_module_t* module,
            struct sensor_t const** list);

    /**
     *  Place the module in a specific mode. The following modes are defined
     *
     *  0 - Normal operation. Default state of the module.
     *  1 - Loopback mode. Data is injected for the supported
     *      sensors by the sensor service in this mode.
     * return 0 on success
     *         -EINVAL if requested mode is not supported
     *         -EPERM if operation is not allowed
     */
    int (*set_operation_mode)(unsigned int mode);
};

struct sensor_t {

    /* Name of this sensor.
     * All sensors of the same "type" must have a different "name".
     */
    const char*     name;

    /* vendor of the hardware part */
    const char*     vendor;

    /* version of the hardware part + driver. The value of this field
     * must increase when the driver is updated in a way that changes the
     * output of this sensor. This is important for fused sensors when the
     * fusion algorithm is updated.
     */
    int             version;

    /* handle that identifies this sensors. This handle is used to reference
     * this sensor throughout the HAL API.
     */
    int             handle;

    /* this sensor's type. */
    int             type;

    /* maximum range of this sensor's value in SI units */
    float           maxRange;

    /* smallest difference between two values reported by this sensor */
    float           resolution;

    /* rough estimate of this sensor's power consumption in mA */
    float           power;

    /* this value depends on the reporting mode:
     *
     *   continuous: minimum sample period allowed in microseconds
     *   on-change : 0
     *   one-shot  :-1
     *   special   : 0, unless otherwise noted
     */
    int32_t         minDelay;

    /* number of events reserved for this sensor in the batch mode FIFO.
     * If there is a dedicated FIFO for this sensor, then this is the
     * size of this FIFO. If the FIFO is shared with other sensors,
     * this is the size reserved for that sensor and it can be zero.
     */
    uint32_t        fifoReservedEventCount;

    /* maximum number of events of this sensor that could be batched.
     * This is especially relevant when the FIFO is shared between
     * several sensors; this value is then set to the size of that FIFO.
     */
    uint32_t        fifoMaxEventCount;

    /* type of this sensor as a string.
     *
     * If type is OEM specific or sensor manufacturer specific type
     * (>=SENSOR_TYPE_DEVICE_PRIVATE_BASE), this string must be defined with reserved domain of
     * vendor/OEM as a prefix, e.g. com.google.glass.onheaddetector
     *
     * For sensors of Android defined types, Android framework will override this value. It is ok to
     * leave it pointing to an empty string.
     */
    const char*    stringType;

    /* permission required to see this sensor, register to it and receive data.
     * Set to "" if no permission is required. Some sensor types like the
     * heart rate monitor have a mandatory require_permission.
     * For sensors that always require a specific permission, like the heart
     * rate monitor, the android framework might overwrite this string
     * automatically.
     */
    const char*    requiredPermission;

    /* This value is defined only for continuous mode and on-change sensors. It is the delay between
     * two sensor events corresponding to the lowest frequency that this sensor supports. When lower
     * frequencies are requested through batch()/setDelay() the events will be generated at this
     * frequency instead. It can be used by the framework or applications to estimate when the batch
     * FIFO may be full.
     *
     * NOTE: 1) period_ns is in nanoseconds where as maxDelay/minDelay are in microseconds.
     *              continuous, on-change: maximum sampling period allowed in microseconds.
     *              one-shot, special : 0
     *   2) maxDelay should always fit within a 32 bit signed integer. It is declared as 64 bit
     *      on 64 bit architectures only for binary compatibility reasons.
     * Availability: SENSORS_DEVICE_API_VERSION_1_3
     */
    #ifdef __LP64__
       int64_t maxDelay;
    #else
       int32_t maxDelay;
    #endif

    /* Flags for sensor. See SENSOR_FLAG_* above. Only the least significant 32 bits are used here.
     * It is declared as 64 bit on 64 bit architectures only for binary compatibility reasons.
     * Availability: SENSORS_DEVICE_API_VERSION_1_3
     */
    #ifdef __LP64__
       uint64_t flags;
    #else
       uint32_t flags;
    #endif

    /* reserved fields, must be zero */
    void*           reserved[2];
};

/**
 * Shared memory information for a direct channel
 */
struct sensors_direct_mem_t {
    int type;                           // enum SENSOR_DIRECT_MEM_...
    int format;                         // enum SENSOR_DIRECT_FMT_...
    size_t size;                        // size of the memory region, in bytes
    const struct native_handle *handle; // shared memory handle, which is interpreted differently
                                        // depending on type
};

/**
 * Direct channel report configuration
 */
struct sensors_direct_cfg_t {
    int rate_level;             // enum SENSOR_DIRECT_RATE_...
};

/*
 * sensors_poll_device_t is used with SENSORS_DEVICE_API_VERSION_0_1
 * and is present for backward binary and source compatibility.
 * See the Sensors HAL interface section for complete descriptions of the
 * following functions:
 * http://source.android.com/devices/sensors/index.html#hal
 */
struct sensors_poll_device_t {
    struct hw_device_t common;
    int (*activate)(struct sensors_poll_device_t *dev,
            int sensor_handle, int enabled);
    int (*setDelay)(struct sensors_poll_device_t *dev,
            int sensor_handle, int64_t sampling_period_ns);
    int (*poll)(struct sensors_poll_device_t *dev,
            sensors_event_t* data, int count);
};

/*
 * struct sensors_poll_device_1 is used in HAL versions >= SENSORS_DEVICE_API_VERSION_1_0
 */
typedef struct sensors_poll_device_1 {
    union {
        /* sensors_poll_device_1 is compatible with sensors_poll_device_t,
         * and can be down-cast to it
         */
        struct sensors_poll_device_t v0;

        struct {
            struct hw_device_t common;

            /* Activate/de-activate one sensor.
             *
             * sensor_handle is the handle of the sensor to change.
             * enabled set to 1 to enable, or 0 to disable the sensor.
             *
             * Before sensor activation, existing sensor events that have not
             * been picked up by poll() should be abandoned so that application
             * upon new activation request will not get stale events.
             * (events that are generated during latter activation or during
             * data injection mode after sensor deactivation)
             *
             * Return 0 on success, negative errno code otherwise.
             */
            int (*activate)(struct sensors_poll_device_t *dev,
                    int sensor_handle, int enabled);

            /**
             * Set the events's period in nanoseconds for a given sensor.
             * If sampling_period_ns > max_delay it will be truncated to
             * max_delay and if sampling_period_ns < min_delay it will be
             * replaced by min_delay.
             */
            int (*setDelay)(struct sensors_poll_device_t *dev,
                    int sensor_handle, int64_t sampling_period_ns);

            /**
             * Write an array of sensor_event_t to data. The size of the
             * available buffer is specified by count. Returns number of
             * valid sensor_event_t.
             *
             * This function should block if there is no sensor event
             * available when being called. Thus, return value should always be
             * positive.
             */
            int (*poll)(struct sensors_poll_device_t *dev,
                    sensors_event_t* data, int count);
        };
    };


    /*
     * Sets a sensor’s parameters, including sampling frequency and maximum
     * report latency. This function can be called while the sensor is
     * activated, in which case it must not cause any sensor measurements to
     * be lost: transitioning from one sampling rate to the other cannot cause
     * lost events, nor can transitioning from a high maximum report latency to
     * a low maximum report latency.
     * See the Batching sensor results page for details:
     * http://source.android.com/devices/sensors/batching.html
     */
    int (*batch)(struct sensors_poll_device_1* dev,
            int sensor_handle, int flags, int64_t sampling_period_ns,
            int64_t max_report_latency_ns);

    /*
     * Flush adds a META_DATA_FLUSH_COMPLETE event (sensors_event_meta_data_t)
     * to the end of the "batch mode" FIFO for the specified sensor and flushes
     * the FIFO.
     * If the FIFO is empty or if the sensor doesn't support batching (FIFO size zero),
     * it should return SUCCESS along with a trivial META_DATA_FLUSH_COMPLETE event added to the
     * event stream. This applies to all sensors other than one-shot sensors.
     * If the sensor is a one-shot sensor, flush must return -EINVAL and not generate
     * any flush complete metadata.
     * If the sensor is not active at the time flush() is called, flush() should return
     * -EINVAL.
     */
    int (*flush)(struct sensors_poll_device_1* dev, int sensor_handle);

    /*
     * Inject a single sensor sample to be to this device.
     * data points to the sensor event to be injected
     * return 0 on success
     *         -EPERM if operation is not allowed
     *         -EINVAL if sensor event cannot be injected
     */
    int (*inject_sensor_data)(struct sensors_poll_device_1 *dev, const sensors_event_t *data);

    /*
     * Register/unregister direct report channel.
     *
     * A HAL declares support for direct report by setting non-NULL values for both
     * register_direct_channel and config_direct_report.
     *
     * This function has two operation modes:
     *
     * Register: mem != NULL, register a channel using supplied shared memory information. By the
     * time this function returns, sensors must finish initializing shared memory content
     * (format dependent, see SENSOR_DIRECT_FMT_*).
     *      Parameters:
     *          mem             points to a valid struct sensors_direct_mem_t.
     *          channel_handle  is ignored.
     *      Return value:
     *          A handle of channel (>0, <INT32_MAX) when success, which later can be referred in
     *          unregister or config_direct_report call, or error code (<0) when failed
     * Unregister: mem == NULL, unregister a previously registered channel.
     *      Parameters:
     *          mem             set to NULL
     *          channel_handle  contains handle of channel to be unregistered
     *      Return value:
     *          0, even if the channel_handle is invalid, in which case it will be a no-op.
     */
    int (*register_direct_channel)(struct sensors_poll_device_1 *dev,
            const struct sensors_direct_mem_t* mem, int channel_handle);

    /*
     * Configure direct sensor event report in direct channel.
     *
     * Start, modify rate or stop direct report of a sensor in a certain direct channel. A special
     * case is setting sensor handle -1 to stop means to stop all active sensor report on the
     * channel specified.
     *
     * A HAL declares support for direct report by setting non-NULL values for both
     * register_direct_channel and config_direct_report.
     *
     * Parameters:
     *      sensor_handle   sensor to be configured. The sensor has to support direct report
     *                      mode by setting flags of sensor_t. Also, direct report mode is only
     *                      defined for continuous reporting mode sensors.
     *      channel_handle  channel handle to be configured.
     *      config          direct report parameters, see sensor_direct_cfg_t.
     * Return value:
     *      - when sensor is started or sensor rate level is changed: return positive identifier of
     *        sensor in specified channel if successful, otherwise return negative error code.
     *      - when sensor is stopped: return 0 for success or negative error code for failure.
     */
    int (*config_direct_report)(struct sensors_poll_device_1 *dev,
            int sensor_handle, int channel_handle, const struct sensors_direct_cfg_t *config);

    /*
     * Reserved for future use, must be zero.
     */
    void (*reserved_procs[5])(void);

} sensors_poll_device_1_t;


/** convenience API for opening and closing a device */

static inline int sensors_open(const struct hw_module_t* module,
        struct sensors_poll_device_t** device) {
    return module->methods->open(module,
            SENSORS_HARDWARE_POLL, TO_HW_DEVICE_T_OPEN(device));
}

static inline int sensors_close(struct sensors_poll_device_t* device) {
    return device->common.close(&device->common);
}

static inline int sensors_open_1(const struct hw_module_t* module,
        sensors_poll_device_1_t** device) {
    return module->methods->open(module,
            SENSORS_HARDWARE_POLL, TO_HW_DEVICE_T_OPEN(device));
}

static inline int sensors_close_1(sensors_poll_device_1_t* device) {
    return device->common.close(&device->common);
}

__END_DECLS

#endif  // ANDROID_SENSORS_INTERFACE_H
