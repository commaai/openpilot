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

#ifndef CONTEXT_HUB_H
#define CONTEXT_HUB_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>

/**
 * This header file defines the interface of a Context Hub Implementation to
 * the Android service exposing Context hub capabilities to applications.
 * The Context hub is expected to a low power compute domain with the following
 * defining charecteristics -
 *
 *    1) Access to sensors like accelerometer, gyroscope, magenetometer.
 *    2) Access to radios like GPS, Wifi, Bluetooth etc.
 *    3) Access to low power audio sensing.
 *
 * Implementations of this HAL can add additional sensors not defined by the
 * Android API. Such information sources shall be private to the implementation.
 *
 * The Context Hub HAL exposes the construct of code download. A piece of binary
 * code can be pushed to the context hub through the supported APIs.
 *
 * This version of the HAL designs in the possibility of multiple context hubs.
 */

__BEGIN_DECLS

/*****************************************************************************/

#define CONTEXT_HUB_HEADER_MAJOR_VERSION          1
#define CONTEXT_HUB_HEADER_MINOR_VERSION          1
#define CONTEXT_HUB_DEVICE_API_VERSION \
     HARDWARE_DEVICE_API_VERSION(CONTEXT_HUB_HEADER_MAJOR_VERSION, \
                                 CONTEXT_HUB_HEADER_MINOR_VERSION)

#define CONTEXT_HUB_DEVICE_API_VERSION_1_0  HARDWARE_DEVICE_API_VERSION(1, 0)
#define CONTEXT_HUB_DEVICE_API_VERSION_1_1  HARDWARE_DEVICE_API_VERSION(1, 1)

/**
 * The id of this module
 */
#define CONTEXT_HUB_MODULE_ID         "context_hub"

/**
 * Name of the device to open
 */
#define CONTEXT_HUB_HARDWARE_POLL     "ctxt_poll"

/**
 * Memory types for code upload. Device-specific. At least HUB_MEM_TYPE_MAIN must be supported
 */
#define HUB_MEM_TYPE_MAIN             0
#define HUB_MEM_TYPE_SECONDARY        1
#define HUB_MEM_TYPE_TCM              2


#define HUB_MEM_TYPE_FIRST_VENDOR     0x80000000ul

#define NANOAPP_VENDORS_ALL           0xFFFFFFFFFF000000ULL
#define NANOAPP_VENDOR_ALL_APPS       0x0000000000FFFFFFULL

#define NANOAPP_VENDOR(name) \
    (((uint64_t)(name)[0] << 56) | \
    ((uint64_t)(name)[1] << 48) | \
    ((uint64_t)(name)[2] << 40) | \
    ((uint64_t)(name)[3] << 32) | \
    ((uint64_t)(name)[4] << 24))

/*
 * generates the NANOAPP ID from vendor id and app seq# id
 */
#define NANO_APP_ID(vendor, seq_id) \
	(((uint64_t)(vendor) & NANOAPP_VENDORS_ALL) | ((uint64_t)(seq_id) & NANOAPP_VENDOR_ALL_APPS))

struct hub_app_name_t {
    uint64_t id;
};

/**
 * Other memory types (likely not writeable, informational only)
 */
#define HUB_MEM_TYPE_BOOTLOADER       0xfffffffful
#define HUB_MEM_TYPE_OS               0xfffffffeul
#define HUB_MEM_TYPE_EEDATA           0xfffffffdul
#define HUB_MEM_TYPE_RAM              0xfffffffcul

/**
 * Types of memory blocks on the context hub
 * */
#define MEM_FLAG_READ  0x1  // Memory can be written to
#define MEM_FLAG_WRITE 0x2  // Memory can be written to
#define MEM_FLAG_EXEC  0x4  // Memory can be executed from

/**
 * The following structure defines each memory block in detail
 */
struct mem_range_t {
    uint32_t total_bytes;
    uint32_t free_bytes;
    uint32_t type;        // HUB_MEM_TYPE_*
    uint32_t mem_flags;   // MEM_FLAG_*
};

#define NANOAPP_SIGNED_FLAG    0x1
#define NANOAPP_ENCRYPTED_FLAG 0x2
#define NANOAPP_MAGIC (((uint32_t)'N' <<  0) | ((uint32_t)'A' <<  8) | ((uint32_t)'N' << 16) | ((uint32_t)'O' << 24))

// The binary format below is in little endian format
struct nano_app_binary_t {
    uint32_t header_version;       // 0x1 for this version
    uint32_t magic;                // "NANO"
    struct hub_app_name_t app_id;  // App Id contains vendor id
    uint32_t app_version;          // Version of the app
    uint32_t flags;                // Signed, encrypted
    uint64_t hw_hub_type;          // which hub type is this compiled for

    // The version of the CHRE API that this nanoapp was compiled against.
    // If these values are both set to 0, then they must be interpreted the same
    // as if major version were set to 1, and minor 0 (the first valid CHRE API
    // version).
    uint8_t target_chre_api_major_version;
    uint8_t target_chre_api_minor_version;

    uint8_t reserved[6];           // Should be all zeroes
    uint8_t custom_binary[0];      // start of custom binary data
} __attribute__((packed));

struct hub_app_info {
    struct hub_app_name_t app_name;
    uint32_t version;
    uint32_t num_mem_ranges;
    struct mem_range_t mem_usage[2]; // Apps could only have RAM and SHARED_DATA
};

/**
 * Following enum defines the types of sensors that a hub may declare support
 * for. Declaration for support would mean that the hub can access and process
 * data from that particular sensor type.
 */

typedef enum {
    CONTEXT_SENSOR_RESERVED,             // 0
    CONTEXT_SENSOR_ACCELEROMETER,        // 1
    CONTEXT_SENSOR_GYROSCOPE,            // 2
    CONTEXT_SENSOR_MAGNETOMETER,         // 3
    CONTEXT_SENSOR_BAROMETER,            // 4
    CONTEXT_SENSOR_PROXIMITY_SENSOR,     // 5
    CONTEXT_SENSOR_AMBIENT_LIGHT_SENSOR, // 6

    CONTEXT_SENSOR_GPS = 0x100,          // 0x100
    // Reserving this space for variants on GPS
    CONTEXT_SENSOR_WIFI = 0x200,         // 0x200
    // Reserving this space for variants on WIFI
    CONTEXT_SENSOR_AUDIO = 0x300,        // 0x300
    // Reserving this space for variants on Audio
    CONTEXT_SENSOR_CAMERA = 0x400,       // 0x400
    // Reserving this space for variants on Camera
    CONTEXT_SENSOR_BLE = 0x500,          // 0x500

    CONTEXT_SENSOR_MAX = 0xffffffff,     //make sure enum size is set
} context_sensor_e;

/**
 * Sensor types beyond CONTEXT_HUB_TYPE_PRIVATE_SENSOR_BASE are custom types
 */
#define CONTEXT_HUB_TYPE_PRIVATE_SENSOR_BASE 0x10000

/**
 * The following structure describes a sensor
 */
struct physical_sensor_description_t {
    uint32_t sensor_type;           // From the definitions above eg: 100
    const char *type_string;        // Type as a string. eg: "GPS"
    const char *name;               // Identifier eg: "Bosch BMI160"
    const char *vendor;             // Vendor : eg "STM"
    uint32_t version;               // Version : eg 0x1001
    uint32_t fifo_reserved_count;   // Batching possible in hardware. Please
                                    // note that here hardware does not include
                                    // the context hub itself. Thus, this
                                    // definition may be different from say the
                                    // number advertised in the sensors HAL
                                    // which allows for batching in a hub.
    uint32_t fifo_max_count;        // maximum number of batchable events.
    uint64_t min_delay_ms;          // in milliseconds, corresponding to highest
                                    // sampling freq.
    uint64_t max_delay_ms;          // in milliseconds, corresponds to minimum
                                    // sampling frequency
    float peak_power_mw;            // At max frequency & no batching, power
                                    // in milliwatts
};

struct connected_sensor_t {
    uint32_t sensor_id;             // identifier for this sensor

    /* This union may be extended to other sensor types */
    union {
        struct physical_sensor_description_t physical_sensor;
    };
};

struct hub_message_t {
    struct hub_app_name_t app_name; /* To/From this nanoapp */
    uint32_t message_type;
    uint32_t message_len;
    const void *message;
};

/**
 * Definition of a context hub. A device may contain more than one low
 * power domain. In that case, please add an entry for each hub. However,
 * it is perfectly OK for a device to declare one context hub and manage
 * them internally as several
 */

struct context_hub_t {
    const char *name;                // descriptive name eg: "Awesome Hub #1"
    const char *vendor;              // hub hardware vendor eg: "Qualcomm"
    const char *toolchain;           // toolchain to make binaries eg:"gcc ARM"
    uint32_t platform_version;       // Version of the hardware : eg 0x20
    uint32_t toolchain_version;      // Version of the toolchain : eg: 0x484
    uint32_t hub_id;                 // a device unique id for this hub

    float peak_mips;                 // Peak MIPS platform can deliver
    float stopped_power_draw_mw;     // if stopped, retention power, milliwatts
    float sleep_power_draw_mw;       // if sleeping, retention power, milliwatts
    float peak_power_draw_mw;        // for a busy CPUm power in milliwatts

    const struct connected_sensor_t *connected_sensors; // array of connected sensors
    uint32_t num_connected_sensors;  // number of connected sensors

    const struct hub_app_name_t os_app_name; /* send msgs here for OS functions */
    uint32_t max_supported_msg_len;  // This is the maximum size of the message that can
                                     // be sent to the hub in one chunk (in bytes)
};

/**
 * Definitions of message payloads, see hub_messages_e
 */

struct status_response_t {
    int32_t result; // 0 on success, < 0 : error on failure. > 0 for any descriptive status
};

struct apps_enable_request_t {
    struct hub_app_name_t app_name;
};

struct apps_disable_request_t {
    struct hub_app_name_t app_name;
};

struct load_app_request_t {
    struct nano_app_binary_t app_binary;
};

struct unload_app_request_t {
    struct hub_app_name_t app_name;
};

struct query_apps_request_t {
    struct hub_app_name_t app_name;
};

/**
 * CONTEXT_HUB_APPS_ENABLE
 * Enables the specified nano-app(s)
 *
 * Payload : apps_enable_request_t
 *
 * Response : status_response_t
 *            On receipt of a successful response, it is
 *               expected that
 *
 *               i) the app is executing and able to receive
 *                  any messages.
 *
 *              ii) the system should be able to respond to an
 *                  CONTEXT_HUB_QUERY_APPS request.
 *
 */

/**
 * CONTEXT_HUB_APPS_DISABLE
 * Stops the specified nano-app(s)
 *
 * Payload : apps_disable_request_t
 *
 * Response : status_response_t
 *            On receipt of a successful response,
 *               i) No further events are delivered to the
 *                  nanoapp.
 *
 *              ii) The app should not show up in a
 *                  CONTEXT_HUB_QUERY_APPS request.
 */

/**
 * CONTEXT_HUB_LOAD_APP
 * Loads a nanoApp. Upon loading the nanoApp's init method is
 * called.
 *
 *
 * Payload : load_app_request_t
 *
 * Response : status_response_t On receipt of a successful
 *               response, it is expected that
 *               i) the app is executing and able to receive
 *                  messages.
 *
 *              ii) the system should be able to respond to a
 *                  CONTEXT_HUB_QUERY_APPS.
 */

/**
 * CONTEXT_HUB_UNLOAD_APP
 * Unloads a nanoApp. Before the unload, the app's deinit method
 * is called.
 *
 * Payload : unload_app_request_t.
 *
 * Response : status_response_t On receipt of a
 *            successful response, it is expected that
 *               i) No further events are delivered to the
 *                  nanoapp.
 *
 *              ii) the system does not list the app in a
 *                  response to a CONTEXT_HUB_QUERY_APPS.
 *
 *             iii) Any resources used by the app should be
 *                  freed up and available to the system.
 */

/**
 * CONTEXT_HUB_QUERY_APPS Queries for status of apps
 *
 * Payload : query_apps_request_t
 *
 * Response : struct hub_app_info[]
 */

/**
 * CONTEXT_HUB_QUERY_MEMORY Queries for memory regions on the
 * hub
 *
 * Payload : NULL
 *
 * Response : struct mem_range_t[]
 */

/**
 * CONTEXT_HUB_OS_REBOOT
 * Reboots context hub OS, restarts all the nanoApps.
 * No reboot notification is sent to nanoApps; reboot happens immediately and
 * unconditionally; all volatile FW state and any data is lost as a result
 *
 * Payload : none
 *
 * Response : status_response_t
 *            On receipt of a successful response, it is
 *               expected that
 *
 *               i) system reboot has completed;
 *                  status contains reboot reason code (platform-specific)
 *
 * Unsolicited response:
 *            System may send unsolicited response at any time;
 *            this should be interpreted as FW reboot, and necessary setup
 *            has to be done (same or similar to the setup done on system boot)
 */

/**
 * All communication between the context hubs and the Context Hub Service is in
 * the form of messages. Some message types are distinguished and their
 * Semantics shall be well defined.
 * Custom message types should be defined starting above
 * CONTEXT_HUB_PRIVATE_MSG_BASE
 */

typedef enum {
    CONTEXT_HUB_APPS_ENABLE  = 1, // Enables loaded nano-app(s)
    CONTEXT_HUB_APPS_DISABLE = 2, // Disables loaded nano-app(s)
    CONTEXT_HUB_LOAD_APP     = 3, // Load a supplied app
    CONTEXT_HUB_UNLOAD_APP   = 4, // Unload a specified app
    CONTEXT_HUB_QUERY_APPS   = 5, // Query for app(s) info on hub
    CONTEXT_HUB_QUERY_MEMORY = 6, // Query for memory info
    CONTEXT_HUB_OS_REBOOT    = 7, // Request to reboot context HUB OS
} hub_messages_e;

#define CONTEXT_HUB_TYPE_PRIVATE_MSG_BASE 0x00400

/**
 * A callback registers with the context hub service to pass messages
 * coming from the hub to the service/clients.
 */
typedef int context_hub_callback(uint32_t hub_id, const struct hub_message_t *rxed_msg, void *cookie);


/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
struct context_hub_module_t {
    struct hw_module_t common;

    /**
     * Enumerate all available hubs.The list is returned in "list".
     * @return result : number of hubs in list or error  (negative)
     *
     * This method shall be called at device bootup.
     */
    int (*get_hubs)(struct context_hub_module_t* module, const struct context_hub_t ** list);

    /**
     * Registers a callback for the HAL implementation to communicate
     * with the context hub service.
     * @return result : 0 if successful, error code otherwise
     */
    int (*subscribe_messages)(uint32_t hub_id, context_hub_callback cbk, void *cookie);

    /**
     * Send a message to a hub
     * @return result : 0 if successful, error code otherwise
     */
    int (*send_message)(uint32_t hub_id, const struct hub_message_t *msg);

};

__END_DECLS

#endif  // CONTEXT_HUB_SENSORS_INTERFACE_H
