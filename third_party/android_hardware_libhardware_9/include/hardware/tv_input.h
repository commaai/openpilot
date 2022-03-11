/*
 * Copyright 2014 The Android Open Source Project
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

#ifndef ANDROID_TV_INPUT_INTERFACE_H
#define ANDROID_TV_INPUT_INTERFACE_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>
#include <system/audio.h>
#include <cutils/native_handle.h>

__BEGIN_DECLS

/*
 * Module versioning information for the TV input hardware module, based on
 * tv_input_module_t.common.module_api_version.
 *
 * Version History:
 *
 * TV_INPUT_MODULE_API_VERSION_0_1:
 * Initial TV input hardware module API.
 *
 */

#define TV_INPUT_MODULE_API_VERSION_0_1  HARDWARE_MODULE_API_VERSION(0, 1)

#define TV_INPUT_DEVICE_API_VERSION_0_1  HARDWARE_DEVICE_API_VERSION(0, 1)

/*
 * The id of this module
 */
#define TV_INPUT_HARDWARE_MODULE_ID "tv_input"

#define TV_INPUT_DEFAULT_DEVICE "default"

/*****************************************************************************/

/*
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
typedef struct tv_input_module {
    struct hw_module_t common;
} tv_input_module_t;

/*****************************************************************************/

enum {
    /* Generic hardware. */
    TV_INPUT_TYPE_OTHER_HARDWARE = 1,
    /* Tuner. (e.g. built-in terrestrial tuner) */
    TV_INPUT_TYPE_TUNER = 2,
    TV_INPUT_TYPE_COMPOSITE = 3,
    TV_INPUT_TYPE_SVIDEO = 4,
    TV_INPUT_TYPE_SCART = 5,
    TV_INPUT_TYPE_COMPONENT = 6,
    TV_INPUT_TYPE_VGA = 7,
    TV_INPUT_TYPE_DVI = 8,
    /* Physical HDMI port. (e.g. HDMI 1) */
    TV_INPUT_TYPE_HDMI = 9,
    TV_INPUT_TYPE_DISPLAY_PORT = 10,
};
typedef uint32_t tv_input_type_t;

typedef struct tv_input_device_info {
    /* Device ID */
    int device_id;

    /* Type of physical TV input. */
    tv_input_type_t type;

    union {
        struct {
            /* HDMI port ID number */
            uint32_t port_id;
        } hdmi;

        /* TODO: add other type specific information. */

        int32_t type_info_reserved[16];
    };

    /* TODO: Add capability if necessary. */

    /*
     * Audio info
     *
     * audio_type == AUDIO_DEVICE_NONE if this input has no audio.
     */
    audio_devices_t audio_type;
    const char* audio_address;

    int32_t reserved[16];
} tv_input_device_info_t;

/* See tv_input_event_t for more details. */
enum {
    /*
     * Hardware notifies the framework that a device is available.
     *
     * Note that DEVICE_AVAILABLE and DEVICE_UNAVAILABLE events do not represent
     * hotplug events (i.e. plugging cable into or out of the physical port).
     * These events notify the framework whether the port is available or not.
     * For a concrete example, when a user plugs in or pulls out the HDMI cable
     * from a HDMI port, it does not generate DEVICE_AVAILABLE and/or
     * DEVICE_UNAVAILABLE events. However, if a user inserts a pluggable USB
     * tuner into the Android device, it will generate a DEVICE_AVAILABLE event
     * and when the port is removed, it should generate a DEVICE_UNAVAILABLE
     * event.
     *
     * For hotplug events, please see STREAM_CONFIGURATION_CHANGED for more
     * details.
     *
     * HAL implementation should register devices by using this event when the
     * device boots up. The framework will recognize device reported via this
     * event only. In addition, the implementation could use this event to
     * notify the framework that a removable TV input device (such as USB tuner
     * as stated in the example above) is attached.
     */
    TV_INPUT_EVENT_DEVICE_AVAILABLE = 1,
    /*
     * Hardware notifies the framework that a device is unavailable.
     *
     * HAL implementation should generate this event when a device registered
     * by TV_INPUT_EVENT_DEVICE_AVAILABLE is no longer available. For example,
     * the event can indicate that a USB tuner is plugged out from the Android
     * device.
     *
     * Note that this event is not for indicating cable plugged out of the port;
     * for that purpose, the implementation should use
     * STREAM_CONFIGURATION_CHANGED event. This event represents the port itself
     * being no longer available.
     */
    TV_INPUT_EVENT_DEVICE_UNAVAILABLE = 2,
    /*
     * Stream configurations are changed. Client should regard all open streams
     * at the specific device are closed, and should call
     * get_stream_configurations() again, opening some of them if necessary.
     *
     * HAL implementation should generate this event when the available stream
     * configurations change for any reason. A typical use case of this event
     * would be to notify the framework that the input signal has changed
     * resolution, or that the cable is plugged out so that the number of
     * available streams is 0.
     *
     * The implementation may use this event to indicate hotplug status of the
     * port. the framework regards input devices with no available streams as
     * disconnected, so the implementation can generate this event with no
     * available streams to indicate that this device is disconnected, and vice
     * versa.
     */
    TV_INPUT_EVENT_STREAM_CONFIGURATIONS_CHANGED = 3,
    /*
     * Hardware is done with capture request with the buffer. Client can assume
     * ownership of the buffer again.
     *
     * HAL implementation should generate this event after request_capture() if
     * it succeeded. The event shall have the buffer with the captured image.
     */
    TV_INPUT_EVENT_CAPTURE_SUCCEEDED = 4,
    /*
     * Hardware met a failure while processing a capture request or client
     * canceled the request. Client can assume ownership of the buffer again.
     *
     * The event is similar to TV_INPUT_EVENT_CAPTURE_SUCCEEDED, but HAL
     * implementation generates this event upon a failure to process
     * request_capture(), or a request cancellation.
     */
    TV_INPUT_EVENT_CAPTURE_FAILED = 5,
};
typedef uint32_t tv_input_event_type_t;

typedef struct tv_input_capture_result {
    /* Device ID */
    int device_id;

    /* Stream ID */
    int stream_id;

    /* Sequence number of the request */
    uint32_t seq;

    /*
     * The buffer passed to hardware in request_capture(). The content of
     * buffer is undefined (although buffer itself is valid) for
     * TV_INPUT_CAPTURE_FAILED event.
     */
    buffer_handle_t buffer;

    /*
     * Error code for the request. -ECANCELED if request is cancelled; other
     * error codes are unknown errors.
     */
    int error_code;
} tv_input_capture_result_t;

typedef struct tv_input_event {
    tv_input_event_type_t type;

    union {
        /*
         * TV_INPUT_EVENT_DEVICE_AVAILABLE: all fields are relevant
         * TV_INPUT_EVENT_DEVICE_UNAVAILABLE: only device_id is relevant
         * TV_INPUT_EVENT_STREAM_CONFIGURATIONS_CHANGED: only device_id is
         *    relevant
         */
        tv_input_device_info_t device_info;
        /*
         * TV_INPUT_EVENT_CAPTURE_SUCCEEDED: error_code is not relevant
         * TV_INPUT_EVENT_CAPTURE_FAILED: all fields are relevant
         */
        tv_input_capture_result_t capture_result;
    };
} tv_input_event_t;

typedef struct tv_input_callback_ops {
    /*
     * event contains the type of the event and additional data if necessary.
     * The event object is guaranteed to be valid only for the duration of the
     * call.
     *
     * data is an object supplied at device initialization, opaque to the
     * hardware.
     */
    void (*notify)(struct tv_input_device* dev,
            tv_input_event_t* event, void* data);
} tv_input_callback_ops_t;

enum {
    TV_STREAM_TYPE_INDEPENDENT_VIDEO_SOURCE = 1,
    TV_STREAM_TYPE_BUFFER_PRODUCER = 2,
};
typedef uint32_t tv_stream_type_t;

typedef struct tv_stream_config {
    /*
     * ID number of the stream. This value is used to identify the whole stream
     * configuration.
     */
    int stream_id;

    /* Type of the stream */
    tv_stream_type_t type;

    /* Max width/height of the stream. */
    uint32_t max_video_width;
    uint32_t max_video_height;
} tv_stream_config_t;

typedef struct buffer_producer_stream {
    /*
     * IN/OUT: Width / height of the stream. Client may request for specific
     * size but hardware may change it. Client must allocate buffers with
     * specified width and height.
     */
    uint32_t width;
    uint32_t height;

    /* OUT: Client must set this usage when allocating buffer. */
    uint32_t usage;

    /* OUT: Client must allocate a buffer with this format. */
    uint32_t format;
} buffer_producer_stream_t;

typedef struct tv_stream {
    /* IN: ID in the stream configuration */
    int stream_id;

    /* OUT: Type of the stream (for convenience) */
    tv_stream_type_t type;

    /* Data associated with the stream for client's use */
    union {
        /* OUT: A native handle describing the sideband stream source */
        native_handle_t* sideband_stream_source_handle;

        /* IN/OUT: Details are in buffer_producer_stream_t */
        buffer_producer_stream_t buffer_producer;
    };
} tv_stream_t;

/*
 * Every device data structure must begin with hw_device_t
 * followed by module specific public methods and attributes.
 */
typedef struct tv_input_device {
    struct hw_device_t common;

    /*
     * initialize:
     *
     * Provide callbacks to the device and start operation. At first, no device
     * is available and after initialize() completes, currently available
     * devices including static devices should notify via callback.
     *
     * Framework owns callbacks object.
     *
     * data is a framework-owned object which would be sent back to the
     * framework for each callback notifications.
     *
     * Return 0 on success.
     */
    int (*initialize)(struct tv_input_device* dev,
            const tv_input_callback_ops_t* callback, void* data);

    /*
     * get_stream_configurations:
     *
     * Get stream configurations for a specific device. An input device may have
     * multiple configurations.
     *
     * The configs object is guaranteed to be valid only until the next call to
     * get_stream_configurations() or STREAM_CONFIGURATIONS_CHANGED event.
     *
     * Return 0 on success.
     */
    int (*get_stream_configurations)(const struct tv_input_device* dev,
            int device_id, int* num_configurations,
            const tv_stream_config_t** configs);

    /*
     * open_stream:
     *
     * Open a stream with given stream ID. Caller owns stream object, and the
     * populated data is only valid until the stream is closed.
     *
     * Return 0 on success; -EBUSY if the client should close other streams to
     * open the stream; -EEXIST if the stream with the given ID is already open;
     * -EINVAL if device_id and/or stream_id are invalid; other non-zero value
     * denotes unknown error.
     */
    int (*open_stream)(struct tv_input_device* dev, int device_id,
            tv_stream_t* stream);

    /*
     * close_stream:
     *
     * Close a stream to a device. data in tv_stream_t* object associated with
     * the stream_id is obsolete once this call finishes.
     *
     * Return 0 on success; -ENOENT if the stream is not open; -EINVAL if
     * device_id and/or stream_id are invalid.
     */
    int (*close_stream)(struct tv_input_device* dev, int device_id,
            int stream_id);

    /*
     * request_capture:
     *
     * Request buffer capture for a stream. This is only valid for buffer
     * producer streams. The buffer should be created with size, format and
     * usage specified in the stream. Framework provides seq in an
     * increasing sequence per each stream. Hardware should provide the picture
     * in a chronological order according to seq. For example, if two
     * requests are being processed at the same time, the request with the
     * smaller seq should get an earlier frame.
     *
     * The framework releases the ownership of the buffer upon calling this
     * function. When the buffer is filled, hardware notifies the framework
     * via TV_INPUT_EVENT_CAPTURE_FINISHED callback, and the ownership is
     * transferred back to framework at that time.
     *
     * Return 0 on success; -ENOENT if the stream is not open; -EINVAL if
     * device_id and/or stream_id are invalid; -EWOULDBLOCK if HAL cannot take
     * additional requests until it releases a buffer.
     */
    int (*request_capture)(struct tv_input_device* dev, int device_id,
            int stream_id, buffer_handle_t buffer, uint32_t seq);

    /*
     * cancel_capture:
     *
     * Cancel an ongoing capture. Hardware should release the buffer as soon as
     * possible via TV_INPUT_EVENT_CAPTURE_FAILED callback.
     *
     * Return 0 on success; -ENOENT if the stream is not open; -EINVAL if
     * device_id, stream_id, and/or seq are invalid.
     */
    int (*cancel_capture)(struct tv_input_device* dev, int device_id,
            int stream_id, uint32_t seq);

    void* reserved[16];
} tv_input_device_t;

__END_DECLS

#endif  // ANDROID_TV_INPUT_INTERFACE_H
