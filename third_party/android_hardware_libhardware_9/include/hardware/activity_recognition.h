/*
 * Copyright (C) 2014 The Android Open Source Project
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

/*
 * Activity Recognition HAL. The goal is to provide low power, low latency, always-on activity
 * recognition implemented in hardware (i.e. these activity recognition algorithms/classifers
 * should NOT be run on the AP). By low power we mean that this may be activated 24/7 without
 * impacting the battery drain speed (goal in order of 1mW including the power for sensors).
 * This HAL does not specify the input sources that are used towards detecting these activities.
 * It has one monitor interface which can be used to batch activities for always-on
 * activity_recognition and if the latency is zero, the same interface can be used for low latency
 * detection.
 */

#ifndef ANDROID_ACTIVITY_RECOGNITION_INTERFACE_H
#define ANDROID_ACTIVITY_RECOGNITION_INTERFACE_H

#include <hardware/hardware.h>

__BEGIN_DECLS

#define ACTIVITY_RECOGNITION_HEADER_VERSION   1
#define ACTIVITY_RECOGNITION_API_VERSION_0_1  HARDWARE_DEVICE_API_VERSION_2(0, 1, ACTIVITY_RECOGNITION_HEADER_VERSION)

#define ACTIVITY_RECOGNITION_HARDWARE_MODULE_ID "activity_recognition"
#define ACTIVITY_RECOGNITION_HARDWARE_INTERFACE "activity_recognition_hw_if"

/*
 * Define types for various activities. Multiple activities may be active at the same time and
 * sometimes none of these activities may be active.
 *
 * Each activity has a corresponding type. Only activities that are defined here should use
 * android.activity_recognition.* prefix. OEM defined activities should not use this prefix.
 * Activity type of OEM-defined activities should start with the reverse domain name of the entity
 * defining the activity.
 *
 * When android introduces a new activity type that can potentially replace an OEM-defined activity
 * type, the OEM must use the official activity type on versions of the HAL that support this new
 * official activity type.
 *
 * Example (made up): Suppose Google's Glass team wants to detect nodding activity.
 *  - Such an activity is not officially supported in android L
 *  - Glass devices launching on L can implement a custom activity with
 *    type = "com.google.glass.nodding"
 *  - In M android release, if android decides to define ACITIVITY_TYPE_NODDING, those types
 *    should replace the Glass-team-specific types in all future launches.
 *  - When launching glass on the M release, Google should now use the official activity type
 *  - This way, other applications can use this activity.
 */

#define ACTIVITY_TYPE_IN_VEHICLE       "android.activity_recognition.in_vehicle"

#define ACTIVITY_TYPE_ON_BICYCLE       "android.activity_recognition.on_bicycle"

#define ACTIVITY_TYPE_WALKING          "android.activity_recognition.walking"

#define ACTIVITY_TYPE_RUNNING          "android.activity_recognition.running"

#define ACTIVITY_TYPE_STILL            "android.activity_recognition.still"

#define ACTIVITY_TYPE_TILTING          "android.activity_recognition.tilting"

/* Values for activity_event.event_types. */
enum {
    /*
     * A flush_complete event which indicates that a flush() has been successfully completed. This
     * does not correspond to any activity/event. An event of this type should be added to the end
     * of a batch FIFO and it indicates that all the events in the batch FIFO have been successfully
     * reported to the framework. An event of this type should be generated only if flush() has been
     * explicitly called and if the FIFO is empty at the time flush() is called it should trivially
     * return a flush_complete_event to indicate that the FIFO is empty.
     *
     * A flush complete event should have the following parameters set.
     * activity_event_t.event_type = ACTIVITY_EVENT_FLUSH_COMPLETE
     * activity_event_t.activity = 0
     * activity_event_t.timestamp = 0
     * activity_event_t.reserved = 0
     * See (*flush)() for more details.
     */
    ACTIVITY_EVENT_FLUSH_COMPLETE = 0,

    /* Signifies entering an activity. */
    ACTIVITY_EVENT_ENTER = 1,

    /* Signifies exiting an activity. */
    ACTIVITY_EVENT_EXIT  = 2
};

/*
 * Each event is a separate activity with event_type indicating whether this activity has started
 * or ended. Eg event: (event_type="enter", activity="ON_FOOT", timestamp)
 */
typedef struct activity_event {
    /* One of the ACTIVITY_EVENT_* constants defined above. */
    uint32_t event_type;

    /*
     * Index of the activity in the list returned by get_supported_activities_list. If this event
     * is a flush complete event, this should be set to zero.
     */
    uint32_t activity;

    /* Time at which the transition/event has occurred in nanoseconds using elapsedRealTimeNano. */
    int64_t timestamp;

    /* Set to zero. */
    int32_t reserved[4];
} activity_event_t;

typedef struct activity_recognition_module {
    /**
     * Common methods of the activity recognition module.  This *must* be the first member of
     * activity_recognition_module as users of this structure will cast a hw_module_t to
     * activity_recognition_module pointer in contexts where it's known the hw_module_t
     * references an activity_recognition_module.
     */
    hw_module_t common;

    /*
     * List of all activities supported by this module including OEM defined activities. Each
     * activity is represented using a string defined above. Each string should be null terminated.
     * The index of the activity in this array is used as a "handle" for enabling/disabling and
     * event delivery.
     * Return value is the size of this list.
     */
    int (*get_supported_activities_list)(struct activity_recognition_module* module,
            char const* const* *activity_list);
} activity_recognition_module_t;

struct activity_recognition_device;

typedef struct activity_recognition_callback_procs {
    // Callback for activity_data. This is guaranteed to not invoke any HAL methods.
    // Memory allocated for the events can be reused after this method returns.
    //    events - Array of activity_event_t s that are reported.
    //    count  - size of the array.
    void (*activity_callback)(const struct activity_recognition_callback_procs* procs,
            const activity_event_t* events, int count);
} activity_recognition_callback_procs_t;

typedef struct activity_recognition_device {
    /**
     * Common methods of the activity recognition device.  This *must* be the first member of
     * activity_recognition_device as users of this structure will cast a hw_device_t to
     * activity_recognition_device pointer in contexts where it's known the hw_device_t
     * references an activity_recognition_device.
     */
    hw_device_t common;

    /*
     * Sets the callback to invoke when there are events to report. This call overwrites the
     * previously registered callback (if any).
     */
    void (*register_activity_callback)(const struct activity_recognition_device* dev,
            const activity_recognition_callback_procs_t* callback);

    /*
     * Activates monitoring of activity transitions. Activities need not be reported as soon as they
     * are detected. The detected activities are stored in a FIFO and reported in batches when the
     * "max_batch_report_latency" expires or when the batch FIFO is full. The implementation should
     * allow the AP to go into suspend mode while the activities are detected and stored in the
     * batch FIFO. Whenever events need to be reported (like when the FIFO is full or when the
     * max_batch_report_latency has expired for an activity, event pair), it should wake_up the AP
     * so that no events are lost. Activities are stored as transitions and they are allowed to
     * overlap with each other. Each (activity, event_type) pair can be activated or deactivated
     * independently of the other. The HAL implementation needs to keep track of which pairs are
     * currently active and needs to detect only those pairs.
     *
     * At the first detection after this function gets called, the hardware should know whether the
     * user is in the activity.
     * - If event_type is ACTIVITY_EVENT_ENTER and the user is in the activity, then an
     *   (ACTIVITY_EVENT_ENTER, activity) event should be added to the FIFO.
     * - If event_type is ACTIVITY_EVENT_EXIT and the user is not in the activity, then an
     *   (ACTIVITY_EVENT_EXIT, activity) event should be added to the FIFO.
     * For example, suppose get_supported_activities_list contains on_bicyle and running, and the
     * user is biking. Consider the following four calls that could happen in any order.
     * - When enable_activity_event(on_bicycle, ACTIVITY_EVENT_ENTER) is called,
     *   (ACTIVITY_EVENT_ENTER, on_bicycle) should be added to the FIFO.
     * - When enable_activity_event(on_bicycle, ACTIVITY_EVENT_EXIT) is called, nothing should be
     *   added to the FIFO.
     * - When enable_activity_event(running, ACTIVITY_EVENT_ENTER) is called, nothing should be
     *   added to the FIFO.
     * - When enable_activity_event(running, ACTIVITY_EVENT_EXIT) is called,
     *   (ACTIVITY_EVENT_EXIT, running) should be added to the FIFO.
     *
     * activity_handle - Index of the specific activity that needs to be detected in the list
     *                   returned by get_supported_activities_list.
     * event_type - Specific transition of the activity that needs to be detected. It should be
     *              either ACTIVITY_EVENT_ENTER or ACTIVITY_EVENT_EXIT.
     * max_batch_report_latency_ns - a transition can be delayed by at most
     *                               “max_batch_report_latency” nanoseconds.
     * Return 0 on success, negative errno code otherwise.
     */
    int (*enable_activity_event)(const struct activity_recognition_device* dev,
            uint32_t activity_handle, uint32_t event_type, int64_t max_batch_report_latency_ns);

    /*
     * Disables detection of a specific (activity, event_type) pair. All the (activity, event_type)
     * events in the FIFO are discarded.
     */
    int (*disable_activity_event)(const struct activity_recognition_device* dev,
            uint32_t activity_handle, uint32_t event_type);

    /*
     * Flush all the batch FIFOs. Report all the activities that were stored in the FIFO so far as
     * if max_batch_report_latency had expired. This shouldn't change the latency in any way. Add
     * a flush_complete_event to indicate the end of the FIFO after all events are delivered.
     * activity_callback should be called before this function returns successfully.
     * See ACTIVITY_EVENT_FLUSH_COMPLETE for more details.
     * Return 0 on success, negative errno code otherwise.
     */
    int (*flush)(const struct activity_recognition_device* dev);

    // Must be set to NULL.
    void (*reserved_procs[16 - 4])(void);
} activity_recognition_device_t;

static inline int activity_recognition_open(const hw_module_t* module,
        activity_recognition_device_t** device) {
    return module->methods->open(module,
            ACTIVITY_RECOGNITION_HARDWARE_INTERFACE, (hw_device_t**)device);
}

static inline int activity_recognition_close(activity_recognition_device_t* device) {
    return device->common.close(&device->common);
}

__END_DECLS

#endif // ANDROID_ACTIVITY_RECOGNITION_INTERFACE_H
