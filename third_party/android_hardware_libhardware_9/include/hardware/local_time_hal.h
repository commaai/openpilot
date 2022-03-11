/*
 * Copyright (C) 2011 The Android Open Source Project
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


#ifndef ANDROID_LOCAL_TIME_HAL_INTERFACE_H
#define ANDROID_LOCAL_TIME_HAL_INTERFACE_H

#include <stdint.h>

#include <hardware/hardware.h>

__BEGIN_DECLS

/**
 * The id of this module
 */
#define LOCAL_TIME_HARDWARE_MODULE_ID "local_time"

/**
 * Name of the local time devices to open
 */
#define LOCAL_TIME_HARDWARE_INTERFACE "local_time_hw_if"

/**********************************************************************/

/**
 * A structure used to collect low level sync data in a lab environment.  Most
 * HAL implementations will never need this structure.
 */
struct local_time_debug_event {
    int64_t local_timesync_event_id;
    int64_t local_time;
};

/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
struct local_time_module {
    struct hw_module_t common;
};

struct local_time_hw_device {
    /**
     * Common methods of the local time hardware device.  This *must* be the first member of
     * local_time_hw_device as users of this structure will cast a hw_device_t to
     * local_time_hw_device pointer in contexts where it's known the hw_device_t references a
     * local_time_hw_device.
     */
    struct hw_device_t common;

    /**
     *
     * Returns the current value of the system wide local time counter
     */
    int64_t (*get_local_time)(struct local_time_hw_device* dev);

    /**
     *
     * Returns the nominal frequency (in hertz) of the system wide local time
     * counter
     */
    uint64_t (*get_local_freq)(struct local_time_hw_device* dev);

    /**
     *
     * Sets the HW slew rate of oscillator which drives the system wide local
     * time counter.  On success, platforms should return 0.  Platforms which
     * do not support HW slew should leave this method set to NULL.
     *
     * Valid values for rate range from MIN_INT16 to MAX_INT16.  Platform
     * implementations should attempt map this range linearly to the min/max
     * slew rate of their hardware.
     */
    int (*set_local_slew)(struct local_time_hw_device* dev, int16_t rate);

    /**
     *
     * A method used to collect low level sync data in a lab environments.
     * Most HAL implementations will simply set this member to NULL, or return
     * -EINVAL to indicate that this functionality is not supported.
     * Production HALs should never support this method.
     */
    int (*get_debug_log)(struct local_time_hw_device* dev,
                         struct local_time_debug_event* records,
                         int max_records);
};

typedef struct local_time_hw_device local_time_hw_device_t;

/** convenience API for opening and closing a supported device */

static inline int local_time_hw_device_open(
        const struct hw_module_t* module,
        struct local_time_hw_device** device)
{
    return module->methods->open(module, LOCAL_TIME_HARDWARE_INTERFACE,
                                 TO_HW_DEVICE_T_OPEN(device));
}

static inline int local_time_hw_device_close(struct local_time_hw_device* device)
{
    return device->common.close(&device->common);
}


__END_DECLS

#endif  // ANDROID_LOCAL_TIME_INTERFACE_H
