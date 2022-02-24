/*
 * Copyright (C) 2015, The CyanogenMod Project
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

#ifndef CM_AUDIO_AMPLIFIER_INTERFACE_H
#define CM_AUDIO_AMPLIFIER_INTERFACE_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/audio.h>
#include <hardware/hardware.h>

#include <system/audio.h>

__BEGIN_DECLS

#define AMPLIFIER_HARDWARE_MODULE_ID "audio_amplifier"

#define AMPLIFIER_HARDWARE_INTERFACE "audio_amplifier_hw_if"

#define AMPLIFIER_MODULE_API_VERSION_0_1 HARDWARE_MODULE_API_VERSION(0, 1)

#define AMPLIFIER_DEVICE_API_VERSION_1_0 HARDWARE_DEVICE_API_VERSION(1, 0)
#define AMPLIFIER_DEVICE_API_VERSION_2_0 HARDWARE_DEVICE_API_VERSION(2, 0)
#define AMPLIFIER_DEVICE_API_VERSION_CURRENT AMPLIFIER_DEVICE_API_VERSION_2_0

struct str_parms;

typedef struct amplifier_device {
    /**
     * Common methods of the amplifier device. This *must* be the first member
     * of amplifier_device as users of this structure will cast a hw_device_t
     * to amplifier_device pointer in contexts where it's known
     * the hw_device_t references a amplifier_device.
     */
    struct hw_device_t common;

    /**
     * Notify amplifier device of current input devices
     *
     * This function should handle only input devices.
     */
    int (*set_input_devices)(struct amplifier_device *device, uint32_t devices);

    /**
     * Notify amplifier device of current output devices
     *
     * This function should handle only output devices.
     */
    int (*set_output_devices)(struct amplifier_device *device, uint32_t devices);

    /**
     * Notify amplifier device of output device enable/disable
     *
     * This function should handle only output devices.
     */
    int (*enable_output_devices)(struct amplifier_device *device,
            uint32_t devices, bool enable);

    /**
     * Notify amplifier device of input device enable/disable
     *
     * This function should handle only input devices.
     */
    int (*enable_input_devices)(struct amplifier_device *device,
            uint32_t devices, bool enable);

    /**
     * Notify amplifier device about current audio mode
     */
    int (*set_mode)(struct amplifier_device *device, audio_mode_t mode);

    /**
     * Notify amplifier device that an output stream has started
     */
    int (*output_stream_start)(struct amplifier_device *device,
            struct audio_stream_out *stream, bool offload);

    /**
     * Notify amplifier device that an input stream has started
     */
    int (*input_stream_start)(struct amplifier_device *device,
            struct audio_stream_in *stream);

    /**
     * Notify amplifier device that an output stream has stopped
     */
    int (*output_stream_standby)(struct amplifier_device *device,
            struct audio_stream_out *stream);

    /**
     * Notify amplifier device that an input stream has stopped
     */
    int (*input_stream_standby)(struct amplifier_device *device,
            struct audio_stream_in *stream);

    /**
     * set/get output audio device parameters.
     */
    int (*set_parameters)(struct amplifier_device *device,
        struct str_parms *parms);
} amplifier_device_t;

typedef struct amplifier_module {
    /**
     * Common methods of the amplifier module. This *must* be the first member
     * of amplifier_module as users of this structure will cast a hw_module_t
     * to amplifier_module pointer in contexts where it's known
     * the hw_module_t references a amplifier_module.
     */
    struct hw_module_t common;
} amplifier_module_t;

/** convenience API for opening and closing a supported device */

static inline int amplifier_device_open(const struct hw_module_t *module,
        struct amplifier_device **device)
{
    return module->methods->open(module, AMPLIFIER_HARDWARE_INTERFACE,
            (struct hw_device_t **) device);
}

static inline int amplifier_device_close(struct amplifier_device *device)
{
    return device->common.close(&device->common);
}

__END_DECLS

#endif  // CM_AUDIO_AMPLIFIER_INTERFACE_H
