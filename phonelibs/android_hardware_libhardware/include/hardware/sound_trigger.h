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

#include <system/audio.h>
#include <system/sound_trigger.h>
#include <hardware/hardware.h>

#ifndef ANDROID_SOUND_TRIGGER_HAL_H
#define ANDROID_SOUND_TRIGGER_HAL_H


__BEGIN_DECLS

/**
 * The id of this module
 */
#define SOUND_TRIGGER_HARDWARE_MODULE_ID "sound_trigger"

/**
 * Name of the audio devices to open
 */
#define SOUND_TRIGGER_HARDWARE_INTERFACE "sound_trigger_hw_if"

#define SOUND_TRIGGER_MODULE_API_VERSION_1_0 HARDWARE_MODULE_API_VERSION(1, 0)
#define SOUND_TRIGGER_MODULE_API_VERSION_CURRENT SOUND_TRIGGER_MODULE_API_VERSION_1_0


#define SOUND_TRIGGER_DEVICE_API_VERSION_1_0 HARDWARE_DEVICE_API_VERSION(1, 0)
#define SOUND_TRIGGER_DEVICE_API_VERSION_CURRENT SOUND_TRIGGER_DEVICE_API_VERSION_1_0

/**
 * List of known sound trigger HAL modules. This is the base name of the sound_trigger HAL
 * library composed of the "sound_trigger." prefix, one of the base names below and
 * a suffix specific to the device.
 * e.g: sondtrigger.primary.goldfish.so or sound_trigger.primary.default.so
 */

#define SOUND_TRIGGER_HARDWARE_MODULE_ID_PRIMARY "primary"


/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
struct sound_trigger_module {
    struct hw_module_t common;
};

typedef void (*recognition_callback_t)(struct sound_trigger_recognition_event *event, void *cookie);
typedef void (*sound_model_callback_t)(struct sound_trigger_model_event *event, void *cookie);

struct sound_trigger_hw_device {
    struct hw_device_t common;

    /*
     * Retrieve implementation properties.
     */
    int (*get_properties)(const struct sound_trigger_hw_device *dev,
                          struct sound_trigger_properties *properties);

    /*
     * Load a sound model. Once loaded, recognition of this model can be started and stopped.
     * Only one active recognition per model at a time. The SoundTrigger service will handle
     * concurrent recognition requests by different users/applications on the same model.
     * The implementation returns a unique handle used by other functions (unload_sound_model(),
     * start_recognition(), etc...
     */
    int (*load_sound_model)(const struct sound_trigger_hw_device *dev,
                            struct sound_trigger_sound_model *sound_model,
                            sound_model_callback_t callback,
                            void *cookie,
                            sound_model_handle_t *handle);

    /*
     * Unload a sound model. A sound model can be unloaded to make room for a new one to overcome
     * implementation limitations.
     */
    int (*unload_sound_model)(const struct sound_trigger_hw_device *dev,
                              sound_model_handle_t handle);

    /* Start recognition on a given model. Only one recognition active at a time per model.
     * Once recognition succeeds of fails, the callback is called.
     * TODO: group recognition configuration parameters into one struct and add key phrase options.
     */
    int (*start_recognition)(const struct sound_trigger_hw_device *dev,
                             sound_model_handle_t sound_model_handle,
                             const struct sound_trigger_recognition_config *config,
                             recognition_callback_t callback,
                             void *cookie);

    /* Stop recognition on a given model.
     * The implementation does not have to call the callback when stopped via this method.
     */
    int (*stop_recognition)(const struct sound_trigger_hw_device *dev,
                           sound_model_handle_t sound_model_handle);
};

typedef struct sound_trigger_hw_device sound_trigger_hw_device_t;

/** convenience API for opening and closing a supported device */

static inline int sound_trigger_hw_device_open(const struct hw_module_t* module,
                                       struct sound_trigger_hw_device** device)
{
    return module->methods->open(module, SOUND_TRIGGER_HARDWARE_INTERFACE,
                                 (struct hw_device_t**)device);
}

static inline int sound_trigger_hw_device_close(struct sound_trigger_hw_device* device)
{
    return device->common.close(&device->common);
}

__END_DECLS

#endif  // ANDROID_SOUND_TRIGGER_HAL_H
