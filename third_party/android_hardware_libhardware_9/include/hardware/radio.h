/*
 * Copyright (C) 2015 The Android Open Source Project
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

#include <system/radio.h>
#include <hardware/hardware.h>

#ifndef ANDROID_RADIO_HAL_H
#define ANDROID_RADIO_HAL_H


__BEGIN_DECLS

/**
 * The id of this module
 */
#define RADIO_HARDWARE_MODULE_ID "radio"

/**
 * Name of the audio devices to open
 */
#define RADIO_HARDWARE_DEVICE "radio_hw_device"

#define RADIO_MODULE_API_VERSION_1_0 HARDWARE_MODULE_API_VERSION(1, 0)
#define RADIO_MODULE_API_VERSION_CURRENT RADIO_MODULE_API_VERSION_1_0


#define RADIO_DEVICE_API_VERSION_1_0 HARDWARE_DEVICE_API_VERSION(1, 0)
#define RADIO_DEVICE_API_VERSION_CURRENT RADIO_DEVICE_API_VERSION_1_0

/**
 * List of known radio HAL modules. This is the base name of the radio HAL
 * library composed of the "radio." prefix, one of the base names below and
 * a suffix specific to the device.
 * E.g: radio.fm.default.so
 */

#define RADIO_HARDWARE_MODULE_ID_FM "fm" /* corresponds to RADIO_CLASS_AM_FM */
#define RADIO_HARDWARE_MODULE_ID_SAT "sat" /* corresponds to RADIO_CLASS_SAT */
#define RADIO_HARDWARE_MODULE_ID_DT "dt" /* corresponds to RADIO_CLASS_DT */


/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
struct radio_module {
    struct hw_module_t common;
};

/*
 * Callback function called by the HAL when one of the following occurs:
 * - event RADIO_EVENT_HW_FAILURE: radio chip of driver failure requiring
 * closing and reopening of the tuner interface.
 * - event RADIO_EVENT_CONFIG: new configuration applied in response to open_tuner(),
 * or set_configuration(). The event status is 0 (no error) if the configuration has been applied,
 * -EINVAL is not or -ETIMEDOUT in case of time out.
 * - event RADIO_EVENT_TUNED: tune locked on new station/frequency following scan(),
 * step(), tune() or auto AF switching. The event status is 0 (no error) if in tune,
 * -EINVAL is not tuned and data in radio_program_info is not valid or -ETIMEDOUT if scan()
 * timed out.
 * - event RADIO_EVENT_TA: at the beginning and end of traffic announcement if current
 * configuration enables TA.
 * - event RADIO_EVENT_AF: after automatic switching to alternate frequency if current
 * configuration enables AF switching.
 * - event RADIO_EVENT_ANTENNA: when the antenna is connected or disconnected.
 * - event RADIO_EVENT_METADATA: when new meta data are received from the tuned station.
 * The callback MUST NOT be called synchronously while executing a HAL function but from
 * a separate thread.
 */
typedef void (*radio_callback_t)(radio_hal_event_t *event, void *cookie);

/* control interface for a radio tuner */
struct radio_tuner {
    /*
     * Apply current radio band configuration (band, range, channel spacing ...).
     *
     * arguments:
     * - config: the band configuration to apply
     *
     * returns:
     *  0 if configuration could be applied
     *  -EINVAL if configuration requested is invalid
     *
     * Automatically cancels pending scan, step or tune.
     *
     * Callback function with event RADIO_EVENT_CONFIG MUST be called once the
     * configuration is applied or a failure occurs or after a time out.
     */
    int (*set_configuration)(const struct radio_tuner *tuner,
                             const radio_hal_band_config_t *config);

    /*
     * Retrieve current radio band configuration.
     *
     * arguments:
     * - config: where to return the band configuration
     *
     * returns:
     *  0 if valid configuration is returned
     *  -EINVAL if invalid arguments are passed
     */
    int (*get_configuration)(const struct radio_tuner *tuner,
                             radio_hal_band_config_t *config);

    /*
     * Start scanning up to next valid station.
     * Must be called when a valid configuration has been applied.
     *
     * arguments:
     * - direction: RADIO_DIRECTION_UP or RADIO_DIRECTION_DOWN
     * - skip_sub_channel: valid for HD radio or digital radios only: ignore sub channels
     *  (e.g SPS for HD radio).
     *
     * returns:
     *  0 if scan successfully started
     *  -ENOSYS if called out of sequence
     *  -ENODEV if another error occurs
     *
     * Automatically cancels pending scan, step or tune.
     *
     *  Callback function with event RADIO_EVENT_TUNED MUST be called once
     *  locked on a station or after a time out or full frequency scan if
     *  no station found. The event status should indicate if a valid station
     *  is tuned or not.
     */
    int (*scan)(const struct radio_tuner *tuner,
                radio_direction_t direction, bool skip_sub_channel);

    /*
     * Move one channel spacing up or down.
     * Must be called when a valid configuration has been applied.
     *
     * arguments:
     * - direction: RADIO_DIRECTION_UP or RADIO_DIRECTION_DOWN
     * - skip_sub_channel: valid for HD radio or digital radios only: ignore sub channels
     *  (e.g SPS for HD radio).
     *
     * returns:
     *  0 if step successfully started
     *  -ENOSYS if called out of sequence
     *  -ENODEV if another error occurs
     *
     * Automatically cancels pending scan, step or tune.
     *
     * Callback function with event RADIO_EVENT_TUNED MUST be called once
     * step completed or after a time out. The event status should indicate
     * if a valid station is tuned or not.
     */
    int (*step)(const struct radio_tuner *tuner,
                radio_direction_t direction, bool skip_sub_channel);

    /*
     * Tune to specified frequency.
     * Must be called when a valid configuration has been applied.
     *
     * arguments:
     * - channel: channel to tune to. A frequency in kHz for AM/FM/HD Radio bands.
     * - sub_channel: valid for HD radio or digital radios only: (e.g SPS number for HD radio).
     *
     * returns:
     *  0 if tune successfully started
     *  -ENOSYS if called out of sequence
     *  -EINVAL if invalid arguments are passed
     *  -ENODEV if another error occurs
     *
     * Automatically cancels pending scan, step or tune.
     *
     * Callback function with event RADIO_EVENT_TUNED MUST be called once
     * tuned or after a time out. The event status should indicate
     * if a valid station is tuned or not.
     */
    int (*tune)(const struct radio_tuner *tuner,
                unsigned int channel, unsigned int sub_channel);

    /*
     * Cancel a scan, step or tune operation.
     * Must be called while a scan, step or tune operation is pending
     * (callback not yet sent).
     *
     * returns:
     *  0 if successful
     *  -ENOSYS if called out of sequence
     *  -ENODEV if another error occurs
     *
     * The callback is not sent.
     */
    int (*cancel)(const struct radio_tuner *tuner);

    /*
     * Retrieve current station information.
     *
     * arguments:
     * - info: where to return the program info.
     * If info->metadata is NULL. no meta data should be returned.
     * If meta data must be returned, they should be added to or cloned to
     * info->metadata, not passed from a newly created meta data buffer.
     *
     * returns:
     *  0 if tuned and information available
     *  -EINVAL if invalid arguments are passed
     *  -ENODEV if another error occurs
     */
    int (*get_program_information)(const struct radio_tuner *tuner,
                                   radio_program_info_t *info);
};

struct radio_hw_device {
    struct hw_device_t common;

    /*
     * Retrieve implementation properties.
     *
     * arguments:
     * - properties: where to return the module properties
     *
     * returns:
     *  0 if no error
     *  -EINVAL if invalid arguments are passed
     */
    int (*get_properties)(const struct radio_hw_device *dev,
                          radio_hal_properties_t *properties);

    /*
     * Open a tuner interface for the requested configuration.
     * If no other tuner is opened, this will activate the radio module.
     *
     * arguments:
     * - config: the band configuration to apply
     * - audio: this tuner will be used for live radio listening and should be connected to
     * the radio audio source.
     * - callback: the event callback
     * - cookie: the cookie to pass when calling the callback
     * - tuner: where to return the tuner interface
     *
     * returns:
     *  0 if HW was powered up and configuration could be applied
     *  -EINVAL if configuration requested is invalid
     *  -ENOSYS if called out of sequence
     *
     * Callback function with event RADIO_EVENT_CONFIG MUST be called once the
     * configuration is applied or a failure occurs or after a time out.
     */
    int (*open_tuner)(const struct radio_hw_device *dev,
                    const radio_hal_band_config_t *config,
                    bool audio,
                    radio_callback_t callback,
                    void *cookie,
                    const struct radio_tuner **tuner);

    /*
     * Close a tuner interface.
     * If the last tuner is closed, the radio module is deactivated.
     *
     * arguments:
     * - tuner: the tuner interface to close
     *
     * returns:
     *  0 if powered down successfully.
     *  -EINVAL if an invalid argument is passed
     *  -ENOSYS if called out of sequence
     */
    int (*close_tuner)(const struct radio_hw_device *dev, const struct radio_tuner *tuner);

};

typedef struct  radio_hw_device  radio_hw_device_t;

/** convenience API for opening and closing a supported device */

static inline int radio_hw_device_open(const struct hw_module_t* module,
                                       struct radio_hw_device** device)
{
    return module->methods->open(module, RADIO_HARDWARE_DEVICE,
                                 TO_HW_DEVICE_T_OPEN(device));
}

static inline int radio_hw_device_close(const struct radio_hw_device* device)
{
    return device->common.close((struct hw_device_t *)&device->common);
}

__END_DECLS

#endif  // ANDROID_RADIO_HAL_H
