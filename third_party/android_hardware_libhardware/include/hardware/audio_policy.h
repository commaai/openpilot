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


#ifndef ANDROID_AUDIO_POLICY_INTERFACE_H
#define ANDROID_AUDIO_POLICY_INTERFACE_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>

#include <system/audio.h>
#include <system/audio_policy.h>

__BEGIN_DECLS

/**
 * The id of this module
 */
#define AUDIO_POLICY_HARDWARE_MODULE_ID "audio_policy"

/**
 * Name of the audio devices to open
 */
#define AUDIO_POLICY_INTERFACE "policy"

/* ---------------------------------------------------------------------------- */

/*
 * The audio_policy and audio_policy_service_ops structs define the
 * communication interfaces between the platform specific audio policy manager
 * and Android generic audio policy manager.
 * The platform specific audio policy manager must implement methods of the
 * audio_policy struct.
 * This implementation makes use of the audio_policy_service_ops to control
 * the activity and configuration of audio input and output streams.
 *
 * The platform specific audio policy manager is in charge of the audio
 * routing and volume control policies for a given platform.
 * The main roles of this module are:
 *   - keep track of current system state (removable device connections, phone
 *     state, user requests...).
 *   System state changes and user actions are notified to audio policy
 *   manager with methods of the audio_policy.
 *
 *   - process get_output() queries received when AudioTrack objects are
 *     created: Those queries return a handler on an output that has been
 *     selected, configured and opened by the audio policy manager and that
 *     must be used by the AudioTrack when registering to the AudioFlinger
 *     with the createTrack() method.
 *   When the AudioTrack object is released, a release_output() query
 *   is received and the audio policy manager can decide to close or
 *   reconfigure the output depending on other streams using this output and
 *   current system state.
 *
 *   - similarly process get_input() and release_input() queries received from
 *     AudioRecord objects and configure audio inputs.
 *   - process volume control requests: the stream volume is converted from
 *     an index value (received from UI) to a float value applicable to each
 *     output as a function of platform specific settings and current output
 *     route (destination device). It also make sure that streams are not
 *     muted if not allowed (e.g. camera shutter sound in some countries).
 */

/* XXX: this should be defined OUTSIDE of frameworks/base */
struct effect_descriptor_s;

struct audio_policy {
    /*
     * configuration functions
     */

    /* indicate a change in device connection status */
    int (*set_device_connection_state)(struct audio_policy *pol,
                                       audio_devices_t device,
                                       audio_policy_dev_state_t state,
                                       const char *device_address);

    /* retrieve a device connection status */
    audio_policy_dev_state_t (*get_device_connection_state)(
                                            const struct audio_policy *pol,
                                            audio_devices_t device,
                                            const char *device_address);

    /* indicate a change in phone state. Valid phones states are defined
     * by audio_mode_t */
    void (*set_phone_state)(struct audio_policy *pol, audio_mode_t state);

    /* deprecated, never called (was "indicate a change in ringer mode") */
    void (*set_ringer_mode)(struct audio_policy *pol, uint32_t mode,
                            uint32_t mask);

    /* force using a specific device category for the specified usage */
    void (*set_force_use)(struct audio_policy *pol,
                          audio_policy_force_use_t usage,
                          audio_policy_forced_cfg_t config);

    /* retrieve current device category forced for a given usage */
    audio_policy_forced_cfg_t (*get_force_use)(const struct audio_policy *pol,
                                               audio_policy_force_use_t usage);

    /* if can_mute is true, then audio streams that are marked ENFORCED_AUDIBLE
     * can still be muted. */
    void (*set_can_mute_enforced_audible)(struct audio_policy *pol,
                                          bool can_mute);

    /* check proper initialization */
    int (*init_check)(const struct audio_policy *pol);

    /*
     * Audio routing query functions
     */

    /* request an output appropriate for playback of the supplied stream type and
     * parameters */
    audio_io_handle_t (*get_output)(struct audio_policy *pol,
                                    audio_stream_type_t stream,
                                    uint32_t samplingRate,
                                    audio_format_t format,
                                    audio_channel_mask_t channelMask,
                                    audio_output_flags_t flags,
                                    const audio_offload_info_t *offloadInfo);

    /* indicates to the audio policy manager that the output starts being used
     * by corresponding stream. */
    int (*start_output)(struct audio_policy *pol,
                        audio_io_handle_t output,
                        audio_stream_type_t stream,
                        int session);

    /* indicates to the audio policy manager that the output stops being used
     * by corresponding stream. */
    int (*stop_output)(struct audio_policy *pol,
                       audio_io_handle_t output,
                       audio_stream_type_t stream,
                       int session);

    /* releases the output. */
    void (*release_output)(struct audio_policy *pol, audio_io_handle_t output);

    /* request an input appropriate for record from the supplied device with
     * supplied parameters. */
    audio_io_handle_t (*get_input)(struct audio_policy *pol, audio_source_t inputSource,
                                   uint32_t samplingRate,
                                   audio_format_t format,
                                   audio_channel_mask_t channelMask,
                                   audio_in_acoustics_t acoustics);

    /* indicates to the audio policy manager that the input starts being used */
    int (*start_input)(struct audio_policy *pol, audio_io_handle_t input);

    /* indicates to the audio policy manager that the input stops being used. */
    int (*stop_input)(struct audio_policy *pol, audio_io_handle_t input);

    /* releases the input. */
    void (*release_input)(struct audio_policy *pol, audio_io_handle_t input);

    /*
     * volume control functions
     */

    /* initialises stream volume conversion parameters by specifying volume
     * index range. The index range for each stream is defined by AudioService. */
    void (*init_stream_volume)(struct audio_policy *pol,
                               audio_stream_type_t stream,
                               int index_min,
                               int index_max);

    /* sets the new stream volume at a level corresponding to the supplied
     * index. The index is within the range specified by init_stream_volume() */
    int (*set_stream_volume_index)(struct audio_policy *pol,
                                   audio_stream_type_t stream,
                                   int index);

    /* retrieve current volume index for the specified stream */
    int (*get_stream_volume_index)(const struct audio_policy *pol,
                                   audio_stream_type_t stream,
                                   int *index);

    /* sets the new stream volume at a level corresponding to the supplied
     * index for the specified device.
     * The index is within the range specified by init_stream_volume() */
    int (*set_stream_volume_index_for_device)(struct audio_policy *pol,
                                   audio_stream_type_t stream,
                                   int index,
                                   audio_devices_t device);

    /* retrieve current volume index for the specified stream for the specified device */
    int (*get_stream_volume_index_for_device)(const struct audio_policy *pol,
                                   audio_stream_type_t stream,
                                   int *index,
                                   audio_devices_t device);

    /* return the strategy corresponding to a given stream type */
    uint32_t (*get_strategy_for_stream)(const struct audio_policy *pol,
                                        audio_stream_type_t stream);

    /* return the enabled output devices for the given stream type */
    audio_devices_t (*get_devices_for_stream)(const struct audio_policy *pol,
                                       audio_stream_type_t stream);

    /* Audio effect management */
    audio_io_handle_t (*get_output_for_effect)(struct audio_policy *pol,
                                            const struct effect_descriptor_s *desc);

    int (*register_effect)(struct audio_policy *pol,
                           const struct effect_descriptor_s *desc,
                           audio_io_handle_t output,
                           uint32_t strategy,
                           int session,
                           int id);

    int (*unregister_effect)(struct audio_policy *pol, int id);

    int (*set_effect_enabled)(struct audio_policy *pol, int id, bool enabled);

    bool (*is_stream_active)(const struct audio_policy *pol,
            audio_stream_type_t stream,
            uint32_t in_past_ms);

    bool (*is_stream_active_remotely)(const struct audio_policy *pol,
            audio_stream_type_t stream,
            uint32_t in_past_ms);

    bool (*is_source_active)(const struct audio_policy *pol,
            audio_source_t source);

    /* dump state */
    int (*dump)(const struct audio_policy *pol, int fd);

    /* check if offload is possible for given sample rate, bitrate, duration, ... */
    bool (*is_offload_supported)(const struct audio_policy *pol,
                                const audio_offload_info_t *info);
};


struct audio_policy_service_ops {
    /*
     * Audio output Control functions
     */

    /* Opens an audio output with the requested parameters.
     *
     * The parameter values can indicate to use the default values in case the
     * audio policy manager has no specific requirements for the output being
     * opened.
     *
     * When the function returns, the parameter values reflect the actual
     * values used by the audio hardware output stream.
     *
     * The audio policy manager can check if the proposed parameters are
     * suitable or not and act accordingly.
     */
    audio_io_handle_t (*open_output)(void *service,
                                     audio_devices_t *pDevices,
                                     uint32_t *pSamplingRate,
                                     audio_format_t *pFormat,
                                     audio_channel_mask_t *pChannelMask,
                                     uint32_t *pLatencyMs,
                                     audio_output_flags_t flags);

    /* creates a special output that is duplicated to the two outputs passed as
     * arguments. The duplication is performed by
     * a special mixer thread in the AudioFlinger.
     */
    audio_io_handle_t (*open_duplicate_output)(void *service,
                                               audio_io_handle_t output1,
                                               audio_io_handle_t output2);

    /* closes the output stream */
    int (*close_output)(void *service, audio_io_handle_t output);

    /* suspends the output.
     *
     * When an output is suspended, the corresponding audio hardware output
     * stream is placed in standby and the AudioTracks attached to the mixer
     * thread are still processed but the output mix is discarded.
     */
    int (*suspend_output)(void *service, audio_io_handle_t output);

    /* restores a suspended output. */
    int (*restore_output)(void *service, audio_io_handle_t output);

    /* */
    /* Audio input Control functions */
    /* */

    /* opens an audio input
     * deprecated - new implementations should use open_input_on_module,
     * and the acoustics parameter is ignored
     */
    audio_io_handle_t (*open_input)(void *service,
                                    audio_devices_t *pDevices,
                                    uint32_t *pSamplingRate,
                                    audio_format_t *pFormat,
                                    audio_channel_mask_t *pChannelMask,
                                    audio_in_acoustics_t acoustics);

    /* closes an audio input */
    int (*close_input)(void *service, audio_io_handle_t input);

    /* */
    /* misc control functions */
    /* */

    /* set a stream volume for a particular output.
     *
     * For the same user setting, a given stream type can have different
     * volumes for each output (destination device) it is attached to.
     */
    int (*set_stream_volume)(void *service,
                             audio_stream_type_t stream,
                             float volume,
                             audio_io_handle_t output,
                             int delay_ms);

    /* invalidate a stream type, causing a reroute to an unspecified new output */
    int (*invalidate_stream)(void *service,
                             audio_stream_type_t stream);

    /* function enabling to send proprietary informations directly from audio
     * policy manager to audio hardware interface. */
    void (*set_parameters)(void *service,
                           audio_io_handle_t io_handle,
                           const char *kv_pairs,
                           int delay_ms);

    /* function enabling to receive proprietary informations directly from
     * audio hardware interface to audio policy manager.
     *
     * Returns a pointer to a heap allocated string. The caller is responsible
     * for freeing the memory for it using free().
     */

    char * (*get_parameters)(void *service, audio_io_handle_t io_handle,
                             const char *keys);

    /* request the playback of a tone on the specified stream.
     * used for instance to replace notification sounds when playing over a
     * telephony device during a phone call.
     */
    int (*start_tone)(void *service,
                      audio_policy_tone_t tone,
                      audio_stream_type_t stream);

    int (*stop_tone)(void *service);

    /* set down link audio volume. */
    int (*set_voice_volume)(void *service,
                            float volume,
                            int delay_ms);

    /* move effect to the specified output */
    int (*move_effects)(void *service,
                        int session,
                        audio_io_handle_t src_output,
                        audio_io_handle_t dst_output);

    /* loads an audio hw module.
     *
     * The module name passed is the base name of the HW module library, e.g "primary" or "a2dp".
     * The function returns a handle on the module that will be used to specify a particular
     * module when calling open_output_on_module() or open_input_on_module()
     */
    audio_module_handle_t (*load_hw_module)(void *service,
                                              const char *name);

    /* Opens an audio output on a particular HW module.
     *
     * Same as open_output() but specifying a specific HW module on which the output must be opened.
     */
    audio_io_handle_t (*open_output_on_module)(void *service,
                                     audio_module_handle_t module,
                                     audio_devices_t *pDevices,
                                     uint32_t *pSamplingRate,
                                     audio_format_t *pFormat,
                                     audio_channel_mask_t *pChannelMask,
                                     uint32_t *pLatencyMs,
                                     audio_output_flags_t flags,
                                     const audio_offload_info_t *offloadInfo);

    /* Opens an audio input on a particular HW module.
     *
     * Same as open_input() but specifying a specific HW module on which the input must be opened.
     * Also removed deprecated acoustics parameter
     */
    audio_io_handle_t (*open_input_on_module)(void *service,
                                    audio_module_handle_t module,
                                    audio_devices_t *pDevices,
                                    uint32_t *pSamplingRate,
                                    audio_format_t *pFormat,
                                    audio_channel_mask_t *pChannelMask);

};

/**********************************************************************/

/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
typedef struct audio_policy_module {
    struct hw_module_t common;
} audio_policy_module_t;

struct audio_policy_device {
    /**
     * Common methods of the audio policy device.  This *must* be the first member of
     * audio_policy_device as users of this structure will cast a hw_device_t to
     * audio_policy_device pointer in contexts where it's known the hw_device_t references an
     * audio_policy_device.
     */
    struct hw_device_t common;

    int (*create_audio_policy)(const struct audio_policy_device *device,
                               struct audio_policy_service_ops *aps_ops,
                               void *service,
                               struct audio_policy **ap);

    int (*destroy_audio_policy)(const struct audio_policy_device *device,
                                struct audio_policy *ap);
};

/** convenience API for opening and closing a supported device */

static inline int audio_policy_dev_open(const hw_module_t* module,
                                    struct audio_policy_device** device)
{
    return module->methods->open(module, AUDIO_POLICY_INTERFACE,
                                 (hw_device_t**)device);
}

static inline int audio_policy_dev_close(struct audio_policy_device* device)
{
    return device->common.close(&device->common);
}


__END_DECLS

#endif  // ANDROID_AUDIO_POLICY_INTERFACE_H
