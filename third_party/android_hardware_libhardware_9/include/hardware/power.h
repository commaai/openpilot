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

#ifndef ANDROID_INCLUDE_HARDWARE_POWER_H
#define ANDROID_INCLUDE_HARDWARE_POWER_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>

__BEGIN_DECLS

#define POWER_MODULE_API_VERSION_0_1  HARDWARE_MODULE_API_VERSION(0, 1)
#define POWER_MODULE_API_VERSION_0_2  HARDWARE_MODULE_API_VERSION(0, 2)
#define POWER_MODULE_API_VERSION_0_3  HARDWARE_MODULE_API_VERSION(0, 3)
#define POWER_MODULE_API_VERSION_0_4  HARDWARE_MODULE_API_VERSION(0, 4)
#define POWER_MODULE_API_VERSION_0_5  HARDWARE_MODULE_API_VERSION(0, 5)

/**
 * The id of this module
 */
#define POWER_HARDWARE_MODULE_ID "power"

/*
 * Platform-level sleep state stats.
 * Maximum length of Platform-level sleep state name.
 */
#define POWER_STATE_NAME_MAX_LENGTH 100

/*
 * Platform-level sleep state stats.
 * Maximum length of Platform-level sleep state voter name.
 */
#define POWER_STATE_VOTER_NAME_MAX_LENGTH 100

/*
 * Power hint identifiers passed to (*powerHint)
 */

typedef enum {
    POWER_HINT_VSYNC = 0x00000001,
    POWER_HINT_INTERACTION = 0x00000002,
    /* DO NOT USE POWER_HINT_VIDEO_ENCODE/_DECODE!  They will be removed in
     * KLP.
     */
    POWER_HINT_VIDEO_ENCODE = 0x00000003,
    POWER_HINT_VIDEO_DECODE = 0x00000004,
    POWER_HINT_LOW_POWER = 0x00000005,
    POWER_HINT_SUSTAINED_PERFORMANCE = 0x00000006,
    POWER_HINT_VR_MODE = 0x00000007,
    POWER_HINT_LAUNCH = 0x00000008,
    POWER_HINT_DISABLE_TOUCH = 0x00000009
} power_hint_t;

typedef enum {
    POWER_FEATURE_DOUBLE_TAP_TO_WAKE = 0x00000001
} feature_t;

/*
 * Platform-level sleep state stats:
 * power_state_voter_t struct is useful for describing the individual voters when a
 * Platform-level sleep state is chosen by aggregation of votes from multiple
 * clients/system conditions.
 *
 * This helps in attirbuting what in the device is blocking the device from
 * entering the lowest Platform-level sleep state.
 */
typedef struct {
    /*
     * Name of the voter.
     */
     char name[POWER_STATE_VOTER_NAME_MAX_LENGTH];

    /*
     * Total time in msec the voter voted for the platform sleep state since boot.
     */
     uint64_t total_time_in_msec_voted_for_since_boot;

    /*
     * Number of times the voter voted for the platform sleep state since boot.
     */
     uint64_t total_number_of_times_voted_since_boot;
} power_state_voter_t;

/*
 * Platform-level sleep state stats:
 * power_state_platform_sleep_state_t represents the Platform-level sleep state the
 * device is capable of getting into.
 *
 * SoCs usually have more than one Platform-level sleep state.
 *
 * The caller calls the get_number_of_platform_modes function to figure out the size
 * of power_state_platform_sleep_state_t array where each array element represents
 * a specific Platform-level sleep state.
 *
 * Higher the index deeper the state is i.e. lesser steady-state power is consumed
 * by the platform to be resident in that state.
 *
 * Caller allocates power_state_voter_t *voters for each Platform-level sleep state by
 * calling get_voter_list.
 */
typedef struct {
    /*
     * Platform-level Sleep state name.
     */
    char name[POWER_STATE_NAME_MAX_LENGTH];

    /*
     * Time spent in msec at this platform-level sleep state since boot.
     */
    uint64_t residency_in_msec_since_boot;

    /*
     * Total number of times system entered this state.
     */
    uint64_t total_transitions;

    /*
     * This platform-level sleep state can only be reached during system suspend.
     */
    bool supported_only_in_suspend;

    /*
     * The following fields are useful if the Platform-level sleep state
     * is chosen by aggregation votes from multiple clients/system conditions.
     * All the voters have to say yes or all the system conditions need to be
     * met to enter a platform-level sleep state.
     *
     * Setting number_of_voters to zero implies either the info is not available
     * or the system does not follow a voting mechanism to choose this
     * Platform-level sleep state.
     */
    uint32_t number_of_voters;

    /*
     * Voter list - Has to be allocated by the caller.
     *
     * Caller allocates power_state_voter_t *voters for each Platform-level sleep state
     * by calling get_voter_list.
     */
    power_state_voter_t *voters;
} power_state_platform_sleep_state_t;

/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
typedef struct power_module {
    struct hw_module_t common;

    /*
     * (*init)() performs power management setup actions at runtime
     * startup, such as to set default cpufreq parameters.  This is
     * called only by the Power HAL instance loaded by
     * PowerManagerService.
     *
     * Platform-level sleep state stats:
     * Can Also be used to initiate device specific Platform-level
     * Sleep state nodes from version 0.5 onwards.
     */
    void (*init)(struct power_module *module);

    /*
     * (*setInteractive)() performs power management actions upon the
     * system entering interactive state (that is, the system is awake
     * and ready for interaction, often with UI devices such as
     * display and touchscreen enabled) or non-interactive state (the
     * system appears asleep, display usually turned off).  The
     * non-interactive state is usually entered after a period of
     * inactivity, in order to conserve battery power during
     * such inactive periods.
     *
     * Typical actions are to turn on or off devices and adjust
     * cpufreq parameters.  This function may also call the
     * appropriate interfaces to allow the kernel to suspend the
     * system to low-power sleep state when entering non-interactive
     * state, and to disallow low-power suspend when the system is in
     * interactive state.  When low-power suspend state is allowed, the
     * kernel may suspend the system whenever no wakelocks are held.
     *
     * on is non-zero when the system is transitioning to an
     * interactive / awake state, and zero when transitioning to a
     * non-interactive / asleep state.
     *
     * This function is called to enter non-interactive state after
     * turning off the screen (if present), and called to enter
     * interactive state prior to turning on the screen.
     */
    void (*setInteractive)(struct power_module *module, int on);

    /*
     * (*powerHint) is called to pass hints on power requirements, which
     * may result in adjustment of power/performance parameters of the
     * cpufreq governor and other controls.  The possible hints are:
     *
     * POWER_HINT_VSYNC
     *
     *     Foreground app has started or stopped requesting a VSYNC pulse
     *     from SurfaceFlinger.  If the app has started requesting VSYNC
     *     then CPU and GPU load is expected soon, and it may be appropriate
     *     to raise speeds of CPU, memory bus, etc.  The data parameter is
     *     non-zero to indicate VSYNC pulse is now requested, or zero for
     *     VSYNC pulse no longer requested.
     *
     * POWER_HINT_INTERACTION
     *
     *     User is interacting with the device, for example, touchscreen
     *     events are incoming.  CPU and GPU load may be expected soon,
     *     and it may be appropriate to raise speeds of CPU, memory bus,
     *     etc.  The data parameter is the estimated length of the interaction
     *     in milliseconds, or 0 if unknown.
     *
     * POWER_HINT_LOW_POWER
     *
     *     Low power mode is activated or deactivated. Low power mode
     *     is intended to save battery at the cost of performance. The data
     *     parameter is non-zero when low power mode is activated, and zero
     *     when deactivated.
     *
     * POWER_HINT_SUSTAINED_PERFORMANCE
     *
     *     Sustained Performance mode is actived or deactivated. Sustained
     *     performance mode is intended to provide a consistent level of
     *     performance for a prolonged amount of time. The data parameter is
     *     non-zero when sustained performance mode is activated, and zero
     *     when deactivated.
     *
     * POWER_HINT_VR_MODE
     *
     *     VR Mode is activated or deactivated. VR mode is intended to
     *     provide minimum guarantee for performance for the amount of time the
     *     device can sustain it. The data parameter is non-zero when the mode
     *     is activated and zero when deactivated.
     *
     * POWER_HINT_DISABLE_TOUCH
     *
     *     When device enters some special modes, e.g. theater mode in Android
     *     Wear, there is no touch interaction expected between device and user.
     *     Touch controller could be disabled in those modes to save power.
     *     The data parameter is non-zero when touch could be disabled, and zero
     *     when touch needs to be re-enabled.
     *
     * A particular platform may choose to ignore any hint.
     *
     * availability: version 0.2
     *
     */
    void (*powerHint)(struct power_module *module, power_hint_t hint,
                      void *data);

    /*
     * (*setFeature) is called to turn on or off a particular feature
     * depending on the state parameter. The possible features are:
     *
     * FEATURE_DOUBLE_TAP_TO_WAKE
     *
     *    Enabling/Disabling this feature will allow/disallow the system
     *    to wake up by tapping the screen twice.
     *
     * availability: version 0.3
     *
     */
    void (*setFeature)(struct power_module *module, feature_t feature, int state);

    /*
     * Platform-level sleep state stats:
     * Report cumulative info on the statistics on platform-level sleep states since boot.
     *
     * Caller of the function queries the get_number_of_sleep_states and allocates the
     * memory for the power_state_platform_sleep_state_t *list before calling this function.
     *
     * power_stats module is responsible to assign values to all the fields as
     * necessary.
     *
     * Higher the index deeper the state is i.e. lesser steady-state power is consumed
     * by the platform to be resident in that state.
     *
     * The function returns 0 on success or negative value -errno on error.
     * EINVAL - *list is NULL.
     * EIO - filesystem nodes access error.
     *
     * availability: version 0.5
     */
    int (*get_platform_low_power_stats)(struct power_module *module,
        power_state_platform_sleep_state_t *list);

    /*
     * Platform-level sleep state stats:
     * This function is called to determine the number of platform-level sleep states
     * for get_platform_low_power_stats.
     *
     * The value returned by this function is used to allocate memory for
     * power_state_platform_sleep_state_t *list for get_platform_low_power_stats.
     *
     * The number of parameters must not change for successive calls.
     *
     * Return number of parameters on success or negative value -errno on error.
     * EIO - filesystem nodes access error.
     *
     * availability: version 0.5
     */
    ssize_t (*get_number_of_platform_modes)(struct power_module *module);

    /*
     * Platform-level sleep state stats:
     * Provides the number of voters for each of the Platform-level sleep state.
     *
     * Caller uses this function to allocate memory for the power_state_voter_t list.
     *
     * Caller has to allocate the space for the *voter array which is
     * get_number_of_platform_modes() long.
     *
     * Return 0 on success or negative value -errno on error.
     * EINVAL - *voter is NULL.
     * EIO - filesystem nodes access error.
     *
     * availability: version 0.5
     */
    int (*get_voter_list)(struct power_module *module, size_t *voter);

} power_module_t;


__END_DECLS

#endif  // ANDROID_INCLUDE_HARDWARE_POWER_H
