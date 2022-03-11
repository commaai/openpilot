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

#ifndef ANDROID_INCLUDE_HARDWARE_THERMAL_H
#define ANDROID_INCLUDE_HARDWARE_THERMAL_H

#include <stdbool.h>
#include <stdint.h>
#include <float.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>

__BEGIN_DECLS

#define THERMAL_HARDWARE_MODULE_API_VERSION_0_1 HARDWARE_MODULE_API_VERSION(0, 1)

#define THERMAL_HARDWARE_MODULE_ID "thermal"

// This value is returned if a desired temperature is not available.
#define UNKNOWN_TEMPERATURE -FLT_MAX

/** Device temperature types. Must be kept in sync with
 * framework/base/core/java/android/os/HardwarePropertiesManager.java
 */
enum temperature_type {
    DEVICE_TEMPERATURE_UNKNOWN  = -1,
    DEVICE_TEMPERATURE_CPU      = 0,
    DEVICE_TEMPERATURE_GPU      = 1,
    DEVICE_TEMPERATURE_BATTERY  = 2,
    DEVICE_TEMPERATURE_SKIN     = 3
};

enum cooling_type {
    /** Fan cooling device speed in RPM. */
    FAN_RPM                     = 0,
};

typedef struct {
  /**
   * This temperature's type.
   */
  enum temperature_type type;

  /**
   * Name of this temperature.
   * All temperatures of the same "type" must have a different "name".
   */
  const char *name;

  /**
   * Current temperature in Celsius. If not available set by HAL to
   * UNKNOWN_TEMPERATURE.
   * Current temperature can be in any units if
   * type=DEVICE_TEMPERATURE_UNKNOWN.
   */
  float current_value;

  /**
   * Throttling temperature constant for this temperature.
   * If not available, set by HAL to UNKNOWN_TEMPERATURE.
   */
  float throttling_threshold;

  /**
   * Shutdown temperature constant for this temperature.
   * If not available, set by HAL to UNKNOWN_TEMPERATURE.
   */
  float shutdown_threshold;

  /**
   * Threshold temperature above which the VR mode clockrate minimums cannot
   * be maintained for this device.
   * If not available, set by HAL to UNKNOWN_TEMPERATURE.
   */
  float vr_throttling_threshold;
} temperature_t;

typedef struct {
    /**
     * This cooling device type.
     */
    enum cooling_type type;

    /**
     * Name of this cooling device.
     * All cooling devices of the same "type" must have a different "name".
     */
    const char *name;

    /**
     * Current cooling device value. Units depend on cooling device "type".
     */
    float current_value;
} cooling_device_t;

typedef struct {
    /**
     * Name of this CPU.
     * All CPUs must have a different "name".
     */
    const char *name;

    /**
     * Active time since the last boot in ms.
     */
    uint64_t active;

    /**
     * Total time since the last boot in ms.
     */
    uint64_t total;

    /**
     * Is set to true when a core is online.
     * If the core is offline, all other members except |name| should be ignored.
     */
    bool is_online;
} cpu_usage_t;

typedef struct thermal_module {
    struct hw_module_t common;

    /*
     * (*getTemperatures) is called to get temperatures in Celsius.
     *
     * @param list If NULL, this method only returns number of temperatures
     *     and caller should allocate a temperature_t array with that number
     *     of elements.
     *     Caller is responsible for allocating temperature_t array |list| of
     *     large enough size (not less than returned number of temperatures).
     *     If |list| is not NULL and this method returns non-negative value,
     *     it's filled with the current temperatures. If the resulting
     *     temperature list is longer than |size| elements, the remaining
     *     temperatures are discarded and not stored, but counted for the value
     *     returned by this method.
     *     The order of temperatures of built-in devices (such as CPUs, GPUs and
     *     etc.) in the |list| is kept the same regardless the number of calls
     *     to this method even if they go offline, if these devices exist on
     *     boot. The method always returns and never removes such temperatures.
     * @param size The capacity of |list|, in elements, if |list| is not NULL.
     *
     * @return number of temperatures or negative value -errno on error.
     *
     */
    ssize_t (*getTemperatures)(struct thermal_module *module, temperature_t *list, size_t size);

    /*
     * (*getCpuUsages) is called to get CPU usage information of each core:
     *     active and total times in ms since first boot.
     *
     * @param list If NULL, this method only returns number of cores and caller
     *     should allocate a cpu_usage_t array with that number of elements.
     *     Caller is responsible for allocating cpu_usage_t array |list| of
     *     large enough size (not less than returned number of CPUs).
     *     If |list| is not NULL and this method returns non-negative value,
     *     it's filled with the current CPU usages.
     *     The order of CPUs in the |list| is kept the same regardless the
     *     number of calls to this method.
     *
     * @return constant number of CPUs or negative value -errno on error.
     *
     */
    ssize_t (*getCpuUsages)(struct thermal_module *module, cpu_usage_t *list);

    /*
     * (*getCoolingDevices) is called to get the cooling devices information.
     *
     * @param list If NULL, this method only returns number of cooling devices
     *     and caller should allocate a cooling_device_t array with that number
     *     of elements.
     *     Caller is responsible for allocating cooling_device_t array |list| of
     *     large enough size (not less than returned number of cooling devices).
     *     If |list| is not NULL and this method returns non-negative value,
     *     it's filled with the current cooling device information. If the
     *     resulting cooling device list is longer than |size| elements, the
     *     remaining cooling device informations are discarded and not stored,
     *     but counted for the value returned by this method.
     *     The order of built-in coolling devices in the |list| is kept the same
     *     regardless the number of calls to this method even if they go
     *     offline, if these devices exist on boot. The method always returns
     *     and never removes from the list such coolling devices.
     * @param size The capacity of |list|, in elements, if |list| is not NULL.
     *
     * @return number of cooling devices or negative value -errno on error.
     *
     */
    ssize_t (*getCoolingDevices)(struct thermal_module *module, cooling_device_t *list,
                                 size_t size);

} thermal_module_t;

__END_DECLS

#endif  // ANDROID_INCLUDE_HARDWARE_THERMAL_H
