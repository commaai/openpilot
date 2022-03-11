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

#ifndef ANDROID_INCLUDE_HARDWARE_BOOT_CONTROL_H
#define ANDROID_INCLUDE_HARDWARE_BOOT_CONTROL_H

#include <hardware/hardware.h>

__BEGIN_DECLS

#define BOOT_CONTROL_MODULE_API_VERSION_0_1  HARDWARE_MODULE_API_VERSION(0, 1)

/**
 * The id of this module
 */
#define BOOT_CONTROL_HARDWARE_MODULE_ID "bootctrl"

/*
 * The Boot Control HAL is designed to allow for managing sets of redundant
 * partitions, called slots, that can be booted from independantly. Slots
 * are sets of partitions whose names differ only by a given suffix.
 * They are identified here by a 0 indexed number, and associated with their
 * suffix, which can be appended to the base name for any particular partition
 * to find the one associated with that slot. The bootloader must pass the suffix
 * of the currently active slot either through a kernel command line property at
 * androidboot.slot_suffix, or the device tree at /firmware/android/slot_suffix.
 * The primary use of this set up is to allow for background updates while the
 * device is running, and to provide a fallback in the event that the update fails.
 */


/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
typedef struct boot_control_module {
    struct hw_module_t common;

    /*
     * (*init)() perform any initialization tasks needed for the HAL.
     * This is called only once.
     */
    void (*init)(struct boot_control_module *module);

    /*
     * (*getNumberSlots)() returns the number of available slots.
     * For instance, a system with a single set of partitions would return
     * 1, a system with A/B would return 2, A/B/C -> 3...
     */
    unsigned (*getNumberSlots)(struct boot_control_module *module);

    /*
     * (*getCurrentSlot)() returns the value letting the system know
     * whether the current slot is A or B. The meaning of A and B is
     * left up to the implementer. It is assumed that if the current slot
     * is A, then the block devices underlying B can be accessed directly
     * without any risk of corruption.
     * The returned value is always guaranteed to be strictly less than the
     * value returned by getNumberSlots. Slots start at 0 and
     * finish at getNumberSlots() - 1
     */
    unsigned (*getCurrentSlot)(struct boot_control_module *module);

    /*
     * (*markBootSuccessful)() marks the current slot
     * as having booted successfully
     *
     * Returns 0 on success, -errno on error.
     */
    int (*markBootSuccessful)(struct boot_control_module *module);

    /*
     * (*setActiveBootSlot)() marks the slot passed in parameter as
     * the active boot slot (see getCurrentSlot for an explanation
     * of the "slot" parameter). This overrides any previous call to
     * setSlotAsUnbootable.
     * Returns 0 on success, -errno on error.
     */
    int (*setActiveBootSlot)(struct boot_control_module *module, unsigned slot);

    /*
     * (*setSlotAsUnbootable)() marks the slot passed in parameter as
     * an unbootable. This can be used while updating the contents of the slot's
     * partitions, so that the system will not attempt to boot a known bad set up.
     * Returns 0 on success, -errno on error.
     */
    int (*setSlotAsUnbootable)(struct boot_control_module *module, unsigned slot);

    /*
     * (*isSlotBootable)() returns if the slot passed in parameter is
     * bootable. Note that slots can be made unbootable by both the
     * bootloader and by the OS using setSlotAsUnbootable.
     * Returns 1 if the slot is bootable, 0 if it's not, and -errno on
     * error.
     */
    int (*isSlotBootable)(struct boot_control_module *module, unsigned slot);

    /*
     * (*getSuffix)() returns the string suffix used by partitions that
     * correspond to the slot number passed in parameter. The returned string
     * is expected to be statically allocated and not need to be freed.
     * Returns NULL if slot does not match an existing slot.
     */
    const char* (*getSuffix)(struct boot_control_module *module, unsigned slot);

    /*
     * (*isSlotMarkedSucessful)() returns if the slot passed in parameter has
     * been marked as successful using markBootSuccessful.
     * Returns 1 if the slot has been marked as successful, 0 if it's
     * not the case, and -errno on error.
     */
    int (*isSlotMarkedSuccessful)(struct boot_control_module *module, unsigned slot);

    void* reserved[31];
} boot_control_module_t;


__END_DECLS

#endif  // ANDROID_INCLUDE_HARDWARE_BOOT_CONTROL_H
