/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_HARDWARE_CONSUMERIR_H
#define ANDROID_INCLUDE_HARDWARE_CONSUMERIR_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <hardware/hardware.h>
#include <hardware/hwcomposer_defs.h>

#define CONSUMERIR_MODULE_API_VERSION_1_0 HARDWARE_MODULE_API_VERSION(1, 0)
#define CONSUMERIR_HARDWARE_MODULE_ID "consumerir"
#define CONSUMERIR_TRANSMITTER "transmitter"

typedef struct consumerir_freq_range {
    int min;
    int max;
} consumerir_freq_range_t;

typedef struct consumerir_module {
    /**
     * Common methods of the consumer IR module.  This *must* be the first member of
     * consumerir_module as users of this structure will cast a hw_module_t to
     * consumerir_module pointer in contexts where it's known the hw_module_t references a
     * consumerir_module.
     */
    struct hw_module_t common;
} consumerir_module_t;

typedef struct consumerir_device {
    /**
     * Common methods of the consumer IR device.  This *must* be the first member of
     * consumerir_device as users of this structure will cast a hw_device_t to
     * consumerir_device pointer in contexts where it's known the hw_device_t references a
     * consumerir_device.
     */
    struct hw_device_t common;

    /*
     * (*transmit)() is called to by the ConsumerIrService to send an IR pattern
     * at a given carrier_freq.
     *
     * The pattern is alternating series of carrier on and off periods measured in
     * microseconds.  The carrier should be turned off at the end of a transmit
     * even if there are and odd number of entries in the pattern array.
     *
     * This call should return when the transmit is complete or encounters an error.
     *
     * returns: 0 on success. A negative error code on error.
     */
    int (*transmit)(struct consumerir_device *dev, int carrier_freq,
            const int pattern[], int pattern_len);

    /*
     * (*get_num_carrier_freqs)() is called by the ConsumerIrService to get the
     * number of carrier freqs to allocate space for, which is then filled by
     * a subsequent call to (*get_carrier_freqs)().
     *
     * returns: the number of ranges on success. A negative error code on error.
     */
    int (*get_num_carrier_freqs)(struct consumerir_device *dev);

    /*
     * (*get_carrier_freqs)() is called by the ConsumerIrService to enumerate
     * which frequencies the IR transmitter supports.  The HAL implementation
     * should fill an array of consumerir_freq_range structs with the
     * appropriate values for the transmitter, up to len elements.
     *
     * returns: the number of ranges on success. A negative error code on error.
     */
    int (*get_carrier_freqs)(struct consumerir_device *dev,
            size_t len, consumerir_freq_range_t *ranges);

    /* Reserved for future use. Must be NULL. */
    void* reserved[8 - 3];
} consumerir_device_t;

#endif /* ANDROID_INCLUDE_HARDWARE_CONSUMERIR_H */
