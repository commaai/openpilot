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

#ifndef ANDROID_NFC_TAG_HAL_INTERFACE_H
#define ANDROID_NFC_TAG_HAL_INTERFACE_H

#include <stdint.h>

#include <hardware/hardware.h>

__BEGIN_DECLS

/*
 * HAL for programmable NFC tags.
 *
 */

#define NFC_TAG_HARDWARE_MODULE_ID "nfc_tag"
#define NFC_TAG_ID "tag"

typedef struct nfc_tag_module_t {
    /**
     * Common methods of the NFC tag module.  This *must* be the first member of
     * nfc_tag_module_t as users of this structure will cast a hw_module_t to
     * nfc_tag_module_t pointer in contexts where it's known the hw_module_t references a
     * nfc_tag_module_t.
     */
    struct hw_module_t common;
} nfc_tag_module_t;

typedef struct nfc_tag_device {
    /**
     * Common methods of the NFC tag device.  This *must* be the first member of
     * nfc_tag_device_t as users of this structure will cast a hw_device_t to
     * nfc_tag_device_t pointer in contexts where it's known the hw_device_t references a
     * nfc_tag_device_t.
     */
    struct hw_device_t common;

    /**
     * Initialize the NFC tag.
     *
     * The driver must:
     *   * Set the static lock bytes to read only
     *   * Configure the Capability Container to disable write acess
     *         eg: 0xE1 0x10 <size> 0x0F
     *
     * This function is called once before any calls to setContent().
     *
     * Return 0 on success or -errno on error.
     */
    int (*init)(const struct nfc_tag_device *dev);

    /**
     * Set the NFC tag content.
     *
     * The driver must write <data> in the data area of the tag starting at
     * byte 0 of block 4 and zero the rest of the data area.
     *
     * Returns 0 on success or -errno on error.
     */
    int (*setContent)(const struct nfc_tag_device *dev, const uint8_t *data, size_t len);

    /**
     * Returns the memory size of the data area.
     */
    int (*getMemorySize)(const struct nfc_tag_device *dev);
} nfc_tag_device_t;

static inline int nfc_tag_open(const struct hw_module_t* module,
                               nfc_tag_device_t** dev) {
    return module->methods->open(module, NFC_TAG_ID,
                                 (struct hw_device_t**)dev);
}

static inline int nfc_tag_close(nfc_tag_device_t* dev) {
    return dev->common.close(&dev->common);
}

__END_DECLS

#endif // ANDROID_NFC_TAG_HAL_INTERFACE_H
