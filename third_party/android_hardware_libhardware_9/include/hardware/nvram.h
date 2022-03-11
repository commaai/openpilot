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

#ifndef ANDROID_HARDWARE_NVRAM_H
#define ANDROID_HARDWARE_NVRAM_H

#include <stdint.h>
#include <sys/cdefs.h>

#include <hardware/hardware.h>
#include <hardware/nvram_defs.h>

__BEGIN_DECLS

/* The id of this module. */
#define NVRAM_HARDWARE_MODULE_ID "nvram"
#define NVRAM_HARDWARE_DEVICE_ID "nvram-dev"

/* The version of this module. */
#define NVRAM_MODULE_API_VERSION_0_1 HARDWARE_MODULE_API_VERSION(0, 1)
#define NVRAM_DEVICE_API_VERSION_1_1 HARDWARE_DEVICE_API_VERSION(1, 1)

struct nvram_module {
    /**
     * Common methods of the nvram_module. This *must* be the first member of
     * nvram_module as users of this structure will cast a hw_module_t to
     * nvram_module pointer in contexts where it's known the hw_module_t
     * references a nvram_module.
     */
    hw_module_t common;

    /* There are no module methods other than the common ones. */
};

struct nvram_device {
    /**
     * Common methods of the nvram_device.  This *must* be the first member of
     * nvram_device as users of this structure will cast a hw_device_t to
     * nvram_device pointer in contexts where it's known the hw_device_t
     * references a nvram_device.
     */
    struct hw_device_t common;

    /**
     * Outputs the total number of bytes available in NVRAM. This will
     * always be at least 2048. If an implementation does not know the
     * total size it may provide an estimate or 2048.
     *
     *   device - The nvram_device instance.
     *   total_size - Receives the output. Cannot be NULL.
     */
    nvram_result_t (*get_total_size_in_bytes)(const struct nvram_device* device,
                                              uint64_t* total_size);

    /**
     * Outputs the unallocated number of bytes available in NVRAM. If an
     * implementation does not know the available size it may provide an
     * estimate or the total size.
     *
     *   device - The nvram_device instance.
     *   available_size - Receives the output. Cannot be NULL.
     */
    nvram_result_t (*get_available_size_in_bytes)(
        const struct nvram_device* device, uint64_t* available_size);

    /**
     * Outputs the maximum number of bytes that can be allocated for a single
     * space. This will always be at least 32. If an implementation does not
     * limit the maximum size it may provide the total size.
     *
     *   device - The nvram_device instance.
     *   max_space_size - Receives the output. Cannot be NULL.
     */
    nvram_result_t (*get_max_space_size_in_bytes)(
        const struct nvram_device* device, uint64_t* max_space_size);

    /**
     * Outputs the maximum total number of spaces that may be allocated.
     * This will always be at least 8. Outputs NV_UNLIMITED_SPACES if any
     * number of spaces are supported (limited only to available NVRAM
     * bytes).
     *
     *   device - The nvram_device instance.
     *   num_spaces - Receives the output. Cannot be NULL.
     */
    nvram_result_t (*get_max_spaces)(const struct nvram_device* device,
                                     uint32_t* num_spaces);

    /**
     * Outputs a list of created space indices. If |max_list_size| is
     * 0, only |list_size| is populated.
     *
     *   device - The nvram_device instance.
     *   max_list_size - The number of items in the |space_index_list|
     *                   array.
     *   space_index_list - Receives the list of created spaces up to the
     *                      given |max_list_size|. May be NULL if
     *                      |max_list_size| is 0.
     *   list_size - Receives the number of items populated in
     *               |space_index_list|, or the number of items available
     *               if |space_index_list| is NULL.
     */
    nvram_result_t (*get_space_list)(const struct nvram_device* device,
                                     uint32_t max_list_size,
                                     uint32_t* space_index_list,
                                     uint32_t* list_size);

    /**
     * Outputs the size, in bytes, of a given space.
     *
     *   device - The nvram_device instance.
     *   index - The space index.
     *   size - Receives the output. Cannot be NULL.
     */
    nvram_result_t (*get_space_size)(const struct nvram_device* device,
                                     uint32_t index, uint64_t* size);

    /**
     * Outputs the list of controls associated with a given space.
     *
     *   device - The nvram_device instance.
     *   index - The space index.
     *   max_list_size - The number of items in the |control_list| array.
     *   control_list - Receives the list of controls up to the given
     *                  |max_list_size|. May be NULL if |max_list_size|
     *                  is 0.
     *   list_size - Receives the number of items populated in
     *               |control_list|, or the number of items available if
     *               |control_list| is NULL.
     */
    nvram_result_t (*get_space_controls)(const struct nvram_device* device,
                                         uint32_t index, uint32_t max_list_size,
                                         nvram_control_t* control_list,
                                         uint32_t* list_size);

    /**
     * Outputs whether locks are enabled for the given space. When a lock
     * is enabled, the operation is disabled and any attempt to perform that
     * operation will result in NV_RESULT_OPERATION_DISABLED.
     *
     *   device - The nvram_device instance.
     *   index - The space index.
     *   write_lock_enabled - Will be set to non-zero iff write
     *                        operations are currently disabled.
     *   read_lock_enabled - Will be set to non-zero iff read operations
     *                       are currently disabled.
     */
    nvram_result_t (*is_space_locked)(const struct nvram_device* device,
                                      uint32_t index, int* write_lock_enabled,
                                      int* read_lock_enabled);

    /**
     * Creates a new space with the given index, size, controls, and
     * authorization value.
     *
     *   device - The nvram_device instance.
     *   index - An index for the new space. The index can be any 32-bit
     *           value but must not already be assigned to an existing
     *           space.
     *   size_in_bytes - The number of bytes to allocate for the space.
     *   control_list - An array of controls to enforce for the space.
     *   list_size - The number of items in |control_list|.
     *   authorization_value - If |control_list| contains
     *                         NV_CONTROL_READ_AUTHORIZATION and / or
     *                         NV_CONTROL_WRITE_AUTHORIZATION, then this
     *                         parameter provides the authorization value
     *                         for these policies (if both controls are
     *                         set then this value applies to both).
     *                         Otherwise, this value is ignored and may
     *                         be NULL.
     *   authorization_value_size - The number of bytes in
     *                              |authorization_value|.
     */
    nvram_result_t (*create_space)(const struct nvram_device* device,
                                   uint32_t index, uint64_t size_in_bytes,
                                   const nvram_control_t* control_list,
                                   uint32_t list_size,
                                   const uint8_t* authorization_value,
                                   uint32_t authorization_value_size);

    /**
     * Deletes a space.
     *
     *   device - The nvram_device instance.
     *   index - The space index.
     *   authorization_value - If the space has the
     *                         NV_CONTROL_WRITE_AUTHORIZATION policy,
     *                         then this parameter provides the
     *                         authorization value. Otherwise, this value
     *                         is ignored and may be NULL.
     *   authorization_value_size - The number of bytes in
     *                              |authorization_value|.
     */
    nvram_result_t (*delete_space)(const struct nvram_device* device,
                                   uint32_t index,
                                   const uint8_t* authorization_value,
                                   uint32_t authorization_value_size);

    /**
     * Disables any further creation of spaces until the next full device
     * reset (as in factory reset, not reboot). Subsequent calls to
     * NV_CreateSpace should return NV_RESULT_OPERATION_DISABLED.
     *
     *   device - The nvram_device instance.
     */
    nvram_result_t (*disable_create)(const struct nvram_device* device);

    /**
     * Writes the contents of a space. If the space is configured with
     * NV_CONTROL_WRITE_EXTEND then the input data is used to extend the
     * current data.
     *
     *   device - The nvram_device instance.
     *   index - The space index.
     *   buffer - The data to write.
     *   buffer_size - The number of bytes in |buffer|. If this is less
     *                 than the size of the space, the remaining bytes
     *                 will be set to 0x00. If this is more than the size
     *                 of the space, returns NV_RESULT_INVALID_PARAMETER.
     *   authorization_value - If the space has the
     *                         NV_CONTROL_WRITE_AUTHORIZATION policy,
     *                         then this parameter provides the
     *                         authorization value. Otherwise, this value
     *                         is ignored and may be NULL.
     *   authorization_value_size - The number of bytes in
     *                              |authorization_value|.
     */
    nvram_result_t (*write_space)(const struct nvram_device* device,
                                  uint32_t index, const uint8_t* buffer,
                                  uint64_t buffer_size,
                                  const uint8_t* authorization_value,
                                  uint32_t authorization_value_size);

    /**
     * Reads the contents of a space. If the space has never been
     * written, all bytes read will be 0x00.
     *
     *   device - The nvram_device instance.
     *   index - The space index.
     *   num_bytes_to_read - The number of bytes to read; |buffer| must
     *                       be large enough to hold this many bytes. If
     *                       this is more than the size of the space, the
     *                       entire space is read. If this is less than
     *                       the size of the space, the first bytes in
     *                       the space are read.
     *   authorization_value - If the space has the
     *                         NV_CONTROL_READ_AUTHORIZATION policy, then
     *                         this parameter provides the authorization
     *                         value. Otherwise, this value is ignored
     *                         and may be NULL.
     *   authorization_value_size - The number of bytes in
     *                              |authorization_value|.
     *   buffer - Receives the data read from the space. Must be at least
     *            |num_bytes_to_read| bytes in size.
     *   bytes_read - The number of bytes read. If NV_RESULT_SUCCESS is
     *                returned this will be set to the smaller of
     *                |num_bytes_to_read| or the size of the space.
     */
    nvram_result_t (*read_space)(const struct nvram_device* device,
                                 uint32_t index, uint64_t num_bytes_to_read,
                                 const uint8_t* authorization_value,
                                 uint32_t authorization_value_size,
                                 uint8_t* buffer, uint64_t* bytes_read);

    /**
     * Enables a write lock for the given space according to its policy.
     * If the space does not have NV_CONTROL_PERSISTENT_WRITE_LOCK or
     * NV_CONTROL_BOOT_WRITE_LOCK set then this function has no effect
     * and may return an error.
     *
     *   device - The nvram_device instance.
     *   index - The space index.
     *   authorization_value - If the space has the
     *                         NV_CONTROL_WRITE_AUTHORIZATION policy,
     *                         then this parameter provides the
     *                         authorization value. Otherwise, this value
     *                         is ignored and may be NULL.
     *   authorization_value_size - The number of bytes in
     *                              |authorization_value|.
     */
    nvram_result_t (*enable_write_lock)(const struct nvram_device* device,
                                        uint32_t index,
                                        const uint8_t* authorization_value,
                                        uint32_t authorization_value_size);

    /**
     * Enables a read lock for the given space according to its policy.
     * If the space does not have NV_CONTROL_BOOT_READ_LOCK set then this
     * function has no effect and may return an error.
     *
     *   device - The nvram_device instance.
     *   index - The space index.
     *   authorization_value - If the space has the
     *                         NV_CONTROL_READ_AUTHORIZATION policy, then
     *                         this parameter provides the authorization
     *                         value. (Note that there is no requirement
     *                         for write access in order to lock for
     *                         reading. A read lock is always volatile.)
     *                         Otherwise, this value is ignored and may
     *                         be NULL.
     *   authorization_value_size - The number of bytes in
     *                              |authorization_value|.
     */
    nvram_result_t (*enable_read_lock)(const struct nvram_device* device,
                                       uint32_t index,
                                       const uint8_t* authorization_value,
                                       uint32_t authorization_value_size);
};

typedef struct nvram_device nvram_device_t;

/* Convenience API for opening and closing nvram devices. */
static inline int nvram_open(const struct hw_module_t* module,
                             nvram_device_t** device) {
    return module->methods->open(module, NVRAM_HARDWARE_DEVICE_ID,
                                 TO_HW_DEVICE_T_OPEN(device));
}

static inline int nvram_close(nvram_device_t* device) {
    return device->common.close(&device->common);
}

__END_DECLS

#endif  // ANDROID_HARDWARE_NVRAM_H
