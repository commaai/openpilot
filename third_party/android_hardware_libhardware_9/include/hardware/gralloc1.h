/*
 * Copyright 2015 The Android Open Source Project
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

#ifndef ANDROID_HARDWARE_GRALLOC1_H
#define ANDROID_HARDWARE_GRALLOC1_H

#include <hardware/hardware.h>
#include <cutils/native_handle.h>

__BEGIN_DECLS

#define GRALLOC_MODULE_API_VERSION_1_0 HARDWARE_MODULE_API_VERSION(1, 0)
#define GRALLOC_HARDWARE_MODULE_ID "gralloc"

/*
 * Enums
 */

typedef enum {
    GRALLOC1_CAPABILITY_INVALID = 0,

    /* If this capability is supported, then the outBuffers parameter to
     * allocate may be NULL, which instructs the device to report whether the
     * given allocation is possible or not. */
    GRALLOC1_CAPABILITY_TEST_ALLOCATE = 1,

    /* If this capability is supported, then the implementation supports
     * allocating buffers with more than one image layer. */
    GRALLOC1_CAPABILITY_LAYERED_BUFFERS = 2,

    /* If this capability is supported, then the implementation always closes
     * and deletes a buffer handle whenever the last reference is removed.
     *
     * Supporting this capability is strongly recommended.  It will become
     * mandatory in future releases. */
    GRALLOC1_CAPABILITY_RELEASE_IMPLY_DELETE = 3,

    GRALLOC1_LAST_CAPABILITY = 3,
} gralloc1_capability_t;

typedef enum {
    GRALLOC1_CONSUMER_USAGE_NONE = 0,
    GRALLOC1_CONSUMER_USAGE_CPU_READ_NEVER = 0,
    /* 1ULL << 0 */
    GRALLOC1_CONSUMER_USAGE_CPU_READ = 1ULL << 1,
    GRALLOC1_CONSUMER_USAGE_CPU_READ_OFTEN = 1ULL << 2 |
            GRALLOC1_CONSUMER_USAGE_CPU_READ,
    /* 1ULL << 3 */
    /* 1ULL << 4 */
    /* 1ULL << 5 */
    /* 1ULL << 6 */
    /* 1ULL << 7 */
    GRALLOC1_CONSUMER_USAGE_GPU_TEXTURE = 1ULL << 8,
    /* 1ULL << 9 */
    /* 1ULL << 10 */
    GRALLOC1_CONSUMER_USAGE_HWCOMPOSER = 1ULL << 11,
    GRALLOC1_CONSUMER_USAGE_CLIENT_TARGET = 1ULL << 12,
    /* 1ULL << 13 */
    /* 1ULL << 14 */
    GRALLOC1_CONSUMER_USAGE_CURSOR = 1ULL << 15,
    GRALLOC1_CONSUMER_USAGE_VIDEO_ENCODER = 1ULL << 16,
    /* 1ULL << 17 */
    GRALLOC1_CONSUMER_USAGE_CAMERA = 1ULL << 18,
    /* 1ULL << 19 */
    GRALLOC1_CONSUMER_USAGE_RENDERSCRIPT = 1ULL << 20,

    /* Indicates that the consumer may attach buffers to their end of the
     * BufferQueue, which means that the producer may never have seen a given
     * dequeued buffer before. May be ignored by the gralloc device. */
    GRALLOC1_CONSUMER_USAGE_FOREIGN_BUFFERS = 1ULL << 21,

    /* 1ULL << 22 */
    GRALLOC1_CONSUMER_USAGE_GPU_DATA_BUFFER = 1ULL << 23,
    /* 1ULL << 24 */
    /* 1ULL << 25 */
    /* 1ULL << 26 */
    /* 1ULL << 27 */

    /* Bits reserved for implementation-specific usage flags */
    GRALLOC1_CONSUMER_USAGE_PRIVATE_0 = 1ULL << 28,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_1 = 1ULL << 29,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_2 = 1ULL << 30,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_3 = 1ULL << 31,

    /* 1ULL << 32 */
    /* 1ULL << 33 */
    /* 1ULL << 34 */
    /* 1ULL << 35 */
    /* 1ULL << 36 */
    /* 1ULL << 37 */
    /* 1ULL << 38 */
    /* 1ULL << 39 */
    /* 1ULL << 40 */
    /* 1ULL << 41 */
    /* 1ULL << 42 */
    /* 1ULL << 43 */
    /* 1ULL << 44 */
    /* 1ULL << 45 */
    /* 1ULL << 46 */
    /* 1ULL << 47 */

    /* Bits reserved for implementation-specific usage flags */
    GRALLOC1_CONSUMER_USAGE_PRIVATE_19 = 1ULL << 48,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_18 = 1ULL << 49,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_17 = 1ULL << 50,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_16 = 1ULL << 51,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_15 = 1ULL << 52,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_14 = 1ULL << 53,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_13 = 1ULL << 54,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_12 = 1ULL << 55,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_11 = 1ULL << 56,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_10 = 1ULL << 57,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_9 = 1ULL << 58,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_8 = 1ULL << 59,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_7 = 1ULL << 60,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_6 = 1ULL << 61,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_5 = 1ULL << 62,
    GRALLOC1_CONSUMER_USAGE_PRIVATE_4 = 1ULL << 63,
} gralloc1_consumer_usage_t;

typedef enum {
    GRALLOC1_FUNCTION_INVALID = 0,
    GRALLOC1_FUNCTION_DUMP = 1,
    GRALLOC1_FUNCTION_CREATE_DESCRIPTOR = 2,
    GRALLOC1_FUNCTION_DESTROY_DESCRIPTOR = 3,
    GRALLOC1_FUNCTION_SET_CONSUMER_USAGE = 4,
    GRALLOC1_FUNCTION_SET_DIMENSIONS = 5,
    GRALLOC1_FUNCTION_SET_FORMAT = 6,
    GRALLOC1_FUNCTION_SET_PRODUCER_USAGE = 7,
    GRALLOC1_FUNCTION_GET_BACKING_STORE = 8,
    GRALLOC1_FUNCTION_GET_CONSUMER_USAGE = 9,
    GRALLOC1_FUNCTION_GET_DIMENSIONS = 10,
    GRALLOC1_FUNCTION_GET_FORMAT = 11,
    GRALLOC1_FUNCTION_GET_PRODUCER_USAGE = 12,
    GRALLOC1_FUNCTION_GET_STRIDE = 13,
    GRALLOC1_FUNCTION_ALLOCATE = 14,
    GRALLOC1_FUNCTION_RETAIN = 15,
    GRALLOC1_FUNCTION_RELEASE = 16,
    GRALLOC1_FUNCTION_GET_NUM_FLEX_PLANES = 17,
    GRALLOC1_FUNCTION_LOCK = 18,
    GRALLOC1_FUNCTION_LOCK_FLEX = 19,
    GRALLOC1_FUNCTION_UNLOCK = 20,
    GRALLOC1_FUNCTION_SET_LAYER_COUNT = 21,
    GRALLOC1_FUNCTION_GET_LAYER_COUNT = 22,
    GRALLOC1_FUNCTION_VALIDATE_BUFFER_SIZE = 23,
    GRALLOC1_FUNCTION_GET_TRANSPORT_SIZE = 24,
    GRALLOC1_FUNCTION_IMPORT_BUFFER = 25,
    GRALLOC1_LAST_FUNCTION = 25,
} gralloc1_function_descriptor_t;

typedef enum {
    GRALLOC1_ERROR_NONE = 0,
    GRALLOC1_ERROR_BAD_DESCRIPTOR = 1,
    GRALLOC1_ERROR_BAD_HANDLE = 2,
    GRALLOC1_ERROR_BAD_VALUE = 3,
    GRALLOC1_ERROR_NOT_SHARED = 4,
    GRALLOC1_ERROR_NO_RESOURCES = 5,
    GRALLOC1_ERROR_UNDEFINED = 6,
    GRALLOC1_ERROR_UNSUPPORTED = 7,
} gralloc1_error_t;

typedef enum {
    GRALLOC1_PRODUCER_USAGE_NONE = 0,
    GRALLOC1_PRODUCER_USAGE_CPU_WRITE_NEVER = 0,
    /* 1ULL << 0 */
    GRALLOC1_PRODUCER_USAGE_CPU_READ = 1ULL << 1,
    GRALLOC1_PRODUCER_USAGE_CPU_READ_OFTEN = 1ULL << 2 |
            GRALLOC1_PRODUCER_USAGE_CPU_READ,
    /* 1ULL << 3 */
    /* 1ULL << 4 */
    GRALLOC1_PRODUCER_USAGE_CPU_WRITE = 1ULL << 5,
    GRALLOC1_PRODUCER_USAGE_CPU_WRITE_OFTEN = 1ULL << 6 |
            GRALLOC1_PRODUCER_USAGE_CPU_WRITE,
    /* 1ULL << 7 */
    /* 1ULL << 8 */
    GRALLOC1_PRODUCER_USAGE_GPU_RENDER_TARGET = 1ULL << 9,
    /* 1ULL << 10 */
    /* 1ULL << 11 */
    /* 1ULL << 12 */
    /* 1ULL << 13 */

    /* The consumer must have a hardware-protected path to an external display
     * sink for this buffer. If a hardware-protected path is not available, then
     * do not attempt to display this buffer. */
    GRALLOC1_PRODUCER_USAGE_PROTECTED = 1ULL << 14,

    /* 1ULL << 15 */
    /* 1ULL << 16 */
    GRALLOC1_PRODUCER_USAGE_CAMERA = 1ULL << 17,
    /* 1ULL << 18 */
    /* 1ULL << 19 */
    /* 1ULL << 20 */
    /* 1ULL << 21 */
    GRALLOC1_PRODUCER_USAGE_VIDEO_DECODER = 1ULL << 22,
    GRALLOC1_PRODUCER_USAGE_SENSOR_DIRECT_DATA = 1ULL << 23,
    /* 1ULL << 24 */
    /* 1ULL << 25 */
    /* 1ULL << 26 */
    /* 1ULL << 27 */

    /* Bits reserved for implementation-specific usage flags */
    GRALLOC1_PRODUCER_USAGE_PRIVATE_0 = 1ULL << 28,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_1 = 1ULL << 29,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_2 = 1ULL << 30,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_3 = 1ULL << 31,

    /* 1ULL << 32 */
    /* 1ULL << 33 */
    /* 1ULL << 34 */
    /* 1ULL << 35 */
    /* 1ULL << 36 */
    /* 1ULL << 37 */
    /* 1ULL << 38 */
    /* 1ULL << 39 */
    /* 1ULL << 40 */
    /* 1ULL << 41 */
    /* 1ULL << 42 */
    /* 1ULL << 43 */
    /* 1ULL << 44 */
    /* 1ULL << 45 */
    /* 1ULL << 46 */
    /* 1ULL << 47 */

    /* Bits reserved for implementation-specific usage flags */
    GRALLOC1_PRODUCER_USAGE_PRIVATE_19 = 1ULL << 48,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_18 = 1ULL << 49,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_17 = 1ULL << 50,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_16 = 1ULL << 51,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_15 = 1ULL << 52,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_14 = 1ULL << 53,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_13 = 1ULL << 54,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_12 = 1ULL << 55,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_11 = 1ULL << 56,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_10 = 1ULL << 57,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_9 = 1ULL << 58,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_8 = 1ULL << 59,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_7 = 1ULL << 60,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_6 = 1ULL << 61,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_5 = 1ULL << 62,
    GRALLOC1_PRODUCER_USAGE_PRIVATE_4 = 1ULL << 63,
} gralloc1_producer_usage_t;

/*
 * Typedefs
 */

typedef void (*gralloc1_function_pointer_t)();

typedef uint64_t gralloc1_backing_store_t;
typedef uint64_t gralloc1_buffer_descriptor_t;

/*
 * Device Struct
 */

typedef struct gralloc1_device {
    /* Must be the first member of this struct, since a pointer to this struct
     * will be generated by casting from a hw_device_t* */
    struct hw_device_t common;

    /* getCapabilities(..., outCount, outCapabilities)
     *
     * Provides a list of capabilities (described in the definition of
     * gralloc1_capability_t above) supported by this device. This list must not
     * change after the device has been loaded.
     *
     * Parameters:
     *   outCount - if outCapabilities was NULL, the number of capabilities
     *       which would have been returned; if outCapabilities was not NULL,
     *       the number of capabilities returned, which must not exceed the
     *       value stored in outCount prior to the call
     *   outCapabilities - a list of capabilities supported by this device; may
     *       be NULL, in which case this function must write into outCount the
     *       number of capabilities which would have been written into
     *       outCapabilities
     */
    void (*getCapabilities)(struct gralloc1_device* device, uint32_t* outCount,
            int32_t* /*gralloc1_capability_t*/ outCapabilities);

    /* getFunction(..., descriptor)
     *
     * Returns a function pointer which implements the requested description.
     *
     * Parameters:
     *   descriptor - the function to return
     *
     * Returns either a function pointer implementing the requested descriptor
     *   or NULL if the described function is not supported by this device.
     */
    gralloc1_function_pointer_t (*getFunction)(struct gralloc1_device* device,
            int32_t /*gralloc1_function_descriptor_t*/ descriptor);
} gralloc1_device_t;

static inline int gralloc1_open(const struct hw_module_t* module,
        gralloc1_device_t** device) {
    return module->methods->open(module, GRALLOC_HARDWARE_MODULE_ID,
            TO_HW_DEVICE_T_OPEN(device));
}

static inline int gralloc1_close(gralloc1_device_t* device) {
    return device->common.close(&device->common);
}

/* dump(..., outSize, outBuffer)
 * Function descriptor: GRALLOC1_FUNCTION_DUMP
 * Must be provided by all gralloc1 devices
 *
 * Retrieves implementation-defined debug information, which will be displayed
 * during, for example, `dumpsys SurfaceFlinger`.
 *
 * If called with outBuffer == NULL, the device should store a copy of the
 * desired output and return its length in bytes in outSize. If the device
 * already has a stored copy, that copy should be purged and replaced with a
 * fresh copy.
 *
 * If called with outBuffer != NULL, the device should copy its stored version
 * of the output into outBuffer and store how many bytes of data it copied into
 * outSize. Prior to this call, the client will have populated outSize with the
 * maximum number of bytes outBuffer can hold. The device must not write more
 * than this amount into outBuffer. If the device does not currently have a
 * stored copy, then it should return 0 in outSize.
 *
 * Any data written into outBuffer need not be null-terminated.
 *
 * Parameters:
 *   outSize - if outBuffer was NULL, the number of bytes needed to copy the
 *       device's stored output; if outBuffer was not NULL, the number of bytes
 *       written into it, which must not exceed the value stored in outSize
 *       prior to the call; pointer will be non-NULL
 *   outBuffer - the buffer to write the dump output into; may be NULL as
 *       described above; data written into this buffer need not be
 *       null-terminated
 */
typedef void (*GRALLOC1_PFN_DUMP)(gralloc1_device_t* device, uint32_t* outSize,
        char* outBuffer);

/*
 * Buffer descriptor lifecycle functions
 *
 * All of these functions take as their first parameter a device pointer, so
 * this parameter is omitted from the described parameter lists.
 */

/* createDescriptor(..., outDescriptor)
 * Function descriptor: GRALLOC1_FUNCTION_CREATE_DESCRIPTOR
 * Must be provided by all gralloc1 devices
 *
 * Creates a new, empty buffer descriptor.
 *
 * Parameters:
 *   outDescriptor - the new buffer descriptor
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_NO_RESOURCES - no more descriptors can currently be created
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_CREATE_DESCRIPTOR)(
        gralloc1_device_t* device, gralloc1_buffer_descriptor_t* outDescriptor);

/* destroyDescriptor(..., descriptor)
 * Function descriptor: GRALLOC1_FUNCTION_DESTROY_DESCRIPTOR
 * Must be provided by all gralloc1 devices
 *
 * Destroys an existing buffer descriptor.
 *
 * Parameters:
 *   descriptor - the buffer descriptor to destroy
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_DESCRIPTOR - descriptor does not refer to a valid
 *       buffer descriptor
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_DESTROY_DESCRIPTOR)(
        gralloc1_device_t* device, gralloc1_buffer_descriptor_t descriptor);

/*
 * Buffer descriptor modification functions
 *
 * All of these functions take as their first two parameters a device pointer
 * and a buffer descriptor, so these parameters are omitted from the described
 * parameter lists.
 */

/* setConsumerUsage(..., usage)
 * Function descriptor: GRALLOC1_FUNCTION_SET_CONSUMER_USAGE
 * Must be provided by all gralloc1 devices
 *
 * Sets the desired consumer usage flags of the buffer.
 *
 * Valid usage flags can be found in the definition of gralloc1_consumer_usage_t
 * above.
 *
 * Parameters:
 *   usage - the desired consumer usage flags
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_DESCRIPTOR - the buffer descriptor is invalid
 *   GRALLOC1_ERROR_BAD_VALUE - an invalid usage flag was passed in
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_SET_CONSUMER_USAGE)(
        gralloc1_device_t* device, gralloc1_buffer_descriptor_t descriptor,
        uint64_t /*gralloc1_consumer_usage_t*/ usage);

/* setDimensions(..., width, height)
 * Function descriptor: GRALLOC1_FUNCTION_SET_DIMENSIONS
 * Must be provided by all gralloc1 devices
 *
 * Sets the desired width and height of the buffer in pixels.
 *
 * The width specifies how many columns of pixels should be in the allocated
 * buffer, but does not necessarily represent the offset in columns between the
 * same column in adjacent rows. If this offset is required, consult getStride
 * below.
 *
 * The height specifies how many rows of pixels should be in the allocated
 * buffer.
 *
 * Parameters:
 *   width - the desired width in pixels
 *   height - the desired height in pixels
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_DESCRIPTOR - the buffer descriptor is invalid
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_SET_DIMENSIONS)(
        gralloc1_device_t* device, gralloc1_buffer_descriptor_t descriptor,
        uint32_t width, uint32_t height);

/* setFormat(..., format)
 * Function descriptor: GRALLOC1_FUNCTION_SET_FORMAT
 * Must be provided by all gralloc1 devices
 *
 * Sets the desired format of the buffer.
 *
 * The valid formats can be found in <system/graphics.h>.
 *
 * Parameters:
 *   format - the desired format
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_DESCRIPTOR - the buffer descriptor is invalid
 *   GRALLOC1_ERROR_BAD_VALUE - format is invalid
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_SET_FORMAT)(
        gralloc1_device_t* device, gralloc1_buffer_descriptor_t descriptor,
        int32_t /*android_pixel_format_t*/ format);

/* setLayerCount(..., layerCount)
 * Function descriptor: GRALLOC1_FUNCTION_SET_LAYER_COUNT
 * Must be provided by all gralloc1 devices that provide the
 * GRALLOC1_CAPABILITY_LAYERED_BUFFERS capability.
 *
 * Sets the number of layers in the buffer.
 *
 * A buffer with multiple layers may be used as the backing store of an array
 * texture. All layers of a buffer share the same characteristics (e.g.,
 * dimensions, format, usage). Devices that do not support
 * GRALLOC1_CAPABILITY_LAYERED_BUFFERS must allocate only buffers with a single
 * layer.
 *
 * Parameters:
 *   layerCount - the desired number of layers, must be non-zero
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_DESCRIPTOR - the buffer descriptor is invalid
 *   GRALLOC1_ERROR_BAD_VALUE - the layer count is invalid
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_SET_LAYER_COUNT)(
        gralloc1_device_t* device, gralloc1_buffer_descriptor_t descriptor,
        uint32_t layerCount);

/* setProducerUsage(..., usage)
 * Function descriptor: GRALLOC1_FUNCTION_SET_PRODUCER_USAGE
 * Must be provided by all gralloc1 devices
 *
 * Sets the desired producer usage flags of the buffer.
 *
 * Valid usage flags can be found in the definition of gralloc1_producer_usage_t
 * above.
 *
 * Parameters:
 *   usage - the desired producer usage flags
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_DESCRIPTOR - the buffer descriptor is invalid
 *   GRALLOC1_ERROR_BAD_VALUE - an invalid usage flag was passed in
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_SET_PRODUCER_USAGE)(
        gralloc1_device_t* device, gralloc1_buffer_descriptor_t descriptor,
        uint64_t /*gralloc1_producer_usage_t*/ usage);

/*
 * Buffer handle query functions
 *
 * All of these functions take as their first two parameters a device pointer
 * and a buffer handle, so these parameters are omitted from the described
 * parameter lists.
 *
 * [1] Currently many of these functions may return GRALLOC1_ERROR_UNSUPPORTED,
 * which means that the device is not able to retrieve the requested information
 * from the buffer. This is necessary to enable a smooth transition from earlier
 * versions of the gralloc HAL, but gralloc1 implementers are strongly
 * discouraged from returning this value, as future versions of the platform
 * code will require all of these functions to succeed given a valid handle.
 */

/* getBackingStore(..., outStore)
 * Function descriptor: GRALLOC1_FUNCTION_GET_BACKING_STORE
 * Must be provided by all gralloc1 devices
 *
 * Gets a value that uniquely identifies the backing store of the given buffer.
 *
 * Buffers which share a backing store should return the same value from this
 * function. If the buffer is present in more than one process, the backing
 * store value for that buffer is not required to be the same in every process.
 *
 * Parameters:
 *   outStore - the backing store identifier for this buffer
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to retrieve the
 *       backing store identifier from the buffer; see note [1] in this
 *       section's header for more information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_BACKING_STORE)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        gralloc1_backing_store_t* outStore);

/* getConsumerUsage(..., outUsage)
 * Function descriptor: GRALLOC1_FUNCTION_GET_CONSUMER_USAGE
 * Must be provided by all gralloc1 devices
 *
 * Gets the consumer usage flags which were used to allocate this buffer.
 *
 * Usage flags can be found in the definition of gralloc1_consumer_usage_t above
 *
 * Parameters:
 *   outUsage - the consumer usage flags used to allocate this buffer; must be
 *       non-NULL
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to retrieve the
 *       dimensions from the buffer; see note [1] in this section's header for
 *       more information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_CONSUMER_USAGE)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        uint64_t* /*gralloc1_consumer_usage_t*/ outUsage);

/* getDimensions(..., outWidth, outHeight)
 * Function descriptor: GRALLOC1_FUNCTION_GET_DIMENSIONS
 * Must be provided by all gralloc1 devices
 *
 * Gets the width and height of the buffer in pixels.
 *
 * See setDimensions for more information about these values.
 *
 * Parameters:
 *   outWidth - the width of the buffer in pixels, must be non-NULL
 *   outHeight - the height of the buffer in pixels, must be non-NULL
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to retrieve the
 *       dimensions from the buffer; see note [1] in this section's header for
 *       more information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_DIMENSIONS)(
        gralloc1_device_t* device, buffer_handle_t buffer, uint32_t* outWidth,
        uint32_t* outHeight);

/* getFormat(..., outFormat)
 * Function descriptor: GRALLOC1_FUNCTION_GET_FORMAT
 * Must be provided by all gralloc1 devices
 *
 * Gets the format of the buffer.
 *
 * The valid formats can be found in the HAL_PIXEL_FORMAT_* enum in
 * system/graphics.h.
 *
 * Parameters:
 *   outFormat - the format of the buffer; must be non-NULL
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to retrieve the format
 *       from the buffer; see note [1] in this section's header for more
 *       information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_FORMAT)(
        gralloc1_device_t* device, buffer_handle_t descriptor,
        int32_t* outFormat);

/* getLayerCount(..., outLayerCount)
 * Function descriptor: GRALLOC1_FUNCTION_GET_LAYER_COUNT
 * Must be provided by all gralloc1 devices that provide the
 * GRALLOC1_CAPABILITY_LAYERED_BUFFERS capability.
 *
 * Gets the number of layers of the buffer.
 *
 * See setLayerCount for more information about this value.
 *
 * Parameters:
 *   outLayerCount - the number of layers in the image, must be non-NULL
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to retrieve the
 *       layer count from the buffer; see note [1] in this section's header for
 *       more information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_LAYER_COUNT)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        uint32_t* outLayerCount);

/* getProducerUsage(..., outUsage)
 * Function descriptor: GRALLOC1_FUNCTION_GET_PRODUCER_USAGE
 * Must be provided by all gralloc1 devices
 *
 * Gets the producer usage flags which were used to allocate this buffer.
 *
 * Usage flags can be found in the definition of gralloc1_producer_usage_t above
 *
 * Parameters:
 *   outUsage - the producer usage flags used to allocate this buffer; must be
 *       non-NULL
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to retrieve the usage
 *       from the buffer; see note [1] in this section's header for more
 *       information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_PRODUCER_USAGE)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        uint64_t* /*gralloc1_producer_usage_t*/ outUsage);

/* getStride(..., outStride)
 * Function descriptor: GRALLOC1_FUNCTION_GET_STRIDE
 * Must be provided by all gralloc1 devices
 *
 * Gets the stride of the buffer in pixels.
 *
 * The stride is the offset in pixel-sized elements between the same column in
 * two adjacent rows of pixels. This may not be equal to the width of the
 * buffer.
 *
 * Parameters:
 *   outStride - the stride in pixels; must be non-NULL
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNDEFINED - the notion of a stride is not meaningful for
 *       this format
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to retrieve the stride
 *       from the descriptor; see note [1] in this section's header for more
 *       information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_STRIDE)(
        gralloc1_device_t* device, buffer_handle_t buffer, uint32_t* outStride);

/* getTransportSize(..., outNumFds, outNumInts)
 * Function descriptor: GRALLOC1_FUNCTION_GET_TRANSPORT_SIZE
 * This function is optional for all gralloc1 devices.
 *
 * Get the transport size of a buffer. An imported buffer handle is a raw
 * buffer handle with the process-local runtime data appended. This
 * function, for example, allows a caller to omit the process-local
 * runtime data at the tail when serializing the imported buffer handle.
 *
 * Note that a client might or might not omit the process-local runtime
 * data when sending an imported buffer handle. The mapper must support
 * both cases on the receiving end.
 *
 * Parameters:
 *   outNumFds - the number of file descriptors needed for transport
 *   outNumInts - the number of integers needed for transport
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to retrieve the numFds
 *       and numInts; see note [1] in this section's header for more information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_TRANSPORT_SIZE)(
        gralloc1_device_t* device, buffer_handle_t buffer, uint32_t *outNumFds,
        uint32_t *outNumInts);

typedef struct gralloc1_buffer_descriptor_info {
    uint32_t width;
    uint32_t height;
    uint32_t layerCount;
    int32_t /*android_pixel_format_t*/ format;
    uint64_t producerUsage;
    uint64_t consumerUsage;
} gralloc1_buffer_descriptor_info_t;

/* validateBufferSize(..., )
 * Function descriptor: GRALLOC1_FUNCTION_VALIDATE_BUFFER_SIZE
 * This function is optional for all gralloc1 devices.
 *
 * Validate that the buffer can be safely accessed by a caller who assumes
 * the specified descriptorInfo and stride. This must at least validate
 * that the buffer size is large enough. Validating the buffer against
 * individual buffer attributes is optional.
 *
 * Parameters:
 *   descriptor - specifies the attributes of the buffer
 *   stride - the buffer stride returned by IAllocator::allocate
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_BAD_VALUE - when buffer cannot be safely accessed
 *   GRALLOC1_ERROR_UNSUPPORTED - the device is unable to validate the buffer
 *       size; see note [1] in this section's header for more information
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_VALIDATE_BUFFER_SIZE)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        const gralloc1_buffer_descriptor_info_t* descriptorInfo,
        uint32_t stride);

/*
 * Buffer management functions
 */

/* allocate(..., numDescriptors, descriptors, outBuffers)
 * Function descriptor: GRALLOC1_FUNCTION_ALLOCATE
 * Must be provided by all gralloc1 devices
 *
 * Attempts to allocate a number of buffers sharing a backing store.
 *
 * Each buffer will correspond to one of the descriptors passed into the
 * function. If the device is unable to share the backing store between the
 * buffers, it should attempt to allocate the buffers with different backing
 * stores and return GRALLOC1_ERROR_NOT_SHARED if it is successful.
 *
 * If this call is successful, the client is responsible for freeing the
 * buffer_handle_t using release() when it is finished with the buffer. It is
 * not necessary to call retain() on the returned buffers, as they must have a
 * reference added by the device before returning.
 *
 * If GRALLOC1_CAPABILITY_TEST_ALLOCATE is supported by this device, outBuffers
 * may be NULL. In this case, the device must not attempt to allocate any
 * buffers, but instead must return either GRALLOC1_ERROR_NONE if such an
 * allocation is possible (ignoring potential resource contention which might
 * lead to a GRALLOC1_ERROR_NO_RESOURCES error), GRALLOC1_ERROR_NOT_SHARED if
 * the buffers can be allocated, but cannot share a backing store, or
 * GRALLOC1_ERROR_UNSUPPORTED if one or more of the descriptors can never be
 * allocated by the device.
 *
 * Parameters:
 *   numDescriptors - the number of buffer descriptors, which must also be equal
 *       to the size of the outBuffers array
 *   descriptors - the buffer descriptors to attempt to allocate
 *   outBuffers - the allocated buffers; must be non-NULL unless the device
 *       supports GRALLOC1_CAPABILITY_TEST_ALLOCATE (see above), and must not be
 *       modified by the device if allocation is unsuccessful
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_DESCRIPTOR - one of the descriptors does not refer to a
 *      valid buffer descriptor
 *   GRALLOC1_ERROR_NOT_SHARED - allocation was successful, but required more
 *       than one backing store to satisfy all of the buffer descriptors
 *   GRALLOC1_ERROR_NO_RESOURCES - allocation failed because one or more of the
 *       backing stores could not be created at this time (but this allocation
 *       might succeed at a future time)
 *   GRALLOC1_ERROR_UNSUPPORTED - one or more of the descriptors can never be
 *       satisfied by the device
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_ALLOCATE)(
        gralloc1_device_t* device, uint32_t numDescriptors,
        const gralloc1_buffer_descriptor_t* descriptors,
        buffer_handle_t* outBuffers);

/* importBuffer(..., rawHandle, outBuffer);
 * Function descriptor: GRALLOC1_FUNCTION_IMPORT_BUFFER
 * This function is optional for all gralloc1 devices.
 * When supported, GRALLOC1_CAPABILITY_RELEASE_IMPLY_DELETE must also be
 * supported.
 *
 * Explictly imports a buffer into a proccess.
 *
 * This function can be called in place of retain when a raw buffer handle is
 * received by a remote process. Import producess a import handle that can
 * be used to access the underlying graphic buffer. The new import handle has a
 * ref count of 1.
 *
 * This function must at least validate the raw handle before creating the
 * imported handle. It must also support importing the same raw handle
 * multiple times to create multiple imported handles. The imported handle
 * must be considered valid everywhere in the process.
 *
 * Parameters:
 *   rawHandle - the raw buffer handle to import
 *   outBuffer - a handle to the newly imported buffer
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_NO_RESOURCES - it is not possible to add a import to this
 *       buffer at this time
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_IMPORT_BUFFER)(
        gralloc1_device_t* device, const buffer_handle_t rawHandle,
        buffer_handle_t* outBuffer);

/* retain(..., buffer)
 * Function descriptor: GRALLOC1_FUNCTION_RETAIN
 * Must be provided by all gralloc1 devices
 *
 * Adds a reference to the given buffer.
 *
 * This function must be called when a buffer_handle_t is received from a remote
 * process to prevent the buffer's data from being freed when the remote process
 * releases the buffer. It may also be called to increase the reference count if
 * two components in the same process want to interact with the buffer
 * independently.
 *
 * Parameters:
 *   buffer - the buffer to which a reference should be added
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_NO_RESOURCES - it is not possible to add a reference to this
 *       buffer at this time
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_RETAIN)(
        gralloc1_device_t* device, buffer_handle_t buffer);

/* release(..., buffer)
 * Function descriptor: GRALLOC1_FUNCTION_RELEASE
 * Must be provided by all gralloc1 devices
 *
 * Removes a reference from the given buffer.
 *
 * If no references remain, the buffer should be freed. When the last buffer
 * referring to a particular backing store is freed, that backing store should
 * also be freed.
 *
 * When GRALLOC1_CAPABILITY_RELEASE_IMPLY_DELETE is supported,
 * native_handle_close and native_handle_delete must always be called by the
 * implementation whenever the last reference is removed.  Otherwise, a call
 * to release() will be followed by native_handle_close and native_handle_delete
 * by the caller when the buffer is not allocated locally through allocate().
 *
 * Parameters:
 *   buffer - the buffer from which a reference should be removed
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_RELEASE)(
        gralloc1_device_t* device, buffer_handle_t buffer);

/*
 * Buffer access functions
 *
 * All of these functions take as their first parameter a device pointer, so
 * this parameter is omitted from the described parameter lists.
 */

typedef struct gralloc1_rect {
    int32_t left;
    int32_t top;
    int32_t width;
    int32_t height;
} gralloc1_rect_t;

/* getNumFlexPlanes(..., buffer, outNumPlanes)
 * Function descriptor: GRALLOC1_FUNCTION_GET_NUM_FLEX_PLANES
 * Must be provided by all gralloc1 devices
 *
 * Returns the number of flex layout planes which are needed to represent the
 * given buffer. This may be used to efficiently allocate only as many plane
 * structures as necessary before calling into lockFlex.
 *
 * If the given buffer cannot be locked as a flex format, this function may
 * return GRALLOC1_ERROR_UNSUPPORTED (as lockFlex would).
 *
 * Parameters:
 *   buffer - the buffers for which the number of planes should be queried
 *   outNumPlanes - the number of flex planes required to describe the given
 *       buffer; must be non-NULL
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_UNSUPPORTED - the buffer's format cannot be represented in a
 *       flex layout
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_GET_NUM_FLEX_PLANES)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        uint32_t* outNumPlanes);

/* lock(..., buffer, producerUsage, consumerUsage, accessRegion, outData,
 *     acquireFence)
 * Function descriptor: GRALLOC1_FUNCTION_LOCK
 * Must be provided by all gralloc1 devices
 *
 * Locks the given buffer for the specified CPU usage.
 *
 * Exactly one of producerUsage and consumerUsage must be *_USAGE_NONE. The
 * usage which is not *_USAGE_NONE must be one of the *_USAGE_CPU_* values, as
 * applicable. Locking a buffer for a non-CPU usage is not supported.
 *
 * Locking the same buffer simultaneously from multiple threads is permitted,
 * but if any of the threads attempt to lock the buffer for writing, the
 * behavior is undefined, except that it must not cause process termination or
 * block the client indefinitely. Leaving the buffer content in an indeterminate
 * state or returning an error are both acceptable.
 *
 * The client must not modify the content of the buffer outside of accessRegion,
 * and the device need not guarantee that content outside of accessRegion is
 * valid for reading. The result of reading or writing outside of accessRegion
 * is undefined, except that it must not cause process termination.
 *
 * outData must be a non-NULL pointer, the contents of which which will be
 * filled with a pointer to the locked buffer memory. This address will
 * represent the top-left corner of the entire buffer, even if accessRegion does
 * not begin at the top-left corner.
 *
 * acquireFence is a file descriptor referring to a acquire sync fence object,
 * which will be signaled when it is safe for the device to access the contents
 * of the buffer (prior to locking). If it is already safe to access the buffer
 * contents, -1 may be passed instead.
 *
 * Parameters:
 *   buffer - the buffer to lock
 *   producerUsage - the producer usage flags to request; either this or
 *       consumerUsage must be GRALLOC1_*_USAGE_NONE, and the other must be a
 *       CPU usage
 *   consumerUsage - the consumer usage flags to request; either this or
 *       producerUsage must be GRALLOC1_*_USAGE_NONE, and the other must be a
 *       CPU usage
 *   accessRegion - the portion of the buffer that the client intends to access;
 *       must be non-NULL
 *   outData - will be filled with a CPU-accessible pointer to the buffer data;
 *       must be non-NULL
 *   acquireFence - a sync fence file descriptor as described above
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_BAD_VALUE - neither or both of producerUsage and
 *       consumerUsage were GRALLOC1_*_USAGE_NONE, or the usage which was not
 *       *_USAGE_NONE was not a CPU usage
 *   GRALLOC1_ERROR_NO_RESOURCES - the buffer cannot be locked at this time, but
 *       locking may succeed at a future time
 *   GRALLOC1_ERROR_UNSUPPORTED - the buffer cannot be locked with the given
 *       usage, and any future attempts at locking will also fail
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_LOCK)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        uint64_t /*gralloc1_producer_usage_t*/ producerUsage,
        uint64_t /*gralloc1_consumer_usage_t*/ consumerUsage,
        const gralloc1_rect_t* accessRegion, void** outData,
        int32_t acquireFence);

/* lockFlex(..., buffer, producerUsage, consumerUsage, accessRegion,
 *     outFlexLayout, outAcquireFence)
 * Function descriptor: GRALLOC1_FUNCTION_LOCK_FLEX
 * Must be provided by all gralloc1 devices
 *
 * This is largely the same as lock(), except that instead of returning a
 * pointer directly to the buffer data, it returns an android_flex_layout
 * struct describing how to access the data planes.
 *
 * This function must work on buffers with HAL_PIXEL_FORMAT_YCbCr_*_888 if
 * supported by the device, as well as with any other formats requested by
 * multimedia codecs when they are configured with a flexible-YUV-compatible
 * color format.
 *
 * This function may also be called on buffers of other formats, including
 * non-YUV formats, but if the buffer format is not compatible with a flexible
 * representation, it may return GRALLOC1_ERROR_UNSUPPORTED.
 *
 * Parameters:
 *   buffer - the buffer to lock
 *   producerUsage - the producer usage flags to request; either this or
 *       consumerUsage must be GRALLOC1_*_USAGE_NONE, and the other must be a
 *       CPU usage
 *   consumerUsage - the consumer usage flags to request; either this or
 *       producerUsage must be GRALLOC1_*_USAGE_NONE, and the other must be a
 *       CPU usage
 *   accessRegion - the portion of the buffer that the client intends to access;
 *      must be non-NULL
 *   outFlexLayout - will be filled with the description of the planes in the
 *       buffer
 *   acquireFence - a sync fence file descriptor as described in lock()
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 *   GRALLOC1_ERROR_BAD_VALUE - neither or both of producerUsage and
 *       consumerUsage were *_USAGE_NONE, or the usage which was not
 *       *_USAGE_NONE was not a CPU usage
 *   GRALLOC1_ERROR_NO_RESOURCES - the buffer cannot be locked at this time, but
 *       locking may succeed at a future time
 *   GRALLOC1_ERROR_UNSUPPORTED - the buffer cannot be locked with the given
 *       usage, and any future attempts at locking will also fail
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_LOCK_FLEX)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        uint64_t /*gralloc1_producer_usage_t*/ producerUsage,
        uint64_t /*gralloc1_consumer_usage_t*/ consumerUsage,
        const gralloc1_rect_t* accessRegion,
        struct android_flex_layout* outFlexLayout, int32_t acquireFence);

/* unlock(..., buffer, releaseFence)
 * Function descriptor: GRALLOC1_FUNCTION_UNLOCK
 * Must be provided by all gralloc1 devices
 *
 * This function indicates to the device that the client will be done with the
 * buffer when releaseFence signals.
 *
 * outReleaseFence will be filled with a file descriptor referring to a release
 * sync fence object, which will be signaled when it is safe to access the
 * contents of the buffer (after the buffer has been unlocked). If it is already
 * safe to access the buffer contents, then -1 may be returned instead.
 *
 * This function is used to unlock both buffers locked by lock() and those
 * locked by lockFlex().
 *
 * Parameters:
 *   buffer - the buffer to unlock
 *   outReleaseFence - a sync fence file descriptor as described above
 *
 * Returns GRALLOC1_ERROR_NONE or one of the following errors:
 *   GRALLOC1_ERROR_BAD_HANDLE - the buffer handle is invalid
 */
typedef int32_t /*gralloc1_error_t*/ (*GRALLOC1_PFN_UNLOCK)(
        gralloc1_device_t* device, buffer_handle_t buffer,
        int32_t* outReleaseFence);

__END_DECLS

#endif
