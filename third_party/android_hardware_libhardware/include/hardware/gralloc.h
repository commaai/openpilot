/*
 * Copyright (C) 2008 The Android Open Source Project
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


#ifndef ANDROID_GRALLOC_INTERFACE_H
#define ANDROID_GRALLOC_INTERFACE_H

#include <system/window.h>
#include <system/graphics.h>
#include <hardware/hardware.h>

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <cutils/native_handle.h>

#include <hardware/hardware.h>
#include <hardware/fb.h>

__BEGIN_DECLS

/**
 * Module versioning information for the Gralloc hardware module, based on
 * gralloc_module_t.common.module_api_version.
 *
 * Version History:
 *
 * GRALLOC_MODULE_API_VERSION_0_1:
 * Initial Gralloc hardware module API.
 *
 * GRALLOC_MODULE_API_VERSION_0_2:
 * Add support for flexible YCbCr format with (*lock_ycbcr)() method.
 *
 * GRALLOC_MODULE_API_VERSION_0_3:
 * Add support for fence passing to/from lock/unlock.
 */

#define GRALLOC_MODULE_API_VERSION_0_1  HARDWARE_MODULE_API_VERSION(0, 1)
#define GRALLOC_MODULE_API_VERSION_0_2  HARDWARE_MODULE_API_VERSION(0, 2)
#define GRALLOC_MODULE_API_VERSION_0_3  HARDWARE_MODULE_API_VERSION(0, 3)

#define GRALLOC_DEVICE_API_VERSION_0_1  HARDWARE_DEVICE_API_VERSION(0, 1)

/**
 * The id of this module
 */
#define GRALLOC_HARDWARE_MODULE_ID "gralloc"

/**
 * Name of the graphics device to open
 */

#define GRALLOC_HARDWARE_GPU0 "gpu0"

enum {
    /* buffer is never read in software */
    GRALLOC_USAGE_SW_READ_NEVER         = 0x00000000,
    /* buffer is rarely read in software */
    GRALLOC_USAGE_SW_READ_RARELY        = 0x00000002,
    /* buffer is often read in software */
    GRALLOC_USAGE_SW_READ_OFTEN         = 0x00000003,
    /* mask for the software read values */
    GRALLOC_USAGE_SW_READ_MASK          = 0x0000000F,

    /* buffer is never written in software */
    GRALLOC_USAGE_SW_WRITE_NEVER        = 0x00000000,
    /* buffer is rarely written in software */
    GRALLOC_USAGE_SW_WRITE_RARELY       = 0x00000020,
    /* buffer is often written in software */
    GRALLOC_USAGE_SW_WRITE_OFTEN        = 0x00000030,
    /* mask for the software write values */
    GRALLOC_USAGE_SW_WRITE_MASK         = 0x000000F0,

    /* buffer will be used as an OpenGL ES texture */
    GRALLOC_USAGE_HW_TEXTURE            = 0x00000100,
    /* buffer will be used as an OpenGL ES render target */
    GRALLOC_USAGE_HW_RENDER             = 0x00000200,
    /* buffer will be used by the 2D hardware blitter */
    GRALLOC_USAGE_HW_2D                 = 0x00000400,
    /* buffer will be used by the HWComposer HAL module */
    GRALLOC_USAGE_HW_COMPOSER           = 0x00000800,
    /* buffer will be used with the framebuffer device */
    GRALLOC_USAGE_HW_FB                 = 0x00001000,

    /* buffer should be displayed full-screen on an external display when
     * possible */
    GRALLOC_USAGE_EXTERNAL_DISP         = 0x00002000,

    /* Must have a hardware-protected path to external display sink for
     * this buffer.  If a hardware-protected path is not available, then
     * either don't composite only this buffer (preferred) to the
     * external sink, or (less desirable) do not route the entire
     * composition to the external sink.  */
    GRALLOC_USAGE_PROTECTED             = 0x00004000,

    /* buffer may be used as a cursor */
    GRALLOC_USAGE_CURSOR                = 0x00008000,

    /* buffer will be used with the HW video encoder */
    GRALLOC_USAGE_HW_VIDEO_ENCODER      = 0x00010000,
    /* buffer will be written by the HW camera pipeline */
    GRALLOC_USAGE_HW_CAMERA_WRITE       = 0x00020000,
    /* buffer will be read by the HW camera pipeline */
    GRALLOC_USAGE_HW_CAMERA_READ        = 0x00040000,
    /* buffer will be used as part of zero-shutter-lag queue */
    GRALLOC_USAGE_HW_CAMERA_ZSL         = 0x00060000,
    /* mask for the camera access values */
    GRALLOC_USAGE_HW_CAMERA_MASK        = 0x00060000,
    /* mask for the software usage bit-mask */
    GRALLOC_USAGE_HW_MASK               = 0x00071F00,

    /* buffer will be used as a RenderScript Allocation */
    GRALLOC_USAGE_RENDERSCRIPT          = 0x00100000,

    /* Set by the consumer to indicate to the producer that they may attach a
     * buffer that they did not detach from the BufferQueue. Will be filtered
     * out by GRALLOC_USAGE_ALLOC_MASK, so gralloc modules will not need to
     * handle this flag. */
    GRALLOC_USAGE_FOREIGN_BUFFERS       = 0x00200000,

    /* Mask of all flags which could be passed to a gralloc module for buffer
     * allocation. Any flags not in this mask do not need to be handled by
     * gralloc modules. */
    GRALLOC_USAGE_ALLOC_MASK            = ~(GRALLOC_USAGE_FOREIGN_BUFFERS),

    /* implementation-specific private usage flags */
    GRALLOC_USAGE_PRIVATE_0             = 0x10000000,
    GRALLOC_USAGE_PRIVATE_1             = 0x20000000,
    GRALLOC_USAGE_PRIVATE_2             = 0x40000000,
    GRALLOC_USAGE_PRIVATE_3             = 0x80000000,
    GRALLOC_USAGE_PRIVATE_MASK          = 0xF0000000,

#ifdef EXYNOS4_ENHANCEMENTS
    /* SAMSUNG */
    GRALLOC_USAGE_PRIVATE_NONECACHE     = 0x00800000,

    GRALLOC_USAGE_HW_FIMC1              = 0x01000000,
    GRALLOC_USAGE_HW_ION                = 0x02000000,
    GRALLOC_USAGE_YUV_ADDR              = 0x04000000,
    GRALLOC_USAGE_CAMERA                = 0x08000000,

    /* SEC Private usage , for Overlay path at HWC */
    GRALLOC_USAGE_HWC_HWOVERLAY         = 0x20000000,
#endif
};

/*****************************************************************************/

/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
typedef struct gralloc_module_t {
    struct hw_module_t common;
    
    /*
     * (*registerBuffer)() must be called before a buffer_handle_t that has not
     * been created with (*alloc_device_t::alloc)() can be used.
     * 
     * This is intended to be used with buffer_handle_t's that have been
     * received in this process through IPC.
     * 
     * This function checks that the handle is indeed a valid one and prepares
     * it for use with (*lock)() and (*unlock)().
     * 
     * It is not necessary to call (*registerBuffer)() on a handle created 
     * with (*alloc_device_t::alloc)().
     * 
     * returns an error if this buffer_handle_t is not valid.
     */
    int (*registerBuffer)(struct gralloc_module_t const* module,
            buffer_handle_t handle);

    /*
     * (*unregisterBuffer)() is called once this handle is no longer needed in
     * this process. After this call, it is an error to call (*lock)(),
     * (*unlock)(), or (*registerBuffer)().
     * 
     * This function doesn't close or free the handle itself; this is done
     * by other means, usually through libcutils's native_handle_close() and
     * native_handle_free(). 
     * 
     * It is an error to call (*unregisterBuffer)() on a buffer that wasn't
     * explicitly registered first.
     */
    int (*unregisterBuffer)(struct gralloc_module_t const* module,
            buffer_handle_t handle);
    
    /*
     * The (*lock)() method is called before a buffer is accessed for the 
     * specified usage. This call may block, for instance if the h/w needs
     * to finish rendering or if CPU caches need to be synchronized.
     * 
     * The caller promises to modify only pixels in the area specified 
     * by (l,t,w,h).
     * 
     * The content of the buffer outside of the specified area is NOT modified
     * by this call.
     *
     * If usage specifies GRALLOC_USAGE_SW_*, vaddr is filled with the address
     * of the buffer in virtual memory.
     *
     * Note calling (*lock)() on HAL_PIXEL_FORMAT_YCbCr_*_888 buffers will fail
     * and return -EINVAL.  These buffers must be locked with (*lock_ycbcr)()
     * instead.
     *
     * THREADING CONSIDERATIONS:
     *
     * It is legal for several different threads to lock a buffer from 
     * read access, none of the threads are blocked.
     * 
     * However, locking a buffer simultaneously for write or read/write is
     * undefined, but:
     * - shall not result in termination of the process
     * - shall not block the caller
     * It is acceptable to return an error or to leave the buffer's content
     * into an indeterminate state.
     *
     * If the buffer was created with a usage mask incompatible with the
     * requested usage flags here, -EINVAL is returned. 
     * 
     */
    
    int (*lock)(struct gralloc_module_t const* module,
            buffer_handle_t handle, int usage,
            int l, int t, int w, int h,
            void** vaddr);

    
    /*
     * The (*unlock)() method must be called after all changes to the buffer
     * are completed.
     */
    
    int (*unlock)(struct gralloc_module_t const* module,
            buffer_handle_t handle);

#ifdef EXYNOS4_ENHANCEMENTS
    int (*getphys) (struct gralloc_module_t const* module,
            buffer_handle_t handle, void** paddr);
#endif

    /* reserved for future use */
    int (*perform)(struct gralloc_module_t const* module,
            int operation, ... );

    /*
     * The (*lock_ycbcr)() method is like the (*lock)() method, with the
     * difference that it fills a struct ycbcr with a description of the buffer
     * layout, and zeroes out the reserved fields.
     *
     * If the buffer format is not compatible with a flexible YUV format (e.g.
     * the buffer layout cannot be represented with the ycbcr struct), it
     * will return -EINVAL.
     *
     * This method must work on buffers with HAL_PIXEL_FORMAT_YCbCr_*_888
     * if supported by the device, as well as with any other format that is
     * requested by the multimedia codecs when they are configured with a
     * flexible-YUV-compatible color-format with android native buffers.
     *
     * Note that this method may also be called on buffers of other formats,
     * including non-YUV formats.
     *
     * Added in GRALLOC_MODULE_API_VERSION_0_2.
     */

    int (*lock_ycbcr)(struct gralloc_module_t const* module,
            buffer_handle_t handle, int usage,
            int l, int t, int w, int h,
            struct android_ycbcr *ycbcr);

    /*
     * The (*lockAsync)() method is like the (*lock)() method except
     * that the buffer's sync fence object is passed into the lock
     * call instead of requiring the caller to wait for completion.
     *
     * The gralloc implementation takes ownership of the fenceFd and
     * is responsible for closing it when no longer needed.
     *
     * Added in GRALLOC_MODULE_API_VERSION_0_3.
     */
    int (*lockAsync)(struct gralloc_module_t const* module,
            buffer_handle_t handle, int usage,
            int l, int t, int w, int h,
            void** vaddr, int fenceFd);

    /*
     * The (*unlockAsync)() method is like the (*unlock)() method
     * except that a buffer sync fence object is returned from the
     * lock call, representing the completion of any pending work
     * performed by the gralloc implementation.
     *
     * The caller takes ownership of the fenceFd and is responsible
     * for closing it when no longer needed.
     *
     * Added in GRALLOC_MODULE_API_VERSION_0_3.
     */
    int (*unlockAsync)(struct gralloc_module_t const* module,
            buffer_handle_t handle, int* fenceFd);

    /*
     * The (*lockAsync_ycbcr)() method is like the (*lock_ycbcr)()
     * method except that the buffer's sync fence object is passed
     * into the lock call instead of requiring the caller to wait for
     * completion.
     *
     * The gralloc implementation takes ownership of the fenceFd and
     * is responsible for closing it when no longer needed.
     *
     * Added in GRALLOC_MODULE_API_VERSION_0_3.
     */
    int (*lockAsync_ycbcr)(struct gralloc_module_t const* module,
            buffer_handle_t handle, int usage,
            int l, int t, int w, int h,
            struct android_ycbcr *ycbcr, int fenceFd);

    /* reserved for future use */
    void* reserved_proc[3];
} gralloc_module_t;

/*****************************************************************************/

/**
 * Every device data structure must begin with hw_device_t
 * followed by module specific public methods and attributes.
 */

typedef struct alloc_device_t {
    struct hw_device_t common;

    /* 
     * (*alloc)() Allocates a buffer in graphic memory with the requested
     * parameters and returns a buffer_handle_t and the stride in pixels to
     * allow the implementation to satisfy hardware constraints on the width
     * of a pixmap (eg: it may have to be multiple of 8 pixels). 
     * The CALLER TAKES OWNERSHIP of the buffer_handle_t.
     *
     * If format is HAL_PIXEL_FORMAT_YCbCr_420_888, the returned stride must be
     * 0, since the actual strides are available from the android_ycbcr
     * structure.
     * 
     * Returns 0 on success or -errno on error.
     */
    
    int (*alloc)(struct alloc_device_t* dev,
            int w, int h, int format, int usage,
            buffer_handle_t* handle, int* stride);

    /*
     * (*free)() Frees a previously allocated buffer. 
     * Behavior is undefined if the buffer is still mapped in any process,
     * but shall not result in termination of the program or security breaches
     * (allowing a process to get access to another process' buffers).
     * THIS FUNCTION TAKES OWNERSHIP of the buffer_handle_t which becomes
     * invalid after the call. 
     * 
     * Returns 0 on success or -errno on error.
     */
    int (*free)(struct alloc_device_t* dev,
            buffer_handle_t handle);

    /* This hook is OPTIONAL.
     *
     * If non NULL it will be caused by SurfaceFlinger on dumpsys
     */
    void (*dump)(struct alloc_device_t *dev, char *buff, int buff_len);

    void* reserved_proc[7];
} alloc_device_t;


/** convenience API for opening and closing a supported device */

static inline int gralloc_open(const struct hw_module_t* module, 
        struct alloc_device_t** device) {
    return module->methods->open(module, 
            GRALLOC_HARDWARE_GPU0, (struct hw_device_t**)device);
}

static inline int gralloc_close(struct alloc_device_t* device) {
    return device->common.close(&device->common);
}

__END_DECLS

#endif  // ANDROID_GRALLOC_INTERFACE_H
