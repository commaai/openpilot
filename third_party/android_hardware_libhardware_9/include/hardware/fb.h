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


#ifndef ANDROID_FB_INTERFACE_H
#define ANDROID_FB_INTERFACE_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <cutils/native_handle.h>

#include <hardware/hardware.h>

__BEGIN_DECLS

#define GRALLOC_HARDWARE_FB0 "fb0"

/*****************************************************************************/


/*****************************************************************************/

typedef struct framebuffer_device_t {
    /**
     * Common methods of the framebuffer device.  This *must* be the first member of
     * framebuffer_device_t as users of this structure will cast a hw_device_t to
     * framebuffer_device_t pointer in contexts where it's known the hw_device_t references a
     * framebuffer_device_t.
     */
    struct hw_device_t common;

    /* flags describing some attributes of the framebuffer */
    const uint32_t  flags;

    /* dimensions of the framebuffer in pixels */
    const uint32_t  width;
    const uint32_t  height;

    /* frambuffer stride in pixels */
    const int       stride;

    /* framebuffer pixel format */
    const int       format;

    /* resolution of the framebuffer's display panel in pixel per inch*/
    const float     xdpi;
    const float     ydpi;

    /* framebuffer's display panel refresh rate in frames per second */
    const float     fps;

    /* min swap interval supported by this framebuffer */
    const int       minSwapInterval;

    /* max swap interval supported by this framebuffer */
    const int       maxSwapInterval;

    /* Number of framebuffers supported*/
    const int       numFramebuffers;

    int reserved[7];

    /*
     * requests a specific swap-interval (same definition than EGL)
     *
     * Returns 0 on success or -errno on error.
     */
    int (*setSwapInterval)(struct framebuffer_device_t* window,
            int interval);

    /*
     * This hook is OPTIONAL.
     *
     * It is non NULL If the framebuffer driver supports "update-on-demand"
     * and the given rectangle is the area of the screen that gets
     * updated during (*post)().
     *
     * This is useful on devices that are able to DMA only a portion of
     * the screen to the display panel, upon demand -- as opposed to
     * constantly refreshing the panel 60 times per second, for instance.
     *
     * Only the area defined by this rectangle is guaranteed to be valid, that
     * is, the driver is not allowed to post anything outside of this
     * rectangle.
     *
     * The rectangle evaluated during (*post)() and specifies which area
     * of the buffer passed in (*post)() shall to be posted.
     *
     * return -EINVAL if width or height <=0, or if left or top < 0
     */
    int (*setUpdateRect)(struct framebuffer_device_t* window,
            int left, int top, int width, int height);

    /*
     * Post <buffer> to the display (display it on the screen)
     * The buffer must have been allocated with the
     *   GRALLOC_USAGE_HW_FB usage flag.
     * buffer must be the same width and height as the display and must NOT
     * be locked.
     *
     * The buffer is shown during the next VSYNC.
     *
     * If the same buffer is posted again (possibly after some other buffer),
     * post() will block until the the first post is completed.
     *
     * Internally, post() is expected to lock the buffer so that a
     * subsequent call to gralloc_module_t::(*lock)() with USAGE_RENDER or
     * USAGE_*_WRITE will block until it is safe; that is typically once this
     * buffer is shown and another buffer has been posted.
     *
     * Returns 0 on success or -errno on error.
     */
    int (*post)(struct framebuffer_device_t* dev, buffer_handle_t buffer);


    /*
     * The (*compositionComplete)() method must be called after the
     * compositor has finished issuing GL commands for client buffers.
     */

    int (*compositionComplete)(struct framebuffer_device_t* dev);

    /*
     * This hook is OPTIONAL.
     *
     * If non NULL it will be caused by SurfaceFlinger on dumpsys
     */
    void (*dump)(struct framebuffer_device_t* dev, char *buff, int buff_len);

    /*
     * (*enableScreen)() is used to either blank (enable=0) or
     * unblank (enable=1) the screen this framebuffer is attached to.
     *
     * Returns 0 on success or -errno on error.
     */
    int (*enableScreen)(struct framebuffer_device_t* dev, int enable);

    void* reserved_proc[6];

} framebuffer_device_t;


/** convenience API for opening and closing a supported device */

static inline int framebuffer_open(const struct hw_module_t* module,
        struct framebuffer_device_t** device) {
    return module->methods->open(module,
            GRALLOC_HARDWARE_FB0, TO_HW_DEVICE_T_OPEN(device));
}

static inline int framebuffer_close(struct framebuffer_device_t* device) {
    return device->common.close(&device->common);
}


__END_DECLS

#endif  // ANDROID_FB_INTERFACE_H
