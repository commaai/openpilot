/*
 * Copyright (C) 2010 The Android Open Source Project
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

/**
 * @addtogroup NativeActivity Native Activity
 * @{
 */

/**
 * @file native_window.h
 */

#ifndef ANDROID_NATIVE_WINDOW_H
#define ANDROID_NATIVE_WINDOW_H

#include <android/rect.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pixel formats that a window can use.
 */
enum {
    /** Red: 8 bits, Green: 8 bits, Blue: 8 bits, Alpha: 8 bits. **/
    WINDOW_FORMAT_RGBA_8888          = 1,
    /** Red: 8 bits, Green: 8 bits, Blue: 8 bits, Unused: 8 bits. **/
    WINDOW_FORMAT_RGBX_8888          = 2,
    /** Red: 5 bits, Green: 6 bits, Blue: 5 bits. **/
    WINDOW_FORMAT_RGB_565            = 4,
};

struct ANativeWindow;
/**
 * {@link ANativeWindow} is opaque type that provides access to a native window.
 *
 * A pointer can be obtained using ANativeWindow_fromSurface().
 */
typedef struct ANativeWindow ANativeWindow;

/**
 * {@link ANativeWindow} is a struct that represents a windows buffer.
 *
 * A pointer can be obtained using ANativeWindow_lock().
 */
typedef struct ANativeWindow_Buffer {
    // The number of pixels that are show horizontally.
    int32_t width;

    // The number of pixels that are shown vertically.
    int32_t height;

    // The number of *pixels* that a line in the buffer takes in
    // memory.  This may be >= width.
    int32_t stride;

    // The format of the buffer.  One of WINDOW_FORMAT_*
    int32_t format;

    // The actual bits.
    void* bits;

    // Do not touch.
    uint32_t reserved[6];
} ANativeWindow_Buffer;

/**
 * Acquire a reference on the given ANativeWindow object.  This prevents the object
 * from being deleted until the reference is removed.
 */
void ANativeWindow_acquire(ANativeWindow* window);

/**
 * Remove a reference that was previously acquired with ANativeWindow_acquire().
 */
void ANativeWindow_release(ANativeWindow* window);

/**
 * Return the current width in pixels of the window surface.  Returns a
 * negative value on error.
 */
int32_t ANativeWindow_getWidth(ANativeWindow* window);

/**
 * Return the current height in pixels of the window surface.  Returns a
 * negative value on error.
 */
int32_t ANativeWindow_getHeight(ANativeWindow* window);

/**
 * Return the current pixel format of the window surface.  Returns a
 * negative value on error.
 */
int32_t ANativeWindow_getFormat(ANativeWindow* window);

/**
 * Change the format and size of the window buffers.
 *
 * The width and height control the number of pixels in the buffers, not the
 * dimensions of the window on screen.  If these are different than the
 * window's physical size, then it buffer will be scaled to match that size
 * when compositing it to the screen.
 *
 * For all of these parameters, if 0 is supplied then the window's base
 * value will come back in force.
 *
 * width and height must be either both zero or both non-zero.
 *
 */
int32_t ANativeWindow_setBuffersGeometry(ANativeWindow* window,
        int32_t width, int32_t height, int32_t format);

/**
 * Lock the window's next drawing surface for writing.
 * inOutDirtyBounds is used as an in/out parameter, upon entering the
 * function, it contains the dirty region, that is, the region the caller
 * intends to redraw. When the function returns, inOutDirtyBounds is updated
 * with the actual area the caller needs to redraw -- this region is often
 * extended by ANativeWindow_lock.
 */
int32_t ANativeWindow_lock(ANativeWindow* window, ANativeWindow_Buffer* outBuffer,
        ARect* inOutDirtyBounds);

/**
 * Unlock the window's drawing surface after previously locking it,
 * posting the new buffer to the display.
 */
int32_t ANativeWindow_unlockAndPost(ANativeWindow* window);

#ifdef __cplusplus
};
#endif

#endif // ANDROID_NATIVE_WINDOW_H

/** @} */
