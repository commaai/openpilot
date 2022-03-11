/*
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef ANDROID_VNDK_NATIVEWINDOW_ANATIVEWINDOW_H
#define ANDROID_VNDK_NATIVEWINDOW_ANATIVEWINDOW_H

#include <nativebase/nativebase.h>

// vndk is a superset of the NDK
#include <android/native_window.h>


__BEGIN_DECLS

/*
 * Convert this ANativeWindowBuffer into a AHardwareBuffer
 */
AHardwareBuffer* ANativeWindowBuffer_getHardwareBuffer(ANativeWindowBuffer* anwb);

/*****************************************************************************/

/*
 * Stores a value into one of the 4 available slots
 * Retrieve the value with ANativeWindow_OemStorageGet()
 *
 * slot: 0 to 3
 *
 * Returns 0 on success or -errno on error.
 */
int ANativeWindow_OemStorageSet(ANativeWindow* window, uint32_t slot, intptr_t value);


/*
 * Retrieves a value from one of the 4 available slots
 * By default the returned value is 0 if it wasn't set by ANativeWindow_OemStorageSet()
 *
 * slot: 0 to 3
 *
 * Returns 0 on success or -errno on error.
 */
int ANativeWindow_OemStorageGet(ANativeWindow* window, uint32_t slot, intptr_t* value);


/*
 * Set the swap interval for this surface.
 *
 * Returns 0 on success or -errno on error.
 */
int ANativeWindow_setSwapInterval(ANativeWindow* window, int interval);


/*
 * queries that can be used with ANativeWindow_query() and ANativeWindow_queryf()
 */
enum ANativeWindowQuery {
    /* The minimum number of buffers that must remain un-dequeued after a buffer
     * has been queued.  This value applies only if set_buffer_count was used to
     * override the number of buffers and if a buffer has since been queued.
     * Users of the set_buffer_count ANativeWindow method should query this
     * value before calling set_buffer_count.  If it is necessary to have N
     * buffers simultaneously dequeued as part of the steady-state operation,
     * and this query returns M then N+M buffers should be requested via
     * native_window_set_buffer_count.
     *
     * Note that this value does NOT apply until a single buffer has been
     * queued.  In particular this means that it is possible to:
     *
     * 1. Query M = min undequeued buffers
     * 2. Set the buffer count to N + M
     * 3. Dequeue all N + M buffers
     * 4. Cancel M buffers
     * 5. Queue, dequeue, queue, dequeue, ad infinitum
     */
    ANATIVEWINDOW_QUERY_MIN_UNDEQUEUED_BUFFERS = 3,

    /*
     * Default width of ANativeWindow buffers, these are the
     * dimensions of the window buffers irrespective of the
     * ANativeWindow_setBuffersDimensions() call and match the native window
     * size.
     */
    ANATIVEWINDOW_QUERY_DEFAULT_WIDTH = 6,
    ANATIVEWINDOW_QUERY_DEFAULT_HEIGHT = 7,

    /*
     * transformation that will most-likely be applied to buffers. This is only
     * a hint, the actual transformation applied might be different.
     *
     * INTENDED USE:
     *
     * The transform hint can be used by a producer, for instance the GLES
     * driver, to pre-rotate the rendering such that the final transformation
     * in the composer is identity. This can be very useful when used in
     * conjunction with the h/w composer HAL, in situations where it
     * cannot handle arbitrary rotations.
     *
     * 1. Before dequeuing a buffer, the GL driver (or any other ANW client)
     *    queries the ANW for NATIVE_WINDOW_TRANSFORM_HINT.
     *
     * 2. The GL driver overrides the width and height of the ANW to
     *    account for NATIVE_WINDOW_TRANSFORM_HINT. This is done by querying
     *    NATIVE_WINDOW_DEFAULT_{WIDTH | HEIGHT}, swapping the dimensions
     *    according to NATIVE_WINDOW_TRANSFORM_HINT and calling
     *    native_window_set_buffers_dimensions().
     *
     * 3. The GL driver dequeues a buffer of the new pre-rotated size.
     *
     * 4. The GL driver renders to the buffer such that the image is
     *    already transformed, that is applying NATIVE_WINDOW_TRANSFORM_HINT
     *    to the rendering.
     *
     * 5. The GL driver calls native_window_set_transform to apply
     *    inverse transformation to the buffer it just rendered.
     *    In order to do this, the GL driver needs
     *    to calculate the inverse of NATIVE_WINDOW_TRANSFORM_HINT, this is
     *    done easily:
     *
     *        int hintTransform, inverseTransform;
     *        query(..., NATIVE_WINDOW_TRANSFORM_HINT, &hintTransform);
     *        inverseTransform = hintTransform;
     *        if (hintTransform & HAL_TRANSFORM_ROT_90)
     *            inverseTransform ^= HAL_TRANSFORM_ROT_180;
     *
     *
     * 6. The GL driver queues the pre-transformed buffer.
     *
     * 7. The composer combines the buffer transform with the display
     *    transform.  If the buffer transform happens to cancel out the
     *    display transform then no rotation is needed.
     *
     */
    ANATIVEWINDOW_QUERY_TRANSFORM_HINT = 8,

    /*
     * Returns the age of the contents of the most recently dequeued buffer as
     * the number of frames that have elapsed since it was last queued. For
     * example, if the window is double-buffered, the age of any given buffer in
     * steady state will be 2. If the dequeued buffer has never been queued, its
     * age will be 0.
     */
    ANATIVEWINDOW_QUERY_BUFFER_AGE = 13,

    /* min swap interval supported by this compositor */
    ANATIVEWINDOW_QUERY_MIN_SWAP_INTERVAL = 0x10000,

    /* max swap interval supported by this compositor */
    ANATIVEWINDOW_QUERY_MAX_SWAP_INTERVAL = 0x10001,

    /* horizontal resolution in DPI. value is float, use queryf() */
    ANATIVEWINDOW_QUERY_XDPI = 0x10002,

    /* vertical resolution in DPI. value is float, use queryf() */
    ANATIVEWINDOW_QUERY_YDPI = 0x10003,
};

typedef enum ANativeWindowQuery ANativeWindowQuery;

/*
 * hook used to retrieve information about the native window.
 *
 * Returns 0 on success or -errno on error.
 */
int ANativeWindow_query(const ANativeWindow* window, ANativeWindowQuery query, int* value);
int ANativeWindow_queryf(const ANativeWindow* window, ANativeWindowQuery query, float* value);


/*
 * Hook called by EGL to acquire a buffer. This call may block if no
 * buffers are available.
 *
 * The window holds a reference to the buffer between dequeueBuffer and
 * either queueBuffer or cancelBuffer, so clients only need their own
 * reference if they might use the buffer after queueing or canceling it.
 * Holding a reference to a buffer after queueing or canceling it is only
 * allowed if a specific buffer count has been set.
 *
 * The libsync fence file descriptor returned in the int pointed to by the
 * fenceFd argument will refer to the fence that must signal before the
 * dequeued buffer may be written to.  A value of -1 indicates that the
 * caller may access the buffer immediately without waiting on a fence.  If
 * a valid file descriptor is returned (i.e. any value except -1) then the
 * caller is responsible for closing the file descriptor.
 *
 * Returns 0 on success or -errno on error.
 */
int ANativeWindow_dequeueBuffer(ANativeWindow* window, ANativeWindowBuffer** buffer, int* fenceFd);


/*
 * Hook called by EGL when modifications to the render buffer are done.
 * This unlocks and post the buffer.
 *
 * The window holds a reference to the buffer between dequeueBuffer and
 * either queueBuffer or cancelBuffer, so clients only need their own
 * reference if they might use the buffer after queueing or canceling it.
 * Holding a reference to a buffer after queueing or canceling it is only
 * allowed if a specific buffer count has been set.
 *
 * The fenceFd argument specifies a libsync fence file descriptor for a
 * fence that must signal before the buffer can be accessed.  If the buffer
 * can be accessed immediately then a value of -1 should be used.  The
 * caller must not use the file descriptor after it is passed to
 * queueBuffer, and the ANativeWindow implementation is responsible for
 * closing it.
 *
 * Returns 0 on success or -errno on error.
 */
int ANativeWindow_queueBuffer(ANativeWindow* window, ANativeWindowBuffer* buffer, int fenceFd);


/*
 * Hook used to cancel a buffer that has been dequeued.
 * No synchronization is performed between dequeue() and cancel(), so
 * either external synchronization is needed, or these functions must be
 * called from the same thread.
 *
 * The window holds a reference to the buffer between dequeueBuffer and
 * either queueBuffer or cancelBuffer, so clients only need their own
 * reference if they might use the buffer after queueing or canceling it.
 * Holding a reference to a buffer after queueing or canceling it is only
 * allowed if a specific buffer count has been set.
 *
 * The fenceFd argument specifies a libsync fence file decsriptor for a
 * fence that must signal before the buffer can be accessed.  If the buffer
 * can be accessed immediately then a value of -1 should be used.
 *
 * Note that if the client has not waited on the fence that was returned
 * from dequeueBuffer, that same fence should be passed to cancelBuffer to
 * ensure that future uses of the buffer are preceded by a wait on that
 * fence.  The caller must not use the file descriptor after it is passed
 * to cancelBuffer, and the ANativeWindow implementation is responsible for
 * closing it.
 *
 * Returns 0 on success or -errno on error.
 */
int ANativeWindow_cancelBuffer(ANativeWindow* window, ANativeWindowBuffer* buffer, int fenceFd);

/*
 *  Sets the intended usage flags for the next buffers.
 *
 *  usage: one of AHARDWAREBUFFER_USAGE_* constant
 *
 *  By default (if this function is never called), a usage of
 *      AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT
 *  is assumed.
 *
 *  Calling this function will usually cause following buffers to be
 *  reallocated.
 */
int ANativeWindow_setUsage(ANativeWindow* window, uint64_t usage);


/*
 * Sets the number of buffers associated with this native window.
 */
int ANativeWindow_setBufferCount(ANativeWindow* window, size_t bufferCount);


/*
 * All buffers dequeued after this call will have the dimensions specified.
 * In particular, all buffers will have a fixed-size, independent from the
 * native-window size. They will be scaled according to the scaling mode
 * (see native_window_set_scaling_mode) upon window composition.
 *
 * If w and h are 0, the normal behavior is restored. That is, dequeued buffers
 * following this call will be sized to match the window's size.
 *
 * Calling this function will reset the window crop to a NULL value, which
 * disables cropping of the buffers.
 */
int ANativeWindow_setBuffersDimensions(ANativeWindow* window, uint32_t w, uint32_t h);


/*
 * All buffers dequeued after this call will have the format specified.
 * format: one of AHARDWAREBUFFER_FORMAT_* constant
 *
 * If the specified format is 0, the default buffer format will be used.
 */
int ANativeWindow_setBuffersFormat(ANativeWindow* window, int format);


/*
 * All buffers queued after this call will be associated with the timestamp in nanosecond
 * parameter specified. If the timestamp is set to NATIVE_WINDOW_TIMESTAMP_AUTO
 * (the default), timestamps will be generated automatically when queueBuffer is
 * called. The timestamp is measured in nanoseconds, and is normally monotonically
 * increasing. The timestamp should be unaffected by time-of-day adjustments,
 * and for a camera should be strictly monotonic but for a media player may be
 * reset when the position is set.
 */
int ANativeWindow_setBuffersTimestamp(ANativeWindow* window, int64_t timestamp);


/*
 * Enable/disable shared buffer mode
 */
int ANativeWindow_setSharedBufferMode(ANativeWindow* window, bool sharedBufferMode);


/*
 * Enable/disable auto refresh when in shared buffer mode
 */
int ANativeWindow_setAutoRefresh(ANativeWindow* window, bool autoRefresh);


/*****************************************************************************/

__END_DECLS

#endif /* ANDROID_VNDK_NATIVEWINDOW_ANATIVEWINDOW_H */
