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

#ifndef ANDROID_INCLUDE_HARDWARE_HWCOMPOSER_H
#define ANDROID_INCLUDE_HARDWARE_HWCOMPOSER_H

#include <stdint.h>
#include <sys/cdefs.h>

#include <hardware/gralloc.h>
#include <hardware/hardware.h>
#include <cutils/native_handle.h>

#include <hardware/hwcomposer_defs.h>

__BEGIN_DECLS

/*****************************************************************************/

/* for compatibility */
#define HWC_MODULE_API_VERSION      HWC_MODULE_API_VERSION_0_1
#define HWC_DEVICE_API_VERSION      HWC_DEVICE_API_VERSION_0_1
#define HWC_API_VERSION             HWC_DEVICE_API_VERSION

/*****************************************************************************/

/**
 * The id of this module
 */
#define HWC_HARDWARE_MODULE_ID "hwcomposer"

/**
 * Name of the sensors device to open
 */
#define HWC_HARDWARE_COMPOSER   "composer"

typedef struct hwc_rect {
    int left;
    int top;
    int right;
    int bottom;
} hwc_rect_t;

typedef struct hwc_frect {
    float left;
    float top;
    float right;
    float bottom;
} hwc_frect_t;

typedef struct hwc_region {
    size_t numRects;
    hwc_rect_t const* rects;
} hwc_region_t;

typedef struct hwc_color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} hwc_color_t;

typedef struct hwc_layer_1 {
    /*
     * compositionType is used to specify this layer's type and is set by either
     * the hardware composer implementation, or by the caller (see below).
     *
     *  This field is always reset to HWC_BACKGROUND or HWC_FRAMEBUFFER
     *  before (*prepare)() is called when the HWC_GEOMETRY_CHANGED flag is
     *  also set, otherwise, this field is preserved between (*prepare)()
     *  calls.
     *
     * HWC_BACKGROUND
     *   Always set by the caller before calling (*prepare)(), this value
     *   indicates this is a special "background" layer. The only valid field
     *   is backgroundColor.
     *   The HWC can toggle this value to HWC_FRAMEBUFFER to indicate it CANNOT
     *   handle the background color.
     *
     *
     * HWC_FRAMEBUFFER_TARGET
     *   Always set by the caller before calling (*prepare)(), this value
     *   indicates this layer is the framebuffer surface used as the target of
     *   OpenGL ES composition. If the HWC sets all other layers to HWC_OVERLAY
     *   or HWC_BACKGROUND, then no OpenGL ES composition will be done, and
     *   this layer should be ignored during set().
     *
     *   This flag (and the framebuffer surface layer) will only be used if the
     *   HWC version is HWC_DEVICE_API_VERSION_1_1 or higher. In older versions,
     *   the OpenGL ES target surface is communicated by the (dpy, sur) fields
     *   in hwc_compositor_device_1_t.
     *
     *   This value cannot be set by the HWC implementation.
     *
     *
     * HWC_FRAMEBUFFER
     *   Set by the caller before calling (*prepare)() ONLY when the
     *   HWC_GEOMETRY_CHANGED flag is also set.
     *
     *   Set by the HWC implementation during (*prepare)(), this indicates
     *   that the layer will be drawn into the framebuffer using OpenGL ES.
     *   The HWC can toggle this value to HWC_OVERLAY to indicate it will
     *   handle the layer.
     *
     *
     * HWC_OVERLAY
     *   Set by the HWC implementation during (*prepare)(), this indicates
     *   that the layer will be handled by the HWC (ie: it must not be
     *   composited with OpenGL ES).
     *
     *
     * HWC_SIDEBAND
     *   Set by the caller before calling (*prepare)(), this value indicates
     *   the contents of this layer come from a sideband video stream.
     *
     *   The h/w composer is responsible for receiving new image buffers from
     *   the stream at the appropriate time (e.g. synchronized to a separate
     *   audio stream), compositing them with the current contents of other
     *   layers, and displaying the resulting image. This happens
     *   independently of the normal prepare/set cycle. The prepare/set calls
     *   only happen when other layers change, or when properties of the
     *   sideband layer such as position or size change.
     *
     *   If the h/w composer can't handle the layer as a sideband stream for
     *   some reason (e.g. unsupported scaling/blending/rotation, or too many
     *   sideband layers) it can set compositionType to HWC_FRAMEBUFFER in
     *   (*prepare)(). However, doing so will result in the layer being shown
     *   as a solid color since the platform is not currently able to composite
     *   sideband layers with the GPU. This may be improved in future
     *   versions of the platform.
     *
     *
     * HWC_CURSOR_OVERLAY
     *   Set by the HWC implementation during (*prepare)(), this value
     *   indicates the layer's composition will now be handled by the HWC.
     *   Additionally, the client can now asynchronously update the on-screen
     *   position of this layer using the setCursorPositionAsync() api.
     */
    int32_t compositionType;

    /*
     * hints is bit mask set by the HWC implementation during (*prepare)().
     * It is preserved between (*prepare)() calls, unless the
     * HWC_GEOMETRY_CHANGED flag is set, in which case it is reset to 0.
     *
     * see hwc_layer_t::hints
     */
    uint32_t hints;

    /* see hwc_layer_t::flags */
    uint32_t flags;

    union {
        /* color of the background.  hwc_color_t.a is ignored */
        hwc_color_t backgroundColor;

        struct {
            union {
                /* When compositionType is HWC_FRAMEBUFFER, HWC_OVERLAY,
                 * HWC_FRAMEBUFFER_TARGET, this is the handle of the buffer to
                 * compose. This handle is guaranteed to have been allocated
                 * from gralloc using the GRALLOC_USAGE_HW_COMPOSER usage flag.
                 * If the layer's handle is unchanged across two consecutive
                 * prepare calls and the HWC_GEOMETRY_CHANGED flag is not set
                 * for the second call then the HWComposer implementation may
                 * assume that the contents of the buffer have not changed. */
                buffer_handle_t handle;

                /* When compositionType is HWC_SIDEBAND, this is the handle
                 * of the sideband video stream to compose. */
                const native_handle_t* sidebandStream;
            };

            /* transformation to apply to the buffer during composition */
            uint32_t transform;

            /* blending to apply during composition */
            int32_t blending;

            /* area of the source to consider, the origin is the top-left corner of
             * the buffer. As of HWC_DEVICE_API_VERSION_1_3, sourceRect uses floats.
             * If the h/w can't support a non-integer source crop rectangle, it should
             * punt to OpenGL ES composition.
             */
            union {
                // crop rectangle in integer (pre HWC_DEVICE_API_VERSION_1_3)
                hwc_rect_t sourceCropi;
                hwc_rect_t sourceCrop; // just for source compatibility
                // crop rectangle in floats (as of HWC_DEVICE_API_VERSION_1_3)
                hwc_frect_t sourceCropf;
            };

            /* where to composite the sourceCrop onto the display. The sourceCrop
             * is scaled using linear filtering to the displayFrame. The origin is the
             * top-left corner of the screen.
             */
            hwc_rect_t displayFrame;

            /* visible region in screen space. The origin is the
             * top-left corner of the screen.
             * The visible region INCLUDES areas overlapped by a translucent layer.
             */
            hwc_region_t visibleRegionScreen;

            /* Sync fence object that will be signaled when the buffer's
             * contents are available. May be -1 if the contents are already
             * available. This field is only valid during set(), and should be
             * ignored during prepare(). The set() call must not wait for the
             * fence to be signaled before returning, but the HWC must wait for
             * all buffers to be signaled before reading from them.
             *
             * HWC_FRAMEBUFFER layers will never have an acquire fence, since
             * reads from them are complete before the framebuffer is ready for
             * display.
             *
             * HWC_SIDEBAND layers will never have an acquire fence, since
             * synchronization is handled through implementation-defined
             * sideband mechanisms.
             *
             * The HWC takes ownership of the acquireFenceFd and is responsible
             * for closing it when no longer needed.
             */
            int acquireFenceFd;

            /* During set() the HWC must set this field to a file descriptor for
             * a sync fence object that will signal after the HWC has finished
             * reading from the buffer. The field is ignored by prepare(). Each
             * layer should have a unique file descriptor, even if more than one
             * refer to the same underlying fence object; this allows each to be
             * closed independently.
             *
             * If buffer reads can complete at significantly different times,
             * then using independent fences is preferred. For example, if the
             * HWC handles some layers with a blit engine and others with
             * overlays, then the blit layers can be reused immediately after
             * the blit completes, but the overlay layers can't be reused until
             * a subsequent frame has been displayed.
             *
             * Since HWC doesn't read from HWC_FRAMEBUFFER layers, it shouldn't
             * produce a release fence for them. The releaseFenceFd will be -1
             * for these layers when set() is called.
             *
             * Since HWC_SIDEBAND buffers don't pass through the HWC client,
             * the HWC shouldn't produce a release fence for them. The
             * releaseFenceFd will be -1 for these layers when set() is called.
             *
             * The HWC client taks ownership of the releaseFenceFd and is
             * responsible for closing it when no longer needed.
             */
            int releaseFenceFd;

            /*
             * Availability: HWC_DEVICE_API_VERSION_1_2
             *
             * Alpha value applied to the whole layer. The effective
             * value of each pixel is computed as:
             *
             *   if (blending == HWC_BLENDING_PREMULT)
             *      pixel.rgb = pixel.rgb * planeAlpha / 255
             *   pixel.a = pixel.a * planeAlpha / 255
             *
             * Then blending proceeds as usual according to the "blending"
             * field above.
             *
             * NOTE: planeAlpha applies to YUV layers as well:
             *
             *   pixel.rgb = yuv_to_rgb(pixel.yuv)
             *   if (blending == HWC_BLENDING_PREMULT)
             *      pixel.rgb = pixel.rgb * planeAlpha / 255
             *   pixel.a = planeAlpha
             *
             *
             * IMPLEMENTATION NOTE:
             *
             * If the source image doesn't have an alpha channel, then
             * the h/w can use the HWC_BLENDING_COVERAGE equations instead of
             * HWC_BLENDING_PREMULT and simply set the alpha channel to
             * planeAlpha.
             *
             * e.g.:
             *
             *   if (blending == HWC_BLENDING_PREMULT)
             *      blending = HWC_BLENDING_COVERAGE;
             *   pixel.a = planeAlpha;
             *
             */
            uint8_t planeAlpha;

            /* Pad to 32 bits */
            uint8_t _pad[3];

            /*
             * Availability: HWC_DEVICE_API_VERSION_1_5
             *
             * This defines the region of the source buffer that has been
             * modified since the last frame.
             *
             * If surfaceDamage.numRects > 0, then it may be assumed that any
             * portion of the source buffer not covered by one of the rects has
             * not been modified this frame. If surfaceDamage.numRects == 0,
             * then the whole source buffer must be treated as if it had been
             * modified.
             *
             * If the layer's contents are not modified relative to the prior
             * prepare/set cycle, surfaceDamage will contain exactly one empty
             * rect ([0, 0, 0, 0]).
             *
             * The damage rects are relative to the pre-transformed buffer, and
             * their origin is the top-left corner.
             */
            hwc_region_t surfaceDamage;
        };
    };

#ifdef __LP64__
    /*
     * For 64-bit mode, this struct is 120 bytes (and 8-byte aligned), and needs
     * to be padded as such to maintain binary compatibility.
     */
    uint8_t reserved[120 - 112];
#else
    /*
     * For 32-bit mode, this struct is 96 bytes, and needs to be padded as such
     * to maintain binary compatibility.
     */
    uint8_t reserved[96 - 84];
#endif

} hwc_layer_1_t;

/* This represents a display, typically an EGLDisplay object */
typedef void* hwc_display_t;

/* This represents a surface, typically an EGLSurface object  */
typedef void* hwc_surface_t;

/*
 * hwc_display_contents_1_t::flags values
 */
enum {
    /*
     * HWC_GEOMETRY_CHANGED is set by SurfaceFlinger to indicate that the list
     * passed to (*prepare)() has changed by more than just the buffer handles
     * and acquire fences.
     */
    HWC_GEOMETRY_CHANGED = 0x00000001,
};

/*
 * Description of the contents to output on a display.
 *
 * This is the top-level structure passed to the prepare and set calls to
 * negotiate and commit the composition of a display image.
 */
typedef struct hwc_display_contents_1 {
    /* File descriptor referring to a Sync HAL fence object which will signal
     * when this composition is retired. For a physical display, a composition
     * is retired when it has been replaced on-screen by a subsequent set. For
     * a virtual display, the composition is retired when the writes to
     * outputBuffer are complete and can be read. The fence object is created
     * and returned by the set call; this field will be -1 on entry to prepare
     * and set. SurfaceFlinger will close the returned file descriptor.
     */
    int retireFenceFd;

    union {
        /* Fields only relevant for HWC_DEVICE_VERSION_1_0. */
        struct {
            /* (dpy, sur) is the target of SurfaceFlinger's OpenGL ES
             * composition for HWC_DEVICE_VERSION_1_0. They aren't relevant to
             * prepare. The set call should commit this surface atomically to
             * the display along with any overlay layers.
             */
            hwc_display_t dpy;
            hwc_surface_t sur;
        };

        /* These fields are used for virtual displays when the h/w composer
         * version is at least HWC_DEVICE_VERSION_1_3. */
        struct {
            /* outbuf is the buffer that receives the composed image for
             * virtual displays. Writes to the outbuf must wait until
             * outbufAcquireFenceFd signals. A fence that will signal when
             * writes to outbuf are complete should be returned in
             * retireFenceFd.
             *
             * This field is set before prepare(), so properties of the buffer
             * can be used to decide which layers can be handled by h/w
             * composer.
             *
             * If prepare() sets all layers to FRAMEBUFFER, then GLES
             * composition will happen directly to the output buffer. In this
             * case, both outbuf and the FRAMEBUFFER_TARGET layer's buffer will
             * be the same, and set() has no work to do besides managing fences.
             *
             * If the TARGET_FORCE_HWC_FOR_VIRTUAL_DISPLAYS board config
             * variable is defined (not the default), then this behavior is
             * changed: if all layers are marked for FRAMEBUFFER, GLES
             * composition will take place to a scratch framebuffer, and
             * h/w composer must copy it to the output buffer. This allows the
             * h/w composer to do format conversion if there are cases where
             * that is more desirable than doing it in the GLES driver or at the
             * virtual display consumer.
             *
             * If some or all layers are marked OVERLAY, then the framebuffer
             * and output buffer will be different. As with physical displays,
             * the framebuffer handle will not change between frames if all
             * layers are marked for OVERLAY.
             */
            buffer_handle_t outbuf;

            /* File descriptor for a fence that will signal when outbuf is
             * ready to be written. The h/w composer is responsible for closing
             * this when no longer needed.
             *
             * Will be -1 whenever outbuf is NULL, or when the outbuf can be
             * written immediately.
             */
            int outbufAcquireFenceFd;
        };
    };

    /* List of layers that will be composed on the display. The buffer handles
     * in the list will be unique. If numHwLayers is 0, all composition will be
     * performed by SurfaceFlinger.
     */
    uint32_t flags;
    size_t numHwLayers;
    hwc_layer_1_t hwLayers[0];

} hwc_display_contents_1_t;

/* see hwc_composer_device::registerProcs()
 * All of the callbacks are required and non-NULL unless otherwise noted.
 */
typedef struct hwc_procs {
    /*
     * (*invalidate)() triggers a screen refresh, in particular prepare and set
     * will be called shortly after this call is made. Note that there is
     * NO GUARANTEE that the screen refresh will happen after invalidate()
     * returns (in particular, it could happen before).
     * invalidate() is GUARANTEED TO NOT CALL BACK into the h/w composer HAL and
     * it is safe to call invalidate() from any of hwc_composer_device
     * hooks, unless noted otherwise.
     */
    void (*invalidate)(const struct hwc_procs* procs);

    /*
     * (*vsync)() is called by the h/w composer HAL when a vsync event is
     * received and HWC_EVENT_VSYNC is enabled on a display
     * (see: hwc_event_control).
     *
     * the "disp" parameter indicates which display the vsync event is for.
     * the "timestamp" parameter is the system monotonic clock timestamp in
     *   nanosecond of when the vsync event happened.
     *
     * vsync() is GUARANTEED TO NOT CALL BACK into the h/w composer HAL.
     *
     * It is expected that vsync() is called from a thread of at least
     * HAL_PRIORITY_URGENT_DISPLAY with as little latency as possible,
     * typically less than 0.5 ms.
     *
     * It is a (silent) error to have HWC_EVENT_VSYNC enabled when calling
     * hwc_composer_device.set(..., 0, 0, 0) (screen off). The implementation
     * can either stop or continue to process VSYNC events, but must not
     * crash or cause other problems.
     */
    void (*vsync)(const struct hwc_procs* procs, int disp, int64_t timestamp);

    /*
     * (*hotplug)() is called by the h/w composer HAL when a display is
     * connected or disconnected. The PRIMARY display is always connected and
     * the hotplug callback should not be called for it.
     *
     * The disp parameter indicates which display type this event is for.
     * The connected parameter indicates whether the display has just been
     *   connected (1) or disconnected (0).
     *
     * The hotplug() callback may call back into the h/w composer on the same
     * thread to query refresh rate and dpi for the display. Additionally,
     * other threads may be calling into the h/w composer while the callback
     * is in progress.
     *
     * The h/w composer must serialize calls to the hotplug callback; only
     * one thread may call it at a time.
     *
     * This callback will be NULL if the h/w composer is using
     * HWC_DEVICE_API_VERSION_1_0.
     */
    void (*hotplug)(const struct hwc_procs* procs, int disp, int connected);

} hwc_procs_t;


/*****************************************************************************/

typedef struct hwc_module {
    /**
     * Common methods of the hardware composer module.  This *must* be the first member of
     * hwc_module as users of this structure will cast a hw_module_t to
     * hwc_module pointer in contexts where it's known the hw_module_t references a
     * hwc_module.
     */
    struct hw_module_t common;
} hwc_module_t;

typedef struct hwc_composer_device_1 {
    /**
     * Common methods of the hardware composer device.  This *must* be the first member of
     * hwc_composer_device_1 as users of this structure will cast a hw_device_t to
     * hwc_composer_device_1 pointer in contexts where it's known the hw_device_t references a
     * hwc_composer_device_1.
     */
    struct hw_device_t common;

    /*
     * (*prepare)() is called for each frame before composition and is used by
     * SurfaceFlinger to determine what composition steps the HWC can handle.
     *
     * (*prepare)() can be called more than once, the last call prevails.
     *
     * The HWC responds by setting the compositionType field in each layer to
     * either HWC_FRAMEBUFFER, HWC_OVERLAY, or HWC_CURSOR_OVERLAY. For the
     * HWC_FRAMEBUFFER type, composition for the layer is handled by
     * SurfaceFlinger with OpenGL ES. For the latter two overlay types,
     * the HWC will have to handle the layer's composition. compositionType
     * and hints are preserved between (*prepare)() calles unless the
     * HWC_GEOMETRY_CHANGED flag is set.
     *
     * (*prepare)() is called with HWC_GEOMETRY_CHANGED to indicate that the
     * list's geometry has changed, that is, when more than just the buffer's
     * handles have been updated. Typically this happens (but is not limited to)
     * when a window is added, removed, resized or moved. In this case
     * compositionType and hints are reset to their default value.
     *
     * For HWC 1.0, numDisplays will always be one, and displays[0] will be
     * non-NULL.
     *
     * For HWC 1.1, numDisplays will always be HWC_NUM_PHYSICAL_DISPLAY_TYPES.
     * Entries for unsupported or disabled/disconnected display types will be
     * NULL.
     *
     * In HWC 1.3, numDisplays may be up to HWC_NUM_DISPLAY_TYPES. The extra
     * entries correspond to enabled virtual displays, and will be non-NULL.
     *
     * returns: 0 on success. An negative error code on error. If an error is
     * returned, SurfaceFlinger will assume that none of the layer will be
     * handled by the HWC.
     */
    int (*prepare)(struct hwc_composer_device_1 *dev,
                    size_t numDisplays, hwc_display_contents_1_t** displays);

    /*
     * (*set)() is used in place of eglSwapBuffers(), and assumes the same
     * functionality, except it also commits the work list atomically with
     * the actual eglSwapBuffers().
     *
     * The layer lists are guaranteed to be the same as the ones returned from
     * the last call to (*prepare)().
     *
     * When this call returns the caller assumes that the displays will be
     * updated in the near future with the content of their work lists, without
     * artifacts during the transition from the previous frame.
     *
     * A display with zero layers indicates that the entire composition has
     * been handled by SurfaceFlinger with OpenGL ES. In this case, (*set)()
     * behaves just like eglSwapBuffers().
     *
     * For HWC 1.0, numDisplays will always be one, and displays[0] will be
     * non-NULL.
     *
     * For HWC 1.1, numDisplays will always be HWC_NUM_PHYSICAL_DISPLAY_TYPES.
     * Entries for unsupported or disabled/disconnected display types will be
     * NULL.
     *
     * In HWC 1.3, numDisplays may be up to HWC_NUM_DISPLAY_TYPES. The extra
     * entries correspond to enabled virtual displays, and will be non-NULL.
     *
     * IMPORTANT NOTE: There is an implicit layer containing opaque black
     * pixels behind all the layers in the list. It is the responsibility of
     * the hwcomposer module to make sure black pixels are output (or blended
     * from).
     *
     * IMPORTANT NOTE: In the event of an error this call *MUST* still cause
     * any fences returned in the previous call to set to eventually become
     * signaled.  The caller may have already issued wait commands on these
     * fences, and having set return without causing those fences to signal
     * will likely result in a deadlock.
     *
     * returns: 0 on success. A negative error code on error:
     *    HWC_EGL_ERROR: eglGetError() will provide the proper error code (only
     *        allowed prior to HWComposer 1.1)
     *    Another code for non EGL errors.
     */
    int (*set)(struct hwc_composer_device_1 *dev,
                size_t numDisplays, hwc_display_contents_1_t** displays);

    /*
     * eventControl(..., event, enabled)
     * Enables or disables h/w composer events for a display.
     *
     * eventControl can be called from any thread and takes effect
     * immediately.
     *
     *  Supported events are:
     *      HWC_EVENT_VSYNC
     *
     * returns -EINVAL if the "event" parameter is not one of the value above
     * or if the "enabled" parameter is not 0 or 1.
     */
    int (*eventControl)(struct hwc_composer_device_1* dev, int disp,
            int event, int enabled);

    union {
        /*
         * For HWC 1.3 and earlier, the blank() interface is used.
         *
         * blank(..., blank)
         * Blanks or unblanks a display's screen.
         *
         * Turns the screen off when blank is nonzero, on when blank is zero.
         * Multiple sequential calls with the same blank value must be
         * supported.
         * The screen state transition must be be complete when the function
         * returns.
         *
         * returns 0 on success, negative on error.
         */
        int (*blank)(struct hwc_composer_device_1* dev, int disp, int blank);

        /*
         * For HWC 1.4 and above, setPowerMode() will be used in place of
         * blank().
         *
         * setPowerMode(..., mode)
         * Sets the display screen's power state.
         *
         * Refer to the documentation of the HWC_POWER_MODE_* constants
         * for information about each power mode.
         *
         * The functionality is similar to the blank() command in previous
         * versions of HWC, but with support for more power states.
         *
         * The display driver is expected to retain and restore the low power
         * state of the display while entering and exiting from suspend.
         *
         * Multiple sequential calls with the same mode value must be supported.
         *
         * The screen state transition must be be complete when the function
         * returns.
         *
         * returns 0 on success, negative on error.
         */
        int (*setPowerMode)(struct hwc_composer_device_1* dev, int disp,
                int mode);
    };

    /*
     * Used to retrieve information about the h/w composer
     *
     * Returns 0 on success or -errno on error.
     */
    int (*query)(struct hwc_composer_device_1* dev, int what, int* value);

    /*
     * (*registerProcs)() registers callbacks that the h/w composer HAL can
     * later use. It will be called immediately after the composer device is
     * opened with non-NULL procs. It is FORBIDDEN to call any of the callbacks
     * from within registerProcs(). registerProcs() must save the hwc_procs_t
     * pointer which is needed when calling a registered callback.
     */
    void (*registerProcs)(struct hwc_composer_device_1* dev,
            hwc_procs_t const* procs);

    /*
     * This field is OPTIONAL and can be NULL.
     *
     * If non NULL it will be called by SurfaceFlinger on dumpsys
     */
    void (*dump)(struct hwc_composer_device_1* dev, char *buff, int buff_len);

    /*
     * (*getDisplayConfigs)() returns handles for the configurations available
     * on the connected display. These handles must remain valid as long as the
     * display is connected.
     *
     * Configuration handles are written to configs. The number of entries
     * allocated by the caller is passed in *numConfigs; getDisplayConfigs must
     * not try to write more than this number of config handles. On return, the
     * total number of configurations available for the display is returned in
     * *numConfigs. If *numConfigs is zero on entry, then configs may be NULL.
     *
     * Hardware composers implementing HWC_DEVICE_API_VERSION_1_3 or prior
     * shall choose one configuration to activate and report it as the first
     * entry in the returned list. Reporting the inactive configurations is not
     * required.
     *
     * HWC_DEVICE_API_VERSION_1_4 and later provide configuration management
     * through SurfaceFlinger, and hardware composers implementing these APIs
     * must also provide getActiveConfig and setActiveConfig. Hardware composers
     * implementing these API versions may choose not to activate any
     * configuration, leaving configuration selection to higher levels of the
     * framework.
     *
     * Returns 0 on success or a negative error code on error. If disp is a
     * hotpluggable display type and no display is connected, an error shall be
     * returned.
     *
     * This field is REQUIRED for HWC_DEVICE_API_VERSION_1_1 and later.
     * It shall be NULL for previous versions.
     */
    int (*getDisplayConfigs)(struct hwc_composer_device_1* dev, int disp,
            uint32_t* configs, size_t* numConfigs);

    /*
     * (*getDisplayAttributes)() returns attributes for a specific config of a
     * connected display. The config parameter is one of the config handles
     * returned by getDisplayConfigs.
     *
     * The list of attributes to return is provided in the attributes
     * parameter, terminated by HWC_DISPLAY_NO_ATTRIBUTE. The value for each
     * requested attribute is written in order to the values array. The
     * HWC_DISPLAY_NO_ATTRIBUTE attribute does not have a value, so the values
     * array will have one less value than the attributes array.
     *
     * This field is REQUIRED for HWC_DEVICE_API_VERSION_1_1 and later.
     * It shall be NULL for previous versions.
     *
     * If disp is a hotpluggable display type and no display is connected,
     * or if config is not a valid configuration for the display, a negative
     * error code shall be returned.
     */
    int (*getDisplayAttributes)(struct hwc_composer_device_1* dev, int disp,
            uint32_t config, const uint32_t* attributes, int32_t* values);

    /*
     * (*getActiveConfig)() returns the index of the configuration that is
     * currently active on the connected display. The index is relative to
     * the list of configuration handles returned by getDisplayConfigs. If there
     * is no active configuration, -1 shall be returned.
     *
     * Returns the configuration index on success or -1 on error.
     *
     * This field is REQUIRED for HWC_DEVICE_API_VERSION_1_4 and later.
     * It shall be NULL for previous versions.
     */
    int (*getActiveConfig)(struct hwc_composer_device_1* dev, int disp);

    /*
     * (*setActiveConfig)() instructs the hardware composer to switch to the
     * display configuration at the given index in the list of configuration
     * handles returned by getDisplayConfigs.
     *
     * If this function returns without error, any subsequent calls to
     * getActiveConfig shall return the index set by this function until one
     * of the following occurs:
     *   1) Another successful call of this function
     *   2) The display is disconnected
     *
     * Returns 0 on success or a negative error code on error. If disp is a
     * hotpluggable display type and no display is connected, or if index is
     * outside of the range of hardware configurations returned by
     * getDisplayConfigs, an error shall be returned.
     *
     * This field is REQUIRED for HWC_DEVICE_API_VERSION_1_4 and later.
     * It shall be NULL for previous versions.
     */
    int (*setActiveConfig)(struct hwc_composer_device_1* dev, int disp,
            int index);
    /*
     * Asynchronously update the location of the cursor layer.
     *
     * Within the standard prepare()/set() composition loop, the client
     * (surfaceflinger) can request that a given layer uses dedicated cursor
     * composition hardware by specifiying the HWC_IS_CURSOR_LAYER flag. Only
     * one layer per display can have this flag set. If the layer is suitable
     * for the platform's cursor hardware, hwcomposer will return from prepare()
     * a composition type of HWC_CURSOR_OVERLAY for that layer. This indicates
     * not only that the client is not responsible for compositing that layer,
     * but also that the client can continue to update the position of that layer
     * after a call to set(). This can reduce the visible latency of mouse
     * movement to visible, on-screen cursor updates. Calls to
     * setCursorPositionAsync() may be made from a different thread doing the
     * prepare()/set() composition loop, but care must be taken to not interleave
     * calls of setCursorPositionAsync() between calls of set()/prepare().
     *
     * Notes:
     * - Only one layer per display can be specified as a cursor layer with
     *   HWC_IS_CURSOR_LAYER.
     * - hwcomposer will only return one layer per display as HWC_CURSOR_OVERLAY
     * - This returns 0 on success or -errno on error.
     * - This field is optional for HWC_DEVICE_API_VERSION_1_4 and later. It
     *   should be null for previous versions.
     */
    int (*setCursorPositionAsync)(struct hwc_composer_device_1 *dev, int disp, int x_pos, int y_pos);

    /*
     * Reserved for future use. Must be NULL.
     */
    void* reserved_proc[1];

} hwc_composer_device_1_t;

/** convenience API for opening and closing a device */

static inline int hwc_open_1(const struct hw_module_t* module,
        hwc_composer_device_1_t** device) {
    return module->methods->open(module,
            HWC_HARDWARE_COMPOSER, (struct hw_device_t**)device);
}

static inline int hwc_close_1(hwc_composer_device_1_t* device) {
    return device->common.close(&device->common);
}

/*****************************************************************************/

__END_DECLS

#endif /* ANDROID_INCLUDE_HARDWARE_HWCOMPOSER_H */
