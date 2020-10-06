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

#ifndef ANDROID_INCLUDE_HARDWARE_HWCOMPOSER_DEFS_H
#define ANDROID_INCLUDE_HARDWARE_HWCOMPOSER_DEFS_H

#include <stdint.h>
#include <sys/cdefs.h>

#include <hardware/gralloc.h>
#include <hardware/hardware.h>
#include <cutils/native_handle.h>

__BEGIN_DECLS

/*****************************************************************************/

#define HWC_HEADER_VERSION          1

#define HWC_MODULE_API_VERSION_0_1  HARDWARE_MODULE_API_VERSION(0, 1)

#define HWC_DEVICE_API_VERSION_1_0  HARDWARE_DEVICE_API_VERSION_2(1, 0, HWC_HEADER_VERSION)
#define HWC_DEVICE_API_VERSION_1_1  HARDWARE_DEVICE_API_VERSION_2(1, 1, HWC_HEADER_VERSION)
#define HWC_DEVICE_API_VERSION_1_2  HARDWARE_DEVICE_API_VERSION_2(1, 2, HWC_HEADER_VERSION)
#define HWC_DEVICE_API_VERSION_1_3  HARDWARE_DEVICE_API_VERSION_2(1, 3, HWC_HEADER_VERSION)
#define HWC_DEVICE_API_VERSION_1_4  HARDWARE_DEVICE_API_VERSION_2(1, 4, HWC_HEADER_VERSION)
#define HWC_DEVICE_API_VERSION_1_5  HARDWARE_DEVICE_API_VERSION_2(1, 5, HWC_HEADER_VERSION)

enum {
    /* hwc_composer_device_t::set failed in EGL */
    HWC_EGL_ERROR = -1
};

/*
 * hwc_layer_t::hints values
 * Hints are set by the HAL and read by SurfaceFlinger
 */
enum {
    /*
     * HWC can set the HWC_HINT_TRIPLE_BUFFER hint to indicate to SurfaceFlinger
     * that it should triple buffer this layer. Typically HWC does this when
     * the layer will be unavailable for use for an extended period of time,
     * e.g. if the display will be fetching data directly from the layer and
     * the layer can not be modified until after the next set().
     */
    HWC_HINT_TRIPLE_BUFFER  = 0x00000001,

    /*
     * HWC sets HWC_HINT_CLEAR_FB to tell SurfaceFlinger that it should clear the
     * framebuffer with transparent pixels where this layer would be.
     * SurfaceFlinger will only honor this flag when the layer has no blending
     *
     */
    HWC_HINT_CLEAR_FB       = 0x00000002
};

/*
 * hwc_layer_t::flags values
 * Flags are set by SurfaceFlinger and read by the HAL
 */
enum {
    /*
     * HWC_SKIP_LAYER is set by SurfaceFlnger to indicate that the HAL
     * shall not consider this layer for composition as it will be handled
     * by SurfaceFlinger (just as if compositionType was set to HWC_OVERLAY).
     */
    HWC_SKIP_LAYER = 0x00000001,

    /*
     * HWC_IS_CURSOR_LAYER is set by surfaceflinger to indicate that this
     * layer is being used as a cursor on this particular display, and that
     * surfaceflinger can potentially perform asynchronous position updates for
     * this layer. If a call to prepare() returns HWC_CURSOR_OVERLAY for the
     * composition type of this layer, then the hwcomposer will allow async
     * position updates to this layer via setCursorPositionAsync().
     */
    HWC_IS_CURSOR_LAYER = 0x00000002
};

/*
 * hwc_layer_t::compositionType values
 */
enum {
    /* this layer is to be drawn into the framebuffer by SurfaceFlinger */
    HWC_FRAMEBUFFER = 0,

    /* this layer will be handled in the HWC */
    HWC_OVERLAY = 1,

    /* this is the background layer. it's used to set the background color.
     * there is only a single background layer */
    HWC_BACKGROUND = 2,

    /* this layer holds the result of compositing the HWC_FRAMEBUFFER layers.
     * Added in HWC_DEVICE_API_VERSION_1_1. */
    HWC_FRAMEBUFFER_TARGET = 3,

    /* this layer's contents are taken from a sideband buffer stream.
     * Added in HWC_DEVICE_API_VERSION_1_4. */
    HWC_SIDEBAND = 4,

    /* this layer's composition will be handled by hwcomposer by dedicated
       cursor overlay hardware. hwcomposer will also all async position updates
       of this layer outside of the normal prepare()/set() loop. Added in
       HWC_DEVICE_API_VERSION_1_4. */
    HWC_CURSOR_OVERLAY =  5
 };
/*
 * hwc_layer_t::blending values
 */
enum {
    /* no blending */
    HWC_BLENDING_NONE     = 0x0100,

    /* ONE / ONE_MINUS_SRC_ALPHA */
    HWC_BLENDING_PREMULT  = 0x0105,

    /* SRC_ALPHA / ONE_MINUS_SRC_ALPHA */
    HWC_BLENDING_COVERAGE = 0x0405
};

/*
 * hwc_layer_t::transform values
 */
enum {
    /* flip source image horizontally */
    HWC_TRANSFORM_FLIP_H = HAL_TRANSFORM_FLIP_H,
    /* flip source image vertically */
    HWC_TRANSFORM_FLIP_V = HAL_TRANSFORM_FLIP_V,
    /* rotate source image 90 degrees clock-wise */
    HWC_TRANSFORM_ROT_90 = HAL_TRANSFORM_ROT_90,
    /* rotate source image 180 degrees */
    HWC_TRANSFORM_ROT_180 = HAL_TRANSFORM_ROT_180,
    /* rotate source image 270 degrees clock-wise */
    HWC_TRANSFORM_ROT_270 = HAL_TRANSFORM_ROT_270,
};

/* attributes queriable with query() */
enum {
    /*
     * Must return 1 if the background layer is supported, 0 otherwise.
     */
    HWC_BACKGROUND_LAYER_SUPPORTED      = 0,

    /*
     * Returns the vsync period in nanoseconds.
     *
     * This query is not used for HWC_DEVICE_API_VERSION_1_1 and later.
     * Instead, the per-display attribute HWC_DISPLAY_VSYNC_PERIOD is used.
     */
    HWC_VSYNC_PERIOD                    = 1,

    /*
     * Availability: HWC_DEVICE_API_VERSION_1_1
     * Returns a mask of supported display types.
     */
    HWC_DISPLAY_TYPES_SUPPORTED         = 2,
};

/* display attributes returned by getDisplayAttributes() */
enum {
    /* Indicates the end of an attribute list */
    HWC_DISPLAY_NO_ATTRIBUTE                = 0,

    /* The vsync period in nanoseconds */
    HWC_DISPLAY_VSYNC_PERIOD                = 1,

    /* The number of pixels in the horizontal and vertical directions. */
    HWC_DISPLAY_WIDTH                       = 2,
    HWC_DISPLAY_HEIGHT                      = 3,

    /* The number of pixels per thousand inches of this configuration.
     *
     * Scaling DPI by 1000 allows it to be stored in an int without losing
     * too much precision.
     *
     * If the DPI for a configuration is unavailable or the HWC implementation
     * considers it unreliable, it should set these attributes to zero.
     */
    HWC_DISPLAY_DPI_X                       = 4,
    HWC_DISPLAY_DPI_Y                       = 5,

    /* Indicates which of the vendor-defined color transforms is provided by
     * this configuration. */
    HWC_DISPLAY_COLOR_TRANSFORM             = 6,
};

/* Allowed events for hwc_methods::eventControl() */
enum {
    HWC_EVENT_VSYNC     = 0
};

/* Display types and associated mask bits. */
enum {
    HWC_DISPLAY_PRIMARY     = 0,
    HWC_DISPLAY_EXTERNAL    = 1,    // HDMI, DP, etc.
#ifdef QTI_BSP
    HWC_DISPLAY_TERTIARY    = 2,
    HWC_DISPLAY_VIRTUAL     = 3,

    HWC_NUM_PHYSICAL_DISPLAY_TYPES = 3,
    HWC_NUM_DISPLAY_TYPES          = 4,
#else
    HWC_DISPLAY_VIRTUAL     = 2,

    HWC_NUM_PHYSICAL_DISPLAY_TYPES = 2,
    HWC_NUM_DISPLAY_TYPES          = 3,
#endif
};

enum {
    HWC_DISPLAY_PRIMARY_BIT     = 1 << HWC_DISPLAY_PRIMARY,
    HWC_DISPLAY_EXTERNAL_BIT    = 1 << HWC_DISPLAY_EXTERNAL,
#ifdef QTI_BSP
    HWC_DISPLAY_TERTIARY_BIT    = 1 << HWC_DISPLAY_TERTIARY,
#endif
    HWC_DISPLAY_VIRTUAL_BIT     = 1 << HWC_DISPLAY_VIRTUAL,
};

/* Display power modes */
enum {
    /* The display is turned off (blanked). */
    HWC_POWER_MODE_OFF      = 0,
    /* The display is turned on and configured in a low power state
     * that is suitable for presenting ambient information to the user,
     * possibly with lower fidelity than normal but greater efficiency. */
    HWC_POWER_MODE_DOZE     = 1,
    /* The display is turned on normally. */
    HWC_POWER_MODE_NORMAL   = 2,
    /* The display is configured as in HWC_POWER_MODE_DOZE but may
     * stop applying frame buffer updates from the graphics subsystem.
     * This power mode is effectively a hint from the doze dream to
     * tell the hardware that it is done drawing to the display for the
     * time being and that the display should remain on in a low power
     * state and continue showing its current contents indefinitely
     * until the mode changes.
     *
     * This mode may also be used as a signal to enable hardware-based doze
     * functionality.  In this case, the doze dream is effectively
     * indicating that the hardware is free to take over the display
     * and manage it autonomously to implement low power always-on display
     * functionality. */
    HWC_POWER_MODE_DOZE_SUSPEND  = 3,
};

/*****************************************************************************/

__END_DECLS

#endif /* ANDROID_INCLUDE_HARDWARE_HWCOMPOSER_DEFS_H */
