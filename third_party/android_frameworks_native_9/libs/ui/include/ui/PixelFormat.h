/*
 * Copyright (C) 2005 The Android Open Source Project
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

//

// Pixel formats used across the system.
// These formats might not all be supported by all renderers, for instance
// skia or SurfaceFlinger are not required to support all of these formats
// (either as source or destination)


#ifndef UI_PIXELFORMAT_H
#define UI_PIXELFORMAT_H

#include <hardware/hardware.h>

namespace android {

enum {
    //
    // these constants need to match those
    // in graphics/PixelFormat.java & pixelflinger/format.h
    //
    PIXEL_FORMAT_UNKNOWN    =   0,
    PIXEL_FORMAT_NONE       =   0,

    // logical pixel formats used by the SurfaceFlinger -----------------------
    PIXEL_FORMAT_CUSTOM         = -4,
        // Custom pixel-format described by a PixelFormatInfo structure

    PIXEL_FORMAT_TRANSLUCENT    = -3,
        // System chooses a format that supports translucency (many alpha bits)

    PIXEL_FORMAT_TRANSPARENT    = -2,
        // System chooses a format that supports transparency
        // (at least 1 alpha bit)

    PIXEL_FORMAT_OPAQUE         = -1,
        // System chooses an opaque format (no alpha bits required)

    // real pixel formats supported for rendering -----------------------------

    PIXEL_FORMAT_RGBA_8888    = HAL_PIXEL_FORMAT_RGBA_8888,    // 4x8-bit RGBA
    PIXEL_FORMAT_RGBX_8888    = HAL_PIXEL_FORMAT_RGBX_8888,    // 4x8-bit RGB0
    PIXEL_FORMAT_RGB_888      = HAL_PIXEL_FORMAT_RGB_888,      // 3x8-bit RGB
    PIXEL_FORMAT_RGB_565      = HAL_PIXEL_FORMAT_RGB_565,      // 16-bit RGB
    PIXEL_FORMAT_BGRA_8888    = HAL_PIXEL_FORMAT_BGRA_8888,    // 4x8-bit BGRA
    PIXEL_FORMAT_RGBA_5551    = 6,                             // 16-bit ARGB
    PIXEL_FORMAT_RGBA_4444    = 7,                             // 16-bit ARGB
    PIXEL_FORMAT_RGBA_FP16    = HAL_PIXEL_FORMAT_RGBA_FP16,    // 64-bit RGBA
    PIXEL_FORMAT_RGBA_1010102 = HAL_PIXEL_FORMAT_RGBA_1010102, // 32-bit RGBA
};

typedef int32_t PixelFormat;

uint32_t bytesPerPixel(PixelFormat format);
uint32_t bitsPerPixel(PixelFormat format);

}; // namespace android

#endif // UI_PIXELFORMAT_H
