/*
 * Copyright (C) 2011 The Android Open Source Project
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

#ifndef SYSTEM_CORE_INCLUDE_ANDROID_GRAPHICS_H
#define SYSTEM_CORE_INCLUDE_ANDROID_GRAPHICS_H

#include <stddef.h>
#include <stdint.h>

/*
 * Some of the enums are now defined in HIDL in hardware/interfaces and are
 * generated.
 */
#include "graphics-base.h"
#include "graphics-sw.h"

#ifdef __cplusplus
extern "C" {
#endif

/* for compatibility */
#define HAL_PIXEL_FORMAT_YCbCr_420_888 HAL_PIXEL_FORMAT_YCBCR_420_888
#define HAL_PIXEL_FORMAT_YCbCr_422_SP HAL_PIXEL_FORMAT_YCBCR_422_SP
#define HAL_PIXEL_FORMAT_YCrCb_420_SP HAL_PIXEL_FORMAT_YCRCB_420_SP
#define HAL_PIXEL_FORMAT_YCbCr_422_I HAL_PIXEL_FORMAT_YCBCR_422_I
typedef android_pixel_format_t android_pixel_format;
typedef android_transform_t android_transform;
typedef android_dataspace_t android_dataspace;
typedef android_color_mode_t android_color_mode;
typedef android_color_transform_t android_color_transform;
typedef android_hdr_t android_hdr;

/*
 * If the HAL needs to create service threads to handle graphics related
 * tasks, these threads need to run at HAL_PRIORITY_URGENT_DISPLAY priority
 * if they can block the main rendering thread in any way.
 *
 * the priority of the current thread can be set with:
 *
 *      #include <sys/resource.h>
 *      setpriority(PRIO_PROCESS, 0, HAL_PRIORITY_URGENT_DISPLAY);
 *
 */

#define HAL_PRIORITY_URGENT_DISPLAY     (-8)

/*
 * Structure for describing YCbCr formats for consumption by applications.
 * This is used with HAL_PIXEL_FORMAT_YCbCr_*_888.
 *
 * Buffer chroma subsampling is defined in the format.
 * e.g. HAL_PIXEL_FORMAT_YCbCr_420_888 has subsampling 4:2:0.
 *
 * Buffers must have a 8 bit depth.
 *
 * y, cb, and cr point to the first byte of their respective planes.
 *
 * Stride describes the distance in bytes from the first value of one row of
 * the image to the first value of the next row.  It includes the width of the
 * image plus padding.
 * ystride is the stride of the luma plane.
 * cstride is the stride of the chroma planes.
 *
 * chroma_step is the distance in bytes from one chroma pixel value to the
 * next.  This is 2 bytes for semiplanar (because chroma values are interleaved
 * and each chroma value is one byte) and 1 for planar.
 */

struct android_ycbcr {
    void *y;
    void *cb;
    void *cr;
    size_t ystride;
    size_t cstride;
    size_t chroma_step;

    /** reserved for future use, set to 0 by gralloc's (*lock_ycbcr)() */
    uint32_t reserved[8];
};

/*
 * Structures for describing flexible YUVA/RGBA formats for consumption by
 * applications. Such flexible formats contain a plane for each component (e.g.
 * red, green, blue), where each plane is laid out in a grid-like pattern
 * occupying unique byte addresses and with consistent byte offsets between
 * neighboring pixels.
 *
 * The android_flex_layout structure is used with any pixel format that can be
 * represented by it, such as:
 *  - HAL_PIXEL_FORMAT_YCbCr_*_888
 *  - HAL_PIXEL_FORMAT_FLEX_RGB*_888
 *  - HAL_PIXEL_FORMAT_RGB[AX]_888[8],BGRA_8888,RGB_888
 *  - HAL_PIXEL_FORMAT_YV12,Y8,Y16,YCbCr_422_SP/I,YCrCb_420_SP
 *  - even implementation defined formats that can be represented by
 *    the structures
 *
 * Vertical increment (aka. row increment or stride) describes the distance in
 * bytes from the first pixel of one row to the first pixel of the next row
 * (below) for the component plane. This can be negative.
 *
 * Horizontal increment (aka. column or pixel increment) describes the distance
 * in bytes from one pixel to the next pixel (to the right) on the same row for
 * the component plane. This can be negative.
 *
 * Each plane can be subsampled either vertically or horizontally by
 * a power-of-two factor.
 *
 * The bit-depth of each component can be arbitrary, as long as the pixels are
 * laid out on whole bytes, in native byte-order, using the most significant
 * bits of each unit.
 */

typedef enum android_flex_component {
    /* luma */
    FLEX_COMPONENT_Y = 1 << 0,
    /* chroma blue */
    FLEX_COMPONENT_Cb = 1 << 1,
    /* chroma red */
    FLEX_COMPONENT_Cr = 1 << 2,

    /* red */
    FLEX_COMPONENT_R = 1 << 10,
    /* green */
    FLEX_COMPONENT_G = 1 << 11,
    /* blue */
    FLEX_COMPONENT_B = 1 << 12,

    /* alpha */
    FLEX_COMPONENT_A = 1 << 30,
} android_flex_component_t;

typedef struct android_flex_plane {
    /* pointer to the first byte of the top-left pixel of the plane. */
    uint8_t *top_left;

    android_flex_component_t component;

    /* bits allocated for the component in each pixel. Must be a positive
       multiple of 8. */
    int32_t bits_per_component;
    /* number of the most significant bits used in the format for this
       component. Must be between 1 and bits_per_component, inclusive. */
    int32_t bits_used;

    /* horizontal increment */
    int32_t h_increment;
    /* vertical increment */
    int32_t v_increment;
    /* horizontal subsampling. Must be a positive power of 2. */
    int32_t h_subsampling;
    /* vertical subsampling. Must be a positive power of 2. */
    int32_t v_subsampling;
} android_flex_plane_t;

typedef enum android_flex_format {
    /* not a flexible format */
    FLEX_FORMAT_INVALID = 0x0,
    FLEX_FORMAT_Y = FLEX_COMPONENT_Y,
    FLEX_FORMAT_YCbCr = FLEX_COMPONENT_Y | FLEX_COMPONENT_Cb | FLEX_COMPONENT_Cr,
    FLEX_FORMAT_YCbCrA = FLEX_FORMAT_YCbCr | FLEX_COMPONENT_A,
    FLEX_FORMAT_RGB = FLEX_COMPONENT_R | FLEX_COMPONENT_G | FLEX_COMPONENT_B,
    FLEX_FORMAT_RGBA = FLEX_FORMAT_RGB | FLEX_COMPONENT_A,
} android_flex_format_t;

typedef struct android_flex_layout {
    /* the kind of flexible format */
    android_flex_format_t format;

    /* number of planes; 0 for FLEX_FORMAT_INVALID */
    uint32_t num_planes;
    /* a plane for each component; ordered in increasing component value order.
       E.g. FLEX_FORMAT_RGBA maps 0 -> R, 1 -> G, etc.
       Can be NULL for FLEX_FORMAT_INVALID */
    android_flex_plane_t *planes;
} android_flex_layout_t;

/**
 * Structure used to define depth point clouds for format HAL_PIXEL_FORMAT_BLOB
 * with dataSpace value of HAL_DATASPACE_DEPTH.
 * When locking a native buffer of the above format and dataSpace value,
 * the vaddr pointer can be cast to this structure.
 *
 * A variable-length list of (x,y,z, confidence) 3D points, as floats.  (x, y,
 * z) represents a measured point's position, with the coordinate system defined
 * by the data source.  Confidence represents the estimated likelihood that this
 * measurement is correct. It is between 0.f and 1.f, inclusive, with 1.f ==
 * 100% confidence.
 *
 * num_points is the number of points in the list
 *
 * xyz_points is the flexible array of floating-point values.
 *   It contains (num_points) * 4 floats.
 *
 *   For example:
 *     android_depth_points d = get_depth_buffer();
 *     struct {
 *       float x; float y; float z; float confidence;
 *     } firstPoint, lastPoint;
 *
 *     firstPoint.x = d.xyzc_points[0];
 *     firstPoint.y = d.xyzc_points[1];
 *     firstPoint.z = d.xyzc_points[2];
 *     firstPoint.confidence = d.xyzc_points[3];
 *     lastPoint.x = d.xyzc_points[(d.num_points - 1) * 4 + 0];
 *     lastPoint.y = d.xyzc_points[(d.num_points - 1) * 4 + 1];
 *     lastPoint.z = d.xyzc_points[(d.num_points - 1) * 4 + 2];
 *     lastPoint.confidence = d.xyzc_points[(d.num_points - 1) * 4 + 3];
 */

struct android_depth_points {
    uint32_t num_points;

    /** reserved for future use, set to 0 by gralloc's (*lock)() */
    uint32_t reserved[8];

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif
    float xyzc_points[];
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
};

/**
  * These structures are used to define the reference display's
  * capabilities for HDR content. Display engine can use this
  * to better tone map content to user's display.
  * Color is defined in CIE XYZ coordinates
  */
struct android_xy_color {
    float x;
    float y;
};

struct android_smpte2086_metadata {
    struct android_xy_color displayPrimaryRed;
    struct android_xy_color displayPrimaryGreen;
    struct android_xy_color displayPrimaryBlue;
    struct android_xy_color whitePoint;
    float maxLuminance;
    float minLuminance;
};

struct android_cta861_3_metadata {
    float maxContentLightLevel;
    float maxFrameAverageLightLevel;
};

#ifdef __cplusplus
}
#endif

#endif /* SYSTEM_CORE_INCLUDE_ANDROID_GRAPHICS_H */
