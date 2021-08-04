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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/**
 * pixel format definitions
 */

enum {
    /*
     * "linear" color pixel formats:
     *
     * When used with ANativeWindow, the dataSpace field describes the color
     * space of the buffer.
     *
     * The color space determines, for example, if the formats are linear or
     * gamma-corrected; or whether any special operations are performed when
     * reading or writing into a buffer in one of these formats.
     */
    HAL_PIXEL_FORMAT_RGBA_8888          = 1,
    HAL_PIXEL_FORMAT_RGBX_8888          = 2,
    HAL_PIXEL_FORMAT_RGB_888            = 3,
    HAL_PIXEL_FORMAT_RGB_565            = 4,
    HAL_PIXEL_FORMAT_BGRA_8888          = 5,

    /*
     * 0x100 - 0x1FF
     *
     * This range is reserved for pixel formats that are specific to the HAL
     * implementation.  Implementations can use any value in this range to
     * communicate video pixel formats between their HAL modules.  These formats
     * must not have an alpha channel.  Additionally, an EGLimage created from a
     * gralloc buffer of one of these formats must be supported for use with the
     * GL_OES_EGL_image_external OpenGL ES extension.
     */

    /*
     * Android YUV format:
     *
     * This format is exposed outside of the HAL to software decoders and
     * applications.  EGLImageKHR must support it in conjunction with the
     * OES_EGL_image_external extension.
     *
     * YV12 is a 4:2:0 YCrCb planar format comprised of a WxH Y plane followed
     * by (W/2) x (H/2) Cr and Cb planes.
     *
     * This format assumes
     * - an even width
     * - an even height
     * - a horizontal stride multiple of 16 pixels
     * - a vertical stride equal to the height
     *
     *   y_size = stride * height
     *   c_stride = ALIGN(stride/2, 16)
     *   c_size = c_stride * height/2
     *   size = y_size + c_size * 2
     *   cr_offset = y_size
     *   cb_offset = y_size + c_size
     *
     * When used with ANativeWindow, the dataSpace field describes the color
     * space of the buffer.
     */
    HAL_PIXEL_FORMAT_YV12   = 0x32315659, // YCrCb 4:2:0 Planar


    /*
     * Android Y8 format:
     *
     * This format is exposed outside of the HAL to the framework.
     * The expected gralloc usage flags are SW_* and HW_CAMERA_*,
     * and no other HW_ flags will be used.
     *
     * Y8 is a YUV planar format comprised of a WxH Y plane,
     * with each pixel being represented by 8 bits.
     *
     * It is equivalent to just the Y plane from YV12.
     *
     * This format assumes
     * - an even width
     * - an even height
     * - a horizontal stride multiple of 16 pixels
     * - a vertical stride equal to the height
     *
     *   size = stride * height
     *
     * When used with ANativeWindow, the dataSpace field describes the color
     * space of the buffer.
     */
    HAL_PIXEL_FORMAT_Y8     = 0x20203859,

    /*
     * Android Y16 format:
     *
     * This format is exposed outside of the HAL to the framework.
     * The expected gralloc usage flags are SW_* and HW_CAMERA_*,
     * and no other HW_ flags will be used.
     *
     * Y16 is a YUV planar format comprised of a WxH Y plane,
     * with each pixel being represented by 16 bits.
     *
     * It is just like Y8, but has double the bits per pixel (little endian).
     *
     * This format assumes
     * - an even width
     * - an even height
     * - a horizontal stride multiple of 16 pixels
     * - a vertical stride equal to the height
     * - strides are specified in pixels, not in bytes
     *
     *   size = stride * height * 2
     *
     * When used with ANativeWindow, the dataSpace field describes the color
     * space of the buffer, except that dataSpace field
     * HAL_DATASPACE_DEPTH indicates that this buffer contains a depth
     * image where each sample is a distance value measured by a depth camera,
     * plus an associated confidence value.
     */
    HAL_PIXEL_FORMAT_Y16    = 0x20363159,

    /*
     * Android RAW sensor format:
     *
     * This format is exposed outside of the camera HAL to applications.
     *
     * RAW16 is a single-channel, 16-bit, little endian format, typically
     * representing raw Bayer-pattern images from an image sensor, with minimal
     * processing.
     *
     * The exact pixel layout of the data in the buffer is sensor-dependent, and
     * needs to be queried from the camera device.
     *
     * Generally, not all 16 bits are used; more common values are 10 or 12
     * bits. If not all bits are used, the lower-order bits are filled first.
     * All parameters to interpret the raw data (black and white points,
     * color space, etc) must be queried from the camera device.
     *
     * This format assumes
     * - an even width
     * - an even height
     * - a horizontal stride multiple of 16 pixels
     * - a vertical stride equal to the height
     * - strides are specified in pixels, not in bytes
     *
     *   size = stride * height * 2
     *
     * This format must be accepted by the gralloc module when used with the
     * following usage flags:
     *    - GRALLOC_USAGE_HW_CAMERA_*
     *    - GRALLOC_USAGE_SW_*
     *    - GRALLOC_USAGE_RENDERSCRIPT
     *
     * When used with ANativeWindow, the dataSpace should be
     * HAL_DATASPACE_ARBITRARY, as raw image sensor buffers require substantial
     * extra metadata to define.
     */
    HAL_PIXEL_FORMAT_RAW16 = 0x20,

    /*
     * Android RAW10 format:
     *
     * This format is exposed outside of the camera HAL to applications.
     *
     * RAW10 is a single-channel, 10-bit per pixel, densely packed in each row,
     * unprocessed format, usually representing raw Bayer-pattern images coming from
     * an image sensor.
     *
     * In an image buffer with this format, starting from the first pixel of each
     * row, each 4 consecutive pixels are packed into 5 bytes (40 bits). Each one
     * of the first 4 bytes contains the top 8 bits of each pixel, The fifth byte
     * contains the 2 least significant bits of the 4 pixels, the exact layout data
     * for each 4 consecutive pixels is illustrated below (Pi[j] stands for the jth
     * bit of the ith pixel):
     *
     *          bit 7                                     bit 0
     *          =====|=====|=====|=====|=====|=====|=====|=====|
     * Byte 0: |P0[9]|P0[8]|P0[7]|P0[6]|P0[5]|P0[4]|P0[3]|P0[2]|
     *         |-----|-----|-----|-----|-----|-----|-----|-----|
     * Byte 1: |P1[9]|P1[8]|P1[7]|P1[6]|P1[5]|P1[4]|P1[3]|P1[2]|
     *         |-----|-----|-----|-----|-----|-----|-----|-----|
     * Byte 2: |P2[9]|P2[8]|P2[7]|P2[6]|P2[5]|P2[4]|P2[3]|P2[2]|
     *         |-----|-----|-----|-----|-----|-----|-----|-----|
     * Byte 3: |P3[9]|P3[8]|P3[7]|P3[6]|P3[5]|P3[4]|P3[3]|P3[2]|
     *         |-----|-----|-----|-----|-----|-----|-----|-----|
     * Byte 4: |P3[1]|P3[0]|P2[1]|P2[0]|P1[1]|P1[0]|P0[1]|P0[0]|
     *          ===============================================
     *
     * This format assumes
     * - a width multiple of 4 pixels
     * - an even height
     * - a vertical stride equal to the height
     * - strides are specified in bytes, not in pixels
     *
     *   size = stride * height
     *
     * When stride is equal to width * (10 / 8), there will be no padding bytes at
     * the end of each row, the entire image data is densely packed. When stride is
     * larger than width * (10 / 8), padding bytes will be present at the end of each
     * row (including the last row).
     *
     * This format must be accepted by the gralloc module when used with the
     * following usage flags:
     *    - GRALLOC_USAGE_HW_CAMERA_*
     *    - GRALLOC_USAGE_SW_*
     *    - GRALLOC_USAGE_RENDERSCRIPT
     *
     * When used with ANativeWindow, the dataSpace field should be
     * HAL_DATASPACE_ARBITRARY, as raw image sensor buffers require substantial
     * extra metadata to define.
     */
    HAL_PIXEL_FORMAT_RAW10 = 0x25,

    /*
     * Android RAW12 format:
     *
     * This format is exposed outside of camera HAL to applications.
     *
     * RAW12 is a single-channel, 12-bit per pixel, densely packed in each row,
     * unprocessed format, usually representing raw Bayer-pattern images coming from
     * an image sensor.
     *
     * In an image buffer with this format, starting from the first pixel of each
     * row, each two consecutive pixels are packed into 3 bytes (24 bits). The first
     * and second byte contains the top 8 bits of first and second pixel. The third
     * byte contains the 4 least significant bits of the two pixels, the exact layout
     * data for each two consecutive pixels is illustrated below (Pi[j] stands for
     * the jth bit of the ith pixel):
     *
     *           bit 7                                            bit 0
     *          ======|======|======|======|======|======|======|======|
     * Byte 0: |P0[11]|P0[10]|P0[ 9]|P0[ 8]|P0[ 7]|P0[ 6]|P0[ 5]|P0[ 4]|
     *         |------|------|------|------|------|------|------|------|
     * Byte 1: |P1[11]|P1[10]|P1[ 9]|P1[ 8]|P1[ 7]|P1[ 6]|P1[ 5]|P1[ 4]|
     *         |------|------|------|------|------|------|------|------|
     * Byte 2: |P1[ 3]|P1[ 2]|P1[ 1]|P1[ 0]|P0[ 3]|P0[ 2]|P0[ 1]|P0[ 0]|
     *          =======================================================
     *
     * This format assumes:
     * - a width multiple of 4 pixels
     * - an even height
     * - a vertical stride equal to the height
     * - strides are specified in bytes, not in pixels
     *
     *   size = stride * height
     *
     * When stride is equal to width * (12 / 8), there will be no padding bytes at
     * the end of each row, the entire image data is densely packed. When stride is
     * larger than width * (12 / 8), padding bytes will be present at the end of
     * each row (including the last row).
     *
     * This format must be accepted by the gralloc module when used with the
     * following usage flags:
     *    - GRALLOC_USAGE_HW_CAMERA_*
     *    - GRALLOC_USAGE_SW_*
     *    - GRALLOC_USAGE_RENDERSCRIPT
     *
     * When used with ANativeWindow, the dataSpace field should be
     * HAL_DATASPACE_ARBITRARY, as raw image sensor buffers require substantial
     * extra metadata to define.
     */
    HAL_PIXEL_FORMAT_RAW12 = 0x26,

    /*
     * Android opaque RAW format:
     *
     * This format is exposed outside of the camera HAL to applications.
     *
     * RAW_OPAQUE is a format for unprocessed raw image buffers coming from an
     * image sensor. The actual structure of buffers of this format is
     * implementation-dependent.
     *
     * This format must be accepted by the gralloc module when used with the
     * following usage flags:
     *    - GRALLOC_USAGE_HW_CAMERA_*
     *    - GRALLOC_USAGE_SW_*
     *    - GRALLOC_USAGE_RENDERSCRIPT
     *
     * When used with ANativeWindow, the dataSpace field should be
     * HAL_DATASPACE_ARBITRARY, as raw image sensor buffers require substantial
     * extra metadata to define.
     */
    HAL_PIXEL_FORMAT_RAW_OPAQUE = 0x24,

    /*
     * Android binary blob graphics buffer format:
     *
     * This format is used to carry task-specific data which does not have a
     * standard image structure. The details of the format are left to the two
     * endpoints.
     *
     * A typical use case is for transporting JPEG-compressed images from the
     * Camera HAL to the framework or to applications.
     *
     * Buffers of this format must have a height of 1, and width equal to their
     * size in bytes.
     *
     * When used with ANativeWindow, the mapping of the dataSpace field to
     * buffer contents for BLOB is as follows:
     *
     *  dataSpace value               | Buffer contents
     * -------------------------------+-----------------------------------------
     *  HAL_DATASPACE_JFIF            | An encoded JPEG image
     *  HAL_DATASPACE_DEPTH           | An android_depth_points buffer
     *  Other                         | Unsupported
     *
     */
    HAL_PIXEL_FORMAT_BLOB = 0x21,

    /*
     * Android format indicating that the choice of format is entirely up to the
     * device-specific Gralloc implementation.
     *
     * The Gralloc implementation should examine the usage bits passed in when
     * allocating a buffer with this format, and it should derive the pixel
     * format from those usage flags.  This format will never be used with any
     * of the GRALLOC_USAGE_SW_* usage flags.
     *
     * If a buffer of this format is to be used as an OpenGL ES texture, the
     * framework will assume that sampling the texture will always return an
     * alpha value of 1.0 (i.e. the buffer contains only opaque pixel values).
     *
     * When used with ANativeWindow, the dataSpace field describes the color
     * space of the buffer.
     */
    HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED = 0x22,

    /*
     * Android flexible YCbCr 4:2:0 formats
     *
     * This format allows platforms to use an efficient YCbCr/YCrCb 4:2:0
     * buffer layout, while still describing the general format in a
     * layout-independent manner.  While called YCbCr, it can be
     * used to describe formats with either chromatic ordering, as well as
     * whole planar or semiplanar layouts.
     *
     * struct android_ycbcr (below) is the the struct used to describe it.
     *
     * This format must be accepted by the gralloc module when
     * USAGE_SW_WRITE_* or USAGE_SW_READ_* are set.
     *
     * This format is locked for use by gralloc's (*lock_ycbcr) method, and
     * locking with the (*lock) method will return an error.
     *
     * When used with ANativeWindow, the dataSpace field describes the color
     * space of the buffer.
     */
    HAL_PIXEL_FORMAT_YCbCr_420_888 = 0x23,

    /*
     * Android flexible YCbCr 4:2:2 formats
     *
     * This format allows platforms to use an efficient YCbCr/YCrCb 4:2:2
     * buffer layout, while still describing the general format in a
     * layout-independent manner.  While called YCbCr, it can be
     * used to describe formats with either chromatic ordering, as well as
     * whole planar or semiplanar layouts.
     *
     * This format is currently only used by SW readable buffers
     * produced by MediaCodecs, so the gralloc module can ignore this format.
     */
    HAL_PIXEL_FORMAT_YCbCr_422_888 = 0x27,

    /*
     * Android flexible YCbCr 4:4:4 formats
     *
     * This format allows platforms to use an efficient YCbCr/YCrCb 4:4:4
     * buffer layout, while still describing the general format in a
     * layout-independent manner.  While called YCbCr, it can be
     * used to describe formats with either chromatic ordering, as well as
     * whole planar or semiplanar layouts.
     *
     * This format is currently only used by SW readable buffers
     * produced by MediaCodecs, so the gralloc module can ignore this format.
     */
    HAL_PIXEL_FORMAT_YCbCr_444_888 = 0x28,

    /*
     * Android flexible RGB 888 formats
     *
     * This format allows platforms to use an efficient RGB/BGR/RGBX/BGRX
     * buffer layout, while still describing the general format in a
     * layout-independent manner.  While called RGB, it can be
     * used to describe formats with either color ordering and optional
     * padding, as well as whole planar layout.
     *
     * This format is currently only used by SW readable buffers
     * produced by MediaCodecs, so the gralloc module can ignore this format.
     */
    HAL_PIXEL_FORMAT_FLEX_RGB_888 = 0x29,

    /*
     * Android flexible RGBA 8888 formats
     *
     * This format allows platforms to use an efficient RGBA/BGRA/ARGB/ABGR
     * buffer layout, while still describing the general format in a
     * layout-independent manner.  While called RGBA, it can be
     * used to describe formats with any of the component orderings, as
     * well as whole planar layout.
     *
     * This format is currently only used by SW readable buffers
     * produced by MediaCodecs, so the gralloc module can ignore this format.
     */
    HAL_PIXEL_FORMAT_FLEX_RGBA_8888 = 0x2A,

    /* Legacy formats (deprecated), used by ImageFormat.java */
    HAL_PIXEL_FORMAT_YCbCr_422_SP       = 0x10, // NV16
    HAL_PIXEL_FORMAT_YCrCb_420_SP       = 0x11, // NV21
    HAL_PIXEL_FORMAT_YCbCr_422_I        = 0x14, // YUY2
};

/*
 * Structure for describing YCbCr formats for consumption by applications.
 * This is used with HAL_PIXEL_FORMAT_YCbCr_*_888.
 *
 * Buffer chroma subsampling is defined in the format.
 * e.g. HAL_PIXEL_FORMAT_YCbCr_420_888 has subsampling 4:2:0.
 *
 * Buffers must have a 8 bit depth.
 *
 * @y, @cb, and @cr point to the first byte of their respective planes.
 *
 * Stride describes the distance in bytes from the first value of one row of
 * the image to the first value of the next row.  It includes the width of the
 * image plus padding.
 * @ystride is the stride of the luma plane.
 * @cstride is the stride of the chroma planes.
 *
 * @chroma_step is the distance in bytes from one chroma pixel value to the
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
 * @num_points is the number of points in the list
 *
 * @xyz_points is the flexible array of floating-point values.
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

    float xyzc_points[];
};

/**
 * Transformation definitions
 *
 * IMPORTANT NOTE:
 * HAL_TRANSFORM_ROT_90 is applied CLOCKWISE and AFTER HAL_TRANSFORM_FLIP_{H|V}.
 *
 */

enum {
    /* flip source image horizontally (around the vertical axis) */
    HAL_TRANSFORM_FLIP_H    = 0x01,
    /* flip source image vertically (around the horizontal axis)*/
    HAL_TRANSFORM_FLIP_V    = 0x02,
    /* rotate source image 90 degrees clockwise */
    HAL_TRANSFORM_ROT_90    = 0x04,
    /* rotate source image 180 degrees */
    HAL_TRANSFORM_ROT_180   = 0x03,
    /* rotate source image 270 degrees clockwise */
    HAL_TRANSFORM_ROT_270   = 0x07,
    /* don't use. see system/window.h */
    HAL_TRANSFORM_RESERVED  = 0x08,
};

/**
 * Dataspace Definitions
 * ======================
 *
 * Dataspace is the definition of how pixel values should be interpreted.
 *
 * For many formats, this is the colorspace of the image data, which includes
 * primaries (including white point) and the transfer characteristic function,
 * which describes both gamma curve and numeric range (within the bit depth).
 *
 * Other dataspaces include depth measurement data from a depth camera.
 */

typedef enum android_dataspace {
    /*
     * Default-assumption data space, when not explicitly specified.
     *
     * It is safest to assume the buffer is an image with sRGB primaries and
     * encoding ranges, but the consumer and/or the producer of the data may
     * simply be using defaults. No automatic gamma transform should be
     * expected, except for a possible display gamma transform when drawn to a
     * screen.
     */
    HAL_DATASPACE_UNKNOWN = 0x0,

    /*
     * Arbitrary dataspace with manually defined characteristics.  Definition
     * for colorspaces or other meaning must be communicated separately.
     *
     * This is used when specifying primaries, transfer characteristics,
     * etc. separately.
     *
     * A typical use case is in video encoding parameters (e.g. for H.264),
     * where a colorspace can have separately defined primaries, transfer
     * characteristics, etc.
     */
    HAL_DATASPACE_ARBITRARY = 0x1,

    /*
     * RGB Colorspaces
     * -----------------
     *
     * Primaries are given using (x,y) coordinates in the CIE 1931 definition
     * of x and y specified by ISO 11664-1.
     *
     * Transfer characteristics are the opto-electronic transfer characteristic
     * at the source as a function of linear optical intensity (luminance).
     */

    /*
     * sRGB linear encoding:
     *
     * The red, green, and blue components are stored in sRGB space, but
     * are linear, not gamma-encoded.
     * The RGB primaries and the white point are the same as BT.709.
     *
     * The values are encoded using the full range ([0,255] for 8-bit) for all
     * components.
     */
    HAL_DATASPACE_SRGB_LINEAR = 0x200,

    /*
     * sRGB gamma encoding:
     *
     * The red, green and blue components are stored in sRGB space, and
     * converted to linear space when read, using the standard sRGB to linear
     * equation:
     *
     * Clinear = Csrgb / 12.92                  for Csrgb <= 0.04045
     *         = (Csrgb + 0.055 / 1.055)^2.4    for Csrgb >  0.04045
     *
     * When written the inverse transformation is performed:
     *
     * Csrgb = 12.92 * Clinear                  for Clinear <= 0.0031308
     *       = 1.055 * Clinear^(1/2.4) - 0.055  for Clinear >  0.0031308
     *
     *
     * The alpha component, if present, is always stored in linear space and
     * is left unmodified when read or written.
     *
     * The RGB primaries and the white point are the same as BT.709.
     *
     * The values are encoded using the full range ([0,255] for 8-bit) for all
     * components.
     *
     */
    HAL_DATASPACE_SRGB = 0x201,

    /*
     * YCbCr Colorspaces
     * -----------------
     *
     * Primaries are given using (x,y) coordinates in the CIE 1931 definition
     * of x and y specified by ISO 11664-1.
     *
     * Transfer characteristics are the opto-electronic transfer characteristic
     * at the source as a function of linear optical intensity (luminance).
     */

    /*
     * JPEG File Interchange Format (JFIF)
     *
     * Same model as BT.601-625, but all values (Y, Cb, Cr) range from 0 to 255
     *
     * Transfer characteristic curve:
     *  E = 1.099 * L ^ 0.45 - 0.099, 1.00 >= L >= 0.018
     *  E = 4.500 L, 0.018 > L >= 0
     *      L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *      E - corresponding electrical signal
     *
     * Primaries:       x       y
     *  green           0.290   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     */
    HAL_DATASPACE_JFIF = 0x101,

    /*
     * ITU-R Recommendation 601 (BT.601) - 625-line
     *
     * Standard-definition television, 625 Lines (PAL)
     *
     * For 8-bit-depth formats:
     * Luma (Y) samples should range from 16 to 235, inclusive
     * Chroma (Cb, Cr) samples should range from 16 to 240, inclusive
     *
     * For 10-bit-depth formats:
     * Luma (Y) samples should range from 64 to 940, inclusive
     * Chroma (Cb, Cr) samples should range from 64 to 960, inclusive
     *
     * Transfer characteristic curve:
     *  E = 1.099 * L ^ 0.45 - 0.099, 1.00 >= L >= 0.018
     *  E = 4.500 L, 0.018 > L >= 0
     *      L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *      E - corresponding electrical signal
     *
     * Primaries:       x       y
     *  green           0.290   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     */
    HAL_DATASPACE_BT601_625 = 0x102,

    /*
     * ITU-R Recommendation 601 (BT.601) - 525-line
     *
     * Standard-definition television, 525 Lines (NTSC)
     *
     * For 8-bit-depth formats:
     * Luma (Y) samples should range from 16 to 235, inclusive
     * Chroma (Cb, Cr) samples should range from 16 to 240, inclusive
     *
     * For 10-bit-depth formats:
     * Luma (Y) samples should range from 64 to 940, inclusive
     * Chroma (Cb, Cr) samples should range from 64 to 960, inclusive
     *
     * Transfer characteristic curve:
     *  E = 1.099 * L ^ 0.45 - 0.099, 1.00 >= L >= 0.018
     *  E = 4.500 L, 0.018 > L >= 0
     *      L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *      E - corresponding electrical signal
     *
     * Primaries:       x       y
     *  green           0.310   0.595
     *  blue            0.155   0.070
     *  red             0.630   0.340
     *  white (D65)     0.3127  0.3290
     */
    HAL_DATASPACE_BT601_525 = 0x103,

    /*
     * ITU-R Recommendation 709 (BT.709)
     *
     * High-definition television
     *
     * For 8-bit-depth formats:
     * Luma (Y) samples should range from 16 to 235, inclusive
     * Chroma (Cb, Cr) samples should range from 16 to 240, inclusive
     *
     * For 10-bit-depth formats:
     * Luma (Y) samples should range from 64 to 940, inclusive
     * Chroma (Cb, Cr) samples should range from 64 to 960, inclusive
     *
     * Primaries:       x       y
     *  green           0.300   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     */
    HAL_DATASPACE_BT709 = 0x104,

    /*
     * The buffer contains depth ranging measurements from a depth camera.
     * This value is valid with formats:
     *    HAL_PIXEL_FORMAT_Y16: 16-bit samples, consisting of a depth measurement
     *       and an associated confidence value. The 3 MSBs of the sample make
     *       up the confidence value, and the low 13 LSBs of the sample make up
     *       the depth measurement.
     *       For the confidence section, 0 means 100% confidence, 1 means 0%
     *       confidence. The mapping to a linear float confidence value between
     *       0.f and 1.f can be obtained with
     *         float confidence = (((depthSample >> 13) - 1) & 0x7) / 7.0f;
     *       The depth measurement can be extracted simply with
     *         uint16_t range = (depthSample & 0x1FFF);
     *    HAL_PIXEL_FORMAT_BLOB: A depth point cloud, as
     *       a variable-length float (x,y,z, confidence) coordinate point list.
     *       The point cloud will be represented with the android_depth_points
     *       structure.
     */
    HAL_DATASPACE_DEPTH = 0x1000

} android_dataspace_t;

#ifdef __cplusplus
}
#endif

#endif /* SYSTEM_CORE_INCLUDE_ANDROID_GRAPHICS_H */
