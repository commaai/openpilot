#ifndef HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_0_TYPES_H
#define HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_0_TYPES_H

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hardware {
namespace graphics {
namespace common {
namespace V1_0 {

// Forward declaration for forward reference support:
enum class PixelFormat : int32_t;
enum class BufferUsage : uint64_t;
enum class Transform : int32_t;
enum class Dataspace : int32_t;
enum class ColorMode : int32_t;
enum class ColorTransform : int32_t;
enum class Hdr : int32_t;

/**
 * Common enumeration and structure definitions for all graphics HALs.
 * 
 * 
 * Pixel formats for graphics buffers.
 */
enum class PixelFormat : int32_t {
    /**
     * 32-bit format that has 8-bit R, G, B, and A components, in that order,
     * from the lowest memory address to the highest memory address.
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    RGBA_8888 = 1,
    /**
     * 32-bit format that has 8-bit R, G, B, and unused components, in that
     * order, from the lowest memory address to the highest memory address.
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    RGBX_8888 = 2,
    /**
     * 24-bit format that has 8-bit R, G, and B components, in that order,
     * from the lowest memory address to the highest memory address.
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    RGB_888 = 3,
    /**
     * 16-bit packed format that has 5-bit R, 6-bit G, and 5-bit B components,
     * in that order, from the most-sigfinicant bits to the least-significant
     * bits.
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    RGB_565 = 4,
    /**
     * 32-bit format that has 8-bit B, G, R, and A components, in that order,
     * from the lowest memory address to the highest memory address.
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    BGRA_8888 = 5,
    /**
     * Legacy formats deprecated in favor of YCBCR_420_888.
     */
    YCBCR_422_SP = 16,
    YCRCB_420_SP = 17,
    YCBCR_422_I = 20,
    /**
     * 64-bit format that has 16-bit R, G, B, and A components, in that order,
     * from the lowest memory address to the highest memory address.
     * 
     * The component values are signed floats, whose interpretation is defined
     * by the dataspace.
     */
    RGBA_FP16 = 22,
    /**
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
     * This format must be accepted by the allocator when used with the
     * following usage flags:
     * 
     *    - BufferUsage::CAMERA_*
     *    - BufferUsage::CPU_*
     *    - BufferUsage::RENDERSCRIPT
     * 
     * The mapping of the dataspace to buffer contents for RAW16 is as
     * follows:
     * 
     *  Dataspace value               | Buffer contents
     * -------------------------------+-----------------------------------------
     *  Dataspace::ARBITRARY          | Raw image sensor data, layout is as
     *                                | defined above.
     *  Dataspace::DEPTH              | Unprocessed implementation-dependent raw
     *                                | depth measurements, opaque with 16 bit
     *                                | samples.
     *  Other                         | Unsupported
     */
    RAW16 = 32,
    /**
     * BLOB is used to carry task-specific data which does not have a standard
     * image structure. The details of the format are left to the two
     * endpoints.
     * 
     * A typical use case is for transporting JPEG-compressed images from the
     * Camera HAL to the framework or to applications.
     * 
     * Buffers of this format must have a height of 1, and width equal to their
     * size in bytes.
     * 
     * The mapping of the dataspace to buffer contents for BLOB is as
     * follows:
     * 
     *  Dataspace value               | Buffer contents
     * -------------------------------+-----------------------------------------
     *  Dataspace::JFIF               | An encoded JPEG image
     *  Dataspace::DEPTH              | An android_depth_points buffer
     *  Dataspace::SENSOR             | Sensor event data.
     *  Other                         | Unsupported
     */
    BLOB = 33,
    /**
     * A format indicating that the choice of format is entirely up to the
     * allocator.
     * 
     * The allocator should examine the usage bits passed in when allocating a
     * buffer with this format, and it should derive the pixel format from
     * those usage flags. This format must never be used with any of the
     * BufferUsage::CPU_* usage flags.
     * 
     * Even when the internally chosen format has an alpha component, the
     * clients must assume the alpha vlaue to be 1.0.
     * 
     * The interpretation of the component values is defined by the dataspace.
     */
    IMPLEMENTATION_DEFINED = 34,
    /**
     * This format allows platforms to use an efficient YCbCr/YCrCb 4:2:0
     * buffer layout, while still describing the general format in a
     * layout-independent manner. While called YCbCr, it can be used to
     * describe formats with either chromatic ordering, as well as
     * whole planar or semiplanar layouts.
     * 
     * This format must be accepted by the allocator when BufferUsage::CPU_*
     * are set.
     * 
     * Buffers with this format must be locked with IMapper::lockYCbCr.
     * Locking with IMapper::lock must return an error.
     * 
     * The interpretation of the component values is defined by the dataspace.
     */
    YCBCR_420_888 = 35,
    /**
     * RAW_OPAQUE is a format for unprocessed raw image buffers coming from an
     * image sensor. The actual structure of buffers of this format is
     * implementation-dependent.
     * 
     * This format must be accepted by the allocator when used with the
     * following usage flags:
     * 
     *    - BufferUsage::CAMERA_*
     *    - BufferUsage::CPU_*
     *    - BufferUsage::RENDERSCRIPT
     * 
     * The mapping of the dataspace to buffer contents for RAW_OPAQUE is as
     * follows:
     * 
     *  Dataspace value               | Buffer contents
     * -------------------------------+-----------------------------------------
     *  Dataspace::ARBITRARY          | Raw image sensor data.
     *  Other                         | Unsupported
     */
    RAW_OPAQUE = 36,
    /**
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
     * This format must be accepted by the allocator when used with the
     * following usage flags:
     * 
     *    - BufferUsage::CAMERA_*
     *    - BufferUsage::CPU_*
     *    - BufferUsage::RENDERSCRIPT
     * 
     * The mapping of the dataspace to buffer contents for RAW10 is as
     * follows:
     * 
     *  Dataspace value               | Buffer contents
     * -------------------------------+-----------------------------------------
     *  Dataspace::ARBITRARY          | Raw image sensor data.
     *  Other                         | Unsupported
     */
    RAW10 = 37,
    /**
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
     * This format must be accepted by the allocator when used with the
     * following usage flags:
     * 
     *    - BufferUsage::CAMERA_*
     *    - BufferUsage::CPU_*
     *    - BufferUsage::RENDERSCRIPT
     * 
     * The mapping of the dataspace to buffer contents for RAW12 is as
     * follows:
     * 
     *  Dataspace value               | Buffer contents
     * -------------------------------+-----------------------------------------
     *  Dataspace::ARBITRARY          | Raw image sensor data.
     *  Other                         | Unsupported
     */
    RAW12 = 38,
    /**
     * 0x27 to 0x2A are reserved for flexible formats 
     * 
     * 32-bit packed format that has 2-bit A, 10-bit B, G, and R components,
     * in that order, from the most-sigfinicant bits to the least-significant
     * bits.
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    RGBA_1010102 = 43,
    /**
     * 0x100 - 0x1FF
     * 
     * This range is reserved for vendor extensions. Formats in this range
     * must support BufferUsage::GPU_TEXTURE. Clients must assume they do not
     * have an alpha component.
     * 
     * 
     * Y8 is a YUV planar format comprised of a WxH Y plane, with each pixel
     * being represented by 8 bits. It is equivalent to just the Y plane from
     * YV12.
     * 
     * This format assumes
     * - an even width
     * - an even height
     * - a horizontal stride multiple of 16 pixels
     * - a vertical stride equal to the height
     * 
     *   size = stride * height
     * 
     * This format must be accepted by the allocator when used with the
     * following usage flags:
     * 
     *    - BufferUsage::CAMERA_*
     *    - BufferUsage::CPU_*
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    Y8 = 538982489,
    /**
     * Y16 is a YUV planar format comprised of a WxH Y plane, with each pixel
     * being represented by 16 bits. It is just like Y8, but has double the
     * bits per pixel (little endian).
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
     * This format must be accepted by the allocator when used with the
     * following usage flags:
     * 
     *    - BufferUsage::CAMERA_*
     *    - BufferUsage::CPU_*
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace. When the dataspace is
     * Dataspace::DEPTH, each pixel is a distance value measured by a depth
     * camera, plus an associated confidence value.
     */
    Y16 = 540422489,
    /**
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
     * This range is reserved for vendor extensions. Formats in this range
     * must support BufferUsage::GPU_TEXTURE. Clients must assume they do not
     * have an alpha component.
     * 
     * This format must be accepted by the allocator when used with the
     * following usage flags:
     * 
     *    - BufferUsage::CAMERA_*
     *    - BufferUsage::CPU_*
     *    - BufferUsage::GPU_TEXTURE
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    YV12 = 842094169,
};

/**
 * Buffer usage definitions.
 */
enum class BufferUsage : uint64_t {
    /**
     * bit 0-3 is an enum  */
    CPU_READ_MASK = 15ull,
    /**
     * buffer is never read by CPU  */
    CPU_READ_NEVER = 0ull,
    /**
     * buffer is rarely read by CPU  */
    CPU_READ_RARELY = 2ull,
    /**
     * buffer is often read by CPU  */
    CPU_READ_OFTEN = 3ull,
    /**
     * bit 4-7 is an enum  */
    CPU_WRITE_MASK = 240ull, // (0xfULL << 4)
    /**
     * buffer is never written by CPU  */
    CPU_WRITE_NEVER = 0ull, // (0 << 4)
    /**
     * buffer is rarely written by CPU  */
    CPU_WRITE_RARELY = 32ull, // (2 << 4)
    /**
     * buffer is often written by CPU  */
    CPU_WRITE_OFTEN = 48ull, // (3 << 4)
    /**
     * buffer is used as a GPU texture  */
    GPU_TEXTURE = 256ull, // (1ULL << 8)
    /**
     * buffer is used as a GPU render target  */
    GPU_RENDER_TARGET = 512ull, // (1ULL << 9)
    /**
     * bit 10 must be zero 
     * 
     * buffer is used as a composer HAL overlay layer  */
    COMPOSER_OVERLAY = 2048ull, // (1ULL << 11)
    /**
     * buffer is used as a composer HAL client target  */
    COMPOSER_CLIENT_TARGET = 4096ull, // (1ULL << 12)
    /**
     * bit 13 must be zero 
     * 
     * Buffer is allocated with hardware-level protection against copying the
     * contents (or information derived from the contents) into unprotected
     * memory.
     */
    PROTECTED = 16384ull, // (1ULL << 14)
    /**
     * buffer is used as a hwcomposer HAL cursor layer  */
    COMPOSER_CURSOR = 32768ull, // (1ULL << 15)
    /**
     * buffer is used as a video encoder input  */
    VIDEO_ENCODER = 65536ull, // (1ULL << 16)
    /**
     * buffer is used as a camera HAL output  */
    CAMERA_OUTPUT = 131072ull, // (1ULL << 17)
    /**
     * buffer is used as a camera HAL input  */
    CAMERA_INPUT = 262144ull, // (1ULL << 18)
    /**
     * bit 19 must be zero 
     * 
     * buffer is used as a renderscript allocation  */
    RENDERSCRIPT = 1048576ull, // (1ULL << 20)
    /**
     * bit 21 must be zero 
     * 
     * buffer is used as a video decoder output  */
    VIDEO_DECODER = 4194304ull, // (1ULL << 22)
    /**
     * buffer is used as a sensor direct report output  */
    SENSOR_DIRECT_DATA = 8388608ull, // (1ULL << 23)
    /**
     * buffer is used as as an OpenGL shader storage or uniform
     * buffer object
     */
    GPU_DATA_BUFFER = 16777216ull, // (1ULL << 24)
    /**
     * bits 25-27 must be zero and are reserved for future versions 
     * 
     * bits 28-31 are reserved for vendor extensions  */
    VENDOR_MASK = 4026531840ull, // (0xfULL << 28)
    /**
     * bits 32-47 must be zero and are reserved for future versions 
     * 
     * bits 48-63 are reserved for vendor extensions  */
    VENDOR_MASK_HI = 18446462598732840960ull, // (0xffffULL << 48)
};

/**
 * Transformation definitions
 */
enum class Transform : int32_t {
    /**
     * Horizontal flip. FLIP_H/FLIP_V is applied before ROT_90.
     */
    FLIP_H = 1, // (1 << 0)
    /**
     * Vertical flip. FLIP_H/FLIP_V is applied before ROT_90.
     */
    FLIP_V = 2, // (1 << 1)
    /**
     * 90 degree clockwise rotation. FLIP_H/FLIP_V is applied before ROT_90.
     */
    ROT_90 = 4, // (1 << 2)
    /**
     * Commonly used combinations.
     */
    ROT_180 = 3, // (FLIP_H | FLIP_V)
    ROT_270 = 7, // ((FLIP_H | FLIP_V) | ROT_90)
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
 * 
 * A dataspace is comprised of a number of fields.
 * 
 * Version
 * --------
 * The top 2 bits represent the revision of the field specification. This is
 * currently always 0.
 * 
 * 
 * bits    31-30 29                      -                          0
 *        +-----+----------------------------------------------------+
 * fields | Rev |            Revision specific fields                |
 *        +-----+----------------------------------------------------+
 * 
 * Field layout for version = 0:
 * ----------------------------
 * 
 * A dataspace is comprised of the following fields:
 *      Standard
 *      Transfer function
 *      Range
 * 
 * bits    31-30 29-27 26 -  22 21 -  16 15             -           0
 *        +-----+-----+--------+--------+----------------------------+
 * fields |  0  |Range|Transfer|Standard|    Legacy and custom       |
 *        +-----+-----+--------+--------+----------------------------+
 *          VV    RRR   TTTTT    SSSSSS    LLLLLLLL       LLLLLLLL
 * 
 * If range, transfer and standard fields are all 0 (e.g. top 16 bits are
 * all zeroes), the bottom 16 bits contain either a legacy dataspace value,
 * or a custom value.
 */
enum class Dataspace : int32_t {
    /**
     * Default-assumption data space, when not explicitly specified.
     * 
     * It is safest to assume the buffer is an image with sRGB primaries and
     * encoding ranges, but the consumer and/or the producer of the data may
     * simply be using defaults. No automatic gamma transform should be
     * expected, except for a possible display gamma transform when drawn to a
     * screen.
     */
    UNKNOWN = 0,
    /**
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
    ARBITRARY = 1,
    /**
     * Color-description aspects
     * 
     * The following aspects define various characteristics of the color
     * specification. These represent bitfields, so that a data space value
     * can specify each of them independently.
     */
    STANDARD_SHIFT = 16,
    /**
     * Standard aspect
     * 
     * Defines the chromaticity coordinates of the source primaries in terms of
     * the CIE 1931 definition of x and y specified in ISO 11664-1.
     */
    STANDARD_MASK = 4128768, // (63 << STANDARD_SHIFT)
    /**
     * Chromacity coordinates are unknown or are determined by the application.
     * Implementations shall use the following suggested standards:
     * 
     * All YCbCr formats: BT709 if size is 720p or larger (since most video
     *                    content is letterboxed this corresponds to width is
     *                    1280 or greater, or height is 720 or greater).
     *                    BT601_625 if size is smaller than 720p or is JPEG.
     * All RGB formats:   BT709.
     * 
     * For all other formats standard is undefined, and implementations should use
     * an appropriate standard for the data represented.
     */
    STANDARD_UNSPECIFIED = 0, // (0 << STANDARD_SHIFT)
    /**
     * Primaries:       x       y
     *  green           0.300   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     * 
     * Use the unadjusted KR = 0.2126, KB = 0.0722 luminance interpretation
     * for RGB conversion.
     */
    STANDARD_BT709 = 65536, // (1 << STANDARD_SHIFT)
    /**
     * Primaries:       x       y
     *  green           0.290   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     * 
     *  KR = 0.299, KB = 0.114. This adjusts the luminance interpretation
     *  for RGB conversion from the one purely determined by the primaries
     *  to minimize the color shift into RGB space that uses BT.709
     *  primaries.
     */
    STANDARD_BT601_625 = 131072, // (2 << STANDARD_SHIFT)
    /**
     * Primaries:       x       y
     *  green           0.290   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     * 
     * Use the unadjusted KR = 0.222, KB = 0.071 luminance interpretation
     * for RGB conversion.
     */
    STANDARD_BT601_625_UNADJUSTED = 196608, // (3 << STANDARD_SHIFT)
    /**
     * Primaries:       x       y
     *  green           0.310   0.595
     *  blue            0.155   0.070
     *  red             0.630   0.340
     *  white (D65)     0.3127  0.3290
     * 
     *  KR = 0.299, KB = 0.114. This adjusts the luminance interpretation
     *  for RGB conversion from the one purely determined by the primaries
     *  to minimize the color shift into RGB space that uses BT.709
     *  primaries.
     */
    STANDARD_BT601_525 = 262144, // (4 << STANDARD_SHIFT)
    /**
     * Primaries:       x       y
     *  green           0.310   0.595
     *  blue            0.155   0.070
     *  red             0.630   0.340
     *  white (D65)     0.3127  0.3290
     * 
     * Use the unadjusted KR = 0.212, KB = 0.087 luminance interpretation
     * for RGB conversion (as in SMPTE 240M).
     */
    STANDARD_BT601_525_UNADJUSTED = 327680, // (5 << STANDARD_SHIFT)
    /**
     * Primaries:       x       y
     *  green           0.170   0.797
     *  blue            0.131   0.046
     *  red             0.708   0.292
     *  white (D65)     0.3127  0.3290
     * 
     * Use the unadjusted KR = 0.2627, KB = 0.0593 luminance interpretation
     * for RGB conversion.
     */
    STANDARD_BT2020 = 393216, // (6 << STANDARD_SHIFT)
    /**
     * Primaries:       x       y
     *  green           0.170   0.797
     *  blue            0.131   0.046
     *  red             0.708   0.292
     *  white (D65)     0.3127  0.3290
     * 
     * Use the unadjusted KR = 0.2627, KB = 0.0593 luminance interpretation
     * for RGB conversion using the linear domain.
     */
    STANDARD_BT2020_CONSTANT_LUMINANCE = 458752, // (7 << STANDARD_SHIFT)
    /**
     * Primaries:       x      y
     *  green           0.21   0.71
     *  blue            0.14   0.08
     *  red             0.67   0.33
     *  white (C)       0.310  0.316
     * 
     * Use the unadjusted KR = 0.30, KB = 0.11 luminance interpretation
     * for RGB conversion.
     */
    STANDARD_BT470M = 524288, // (8 << STANDARD_SHIFT)
    /**
     * Primaries:       x       y
     *  green           0.243   0.692
     *  blue            0.145   0.049
     *  red             0.681   0.319
     *  white (C)       0.310   0.316
     * 
     * Use the unadjusted KR = 0.254, KB = 0.068 luminance interpretation
     * for RGB conversion.
     */
    STANDARD_FILM = 589824, // (9 << STANDARD_SHIFT)
    /**
     * SMPTE EG 432-1 and SMPTE RP 431-2. (DCI-P3)
     * Primaries:       x       y
     *  green           0.265   0.690
     *  blue            0.150   0.060
     *  red             0.680   0.320
     *  white (D65)     0.3127  0.3290
     */
    STANDARD_DCI_P3 = 655360, // (10 << STANDARD_SHIFT)
    /**
     * Adobe RGB
     * Primaries:       x       y
     *  green           0.210   0.710
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     */
    STANDARD_ADOBE_RGB = 720896, // (11 << STANDARD_SHIFT)
    TRANSFER_SHIFT = 22,
    /**
     * Transfer aspect
     * 
     * Transfer characteristics are the opto-electronic transfer characteristic
     * at the source as a function of linear optical intensity (luminance).
     * 
     * For digital signals, E corresponds to the recorded value. Normally, the
     * transfer function is applied in RGB space to each of the R, G and B
     * components independently. This may result in color shift that can be
     * minized by applying the transfer function in Lab space only for the L
     * component. Implementation may apply the transfer function in RGB space
     * for all pixel formats if desired.
     */
    TRANSFER_MASK = 130023424, // (31 << TRANSFER_SHIFT)
    /**
     * Transfer characteristics are unknown or are determined by the
     * application.
     * 
     * Implementations should use the following transfer functions:
     * 
     * For YCbCr formats: use TRANSFER_SMPTE_170M
     * For RGB formats: use TRANSFER_SRGB
     * 
     * For all other formats transfer function is undefined, and implementations
     * should use an appropriate standard for the data represented.
     */
    TRANSFER_UNSPECIFIED = 0, // (0 << TRANSFER_SHIFT)
    /**
     * Transfer characteristic curve:
     *  E = L
     *      L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *      E - corresponding electrical signal
     */
    TRANSFER_LINEAR = 4194304, // (1 << TRANSFER_SHIFT)
    /**
     * Transfer characteristic curve:
     * 
     * E = 1.055 * L^(1/2.4) - 0.055  for 0.0031308 <= L <= 1
     *   = 12.92 * L                  for 0 <= L < 0.0031308
     *     L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *     E - corresponding electrical signal
     */
    TRANSFER_SRGB = 8388608, // (2 << TRANSFER_SHIFT)
    /**
     * BT.601 525, BT.601 625, BT.709, BT.2020
     * 
     * Transfer characteristic curve:
     *  E = 1.099 * L ^ 0.45 - 0.099  for 0.018 <= L <= 1
     *    = 4.500 * L                 for 0 <= L < 0.018
     *      L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *      E - corresponding electrical signal
     */
    TRANSFER_SMPTE_170M = 12582912, // (3 << TRANSFER_SHIFT)
    /**
     * Assumed display gamma 2.2.
     * 
     * Transfer characteristic curve:
     *  E = L ^ (1/2.2)
     *      L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *      E - corresponding electrical signal
     */
    TRANSFER_GAMMA2_2 = 16777216, // (4 << TRANSFER_SHIFT)
    /**
     *  display gamma 2.6.
     * 
     * Transfer characteristic curve:
     *  E = L ^ (1/2.6)
     *      L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *      E - corresponding electrical signal
     */
    TRANSFER_GAMMA2_6 = 20971520, // (5 << TRANSFER_SHIFT)
    /**
     *  display gamma 2.8.
     * 
     * Transfer characteristic curve:
     *  E = L ^ (1/2.8)
     *      L - luminance of image 0 <= L <= 1 for conventional colorimetry
     *      E - corresponding electrical signal
     */
    TRANSFER_GAMMA2_8 = 25165824, // (6 << TRANSFER_SHIFT)
    /**
     * SMPTE ST 2084 (Dolby Perceptual Quantizer)
     * 
     * Transfer characteristic curve:
     *  E = ((c1 + c2 * L^n) / (1 + c3 * L^n)) ^ m
     *  c1 = c3 - c2 + 1 = 3424 / 4096 = 0.8359375
     *  c2 = 32 * 2413 / 4096 = 18.8515625
     *  c3 = 32 * 2392 / 4096 = 18.6875
     *  m = 128 * 2523 / 4096 = 78.84375
     *  n = 0.25 * 2610 / 4096 = 0.1593017578125
     *      L - luminance of image 0 <= L <= 1 for HDR colorimetry.
     *          L = 1 corresponds to 10000 cd/m2
     *      E - corresponding electrical signal
     */
    TRANSFER_ST2084 = 29360128, // (7 << TRANSFER_SHIFT)
    /**
     * ARIB STD-B67 Hybrid Log Gamma
     * 
     * Transfer characteristic curve:
     *  E = r * L^0.5                 for 0 <= L <= 1
     *    = a * ln(L - b) + c         for 1 < L
     *  a = 0.17883277
     *  b = 0.28466892
     *  c = 0.55991073
     *  r = 0.5
     *      L - luminance of image 0 <= L for HDR colorimetry. L = 1 corresponds
     *          to reference white level of 100 cd/m2
     *      E - corresponding electrical signal
     */
    TRANSFER_HLG = 33554432, // (8 << TRANSFER_SHIFT)
    RANGE_SHIFT = 27,
    /**
     * Range aspect
     * 
     * Defines the range of values corresponding to the unit range of 0-1.
     * This is defined for YCbCr only, but can be expanded to RGB space.
     */
    RANGE_MASK = 939524096, // (7 << RANGE_SHIFT)
    /**
     * Range is unknown or are determined by the application.  Implementations
     * shall use the following suggested ranges:
     * 
     * All YCbCr formats: limited range.
     * All RGB or RGBA formats (including RAW and Bayer): full range.
     * All Y formats: full range
     * 
     * For all other formats range is undefined, and implementations should use
     * an appropriate range for the data represented.
     */
    RANGE_UNSPECIFIED = 0, // (0 << RANGE_SHIFT)
    /**
     * Full range uses all values for Y, Cb and Cr from
     * 0 to 2^b-1, where b is the bit depth of the color format.
     */
    RANGE_FULL = 134217728, // (1 << RANGE_SHIFT)
    /**
     * Limited range uses values 16/256*2^b to 235/256*2^b for Y, and
     * 1/16*2^b to 15/16*2^b for Cb, Cr, R, G and B, where b is the bit depth of
     * the color format.
     * 
     * E.g. For 8-bit-depth formats:
     * Luma (Y) samples should range from 16 to 235, inclusive
     * Chroma (Cb, Cr) samples should range from 16 to 240, inclusive
     * 
     * For 10-bit-depth formats:
     * Luma (Y) samples should range from 64 to 940, inclusive
     * Chroma (Cb, Cr) samples should range from 64 to 960, inclusive
     */
    RANGE_LIMITED = 268435456, // (2 << RANGE_SHIFT)
    /**
     * Extended range is used for scRGB. Intended for use with
     * floating point pixel formats. [0.0 - 1.0] is the standard
     * sRGB space. Values outside the range 0.0 - 1.0 can encode
     * color outside the sRGB gamut.
     * Used to blend / merge multiple dataspaces on a single display.
     */
    RANGE_EXTENDED = 402653184, // (3 << RANGE_SHIFT)
    /**
     * Legacy dataspaces
     * 
     * 
     * sRGB linear encoding:
     * 
     * The red, green, and blue components are stored in sRGB space, but
     * are linear, not gamma-encoded.
     * The RGB primaries and the white point are the same as BT.709.
     * 
     * The values are encoded using the full range ([0,255] for 8-bit) for all
     * components.
     */
    SRGB_LINEAR = 512,
    V0_SRGB_LINEAR = 138477568, // ((STANDARD_BT709 | TRANSFER_LINEAR) | RANGE_FULL)
    /**
     * scRGB linear encoding:
     * 
     * The red, green, and blue components are stored in extended sRGB space,
     * but are linear, not gamma-encoded.
     * The RGB primaries and the white point are the same as BT.709.
     * 
     * The values are floating point.
     * A pixel value of 1.0, 1.0, 1.0 corresponds to sRGB white (D65) at 80 nits.
     * Values beyond the range [0.0 - 1.0] would correspond to other colors
     * spaces and/or HDR content.
     */
    V0_SCRGB_LINEAR = 406913024, // ((STANDARD_BT709 | TRANSFER_LINEAR) | RANGE_EXTENDED)
    /**
     * sRGB gamma encoding:
     * 
     * The red, green and blue components are stored in sRGB space, and
     * converted to linear space when read, using the SRGB transfer function
     * for each of the R, G and B components. When written, the inverse
     * transformation is performed.
     * 
     * The alpha component, if present, is always stored in linear space and
     * is left unmodified when read or written.
     * 
     * Use full range and BT.709 standard.
     */
    SRGB = 513,
    V0_SRGB = 142671872, // ((STANDARD_BT709 | TRANSFER_SRGB) | RANGE_FULL)
    /**
     * scRGB:
     * 
     * The red, green, and blue components are stored in extended sRGB space,
     * but are linear, not gamma-encoded.
     * The RGB primaries and the white point are the same as BT.709.
     * 
     * The values are floating point.
     * A pixel value of 1.0, 1.0, 1.0 corresponds to sRGB white (D65) at 80 nits.
     * Values beyond the range [0.0 - 1.0] would correspond to other colors
     * spaces and/or HDR content.
     */
    V0_SCRGB = 411107328, // ((STANDARD_BT709 | TRANSFER_SRGB) | RANGE_EXTENDED)
    /**
     * YCbCr Colorspaces
     * -----------------
     * 
     * Primaries are given using (x,y) coordinates in the CIE 1931 definition
     * of x and y specified by ISO 11664-1.
     * 
     * Transfer characteristics are the opto-electronic transfer characteristic
     * at the source as a function of linear optical intensity (luminance).
     * 
     * 
     * JPEG File Interchange Format (JFIF)
     * 
     * Same model as BT.601-625, but all values (Y, Cb, Cr) range from 0 to 255
     * 
     * Use full range, BT.601 transfer and BT.601_625 standard.
     */
    JFIF = 257,
    V0_JFIF = 146931712, // ((STANDARD_BT601_625 | TRANSFER_SMPTE_170M) | RANGE_FULL)
    /**
     * ITU-R Recommendation 601 (BT.601) - 625-line
     * 
     * Standard-definition television, 625 Lines (PAL)
     * 
     * Use limited range, BT.601 transfer and BT.601_625 standard.
     */
    BT601_625 = 258,
    V0_BT601_625 = 281149440, // ((STANDARD_BT601_625 | TRANSFER_SMPTE_170M) | RANGE_LIMITED)
    /**
     * ITU-R Recommendation 601 (BT.601) - 525-line
     * 
     * Standard-definition television, 525 Lines (NTSC)
     * 
     * Use limited range, BT.601 transfer and BT.601_525 standard.
     */
    BT601_525 = 259,
    V0_BT601_525 = 281280512, // ((STANDARD_BT601_525 | TRANSFER_SMPTE_170M) | RANGE_LIMITED)
    /**
     * ITU-R Recommendation 709 (BT.709)
     * 
     * High-definition television
     * 
     * Use limited range, BT.709 transfer and BT.709 standard.
     */
    BT709 = 260,
    V0_BT709 = 281083904, // ((STANDARD_BT709 | TRANSFER_SMPTE_170M) | RANGE_LIMITED)
    /**
     * SMPTE EG 432-1 and SMPTE RP 431-2.
     * 
     * Digital Cinema DCI-P3
     * 
     * Use full range, linear transfer and D65 DCI-P3 standard
     */
    DCI_P3_LINEAR = 139067392, // ((STANDARD_DCI_P3 | TRANSFER_LINEAR) | RANGE_FULL)
    /**
     * SMPTE EG 432-1 and SMPTE RP 431-2.
     * 
     * Digital Cinema DCI-P3
     * 
     * Use full range, gamma 2.6 transfer and D65 DCI-P3 standard
     * Note: Application is responsible for gamma encoding the data as
     * a 2.6 gamma encoding is not supported in HW.
     */
    DCI_P3 = 155844608, // ((STANDARD_DCI_P3 | TRANSFER_GAMMA2_6) | RANGE_FULL)
    /**
     * Display P3
     * 
     * Display P3 uses same primaries and white-point as DCI-P3
     * linear transfer function makes this the same as DCI_P3_LINEAR.
     */
    DISPLAY_P3_LINEAR = 139067392, // ((STANDARD_DCI_P3 | TRANSFER_LINEAR) | RANGE_FULL)
    /**
     * Display P3
     * 
     * Use same primaries and white-point as DCI-P3
     * but sRGB transfer function.
     */
    DISPLAY_P3 = 143261696, // ((STANDARD_DCI_P3 | TRANSFER_SRGB) | RANGE_FULL)
    /**
     * Adobe RGB
     * 
     * Use full range, gamma 2.2 transfer and Adobe RGB primaries
     * Note: Application is responsible for gamma encoding the data as
     * a 2.2 gamma encoding is not supported in HW.
     */
    ADOBE_RGB = 151715840, // ((STANDARD_ADOBE_RGB | TRANSFER_GAMMA2_2) | RANGE_FULL)
    /**
     * ITU-R Recommendation 2020 (BT.2020)
     * 
     * Ultra High-definition television
     * 
     * Use full range, linear transfer and BT2020 standard
     */
    BT2020_LINEAR = 138805248, // ((STANDARD_BT2020 | TRANSFER_LINEAR) | RANGE_FULL)
    /**
     * ITU-R Recommendation 2020 (BT.2020)
     * 
     * Ultra High-definition television
     * 
     * Use full range, BT.709 transfer and BT2020 standard
     */
    BT2020 = 147193856, // ((STANDARD_BT2020 | TRANSFER_SMPTE_170M) | RANGE_FULL)
    /**
     * ITU-R Recommendation 2020 (BT.2020)
     * 
     * Ultra High-definition television
     * 
     * Use full range, SMPTE 2084 (PQ) transfer and BT2020 standard
     */
    BT2020_PQ = 163971072, // ((STANDARD_BT2020 | TRANSFER_ST2084) | RANGE_FULL)
    /**
     * Data spaces for non-color formats
     * 
     * 
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
    DEPTH = 4096,
    /**
     * The buffer contains sensor events from sensor direct report.
     * This value is valid with formats:
     *    HAL_PIXEL_FORMAT_BLOB: an array of sensor event structure that forms
     *       a lock free queue. Format of sensor event structure is specified
     *       in Sensors HAL.
     */
    SENSOR = 4097,
};

/**
 * Color modes that may be supported by a display.
 * 
 * Definitions:
 * Rendering intent generally defines the goal in mapping a source (input)
 * color to a destination device color for a given color mode.
 * 
 *  It is important to keep in mind three cases where mapping may be applied:
 *  1. The source gamut is much smaller than the destination (display) gamut
 *  2. The source gamut is much larger than the destination gamut (this will
 *  ordinarily be handled using colorimetric rendering, below)
 *  3. The source and destination gamuts are roughly equal, although not
 *  completely overlapping
 *  Also, a common requirement for mappings is that skin tones should be
 *  preserved, or at least remain natural in appearance.
 * 
 *  Colorimetric Rendering Intent (All cases):
 *  Colorimetric indicates that colors should be preserved. In the case
 *  that the source gamut lies wholly within the destination gamut or is
 *  about the same (#1, #3), this will simply mean that no manipulations
 *  (no saturation boost, for example) are applied. In the case where some
 *  source colors lie outside the destination gamut (#2, #3), those will
 *  need to be mapped to colors that are within the destination gamut,
 *  while the already in-gamut colors remain unchanged.
 * 
 *  Non-colorimetric transforms can take many forms. There are no hard
 *  rules and it's left to the implementation to define.
 *  Two common intents are described below.
 * 
 *  Stretched-Gamut Enhancement Intent (Source < Destination):
 *  When the destination gamut is much larger than the source gamut (#1), the
 *  source primaries may be redefined to reflect the full extent of the
 *  destination space, or to reflect an intermediate gamut.
 *  Skin-tone preservation would likely be applied. An example might be sRGB
 *  input displayed on a DCI-P3 capable device, with skin-tone preservation.
 * 
 *  Within-Gamut Enhancement Intent (Source >= Destination):
 *  When the device (destination) gamut is not larger than the source gamut
 *  (#2 or #3), but the appearance of a larger gamut is desired, techniques
 *  such as saturation boost may be applied to the source colors. Skin-tone
 *  preservation may be applied. There is no unique method for within-gamut
 *  enhancement; it would be defined within a flexible color mode.
 * 
 */
enum class ColorMode : int32_t {
    /**
     * DEFAULT is the "native" gamut of the display.
     * White Point: Vendor/OEM defined
     * Panel Gamma: Vendor/OEM defined (typically 2.2)
     * Rendering Intent: Vendor/OEM defined (typically 'enhanced')
     */
    NATIVE = 0,
    /**
     * STANDARD_BT601_625 corresponds with display
     * settings that implement the ITU-R Recommendation BT.601
     * or Rec 601. Using 625 line version
     * Rendering Intent: Colorimetric
     * Primaries:
     *                  x       y
     *  green           0.290   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     * 
     *  KR = 0.299, KB = 0.114. This adjusts the luminance interpretation
     *  for RGB conversion from the one purely determined by the primaries
     *  to minimize the color shift into RGB space that uses BT.709
     *  primaries.
     * 
     * Gamma Correction (GC):
     * 
     *  if Vlinear < 0.018
     *    Vnonlinear = 4.500 * Vlinear
     *  else
     *    Vnonlinear = 1.099 * (Vlinear)^(0.45) – 0.099
     */
    STANDARD_BT601_625 = 1,
    /**
     * Primaries:
     *                  x       y
     *  green           0.290   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     * 
     *  Use the unadjusted KR = 0.222, KB = 0.071 luminance interpretation
     *  for RGB conversion.
     * 
     * Gamma Correction (GC):
     * 
     *  if Vlinear < 0.018
     *    Vnonlinear = 4.500 * Vlinear
     *  else
     *    Vnonlinear = 1.099 * (Vlinear)^(0.45) – 0.099
     */
    STANDARD_BT601_625_UNADJUSTED = 2,
    /**
     * Primaries:
     *                  x       y
     *  green           0.310   0.595
     *  blue            0.155   0.070
     *  red             0.630   0.340
     *  white (D65)     0.3127  0.3290
     * 
     *  KR = 0.299, KB = 0.114. This adjusts the luminance interpretation
     *  for RGB conversion from the one purely determined by the primaries
     *  to minimize the color shift into RGB space that uses BT.709
     *  primaries.
     * 
     * Gamma Correction (GC):
     * 
     *  if Vlinear < 0.018
     *    Vnonlinear = 4.500 * Vlinear
     *  else
     *    Vnonlinear = 1.099 * (Vlinear)^(0.45) – 0.099
     */
    STANDARD_BT601_525 = 3,
    /**
     * Primaries:
     *                  x       y
     *  green           0.310   0.595
     *  blue            0.155   0.070
     *  red             0.630   0.340
     *  white (D65)     0.3127  0.3290
     * 
     *  Use the unadjusted KR = 0.212, KB = 0.087 luminance interpretation
     *  for RGB conversion (as in SMPTE 240M).
     * 
     * Gamma Correction (GC):
     * 
     *  if Vlinear < 0.018
     *    Vnonlinear = 4.500 * Vlinear
     *  else
     *    Vnonlinear = 1.099 * (Vlinear)^(0.45) – 0.099
     */
    STANDARD_BT601_525_UNADJUSTED = 4,
    /**
     * REC709 corresponds with display settings that implement
     * the ITU-R Recommendation BT.709 / Rec. 709 for high-definition television.
     * Rendering Intent: Colorimetric
     * Primaries:
     *                  x       y
     *  green           0.300   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     * 
     * HDTV REC709 Inverse Gamma Correction (IGC): V represents normalized
     * (with [0 to 1] range) value of R, G, or B.
     * 
     *  if Vnonlinear < 0.081
     *    Vlinear = Vnonlinear / 4.5
     *  else
     *    Vlinear = ((Vnonlinear + 0.099) / 1.099) ^ (1/0.45)
     * 
     * HDTV REC709 Gamma Correction (GC):
     * 
     *  if Vlinear < 0.018
     *    Vnonlinear = 4.5 * Vlinear
     *  else
     *    Vnonlinear = 1.099 * (Vlinear) ^ 0.45 – 0.099
     */
    STANDARD_BT709 = 5,
    /**
     * DCI_P3 corresponds with display settings that implement
     * SMPTE EG 432-1 and SMPTE RP 431-2
     * Rendering Intent: Colorimetric
     * Primaries:
     *                  x       y
     *  green           0.265   0.690
     *  blue            0.150   0.060
     *  red             0.680   0.320
     *  white (D65)     0.3127  0.3290
     * 
     * Gamma: 2.6
     */
    DCI_P3 = 6,
    /**
     * SRGB corresponds with display settings that implement
     * the sRGB color space. Uses the same primaries as ITU-R Recommendation
     * BT.709
     * Rendering Intent: Colorimetric
     * Primaries:
     *                  x       y
     *  green           0.300   0.600
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     * 
     * PC/Internet (sRGB) Inverse Gamma Correction (IGC):
     * 
     *  if Vnonlinear ≤ 0.03928
     *    Vlinear = Vnonlinear / 12.92
     *  else
     *    Vlinear = ((Vnonlinear + 0.055)/1.055) ^ 2.4
     * 
     * PC/Internet (sRGB) Gamma Correction (GC):
     * 
     *  if Vlinear ≤ 0.0031308
     *    Vnonlinear = 12.92 * Vlinear
     *  else
     *    Vnonlinear = 1.055 * (Vlinear)^(1/2.4) – 0.055
     */
    SRGB = 7,
    /**
     * ADOBE_RGB corresponds with the RGB color space developed
     * by Adobe Systems, Inc. in 1998.
     * Rendering Intent: Colorimetric
     * Primaries:
     *                  x       y
     *  green           0.210   0.710
     *  blue            0.150   0.060
     *  red             0.640   0.330
     *  white (D65)     0.3127  0.3290
     * 
     * Gamma: 2.2
     */
    ADOBE_RGB = 8,
    /**
     * DISPLAY_P3 is a color space that uses the DCI_P3 primaries,
     * the D65 white point and the SRGB transfer functions.
     * Rendering Intent: Colorimetric
     * Primaries:
     *                  x       y
     *  green           0.265   0.690
     *  blue            0.150   0.060
     *  red             0.680   0.320
     *  white (D65)     0.3127  0.3290
     * 
     * PC/Internet (sRGB) Gamma Correction (GC):
     * 
     *  if Vlinear ≤ 0.0030186
     *    Vnonlinear = 12.92 * Vlinear
     *  else
     *    Vnonlinear = 1.055 * (Vlinear)^(1/2.4) – 0.055
     * 
     * Note: In most cases sRGB transfer function will be fine.
     */
    DISPLAY_P3 = 9,
};

/**
 * Color transforms that may be applied by hardware composer to the whole
 * display.
 */
enum class ColorTransform : int32_t {
    /**
     * Applies no transform to the output color  */
    IDENTITY = 0,
    /**
     * Applies an arbitrary transform defined by a 4x4 affine matrix  */
    ARBITRARY_MATRIX = 1,
    /**
     * Applies a transform that inverts the value or luminance of the color, but
     * does not modify hue or saturation  */
    VALUE_INVERSE = 2,
    /**
     * Applies a transform that maps all colors to shades of gray  */
    GRAYSCALE = 3,
    /**
     * Applies a transform which corrects for protanopic color blindness  */
    CORRECT_PROTANOPIA = 4,
    /**
     * Applies a transform which corrects for deuteranopic color blindness  */
    CORRECT_DEUTERANOPIA = 5,
    /**
     * Applies a transform which corrects for tritanopic color blindness  */
    CORRECT_TRITANOPIA = 6,
};

/**
 * Supported HDR formats. Must be kept in sync with equivalents in Display.java.
 */
enum class Hdr : int32_t {
    /**
     * Device supports Dolby Vision HDR  */
    DOLBY_VISION = 1,
    /**
     * Device supports HDR10  */
    HDR10 = 2,
    /**
     * Device supports hybrid log-gamma HDR  */
    HLG = 3,
};

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::PixelFormat lhs, const ::android::hardware::graphics::common::V1_0::PixelFormat rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::PixelFormat rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::PixelFormat lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::PixelFormat lhs, const ::android::hardware::graphics::common::V1_0::PixelFormat rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::PixelFormat rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::PixelFormat lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_0::PixelFormat e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_0::PixelFormat e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_0::PixelFormat>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_0::PixelFormat> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_8888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_8888)) {
        os += (first ? "" : " | ");
        os += "RGBA_8888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_8888;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RGBX_8888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RGBX_8888)) {
        os += (first ? "" : " | ");
        os += "RGBX_8888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RGBX_8888;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RGB_888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RGB_888)) {
        os += (first ? "" : " | ");
        os += "RGB_888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RGB_888;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RGB_565) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RGB_565)) {
        os += (first ? "" : " | ");
        os += "RGB_565";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RGB_565;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::BGRA_8888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::BGRA_8888)) {
        os += (first ? "" : " | ");
        os += "BGRA_8888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::BGRA_8888;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_SP) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_SP)) {
        os += (first ? "" : " | ");
        os += "YCBCR_422_SP";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_SP;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::YCRCB_420_SP) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::YCRCB_420_SP)) {
        os += (first ? "" : " | ");
        os += "YCRCB_420_SP";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::YCRCB_420_SP;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_I) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_I)) {
        os += (first ? "" : " | ");
        os += "YCBCR_422_I";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_I;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_FP16) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_FP16)) {
        os += (first ? "" : " | ");
        os += "RGBA_FP16";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_FP16;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RAW16) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RAW16)) {
        os += (first ? "" : " | ");
        os += "RAW16";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RAW16;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::BLOB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::BLOB)) {
        os += (first ? "" : " | ");
        os += "BLOB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::BLOB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::IMPLEMENTATION_DEFINED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::IMPLEMENTATION_DEFINED)) {
        os += (first ? "" : " | ");
        os += "IMPLEMENTATION_DEFINED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::IMPLEMENTATION_DEFINED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_420_888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_420_888)) {
        os += (first ? "" : " | ");
        os += "YCBCR_420_888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_420_888;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RAW_OPAQUE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RAW_OPAQUE)) {
        os += (first ? "" : " | ");
        os += "RAW_OPAQUE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RAW_OPAQUE;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RAW10) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RAW10)) {
        os += (first ? "" : " | ");
        os += "RAW10";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RAW10;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RAW12) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RAW12)) {
        os += (first ? "" : " | ");
        os += "RAW12";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RAW12;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_1010102) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_1010102)) {
        os += (first ? "" : " | ");
        os += "RGBA_1010102";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_1010102;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::Y8) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::Y8)) {
        os += (first ? "" : " | ");
        os += "Y8";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::Y8;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::Y16) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::Y16)) {
        os += (first ? "" : " | ");
        os += "Y16";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::Y16;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::PixelFormat::YV12) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::PixelFormat::YV12)) {
        os += (first ? "" : " | ");
        os += "YV12";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::PixelFormat::YV12;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_0::PixelFormat o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_8888) {
        return "RGBA_8888";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RGBX_8888) {
        return "RGBX_8888";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RGB_888) {
        return "RGB_888";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RGB_565) {
        return "RGB_565";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::BGRA_8888) {
        return "BGRA_8888";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_SP) {
        return "YCBCR_422_SP";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::YCRCB_420_SP) {
        return "YCRCB_420_SP";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_I) {
        return "YCBCR_422_I";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_FP16) {
        return "RGBA_FP16";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RAW16) {
        return "RAW16";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::BLOB) {
        return "BLOB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::IMPLEMENTATION_DEFINED) {
        return "IMPLEMENTATION_DEFINED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_420_888) {
        return "YCBCR_420_888";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RAW_OPAQUE) {
        return "RAW_OPAQUE";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RAW10) {
        return "RAW10";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RAW12) {
        return "RAW12";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_1010102) {
        return "RGBA_1010102";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::Y8) {
        return "Y8";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::Y16) {
        return "Y16";
    }
    if (o == ::android::hardware::graphics::common::V1_0::PixelFormat::YV12) {
        return "YV12";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

constexpr uint64_t operator|(const ::android::hardware::graphics::common::V1_0::BufferUsage lhs, const ::android::hardware::graphics::common::V1_0::BufferUsage rhs) {
    return static_cast<uint64_t>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

constexpr uint64_t operator|(const uint64_t lhs, const ::android::hardware::graphics::common::V1_0::BufferUsage rhs) {
    return static_cast<uint64_t>(lhs | static_cast<uint64_t>(rhs));
}

constexpr uint64_t operator|(const ::android::hardware::graphics::common::V1_0::BufferUsage lhs, const uint64_t rhs) {
    return static_cast<uint64_t>(static_cast<uint64_t>(lhs) | rhs);
}

constexpr uint64_t operator&(const ::android::hardware::graphics::common::V1_0::BufferUsage lhs, const ::android::hardware::graphics::common::V1_0::BufferUsage rhs) {
    return static_cast<uint64_t>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

constexpr uint64_t operator&(const uint64_t lhs, const ::android::hardware::graphics::common::V1_0::BufferUsage rhs) {
    return static_cast<uint64_t>(lhs & static_cast<uint64_t>(rhs));
}

constexpr uint64_t operator&(const ::android::hardware::graphics::common::V1_0::BufferUsage lhs, const uint64_t rhs) {
    return static_cast<uint64_t>(static_cast<uint64_t>(lhs) & rhs);
}

constexpr uint64_t &operator|=(uint64_t& v, const ::android::hardware::graphics::common::V1_0::BufferUsage e) {
    v |= static_cast<uint64_t>(e);
    return v;
}

constexpr uint64_t &operator&=(uint64_t& v, const ::android::hardware::graphics::common::V1_0::BufferUsage e) {
    v &= static_cast<uint64_t>(e);
    return v;
}

template<typename>
static inline std::string toString(uint64_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_0::BufferUsage>(uint64_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_0::BufferUsage> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_MASK) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_MASK)) {
        os += (first ? "" : " | ");
        os += "CPU_READ_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_NEVER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_NEVER)) {
        os += (first ? "" : " | ");
        os += "CPU_READ_NEVER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_NEVER;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_RARELY) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_RARELY)) {
        os += (first ? "" : " | ");
        os += "CPU_READ_RARELY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_RARELY;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_OFTEN) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_OFTEN)) {
        os += (first ? "" : " | ");
        os += "CPU_READ_OFTEN";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_OFTEN;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_MASK) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_MASK)) {
        os += (first ? "" : " | ");
        os += "CPU_WRITE_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_NEVER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_NEVER)) {
        os += (first ? "" : " | ");
        os += "CPU_WRITE_NEVER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_NEVER;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_RARELY) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_RARELY)) {
        os += (first ? "" : " | ");
        os += "CPU_WRITE_RARELY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_RARELY;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_OFTEN) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_OFTEN)) {
        os += (first ? "" : " | ");
        os += "CPU_WRITE_OFTEN";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_OFTEN;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_TEXTURE) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::GPU_TEXTURE)) {
        os += (first ? "" : " | ");
        os += "GPU_TEXTURE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_TEXTURE;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_RENDER_TARGET) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::GPU_RENDER_TARGET)) {
        os += (first ? "" : " | ");
        os += "GPU_RENDER_TARGET";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_RENDER_TARGET;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_OVERLAY) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_OVERLAY)) {
        os += (first ? "" : " | ");
        os += "COMPOSER_OVERLAY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_OVERLAY;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CLIENT_TARGET) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CLIENT_TARGET)) {
        os += (first ? "" : " | ");
        os += "COMPOSER_CLIENT_TARGET";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CLIENT_TARGET;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::PROTECTED) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::PROTECTED)) {
        os += (first ? "" : " | ");
        os += "PROTECTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::PROTECTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CURSOR) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CURSOR)) {
        os += (first ? "" : " | ");
        os += "COMPOSER_CURSOR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CURSOR;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_ENCODER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_ENCODER)) {
        os += (first ? "" : " | ");
        os += "VIDEO_ENCODER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_ENCODER;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_OUTPUT) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_OUTPUT)) {
        os += (first ? "" : " | ");
        os += "CAMERA_OUTPUT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_OUTPUT;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_INPUT) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_INPUT)) {
        os += (first ? "" : " | ");
        os += "CAMERA_INPUT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_INPUT;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::RENDERSCRIPT) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::RENDERSCRIPT)) {
        os += (first ? "" : " | ");
        os += "RENDERSCRIPT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::RENDERSCRIPT;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_DECODER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_DECODER)) {
        os += (first ? "" : " | ");
        os += "VIDEO_DECODER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_DECODER;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::SENSOR_DIRECT_DATA) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::SENSOR_DIRECT_DATA)) {
        os += (first ? "" : " | ");
        os += "SENSOR_DIRECT_DATA";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::SENSOR_DIRECT_DATA;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_DATA_BUFFER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::GPU_DATA_BUFFER)) {
        os += (first ? "" : " | ");
        os += "GPU_DATA_BUFFER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_DATA_BUFFER;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK)) {
        os += (first ? "" : " | ");
        os += "VENDOR_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK_HI) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK_HI)) {
        os += (first ? "" : " | ");
        os += "VENDOR_MASK_HI";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK_HI;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_0::BufferUsage o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_MASK) {
        return "CPU_READ_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_NEVER) {
        return "CPU_READ_NEVER";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_RARELY) {
        return "CPU_READ_RARELY";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_OFTEN) {
        return "CPU_READ_OFTEN";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_MASK) {
        return "CPU_WRITE_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_NEVER) {
        return "CPU_WRITE_NEVER";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_RARELY) {
        return "CPU_WRITE_RARELY";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_OFTEN) {
        return "CPU_WRITE_OFTEN";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_TEXTURE) {
        return "GPU_TEXTURE";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_RENDER_TARGET) {
        return "GPU_RENDER_TARGET";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_OVERLAY) {
        return "COMPOSER_OVERLAY";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CLIENT_TARGET) {
        return "COMPOSER_CLIENT_TARGET";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::PROTECTED) {
        return "PROTECTED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CURSOR) {
        return "COMPOSER_CURSOR";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_ENCODER) {
        return "VIDEO_ENCODER";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_OUTPUT) {
        return "CAMERA_OUTPUT";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_INPUT) {
        return "CAMERA_INPUT";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::RENDERSCRIPT) {
        return "RENDERSCRIPT";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_DECODER) {
        return "VIDEO_DECODER";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::SENSOR_DIRECT_DATA) {
        return "SENSOR_DIRECT_DATA";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_DATA_BUFFER) {
        return "GPU_DATA_BUFFER";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK) {
        return "VENDOR_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK_HI) {
        return "VENDOR_MASK_HI";
    }
    std::string os;
    os += toHexString(static_cast<uint64_t>(o));
    return os;
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::Transform lhs, const ::android::hardware::graphics::common::V1_0::Transform rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::Transform rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::Transform lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::Transform lhs, const ::android::hardware::graphics::common::V1_0::Transform rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::Transform rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::Transform lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_0::Transform e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_0::Transform e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_0::Transform>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_0::Transform> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_0::Transform::FLIP_H) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Transform::FLIP_H)) {
        os += (first ? "" : " | ");
        os += "FLIP_H";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Transform::FLIP_H;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Transform::FLIP_V) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Transform::FLIP_V)) {
        os += (first ? "" : " | ");
        os += "FLIP_V";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Transform::FLIP_V;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Transform::ROT_90) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Transform::ROT_90)) {
        os += (first ? "" : " | ");
        os += "ROT_90";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Transform::ROT_90;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Transform::ROT_180) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Transform::ROT_180)) {
        os += (first ? "" : " | ");
        os += "ROT_180";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Transform::ROT_180;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Transform::ROT_270) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Transform::ROT_270)) {
        os += (first ? "" : " | ");
        os += "ROT_270";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Transform::ROT_270;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_0::Transform o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_0::Transform::FLIP_H) {
        return "FLIP_H";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Transform::FLIP_V) {
        return "FLIP_V";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Transform::ROT_90) {
        return "ROT_90";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Transform::ROT_180) {
        return "ROT_180";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Transform::ROT_270) {
        return "ROT_270";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::Dataspace lhs, const ::android::hardware::graphics::common::V1_0::Dataspace rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::Dataspace rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::Dataspace lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::Dataspace lhs, const ::android::hardware::graphics::common::V1_0::Dataspace rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::Dataspace rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::Dataspace lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_0::Dataspace e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_0::Dataspace e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_0::Dataspace>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_0::Dataspace> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::UNKNOWN) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::UNKNOWN)) {
        os += (first ? "" : " | ");
        os += "UNKNOWN";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::UNKNOWN;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::ARBITRARY) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::ARBITRARY)) {
        os += (first ? "" : " | ");
        os += "ARBITRARY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::ARBITRARY;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_SHIFT) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_SHIFT)) {
        os += (first ? "" : " | ");
        os += "STANDARD_SHIFT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_SHIFT;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_MASK) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_MASK)) {
        os += (first ? "" : " | ");
        os += "STANDARD_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_UNSPECIFIED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_UNSPECIFIED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_UNSPECIFIED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_UNSPECIFIED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT709) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT709)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT709";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT709;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_625";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625_UNADJUSTED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625_UNADJUSTED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_625_UNADJUSTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625_UNADJUSTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_525";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525_UNADJUSTED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525_UNADJUSTED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_525_UNADJUSTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525_UNADJUSTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT2020";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT2020_CONSTANT_LUMINANCE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT470M) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT470M)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT470M";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT470M;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_FILM) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_FILM)) {
        os += (first ? "" : " | ");
        os += "STANDARD_FILM";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_FILM;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_DCI_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_DCI_P3)) {
        os += (first ? "" : " | ");
        os += "STANDARD_DCI_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_DCI_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_ADOBE_RGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_ADOBE_RGB)) {
        os += (first ? "" : " | ");
        os += "STANDARD_ADOBE_RGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_ADOBE_RGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SHIFT) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SHIFT)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_SHIFT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SHIFT;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_MASK) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_MASK)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_UNSPECIFIED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_UNSPECIFIED)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_UNSPECIFIED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_UNSPECIFIED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_LINEAR)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SRGB)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_SRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SMPTE_170M) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SMPTE_170M)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_SMPTE_170M";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SMPTE_170M;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_2) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_2)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_GAMMA2_2";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_2;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_6) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_6)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_GAMMA2_6";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_6;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_8) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_8)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_GAMMA2_8";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_8;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_ST2084) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_ST2084)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_ST2084";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_ST2084;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_HLG) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_HLG)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_HLG";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_HLG;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_SHIFT) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::RANGE_SHIFT)) {
        os += (first ? "" : " | ");
        os += "RANGE_SHIFT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_SHIFT;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_MASK) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::RANGE_MASK)) {
        os += (first ? "" : " | ");
        os += "RANGE_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_UNSPECIFIED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::RANGE_UNSPECIFIED)) {
        os += (first ? "" : " | ");
        os += "RANGE_UNSPECIFIED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_UNSPECIFIED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_FULL) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::RANGE_FULL)) {
        os += (first ? "" : " | ");
        os += "RANGE_FULL";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_FULL;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_LIMITED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::RANGE_LIMITED)) {
        os += (first ? "" : " | ");
        os += "RANGE_LIMITED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_LIMITED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_EXTENDED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::RANGE_EXTENDED)) {
        os += (first ? "" : " | ");
        os += "RANGE_EXTENDED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_EXTENDED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::SRGB_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::SRGB_LINEAR)) {
        os += (first ? "" : " | ");
        os += "SRGB_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::SRGB_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB_LINEAR)) {
        os += (first ? "" : " | ");
        os += "V0_SRGB_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB_LINEAR)) {
        os += (first ? "" : " | ");
        os += "V0_SCRGB_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::SRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::SRGB)) {
        os += (first ? "" : " | ");
        os += "SRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::SRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB)) {
        os += (first ? "" : " | ");
        os += "V0_SRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB)) {
        os += (first ? "" : " | ");
        os += "V0_SCRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::JFIF) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::JFIF)) {
        os += (first ? "" : " | ");
        os += "JFIF";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::JFIF;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::V0_JFIF) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::V0_JFIF)) {
        os += (first ? "" : " | ");
        os += "V0_JFIF";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::V0_JFIF;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::BT601_625) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::BT601_625)) {
        os += (first ? "" : " | ");
        os += "BT601_625";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::BT601_625;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_625) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_625)) {
        os += (first ? "" : " | ");
        os += "V0_BT601_625";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_625;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::BT601_525) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::BT601_525)) {
        os += (first ? "" : " | ");
        os += "BT601_525";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::BT601_525;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_525) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_525)) {
        os += (first ? "" : " | ");
        os += "V0_BT601_525";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_525;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::BT709) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::BT709)) {
        os += (first ? "" : " | ");
        os += "BT709";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::BT709;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT709) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::V0_BT709)) {
        os += (first ? "" : " | ");
        os += "V0_BT709";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT709;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3_LINEAR)) {
        os += (first ? "" : " | ");
        os += "DCI_P3_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3)) {
        os += (first ? "" : " | ");
        os += "DCI_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3_LINEAR)) {
        os += (first ? "" : " | ");
        os += "DISPLAY_P3_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3)) {
        os += (first ? "" : " | ");
        os += "DISPLAY_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::ADOBE_RGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::ADOBE_RGB)) {
        os += (first ? "" : " | ");
        os += "ADOBE_RGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::ADOBE_RGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::BT2020_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::BT2020_LINEAR)) {
        os += (first ? "" : " | ");
        os += "BT2020_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::BT2020_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::BT2020) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::BT2020)) {
        os += (first ? "" : " | ");
        os += "BT2020";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::BT2020;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::BT2020_PQ) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::BT2020_PQ)) {
        os += (first ? "" : " | ");
        os += "BT2020_PQ";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::BT2020_PQ;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::DEPTH) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::DEPTH)) {
        os += (first ? "" : " | ");
        os += "DEPTH";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::DEPTH;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Dataspace::SENSOR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Dataspace::SENSOR)) {
        os += (first ? "" : " | ");
        os += "SENSOR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Dataspace::SENSOR;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_0::Dataspace o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::UNKNOWN) {
        return "UNKNOWN";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::ARBITRARY) {
        return "ARBITRARY";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_SHIFT) {
        return "STANDARD_SHIFT";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_MASK) {
        return "STANDARD_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_UNSPECIFIED) {
        return "STANDARD_UNSPECIFIED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT709) {
        return "STANDARD_BT709";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625) {
        return "STANDARD_BT601_625";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625_UNADJUSTED) {
        return "STANDARD_BT601_625_UNADJUSTED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525) {
        return "STANDARD_BT601_525";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525_UNADJUSTED) {
        return "STANDARD_BT601_525_UNADJUSTED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020) {
        return "STANDARD_BT2020";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE) {
        return "STANDARD_BT2020_CONSTANT_LUMINANCE";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT470M) {
        return "STANDARD_BT470M";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_FILM) {
        return "STANDARD_FILM";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_DCI_P3) {
        return "STANDARD_DCI_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_ADOBE_RGB) {
        return "STANDARD_ADOBE_RGB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SHIFT) {
        return "TRANSFER_SHIFT";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_MASK) {
        return "TRANSFER_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_UNSPECIFIED) {
        return "TRANSFER_UNSPECIFIED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_LINEAR) {
        return "TRANSFER_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SRGB) {
        return "TRANSFER_SRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SMPTE_170M) {
        return "TRANSFER_SMPTE_170M";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_2) {
        return "TRANSFER_GAMMA2_2";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_6) {
        return "TRANSFER_GAMMA2_6";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_8) {
        return "TRANSFER_GAMMA2_8";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_ST2084) {
        return "TRANSFER_ST2084";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_HLG) {
        return "TRANSFER_HLG";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_SHIFT) {
        return "RANGE_SHIFT";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_MASK) {
        return "RANGE_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_UNSPECIFIED) {
        return "RANGE_UNSPECIFIED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_FULL) {
        return "RANGE_FULL";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_LIMITED) {
        return "RANGE_LIMITED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_EXTENDED) {
        return "RANGE_EXTENDED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::SRGB_LINEAR) {
        return "SRGB_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB_LINEAR) {
        return "V0_SRGB_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB_LINEAR) {
        return "V0_SCRGB_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::SRGB) {
        return "SRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB) {
        return "V0_SRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB) {
        return "V0_SCRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::JFIF) {
        return "JFIF";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::V0_JFIF) {
        return "V0_JFIF";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::BT601_625) {
        return "BT601_625";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_625) {
        return "V0_BT601_625";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::BT601_525) {
        return "BT601_525";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_525) {
        return "V0_BT601_525";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::BT709) {
        return "BT709";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT709) {
        return "V0_BT709";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3_LINEAR) {
        return "DCI_P3_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3) {
        return "DCI_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3_LINEAR) {
        return "DISPLAY_P3_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3) {
        return "DISPLAY_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::ADOBE_RGB) {
        return "ADOBE_RGB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::BT2020_LINEAR) {
        return "BT2020_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::BT2020) {
        return "BT2020";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::BT2020_PQ) {
        return "BT2020_PQ";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::DEPTH) {
        return "DEPTH";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Dataspace::SENSOR) {
        return "SENSOR";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::ColorMode lhs, const ::android::hardware::graphics::common::V1_0::ColorMode rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::ColorMode rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::ColorMode lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::ColorMode lhs, const ::android::hardware::graphics::common::V1_0::ColorMode rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::ColorMode rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::ColorMode lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_0::ColorMode e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_0::ColorMode e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_0::ColorMode>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_0::ColorMode> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::NATIVE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::NATIVE)) {
        os += (first ? "" : " | ");
        os += "NATIVE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::NATIVE;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_625";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625_UNADJUSTED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625_UNADJUSTED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_625_UNADJUSTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625_UNADJUSTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_525";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525_UNADJUSTED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525_UNADJUSTED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_525_UNADJUSTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525_UNADJUSTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT709) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT709)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT709";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT709;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::DCI_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::DCI_P3)) {
        os += (first ? "" : " | ");
        os += "DCI_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::DCI_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::SRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::SRGB)) {
        os += (first ? "" : " | ");
        os += "SRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::SRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::ADOBE_RGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::ADOBE_RGB)) {
        os += (first ? "" : " | ");
        os += "ADOBE_RGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::ADOBE_RGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorMode::DISPLAY_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorMode::DISPLAY_P3)) {
        os += (first ? "" : " | ");
        os += "DISPLAY_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorMode::DISPLAY_P3;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_0::ColorMode o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::NATIVE) {
        return "NATIVE";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625) {
        return "STANDARD_BT601_625";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625_UNADJUSTED) {
        return "STANDARD_BT601_625_UNADJUSTED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525) {
        return "STANDARD_BT601_525";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525_UNADJUSTED) {
        return "STANDARD_BT601_525_UNADJUSTED";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT709) {
        return "STANDARD_BT709";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::DCI_P3) {
        return "DCI_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::SRGB) {
        return "SRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::ADOBE_RGB) {
        return "ADOBE_RGB";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorMode::DISPLAY_P3) {
        return "DISPLAY_P3";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::ColorTransform lhs, const ::android::hardware::graphics::common::V1_0::ColorTransform rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::ColorTransform rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::ColorTransform lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::ColorTransform lhs, const ::android::hardware::graphics::common::V1_0::ColorTransform rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::ColorTransform rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::ColorTransform lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_0::ColorTransform e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_0::ColorTransform e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_0::ColorTransform>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_0::ColorTransform> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_0::ColorTransform::IDENTITY) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorTransform::IDENTITY)) {
        os += (first ? "" : " | ");
        os += "IDENTITY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorTransform::IDENTITY;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorTransform::ARBITRARY_MATRIX) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorTransform::ARBITRARY_MATRIX)) {
        os += (first ? "" : " | ");
        os += "ARBITRARY_MATRIX";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorTransform::ARBITRARY_MATRIX;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorTransform::VALUE_INVERSE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorTransform::VALUE_INVERSE)) {
        os += (first ? "" : " | ");
        os += "VALUE_INVERSE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorTransform::VALUE_INVERSE;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorTransform::GRAYSCALE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorTransform::GRAYSCALE)) {
        os += (first ? "" : " | ");
        os += "GRAYSCALE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorTransform::GRAYSCALE;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_PROTANOPIA) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_PROTANOPIA)) {
        os += (first ? "" : " | ");
        os += "CORRECT_PROTANOPIA";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_PROTANOPIA;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_DEUTERANOPIA) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_DEUTERANOPIA)) {
        os += (first ? "" : " | ");
        os += "CORRECT_DEUTERANOPIA";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_DEUTERANOPIA;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_TRITANOPIA) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_TRITANOPIA)) {
        os += (first ? "" : " | ");
        os += "CORRECT_TRITANOPIA";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_TRITANOPIA;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_0::ColorTransform o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_0::ColorTransform::IDENTITY) {
        return "IDENTITY";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorTransform::ARBITRARY_MATRIX) {
        return "ARBITRARY_MATRIX";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorTransform::VALUE_INVERSE) {
        return "VALUE_INVERSE";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorTransform::GRAYSCALE) {
        return "GRAYSCALE";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_PROTANOPIA) {
        return "CORRECT_PROTANOPIA";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_DEUTERANOPIA) {
        return "CORRECT_DEUTERANOPIA";
    }
    if (o == ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_TRITANOPIA) {
        return "CORRECT_TRITANOPIA";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::Hdr lhs, const ::android::hardware::graphics::common::V1_0::Hdr rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::Hdr rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_0::Hdr lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::Hdr lhs, const ::android::hardware::graphics::common::V1_0::Hdr rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_0::Hdr rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_0::Hdr lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_0::Hdr e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_0::Hdr e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_0::Hdr>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_0::Hdr> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_0::Hdr::DOLBY_VISION) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Hdr::DOLBY_VISION)) {
        os += (first ? "" : " | ");
        os += "DOLBY_VISION";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Hdr::DOLBY_VISION;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Hdr::HDR10) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Hdr::HDR10)) {
        os += (first ? "" : " | ");
        os += "HDR10";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Hdr::HDR10;
    }
    if ((o & ::android::hardware::graphics::common::V1_0::Hdr::HLG) == static_cast<int32_t>(::android::hardware::graphics::common::V1_0::Hdr::HLG)) {
        os += (first ? "" : " | ");
        os += "HLG";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_0::Hdr::HLG;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_0::Hdr o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_0::Hdr::DOLBY_VISION) {
        return "DOLBY_VISION";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Hdr::HDR10) {
        return "HDR10";
    }
    if (o == ::android::hardware::graphics::common::V1_0::Hdr::HLG) {
        return "HLG";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}


}  // namespace V1_0
}  // namespace common
}  // namespace graphics
}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_0::PixelFormat>
{
    const ::android::hardware::graphics::common::V1_0::PixelFormat* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_0::PixelFormat* end() { return begin() + 20; }
    private:
    static const ::android::hardware::graphics::common::V1_0::PixelFormat* static_begin() {
        static const ::android::hardware::graphics::common::V1_0::PixelFormat kVals[20] {
            ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_8888,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RGBX_8888,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RGB_888,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RGB_565,
            ::android::hardware::graphics::common::V1_0::PixelFormat::BGRA_8888,
            ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_SP,
            ::android::hardware::graphics::common::V1_0::PixelFormat::YCRCB_420_SP,
            ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_422_I,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_FP16,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RAW16,
            ::android::hardware::graphics::common::V1_0::PixelFormat::BLOB,
            ::android::hardware::graphics::common::V1_0::PixelFormat::IMPLEMENTATION_DEFINED,
            ::android::hardware::graphics::common::V1_0::PixelFormat::YCBCR_420_888,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RAW_OPAQUE,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RAW10,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RAW12,
            ::android::hardware::graphics::common::V1_0::PixelFormat::RGBA_1010102,
            ::android::hardware::graphics::common::V1_0::PixelFormat::Y8,
            ::android::hardware::graphics::common::V1_0::PixelFormat::Y16,
            ::android::hardware::graphics::common::V1_0::PixelFormat::YV12,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_0::BufferUsage>
{
    const ::android::hardware::graphics::common::V1_0::BufferUsage* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_0::BufferUsage* end() { return begin() + 23; }
    private:
    static const ::android::hardware::graphics::common::V1_0::BufferUsage* static_begin() {
        static const ::android::hardware::graphics::common::V1_0::BufferUsage kVals[23] {
            ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_MASK,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_NEVER,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_RARELY,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_READ_OFTEN,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_MASK,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_NEVER,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_RARELY,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CPU_WRITE_OFTEN,
            ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_TEXTURE,
            ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_RENDER_TARGET,
            ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_OVERLAY,
            ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CLIENT_TARGET,
            ::android::hardware::graphics::common::V1_0::BufferUsage::PROTECTED,
            ::android::hardware::graphics::common::V1_0::BufferUsage::COMPOSER_CURSOR,
            ::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_ENCODER,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_OUTPUT,
            ::android::hardware::graphics::common::V1_0::BufferUsage::CAMERA_INPUT,
            ::android::hardware::graphics::common::V1_0::BufferUsage::RENDERSCRIPT,
            ::android::hardware::graphics::common::V1_0::BufferUsage::VIDEO_DECODER,
            ::android::hardware::graphics::common::V1_0::BufferUsage::SENSOR_DIRECT_DATA,
            ::android::hardware::graphics::common::V1_0::BufferUsage::GPU_DATA_BUFFER,
            ::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK,
            ::android::hardware::graphics::common::V1_0::BufferUsage::VENDOR_MASK_HI,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_0::Transform>
{
    const ::android::hardware::graphics::common::V1_0::Transform* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_0::Transform* end() { return begin() + 5; }
    private:
    static const ::android::hardware::graphics::common::V1_0::Transform* static_begin() {
        static const ::android::hardware::graphics::common::V1_0::Transform kVals[5] {
            ::android::hardware::graphics::common::V1_0::Transform::FLIP_H,
            ::android::hardware::graphics::common::V1_0::Transform::FLIP_V,
            ::android::hardware::graphics::common::V1_0::Transform::ROT_90,
            ::android::hardware::graphics::common::V1_0::Transform::ROT_180,
            ::android::hardware::graphics::common::V1_0::Transform::ROT_270,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_0::Dataspace>
{
    const ::android::hardware::graphics::common::V1_0::Dataspace* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_0::Dataspace* end() { return begin() + 57; }
    private:
    static const ::android::hardware::graphics::common::V1_0::Dataspace* static_begin() {
        static const ::android::hardware::graphics::common::V1_0::Dataspace kVals[57] {
            ::android::hardware::graphics::common::V1_0::Dataspace::UNKNOWN,
            ::android::hardware::graphics::common::V1_0::Dataspace::ARBITRARY,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_SHIFT,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_MASK,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_UNSPECIFIED,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT709,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_625_UNADJUSTED,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT601_525_UNADJUSTED,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_BT470M,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_FILM,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_DCI_P3,
            ::android::hardware::graphics::common::V1_0::Dataspace::STANDARD_ADOBE_RGB,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SHIFT,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_MASK,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_UNSPECIFIED,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_LINEAR,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SRGB,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_SMPTE_170M,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_2,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_6,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_GAMMA2_8,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_ST2084,
            ::android::hardware::graphics::common::V1_0::Dataspace::TRANSFER_HLG,
            ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_SHIFT,
            ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_MASK,
            ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_UNSPECIFIED,
            ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_FULL,
            ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_LIMITED,
            ::android::hardware::graphics::common::V1_0::Dataspace::RANGE_EXTENDED,
            ::android::hardware::graphics::common::V1_0::Dataspace::SRGB_LINEAR,
            ::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB_LINEAR,
            ::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB_LINEAR,
            ::android::hardware::graphics::common::V1_0::Dataspace::SRGB,
            ::android::hardware::graphics::common::V1_0::Dataspace::V0_SRGB,
            ::android::hardware::graphics::common::V1_0::Dataspace::V0_SCRGB,
            ::android::hardware::graphics::common::V1_0::Dataspace::JFIF,
            ::android::hardware::graphics::common::V1_0::Dataspace::V0_JFIF,
            ::android::hardware::graphics::common::V1_0::Dataspace::BT601_625,
            ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_625,
            ::android::hardware::graphics::common::V1_0::Dataspace::BT601_525,
            ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT601_525,
            ::android::hardware::graphics::common::V1_0::Dataspace::BT709,
            ::android::hardware::graphics::common::V1_0::Dataspace::V0_BT709,
            ::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3_LINEAR,
            ::android::hardware::graphics::common::V1_0::Dataspace::DCI_P3,
            ::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3_LINEAR,
            ::android::hardware::graphics::common::V1_0::Dataspace::DISPLAY_P3,
            ::android::hardware::graphics::common::V1_0::Dataspace::ADOBE_RGB,
            ::android::hardware::graphics::common::V1_0::Dataspace::BT2020_LINEAR,
            ::android::hardware::graphics::common::V1_0::Dataspace::BT2020,
            ::android::hardware::graphics::common::V1_0::Dataspace::BT2020_PQ,
            ::android::hardware::graphics::common::V1_0::Dataspace::DEPTH,
            ::android::hardware::graphics::common::V1_0::Dataspace::SENSOR,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_0::ColorMode>
{
    const ::android::hardware::graphics::common::V1_0::ColorMode* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_0::ColorMode* end() { return begin() + 10; }
    private:
    static const ::android::hardware::graphics::common::V1_0::ColorMode* static_begin() {
        static const ::android::hardware::graphics::common::V1_0::ColorMode kVals[10] {
            ::android::hardware::graphics::common::V1_0::ColorMode::NATIVE,
            ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625,
            ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_625_UNADJUSTED,
            ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525,
            ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT601_525_UNADJUSTED,
            ::android::hardware::graphics::common::V1_0::ColorMode::STANDARD_BT709,
            ::android::hardware::graphics::common::V1_0::ColorMode::DCI_P3,
            ::android::hardware::graphics::common::V1_0::ColorMode::SRGB,
            ::android::hardware::graphics::common::V1_0::ColorMode::ADOBE_RGB,
            ::android::hardware::graphics::common::V1_0::ColorMode::DISPLAY_P3,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_0::ColorTransform>
{
    const ::android::hardware::graphics::common::V1_0::ColorTransform* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_0::ColorTransform* end() { return begin() + 7; }
    private:
    static const ::android::hardware::graphics::common::V1_0::ColorTransform* static_begin() {
        static const ::android::hardware::graphics::common::V1_0::ColorTransform kVals[7] {
            ::android::hardware::graphics::common::V1_0::ColorTransform::IDENTITY,
            ::android::hardware::graphics::common::V1_0::ColorTransform::ARBITRARY_MATRIX,
            ::android::hardware::graphics::common::V1_0::ColorTransform::VALUE_INVERSE,
            ::android::hardware::graphics::common::V1_0::ColorTransform::GRAYSCALE,
            ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_PROTANOPIA,
            ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_DEUTERANOPIA,
            ::android::hardware::graphics::common::V1_0::ColorTransform::CORRECT_TRITANOPIA,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_0::Hdr>
{
    const ::android::hardware::graphics::common::V1_0::Hdr* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_0::Hdr* end() { return begin() + 3; }
    private:
    static const ::android::hardware::graphics::common::V1_0::Hdr* static_begin() {
        static const ::android::hardware::graphics::common::V1_0::Hdr kVals[3] {
            ::android::hardware::graphics::common::V1_0::Hdr::DOLBY_VISION,
            ::android::hardware::graphics::common::V1_0::Hdr::HDR10,
            ::android::hardware::graphics::common::V1_0::Hdr::HLG,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_0_TYPES_H
