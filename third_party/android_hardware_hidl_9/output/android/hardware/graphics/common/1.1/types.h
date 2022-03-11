#ifndef HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_1_TYPES_H
#define HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_1_TYPES_H

#include <android/hardware/graphics/common/1.0/types.h>

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hardware {
namespace graphics {
namespace common {
namespace V1_1 {

// Forward declaration for forward reference support:
enum class PixelFormat : int32_t;
enum class BufferUsage : uint64_t;
enum class Dataspace : int32_t;
enum class ColorMode : int32_t;
enum class RenderIntent : int32_t;

/**
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
    /**
     * 16-bit format that has a single 16-bit depth component.
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    DEPTH_16 = 48,
    /**
     * 32-bit format that has a single 24-bit depth component and, optionally,
     * 8 bits that are unused.
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     */
    DEPTH_24 = 49,
    /**
     * 32-bit format that has a 24-bit depth component and an 8-bit stencil
     * component packed into 32-bits.
     * 
     * The depth component values are unsigned normalized to the range [0, 1],
     * whose interpretation is defined by the dataspace. The stencil values are
     * unsigned integers, whose interpretation is defined by the dataspace.
     */
    DEPTH_24_STENCIL_8 = 50,
    /**
     * 32-bit format that has a single 32-bit depth component.
     * 
     * The component values are signed floats, whose interpretation is defined
     * by the dataspace.
     */
    DEPTH_32F = 51,
    /**
     * Two-component format that has a 32-bit depth component, an 8-bit stencil
     * component, and optionally 24-bits unused.
     * 
     * The depth component values are signed floats, whose interpretation is
     * defined by the dataspace. The stencil bits are unsigned integers, whose
     * interpretation is defined by the dataspace.
     */
    DEPTH_32F_STENCIL_8 = 52,
    /**
     * 8-bit format that has a single 8-bit stencil component.
     * 
     * The component values are unsigned integers, whose interpretation is
     * defined by the dataspace.
     */
    STENCIL_8 = 53,
    /**
     * P010 is a 4:2:0 YCbCr semiplanar format comprised of a WxH Y plane
     * followed immediately by a Wx(H/2) CbCr plane. Each sample is
     * represented by a 16-bit little-endian value, with the lower 6 bits set
     * to zero.
     * 
     * This format assumes
     * - an even height
     * - a vertical stride equal to the height
     * 
     *   stride_in_bytes = stride * 2
     *   y_size = stride_in_bytes * height
     *   cbcr_size = stride_in_bytes * (height / 2)
     *   cb_offset = y_size
     *   cr_offset = cb_offset + 2
     * 
     * This format must be accepted by the allocator when used with the
     * following usage flags:
     * 
     *    - BufferUsage::VIDEO_*
     *    - BufferUsage::CPU_*
     *    - BufferUsage::GPU_TEXTURE
     * 
     * The component values are unsigned normalized to the range [0, 1], whose
     * interpretation is defined by the dataspace.
     * 
     * This format is appropriate for 10bit video content.
     * 
     * Buffers with this format must be locked with IMapper::lockYCbCr
     * or with IMapper::lock.
     */
    YCBCR_P010 = 54,
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
    /**
     * buffer is used as a cube map texture  */
    GPU_CUBE_MAP = 33554432ull, // (1ULL << 25)
    /**
     * buffer contains a complete mipmap hierarchy  */
    GPU_MIPMAP_COMPLETE = 67108864ull, // (1ULL << 26)
};

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
    /**
     * ITU-R Recommendation 2020 (BT.2020)
     * 
     * Ultra High-definition television
     * 
     * Use limited range, BT.709 transfer and BT2020 standard
     */
    BT2020_ITU = 281411584, // ((STANDARD_BT2020 | TRANSFER_SMPTE_170M) | RANGE_LIMITED)
    /**
     * ITU-R Recommendation 2100 (BT.2100)
     * 
     * High dynamic range television
     * 
     * Use limited/full range, PQ/HLG transfer, and BT2020 standard
     * limited range is the preferred / normative definition for BT.2100
     */
    BT2020_ITU_PQ = 298188800, // ((STANDARD_BT2020 | TRANSFER_ST2084) | RANGE_LIMITED)
    BT2020_ITU_HLG = 302383104, // ((STANDARD_BT2020 | TRANSFER_HLG) | RANGE_LIMITED)
    BT2020_HLG = 168165376, // ((STANDARD_BT2020 | TRANSFER_HLG) | RANGE_FULL)
};

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
    /**
     * BT2020 corresponds with display settings that implement the ITU-R
     * Recommendation BT.2020 / Rec. 2020 for UHDTV.
     * 
     * Primaries:
     *                  x       y
     *  green           0.170   0.797
     *  blue            0.131   0.046
     *  red             0.708   0.292
     *  white (D65)     0.3127  0.3290
     * 
     * Inverse Gamma Correction (IGC): V represents normalized (with [0 to 1]
     * range) value of R, G, or B.
     * 
     *  if Vnonlinear < b * 4.5
     *    Vlinear = Vnonlinear / 4.5
     *  else
     *    Vlinear = ((Vnonlinear + (a - 1)) / a) ^ (1/0.45)
     * 
     * Gamma Correction (GC):
     * 
     *  if Vlinear < b
     *    Vnonlinear = 4.5 * Vlinear
     *  else
     *    Vnonlinear = a * Vlinear ^ 0.45 - (a - 1)
     * 
     * where
     * 
     *   a = 1.09929682680944, b = 0.018053968510807
     * 
     * For practical purposes, these a/b values can be used instead
     * 
     *   a = 1.099, b = 0.018 for 10-bit display systems
     *   a = 1.0993, b = 0.0181 for 12-bit display systems
     */
    BT2020 = 10,
    /**
     * BT2100_PQ and BT2100_HLG correspond with display settings that
     * implement the ITU-R Recommendation BT.2100 / Rec. 2100 for HDR TV.
     * 
     * Primaries:
     *                  x       y
     *  green           0.170   0.797
     *  blue            0.131   0.046
     *  red             0.708   0.292
     *  white (D65)     0.3127  0.3290
     * 
     * For BT2100_PQ, the transfer function is Perceptual Quantizer (PQ). For
     * BT2100_HLG, the transfer function is Hybrid Log-Gamma (HLG).
     */
    BT2100_PQ = 11,
    BT2100_HLG = 12,
};

/**
 * RenderIntent defines the mapping from color mode colors to display colors.
 * 
 * A render intent must not change how it maps colors when the color mode
 * changes. That is to say that when a render intent maps color C to color C',
 * the fact that color C can have different pixel values in different color
 * modes should not affect the mapping.
 * 
 * RenderIntent overrides the render intents defined for individual color
 * modes. It is ignored when the color mode is ColorMode::NATIVE, because
 * ColorMode::NATIVE colors are already display colors.
 */
enum class RenderIntent : int32_t {
    /**
     * Colors in the display gamut are unchanged. Colors out of the display
     * gamut are hard-clipped.
     * 
     * This implies that the display must have been calibrated unless
     * ColorMode::NATIVE is the only supported color mode.
     */
    COLORIMETRIC = 0,
    /**
     * Enhance colors that are in the display gamut. Colors out of the display
     * gamut are hard-clipped.
     * 
     * The enhancement typically picks the biggest standard color space (e.g.
     * DCI-P3) that is narrower than the display gamut and stretches it to the
     * display gamut. The stretching is recommended to preserve skin tones.
     */
    ENHANCE = 1,
    /**
     * Tone map high-dynamic-range colors to the display's dynamic range. The
     * dynamic range of the colors are communicated separately. After tone
     * mapping, the mapping to the display gamut is as defined in
     * COLORIMETRIC.
     */
    TONE_MAP_COLORIMETRIC = 2,
    /**
     * Tone map high-dynamic-range colors to the display's dynamic range. The
     * dynamic range of the colors are communicated separately. After tone
     * mapping, the mapping to the display gamut is as defined in ENHANCE.
     * 
     * The tone mapping step and the enhancing step must match
     * TONE_MAP_COLORIMETRIC and ENHANCE respectively when they are also
     * supported.
     */
    TONE_MAP_ENHANCE = 3,
};

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_1::PixelFormat lhs, const ::android::hardware::graphics::common::V1_1::PixelFormat rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_1::PixelFormat rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_1::PixelFormat lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_1::PixelFormat lhs, const ::android::hardware::graphics::common::V1_1::PixelFormat rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_1::PixelFormat rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_1::PixelFormat lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_1::PixelFormat e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_1::PixelFormat e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_1::PixelFormat>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_1::PixelFormat> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_8888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_8888)) {
        os += (first ? "" : " | ");
        os += "RGBA_8888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_8888;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RGBX_8888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RGBX_8888)) {
        os += (first ? "" : " | ");
        os += "RGBX_8888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RGBX_8888;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RGB_888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RGB_888)) {
        os += (first ? "" : " | ");
        os += "RGB_888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RGB_888;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RGB_565) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RGB_565)) {
        os += (first ? "" : " | ");
        os += "RGB_565";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RGB_565;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::BGRA_8888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::BGRA_8888)) {
        os += (first ? "" : " | ");
        os += "BGRA_8888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::BGRA_8888;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_SP) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_SP)) {
        os += (first ? "" : " | ");
        os += "YCBCR_422_SP";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_SP;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::YCRCB_420_SP) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::YCRCB_420_SP)) {
        os += (first ? "" : " | ");
        os += "YCRCB_420_SP";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::YCRCB_420_SP;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_I) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_I)) {
        os += (first ? "" : " | ");
        os += "YCBCR_422_I";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_I;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_FP16) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_FP16)) {
        os += (first ? "" : " | ");
        os += "RGBA_FP16";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_FP16;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RAW16) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RAW16)) {
        os += (first ? "" : " | ");
        os += "RAW16";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RAW16;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::BLOB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::BLOB)) {
        os += (first ? "" : " | ");
        os += "BLOB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::BLOB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::IMPLEMENTATION_DEFINED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::IMPLEMENTATION_DEFINED)) {
        os += (first ? "" : " | ");
        os += "IMPLEMENTATION_DEFINED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::IMPLEMENTATION_DEFINED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_420_888) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_420_888)) {
        os += (first ? "" : " | ");
        os += "YCBCR_420_888";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_420_888;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RAW_OPAQUE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RAW_OPAQUE)) {
        os += (first ? "" : " | ");
        os += "RAW_OPAQUE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RAW_OPAQUE;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RAW10) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RAW10)) {
        os += (first ? "" : " | ");
        os += "RAW10";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RAW10;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RAW12) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RAW12)) {
        os += (first ? "" : " | ");
        os += "RAW12";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RAW12;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_1010102) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_1010102)) {
        os += (first ? "" : " | ");
        os += "RGBA_1010102";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_1010102;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::Y8) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::Y8)) {
        os += (first ? "" : " | ");
        os += "Y8";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::Y8;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::Y16) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::Y16)) {
        os += (first ? "" : " | ");
        os += "Y16";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::Y16;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::YV12) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::YV12)) {
        os += (first ? "" : " | ");
        os += "YV12";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::YV12;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_16) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_16)) {
        os += (first ? "" : " | ");
        os += "DEPTH_16";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_16;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24)) {
        os += (first ? "" : " | ");
        os += "DEPTH_24";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24_STENCIL_8) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24_STENCIL_8)) {
        os += (first ? "" : " | ");
        os += "DEPTH_24_STENCIL_8";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24_STENCIL_8;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F)) {
        os += (first ? "" : " | ");
        os += "DEPTH_32F";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F_STENCIL_8) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F_STENCIL_8)) {
        os += (first ? "" : " | ");
        os += "DEPTH_32F_STENCIL_8";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F_STENCIL_8;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::STENCIL_8) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::STENCIL_8)) {
        os += (first ? "" : " | ");
        os += "STENCIL_8";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::STENCIL_8;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_P010) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_P010)) {
        os += (first ? "" : " | ");
        os += "YCBCR_P010";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_P010;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_1::PixelFormat o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_8888) {
        return "RGBA_8888";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RGBX_8888) {
        return "RGBX_8888";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RGB_888) {
        return "RGB_888";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RGB_565) {
        return "RGB_565";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::BGRA_8888) {
        return "BGRA_8888";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_SP) {
        return "YCBCR_422_SP";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::YCRCB_420_SP) {
        return "YCRCB_420_SP";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_I) {
        return "YCBCR_422_I";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_FP16) {
        return "RGBA_FP16";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RAW16) {
        return "RAW16";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::BLOB) {
        return "BLOB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::IMPLEMENTATION_DEFINED) {
        return "IMPLEMENTATION_DEFINED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_420_888) {
        return "YCBCR_420_888";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RAW_OPAQUE) {
        return "RAW_OPAQUE";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RAW10) {
        return "RAW10";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RAW12) {
        return "RAW12";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_1010102) {
        return "RGBA_1010102";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::Y8) {
        return "Y8";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::Y16) {
        return "Y16";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::YV12) {
        return "YV12";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_16) {
        return "DEPTH_16";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24) {
        return "DEPTH_24";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24_STENCIL_8) {
        return "DEPTH_24_STENCIL_8";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F) {
        return "DEPTH_32F";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F_STENCIL_8) {
        return "DEPTH_32F_STENCIL_8";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::STENCIL_8) {
        return "STENCIL_8";
    }
    if (o == ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_P010) {
        return "YCBCR_P010";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

constexpr uint64_t operator|(const ::android::hardware::graphics::common::V1_1::BufferUsage lhs, const ::android::hardware::graphics::common::V1_1::BufferUsage rhs) {
    return static_cast<uint64_t>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

constexpr uint64_t operator|(const uint64_t lhs, const ::android::hardware::graphics::common::V1_1::BufferUsage rhs) {
    return static_cast<uint64_t>(lhs | static_cast<uint64_t>(rhs));
}

constexpr uint64_t operator|(const ::android::hardware::graphics::common::V1_1::BufferUsage lhs, const uint64_t rhs) {
    return static_cast<uint64_t>(static_cast<uint64_t>(lhs) | rhs);
}

constexpr uint64_t operator&(const ::android::hardware::graphics::common::V1_1::BufferUsage lhs, const ::android::hardware::graphics::common::V1_1::BufferUsage rhs) {
    return static_cast<uint64_t>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

constexpr uint64_t operator&(const uint64_t lhs, const ::android::hardware::graphics::common::V1_1::BufferUsage rhs) {
    return static_cast<uint64_t>(lhs & static_cast<uint64_t>(rhs));
}

constexpr uint64_t operator&(const ::android::hardware::graphics::common::V1_1::BufferUsage lhs, const uint64_t rhs) {
    return static_cast<uint64_t>(static_cast<uint64_t>(lhs) & rhs);
}

constexpr uint64_t &operator|=(uint64_t& v, const ::android::hardware::graphics::common::V1_1::BufferUsage e) {
    v |= static_cast<uint64_t>(e);
    return v;
}

constexpr uint64_t &operator&=(uint64_t& v, const ::android::hardware::graphics::common::V1_1::BufferUsage e) {
    v &= static_cast<uint64_t>(e);
    return v;
}

template<typename>
static inline std::string toString(uint64_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_1::BufferUsage>(uint64_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_1::BufferUsage> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_MASK) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_MASK)) {
        os += (first ? "" : " | ");
        os += "CPU_READ_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_NEVER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_NEVER)) {
        os += (first ? "" : " | ");
        os += "CPU_READ_NEVER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_NEVER;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_RARELY) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_RARELY)) {
        os += (first ? "" : " | ");
        os += "CPU_READ_RARELY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_RARELY;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_OFTEN) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_OFTEN)) {
        os += (first ? "" : " | ");
        os += "CPU_READ_OFTEN";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_OFTEN;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_MASK) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_MASK)) {
        os += (first ? "" : " | ");
        os += "CPU_WRITE_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_NEVER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_NEVER)) {
        os += (first ? "" : " | ");
        os += "CPU_WRITE_NEVER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_NEVER;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_RARELY) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_RARELY)) {
        os += (first ? "" : " | ");
        os += "CPU_WRITE_RARELY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_RARELY;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_OFTEN) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_OFTEN)) {
        os += (first ? "" : " | ");
        os += "CPU_WRITE_OFTEN";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_OFTEN;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_TEXTURE) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::GPU_TEXTURE)) {
        os += (first ? "" : " | ");
        os += "GPU_TEXTURE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_TEXTURE;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_RENDER_TARGET) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::GPU_RENDER_TARGET)) {
        os += (first ? "" : " | ");
        os += "GPU_RENDER_TARGET";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_RENDER_TARGET;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_OVERLAY) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_OVERLAY)) {
        os += (first ? "" : " | ");
        os += "COMPOSER_OVERLAY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_OVERLAY;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CLIENT_TARGET) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CLIENT_TARGET)) {
        os += (first ? "" : " | ");
        os += "COMPOSER_CLIENT_TARGET";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CLIENT_TARGET;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::PROTECTED) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::PROTECTED)) {
        os += (first ? "" : " | ");
        os += "PROTECTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::PROTECTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CURSOR) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CURSOR)) {
        os += (first ? "" : " | ");
        os += "COMPOSER_CURSOR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CURSOR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_ENCODER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_ENCODER)) {
        os += (first ? "" : " | ");
        os += "VIDEO_ENCODER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_ENCODER;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_OUTPUT) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_OUTPUT)) {
        os += (first ? "" : " | ");
        os += "CAMERA_OUTPUT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_OUTPUT;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_INPUT) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_INPUT)) {
        os += (first ? "" : " | ");
        os += "CAMERA_INPUT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_INPUT;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::RENDERSCRIPT) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::RENDERSCRIPT)) {
        os += (first ? "" : " | ");
        os += "RENDERSCRIPT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::RENDERSCRIPT;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_DECODER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_DECODER)) {
        os += (first ? "" : " | ");
        os += "VIDEO_DECODER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_DECODER;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::SENSOR_DIRECT_DATA) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::SENSOR_DIRECT_DATA)) {
        os += (first ? "" : " | ");
        os += "SENSOR_DIRECT_DATA";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::SENSOR_DIRECT_DATA;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_DATA_BUFFER) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::GPU_DATA_BUFFER)) {
        os += (first ? "" : " | ");
        os += "GPU_DATA_BUFFER";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_DATA_BUFFER;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK)) {
        os += (first ? "" : " | ");
        os += "VENDOR_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK_HI) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK_HI)) {
        os += (first ? "" : " | ");
        os += "VENDOR_MASK_HI";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK_HI;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_CUBE_MAP) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::GPU_CUBE_MAP)) {
        os += (first ? "" : " | ");
        os += "GPU_CUBE_MAP";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_CUBE_MAP;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_MIPMAP_COMPLETE) == static_cast<uint64_t>(::android::hardware::graphics::common::V1_1::BufferUsage::GPU_MIPMAP_COMPLETE)) {
        os += (first ? "" : " | ");
        os += "GPU_MIPMAP_COMPLETE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_MIPMAP_COMPLETE;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_1::BufferUsage o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_MASK) {
        return "CPU_READ_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_NEVER) {
        return "CPU_READ_NEVER";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_RARELY) {
        return "CPU_READ_RARELY";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_OFTEN) {
        return "CPU_READ_OFTEN";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_MASK) {
        return "CPU_WRITE_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_NEVER) {
        return "CPU_WRITE_NEVER";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_RARELY) {
        return "CPU_WRITE_RARELY";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_OFTEN) {
        return "CPU_WRITE_OFTEN";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_TEXTURE) {
        return "GPU_TEXTURE";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_RENDER_TARGET) {
        return "GPU_RENDER_TARGET";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_OVERLAY) {
        return "COMPOSER_OVERLAY";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CLIENT_TARGET) {
        return "COMPOSER_CLIENT_TARGET";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::PROTECTED) {
        return "PROTECTED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CURSOR) {
        return "COMPOSER_CURSOR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_ENCODER) {
        return "VIDEO_ENCODER";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_OUTPUT) {
        return "CAMERA_OUTPUT";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_INPUT) {
        return "CAMERA_INPUT";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::RENDERSCRIPT) {
        return "RENDERSCRIPT";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_DECODER) {
        return "VIDEO_DECODER";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::SENSOR_DIRECT_DATA) {
        return "SENSOR_DIRECT_DATA";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_DATA_BUFFER) {
        return "GPU_DATA_BUFFER";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK) {
        return "VENDOR_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK_HI) {
        return "VENDOR_MASK_HI";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_CUBE_MAP) {
        return "GPU_CUBE_MAP";
    }
    if (o == ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_MIPMAP_COMPLETE) {
        return "GPU_MIPMAP_COMPLETE";
    }
    std::string os;
    os += toHexString(static_cast<uint64_t>(o));
    return os;
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_1::Dataspace lhs, const ::android::hardware::graphics::common::V1_1::Dataspace rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_1::Dataspace rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_1::Dataspace lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_1::Dataspace lhs, const ::android::hardware::graphics::common::V1_1::Dataspace rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_1::Dataspace rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_1::Dataspace lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_1::Dataspace e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_1::Dataspace e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_1::Dataspace>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_1::Dataspace> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::UNKNOWN) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::UNKNOWN)) {
        os += (first ? "" : " | ");
        os += "UNKNOWN";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::UNKNOWN;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::ARBITRARY) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::ARBITRARY)) {
        os += (first ? "" : " | ");
        os += "ARBITRARY";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::ARBITRARY;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_SHIFT) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_SHIFT)) {
        os += (first ? "" : " | ");
        os += "STANDARD_SHIFT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_SHIFT;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_MASK) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_MASK)) {
        os += (first ? "" : " | ");
        os += "STANDARD_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_UNSPECIFIED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_UNSPECIFIED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_UNSPECIFIED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_UNSPECIFIED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT709) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT709)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT709";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT709;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_625";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625_UNADJUSTED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625_UNADJUSTED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_625_UNADJUSTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625_UNADJUSTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_525";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525_UNADJUSTED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525_UNADJUSTED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_525_UNADJUSTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525_UNADJUSTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT2020";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT2020_CONSTANT_LUMINANCE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT470M) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT470M)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT470M";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT470M;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_FILM) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_FILM)) {
        os += (first ? "" : " | ");
        os += "STANDARD_FILM";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_FILM;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_DCI_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_DCI_P3)) {
        os += (first ? "" : " | ");
        os += "STANDARD_DCI_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_DCI_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_ADOBE_RGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_ADOBE_RGB)) {
        os += (first ? "" : " | ");
        os += "STANDARD_ADOBE_RGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_ADOBE_RGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SHIFT) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SHIFT)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_SHIFT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SHIFT;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_MASK) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_MASK)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_UNSPECIFIED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_UNSPECIFIED)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_UNSPECIFIED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_UNSPECIFIED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_LINEAR)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SRGB)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_SRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SMPTE_170M) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SMPTE_170M)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_SMPTE_170M";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SMPTE_170M;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_2) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_2)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_GAMMA2_2";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_2;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_6) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_6)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_GAMMA2_6";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_6;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_8) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_8)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_GAMMA2_8";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_8;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_ST2084) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_ST2084)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_ST2084";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_ST2084;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_HLG) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_HLG)) {
        os += (first ? "" : " | ");
        os += "TRANSFER_HLG";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_HLG;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_SHIFT) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::RANGE_SHIFT)) {
        os += (first ? "" : " | ");
        os += "RANGE_SHIFT";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_SHIFT;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_MASK) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::RANGE_MASK)) {
        os += (first ? "" : " | ");
        os += "RANGE_MASK";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_MASK;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_UNSPECIFIED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::RANGE_UNSPECIFIED)) {
        os += (first ? "" : " | ");
        os += "RANGE_UNSPECIFIED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_UNSPECIFIED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_FULL) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::RANGE_FULL)) {
        os += (first ? "" : " | ");
        os += "RANGE_FULL";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_FULL;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_LIMITED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::RANGE_LIMITED)) {
        os += (first ? "" : " | ");
        os += "RANGE_LIMITED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_LIMITED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_EXTENDED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::RANGE_EXTENDED)) {
        os += (first ? "" : " | ");
        os += "RANGE_EXTENDED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_EXTENDED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::SRGB_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::SRGB_LINEAR)) {
        os += (first ? "" : " | ");
        os += "SRGB_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::SRGB_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB_LINEAR)) {
        os += (first ? "" : " | ");
        os += "V0_SRGB_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB_LINEAR)) {
        os += (first ? "" : " | ");
        os += "V0_SCRGB_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::SRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::SRGB)) {
        os += (first ? "" : " | ");
        os += "SRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::SRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB)) {
        os += (first ? "" : " | ");
        os += "V0_SRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB)) {
        os += (first ? "" : " | ");
        os += "V0_SCRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::JFIF) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::JFIF)) {
        os += (first ? "" : " | ");
        os += "JFIF";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::JFIF;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::V0_JFIF) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::V0_JFIF)) {
        os += (first ? "" : " | ");
        os += "V0_JFIF";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::V0_JFIF;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT601_625) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT601_625)) {
        os += (first ? "" : " | ");
        os += "BT601_625";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT601_625;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_625) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_625)) {
        os += (first ? "" : " | ");
        os += "V0_BT601_625";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_625;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT601_525) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT601_525)) {
        os += (first ? "" : " | ");
        os += "BT601_525";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT601_525;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_525) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_525)) {
        os += (first ? "" : " | ");
        os += "V0_BT601_525";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_525;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT709) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT709)) {
        os += (first ? "" : " | ");
        os += "BT709";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT709;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT709) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::V0_BT709)) {
        os += (first ? "" : " | ");
        os += "V0_BT709";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT709;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3_LINEAR)) {
        os += (first ? "" : " | ");
        os += "DCI_P3_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3)) {
        os += (first ? "" : " | ");
        os += "DCI_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3_LINEAR)) {
        os += (first ? "" : " | ");
        os += "DISPLAY_P3_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3)) {
        os += (first ? "" : " | ");
        os += "DISPLAY_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::ADOBE_RGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::ADOBE_RGB)) {
        os += (first ? "" : " | ");
        os += "ADOBE_RGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::ADOBE_RGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_LINEAR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT2020_LINEAR)) {
        os += (first ? "" : " | ");
        os += "BT2020_LINEAR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_LINEAR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT2020) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT2020)) {
        os += (first ? "" : " | ");
        os += "BT2020";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT2020;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_PQ) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT2020_PQ)) {
        os += (first ? "" : " | ");
        os += "BT2020_PQ";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_PQ;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::DEPTH) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::DEPTH)) {
        os += (first ? "" : " | ");
        os += "DEPTH";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::DEPTH;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::SENSOR) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::SENSOR)) {
        os += (first ? "" : " | ");
        os += "SENSOR";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::SENSOR;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU)) {
        os += (first ? "" : " | ");
        os += "BT2020_ITU";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_PQ) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_PQ)) {
        os += (first ? "" : " | ");
        os += "BT2020_ITU_PQ";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_PQ;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_HLG) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_HLG)) {
        os += (first ? "" : " | ");
        os += "BT2020_ITU_HLG";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_HLG;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_HLG) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::Dataspace::BT2020_HLG)) {
        os += (first ? "" : " | ");
        os += "BT2020_HLG";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_HLG;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_1::Dataspace o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::UNKNOWN) {
        return "UNKNOWN";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::ARBITRARY) {
        return "ARBITRARY";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_SHIFT) {
        return "STANDARD_SHIFT";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_MASK) {
        return "STANDARD_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_UNSPECIFIED) {
        return "STANDARD_UNSPECIFIED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT709) {
        return "STANDARD_BT709";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625) {
        return "STANDARD_BT601_625";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625_UNADJUSTED) {
        return "STANDARD_BT601_625_UNADJUSTED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525) {
        return "STANDARD_BT601_525";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525_UNADJUSTED) {
        return "STANDARD_BT601_525_UNADJUSTED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020) {
        return "STANDARD_BT2020";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE) {
        return "STANDARD_BT2020_CONSTANT_LUMINANCE";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT470M) {
        return "STANDARD_BT470M";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_FILM) {
        return "STANDARD_FILM";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_DCI_P3) {
        return "STANDARD_DCI_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_ADOBE_RGB) {
        return "STANDARD_ADOBE_RGB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SHIFT) {
        return "TRANSFER_SHIFT";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_MASK) {
        return "TRANSFER_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_UNSPECIFIED) {
        return "TRANSFER_UNSPECIFIED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_LINEAR) {
        return "TRANSFER_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SRGB) {
        return "TRANSFER_SRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SMPTE_170M) {
        return "TRANSFER_SMPTE_170M";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_2) {
        return "TRANSFER_GAMMA2_2";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_6) {
        return "TRANSFER_GAMMA2_6";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_8) {
        return "TRANSFER_GAMMA2_8";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_ST2084) {
        return "TRANSFER_ST2084";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_HLG) {
        return "TRANSFER_HLG";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_SHIFT) {
        return "RANGE_SHIFT";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_MASK) {
        return "RANGE_MASK";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_UNSPECIFIED) {
        return "RANGE_UNSPECIFIED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_FULL) {
        return "RANGE_FULL";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_LIMITED) {
        return "RANGE_LIMITED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_EXTENDED) {
        return "RANGE_EXTENDED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::SRGB_LINEAR) {
        return "SRGB_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB_LINEAR) {
        return "V0_SRGB_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB_LINEAR) {
        return "V0_SCRGB_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::SRGB) {
        return "SRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB) {
        return "V0_SRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB) {
        return "V0_SCRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::JFIF) {
        return "JFIF";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::V0_JFIF) {
        return "V0_JFIF";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT601_625) {
        return "BT601_625";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_625) {
        return "V0_BT601_625";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT601_525) {
        return "BT601_525";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_525) {
        return "V0_BT601_525";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT709) {
        return "BT709";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT709) {
        return "V0_BT709";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3_LINEAR) {
        return "DCI_P3_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3) {
        return "DCI_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3_LINEAR) {
        return "DISPLAY_P3_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3) {
        return "DISPLAY_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::ADOBE_RGB) {
        return "ADOBE_RGB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_LINEAR) {
        return "BT2020_LINEAR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT2020) {
        return "BT2020";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_PQ) {
        return "BT2020_PQ";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::DEPTH) {
        return "DEPTH";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::SENSOR) {
        return "SENSOR";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU) {
        return "BT2020_ITU";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_PQ) {
        return "BT2020_ITU_PQ";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_HLG) {
        return "BT2020_ITU_HLG";
    }
    if (o == ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_HLG) {
        return "BT2020_HLG";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_1::ColorMode lhs, const ::android::hardware::graphics::common::V1_1::ColorMode rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_1::ColorMode rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_1::ColorMode lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_1::ColorMode lhs, const ::android::hardware::graphics::common::V1_1::ColorMode rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_1::ColorMode rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_1::ColorMode lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_1::ColorMode e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_1::ColorMode e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_1::ColorMode>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_1::ColorMode> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::NATIVE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::NATIVE)) {
        os += (first ? "" : " | ");
        os += "NATIVE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::NATIVE;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_625";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625_UNADJUSTED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625_UNADJUSTED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_625_UNADJUSTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625_UNADJUSTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_525";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525_UNADJUSTED) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525_UNADJUSTED)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT601_525_UNADJUSTED";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525_UNADJUSTED;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT709) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT709)) {
        os += (first ? "" : " | ");
        os += "STANDARD_BT709";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT709;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::DCI_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::DCI_P3)) {
        os += (first ? "" : " | ");
        os += "DCI_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::DCI_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::SRGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::SRGB)) {
        os += (first ? "" : " | ");
        os += "SRGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::SRGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::ADOBE_RGB) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::ADOBE_RGB)) {
        os += (first ? "" : " | ");
        os += "ADOBE_RGB";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::ADOBE_RGB;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::DISPLAY_P3) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::DISPLAY_P3)) {
        os += (first ? "" : " | ");
        os += "DISPLAY_P3";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::DISPLAY_P3;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::BT2020) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::BT2020)) {
        os += (first ? "" : " | ");
        os += "BT2020";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::BT2020;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::BT2100_PQ) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::BT2100_PQ)) {
        os += (first ? "" : " | ");
        os += "BT2100_PQ";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::BT2100_PQ;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::ColorMode::BT2100_HLG) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::ColorMode::BT2100_HLG)) {
        os += (first ? "" : " | ");
        os += "BT2100_HLG";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::ColorMode::BT2100_HLG;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_1::ColorMode o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::NATIVE) {
        return "NATIVE";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625) {
        return "STANDARD_BT601_625";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625_UNADJUSTED) {
        return "STANDARD_BT601_625_UNADJUSTED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525) {
        return "STANDARD_BT601_525";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525_UNADJUSTED) {
        return "STANDARD_BT601_525_UNADJUSTED";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT709) {
        return "STANDARD_BT709";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::DCI_P3) {
        return "DCI_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::SRGB) {
        return "SRGB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::ADOBE_RGB) {
        return "ADOBE_RGB";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::DISPLAY_P3) {
        return "DISPLAY_P3";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::BT2020) {
        return "BT2020";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::BT2100_PQ) {
        return "BT2100_PQ";
    }
    if (o == ::android::hardware::graphics::common::V1_1::ColorMode::BT2100_HLG) {
        return "BT2100_HLG";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_1::RenderIntent lhs, const ::android::hardware::graphics::common::V1_1::RenderIntent rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::common::V1_1::RenderIntent rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::common::V1_1::RenderIntent lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_1::RenderIntent lhs, const ::android::hardware::graphics::common::V1_1::RenderIntent rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::common::V1_1::RenderIntent rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::common::V1_1::RenderIntent lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::common::V1_1::RenderIntent e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::common::V1_1::RenderIntent e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::common::V1_1::RenderIntent>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::common::V1_1::RenderIntent> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::common::V1_1::RenderIntent::COLORIMETRIC) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::RenderIntent::COLORIMETRIC)) {
        os += (first ? "" : " | ");
        os += "COLORIMETRIC";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::RenderIntent::COLORIMETRIC;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::RenderIntent::ENHANCE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::RenderIntent::ENHANCE)) {
        os += (first ? "" : " | ");
        os += "ENHANCE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::RenderIntent::ENHANCE;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_COLORIMETRIC) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_COLORIMETRIC)) {
        os += (first ? "" : " | ");
        os += "TONE_MAP_COLORIMETRIC";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_COLORIMETRIC;
    }
    if ((o & ::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_ENHANCE) == static_cast<int32_t>(::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_ENHANCE)) {
        os += (first ? "" : " | ");
        os += "TONE_MAP_ENHANCE";
        first = false;
        flipped |= ::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_ENHANCE;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::common::V1_1::RenderIntent o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::common::V1_1::RenderIntent::COLORIMETRIC) {
        return "COLORIMETRIC";
    }
    if (o == ::android::hardware::graphics::common::V1_1::RenderIntent::ENHANCE) {
        return "ENHANCE";
    }
    if (o == ::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_COLORIMETRIC) {
        return "TONE_MAP_COLORIMETRIC";
    }
    if (o == ::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_ENHANCE) {
        return "TONE_MAP_ENHANCE";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}


}  // namespace V1_1
}  // namespace common
}  // namespace graphics
}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_1::PixelFormat>
{
    const ::android::hardware::graphics::common::V1_1::PixelFormat* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_1::PixelFormat* end() { return begin() + 27; }
    private:
    static const ::android::hardware::graphics::common::V1_1::PixelFormat* static_begin() {
        static const ::android::hardware::graphics::common::V1_1::PixelFormat kVals[27] {
            ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_8888,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RGBX_8888,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RGB_888,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RGB_565,
            ::android::hardware::graphics::common::V1_1::PixelFormat::BGRA_8888,
            ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_SP,
            ::android::hardware::graphics::common::V1_1::PixelFormat::YCRCB_420_SP,
            ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_422_I,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_FP16,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RAW16,
            ::android::hardware::graphics::common::V1_1::PixelFormat::BLOB,
            ::android::hardware::graphics::common::V1_1::PixelFormat::IMPLEMENTATION_DEFINED,
            ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_420_888,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RAW_OPAQUE,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RAW10,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RAW12,
            ::android::hardware::graphics::common::V1_1::PixelFormat::RGBA_1010102,
            ::android::hardware::graphics::common::V1_1::PixelFormat::Y8,
            ::android::hardware::graphics::common::V1_1::PixelFormat::Y16,
            ::android::hardware::graphics::common::V1_1::PixelFormat::YV12,
            ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_16,
            ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24,
            ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_24_STENCIL_8,
            ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F,
            ::android::hardware::graphics::common::V1_1::PixelFormat::DEPTH_32F_STENCIL_8,
            ::android::hardware::graphics::common::V1_1::PixelFormat::STENCIL_8,
            ::android::hardware::graphics::common::V1_1::PixelFormat::YCBCR_P010,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_1::BufferUsage>
{
    const ::android::hardware::graphics::common::V1_1::BufferUsage* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_1::BufferUsage* end() { return begin() + 25; }
    private:
    static const ::android::hardware::graphics::common::V1_1::BufferUsage* static_begin() {
        static const ::android::hardware::graphics::common::V1_1::BufferUsage kVals[25] {
            ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_MASK,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_NEVER,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_RARELY,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_READ_OFTEN,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_MASK,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_NEVER,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_RARELY,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CPU_WRITE_OFTEN,
            ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_TEXTURE,
            ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_RENDER_TARGET,
            ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_OVERLAY,
            ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CLIENT_TARGET,
            ::android::hardware::graphics::common::V1_1::BufferUsage::PROTECTED,
            ::android::hardware::graphics::common::V1_1::BufferUsage::COMPOSER_CURSOR,
            ::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_ENCODER,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_OUTPUT,
            ::android::hardware::graphics::common::V1_1::BufferUsage::CAMERA_INPUT,
            ::android::hardware::graphics::common::V1_1::BufferUsage::RENDERSCRIPT,
            ::android::hardware::graphics::common::V1_1::BufferUsage::VIDEO_DECODER,
            ::android::hardware::graphics::common::V1_1::BufferUsage::SENSOR_DIRECT_DATA,
            ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_DATA_BUFFER,
            ::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK,
            ::android::hardware::graphics::common::V1_1::BufferUsage::VENDOR_MASK_HI,
            ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_CUBE_MAP,
            ::android::hardware::graphics::common::V1_1::BufferUsage::GPU_MIPMAP_COMPLETE,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_1::Dataspace>
{
    const ::android::hardware::graphics::common::V1_1::Dataspace* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_1::Dataspace* end() { return begin() + 61; }
    private:
    static const ::android::hardware::graphics::common::V1_1::Dataspace* static_begin() {
        static const ::android::hardware::graphics::common::V1_1::Dataspace kVals[61] {
            ::android::hardware::graphics::common::V1_1::Dataspace::UNKNOWN,
            ::android::hardware::graphics::common::V1_1::Dataspace::ARBITRARY,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_SHIFT,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_MASK,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_UNSPECIFIED,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT709,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_625_UNADJUSTED,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT601_525_UNADJUSTED,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_BT470M,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_FILM,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_DCI_P3,
            ::android::hardware::graphics::common::V1_1::Dataspace::STANDARD_ADOBE_RGB,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SHIFT,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_MASK,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_UNSPECIFIED,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_LINEAR,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SRGB,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_SMPTE_170M,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_2,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_6,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_GAMMA2_8,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_ST2084,
            ::android::hardware::graphics::common::V1_1::Dataspace::TRANSFER_HLG,
            ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_SHIFT,
            ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_MASK,
            ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_UNSPECIFIED,
            ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_FULL,
            ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_LIMITED,
            ::android::hardware::graphics::common::V1_1::Dataspace::RANGE_EXTENDED,
            ::android::hardware::graphics::common::V1_1::Dataspace::SRGB_LINEAR,
            ::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB_LINEAR,
            ::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB_LINEAR,
            ::android::hardware::graphics::common::V1_1::Dataspace::SRGB,
            ::android::hardware::graphics::common::V1_1::Dataspace::V0_SRGB,
            ::android::hardware::graphics::common::V1_1::Dataspace::V0_SCRGB,
            ::android::hardware::graphics::common::V1_1::Dataspace::JFIF,
            ::android::hardware::graphics::common::V1_1::Dataspace::V0_JFIF,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT601_625,
            ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_625,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT601_525,
            ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT601_525,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT709,
            ::android::hardware::graphics::common::V1_1::Dataspace::V0_BT709,
            ::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3_LINEAR,
            ::android::hardware::graphics::common::V1_1::Dataspace::DCI_P3,
            ::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3_LINEAR,
            ::android::hardware::graphics::common::V1_1::Dataspace::DISPLAY_P3,
            ::android::hardware::graphics::common::V1_1::Dataspace::ADOBE_RGB,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_LINEAR,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT2020,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_PQ,
            ::android::hardware::graphics::common::V1_1::Dataspace::DEPTH,
            ::android::hardware::graphics::common::V1_1::Dataspace::SENSOR,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_PQ,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_ITU_HLG,
            ::android::hardware::graphics::common::V1_1::Dataspace::BT2020_HLG,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_1::ColorMode>
{
    const ::android::hardware::graphics::common::V1_1::ColorMode* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_1::ColorMode* end() { return begin() + 13; }
    private:
    static const ::android::hardware::graphics::common::V1_1::ColorMode* static_begin() {
        static const ::android::hardware::graphics::common::V1_1::ColorMode kVals[13] {
            ::android::hardware::graphics::common::V1_1::ColorMode::NATIVE,
            ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625,
            ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_625_UNADJUSTED,
            ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525,
            ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT601_525_UNADJUSTED,
            ::android::hardware::graphics::common::V1_1::ColorMode::STANDARD_BT709,
            ::android::hardware::graphics::common::V1_1::ColorMode::DCI_P3,
            ::android::hardware::graphics::common::V1_1::ColorMode::SRGB,
            ::android::hardware::graphics::common::V1_1::ColorMode::ADOBE_RGB,
            ::android::hardware::graphics::common::V1_1::ColorMode::DISPLAY_P3,
            ::android::hardware::graphics::common::V1_1::ColorMode::BT2020,
            ::android::hardware::graphics::common::V1_1::ColorMode::BT2100_PQ,
            ::android::hardware::graphics::common::V1_1::ColorMode::BT2100_HLG,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::common::V1_1::RenderIntent>
{
    const ::android::hardware::graphics::common::V1_1::RenderIntent* begin() { return static_begin(); }
    const ::android::hardware::graphics::common::V1_1::RenderIntent* end() { return begin() + 4; }
    private:
    static const ::android::hardware::graphics::common::V1_1::RenderIntent* static_begin() {
        static const ::android::hardware::graphics::common::V1_1::RenderIntent kVals[4] {
            ::android::hardware::graphics::common::V1_1::RenderIntent::COLORIMETRIC,
            ::android::hardware::graphics::common::V1_1::RenderIntent::ENHANCE,
            ::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_COLORIMETRIC,
            ::android::hardware::graphics::common::V1_1::RenderIntent::TONE_MAP_ENHANCE,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_1_TYPES_H
