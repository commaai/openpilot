#ifndef HIDL_GENERATED_ANDROID_HARDWARE_MEDIA_V1_0_TYPES_H
#define HIDL_GENERATED_ANDROID_HARDWARE_MEDIA_V1_0_TYPES_H

#include <android/hardware/graphics/common/1.0/types.h>

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hardware {
namespace media {
namespace V1_0 {

// Forward declaration for forward reference support:
struct AnwBufferAttributes;
struct AnwBuffer;
struct Rect;

/**
 * Aliases
 */
typedef ::android::hardware::hidl_handle FileDescriptor;

typedef ::android::hardware::hidl_handle Fence;

typedef ::android::hardware::hidl_vec<uint8_t> Bytes;

/**
 * Ref: frameworks/native/include/ui/GraphicBuffer.h
 * Ref: system/core/include/system/window.h: ANativeWindowBuffer
 * 
 * 
 * This struct contains attributes for a gralloc buffer that can be put into a
 * union.
 */
struct AnwBufferAttributes final {
    uint32_t width __attribute__ ((aligned(4)));
    uint32_t height __attribute__ ((aligned(4)));
    uint32_t stride __attribute__ ((aligned(4)));
    ::android::hardware::graphics::common::V1_0::PixelFormat format __attribute__ ((aligned(4)));
    uint32_t usage __attribute__ ((aligned(4)));
    uint32_t generationNumber __attribute__ ((aligned(4)));
    uint64_t layerCount __attribute__ ((aligned(8)));
    uint64_t id __attribute__ ((aligned(8)));
};

static_assert(offsetof(::android::hardware::media::V1_0::AnwBufferAttributes, width) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::AnwBufferAttributes, height) == 4, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::AnwBufferAttributes, stride) == 8, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::AnwBufferAttributes, format) == 12, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::AnwBufferAttributes, usage) == 16, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::AnwBufferAttributes, generationNumber) == 20, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::AnwBufferAttributes, layerCount) == 24, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::AnwBufferAttributes, id) == 32, "wrong offset");
static_assert(sizeof(::android::hardware::media::V1_0::AnwBufferAttributes) == 40, "wrong size");
static_assert(__alignof(::android::hardware::media::V1_0::AnwBufferAttributes) == 8, "wrong alignment");

/**
 * An AnwBuffer is simply AnwBufferAttributes plus a native handle.
 */
struct AnwBuffer final {
    ::android::hardware::hidl_handle nativeHandle __attribute__ ((aligned(8)));
    ::android::hardware::media::V1_0::AnwBufferAttributes attr __attribute__ ((aligned(8)));
};

static_assert(offsetof(::android::hardware::media::V1_0::AnwBuffer, nativeHandle) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::AnwBuffer, attr) == 16, "wrong offset");
static_assert(sizeof(::android::hardware::media::V1_0::AnwBuffer) == 56, "wrong size");
static_assert(__alignof(::android::hardware::media::V1_0::AnwBuffer) == 8, "wrong alignment");

/**
 * Ref: frameworks/native/include/android/rect.h
 * Ref: frameworks/native/include/ui/Rect.h
 */
struct Rect final {
    int32_t left __attribute__ ((aligned(4)));
    int32_t top __attribute__ ((aligned(4)));
    int32_t right __attribute__ ((aligned(4)));
    int32_t bottom __attribute__ ((aligned(4)));
};

static_assert(offsetof(::android::hardware::media::V1_0::Rect, left) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::Rect, top) == 4, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::Rect, right) == 8, "wrong offset");
static_assert(offsetof(::android::hardware::media::V1_0::Rect, bottom) == 12, "wrong offset");
static_assert(sizeof(::android::hardware::media::V1_0::Rect) == 16, "wrong size");
static_assert(__alignof(::android::hardware::media::V1_0::Rect) == 4, "wrong alignment");

/**
 * Ref: frameworks/native/include/ui/Region.h
 */
typedef ::android::hardware::hidl_vec<::android::hardware::media::V1_0::Rect> Region;

static inline std::string toString(const ::android::hardware::media::V1_0::AnwBufferAttributes& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".width = ";
    os += ::android::hardware::toString(o.width);
    os += ", .height = ";
    os += ::android::hardware::toString(o.height);
    os += ", .stride = ";
    os += ::android::hardware::toString(o.stride);
    os += ", .format = ";
    os += ::android::hardware::graphics::common::V1_0::toString(o.format);
    os += ", .usage = ";
    os += ::android::hardware::toString(o.usage);
    os += ", .generationNumber = ";
    os += ::android::hardware::toString(o.generationNumber);
    os += ", .layerCount = ";
    os += ::android::hardware::toString(o.layerCount);
    os += ", .id = ";
    os += ::android::hardware::toString(o.id);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::media::V1_0::AnwBufferAttributes& lhs, const ::android::hardware::media::V1_0::AnwBufferAttributes& rhs) {
    if (lhs.width != rhs.width) {
        return false;
    }
    if (lhs.height != rhs.height) {
        return false;
    }
    if (lhs.stride != rhs.stride) {
        return false;
    }
    if (lhs.format != rhs.format) {
        return false;
    }
    if (lhs.usage != rhs.usage) {
        return false;
    }
    if (lhs.generationNumber != rhs.generationNumber) {
        return false;
    }
    if (lhs.layerCount != rhs.layerCount) {
        return false;
    }
    if (lhs.id != rhs.id) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::media::V1_0::AnwBufferAttributes& lhs,const ::android::hardware::media::V1_0::AnwBufferAttributes& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::media::V1_0::AnwBuffer& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".nativeHandle = ";
    os += ::android::hardware::toString(o.nativeHandle);
    os += ", .attr = ";
    os += ::android::hardware::media::V1_0::toString(o.attr);
    os += "}"; return os;
}

// operator== and operator!= are not generated for AnwBuffer

static inline std::string toString(const ::android::hardware::media::V1_0::Rect& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".left = ";
    os += ::android::hardware::toString(o.left);
    os += ", .top = ";
    os += ::android::hardware::toString(o.top);
    os += ", .right = ";
    os += ::android::hardware::toString(o.right);
    os += ", .bottom = ";
    os += ::android::hardware::toString(o.bottom);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::media::V1_0::Rect& lhs, const ::android::hardware::media::V1_0::Rect& rhs) {
    if (lhs.left != rhs.left) {
        return false;
    }
    if (lhs.top != rhs.top) {
        return false;
    }
    if (lhs.right != rhs.right) {
        return false;
    }
    if (lhs.bottom != rhs.bottom) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::media::V1_0::Rect& lhs,const ::android::hardware::media::V1_0::Rect& rhs){
    return !(lhs == rhs);
}


}  // namespace V1_0
}  // namespace media
}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_MEDIA_V1_0_TYPES_H
