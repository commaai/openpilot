#ifndef HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_TYPES_H
#define HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_TYPES_H

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hidl {
namespace base {
namespace V1_0 {

// Forward declaration for forward reference support:
struct DebugInfo;

struct DebugInfo final {
    // Forward declaration for forward reference support:
    enum class Architecture : int32_t;

    enum class Architecture : int32_t {
        UNKNOWN = 0,
        IS_64BIT = 1, // (::android::hidl::base::V1_0::DebugInfo::Architecture.UNKNOWN implicitly + 1)
        IS_32BIT = 2, // (::android::hidl::base::V1_0::DebugInfo::Architecture.IS_64BIT implicitly + 1)
    };

    int32_t pid __attribute__ ((aligned(4)));
    uint64_t ptr __attribute__ ((aligned(8)));
    ::android::hidl::base::V1_0::DebugInfo::Architecture arch __attribute__ ((aligned(4)));
};

static_assert(offsetof(::android::hidl::base::V1_0::DebugInfo, pid) == 0, "wrong offset");
static_assert(offsetof(::android::hidl::base::V1_0::DebugInfo, ptr) == 8, "wrong offset");
static_assert(offsetof(::android::hidl::base::V1_0::DebugInfo, arch) == 16, "wrong offset");
static_assert(sizeof(::android::hidl::base::V1_0::DebugInfo) == 24, "wrong size");
static_assert(__alignof(::android::hidl::base::V1_0::DebugInfo) == 8, "wrong alignment");

constexpr int32_t operator|(const ::android::hidl::base::V1_0::DebugInfo::Architecture lhs, const ::android::hidl::base::V1_0::DebugInfo::Architecture rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hidl::base::V1_0::DebugInfo::Architecture rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hidl::base::V1_0::DebugInfo::Architecture lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hidl::base::V1_0::DebugInfo::Architecture lhs, const ::android::hidl::base::V1_0::DebugInfo::Architecture rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hidl::base::V1_0::DebugInfo::Architecture rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hidl::base::V1_0::DebugInfo::Architecture lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hidl::base::V1_0::DebugInfo::Architecture e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hidl::base::V1_0::DebugInfo::Architecture e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hidl::base::V1_0::DebugInfo::Architecture>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hidl::base::V1_0::DebugInfo::Architecture> flipped = 0;
    bool first = true;
    if ((o & ::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN) == static_cast<int32_t>(::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN)) {
        os += (first ? "" : " | ");
        os += "UNKNOWN";
        first = false;
        flipped |= ::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN;
    }
    if ((o & ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT) == static_cast<int32_t>(::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT)) {
        os += (first ? "" : " | ");
        os += "IS_64BIT";
        first = false;
        flipped |= ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT;
    }
    if ((o & ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT) == static_cast<int32_t>(::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT)) {
        os += (first ? "" : " | ");
        os += "IS_32BIT";
        first = false;
        flipped |= ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hidl::base::V1_0::DebugInfo::Architecture o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN) {
        return "UNKNOWN";
    }
    if (o == ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT) {
        return "IS_64BIT";
    }
    if (o == ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT) {
        return "IS_32BIT";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

static inline std::string toString(const ::android::hidl::base::V1_0::DebugInfo& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".pid = ";
    os += ::android::hardware::toString(o.pid);
    os += ", .ptr = ";
    os += ::android::hardware::toString(o.ptr);
    os += ", .arch = ";
    os += ::android::hidl::base::V1_0::toString(o.arch);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hidl::base::V1_0::DebugInfo& lhs, const ::android::hidl::base::V1_0::DebugInfo& rhs) {
    if (lhs.pid != rhs.pid) {
        return false;
    }
    if (lhs.ptr != rhs.ptr) {
        return false;
    }
    if (lhs.arch != rhs.arch) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hidl::base::V1_0::DebugInfo& lhs,const ::android::hidl::base::V1_0::DebugInfo& rhs){
    return !(lhs == rhs);
}


}  // namespace V1_0
}  // namespace base
}  // namespace hidl
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hidl::base::V1_0::DebugInfo::Architecture>
{
    const ::android::hidl::base::V1_0::DebugInfo::Architecture* begin() { return static_begin(); }
    const ::android::hidl::base::V1_0::DebugInfo::Architecture* end() { return begin() + 3; }
    private:
    static const ::android::hidl::base::V1_0::DebugInfo::Architecture* static_begin() {
        static const ::android::hidl::base::V1_0::DebugInfo::Architecture kVals[3] {
            ::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN,
            ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT,
            ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_TYPES_H
