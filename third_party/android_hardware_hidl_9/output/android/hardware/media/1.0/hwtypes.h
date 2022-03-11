#ifndef HIDL_GENERATED_ANDROID_HARDWARE_MEDIA_V1_0_HWTYPES_H
#define HIDL_GENERATED_ANDROID_HARDWARE_MEDIA_V1_0_HWTYPES_H

#include <android/hardware/media/1.0/types.h>

#include <android/hardware/graphics/common/1.0/hwtypes.h>

#include <hidl/Status.h>
#include <hwbinder/IBinder.h>
#include <hwbinder/Parcel.h>

namespace android {
namespace hardware {
namespace media {
namespace V1_0 {
::android::status_t readEmbeddedFromParcel(
        const ::android::hardware::media::V1_0::AnwBuffer &obj,
        const ::android::hardware::Parcel &parcel,
        size_t parentHandle,
        size_t parentOffset);

::android::status_t writeEmbeddedToParcel(
        const ::android::hardware::media::V1_0::AnwBuffer &obj,
        ::android::hardware::Parcel *parcel,
        size_t parentHandle,
        size_t parentOffset);

}  // namespace V1_0
}  // namespace media
}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_MEDIA_V1_0_HWTYPES_H
