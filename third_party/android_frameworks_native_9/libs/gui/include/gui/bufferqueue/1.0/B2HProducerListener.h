/*
 * Copyright 2016, The Android Open Source Project
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

#ifndef ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_B2HPRODUCERLISTENER_H
#define ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_B2HPRODUCERLISTENER_H

#include <hidl/MQDescriptor.h>
#include <hidl/Status.h>

#include <binder/IBinder.h>
#include <gui/IProducerListener.h>

#include <android/hidl/base/1.0/IBase.h>
#include <android/hardware/graphics/bufferqueue/1.0/IProducerListener.h>

namespace android {
namespace hardware {
namespace graphics {
namespace bufferqueue {
namespace V1_0 {
namespace utils {

using ::android::hidl::base::V1_0::IBase;
using ::android::hardware::hidl_array;
using ::android::hardware::hidl_memory;
using ::android::hardware::hidl_string;
using ::android::hardware::hidl_vec;
using ::android::hardware::Return;
using ::android::hardware::Void;
using ::android::sp;

typedef ::android::hardware::graphics::bufferqueue::V1_0::IProducerListener
        HProducerListener;

typedef ::android::IProducerListener
        BProducerListener;

struct B2HProducerListener : public HProducerListener {
    sp<BProducerListener> mBase;
    B2HProducerListener(sp<BProducerListener> const& base);
    Return<void> onBufferReleased() override;
    Return<bool> needsReleaseNotify() override;
};

}  // namespace utils
}  // namespace V1_0
}  // namespace omx
}  // namespace media
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_B2HPRODUCERLISTENER_H

