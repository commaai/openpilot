/*
 * Copyright (C) 2016 The Android Open Source Project
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

#ifndef ANDROID_HIDL_TRANSPORT_UTILS_H
#define ANDROID_HIDL_TRANSPORT_UTILS_H

#include <android/hidl/base/1.0/IBase.h>

namespace android {
namespace hardware {
namespace details {

/*
 * Verifies the interface chain of 'interface' contains 'castTo'
 * @param emitError if emitError is false, return Return<bool>{false} on error; if emitError
 * is true, the Return<bool> object contains the actual error.
 */
Return<bool> canCastInterface(::android::hidl::base::V1_0::IBase* interface,
        const char* castTo, bool emitError = false);

std::string getDescriptor(::android::hidl::base::V1_0::IBase* interface);

}   // namespace details
}   // namespace hardware
}   // namespace android

#endif //ANDROID_HIDL_TRANSPORT_UTILS_H
