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

#ifndef ANDROID_HIDL_PASSTHROUGH_SUPPORT_H
#define ANDROID_HIDL_PASSTHROUGH_SUPPORT_H

#include <android/hidl/base/1.0/IBase.h>

namespace android {
namespace hardware {
namespace details {

/*
 * Wrap the given interface with the lowest BsChild possible. Will return the
 * argument directly if nullptr or isRemote().
 *
 * Note that 'static_cast<IFoo*>(wrapPassthrough(foo).get()) is guaranteed to work'
 * assuming that foo is an instance of IFoo.
 *
 * TODO(b/33754152): calling this method multiple times should not re-wrap.
 */
sp<::android::hidl::base::V1_0::IBase> wrapPassthroughInternal(
    sp<::android::hidl::base::V1_0::IBase> iface);

/**
 * Helper method which provides reasonable code to wrapPassthroughInternal
 * which can be used to call wrapPassthrough.
 */
template <typename IType,
          typename = std::enable_if_t<std::is_same<i_tag, typename IType::_hidl_tag>::value>>
sp<IType> wrapPassthrough(sp<IType> iface) {
    return static_cast<IType*>(wrapPassthroughInternal(iface).get());
}

}  // namespace details
}  // namespace hardware
}  // namespace android


#endif  // ANDROID_HIDL_PASSTHROUGH_SUPPORT_H
