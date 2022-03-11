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

#ifndef ANDROID_HIDL_ASHMEM_ALLOCATOR_V1_0_ALLOCATOR_H
#define ANDROID_HIDL_ASHMEM_ALLOCATOR_V1_0_ALLOCATOR_H

#include <android/hidl/allocator/1.0/IAllocator.h>
#include <hidl/MQDescriptor.h>
#include <hidl/Status.h>

namespace android {
namespace hidl {
namespace allocator {
namespace V1_0 {
namespace implementation {

using ::android::hidl::allocator::V1_0::IAllocator;
using ::android::hardware::hidl_array;
using ::android::hardware::hidl_memory;
using ::android::hardware::hidl_string;
using ::android::hardware::hidl_vec;
using ::android::hardware::Return;
using ::android::hardware::Void;
using ::android::sp;

struct AshmemAllocator : public IAllocator {
    // Methods from ::android::hidl::allocator::V1_0::IAllocator follow.
    Return<void> allocate(uint64_t size, allocate_cb _hidl_cb) override;
    Return<void> batchAllocate(uint64_t size, uint64_t count, batchAllocate_cb _hidl_cb) override;
};

}  // namespace implementation
}  // namespace V1_0
}  // namespace allocator
}  // namespace hidl
}  // namespace android

#endif  // ANDROID_HIDL_ASHMEM_ALLOCATOR_V1_0_ALLOCATOR_H
