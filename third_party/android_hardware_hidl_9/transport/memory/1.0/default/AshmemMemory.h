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

#ifndef ANDROID_HIDL_ASHMEM_MEMORY_V1_0_MEMORY_H
#define ANDROID_HIDL_ASHMEM_MEMORY_V1_0_MEMORY_H

#include <android/hidl/memory/1.0/IMemory.h>
#include <hidl/MQDescriptor.h>
#include <hidl/Status.h>

namespace android {
namespace hidl {
namespace memory {
namespace V1_0 {
namespace implementation {

using ::android::hidl::memory::V1_0::IMemory;
using ::android::hardware::hidl_array;
using ::android::hardware::hidl_memory;
using ::android::hardware::hidl_string;
using ::android::hardware::hidl_vec;
using ::android::hardware::Return;
using ::android::hardware::Void;
using ::android::sp;

struct AshmemMemory : public IMemory {

    AshmemMemory(const hidl_memory& memory, void* mappedMemory);
    ~AshmemMemory();

    // Methods from ::android::hidl::memory::V1_0::IMemory follow.
    Return<void> update() override;
    Return<void> updateRange(uint64_t start, uint64_t length) override;
    Return<void> read() override;
    Return<void> readRange(uint64_t start, uint64_t length) override;
    Return<void> commit() override;
    Return<void*> getPointer() override;
    Return<uint64_t> getSize() override;

private:
    // Holding onto hidl_memory reference because it contains
    // handle and size, and handle will also be required for
    // the remoted case.
    hidl_memory mMemory;

    // Mapped memory in process.
    void* mData;
};

}  // namespace implementation
}  // namespace V1_0
}  // namespace memory
}  // namespace hidl
}  // namespace android

#endif  // ANDROID_HIDL_ASHMEM_MEMORY_V1_0_MEMORY_H
