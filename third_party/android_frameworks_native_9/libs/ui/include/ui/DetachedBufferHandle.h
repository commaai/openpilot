/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef ANDROID_DETACHED_BUFFER_HUB_HANDLE_H
#define ANDROID_DETACHED_BUFFER_HUB_HANDLE_H

#include <pdx/channel_handle.h>

#include <memory>

namespace android {

// A wrapper that holds a pdx::LocalChannelHandle object. From the handle, a BufferHub buffer can be
// created. Current implementation assumes that the underlying transport is using libpdx (thus
// holding a pdx::LocalChannelHandle object), but future implementation can change it to a Binder
// backend if ever needed.
class DetachedBufferHandle {
public:
    static std::unique_ptr<DetachedBufferHandle> Create(pdx::LocalChannelHandle handle) {
        return std::unique_ptr<DetachedBufferHandle>(new DetachedBufferHandle(std::move(handle)));
    }

    // Accessors to get or take the internal pdx::LocalChannelHandle.
    pdx::LocalChannelHandle& handle() { return mHandle; }
    const pdx::LocalChannelHandle& handle() const { return mHandle; }

    // Returns whether the DetachedBufferHandle holds a BufferHub channel.
    bool isValid() const { return mHandle.valid(); }

private:
    // Constructs a DetachedBufferHandle from a pdx::LocalChannelHandle.
    explicit DetachedBufferHandle(pdx::LocalChannelHandle handle) : mHandle(std::move(handle)) {}

    pdx::LocalChannelHandle mHandle;
};

} // namespace android

#endif // ANDROID_DETACHED_BUFFER_HUB_HANDLE_H
