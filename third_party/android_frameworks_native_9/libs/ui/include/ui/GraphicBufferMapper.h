/*
 * Copyright (C) 2007 The Android Open Source Project
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

#ifndef ANDROID_UI_BUFFER_MAPPER_H
#define ANDROID_UI_BUFFER_MAPPER_H

#include <stdint.h>
#include <sys/types.h>

#include <memory>

#include <ui/PixelFormat.h>
#include <utils/Singleton.h>


// Needed by code that still uses the GRALLOC_USAGE_* constants.
// when/if we get rid of gralloc, we should provide aliases or fix call sites.
#include <hardware/gralloc.h>


namespace android {

// ---------------------------------------------------------------------------

namespace Gralloc2 {
class Mapper;
}

class Rect;

class GraphicBufferMapper : public Singleton<GraphicBufferMapper>
{
public:
    static void preloadHal();
    static inline GraphicBufferMapper& get() { return getInstance(); }

    // The imported outHandle must be freed with freeBuffer when no longer
    // needed. rawHandle is owned by the caller.
    status_t importBuffer(buffer_handle_t rawHandle,
            uint32_t width, uint32_t height, uint32_t layerCount,
            PixelFormat format, uint64_t usage, uint32_t stride,
            buffer_handle_t* outHandle);

    status_t freeBuffer(buffer_handle_t handle);

    void getTransportSize(buffer_handle_t handle,
            uint32_t* outTransportNumFds, uint32_t* outTransportNumInts);

    status_t lock(buffer_handle_t handle,
            uint32_t usage, const Rect& bounds, void** vaddr);

    status_t lockYCbCr(buffer_handle_t handle,
            uint32_t usage, const Rect& bounds, android_ycbcr *ycbcr);

    status_t unlock(buffer_handle_t handle);

    status_t lockAsync(buffer_handle_t handle,
            uint32_t usage, const Rect& bounds, void** vaddr, int fenceFd);

    status_t lockAsync(buffer_handle_t handle,
            uint64_t producerUsage, uint64_t consumerUsage, const Rect& bounds,
            void** vaddr, int fenceFd);

    status_t lockAsyncYCbCr(buffer_handle_t handle,
            uint32_t usage, const Rect& bounds, android_ycbcr *ycbcr,
            int fenceFd);

    status_t unlockAsync(buffer_handle_t handle, int *fenceFd);

    const Gralloc2::Mapper& getGrallocMapper() const
    {
        return *mMapper;
    }

private:
    friend class Singleton<GraphicBufferMapper>;

    GraphicBufferMapper();

    const std::unique_ptr<const Gralloc2::Mapper> mMapper;
};

// ---------------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_UI_BUFFER_MAPPER_H

