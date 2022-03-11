/*
**
** Copyright 2009, The Android Open Source Project
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
**     http://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*/

#ifndef ANDROID_BUFFER_ALLOCATOR_H
#define ANDROID_BUFFER_ALLOCATOR_H

#include <stdint.h>

#include <memory>
#include <string>

#include <cutils/native_handle.h>

#include <ui/PixelFormat.h>

#include <utils/Errors.h>
#include <utils/KeyedVector.h>
#include <utils/Mutex.h>
#include <utils/Singleton.h>

namespace android {

namespace Gralloc2 {
class Allocator;
}

class GraphicBufferMapper;
class String8;

class GraphicBufferAllocator : public Singleton<GraphicBufferAllocator>
{
public:
    static inline GraphicBufferAllocator& get() { return getInstance(); }

    status_t allocate(uint32_t w, uint32_t h, PixelFormat format,
            uint32_t layerCount, uint64_t usage,
            buffer_handle_t* handle, uint32_t* stride, uint64_t graphicBufferId,
            std::string requestorName);

    status_t free(buffer_handle_t handle);

    void dump(String8& res) const;
    static void dumpToSystemLog();

private:
    struct alloc_rec_t {
        uint32_t width;
        uint32_t height;
        uint32_t stride;
        PixelFormat format;
        uint32_t layerCount;
        uint64_t usage;
        size_t size;
        std::string requestorName;
    };

    static Mutex sLock;
    static KeyedVector<buffer_handle_t, alloc_rec_t> sAllocList;

    friend class Singleton<GraphicBufferAllocator>;
    GraphicBufferAllocator();
    ~GraphicBufferAllocator();

    GraphicBufferMapper& mMapper;
    const std::unique_ptr<const Gralloc2::Allocator> mAllocator;
};

// ---------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_BUFFER_ALLOCATOR_H
