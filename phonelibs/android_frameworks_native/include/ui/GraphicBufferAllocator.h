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

#include <cutils/native_handle.h>

#include <utils/Errors.h>
#include <utils/KeyedVector.h>
#include <utils/threads.h>
#include <utils/Singleton.h>

#include <ui/PixelFormat.h>

#include <hardware/gralloc.h>


namespace android {
// ---------------------------------------------------------------------------

class String8;

class GraphicBufferAllocator : public Singleton<GraphicBufferAllocator>
{
public:
    enum {
        USAGE_SW_READ_NEVER     = GRALLOC_USAGE_SW_READ_NEVER,
        USAGE_SW_READ_RARELY    = GRALLOC_USAGE_SW_READ_RARELY,
        USAGE_SW_READ_OFTEN     = GRALLOC_USAGE_SW_READ_OFTEN,
        USAGE_SW_READ_MASK      = GRALLOC_USAGE_SW_READ_MASK,

        USAGE_SW_WRITE_NEVER    = GRALLOC_USAGE_SW_WRITE_NEVER,
        USAGE_SW_WRITE_RARELY   = GRALLOC_USAGE_SW_WRITE_RARELY,
        USAGE_SW_WRITE_OFTEN    = GRALLOC_USAGE_SW_WRITE_OFTEN,
        USAGE_SW_WRITE_MASK     = GRALLOC_USAGE_SW_WRITE_MASK,

        USAGE_SOFTWARE_MASK     = USAGE_SW_READ_MASK|USAGE_SW_WRITE_MASK,

        USAGE_HW_TEXTURE        = GRALLOC_USAGE_HW_TEXTURE,
        USAGE_HW_RENDER         = GRALLOC_USAGE_HW_RENDER,
        USAGE_HW_2D             = GRALLOC_USAGE_HW_2D,
        USAGE_HW_MASK           = GRALLOC_USAGE_HW_MASK
    };

    static inline GraphicBufferAllocator& get() { return getInstance(); }

    status_t alloc(uint32_t w, uint32_t h, PixelFormat format, uint32_t usage,
            buffer_handle_t* handle, uint32_t* stride);

    status_t free(buffer_handle_t handle);

    void dump(String8& res) const;
    static void dumpToSystemLog();

private:
    struct alloc_rec_t {
        uint32_t width;
        uint32_t height;
        uint32_t stride;
        PixelFormat format;
        uint32_t usage;
        size_t size;
    };

    static Mutex sLock;
    static KeyedVector<buffer_handle_t, alloc_rec_t> sAllocList;

    friend class Singleton<GraphicBufferAllocator>;
    GraphicBufferAllocator();
    ~GraphicBufferAllocator();

    alloc_device_t  *mAllocDev;
};

// ---------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_BUFFER_ALLOCATOR_H
