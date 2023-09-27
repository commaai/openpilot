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

#ifndef ANDROID_GRAPHIC_BUFFER_H
#define ANDROID_GRAPHIC_BUFFER_H

#include <stdint.h>
#include <sys/types.h>

#include <ui/ANativeObjectBase.h>
#include <ui/PixelFormat.h>
#include <ui/Rect.h>
#include <utils/Flattenable.h>
#include <utils/RefBase.h>


struct ANativeWindowBuffer;

namespace android {

class GraphicBufferMapper;

// ===========================================================================
// GraphicBuffer
// ===========================================================================

class GraphicBuffer
    : public ANativeObjectBase< ANativeWindowBuffer, GraphicBuffer, RefBase >,
      public Flattenable<GraphicBuffer>
{
    friend class Flattenable<GraphicBuffer>;
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

        USAGE_PROTECTED         = GRALLOC_USAGE_PROTECTED,

        USAGE_HW_TEXTURE        = GRALLOC_USAGE_HW_TEXTURE,
        USAGE_HW_RENDER         = GRALLOC_USAGE_HW_RENDER,
        USAGE_HW_2D             = GRALLOC_USAGE_HW_2D,
        USAGE_HW_COMPOSER       = GRALLOC_USAGE_HW_COMPOSER,
        USAGE_HW_VIDEO_ENCODER  = GRALLOC_USAGE_HW_VIDEO_ENCODER,
        USAGE_HW_MASK           = GRALLOC_USAGE_HW_MASK,

        USAGE_CURSOR            = GRALLOC_USAGE_CURSOR,
    };

    GraphicBuffer();

    // creates w * h buffer
    GraphicBuffer(uint32_t inWidth, uint32_t inHeight, PixelFormat inFormat,
            uint32_t inUsage);

    // create a buffer from an existing handle
    GraphicBuffer(uint32_t inWidth, uint32_t inHeight, PixelFormat inFormat,
            uint32_t inUsage, uint32_t inStride, native_handle_t* inHandle,
            bool keepOwnership);

    // create a buffer from an existing ANativeWindowBuffer
    GraphicBuffer(ANativeWindowBuffer* buffer, bool keepOwnership);

    // return status
    status_t initCheck() const;

    uint32_t getWidth() const           { return static_cast<uint32_t>(width); }
    uint32_t getHeight() const          { return static_cast<uint32_t>(height); }
    uint32_t getStride() const          { return static_cast<uint32_t>(stride); }
    uint32_t getUsage() const           { return static_cast<uint32_t>(usage); }
    PixelFormat getPixelFormat() const  { return format; }
    Rect getBounds() const              { return Rect(width, height); }
    uint64_t getId() const              { return mId; }

    uint32_t getGenerationNumber() const { return mGenerationNumber; }
    void setGenerationNumber(uint32_t generation) {
        mGenerationNumber = generation;
    }

    status_t reallocate(uint32_t inWidth, uint32_t inHeight,
            PixelFormat inFormat, uint32_t inUsage);

    bool needsReallocation(uint32_t inWidth, uint32_t inHeight,
            PixelFormat inFormat, uint32_t inUsage);

    status_t lock(uint32_t inUsage, void** vaddr);
    status_t lock(uint32_t inUsage, const Rect& rect, void** vaddr);
    // For HAL_PIXEL_FORMAT_YCbCr_420_888
    status_t lockYCbCr(uint32_t inUsage, android_ycbcr *ycbcr);
    status_t lockYCbCr(uint32_t inUsage, const Rect& rect,
            android_ycbcr *ycbcr);
    status_t unlock();
    status_t lockAsync(uint32_t inUsage, void** vaddr, int fenceFd);
    status_t lockAsync(uint32_t inUsage, const Rect& rect, void** vaddr,
            int fenceFd);
    status_t lockAsyncYCbCr(uint32_t inUsage, android_ycbcr *ycbcr,
            int fenceFd);
    status_t lockAsyncYCbCr(uint32_t inUsage, const Rect& rect,
            android_ycbcr *ycbcr, int fenceFd);
    status_t unlockAsync(int *fenceFd);

    ANativeWindowBuffer* getNativeBuffer() const;

    // for debugging
    static void dumpAllocationsToSystemLog();

    // Flattenable protocol
    size_t getFlattenedSize() const;
    size_t getFdCount() const;
    status_t flatten(void*& buffer, size_t& size, int*& fds, size_t& count) const;
    status_t unflatten(void const*& buffer, size_t& size, int const*& fds, size_t& count);

private:
    ~GraphicBuffer();

    enum {
        ownNone   = 0,
        ownHandle = 1,
        ownData   = 2,
    };

    inline const GraphicBufferMapper& getBufferMapper() const {
        return mBufferMapper;
    }
    inline GraphicBufferMapper& getBufferMapper() {
        return mBufferMapper;
    }
    uint8_t mOwner;

private:
    friend class Surface;
    friend class BpSurface;
    friend class BnSurface;
    friend class LightRefBase<GraphicBuffer>;
    GraphicBuffer(const GraphicBuffer& rhs);
    GraphicBuffer& operator = (const GraphicBuffer& rhs);
    const GraphicBuffer& operator = (const GraphicBuffer& rhs) const;

    status_t initSize(uint32_t inWidth, uint32_t inHeight, PixelFormat inFormat,
            uint32_t inUsage);

    void free_handle();

    GraphicBufferMapper& mBufferMapper;
    ssize_t mInitCheck;

    // If we're wrapping another buffer then this reference will make sure it
    // doesn't get freed.
    sp<ANativeWindowBuffer> mWrappedBuffer;

    uint64_t mId;

    // Stores the generation number of this buffer. If this number does not
    // match the BufferQueue's internal generation number (set through
    // IGBP::setGenerationNumber), attempts to attach the buffer will fail.
    uint32_t mGenerationNumber;
};

}; // namespace android

#endif // ANDROID_GRAPHIC_BUFFER_H
