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

#include <string>

#include <ui/ANativeObjectBase.h>
#include <ui/PixelFormat.h>
#include <ui/Rect.h>
#include <utils/Flattenable.h>
#include <utils/RefBase.h>

#include <nativebase/nativebase.h>

#include <hardware/gralloc.h>

namespace android {

class DetachedBufferHandle;
class GraphicBufferMapper;

// ===========================================================================
// GraphicBuffer
// ===========================================================================

class GraphicBuffer
    : public ANativeObjectBase<ANativeWindowBuffer, GraphicBuffer, RefBase>,
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

    static sp<GraphicBuffer> from(ANativeWindowBuffer *);


    // Create a GraphicBuffer to be unflatten'ed into or be reallocated.
    GraphicBuffer();

    // Create a GraphicBuffer by allocating and managing a buffer internally.
    // This function is privileged.  See reallocate for details.
    GraphicBuffer(uint32_t inWidth, uint32_t inHeight, PixelFormat inFormat,
            uint32_t inLayerCount, uint64_t inUsage,
            std::string requestorName = "<Unknown>");

    // Create a GraphicBuffer from an existing handle.
    enum HandleWrapMethod : uint8_t {
        // Wrap and use the handle directly.  It assumes the handle has been
        // registered and never fails.  The handle must have a longer lifetime
        // than this wrapping GraphicBuffer.
        //
        // This can be used when, for example, you want to wrap a handle that
        // is already managed by another GraphicBuffer.
        WRAP_HANDLE,

        // Take ownership of the handle and use it directly.  It assumes the
        // handle has been registered and never fails.
        //
        // This can be used to manage an already registered handle with
        // GraphicBuffer.
        TAKE_HANDLE,

        // Take onwership of an unregistered handle and use it directly.  It
        // can fail when the buffer does not register.  There is no ownership
        // transfer on failures.
        //
        // This can be used to, for example, create a GraphicBuffer from a
        // handle returned by Parcel::readNativeHandle.
        TAKE_UNREGISTERED_HANDLE,

        // Make a clone of the handle and use the cloned handle.  It can fail
        // when cloning fails or when the buffer does not register.  There is
        // never ownership transfer.
        //
        // This can be used to create a GraphicBuffer from a handle that
        // cannot be used directly, such as one from hidl_handle.
        CLONE_HANDLE,
    };
    GraphicBuffer(const native_handle_t* handle, HandleWrapMethod method,
            uint32_t width, uint32_t height,
            PixelFormat format, uint32_t layerCount,
            uint64_t usage, uint32_t stride);

    // These functions are deprecated because they only take 32 bits of usage
    GraphicBuffer(const native_handle_t* handle, HandleWrapMethod method,
            uint32_t width, uint32_t height,
            PixelFormat format, uint32_t layerCount,
            uint32_t usage, uint32_t stride)
        : GraphicBuffer(handle, method, width, height, format, layerCount,
                static_cast<uint64_t>(usage), stride) {}
    GraphicBuffer(uint32_t inWidth, uint32_t inHeight, PixelFormat inFormat,
            uint32_t inLayerCount, uint32_t inUsage, uint32_t inStride,
            native_handle_t* inHandle, bool keepOwnership);
    GraphicBuffer(uint32_t inWidth, uint32_t inHeight, PixelFormat inFormat,
            uint32_t inUsage, std::string requestorName = "<Unknown>");

    // return status
    status_t initCheck() const;

    uint32_t getWidth() const           { return static_cast<uint32_t>(width); }
    uint32_t getHeight() const          { return static_cast<uint32_t>(height); }
    uint32_t getStride() const          { return static_cast<uint32_t>(stride); }
    uint64_t getUsage() const           { return usage; }
    PixelFormat getPixelFormat() const  { return format; }
    uint32_t getLayerCount() const      { return static_cast<uint32_t>(layerCount); }
    Rect getBounds() const              { return Rect(width, height); }
    uint64_t getId() const              { return mId; }

    uint32_t getGenerationNumber() const { return mGenerationNumber; }
    void setGenerationNumber(uint32_t generation) {
        mGenerationNumber = generation;
    }

    // This function is privileged.  It requires access to the allocator
    // device or service, which usually involves adding suitable selinux
    // rules.
    status_t reallocate(uint32_t inWidth, uint32_t inHeight,
            PixelFormat inFormat, uint32_t inLayerCount, uint64_t inUsage);

    bool needsReallocation(uint32_t inWidth, uint32_t inHeight,
            PixelFormat inFormat, uint32_t inLayerCount, uint64_t inUsage);

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
    status_t lockAsync(uint64_t inProducerUsage, uint64_t inConsumerUsage,
            const Rect& rect, void** vaddr, int fenceFd);
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

    // Sets and takes DetachedBuffer. Should only be called from BufferHub.
    bool isDetachedBuffer() const;
    status_t setDetachedBufferHandle(std::unique_ptr<DetachedBufferHandle> detachedBuffer);
    std::unique_ptr<DetachedBufferHandle> takeDetachedBufferHandle();

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

    status_t initWithSize(uint32_t inWidth, uint32_t inHeight,
            PixelFormat inFormat, uint32_t inLayerCount,
            uint64_t inUsage, std::string requestorName);

    status_t initWithHandle(const native_handle_t* handle,
            HandleWrapMethod method, uint32_t width, uint32_t height,
            PixelFormat format, uint32_t layerCount,
            uint64_t usage, uint32_t stride);

    void free_handle();

    GraphicBufferMapper& mBufferMapper;
    ssize_t mInitCheck;

    // numbers of fds/ints in native_handle_t to flatten
    uint32_t mTransportNumFds;
    uint32_t mTransportNumInts;

    uint64_t mId;

    // Stores the generation number of this buffer. If this number does not
    // match the BufferQueue's internal generation number (set through
    // IGBP::setGenerationNumber), attempts to attach the buffer will fail.
    uint32_t mGenerationNumber;

    // Stores a BufferHub handle that can be used to re-attach this GraphicBuffer back into a
    // BufferHub producer/consumer set. In terms of GraphicBuffer's relationship with BufferHub,
    // there are three different modes:
    // 1. Legacy mode: GraphicBuffer is not backed by BufferHub and mDetachedBufferHandle must be
    //    invalid.
    // 2. Detached mode: GraphicBuffer is backed by BufferHub, but not part of a producer/consumer
    //    set. In this mode, mDetachedBufferHandle must be valid.
    // 3. Attached mode: GraphicBuffer is backed by BufferHub and it's part of a producer/consumer
    //    set. In this mode, mDetachedBufferHandle must be invalid.
    std::unique_ptr<DetachedBufferHandle> mDetachedBufferHandle;
};

}; // namespace android

#endif // ANDROID_GRAPHIC_BUFFER_H
