/*
 * Copyright 2014 The Android Open Source Project
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

#ifndef ANDROID_GUI_BUFFERQUEUECONSUMER_H
#define ANDROID_GUI_BUFFERQUEUECONSUMER_H

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <gui/BufferQueueDefs.h>
#include <gui/IGraphicBufferConsumer.h>
#include <utils/String8.h>

namespace android {

class BufferQueueCore;

class BufferQueueConsumer : public BnGraphicBufferConsumer {

public:
    BufferQueueConsumer(const sp<BufferQueueCore>& core);
    ~BufferQueueConsumer() override;

    // acquireBuffer attempts to acquire ownership of the next pending buffer in
    // the BufferQueue. If no buffer is pending then it returns
    // NO_BUFFER_AVAILABLE. If a buffer is successfully acquired, the
    // information about the buffer is returned in BufferItem.  If the buffer
    // returned had previously been acquired then the BufferItem::mGraphicBuffer
    // field of buffer is set to NULL and it is assumed that the consumer still
    // holds a reference to the buffer.
    //
    // If expectedPresent is nonzero, it indicates the time when the buffer
    // will be displayed on screen. If the buffer's timestamp is farther in the
    // future, the buffer won't be acquired, and PRESENT_LATER will be
    // returned.  The presentation time is in nanoseconds, and the time base
    // is CLOCK_MONOTONIC.
    virtual status_t acquireBuffer(BufferItem* outBuffer,
            nsecs_t expectedPresent, uint64_t maxFrameNumber = 0) override;

    // See IGraphicBufferConsumer::detachBuffer
    virtual status_t detachBuffer(int slot);

    // See IGraphicBufferConsumer::attachBuffer
    virtual status_t attachBuffer(int* slot, const sp<GraphicBuffer>& buffer);

    // releaseBuffer releases a buffer slot from the consumer back to the
    // BufferQueue.  This may be done while the buffer's contents are still
    // being accessed.  The fence will signal when the buffer is no longer
    // in use. frameNumber is used to indentify the exact buffer returned.
    //
    // If releaseBuffer returns STALE_BUFFER_SLOT, then the consumer must free
    // any references to the just-released buffer that it might have, as if it
    // had received a onBuffersReleased() call with a mask set for the released
    // buffer.
    //
    // Note that the dependencies on EGL will be removed once we switch to using
    // the Android HW Sync HAL.
    virtual status_t releaseBuffer(int slot, uint64_t frameNumber,
            const sp<Fence>& releaseFence, EGLDisplay display,
            EGLSyncKHR fence);

    // connect connects a consumer to the BufferQueue.  Only one
    // consumer may be connected, and when that consumer disconnects the
    // BufferQueue is placed into the "abandoned" state, causing most
    // interactions with the BufferQueue by the producer to fail.
    // controlledByApp indicates whether the consumer is controlled by
    // the application.
    //
    // consumerListener may not be NULL.
    virtual status_t connect(const sp<IConsumerListener>& consumerListener,
            bool controlledByApp);

    // disconnect disconnects a consumer from the BufferQueue. All
    // buffers will be freed and the BufferQueue is placed in the "abandoned"
    // state, causing most interactions with the BufferQueue by the producer to
    // fail.
    virtual status_t disconnect();

    // getReleasedBuffers sets the value pointed to by outSlotMask to a bit mask
    // indicating which buffer slots have been released by the BufferQueue
    // but have not yet been released by the consumer.
    //
    // This should be called from the onBuffersReleased() callback.
    virtual status_t getReleasedBuffers(uint64_t* outSlotMask);

    // setDefaultBufferSize is used to set the size of buffers returned by
    // dequeueBuffer when a width and height of zero is requested.  Default
    // is 1x1.
    virtual status_t setDefaultBufferSize(uint32_t width, uint32_t height);

    // see IGraphicBufferConsumer::setMaxBufferCount
    virtual status_t setMaxBufferCount(int bufferCount);

    // setMaxAcquiredBufferCount sets the maximum number of buffers that can
    // be acquired by the consumer at one time (default 1).  This call will
    // fail if a producer is connected to the BufferQueue.
    virtual status_t setMaxAcquiredBufferCount(int maxAcquiredBuffers);

    // setConsumerName sets the name used in logging
    status_t setConsumerName(const String8& name) override;

    // setDefaultBufferFormat allows the BufferQueue to create
    // GraphicBuffers of a defaultFormat if no format is specified
    // in dequeueBuffer. The initial default is HAL_PIXEL_FORMAT_RGBA_8888.
    virtual status_t setDefaultBufferFormat(PixelFormat defaultFormat);

    // setDefaultBufferDataSpace allows the BufferQueue to create
    // GraphicBuffers of a defaultDataSpace if no data space is specified
    // in queueBuffer.
    // The initial default is HAL_DATASPACE_UNKNOWN
    virtual status_t setDefaultBufferDataSpace(android_dataspace defaultDataSpace);

    // setConsumerUsageBits will turn on additional usage bits for dequeueBuffer.
    // These are merged with the bits passed to dequeueBuffer.  The values are
    // enumerated in gralloc.h, e.g. GRALLOC_USAGE_HW_RENDER; the default is 0.
    virtual status_t setConsumerUsageBits(uint64_t usage) override;

    // setConsumerIsProtected will turn on an internal bit that indicates whether
    // the consumer can handle protected gralloc buffers (i.e. with
    // GRALLOC_USAGE_PROTECTED set). IGraphicBufferProducer can query this
    // capability using NATIVE_WINDOW_CONSUMER_IS_PROTECTED.
    virtual status_t setConsumerIsProtected(bool isProtected);

    // setTransformHint bakes in rotation to buffers so overlays can be used.
    // The values are enumerated in window.h, e.g.
    // NATIVE_WINDOW_TRANSFORM_ROT_90.  The default is 0 (no transform).
    virtual status_t setTransformHint(uint32_t hint);

    // Retrieve the sideband buffer stream, if any.
    status_t getSidebandStream(sp<NativeHandle>* outStream) const override;

    // See IGraphicBufferConsumer::getOccupancyHistory
    virtual status_t getOccupancyHistory(bool forceFlush,
            std::vector<OccupancyTracker::Segment>* outHistory) override;

    // See IGraphicBufferConsumer::discardFreeBuffers
    virtual status_t discardFreeBuffers() override;

    // dump our state in a String
    status_t dumpState(const String8& prefix, String8* outResult) const override;

    // Functions required for backwards compatibility.
    // These will be modified/renamed in IGraphicBufferConsumer and will be
    // removed from this class at that time. See b/13306289.

    virtual status_t releaseBuffer(int buf, uint64_t frameNumber,
            EGLDisplay display, EGLSyncKHR fence,
            const sp<Fence>& releaseFence) {
        return releaseBuffer(buf, frameNumber, releaseFence, display, fence);
    }

    virtual status_t consumerConnect(const sp<IConsumerListener>& consumer,
            bool controlledByApp) {
        return connect(consumer, controlledByApp);
    }

    virtual status_t consumerDisconnect() { return disconnect(); }

    // End functions required for backwards compatibility

private:
    sp<BufferQueueCore> mCore;

    // This references mCore->mSlots. Lock mCore->mMutex while accessing.
    BufferQueueDefs::SlotsType& mSlots;

    // This is a cached copy of the name stored in the BufferQueueCore.
    // It's updated during setConsumerName.
    String8 mConsumerName;

}; // class BufferQueueConsumer

} // namespace android

#endif
