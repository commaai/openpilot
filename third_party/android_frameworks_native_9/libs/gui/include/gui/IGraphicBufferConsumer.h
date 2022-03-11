/*
 * Copyright (C) 2013 The Android Open Source Project
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

#pragma once

#include <gui/OccupancyTracker.h>

#include <binder/IInterface.h>
#include <binder/SafeInterface.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <ui/PixelFormat.h>

#include <utils/Errors.h>

namespace android {

class BufferItem;
class Fence;
class GraphicBuffer;
class IConsumerListener;
class NativeHandle;

class IGraphicBufferConsumer : public IInterface {
public:
    DECLARE_META_INTERFACE(GraphicBufferConsumer)

    enum {
        // Returned by releaseBuffer, after which the consumer must free any references to the
        // just-released buffer that it might have.
        STALE_BUFFER_SLOT = 1,
        // Returned by dequeueBuffer if there are no pending buffers available.
        NO_BUFFER_AVAILABLE,
        // Returned by dequeueBuffer if it's too early for the buffer to be acquired.
        PRESENT_LATER,
    };

    // acquireBuffer attempts to acquire ownership of the next pending buffer in the BufferQueue.
    // If no buffer is pending then it returns NO_BUFFER_AVAILABLE. If a buffer is successfully
    // acquired, the information about the buffer is returned in BufferItem.
    //
    // If the buffer returned had previously been acquired then the BufferItem::mGraphicBuffer field
    // of buffer is set to NULL and it is assumed that the consumer still holds a reference to the
    // buffer.
    //
    // If presentWhen is non-zero, it indicates the time when the buffer will be displayed on
    // screen. If the buffer's timestamp is farther in the future, the buffer won't be acquired, and
    // PRESENT_LATER will be returned. The presentation time is in nanoseconds, and the time base
    // is CLOCK_MONOTONIC.
    //
    // If maxFrameNumber is non-zero, it indicates that acquireBuffer should only return a buffer
    // with a frame number less than or equal to maxFrameNumber. If no such frame is available
    // (such as when a buffer has been replaced but the consumer has not received the
    // onFrameReplaced callback), then PRESENT_LATER will be returned.
    //
    // Return of NO_ERROR means the operation completed as normal.
    //
    // Return of a positive value means the operation could not be completed at this time, but the
    // user should try again later:
    // * NO_BUFFER_AVAILABLE - no buffer is pending (nothing queued by producer)
    // * PRESENT_LATER - the buffer's timestamp is farther in the future
    //
    // Return of a negative value means an error has occurred:
    // * INVALID_OPERATION - too many buffers have been acquired
    virtual status_t acquireBuffer(BufferItem* buffer, nsecs_t presentWhen,
                                   uint64_t maxFrameNumber = 0) = 0;

    // detachBuffer attempts to remove all ownership of the buffer in the given slot from the buffer
    // queue. If this call succeeds, the slot will be freed, and there will be no way to obtain the
    // buffer from this interface. The freed slot will remain unallocated until either it is
    // selected to hold a freshly allocated buffer in dequeueBuffer or a buffer is attached to the
    // slot. The buffer must have already been acquired.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * BAD_VALUE - the given slot number is invalid, either because it is out of the range
    //               [0, NUM_BUFFER_SLOTS) or because the slot it refers to is not
    //               currently acquired.
    virtual status_t detachBuffer(int slot) = 0;

    // attachBuffer attempts to transfer ownership of a buffer to the BufferQueue. If this call
    // succeeds, it will be as if this buffer was acquired from the returned slot number. As such,
    // this call will fail if attaching this buffer would cause too many buffers to be
    // simultaneously acquired.
    //
    // If the buffer is successfully attached, its frameNumber is initialized to 0. This must be
    // passed into the releaseBuffer call or else the buffer will be deallocated as stale.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * BAD_VALUE - outSlot or buffer were NULL, or the generation number of the buffer did not
    //               match the BufferQueue.
    // * INVALID_OPERATION - cannot attach the buffer because it would cause too many buffers
    //                       to be acquired.
    // * NO_MEMORY - no free slots available
    virtual status_t attachBuffer(int* outSlot, const sp<GraphicBuffer>& buffer) = 0;

    // releaseBuffer releases a buffer slot from the consumer back to the BufferQueue. This may be
    // done while the buffer's contents are still being accessed. The fence will signal when the
    // buffer is no longer in use. frameNumber is used to identify the exact buffer returned.
    //
    // If releaseBuffer returns STALE_BUFFER_SLOT, then the consumer must free any references to the
    // just-released buffer that it might have, as if it had received a onBuffersReleased() call
    // with a mask set for the released buffer.
    //
    // Note that the dependencies on EGL will be removed once we switch to using the Android HW
    // Sync HAL.
    //
    // Return of NO_ERROR means the operation completed as normal.
    //
    // Return of a positive value means the operation could not be completed at this time, but the
    // user should try again later:
    // * STALE_BUFFER_SLOT - see above (second paragraph)
    //
    // Return of a negative value means an error has occurred:
    // * BAD_VALUE - one of the following could've happened:
    //               * the buffer slot was invalid
    //               * the fence was NULL
    //               * the buffer slot specified is not in the acquired state
    virtual status_t releaseBuffer(int buf, uint64_t frameNumber, EGLDisplay display,
                                   EGLSyncKHR fence, const sp<Fence>& releaseFence) = 0;

    status_t releaseHelper(int buf, uint64_t frameNumber, const sp<Fence>& releaseFence) {
        return releaseBuffer(buf, frameNumber, EGL_NO_DISPLAY, EGL_NO_SYNC_KHR, releaseFence);
    }
    // This is explicitly *not* the actual signature of IGBC::releaseBuffer, but:
    //     1) We have no easy way to send the EGL objects across Binder
    //     2) This has always been broken, probably because
    //     3) IGBC is rarely remoted
    // For now, we will choose to bury our heads in the sand and ignore this problem until such time
    // as we can finally finish converting away from EGL sync to native Android sync
    using ReleaseBuffer = decltype(&IGraphicBufferConsumer::releaseHelper);

    // consumerConnect connects a consumer to the BufferQueue. Only one consumer may be connected,
    // and when that consumer disconnects the BufferQueue is placed into the "abandoned" state,
    // causing most interactions with the BufferQueue by the producer to fail. controlledByApp
    // indicates whether the consumer is controlled by the application.
    //
    // consumer may not be NULL.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the BufferQueue has been abandoned
    // * BAD_VALUE - a NULL consumer was provided
    virtual status_t consumerConnect(const sp<IConsumerListener>& consumer,
                                     bool controlledByApp) = 0;

    // consumerDisconnect disconnects a consumer from the BufferQueue. All buffers will be freed and
    // the BufferQueue is placed in the "abandoned" state, causing most interactions with the
    // BufferQueue by the producer to fail.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * BAD_VALUE - no consumer is currently connected
    virtual status_t consumerDisconnect() = 0;

    // getReleasedBuffers sets the value pointed to by slotMask to a bit set. Each bit index with a
    // 1 corresponds to a released buffer slot with that index value. In particular, a released
    // buffer is one that has been released by the BufferQueue but has not yet been released by
    // the consumer.
    //
    // This should be called from the onBuffersReleased() callback.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the BufferQueue has been abandoned.
    virtual status_t getReleasedBuffers(uint64_t* slotMask) = 0;

    // setDefaultBufferSize is used to set the size of buffers returned by dequeueBuffer when a
    // width and height of zero is requested. Default is 1x1.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * BAD_VALUE - either w or h was zero
    virtual status_t setDefaultBufferSize(uint32_t w, uint32_t h) = 0;

    // setMaxBufferCount sets the maximum value for the number of buffers used in the BufferQueue
    // (the initial default is NUM_BUFFER_SLOTS). If a call to setMaxAcquiredBufferCount (by the
    // consumer), or a call to setAsyncMode or setMaxDequeuedBufferCount (by the producer), would
    // cause this value to be exceeded then that call will fail. This call will fail if a producer
    // is connected to the BufferQueue.
    //
    // The count must be between 1 and NUM_BUFFER_SLOTS, inclusive. The count cannot be less than
    // maxAcquiredBufferCount.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * BAD_VALUE - one of the below conditions occurred:
    //               * bufferCount was out of range (see above).
    //               * failure to adjust the number of available slots.
    // * INVALID_OPERATION - attempting to call this after a producer connected.
    virtual status_t setMaxBufferCount(int bufferCount) = 0;

    // setMaxAcquiredBufferCount sets the maximum number of buffers that can be acquired by the
    // consumer at one time (default 1). If this method succeeds, any new buffer slots will be both
    // unallocated and owned by the BufferQueue object (i.e. they are not owned by the producer or
    // consumer). Calling this may also cause some buffer slots to be emptied.
    //
    // This function should not be called with a value of maxAcquiredBuffers that is less than the
    // number of currently acquired buffer slots. Doing so will result in a BAD_VALUE error.
    //
    // maxAcquiredBuffers must be (inclusive) between 1 and MAX_MAX_ACQUIRED_BUFFERS. It also cannot
    // cause the maxBufferCount value to be exceeded.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the BufferQueue has been abandoned
    // * BAD_VALUE - one of the below conditions occurred:
    //               * maxAcquiredBuffers was out of range (see above).
    //               * failure to adjust the number of available slots.
    //               * client would have more than the requested number of acquired buffers after
    //                 this call
    // * INVALID_OPERATION - attempting to call this after a producer connected.
    virtual status_t setMaxAcquiredBufferCount(int maxAcquiredBuffers) = 0;

    // setConsumerName sets the name used in logging
    virtual status_t setConsumerName(const String8& name) = 0;

    // setDefaultBufferFormat allows the BufferQueue to create GraphicBuffers of a defaultFormat if
    // no format is specified in dequeueBuffer. The initial default is PIXEL_FORMAT_RGBA_8888.
    //
    // Return of a value other than NO_ERROR means an unknown error has occurred.
    virtual status_t setDefaultBufferFormat(PixelFormat defaultFormat) = 0;

    // setDefaultBufferDataSpace is a request to the producer to provide buffers of the indicated
    // dataSpace. The producer may ignore this request. The initial default is
    // HAL_DATASPACE_UNKNOWN.
    //
    // Return of a value other than NO_ERROR means an unknown error has occurred.
    virtual status_t setDefaultBufferDataSpace(android_dataspace defaultDataSpace) = 0;

    // setConsumerUsageBits will turn on additional usage bits for dequeueBuffer. These are merged
    // with the bits passed to dequeueBuffer. The values are enumerated in gralloc.h,
    // e.g. GRALLOC_USAGE_HW_RENDER; the default is 0.
    //
    // Return of a value other than NO_ERROR means an unknown error has occurred.
    virtual status_t setConsumerUsageBits(uint64_t usage) = 0;

    // setConsumerIsProtected will turn on an internal bit that indicates whether
    // the consumer can handle protected gralloc buffers (i.e. with
    // GRALLOC_USAGE_PROTECTED set). IGraphicBufferProducer can query this
    // capability using NATIVE_WINDOW_CONSUMER_IS_PROTECTED.
    virtual status_t setConsumerIsProtected(bool isProtected) = 0;

    // setTransformHint bakes in rotation to buffers so overlays can be used. The values are
    // enumerated in window.h, e.g. NATIVE_WINDOW_TRANSFORM_ROT_90. The default is 0
    // (no transform).
    //
    // Return of a value other than NO_ERROR means an unknown error has occurred.
    virtual status_t setTransformHint(uint32_t hint) = 0;

    // Retrieve the sideband buffer stream, if any.
    virtual status_t getSidebandStream(sp<NativeHandle>* outStream) const = 0;

    // Retrieves any stored segments of the occupancy history of this BufferQueue and clears them.
    // Optionally closes out the pending segment if forceFlush is true.
    virtual status_t getOccupancyHistory(bool forceFlush,
                                         std::vector<OccupancyTracker::Segment>* outHistory) = 0;

    // discardFreeBuffers releases all currently-free buffers held by the BufferQueue, in order to
    // reduce the memory consumption of the BufferQueue to the minimum possible without
    // discarding data.
    // The consumer invoking this method is responsible for calling getReleasedBuffers() after this
    // call to free up any of its locally cached buffers.
    virtual status_t discardFreeBuffers() = 0;

    // dump state into a string
    virtual status_t dumpState(const String8& prefix, String8* outResult) const = 0;

    // Provide backwards source compatibility
    void dumpState(String8& result, const char* prefix) {
        String8 returned;
        dumpState(String8(prefix), &returned);
        result.append(returned);
    }
};

class BnGraphicBufferConsumer : public SafeBnInterface<IGraphicBufferConsumer> {
public:
    BnGraphicBufferConsumer()
          : SafeBnInterface<IGraphicBufferConsumer>("BnGraphicBufferConsumer") {}

    status_t onTransact(uint32_t code, const Parcel& data, Parcel* reply,
                        uint32_t flags = 0) override;
};

} // namespace android
