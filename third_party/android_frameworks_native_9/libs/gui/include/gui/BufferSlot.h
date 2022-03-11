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

#ifndef ANDROID_GUI_BUFFERSLOT_H
#define ANDROID_GUI_BUFFERSLOT_H

#include <ui/Fence.h>
#include <ui/GraphicBuffer.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <utils/StrongPointer.h>

namespace android {

class Fence;

// BufferState tracks the states in which a buffer slot can be.
struct BufferState {

    // All slots are initially FREE (not dequeued, queued, acquired, or shared).
    BufferState()
    : mDequeueCount(0),
      mQueueCount(0),
      mAcquireCount(0),
      mShared(false) {
    }

    uint32_t mDequeueCount;
    uint32_t mQueueCount;
    uint32_t mAcquireCount;
    bool mShared;

    // A buffer can be in one of five states, represented as below:
    //
    //         | mShared | mDequeueCount | mQueueCount | mAcquireCount |
    // --------|---------|---------------|-------------|---------------|
    // FREE    |  false  |       0       |      0      |       0       |
    // DEQUEUED|  false  |       1       |      0      |       0       |
    // QUEUED  |  false  |       0       |      1      |       0       |
    // ACQUIRED|  false  |       0       |      0      |       1       |
    // SHARED  |  true   |      any      |     any     |      any      |
    //
    // FREE indicates that the buffer is available to be dequeued by the
    // producer. The slot is "owned" by BufferQueue. It transitions to DEQUEUED
    // when dequeueBuffer is called.
    //
    // DEQUEUED indicates that the buffer has been dequeued by the producer, but
    // has not yet been queued or canceled. The producer may modify the
    // buffer's contents as soon as the associated release fence is signaled.
    // The slot is "owned" by the producer. It can transition to QUEUED (via
    // queueBuffer or attachBuffer) or back to FREE (via cancelBuffer or
    // detachBuffer).
    //
    // QUEUED indicates that the buffer has been filled by the producer and
    // queued for use by the consumer. The buffer contents may continue to be
    // modified for a finite time, so the contents must not be accessed until
    // the associated fence is signaled. The slot is "owned" by BufferQueue. It
    // can transition to ACQUIRED (via acquireBuffer) or to FREE (if another
    // buffer is queued in asynchronous mode).
    //
    // ACQUIRED indicates that the buffer has been acquired by the consumer. As
    // with QUEUED, the contents must not be accessed by the consumer until the
    // acquire fence is signaled. The slot is "owned" by the consumer. It
    // transitions to FREE when releaseBuffer (or detachBuffer) is called. A
    // detached buffer can also enter the ACQUIRED state via attachBuffer.
    //
    // SHARED indicates that this buffer is being used in shared buffer
    // mode. It can be in any combination of the other states at the same time,
    // except for FREE (since that excludes being in any other state). It can
    // also be dequeued, queued, or acquired multiple times.

    inline bool isFree() const {
        return !isAcquired() && !isDequeued() && !isQueued();
    }

    inline bool isDequeued() const {
        return mDequeueCount > 0;
    }

    inline bool isQueued() const {
        return mQueueCount > 0;
    }

    inline bool isAcquired() const {
        return mAcquireCount > 0;
    }

    inline bool isShared() const {
        return mShared;
    }

    inline void reset() {
        *this = BufferState();
    }

    const char* string() const;

    inline void dequeue() {
        mDequeueCount++;
    }

    inline void detachProducer() {
        if (mDequeueCount > 0) {
            mDequeueCount--;
        }
    }

    inline void attachProducer() {
        mDequeueCount++;
    }

    inline void queue() {
        if (mDequeueCount > 0) {
            mDequeueCount--;
        }
        mQueueCount++;
    }

    inline void cancel() {
        if (mDequeueCount > 0) {
            mDequeueCount--;
        }
    }

    inline void freeQueued() {
        if (mQueueCount > 0) {
            mQueueCount--;
        }
    }

    inline void acquire() {
        if (mQueueCount > 0) {
            mQueueCount--;
        }
        mAcquireCount++;
    }

    inline void acquireNotInQueue() {
        mAcquireCount++;
    }

    inline void release() {
        if (mAcquireCount > 0) {
            mAcquireCount--;
        }
    }

    inline void detachConsumer() {
        if (mAcquireCount > 0) {
            mAcquireCount--;
        }
    }

    inline void attachConsumer() {
        mAcquireCount++;
    }
};

struct BufferSlot {

    BufferSlot()
    : mGraphicBuffer(nullptr),
      mEglDisplay(EGL_NO_DISPLAY),
      mBufferState(),
      mRequestBufferCalled(false),
      mFrameNumber(0),
      mEglFence(EGL_NO_SYNC_KHR),
      mFence(Fence::NO_FENCE),
      mAcquireCalled(false),
      mNeedsReallocation(false) {
    }

    // mGraphicBuffer points to the buffer allocated for this slot or is NULL
    // if no buffer has been allocated.
    sp<GraphicBuffer> mGraphicBuffer;

    // mEglDisplay is the EGLDisplay used to create EGLSyncKHR objects.
    EGLDisplay mEglDisplay;

    // mBufferState is the current state of this buffer slot.
    BufferState mBufferState;

    // mRequestBufferCalled is used for validating that the producer did
    // call requestBuffer() when told to do so. Technically this is not
    // needed but useful for debugging and catching producer bugs.
    bool mRequestBufferCalled;

    // mFrameNumber is the number of the queued frame for this slot.  This
    // is used to dequeue buffers in LRU order (useful because buffers
    // may be released before their release fence is signaled).
    uint64_t mFrameNumber;

    // mEglFence is the EGL sync object that must signal before the buffer
    // associated with this buffer slot may be dequeued. It is initialized
    // to EGL_NO_SYNC_KHR when the buffer is created and may be set to a
    // new sync object in releaseBuffer.  (This is deprecated in favor of
    // mFence, below.)
    EGLSyncKHR mEglFence;

    // mFence is a fence which will signal when work initiated by the
    // previous owner of the buffer is finished. When the buffer is FREE,
    // the fence indicates when the consumer has finished reading
    // from the buffer, or when the producer has finished writing if it
    // called cancelBuffer after queueing some writes. When the buffer is
    // QUEUED, it indicates when the producer has finished filling the
    // buffer. When the buffer is DEQUEUED or ACQUIRED, the fence has been
    // passed to the consumer or producer along with ownership of the
    // buffer, and mFence is set to NO_FENCE.
    sp<Fence> mFence;

    // Indicates whether this buffer has been seen by a consumer yet
    bool mAcquireCalled;

    // Indicates whether the buffer was re-allocated without notifying the
    // producer. If so, it needs to set the BUFFER_NEEDS_REALLOCATION flag when
    // dequeued to prevent the producer from using a stale cached buffer.
    bool mNeedsReallocation;
};

} // namespace android

#endif
