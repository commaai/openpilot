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

struct BufferSlot {

    BufferSlot()
    : mEglDisplay(EGL_NO_DISPLAY),
      mBufferState(BufferSlot::FREE),
      mRequestBufferCalled(false),
      mFrameNumber(0),
      mEglFence(EGL_NO_SYNC_KHR),
      mAcquireCalled(false),
      mNeedsCleanupOnRelease(false),
      mAttachedByConsumer(false) {
    }

    // mGraphicBuffer points to the buffer allocated for this slot or is NULL
    // if no buffer has been allocated.
    sp<GraphicBuffer> mGraphicBuffer;

    // mEglDisplay is the EGLDisplay used to create EGLSyncKHR objects.
    EGLDisplay mEglDisplay;

    // BufferState represents the different states in which a buffer slot
    // can be.  All slots are initially FREE.
    enum BufferState {
        // FREE indicates that the buffer is available to be dequeued
        // by the producer.  The buffer may be in use by the consumer for
        // a finite time, so the buffer must not be modified until the
        // associated fence is signaled.
        //
        // The slot is "owned" by BufferQueue.  It transitions to DEQUEUED
        // when dequeueBuffer is called.
        FREE = 0,

        // DEQUEUED indicates that the buffer has been dequeued by the
        // producer, but has not yet been queued or canceled.  The
        // producer may modify the buffer's contents as soon as the
        // associated ready fence is signaled.
        //
        // The slot is "owned" by the producer.  It can transition to
        // QUEUED (via queueBuffer) or back to FREE (via cancelBuffer).
        DEQUEUED = 1,

        // QUEUED indicates that the buffer has been filled by the
        // producer and queued for use by the consumer.  The buffer
        // contents may continue to be modified for a finite time, so
        // the contents must not be accessed until the associated fence
        // is signaled.
        //
        // The slot is "owned" by BufferQueue.  It can transition to
        // ACQUIRED (via acquireBuffer) or to FREE (if another buffer is
        // queued in asynchronous mode).
        QUEUED = 2,

        // ACQUIRED indicates that the buffer has been acquired by the
        // consumer.  As with QUEUED, the contents must not be accessed
        // by the consumer until the fence is signaled.
        //
        // The slot is "owned" by the consumer.  It transitions to FREE
        // when releaseBuffer is called.
        ACQUIRED = 3
    };

    static const char* bufferStateName(BufferState state);

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

    // Indicates whether this buffer needs to be cleaned up by the
    // consumer.  This is set when a buffer in ACQUIRED state is freed.
    // It causes releaseBuffer to return STALE_BUFFER_SLOT.
    bool mNeedsCleanupOnRelease;

    // Indicates whether the buffer was attached on the consumer side.
    // If so, it needs to set the BUFFER_NEEDS_REALLOCATION flag when dequeued
    // to prevent the producer from using a stale cached buffer.
    bool mAttachedByConsumer;
};

} // namespace android

#endif
