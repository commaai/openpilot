/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef ANDROID_GUI_BUFFERITEMCONSUMER_H
#define ANDROID_GUI_BUFFERITEMCONSUMER_H

#include <gui/ConsumerBase.h>
#include <gui/BufferQueue.h>

#define ANDROID_GRAPHICS_BUFFERITEMCONSUMER_JNI_ID "mBufferItemConsumer"

namespace android {

class String8;

/**
 * BufferItemConsumer is a BufferQueue consumer endpoint that allows clients
 * access to the whole BufferItem entry from BufferQueue. Multiple buffers may
 * be acquired at once, to be used concurrently by the client. This consumer can
 * operate either in synchronous or asynchronous mode.
 */
class BufferItemConsumer: public ConsumerBase
{
  public:
    typedef ConsumerBase::FrameAvailableListener FrameAvailableListener;

    struct BufferFreedListener : public virtual RefBase {
        virtual void onBufferFreed(const wp<GraphicBuffer>& graphicBuffer) = 0;
    };

    enum { DEFAULT_MAX_BUFFERS = -1 };
    enum { INVALID_BUFFER_SLOT = BufferQueue::INVALID_BUFFER_SLOT };
    enum { NO_BUFFER_AVAILABLE = BufferQueue::NO_BUFFER_AVAILABLE };

    // Create a new buffer item consumer. The consumerUsage parameter determines
    // the consumer usage flags passed to the graphics allocator. The
    // bufferCount parameter specifies how many buffers can be locked for user
    // access at the same time.
    // controlledByApp tells whether this consumer is controlled by the
    // application.
    BufferItemConsumer(const sp<IGraphicBufferConsumer>& consumer,
            uint64_t consumerUsage, int bufferCount = DEFAULT_MAX_BUFFERS,
            bool controlledByApp = false);

    ~BufferItemConsumer() override;

    // setBufferFreedListener sets the listener object that will be notified
    // when an old buffer is being freed.
    void setBufferFreedListener(const wp<BufferFreedListener>& listener);

    // Gets the next graphics buffer from the producer, filling out the
    // passed-in BufferItem structure. Returns NO_BUFFER_AVAILABLE if the queue
    // of buffers is empty, and INVALID_OPERATION if the maximum number of
    // buffers is already acquired.
    //
    // Only a fixed number of buffers can be acquired at a time, determined by
    // the construction-time bufferCount parameter. If INVALID_OPERATION is
    // returned by acquireBuffer, then old buffers must be returned to the
    // queue by calling releaseBuffer before more buffers can be acquired.
    //
    // If waitForFence is true, and the acquired BufferItem has a valid fence object,
    // acquireBuffer will wait on the fence with no timeout before returning.
    status_t acquireBuffer(BufferItem* item, nsecs_t presentWhen,
            bool waitForFence = true);

    // Returns an acquired buffer to the queue, allowing it to be reused. Since
    // only a fixed number of buffers may be acquired at a time, old buffers
    // must be released by calling releaseBuffer to ensure new buffers can be
    // acquired by acquireBuffer. Once a BufferItem is released, the caller must
    // not access any members of the BufferItem, and should immediately remove
    // all of its references to the BufferItem itself.
    status_t releaseBuffer(const BufferItem &item,
            const sp<Fence>& releaseFence = Fence::NO_FENCE);

   private:
    void freeBufferLocked(int slotIndex) override;

    // mBufferFreedListener is the listener object that will be called when
    // an old buffer is being freed. If it is not NULL it will be called from
    // freeBufferLocked.
    wp<BufferFreedListener> mBufferFreedListener;
};

} // namespace android

#endif // ANDROID_GUI_CPUCONSUMER_H
