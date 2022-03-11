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

#ifndef ANDROID_GUI_BUFFERQUEUE_H
#define ANDROID_GUI_BUFFERQUEUE_H

#include <gui/BufferItem.h>
#include <gui/BufferQueueDefs.h>
#include <gui/IGraphicBufferConsumer.h>
#include <gui/IGraphicBufferProducer.h>
#include <gui/IConsumerListener.h>

namespace android {

class BufferQueue {
public:
    // BufferQueue will keep track of at most this value of buffers.
    // Attempts at runtime to increase the number of buffers past this will fail.
    enum { NUM_BUFFER_SLOTS = BufferQueueDefs::NUM_BUFFER_SLOTS };
    // Used as a placeholder slot# when the value isn't pointing to an existing buffer.
    enum { INVALID_BUFFER_SLOT = BufferItem::INVALID_BUFFER_SLOT };
    // Alias to <IGraphicBufferConsumer.h> -- please scope from there in future code!
    enum {
        NO_BUFFER_AVAILABLE = IGraphicBufferConsumer::NO_BUFFER_AVAILABLE,
        PRESENT_LATER = IGraphicBufferConsumer::PRESENT_LATER,
    };

    // When in async mode we reserve two slots in order to guarantee that the
    // producer and consumer can run asynchronously.
    enum { MAX_MAX_ACQUIRED_BUFFERS = NUM_BUFFER_SLOTS - 2 };

    // for backward source compatibility
    typedef ::android::ConsumerListener ConsumerListener;

    // ProxyConsumerListener is a ConsumerListener implementation that keeps a weak
    // reference to the actual consumer object.  It forwards all calls to that
    // consumer object so long as it exists.
    //
    // This class exists to avoid having a circular reference between the
    // BufferQueue object and the consumer object.  The reason this can't be a weak
    // reference in the BufferQueue class is because we're planning to expose the
    // consumer side of a BufferQueue as a binder interface, which doesn't support
    // weak references.
    class ProxyConsumerListener : public BnConsumerListener {
    public:
        explicit ProxyConsumerListener(const wp<ConsumerListener>& consumerListener);
        ~ProxyConsumerListener() override;
        void onDisconnect() override;
        void onFrameAvailable(const BufferItem& item) override;
        void onFrameReplaced(const BufferItem& item) override;
        void onBuffersReleased() override;
        void onSidebandStreamChanged() override;
        void addAndGetFrameTimestamps(
                const NewFrameEventsEntry* newTimestamps,
                FrameEventHistoryDelta* outDelta) override;
    private:
        // mConsumerListener is a weak reference to the IConsumerListener.  This is
        // the raison d'etre of ProxyConsumerListener.
        wp<ConsumerListener> mConsumerListener;
    };

    // BufferQueue manages a pool of gralloc memory slots to be used by
    // producers and consumers. allocator is used to allocate all the
    // needed gralloc buffers.
    static void createBufferQueue(sp<IGraphicBufferProducer>* outProducer,
            sp<IGraphicBufferConsumer>* outConsumer,
            bool consumerIsSurfaceFlinger = false);

#ifndef NO_BUFFERHUB
    // Creates an IGraphicBufferProducer and IGraphicBufferConsumer pair backed by BufferHub.
    static void createBufferHubQueue(sp<IGraphicBufferProducer>* outProducer,
                                     sp<IGraphicBufferConsumer>* outConsumer);
#endif

    BufferQueue() = delete; // Create through createBufferQueue
};

// ----------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_GUI_BUFFERQUEUE_H
