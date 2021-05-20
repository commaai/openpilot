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

#ifndef ANDROID_GUI_STREAMSPLITTER_H
#define ANDROID_GUI_STREAMSPLITTER_H

#include <gui/IConsumerListener.h>
#include <gui/IProducerListener.h>

#include <utils/Condition.h>
#include <utils/KeyedVector.h>
#include <utils/Mutex.h>
#include <utils/StrongPointer.h>

namespace android {

class GraphicBuffer;
class IGraphicBufferConsumer;
class IGraphicBufferProducer;

// StreamSplitter is an autonomous class that manages one input BufferQueue
// and multiple output BufferQueues. By using the buffer attach and detach logic
// in BufferQueue, it is able to present the illusion of a single split
// BufferQueue, where each buffer queued to the input is available to be
// acquired by each of the outputs, and is able to be dequeued by the input
// again only once all of the outputs have released it.
class StreamSplitter : public BnConsumerListener {
public:
    // createSplitter creates a new splitter, outSplitter, using inputQueue as
    // the input BufferQueue. Output BufferQueues must be added using addOutput
    // before queueing any buffers to the input.
    //
    // A return value other than NO_ERROR means that an error has occurred and
    // outSplitter has not been modified. BAD_VALUE is returned if inputQueue or
    // outSplitter is NULL. See IGraphicBufferConsumer::consumerConnect for
    // explanations of other error codes.
    static status_t createSplitter(const sp<IGraphicBufferConsumer>& inputQueue,
            sp<StreamSplitter>* outSplitter);

    // addOutput adds an output BufferQueue to the splitter. The splitter
    // connects to outputQueue as a CPU producer, and any buffers queued
    // to the input will be queued to each output. It is assumed that all of the
    // outputs are added before any buffers are queued on the input. If any
    // output is abandoned by its consumer, the splitter will abandon its input
    // queue (see onAbandoned).
    //
    // A return value other than NO_ERROR means that an error has occurred and
    // outputQueue has not been added to the splitter. BAD_VALUE is returned if
    // outputQueue is NULL. See IGraphicBufferProducer::connect for explanations
    // of other error codes.
    status_t addOutput(const sp<IGraphicBufferProducer>& outputQueue);

    // setName sets the consumer name of the input queue
    void setName(const String8& name);

private:
    // From IConsumerListener
    //
    // During this callback, we store some tracking information, detach the
    // buffer from the input, and attach it to each of the outputs. This call
    // can block if there are too many outstanding buffers. If it blocks, it
    // will resume when onBufferReleasedByOutput releases a buffer back to the
    // input.
    virtual void onFrameAvailable(const BufferItem& item);

    // From IConsumerListener
    // We don't care about released buffers because we detach each buffer as
    // soon as we acquire it. See the comment for onBufferReleased below for
    // some clarifying notes about the name.
    virtual void onBuffersReleased() {}

    // From IConsumerListener
    // We don't care about sideband streams, since we won't be splitting them
    virtual void onSidebandStreamChanged() {}

    // This is the implementation of the onBufferReleased callback from
    // IProducerListener. It gets called from an OutputListener (see below), and
    // 'from' is which producer interface from which the callback was received.
    //
    // During this callback, we detach the buffer from the output queue that
    // generated the callback, update our state tracking to see if this is the
    // last output releasing the buffer, and if so, release it to the input.
    // If we release the buffer to the input, we allow a blocked
    // onFrameAvailable call to proceed.
    void onBufferReleasedByOutput(const sp<IGraphicBufferProducer>& from);

    // When this is called, the splitter disconnects from (i.e., abandons) its
    // input queue and signals any waiting onFrameAvailable calls to wake up.
    // It still processes callbacks from other outputs, but only detaches their
    // buffers so they can continue operating until they run out of buffers to
    // acquire. This must be called with mMutex locked.
    void onAbandonedLocked();

    // This is a thin wrapper class that lets us determine which BufferQueue
    // the IProducerListener::onBufferReleased callback is associated with. We
    // create one of these per output BufferQueue, and then pass the producer
    // into onBufferReleasedByOutput above.
    class OutputListener : public BnProducerListener,
                           public IBinder::DeathRecipient {
    public:
        OutputListener(const sp<StreamSplitter>& splitter,
                const sp<IGraphicBufferProducer>& output);
        virtual ~OutputListener();

        // From IProducerListener
        virtual void onBufferReleased();

        // From IBinder::DeathRecipient
        virtual void binderDied(const wp<IBinder>& who);

    private:
        sp<StreamSplitter> mSplitter;
        sp<IGraphicBufferProducer> mOutput;
    };

    class BufferTracker : public LightRefBase<BufferTracker> {
    public:
        BufferTracker(const sp<GraphicBuffer>& buffer);

        const sp<GraphicBuffer>& getBuffer() const { return mBuffer; }
        const sp<Fence>& getMergedFence() const { return mMergedFence; }

        void mergeFence(const sp<Fence>& with);

        // Returns the new value
        // Only called while mMutex is held
        size_t incrementReleaseCountLocked() { return ++mReleaseCount; }

    private:
        // Only destroy through LightRefBase
        friend LightRefBase<BufferTracker>;
        ~BufferTracker();

        // Disallow copying
        BufferTracker(const BufferTracker& other);
        BufferTracker& operator=(const BufferTracker& other);

        sp<GraphicBuffer> mBuffer; // One instance that holds this native handle
        sp<Fence> mMergedFence;
        size_t mReleaseCount;
    };

    // Only called from createSplitter
    StreamSplitter(const sp<IGraphicBufferConsumer>& inputQueue);

    // Must be accessed through RefBase
    virtual ~StreamSplitter();

    static const int MAX_OUTSTANDING_BUFFERS = 2;

    // mIsAbandoned is set to true when an output dies. Once the StreamSplitter
    // has been abandoned, it will continue to detach buffers from other
    // outputs, but it will disconnect from the input and not attempt to
    // communicate with it further.
    bool mIsAbandoned;

    Mutex mMutex;
    Condition mReleaseCondition;
    int mOutstandingBuffers;
    sp<IGraphicBufferConsumer> mInput;
    Vector<sp<IGraphicBufferProducer> > mOutputs;

    // Map of GraphicBuffer IDs (GraphicBuffer::getId()) to buffer tracking
    // objects (which are mostly for counting how many outputs have released the
    // buffer, but also contain merged release fences).
    KeyedVector<uint64_t, sp<BufferTracker> > mBuffers;
};

} // namespace android

#endif
