/*
 * Copyright 2018 The Android Open Source Project
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

#ifndef ANDROID_GUI_BUFFERHUBCONSUMER_H_
#define ANDROID_GUI_BUFFERHUBCONSUMER_H_

#include <gui/IGraphicBufferConsumer.h>
#include <private/dvr/buffer_hub_queue_client.h>
#include <private/dvr/buffer_hub_queue_parcelable.h>

namespace android {

class BufferHubConsumer : public IGraphicBufferConsumer {
public:
    // Creates a BufferHubConsumer instance by importing an existing producer queue.
    static sp<BufferHubConsumer> Create(const std::shared_ptr<dvr::ConsumerQueue>& queue);

    // Creates a BufferHubConsumer instance by importing an existing producer
    // parcelable. Note that this call takes the ownership of the parcelable
    // object and is guaranteed to succeed if parcelable object is valid.
    static sp<BufferHubConsumer> Create(dvr::ConsumerQueueParcelable parcelable);

    // See |IGraphicBufferConsumer::acquireBuffer|
    status_t acquireBuffer(BufferItem* buffer, nsecs_t presentWhen,
                           uint64_t maxFrameNumber = 0) override;

    // See |IGraphicBufferConsumer::detachBuffer|
    status_t detachBuffer(int slot) override;

    // See |IGraphicBufferConsumer::attachBuffer|
    status_t attachBuffer(int* outSlot, const sp<GraphicBuffer>& buffer) override;

    // See |IGraphicBufferConsumer::releaseBuffer|
    status_t releaseBuffer(int buf, uint64_t frameNumber, EGLDisplay display, EGLSyncKHR fence,
                           const sp<Fence>& releaseFence) override;

    // See |IGraphicBufferConsumer::consumerConnect|
    status_t consumerConnect(const sp<IConsumerListener>& consumer, bool controlledByApp) override;

    // See |IGraphicBufferConsumer::consumerDisconnect|
    status_t consumerDisconnect() override;

    // See |IGraphicBufferConsumer::getReleasedBuffers|
    status_t getReleasedBuffers(uint64_t* slotMask) override;

    // See |IGraphicBufferConsumer::setDefaultBufferSize|
    status_t setDefaultBufferSize(uint32_t w, uint32_t h) override;

    // See |IGraphicBufferConsumer::setMaxBufferCount|
    status_t setMaxBufferCount(int bufferCount) override;

    // See |IGraphicBufferConsumer::setMaxAcquiredBufferCount|
    status_t setMaxAcquiredBufferCount(int maxAcquiredBuffers) override;

    // See |IGraphicBufferConsumer::setConsumerName|
    status_t setConsumerName(const String8& name) override;

    // See |IGraphicBufferConsumer::setDefaultBufferFormat|
    status_t setDefaultBufferFormat(PixelFormat defaultFormat) override;

    // See |IGraphicBufferConsumer::setDefaultBufferDataSpace|
    status_t setDefaultBufferDataSpace(android_dataspace defaultDataSpace) override;

    // See |IGraphicBufferConsumer::setConsumerUsageBits|
    status_t setConsumerUsageBits(uint64_t usage) override;

    // See |IGraphicBufferConsumer::setConsumerIsProtected|
    status_t setConsumerIsProtected(bool isProtected) override;

    // See |IGraphicBufferConsumer::setTransformHint|
    status_t setTransformHint(uint32_t hint) override;

    // See |IGraphicBufferConsumer::getSidebandStream|
    status_t getSidebandStream(sp<NativeHandle>* outStream) const override;

    // See |IGraphicBufferConsumer::getOccupancyHistory|
    status_t getOccupancyHistory(bool forceFlush,
                                 std::vector<OccupancyTracker::Segment>* outHistory) override;

    // See |IGraphicBufferConsumer::discardFreeBuffers|
    status_t discardFreeBuffers() override;

    // See |IGraphicBufferConsumer::dumpState|
    status_t dumpState(const String8& prefix, String8* outResult) const override;

    // BufferHubConsumer provides its own logic to cast to a binder object.
    IBinder* onAsBinder() override;

private:
    // Private constructor to force use of |Create|.
    BufferHubConsumer() = default;

    // Concrete implementation backed by BufferHubBuffer.
    std::shared_ptr<dvr::ConsumerQueue> mQueue;
};

} // namespace android

#endif // ANDROID_GUI_BUFFERHUBCONSUMER_H_
