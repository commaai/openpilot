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

#ifndef ANDROID_GUI_BUFFERHUBPRODUCER_H_
#define ANDROID_GUI_BUFFERHUBPRODUCER_H_

#include <gui/BufferSlot.h>
#include <gui/IGraphicBufferProducer.h>
#include <private/dvr/buffer_hub_queue_client.h>
#include <private/dvr/buffer_hub_queue_parcelable.h>

namespace android {

class BufferHubProducer : public IGraphicBufferProducer {
public:
    static constexpr int kNoConnectedApi = -1;

    // TODO(b/36187402) The actual implementation of BufferHubQueue's consumer
    // side logic doesn't limit the number of buffer it can acquire
    // simultaneously. We need a way for consumer logic to configure and enforce
    // that.
    static constexpr int kDefaultUndequeuedBuffers = 1;

    // Creates a BufferHubProducer instance by importing an existing prodcuer
    // queue.
    static sp<BufferHubProducer> Create(const std::shared_ptr<dvr::ProducerQueue>& producer);

    // Creates a BufferHubProducer instance by importing an existing prodcuer
    // parcelable. Note that this call takes the ownership of the parcelable
    // object and is guaranteed to succeed if parcelable object is valid.
    static sp<BufferHubProducer> Create(dvr::ProducerQueueParcelable parcelable);

    // See |IGraphicBufferProducer::requestBuffer|
    status_t requestBuffer(int slot, sp<GraphicBuffer>* buf) override;

    // For the BufferHub based implementation. All buffers in the queue are
    // allowed to be dequeued from the consumer side. It call always returns
    // 0 for |NATIVE_WINDOW_MIN_UNDEQUEUED_BUFFERS| query. Thus setting
    // |max_dequeued_buffers| here can be considered the same as setting queue
    // capacity.
    //
    // See |IGraphicBufferProducer::setMaxDequeuedBufferCount| for more info
    status_t setMaxDequeuedBufferCount(int max_dequeued_buffers) override;

    // See |IGraphicBufferProducer::setAsyncMode|
    status_t setAsyncMode(bool async) override;

    // See |IGraphicBufferProducer::dequeueBuffer|
    status_t dequeueBuffer(int* out_slot, sp<Fence>* out_fence, uint32_t width, uint32_t height,
                           PixelFormat format, uint64_t usage, uint64_t* outBufferAge,
                           FrameEventHistoryDelta* outTimestamps) override;

    // See |IGraphicBufferProducer::detachBuffer|
    status_t detachBuffer(int slot) override;

    // See |IGraphicBufferProducer::detachNextBuffer|
    status_t detachNextBuffer(sp<GraphicBuffer>* out_buffer, sp<Fence>* out_fence) override;

    // See |IGraphicBufferProducer::attachBuffer|
    status_t attachBuffer(int* out_slot, const sp<GraphicBuffer>& buffer) override;

    // See |IGraphicBufferProducer::queueBuffer|
    status_t queueBuffer(int slot, const QueueBufferInput& input,
                         QueueBufferOutput* output) override;

    // See |IGraphicBufferProducer::cancelBuffer|
    status_t cancelBuffer(int slot, const sp<Fence>& fence) override;

    // See |IGraphicBufferProducer::query|
    status_t query(int what, int* out_value) override;

    // See |IGraphicBufferProducer::connect|
    status_t connect(const sp<IProducerListener>& listener, int api,
                     bool producer_controlled_by_app, QueueBufferOutput* output) override;

    // See |IGraphicBufferProducer::disconnect|
    status_t disconnect(int api, DisconnectMode mode = DisconnectMode::Api) override;

    // See |IGraphicBufferProducer::setSidebandStream|
    status_t setSidebandStream(const sp<NativeHandle>& stream) override;

    // See |IGraphicBufferProducer::allocateBuffers|
    void allocateBuffers(uint32_t width, uint32_t height, PixelFormat format,
                         uint64_t usage) override;

    // See |IGraphicBufferProducer::allowAllocation|
    status_t allowAllocation(bool allow) override;

    // See |IGraphicBufferProducer::setGenerationNumber|
    status_t setGenerationNumber(uint32_t generation_number) override;

    // See |IGraphicBufferProducer::getConsumerName|
    String8 getConsumerName() const override;

    // See |IGraphicBufferProducer::setSharedBufferMode|
    status_t setSharedBufferMode(bool shared_buffer_mode) override;

    // See |IGraphicBufferProducer::setAutoRefresh|
    status_t setAutoRefresh(bool auto_refresh) override;

    // See |IGraphicBufferProducer::setDequeueTimeout|
    status_t setDequeueTimeout(nsecs_t timeout) override;

    // See |IGraphicBufferProducer::getLastQueuedBuffer|
    status_t getLastQueuedBuffer(sp<GraphicBuffer>* out_buffer, sp<Fence>* out_fence,
                                 float out_transform_matrix[16]) override;

    // See |IGraphicBufferProducer::getFrameTimestamps|
    void getFrameTimestamps(FrameEventHistoryDelta* /*outDelta*/) override;

    // See |IGraphicBufferProducer::getUniqueId|
    status_t getUniqueId(uint64_t* out_id) const override;

    // See |IGraphicBufferProducer::getConsumerUsage|
    status_t getConsumerUsage(uint64_t* out_usage) const override;

    // Takes out the current producer as a binder parcelable object. Note that the
    // producer must be disconnected to be exportable. After successful export,
    // the producer queue can no longer be connected again. Returns NO_ERROR when
    // takeout is successful and out_parcelable will hold the new parcelable
    // object. Also note that out_parcelable cannot be NULL and must points to an
    // invalid parcelable.
    status_t TakeAsParcelable(dvr::ProducerQueueParcelable* out_parcelable);

    IBinder* onAsBinder() override;

protected:
    // See |IGraphicBufferProducer::exportToParcel|
    status_t exportToParcel(Parcel* parcel) override;

private:
    using LocalHandle = pdx::LocalHandle;

    // Private constructor to force use of |Create|.
    BufferHubProducer() {}

    static uint64_t genUniqueId() {
        static std::atomic<uint32_t> counter{0};
        static uint64_t id = static_cast<uint64_t>(getpid()) << 32;
        return id | counter++;
    }

    // Allocate new buffer through BufferHub and add it into |queue_| for
    // bookkeeping.
    status_t AllocateBuffer(uint32_t width, uint32_t height, uint32_t layer_count,
                            PixelFormat format, uint64_t usage);

    // Remove a buffer via BufferHubRPC.
    status_t RemoveBuffer(size_t slot);

    // Free all buffers which are owned by the prodcuer. Note that if graphic
    // buffers are acquired by the consumer, we can't .
    status_t FreeAllBuffers();

    // Concreate implementation backed by BufferHubBuffer.
    std::shared_ptr<dvr::ProducerQueue> queue_;

    // Mutex for thread safety.
    std::mutex mutex_;

    // Connect client API, should be one of the NATIVE_WINDOW_API_* flags.
    int connected_api_{kNoConnectedApi};

    // |max_buffer_count_| sets the capacity of the underlying buffer queue.
    int32_t max_buffer_count_{dvr::BufferHubQueue::kMaxQueueCapacity};

    // |max_dequeued_buffer_count_| set the maximum number of buffers that can
    // be dequeued at the same momment.
    int32_t max_dequeued_buffer_count_{1};

    // Sets how long dequeueBuffer or attachBuffer will block if a buffer or
    // slot is not yet available. The timeout is stored in milliseconds.
    int dequeue_timeout_ms_{dvr::BufferHubQueue::kNoTimeOut};

    // |generation_number_| stores the current generation number of the attached
    // producer. Any attempt to attach a buffer with a different generation
    // number will fail.
    // TOOD(b/38137191) Currently not used as we don't support
    // IGraphicBufferProducer::detachBuffer.
    uint32_t generation_number_{0};

    // |buffers_| stores the buffers that have been dequeued from
    // |dvr::BufferHubQueue|, It is initialized to invalid buffers, and gets
    // filled in with the result of |Dequeue|.
    // TODO(jwcai) The buffer allocated to a slot will also be replaced if the
    // requested buffer usage or geometry differs from that of the buffer
    // allocated to a slot.
    struct BufferHubSlot : public BufferSlot {
        BufferHubSlot() : mBufferProducer(nullptr), mIsReallocating(false) {}
        // BufferSlot comes from android framework, using m prefix to comply with
        // the name convention with the reset of data fields from BufferSlot.
        std::shared_ptr<dvr::BufferProducer> mBufferProducer;
        bool mIsReallocating;
    };
    BufferHubSlot buffers_[dvr::BufferHubQueue::kMaxQueueCapacity];

    // A uniqueId used by IGraphicBufferProducer interface.
    const uint64_t unique_id_{genUniqueId()};

    // A pending parcelable object which keeps the bufferhub channel alive.
    dvr::ProducerQueueParcelable pending_producer_parcelable_;
};

} // namespace android

#endif // ANDROID_GUI_BUFFERHUBPRODUCER_H_
