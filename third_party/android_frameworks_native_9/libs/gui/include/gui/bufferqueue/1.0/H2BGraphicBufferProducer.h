/*
 * Copyright 2016, The Android Open Source Project
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

#ifndef ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_H2BGRAPHICBUFFERPRODUCER_H
#define ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_H2BGRAPHICBUFFERPRODUCER_H

#include <hidl/MQDescriptor.h>
#include <hidl/Status.h>

#include <binder/Binder.h>
#include <gui/IGraphicBufferProducer.h>
#include <gui/IProducerListener.h>

#include <hidl/HybridInterface.h>
#include <android/hardware/graphics/bufferqueue/1.0/IGraphicBufferProducer.h>

namespace android {
namespace hardware {
namespace graphics {
namespace bufferqueue {
namespace V1_0 {
namespace utils {

using ::android::hidl::base::V1_0::IBase;
using ::android::hardware::hidl_array;
using ::android::hardware::hidl_memory;
using ::android::hardware::hidl_string;
using ::android::hardware::hidl_vec;
using ::android::hardware::Return;
using ::android::hardware::Void;
using ::android::sp;

using ::android::hardware::graphics::common::V1_0::PixelFormat;
using ::android::hardware::media::V1_0::AnwBuffer;

typedef ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer
        HGraphicBufferProducer;
typedef ::android::hardware::graphics::bufferqueue::V1_0::IProducerListener
        HProducerListener;

typedef ::android::IGraphicBufferProducer BGraphicBufferProducer;
using ::android::BnGraphicBufferProducer;
using ::android::IProducerListener;

struct H2BGraphicBufferProducer : public ::android::H2BConverter<
        HGraphicBufferProducer,
        BGraphicBufferProducer,
        BnGraphicBufferProducer> {
    H2BGraphicBufferProducer(sp<HGraphicBufferProducer> const& base) : CBase(base) {}

    status_t requestBuffer(int slot, sp<GraphicBuffer>* buf) override;
    status_t setMaxDequeuedBufferCount(int maxDequeuedBuffers) override;
    status_t setAsyncMode(bool async) override;
    status_t dequeueBuffer(int* slot, sp<Fence>* fence, uint32_t w, uint32_t h,
                           ::android::PixelFormat format, uint64_t usage, uint64_t* outBufferAge,
                           FrameEventHistoryDelta* outTimestamps) override;
    status_t detachBuffer(int slot) override;
    status_t detachNextBuffer(sp<GraphicBuffer>* outBuffer, sp<Fence>* outFence)
            override;
    status_t attachBuffer(int* outSlot, const sp<GraphicBuffer>& buffer)
            override;
    status_t queueBuffer(int slot,
            const QueueBufferInput& input,
            QueueBufferOutput* output) override;
    status_t cancelBuffer(int slot, const sp<Fence>& fence) override;
    int query(int what, int* value) override;
    status_t connect(const sp<IProducerListener>& listener, int api,
            bool producerControlledByApp, QueueBufferOutput* output) override;
    status_t disconnect(int api, DisconnectMode mode = DisconnectMode::Api)
            override;
    status_t setSidebandStream(const sp<NativeHandle>& stream) override;
    void allocateBuffers(uint32_t width, uint32_t height,
            ::android::PixelFormat format, uint64_t usage) override;
    status_t allowAllocation(bool allow) override;
    status_t setGenerationNumber(uint32_t generationNumber) override;
    String8 getConsumerName() const override;
    status_t setSharedBufferMode(bool sharedBufferMode) override;
    status_t setAutoRefresh(bool autoRefresh) override;
    status_t setDequeueTimeout(nsecs_t timeout) override;
    status_t getLastQueuedBuffer(sp<GraphicBuffer>* outBuffer,
          sp<Fence>* outFence, float outTransformMatrix[16]) override;
    void getFrameTimestamps(FrameEventHistoryDelta* outDelta) override;
    status_t getUniqueId(uint64_t* outId) const override;
    status_t getConsumerUsage(uint64_t* outUsage) const override;
};

}  // namespace utils
}  // namespace V1_0
}  // namespace bufferqueue
}  // namespace graphics
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_H2BGRAPHICBUFFERPRODUCER_H
