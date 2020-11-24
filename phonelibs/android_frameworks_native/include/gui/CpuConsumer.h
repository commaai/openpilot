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

#ifndef ANDROID_GUI_CPUCONSUMER_H
#define ANDROID_GUI_CPUCONSUMER_H

#include <gui/ConsumerBase.h>

#include <ui/GraphicBuffer.h>

#include <utils/String8.h>
#include <utils/Vector.h>
#include <utils/threads.h>


namespace android {

class BufferQueue;

/**
 * CpuConsumer is a BufferQueue consumer endpoint that allows direct CPU
 * access to the underlying gralloc buffers provided by BufferQueue. Multiple
 * buffers may be acquired by it at once, to be used concurrently by the
 * CpuConsumer owner. Sets gralloc usage flags to be software-read-only.
 * This queue is synchronous by default.
 */

class CpuConsumer : public ConsumerBase
{
  public:
    typedef ConsumerBase::FrameAvailableListener FrameAvailableListener;

    struct LockedBuffer {
        uint8_t    *data;
        uint32_t    width;
        uint32_t    height;
        PixelFormat format;
        uint32_t    stride;
        Rect        crop;
        uint32_t    transform;
        uint32_t    scalingMode;
        int64_t     timestamp;
        android_dataspace dataSpace;
        uint64_t    frameNumber;
        // this is the same as format, except for formats that are compatible with
        // a flexible format (e.g. HAL_PIXEL_FORMAT_YCbCr_420_888). In the latter
        // case this contains that flexible format
        PixelFormat flexFormat;
        // Values below are only valid when using HAL_PIXEL_FORMAT_YCbCr_420_888
        // or compatible format, in which case LockedBuffer::data
        // contains the Y channel, and stride is the Y channel stride. For other
        // formats, these will all be 0.
        uint8_t    *dataCb;
        uint8_t    *dataCr;
        uint32_t    chromaStride;
        uint32_t    chromaStep;
    };

    // Create a new CPU consumer. The maxLockedBuffers parameter specifies
    // how many buffers can be locked for user access at the same time.
    CpuConsumer(const sp<IGraphicBufferConsumer>& bq,
            size_t maxLockedBuffers, bool controlledByApp = false);

    virtual ~CpuConsumer();

    // set the name of the CpuConsumer that will be used to identify it in
    // log messages.
    void setName(const String8& name);

    // Gets the next graphics buffer from the producer and locks it for CPU use,
    // filling out the passed-in locked buffer structure with the native pointer
    // and metadata. Returns BAD_VALUE if no new buffer is available, and
    // NOT_ENOUGH_DATA if the maximum number of buffers is already locked.
    //
    // Only a fixed number of buffers can be locked at a time, determined by the
    // construction-time maxLockedBuffers parameter. If INVALID_OPERATION is
    // returned by lockNextBuffer, then old buffers must be returned to the queue
    // by calling unlockBuffer before more buffers can be acquired.
    status_t lockNextBuffer(LockedBuffer *nativeBuffer);

    // Returns a locked buffer to the queue, allowing it to be reused. Since
    // only a fixed number of buffers may be locked at a time, old buffers must
    // be released by calling unlockBuffer to ensure new buffers can be acquired by
    // lockNextBuffer.
    status_t unlockBuffer(const LockedBuffer &nativeBuffer);

  private:
    // Maximum number of buffers that can be locked at a time
    size_t mMaxLockedBuffers;

    status_t releaseAcquiredBufferLocked(size_t lockedIdx);

    virtual void freeBufferLocked(int slotIndex);

    // Tracking for buffers acquired by the user
    struct AcquiredBuffer {
        // Need to track the original mSlot index and the buffer itself because
        // the mSlot entry may be freed/reused before the acquired buffer is
        // released.
        int mSlot;
        sp<GraphicBuffer> mGraphicBuffer;
        void *mBufferPointer;

        AcquiredBuffer() :
                mSlot(BufferQueue::INVALID_BUFFER_SLOT),
                mBufferPointer(NULL) {
        }
    };
    Vector<AcquiredBuffer> mAcquiredBuffers;

    // Count of currently locked buffers
    size_t mCurrentLockedBuffers;

};

} // namespace android

#endif // ANDROID_GUI_CPUCONSUMER_H
