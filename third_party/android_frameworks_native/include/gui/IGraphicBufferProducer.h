/*
 * Copyright (C) 2010 The Android Open Source Project
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

#ifndef ANDROID_GUI_IGRAPHICBUFFERPRODUCER_H
#define ANDROID_GUI_IGRAPHICBUFFERPRODUCER_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Errors.h>
#include <utils/RefBase.h>

#include <binder/IInterface.h>

#include <ui/Fence.h>
#include <ui/GraphicBuffer.h>
#include <ui/Rect.h>
#include <ui/Region.h>

namespace android {
// ----------------------------------------------------------------------------

class IProducerListener;
class NativeHandle;
class Surface;

/*
 * This class defines the Binder IPC interface for the producer side of
 * a queue of graphics buffers.  It's used to send graphics data from one
 * component to another.  For example, a class that decodes video for
 * playback might use this to provide frames.  This is typically done
 * indirectly, through Surface.
 *
 * The underlying mechanism is a BufferQueue, which implements
 * BnGraphicBufferProducer.  In normal operation, the producer calls
 * dequeueBuffer() to get an empty buffer, fills it with data, then
 * calls queueBuffer() to make it available to the consumer.
 *
 * This class was previously called ISurfaceTexture.
 */
class IGraphicBufferProducer : public IInterface
{
public:
    DECLARE_META_INTERFACE(GraphicBufferProducer);

    enum {
        // A flag returned by dequeueBuffer when the client needs to call
        // requestBuffer immediately thereafter.
        BUFFER_NEEDS_REALLOCATION = 0x1,
        // A flag returned by dequeueBuffer when all mirrored slots should be
        // released by the client. This flag should always be processed first.
        RELEASE_ALL_BUFFERS       = 0x2,
    };

    // requestBuffer requests a new buffer for the given index. The server (i.e.
    // the IGraphicBufferProducer implementation) assigns the newly created
    // buffer to the given slot index, and the client is expected to mirror the
    // slot->buffer mapping so that it's not necessary to transfer a
    // GraphicBuffer for every dequeue operation.
    //
    // The slot must be in the range of [0, NUM_BUFFER_SLOTS).
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the buffer queue has been abandoned.
    // * BAD_VALUE - one of the two conditions occurred:
    //              * slot was out of range (see above)
    //              * buffer specified by the slot is not dequeued
    virtual status_t requestBuffer(int slot, sp<GraphicBuffer>* buf) = 0;

    // setBufferCount sets the number of buffer slots available. Calling this
    // will also cause all buffer slots to be emptied. The caller should empty
    // its mirrored copy of the buffer slots when calling this method.
    //
    // This function should not be called when there are any dequeued buffer
    // slots, doing so will result in a BAD_VALUE error returned.
    //
    // The buffer count should be at most NUM_BUFFER_SLOTS (inclusive), but at least
    // the minimum undequeued buffer count (exclusive). The minimum value
    // can be obtained by calling query(NATIVE_WINDOW_MIN_UNDEQUEUED_BUFFERS).
    // In particular the range is (minUndequeudBuffers, NUM_BUFFER_SLOTS].
    //
    // The buffer count may also be set to 0 (the default), to indicate that
    // the producer does not wish to set a value.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the buffer queue has been abandoned.
    // * BAD_VALUE - one of the below conditions occurred:
    //              * bufferCount was out of range (see above)
    //              * client has one or more buffers dequeued
    virtual status_t setBufferCount(int bufferCount) = 0;

    // dequeueBuffer requests a new buffer slot for the client to use. Ownership
    // of the slot is transfered to the client, meaning that the server will not
    // use the contents of the buffer associated with that slot.
    //
    // The slot index returned may or may not contain a buffer (client-side).
    // If the slot is empty the client should call requestBuffer to assign a new
    // buffer to that slot.
    //
    // Once the client is done filling this buffer, it is expected to transfer
    // buffer ownership back to the server with either cancelBuffer on
    // the dequeued slot or to fill in the contents of its associated buffer
    // contents and call queueBuffer.
    //
    // If dequeueBuffer returns the BUFFER_NEEDS_REALLOCATION flag, the client is
    // expected to call requestBuffer immediately.
    //
    // If dequeueBuffer returns the RELEASE_ALL_BUFFERS flag, the client is
    // expected to release all of the mirrored slot->buffer mappings.
    //
    // The fence parameter will be updated to hold the fence associated with
    // the buffer. The contents of the buffer must not be overwritten until the
    // fence signals. If the fence is Fence::NO_FENCE, the buffer may be written
    // immediately.
    //
    // The async parameter sets whether we're in asynchronous mode for this
    // dequeueBuffer() call.
    //
    // The width and height parameters must be no greater than the minimum of
    // GL_MAX_VIEWPORT_DIMS and GL_MAX_TEXTURE_SIZE (see: glGetIntegerv).
    // An error due to invalid dimensions might not be reported until
    // updateTexImage() is called.  If width and height are both zero, the
    // default values specified by setDefaultBufferSize() are used instead.
    //
    // If the format is 0, the default format will be used.
    //
    // The usage argument specifies gralloc buffer usage flags.  The values
    // are enumerated in <gralloc.h>, e.g. GRALLOC_USAGE_HW_RENDER.  These
    // will be merged with the usage flags specified by
    // IGraphicBufferConsumer::setConsumerUsageBits.
    //
    // This call will block until a buffer is available to be dequeued. If
    // both the producer and consumer are controlled by the app, then this call
    // can never block and will return WOULD_BLOCK if no buffer is available.
    //
    // A non-negative value with flags set (see above) will be returned upon
    // success.
    //
    // Return of a negative means an error has occurred:
    // * NO_INIT - the buffer queue has been abandoned.
    // * BAD_VALUE - both in async mode and buffer count was less than the
    //               max numbers of buffers that can be allocated at once.
    // * INVALID_OPERATION - cannot attach the buffer because it would cause
    //                       too many buffers to be dequeued, either because
    //                       the producer already has a single buffer dequeued
    //                       and did not set a buffer count, or because a
    //                       buffer count was set and this call would cause
    //                       it to be exceeded.
    // * WOULD_BLOCK - no buffer is currently available, and blocking is disabled
    //                 since both the producer/consumer are controlled by app
    // * NO_MEMORY - out of memory, cannot allocate the graphics buffer.
    //
    // All other negative values are an unknown error returned downstream
    // from the graphics allocator (typically errno).
    virtual status_t dequeueBuffer(int* slot, sp<Fence>* fence, bool async,
            uint32_t w, uint32_t h, PixelFormat format, uint32_t usage) = 0;

    // detachBuffer attempts to remove all ownership of the buffer in the given
    // slot from the buffer queue. If this call succeeds, the slot will be
    // freed, and there will be no way to obtain the buffer from this interface.
    // The freed slot will remain unallocated until either it is selected to
    // hold a freshly allocated buffer in dequeueBuffer or a buffer is attached
    // to the slot. The buffer must have already been dequeued, and the caller
    // must already possesses the sp<GraphicBuffer> (i.e., must have called
    // requestBuffer).
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the buffer queue has been abandoned.
    // * BAD_VALUE - the given slot number is invalid, either because it is
    //               out of the range [0, NUM_BUFFER_SLOTS), or because the slot
    //               it refers to is not currently dequeued and requested.
    virtual status_t detachBuffer(int slot) = 0;

    // detachNextBuffer is equivalent to calling dequeueBuffer, requestBuffer,
    // and detachBuffer in sequence, except for two things:
    //
    // 1) It is unnecessary to know the dimensions, format, or usage of the
    //    next buffer.
    // 2) It will not block, since if it cannot find an appropriate buffer to
    //    return, it will return an error instead.
    //
    // Only slots that are free but still contain a GraphicBuffer will be
    // considered, and the oldest of those will be returned. outBuffer is
    // equivalent to outBuffer from the requestBuffer call, and outFence is
    // equivalent to fence from the dequeueBuffer call.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the buffer queue has been abandoned.
    // * BAD_VALUE - either outBuffer or outFence were NULL.
    // * NO_MEMORY - no slots were found that were both free and contained a
    //               GraphicBuffer.
    virtual status_t detachNextBuffer(sp<GraphicBuffer>* outBuffer,
            sp<Fence>* outFence) = 0;

    // attachBuffer attempts to transfer ownership of a buffer to the buffer
    // queue. If this call succeeds, it will be as if this buffer was dequeued
    // from the returned slot number. As such, this call will fail if attaching
    // this buffer would cause too many buffers to be simultaneously dequeued.
    //
    // If attachBuffer returns the RELEASE_ALL_BUFFERS flag, the caller is
    // expected to release all of the mirrored slot->buffer mappings.
    //
    // A non-negative value with flags set (see above) will be returned upon
    // success.
    //
    // Return of a negative value means an error has occurred:
    // * NO_INIT - the buffer queue has been abandoned.
    // * BAD_VALUE - outSlot or buffer were NULL, invalid combination of
    //               async mode and buffer count override, or the generation
    //               number of the buffer did not match the buffer queue.
    // * INVALID_OPERATION - cannot attach the buffer because it would cause
    //                       too many buffers to be dequeued, either because
    //                       the producer already has a single buffer dequeued
    //                       and did not set a buffer count, or because a
    //                       buffer count was set and this call would cause
    //                       it to be exceeded.
    // * WOULD_BLOCK - no buffer slot is currently available, and blocking is
    //                 disabled since both the producer/consumer are
    //                 controlled by the app.
    virtual status_t attachBuffer(int* outSlot,
            const sp<GraphicBuffer>& buffer) = 0;

    // queueBuffer indicates that the client has finished filling in the
    // contents of the buffer associated with slot and transfers ownership of
    // that slot back to the server.
    //
    // It is not valid to call queueBuffer on a slot that is not owned
    // by the client or one for which a buffer associated via requestBuffer
    // (an attempt to do so will fail with a return value of BAD_VALUE).
    //
    // In addition, the input must be described by the client (as documented
    // below). Any other properties (zero point, etc)
    // are client-dependent, and should be documented by the client.
    //
    // The slot must be in the range of [0, NUM_BUFFER_SLOTS).
    //
    // Upon success, the output will be filled with meaningful values
    // (refer to the documentation below).
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the buffer queue has been abandoned.
    // * BAD_VALUE - one of the below conditions occurred:
    //              * fence was NULL
    //              * scaling mode was unknown
    //              * both in async mode and buffer count was less than the
    //                max numbers of buffers that can be allocated at once
    //              * slot index was out of range (see above).
    //              * the slot was not in the dequeued state
    //              * the slot was enqueued without requesting a buffer
    //              * crop rect is out of bounds of the buffer dimensions

    struct QueueBufferInput : public Flattenable<QueueBufferInput> {
        friend class Flattenable<QueueBufferInput>;
        inline QueueBufferInput(const Parcel& parcel);
        // timestamp - a monotonically increasing value in nanoseconds
        // isAutoTimestamp - if the timestamp was synthesized at queue time
        // dataSpace - description of the contents, interpretation depends on format
        // crop - a crop rectangle that's used as a hint to the consumer
        // scalingMode - a set of flags from NATIVE_WINDOW_SCALING_* in <window.h>
        // transform - a set of flags from NATIVE_WINDOW_TRANSFORM_* in <window.h>
        // async - if the buffer is queued in asynchronous mode
        // fence - a fence that the consumer must wait on before reading the buffer,
        //         set this to Fence::NO_FENCE if the buffer is ready immediately
        // sticky - the sticky transform set in Surface (only used by the LEGACY
        //          camera mode).
        inline QueueBufferInput(int64_t timestamp, bool isAutoTimestamp,
                android_dataspace dataSpace, const Rect& crop, int scalingMode,
                uint32_t transform, bool async, const sp<Fence>& fence,
                uint32_t sticky = 0)
                : timestamp(timestamp), isAutoTimestamp(isAutoTimestamp),
                  dataSpace(dataSpace), crop(crop), scalingMode(scalingMode),
                  transform(transform), stickyTransform(sticky),
                  async(async), fence(fence), surfaceDamage() { }
        inline void deflate(int64_t* outTimestamp, bool* outIsAutoTimestamp,
                android_dataspace* outDataSpace,
                Rect* outCrop, int* outScalingMode,
                uint32_t* outTransform, bool* outAsync, sp<Fence>* outFence,
                uint32_t* outStickyTransform = NULL) const {
            *outTimestamp = timestamp;
            *outIsAutoTimestamp = bool(isAutoTimestamp);
            *outDataSpace = dataSpace;
            *outCrop = crop;
            *outScalingMode = scalingMode;
            *outTransform = transform;
            *outAsync = bool(async);
            *outFence = fence;
            if (outStickyTransform != NULL) {
                *outStickyTransform = stickyTransform;
            }
        }

        // Flattenable protocol
        size_t getFlattenedSize() const;
        size_t getFdCount() const;
        status_t flatten(void*& buffer, size_t& size, int*& fds, size_t& count) const;
        status_t unflatten(void const*& buffer, size_t& size, int const*& fds, size_t& count);

        const Region& getSurfaceDamage() const { return surfaceDamage; }
        void setSurfaceDamage(const Region& damage) { surfaceDamage = damage; }

    private:
        int64_t timestamp;
        int isAutoTimestamp;
        android_dataspace dataSpace;
        Rect crop;
        int scalingMode;
        uint32_t transform;
        uint32_t stickyTransform;
        int async;
        sp<Fence> fence;
        Region surfaceDamage;
    };

    // QueueBufferOutput must be a POD structure
    struct __attribute__ ((__packed__)) QueueBufferOutput {
        inline QueueBufferOutput() { }
        // outWidth - filled with default width applied to the buffer
        // outHeight - filled with default height applied to the buffer
        // outTransformHint - filled with default transform applied to the buffer
        // outNumPendingBuffers - num buffers queued that haven't yet been acquired
        //                        (counting the currently queued buffer)
        inline void deflate(uint32_t* outWidth,
                uint32_t* outHeight,
                uint32_t* outTransformHint,
                uint32_t* outNumPendingBuffers) const {
            *outWidth = width;
            *outHeight = height;
            *outTransformHint = transformHint;
            *outNumPendingBuffers = numPendingBuffers;
        }
        inline void inflate(uint32_t inWidth, uint32_t inHeight,
                uint32_t inTransformHint, uint32_t inNumPendingBuffers) {
            width = inWidth;
            height = inHeight;
            transformHint = inTransformHint;
            numPendingBuffers = inNumPendingBuffers;
        }
    private:
        uint32_t width;
        uint32_t height;
        uint32_t transformHint;
        uint32_t numPendingBuffers;
    };

    virtual status_t queueBuffer(int slot,
            const QueueBufferInput& input, QueueBufferOutput* output) = 0;

    // cancelBuffer indicates that the client does not wish to fill in the
    // buffer associated with slot and transfers ownership of the slot back to
    // the server.
    //
    // The buffer is not queued for use by the consumer.
    //
    // The buffer will not be overwritten until the fence signals.  The fence
    // will usually be the one obtained from dequeueBuffer.
    virtual void cancelBuffer(int slot, const sp<Fence>& fence) = 0;

    // query retrieves some information for this surface
    // 'what' tokens allowed are that of NATIVE_WINDOW_* in <window.h>
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - the buffer queue has been abandoned.
    // * BAD_VALUE - what was out of range
    virtual int query(int what, int* value) = 0;

    // connect attempts to connect a client API to the IGraphicBufferProducer.
    // This must be called before any other IGraphicBufferProducer methods are
    // called except for getAllocator. A consumer must be already connected.
    //
    // This method will fail if the connect was previously called on the
    // IGraphicBufferProducer and no corresponding disconnect call was made.
    //
    // The listener is an optional binder callback object that can be used if
    // the producer wants to be notified when the consumer releases a buffer
    // back to the BufferQueue. It is also used to detect the death of the
    // producer. If only the latter functionality is desired, there is a
    // DummyProducerListener class in IProducerListener.h that can be used.
    //
    // The api should be one of the NATIVE_WINDOW_API_* values in <window.h>
    //
    // The producerControlledByApp should be set to true if the producer is hosted
    // by an untrusted process (typically app_process-forked processes). If both
    // the producer and the consumer are app-controlled then all buffer queues
    // will operate in async mode regardless of the async flag.
    //
    // Upon success, the output will be filled with meaningful data
    // (refer to QueueBufferOutput documentation above).
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * NO_INIT - one of the following occurred:
    //             * the buffer queue was abandoned
    //             * no consumer has yet connected
    // * BAD_VALUE - one of the following has occurred:
    //             * the producer is already connected
    //             * api was out of range (see above).
    //             * output was NULL.
    // * DEAD_OBJECT - the token is hosted by an already-dead process
    //
    // Additional negative errors may be returned by the internals, they
    // should be treated as opaque fatal unrecoverable errors.
    virtual status_t connect(const sp<IProducerListener>& listener,
            int api, bool producerControlledByApp, QueueBufferOutput* output) = 0;

    // disconnect attempts to disconnect a client API from the
    // IGraphicBufferProducer.  Calling this method will cause any subsequent
    // calls to other IGraphicBufferProducer methods to fail except for
    // getAllocator and connect.  Successfully calling connect after this will
    // allow the other methods to succeed again.
    //
    // This method will fail if the the IGraphicBufferProducer is not currently
    // connected to the specified client API.
    //
    // The api should be one of the NATIVE_WINDOW_API_* values in <window.h>
    //
    // Disconnecting from an abandoned IGraphicBufferProducer is legal and
    // is considered a no-op.
    //
    // Return of a value other than NO_ERROR means an error has occurred:
    // * BAD_VALUE - one of the following has occurred:
    //             * the api specified does not match the one that was connected
    //             * api was out of range (see above).
    // * DEAD_OBJECT - the token is hosted by an already-dead process
    virtual status_t disconnect(int api) = 0;

    // Attaches a sideband buffer stream to the IGraphicBufferProducer.
    //
    // A sideband stream is a device-specific mechanism for passing buffers
    // from the producer to the consumer without using dequeueBuffer/
    // queueBuffer. If a sideband stream is present, the consumer can choose
    // whether to acquire buffers from the sideband stream or from the queued
    // buffers.
    //
    // Passing NULL or a different stream handle will detach the previous
    // handle if any.
    virtual status_t setSidebandStream(const sp<NativeHandle>& stream) = 0;

    // Allocates buffers based on the given dimensions/format.
    //
    // This function will allocate up to the maximum number of buffers
    // permitted by the current BufferQueue configuration. It will use the
    // given format, dimensions, and usage bits, which are interpreted in the
    // same way as for dequeueBuffer, and the async flag must be set the same
    // way as for dequeueBuffer to ensure that the correct number of buffers are
    // allocated. This is most useful to avoid an allocation delay during
    // dequeueBuffer. If there are already the maximum number of buffers
    // allocated, this function has no effect.
    virtual void allocateBuffers(bool async, uint32_t width, uint32_t height,
            PixelFormat format, uint32_t usage) = 0;

    // Sets whether dequeueBuffer is allowed to allocate new buffers.
    //
    // Normally dequeueBuffer does not discriminate between free slots which
    // already have an allocated buffer and those which do not, and will
    // allocate a new buffer if the slot doesn't have a buffer or if the slot's
    // buffer doesn't match the requested size, format, or usage. This method
    // allows the producer to restrict the eligible slots to those which already
    // have an allocated buffer of the correct size, format, and usage. If no
    // eligible slot is available, dequeueBuffer will block or return an error
    // as usual.
    virtual status_t allowAllocation(bool allow) = 0;

    // Sets the current generation number of the BufferQueue.
    //
    // This generation number will be inserted into any buffers allocated by the
    // BufferQueue, and any attempts to attach a buffer with a different
    // generation number will fail. Buffers already in the queue are not
    // affected and will retain their current generation number. The generation
    // number defaults to 0.
    virtual status_t setGenerationNumber(uint32_t generationNumber) = 0;

    // Returns the name of the connected consumer.
    virtual String8 getConsumerName() const = 0;
};

// ----------------------------------------------------------------------------

class BnGraphicBufferProducer : public BnInterface<IGraphicBufferProducer>
{
public:
    virtual status_t    onTransact( uint32_t code,
                                    const Parcel& data,
                                    Parcel* reply,
                                    uint32_t flags = 0);
};

// ----------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_GUI_IGRAPHICBUFFERPRODUCER_H
