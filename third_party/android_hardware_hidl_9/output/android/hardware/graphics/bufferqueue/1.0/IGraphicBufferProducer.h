#ifndef HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_IGRAPHICBUFFERPRODUCER_H
#define HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_IGRAPHICBUFFERPRODUCER_H

#include <android/hardware/graphics/bufferqueue/1.0/IProducerListener.h>
#include <android/hardware/graphics/common/1.0/types.h>
#include <android/hardware/media/1.0/types.h>
#include <android/hidl/base/1.0/IBase.h>

#include <android/hidl/manager/1.0/IServiceNotification.h>

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <hidl/Status.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hardware {
namespace graphics {
namespace bufferqueue {
namespace V1_0 {

struct IGraphicBufferProducer : public ::android::hidl::base::V1_0::IBase {
    typedef android::hardware::details::i_tag _hidl_tag;

    // Forward declaration for forward reference support:

    /**
     * Ref: frameworks/native/include/gui/IGraphicBufferProducer.h:
     *      IGraphicBufferProducer
     * This is a wrapper/wrapped HAL interface for the actual binder interface.
     */
    // Forward declaration for forward reference support:
    struct FenceTimeSnapshot;
    struct FrameEventsDelta;
    struct CompositorTiming;
    struct FrameEventHistoryDelta;
    enum class DisconnectMode : int32_t;
    struct QueueBufferInput;
    struct QueueBufferOutput;

    /**
     * Type for return values of functions in IGraphicBufferProducer.
     */
    typedef int32_t Status;

    /**
     * Ref: frameworks/native/include/ui/FenceTime.h: FenceTime::Snapshot
     * 
     * An atomic snapshot of the FenceTime that is flattenable.
     */
    struct FenceTimeSnapshot final {
        // Forward declaration for forward reference support:
        enum class State : int32_t;

        enum class State : int32_t {
            EMPTY = 0,
            FENCE = 1, // (::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State.EMPTY implicitly + 1)
            SIGNAL_TIME = 2, // (::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State.FENCE implicitly + 1)
        };

        ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State state __attribute__ ((aligned(4)));
        ::android::hardware::hidl_handle fence __attribute__ ((aligned(8)));
        int64_t signalTimeNs __attribute__ ((aligned(8)));
    };

    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot, state) == 0, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot, fence) == 8, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot, signalTimeNs) == 24, "wrong offset");
    static_assert(sizeof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot) == 32, "wrong size");
    static_assert(__alignof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot) == 8, "wrong alignment");

    /**
     * Ref: frameworks/native/include/gui/FrameTimestamp.h: FrameEventsDelta
     * 
     * A single frame update from the consumer to producer that can be sent
     * through a HIDL interface. Although this may be sent multiple times for
     * the same frame as new timestamps are set, Fences only need to be sent
     * once.
     */
    struct FrameEventsDelta final {
        uint32_t index __attribute__ ((aligned(4)));
        uint64_t frameNumber __attribute__ ((aligned(8)));
        bool addPostCompositeCalled __attribute__ ((aligned(1)));
        bool addRetireCalled __attribute__ ((aligned(1)));
        bool addReleaseCalled __attribute__ ((aligned(1)));
        int64_t postedTimeNs __attribute__ ((aligned(8)));
        int64_t requestedPresentTimeNs __attribute__ ((aligned(8)));
        int64_t latchTimeNs __attribute__ ((aligned(8)));
        int64_t firstRefreshStartTimeNs __attribute__ ((aligned(8)));
        int64_t lastRefreshStartTimeNs __attribute__ ((aligned(8)));
        int64_t dequeueReadyTime __attribute__ ((aligned(8)));
        ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot gpuCompositionDoneFence __attribute__ ((aligned(8)));
        ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot displayPresentFence __attribute__ ((aligned(8)));
        ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot displayRetireFence __attribute__ ((aligned(8)));
        ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot releaseFence __attribute__ ((aligned(8)));
    };

    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, index) == 0, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, frameNumber) == 8, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, addPostCompositeCalled) == 16, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, addRetireCalled) == 17, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, addReleaseCalled) == 18, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, postedTimeNs) == 24, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, requestedPresentTimeNs) == 32, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, latchTimeNs) == 40, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, firstRefreshStartTimeNs) == 48, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, lastRefreshStartTimeNs) == 56, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, dequeueReadyTime) == 64, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, gpuCompositionDoneFence) == 72, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, displayPresentFence) == 104, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, displayRetireFence) == 136, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta, releaseFence) == 168, "wrong offset");
    static_assert(sizeof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta) == 200, "wrong size");
    static_assert(__alignof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta) == 8, "wrong alignment");

    /**
     * Ref: frameworks/native/include/gui/FrameTimestamp.h: CompositorTiming
     * 
     * The most recent compositor timing info sent from consumer to producer
     * through a HIDL interface.
     */
    struct CompositorTiming final {
        int64_t deadlineNs __attribute__ ((aligned(8)));
        int64_t intervalNs __attribute__ ((aligned(8)));
        int64_t presentLatencyNs __attribute__ ((aligned(8)));
    };

    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming, deadlineNs) == 0, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming, intervalNs) == 8, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming, presentLatencyNs) == 16, "wrong offset");
    static_assert(sizeof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming) == 24, "wrong size");
    static_assert(__alignof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming) == 8, "wrong alignment");

    /**
     * Ref: frameworks/native/include/gui/FrameTimestamp.h: FrameEventHistoryDelta
     * 
     * A collection of updates from consumer to producer that can be sent
     * through a HIDL interface.
     */
    struct FrameEventHistoryDelta final {
        ::android::hardware::hidl_vec<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta> deltas __attribute__ ((aligned(8)));
        ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming compositorTiming __attribute__ ((aligned(8)));
    };

    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventHistoryDelta, deltas) == 0, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventHistoryDelta, compositorTiming) == 16, "wrong offset");
    static_assert(sizeof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventHistoryDelta) == 40, "wrong size");
    static_assert(__alignof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventHistoryDelta) == 8, "wrong alignment");

    /**
     * Modes for disconnection.
     */
    enum class DisconnectMode : int32_t {
        /**
         * Disconnect only the specified API.  */
        API = 0,
        /**
         * Disconnect any API originally connected from the process calling
         *  disconnect.  */
        ALL_LOCAL = 1, // (::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode.API implicitly + 1)
    };

    struct QueueBufferInput final {
        int64_t timestamp __attribute__ ((aligned(8)));
        int32_t isAutoTimestamp __attribute__ ((aligned(4)));
        ::android::hardware::graphics::common::V1_0::Dataspace dataSpace __attribute__ ((aligned(4)));
        ::android::hardware::media::V1_0::Rect crop __attribute__ ((aligned(4)));
        int32_t scalingMode __attribute__ ((aligned(4)));
        uint32_t transform __attribute__ ((aligned(4)));
        uint32_t stickyTransform __attribute__ ((aligned(4)));
        ::android::hardware::hidl_handle fence __attribute__ ((aligned(8)));
        ::android::hardware::hidl_vec<::android::hardware::media::V1_0::Rect> surfaceDamage __attribute__ ((aligned(8)));
        bool getFrameTimestamps __attribute__ ((aligned(1)));
    };

    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, timestamp) == 0, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, isAutoTimestamp) == 8, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, dataSpace) == 12, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, crop) == 16, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, scalingMode) == 32, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, transform) == 36, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, stickyTransform) == 40, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, fence) == 48, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, surfaceDamage) == 64, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput, getFrameTimestamps) == 80, "wrong offset");
    static_assert(sizeof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput) == 88, "wrong size");
    static_assert(__alignof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput) == 8, "wrong alignment");

    struct QueueBufferOutput final {
        uint32_t width __attribute__ ((aligned(4)));
        uint32_t height __attribute__ ((aligned(4)));
        uint32_t transformHint __attribute__ ((aligned(4)));
        uint32_t numPendingBuffers __attribute__ ((aligned(4)));
        uint64_t nextFrameNumber __attribute__ ((aligned(8)));
        bool bufferReplaced __attribute__ ((aligned(1)));
        ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventHistoryDelta frameTimestamps __attribute__ ((aligned(8)));
    };

    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput, width) == 0, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput, height) == 4, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput, transformHint) == 8, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput, numPendingBuffers) == 12, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput, nextFrameNumber) == 16, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput, bufferReplaced) == 24, "wrong offset");
    static_assert(offsetof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput, frameTimestamps) == 32, "wrong offset");
    static_assert(sizeof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput) == 72, "wrong size");
    static_assert(__alignof(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput) == 8, "wrong alignment");

    virtual bool isRemote() const override { return false; }


    using requestBuffer_cb = std::function<void(int32_t status, const ::android::hardware::media::V1_0::AnwBuffer& buffer)>;
    /**
     * requestBuffer requests a new buffer for the given index. The server (i.e.
     * the IProducerListener implementation) assigns the newly created
     * buffer to the given slot index, and the client is expected to mirror the
     * slot->buffer mapping so that it's not necessary to transfer an
     * AnwBuffer for every dequeue operation.
     * 
     * The slot must be in the range of [0, NUM_BUFFER_SLOTS).
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned or the producer is not
     *             connected.
     * * BAD_VALUE - one of the two conditions occurred:
     *              * slot was out of range (see above)
     *              * buffer specified by the slot is not dequeued
     */
    virtual ::android::hardware::Return<void> requestBuffer(int32_t slot, requestBuffer_cb _hidl_cb) = 0;

    /**
     * setMaxDequeuedBufferCount sets the maximum number of buffers that can be
     * dequeued by the producer at one time. If this method succeeds, any new
     * buffer slots will be both unallocated and owned by the BufferQueue object
     * (i.e. they are not owned by the producer or consumer). Calling this may
     * also cause some buffer slots to be emptied. If the caller is caching the
     * contents of the buffer slots, it should empty that cache after calling
     * this method.
     * 
     * This function should not be called with a value of maxDequeuedBuffers
     * that is less than the number of currently dequeued buffer slots. Doing so
     * will result in a BAD_VALUE error.
     * 
     * The buffer count should be at least 1 (inclusive), but at most
     * (NUM_BUFFER_SLOTS - the minimum undequeued buffer count) (exclusive). The
     * minimum undequeued buffer count can be obtained by calling
     * query(NATIVE_WINDOW_MIN_UNDEQUEUED_BUFFERS).
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned.
     * * BAD_VALUE - one of the below conditions occurred:
     *     * bufferCount was out of range (see above).
     *     * client would have more than the requested number of dequeued
     *       buffers after this call.
     *     * this call would cause the maxBufferCount value to be exceeded.
     *     * failure to adjust the number of available slots.
     */
    virtual ::android::hardware::Return<int32_t> setMaxDequeuedBufferCount(int32_t maxDequeuedBuffers) = 0;

    /**
     * Set the async flag if the producer intends to asynchronously queue
     * buffers without blocking. Typically this is used for triple-buffering
     * and/or when the swap interval is set to zero.
     * 
     * Enabling async mode will internally allocate an additional buffer to
     * allow for the asynchronous behavior. If it is not enabled queue/dequeue
     * calls may block.
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned.
     * * BAD_VALUE - one of the following has occurred:
     *             * this call would cause the maxBufferCount value to be
     *               exceeded
     *             * failure to adjust the number of available slots.
     */
    virtual ::android::hardware::Return<int32_t> setAsyncMode(bool async) = 0;

    using dequeueBuffer_cb = std::function<void(int32_t status, int32_t slot, const ::android::hardware::hidl_handle& fence, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventHistoryDelta& outTimestamps)>;
    /**
     * dequeueBuffer requests a new buffer slot for the client to use. Ownership
     * of the slot is transfered to the client, meaning that the server will not
     * use the contents of the buffer associated with that slot.
     * 
     * The slot index returned may or may not contain a buffer (client-side).
     * If the slot is empty the client should call requestBuffer to assign a new
     * buffer to that slot.
     * 
     * Once the client is done filling this buffer, it is expected to transfer
     * buffer ownership back to the server with either cancelBuffer on
     * the dequeued slot or to fill in the contents of its associated buffer
     * contents and call queueBuffer.
     * 
     * If dequeueBuffer returns the BUFFER_NEEDS_REALLOCATION flag, the client is
     * expected to call requestBuffer immediately.
     * 
     * If dequeueBuffer returns the RELEASE_ALL_BUFFERS flag, the client is
     * expected to release all of the mirrored slot->buffer mappings.
     * 
     * The fence parameter will be updated to hold the fence associated with
     * the buffer. The contents of the buffer must not be overwritten until the
     * fence signals. If the fence is Fence::NO_FENCE, the buffer may be written
     * immediately.
     * 
     * The width and height parameters must be no greater than the minimum of
     * GL_MAX_VIEWPORT_DIMS and GL_MAX_TEXTURE_SIZE (see: glGetIntegerv).
     * An error due to invalid dimensions might not be reported until
     * updateTexImage() is called.  If width and height are both zero, the
     * default values specified by setDefaultBufferSize() are used instead.
     * 
     * If the format is 0, the default format will be used.
     * 
     * The usage argument specifies gralloc buffer usage flags.  The values
     * are enumerated in <gralloc.h>, e.g. GRALLOC_USAGE_HW_RENDER.  These
     * will be merged with the usage flags specified by
     * IGraphicBufferConsumer::setConsumerUsageBits.
     * 
     * This call will block until a buffer is available to be dequeued. If
     * both the producer and consumer are controlled by the app, then this call
     * can never block and will return WOULD_BLOCK if no buffer is available.
     * 
     * A non-negative value with flags set (see above) will be returned upon
     * success as status.
     * 
     * Return of a negative means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned or the producer is not
     *             connected.
     * * BAD_VALUE - both in async mode and buffer count was less than the
     *               max numbers of buffers that can be allocated at once.
     * * INVALID_OPERATION - cannot attach the buffer because it would cause
     *                       too many buffers to be dequeued, either because
     *                       the producer already has a single buffer dequeued
     *                       and did not set a buffer count, or because a
     *                       buffer count was set and this call would cause
     *                       it to be exceeded.
     * * WOULD_BLOCK - no buffer is currently available, and blocking is disabled
     *                 since both the producer/consumer are controlled by app
     * * NO_MEMORY - out of memory, cannot allocate the graphics buffer.
     * * TIMED_OUT - the timeout set by setDequeueTimeout was exceeded while
     *               waiting for a buffer to become available.
     * 
     * All other negative values are an unknown error returned downstream
     * from the graphics allocator (typically errno).
     */
    virtual ::android::hardware::Return<void> dequeueBuffer(uint32_t width, uint32_t height, ::android::hardware::graphics::common::V1_0::PixelFormat format, uint32_t usage, bool getFrameTimestamps, dequeueBuffer_cb _hidl_cb) = 0;

    /**
     * detachBuffer attempts to remove all ownership of the buffer in the given
     * slot from the buffer queue. If this call succeeds, the slot will be
     * freed, and there will be no way to obtain the buffer from this interface.
     * The freed slot will remain unallocated until either it is selected to
     * hold a freshly allocated buffer in dequeueBuffer or a buffer is attached
     * to the slot. The buffer must have already been dequeued, and the caller
     * must already possesses the sp<AnwBuffer> (i.e., must have called
     * requestBuffer).
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned or the producer is not
     *             connected.
     * * BAD_VALUE - the given slot number is invalid, either because it is
     *               out of the range [0, NUM_BUFFER_SLOTS), or because the slot
     *               it refers to is not currently dequeued and requested.
     */
    virtual ::android::hardware::Return<int32_t> detachBuffer(int32_t slot) = 0;

    using detachNextBuffer_cb = std::function<void(int32_t status, const ::android::hardware::media::V1_0::AnwBuffer& buffer, const ::android::hardware::hidl_handle& fence)>;
    /**
     * detachNextBuffer is equivalent to calling dequeueBuffer, requestBuffer,
     * and detachBuffer in sequence, except for two things:
     * 
     * 1) It is unnecessary to know the dimensions, format, or usage of the
     *    next buffer.
     * 2) It will not block, since if it cannot find an appropriate buffer to
     *    return, it will return an error instead.
     * 
     * Only slots that are free but still contain an AnwBuffer will be
     * considered, and the oldest of those will be returned. buffer is
     * equivalent to buffer from the requestBuffer call, and fence is
     * equivalent to fence from the dequeueBuffer call.
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned or the producer is not
     *             connected.
     * * BAD_VALUE - either outBuffer or outFence were NULL.
     * * NO_MEMORY - no slots were found that were both free and contained a
     *               AnwBuffer.
     */
    virtual ::android::hardware::Return<void> detachNextBuffer(detachNextBuffer_cb _hidl_cb) = 0;

    using attachBuffer_cb = std::function<void(int32_t status, int32_t slot)>;
    /**
     * attachBuffer attempts to transfer ownership of a buffer to the buffer
     * queue. If this call succeeds, it will be as if this buffer was dequeued
     * from the returned slot number. As such, this call will fail if attaching
     * this buffer would cause too many buffers to be simultaneously dequeued.
     * 
     * If attachBuffer returns the RELEASE_ALL_BUFFERS flag, the caller is
     * expected to release all of the mirrored slot->buffer mappings.
     * 
     * A non-negative value with flags set (see above) will be returned upon
     * success.
     * 
     * Return of a negative value means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned or the producer is not
     *             connected.
     * * BAD_VALUE - outSlot or buffer were NULL, invalid combination of
     *               async mode and buffer count override, or the generation
     *               number of the buffer did not match the buffer queue.
     * * INVALID_OPERATION - cannot attach the buffer because it would cause
     *                       too many buffers to be dequeued, either because
     *                       the producer already has a single buffer dequeued
     *                       and did not set a buffer count, or because a
     *                       buffer count was set and this call would cause
     *                       it to be exceeded.
     * * WOULD_BLOCK - no buffer slot is currently available, and blocking is
     *                 disabled since both the producer/consumer are
     *                 controlled by the app.
     * * TIMED_OUT - the timeout set by setDequeueTimeout was exceeded while
     *               waiting for a slot to become available.
     */
    virtual ::android::hardware::Return<void> attachBuffer(const ::android::hardware::media::V1_0::AnwBuffer& buffer, attachBuffer_cb _hidl_cb) = 0;

    using queueBuffer_cb = std::function<void(int32_t status, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput& output)>;
    /**
     * queueBuffer indicates that the client has finished filling in the
     * contents of the buffer associated with slot and transfers ownership of
     * that slot back to the server.
     * 
     * It is not valid to call queueBuffer on a slot that is not owned
     * by the client or one for which a buffer associated via requestBuffer
     * (an attempt to do so will fail with a return value of BAD_VALUE).
     * 
     * In addition, the input must be described by the client (as documented
     * below). Any other properties (zero point, etc)
     * are client-dependent, and should be documented by the client.
     * 
     * The slot must be in the range of [0, NUM_BUFFER_SLOTS).
     * 
     * Upon success, the output will be filled with meaningful values
     * (refer to the documentation below).
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned or the producer is not
     *             connected.
     * * BAD_VALUE - one of the below conditions occurred:
     *              * fence was NULL
     *              * scaling mode was unknown
     *              * both in async mode and buffer count was less than the
     *                max numbers of buffers that can be allocated at once
     *              * slot index was out of range (see above).
     *              * the slot was not in the dequeued state
     *              * the slot was enqueued without requesting a buffer
     *              * crop rect is out of bounds of the buffer dimensions
     */
    virtual ::android::hardware::Return<void> queueBuffer(int32_t slot, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput& input, queueBuffer_cb _hidl_cb) = 0;

    /**
     * cancelBuffer indicates that the client does not wish to fill in the
     * buffer associated with slot and transfers ownership of the slot back to
     * the server.
     * 
     * The buffer is not queued for use by the consumer.
     * 
     * The slot must be in the range of [0, NUM_BUFFER_SLOTS).
     * 
     * The buffer will not be overwritten until the fence signals.  The fence
     * will usually be the one obtained from dequeueBuffer.
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned or the producer is not
     *             connected.
     * * BAD_VALUE - one of the below conditions occurred:
     *              * fence was NULL
     *              * slot index was out of range (see above).
     *              * the slot was not in the dequeued state
     */
    virtual ::android::hardware::Return<int32_t> cancelBuffer(int32_t slot, const ::android::hardware::hidl_handle& fence) = 0;

    using query_cb = std::function<void(int32_t result, int32_t value)>;
    /**
     * query retrieves some information for this surface
     * 'what' tokens allowed are that of NATIVE_WINDOW_* in <window.h>
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - the buffer queue has been abandoned.
     * * BAD_VALUE - what was out of range
     */
    virtual ::android::hardware::Return<void> query(int32_t what, query_cb _hidl_cb) = 0;

    using connect_cb = std::function<void(int32_t status, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput& output)>;
    /**
     * connect attempts to connect a client API to the IGraphicBufferProducer.
     * This must be called before any other IGraphicBufferProducer methods are
     * called except for getAllocator. A consumer must be already connected.
     * 
     * This method will fail if the connect was previously called on the
     * IGraphicBufferProducer and no corresponding disconnect call was made.
     * 
     * The listener is an optional binder callback object that can be used if
     * the producer wants to be notified when the consumer releases a buffer
     * back to the BufferQueue. It is also used to detect the death of the
     * producer. If only the latter functionality is desired, there is a
     * DummyProducerListener class in IProducerListener.h that can be used.
     * 
     * The api should be one of the NATIVE_WINDOW_API_* values in <window.h>
     * 
     * The producerControlledByApp should be set to true if the producer is hosted
     * by an untrusted process (typically app_process-forked processes). If both
     * the producer and the consumer are app-controlled then all buffer queues
     * will operate in async mode regardless of the async flag.
     * 
     * Upon success, the output will be filled with meaningful data
     * (refer to QueueBufferOutput documentation above).
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * NO_INIT - one of the following occurred:
     *             * the buffer queue was abandoned
     *             * no consumer has yet connected
     * * BAD_VALUE - one of the following has occurred:
     *             * the producer is already connected
     *             * api was out of range (see above).
     *             * output was NULL.
     *             * Failure to adjust the number of available slots. This can
     *               happen because of trying to allocate/deallocate the async
     *               buffer in response to the value of producerControlledByApp.
     * * DEAD_OBJECT - the token is hosted by an already-dead process
     * 
     * Additional negative errors may be returned by the internals, they
     * should be treated as opaque fatal unrecoverable errors.
     */
    virtual ::android::hardware::Return<void> connect(const ::android::sp<::android::hardware::graphics::bufferqueue::V1_0::IProducerListener>& listener, int32_t api, bool producerControlledByApp, connect_cb _hidl_cb) = 0;

    /**
     * disconnect attempts to disconnect a client API from the
     * IGraphicBufferProducer.  Calling this method will cause any subsequent
     * calls to other IGraphicBufferProducer methods to fail except for
     * getAllocator and connect.  Successfully calling connect after this will
     * allow the other methods to succeed again.
     * 
     * The api should be one of the NATIVE_WINDOW_API_* values in <window.h>
     * 
     * Alternatively if mode is AllLocal, then the API value is ignored, and any API
     * connected from the same PID calling disconnect will be disconnected.
     * 
     * Disconnecting from an abandoned IGraphicBufferProducer is legal and
     * is considered a no-op.
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * BAD_VALUE - one of the following has occurred:
     *             * the api specified does not match the one that was connected
     *             * api was out of range (see above).
     * * DEAD_OBJECT - the token is hosted by an already-dead process
     */
    virtual ::android::hardware::Return<int32_t> disconnect(int32_t api, ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode mode) = 0;

    /**
     * Attaches a sideband buffer stream to the IGraphicBufferProducer.
     * 
     * A sideband stream is a device-specific mechanism for passing buffers
     * from the producer to the consumer without using dequeueBuffer/
     * queueBuffer. If a sideband stream is present, the consumer can choose
     * whether to acquire buffers from the sideband stream or from the queued
     * buffers.
     * 
     * Passing NULL or a different stream handle will detach the previous
     * handle if any.
     */
    virtual ::android::hardware::Return<int32_t> setSidebandStream(const ::android::hardware::hidl_handle& stream) = 0;

    /**
     * Allocates buffers based on the given dimensions/format.
     * 
     * This function will allocate up to the maximum number of buffers
     * permitted by the current BufferQueue configuration. It will use the
     * given format, dimensions, and usage bits, which are interpreted in the
     * same way as for dequeueBuffer, and the async flag must be set the same
     * way as for dequeueBuffer to ensure that the correct number of buffers are
     * allocated. This is most useful to avoid an allocation delay during
     * dequeueBuffer. If there are already the maximum number of buffers
     * allocated, this function has no effect.
     */
    virtual ::android::hardware::Return<void> allocateBuffers(uint32_t width, uint32_t height, ::android::hardware::graphics::common::V1_0::PixelFormat format, uint32_t usage) = 0;

    /**
     * Sets whether dequeueBuffer is allowed to allocate new buffers.
     * 
     * Normally dequeueBuffer does not discriminate between free slots which
     * already have an allocated buffer and those which do not, and will
     * allocate a new buffer if the slot doesn't have a buffer or if the slot's
     * buffer doesn't match the requested size, format, or usage. This method
     * allows the producer to restrict the eligible slots to those which already
     * have an allocated buffer of the correct size, format, and usage. If no
     * eligible slot is available, dequeueBuffer will block or return an error
     * as usual.
     */
    virtual ::android::hardware::Return<int32_t> allowAllocation(bool allow) = 0;

    /**
     * Sets the current generation number of the BufferQueue.
     * 
     * This generation number will be inserted into any buffers allocated by the
     * BufferQueue, and any attempts to attach a buffer with a different
     * generation number will fail. Buffers already in the queue are not
     * affected and will retain their current generation number. The generation
     * number defaults to 0.
     */
    virtual ::android::hardware::Return<int32_t> setGenerationNumber(uint32_t generationNumber) = 0;

    using getConsumerName_cb = std::function<void(const ::android::hardware::hidl_string& name)>;
    /**
     * Returns the name of the connected consumer.
     */
    virtual ::android::hardware::Return<void> getConsumerName(getConsumerName_cb _hidl_cb) = 0;

    /**
     * Used to enable/disable shared buffer mode.
     * 
     * When shared buffer mode is enabled the first buffer that is queued or
     * dequeued will be cached and returned to all subsequent calls to
     * dequeueBuffer and acquireBuffer. This allows the producer and consumer to
     * simultaneously access the same buffer.
     */
    virtual ::android::hardware::Return<int32_t> setSharedBufferMode(bool sharedBufferMode) = 0;

    /**
     * Used to enable/disable auto-refresh.
     * 
     * Auto refresh has no effect outside of shared buffer mode. In shared
     * buffer mode, when enabled, it indicates to the consumer that it should
     * attempt to acquire buffers even if it is not aware of any being
     * available.
     */
    virtual ::android::hardware::Return<int32_t> setAutoRefresh(bool autoRefresh) = 0;

    /**
     * Sets how long dequeueBuffer will wait for a buffer to become available
     * before returning an error (TIMED_OUT).
     * 
     * This timeout also affects the attachBuffer call, which will block if
     * there is not a free slot available into which the attached buffer can be
     * placed.
     * 
     * By default, the BufferQueue will wait forever, which is indicated by a
     * timeout of -1. If set (to a value other than -1), this will disable
     * non-blocking mode and its corresponding spare buffer (which is used to
     * ensure a buffer is always available).
     * 
     * Return of a value other than NO_ERROR means an error has occurred:
     * * BAD_VALUE - Failure to adjust the number of available slots. This can
     *               happen because of trying to allocate/deallocate the async
     *               buffer.
     */
    virtual ::android::hardware::Return<int32_t> setDequeueTimeout(int64_t timeoutNs) = 0;

    using getLastQueuedBuffer_cb = std::function<void(int32_t status, const ::android::hardware::media::V1_0::AnwBuffer& buffer, const ::android::hardware::hidl_handle& fence, const ::android::hardware::hidl_array<float, 16>& transformMatrix)>;
    /**
     * Returns the last queued buffer along with a fence which must signal
     * before the contents of the buffer are read. If there are no buffers in
     * the queue, buffer.nativeHandle and fence will be null handles.
     * 
     * transformMatrix is meaningless if buffer.nativeHandle is null.
     */
    virtual ::android::hardware::Return<void> getLastQueuedBuffer(getLastQueuedBuffer_cb _hidl_cb) = 0;

    using getFrameTimestamps_cb = std::function<void(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventHistoryDelta& timeStamps)>;
    /**
     * Gets the frame events that haven't already been retrieved.
     */
    virtual ::android::hardware::Return<void> getFrameTimestamps(getFrameTimestamps_cb _hidl_cb) = 0;

    using getUniqueId_cb = std::function<void(int32_t status, uint64_t outId)>;
    /**
     * Returns a unique id for this BufferQueue.
     */
    virtual ::android::hardware::Return<void> getUniqueId(getUniqueId_cb _hidl_cb) = 0;

    using interfaceChain_cb = std::function<void(const ::android::hardware::hidl_vec<::android::hardware::hidl_string>& descriptors)>;
    virtual ::android::hardware::Return<void> interfaceChain(interfaceChain_cb _hidl_cb) override;

    virtual ::android::hardware::Return<void> debug(const ::android::hardware::hidl_handle& fd, const ::android::hardware::hidl_vec<::android::hardware::hidl_string>& options) override;

    using interfaceDescriptor_cb = std::function<void(const ::android::hardware::hidl_string& descriptor)>;
    virtual ::android::hardware::Return<void> interfaceDescriptor(interfaceDescriptor_cb _hidl_cb) override;

    using getHashChain_cb = std::function<void(const ::android::hardware::hidl_vec<::android::hardware::hidl_array<uint8_t, 32>>& hashchain)>;
    virtual ::android::hardware::Return<void> getHashChain(getHashChain_cb _hidl_cb) override;

    virtual ::android::hardware::Return<void> setHALInstrumentation() override;

    virtual ::android::hardware::Return<bool> linkToDeath(const ::android::sp<::android::hardware::hidl_death_recipient>& recipient, uint64_t cookie) override;

    virtual ::android::hardware::Return<void> ping() override;

    using getDebugInfo_cb = std::function<void(const ::android::hidl::base::V1_0::DebugInfo& info)>;
    virtual ::android::hardware::Return<void> getDebugInfo(getDebugInfo_cb _hidl_cb) override;

    virtual ::android::hardware::Return<void> notifySyspropsChanged() override;

    virtual ::android::hardware::Return<bool> unlinkToDeath(const ::android::sp<::android::hardware::hidl_death_recipient>& recipient) override;
    // cast static functions
    static ::android::hardware::Return<::android::sp<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer>> castFrom(const ::android::sp<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer>& parent, bool emitError = false);
    static ::android::hardware::Return<::android::sp<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer>> castFrom(const ::android::sp<::android::hidl::base::V1_0::IBase>& parent, bool emitError = false);

    static const char* descriptor;

    static ::android::sp<IGraphicBufferProducer> tryGetService(const std::string &serviceName="default", bool getStub=false);
    static ::android::sp<IGraphicBufferProducer> tryGetService(const char serviceName[], bool getStub=false)  { std::string str(serviceName ? serviceName : "");      return tryGetService(str, getStub); }
    static ::android::sp<IGraphicBufferProducer> tryGetService(const ::android::hardware::hidl_string& serviceName, bool getStub=false)  { std::string str(serviceName.c_str());      return tryGetService(str, getStub); }
    static ::android::sp<IGraphicBufferProducer> tryGetService(bool getStub) { return tryGetService("default", getStub); }
    static ::android::sp<IGraphicBufferProducer> getService(const std::string &serviceName="default", bool getStub=false);
    static ::android::sp<IGraphicBufferProducer> getService(const char serviceName[], bool getStub=false)  { std::string str(serviceName ? serviceName : "");      return getService(str, getStub); }
    static ::android::sp<IGraphicBufferProducer> getService(const ::android::hardware::hidl_string& serviceName, bool getStub=false)  { std::string str(serviceName.c_str());      return getService(str, getStub); }
    static ::android::sp<IGraphicBufferProducer> getService(bool getStub) { return getService("default", getStub); }
    __attribute__ ((warn_unused_result))::android::status_t registerAsService(const std::string &serviceName="default");
    static bool registerForNotifications(
            const std::string &serviceName,
            const ::android::sp<::android::hidl::manager::V1_0::IServiceNotification> &notification);
};

constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::EMPTY) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::EMPTY)) {
        os += (first ? "" : " | ");
        os += "EMPTY";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::EMPTY;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::FENCE) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::FENCE)) {
        os += (first ? "" : " | ");
        os += "FENCE";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::FENCE;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::SIGNAL_TIME) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::SIGNAL_TIME)) {
        os += (first ? "" : " | ");
        os += "SIGNAL_TIME";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::SIGNAL_TIME;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::EMPTY) {
        return "EMPTY";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::FENCE) {
        return "FENCE";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::SIGNAL_TIME) {
        return "SIGNAL_TIME";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

static inline std::string toString(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".state = ";
    os += ::android::hardware::graphics::bufferqueue::V1_0::toString(o.state);
    os += ", .fence = ";
    os += ::android::hardware::toString(o.fence);
    os += ", .signalTimeNs = ";
    os += ::android::hardware::toString(o.signalTimeNs);
    os += "}"; return os;
}

// operator== and operator!= are not generated for FenceTimeSnapshot

static inline std::string toString(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventsDelta& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".index = ";
    os += ::android::hardware::toString(o.index);
    os += ", .frameNumber = ";
    os += ::android::hardware::toString(o.frameNumber);
    os += ", .addPostCompositeCalled = ";
    os += ::android::hardware::toString(o.addPostCompositeCalled);
    os += ", .addRetireCalled = ";
    os += ::android::hardware::toString(o.addRetireCalled);
    os += ", .addReleaseCalled = ";
    os += ::android::hardware::toString(o.addReleaseCalled);
    os += ", .postedTimeNs = ";
    os += ::android::hardware::toString(o.postedTimeNs);
    os += ", .requestedPresentTimeNs = ";
    os += ::android::hardware::toString(o.requestedPresentTimeNs);
    os += ", .latchTimeNs = ";
    os += ::android::hardware::toString(o.latchTimeNs);
    os += ", .firstRefreshStartTimeNs = ";
    os += ::android::hardware::toString(o.firstRefreshStartTimeNs);
    os += ", .lastRefreshStartTimeNs = ";
    os += ::android::hardware::toString(o.lastRefreshStartTimeNs);
    os += ", .dequeueReadyTime = ";
    os += ::android::hardware::toString(o.dequeueReadyTime);
    os += ", .gpuCompositionDoneFence = ";
    os += ::android::hardware::graphics::bufferqueue::V1_0::toString(o.gpuCompositionDoneFence);
    os += ", .displayPresentFence = ";
    os += ::android::hardware::graphics::bufferqueue::V1_0::toString(o.displayPresentFence);
    os += ", .displayRetireFence = ";
    os += ::android::hardware::graphics::bufferqueue::V1_0::toString(o.displayRetireFence);
    os += ", .releaseFence = ";
    os += ::android::hardware::graphics::bufferqueue::V1_0::toString(o.releaseFence);
    os += "}"; return os;
}

// operator== and operator!= are not generated for FrameEventsDelta

static inline std::string toString(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".deadlineNs = ";
    os += ::android::hardware::toString(o.deadlineNs);
    os += ", .intervalNs = ";
    os += ::android::hardware::toString(o.intervalNs);
    os += ", .presentLatencyNs = ";
    os += ::android::hardware::toString(o.presentLatencyNs);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming& lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming& rhs) {
    if (lhs.deadlineNs != rhs.deadlineNs) {
        return false;
    }
    if (lhs.intervalNs != rhs.intervalNs) {
        return false;
    }
    if (lhs.presentLatencyNs != rhs.presentLatencyNs) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming& lhs,const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::CompositorTiming& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FrameEventHistoryDelta& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".deltas = ";
    os += ::android::hardware::toString(o.deltas);
    os += ", .compositorTiming = ";
    os += ::android::hardware::graphics::bufferqueue::V1_0::toString(o.compositorTiming);
    os += "}"; return os;
}

// operator== and operator!= are not generated for FrameEventHistoryDelta

constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}

constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}

constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}

constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}

constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode e) {
    v |= static_cast<int32_t>(e);
    return v;
}

constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
template<>
inline std::string toString<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::API) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::API)) {
        os += (first ? "" : " | ");
        os += "API";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::API;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::ALL_LOCAL) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::ALL_LOCAL)) {
        os += (first ? "" : " | ");
        os += "ALL_LOCAL";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::ALL_LOCAL;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::API) {
        return "API";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::ALL_LOCAL) {
        return "ALL_LOCAL";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

static inline std::string toString(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferInput& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".timestamp = ";
    os += ::android::hardware::toString(o.timestamp);
    os += ", .isAutoTimestamp = ";
    os += ::android::hardware::toString(o.isAutoTimestamp);
    os += ", .dataSpace = ";
    os += ::android::hardware::graphics::common::V1_0::toString(o.dataSpace);
    os += ", .crop = ";
    os += ::android::hardware::media::V1_0::toString(o.crop);
    os += ", .scalingMode = ";
    os += ::android::hardware::toString(o.scalingMode);
    os += ", .transform = ";
    os += ::android::hardware::toString(o.transform);
    os += ", .stickyTransform = ";
    os += ::android::hardware::toString(o.stickyTransform);
    os += ", .fence = ";
    os += ::android::hardware::toString(o.fence);
    os += ", .surfaceDamage = ";
    os += ::android::hardware::toString(o.surfaceDamage);
    os += ", .getFrameTimestamps = ";
    os += ::android::hardware::toString(o.getFrameTimestamps);
    os += "}"; return os;
}

// operator== and operator!= are not generated for QueueBufferInput

static inline std::string toString(const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::QueueBufferOutput& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".width = ";
    os += ::android::hardware::toString(o.width);
    os += ", .height = ";
    os += ::android::hardware::toString(o.height);
    os += ", .transformHint = ";
    os += ::android::hardware::toString(o.transformHint);
    os += ", .numPendingBuffers = ";
    os += ::android::hardware::toString(o.numPendingBuffers);
    os += ", .nextFrameNumber = ";
    os += ::android::hardware::toString(o.nextFrameNumber);
    os += ", .bufferReplaced = ";
    os += ::android::hardware::toString(o.bufferReplaced);
    os += ", .frameTimestamps = ";
    os += ::android::hardware::graphics::bufferqueue::V1_0::toString(o.frameTimestamps);
    os += "}"; return os;
}

// operator== and operator!= are not generated for QueueBufferOutput

static inline std::string toString(const ::android::sp<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer>& o) {
    std::string os = "[class or subclass of ";
    os += ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::descriptor;
    os += "]";
    os += o->isRemote() ? "@remote" : "@local";
    return os;
}


}  // namespace V1_0
}  // namespace bufferqueue
}  // namespace graphics
}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State>
{
    const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State* begin() { return static_begin(); }
    const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State* end() { return begin() + 3; }
    private:
    static const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State* static_begin() {
        static const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State kVals[3] {
            ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::EMPTY,
            ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::FENCE,
            ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::FenceTimeSnapshot::State::SIGNAL_TIME,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android
namespace android {
namespace hardware {
template<> struct hidl_enum_iterator<::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode>
{
    const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode* begin() { return static_begin(); }
    const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode* end() { return begin() + 2; }
    private:
    static const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode* static_begin() {
        static const ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode kVals[2] {
            ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::API,
            ::android::hardware::graphics::bufferqueue::V1_0::IGraphicBufferProducer::DisconnectMode::ALL_LOCAL,
        };
        return &kVals[0];
    }};

}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_IGRAPHICBUFFERPRODUCER_H
