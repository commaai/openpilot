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

#ifndef ANDROID_GUI_SURFACE_H
#define ANDROID_GUI_SURFACE_H

#include <gui/IGraphicBufferProducer.h>
#include <gui/BufferQueue.h>

#include <ui/ANativeObjectBase.h>
#include <ui/Region.h>

#include <utils/RefBase.h>
#include <utils/threads.h>
#include <utils/KeyedVector.h>

struct ANativeWindow_Buffer;

namespace android {

/*
 * An implementation of ANativeWindow that feeds graphics buffers into a
 * BufferQueue.
 *
 * This is typically used by programs that want to render frames through
 * some means (maybe OpenGL, a software renderer, or a hardware decoder)
 * and have the frames they create forwarded to SurfaceFlinger for
 * compositing.  For example, a video decoder could render a frame and call
 * eglSwapBuffers(), which invokes ANativeWindow callbacks defined by
 * Surface.  Surface then forwards the buffers through Binder IPC
 * to the BufferQueue's producer interface, providing the new frame to a
 * consumer such as GLConsumer.
 */
class Surface
    : public ANativeObjectBase<ANativeWindow, Surface, RefBase>
{
public:

    /*
     * creates a Surface from the given IGraphicBufferProducer (which concrete
     * implementation is a BufferQueue).
     *
     * Surface is mainly state-less while it's disconnected, it can be
     * viewed as a glorified IGraphicBufferProducer holder. It's therefore
     * safe to create other Surfaces from the same IGraphicBufferProducer.
     *
     * However, once a Surface is connected, it'll prevent other Surfaces
     * referring to the same IGraphicBufferProducer to become connected and
     * therefore prevent them to be used as actual producers of buffers.
     *
     * the controlledByApp flag indicates that this Surface (producer) is
     * controlled by the application. This flag is used at connect time.
     */
    Surface(const sp<IGraphicBufferProducer>& bufferProducer, bool controlledByApp = false);

    /* getIGraphicBufferProducer() returns the IGraphicBufferProducer this
     * Surface was created with. Usually it's an error to use the
     * IGraphicBufferProducer while the Surface is connected.
     */
    sp<IGraphicBufferProducer> getIGraphicBufferProducer() const;

    /* convenience function to check that the given surface is non NULL as
     * well as its IGraphicBufferProducer */
    static bool isValid(const sp<Surface>& surface) {
        return surface != NULL && surface->getIGraphicBufferProducer() != NULL;
    }

    /* Attaches a sideband buffer stream to the Surface's IGraphicBufferProducer.
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
    void setSidebandStream(const sp<NativeHandle>& stream);

    /* Allocates buffers based on the current dimensions/format.
     *
     * This function will allocate up to the maximum number of buffers
     * permitted by the current BufferQueue configuration. It will use the
     * default format and dimensions. This is most useful to avoid an allocation
     * delay during dequeueBuffer. If there are already the maximum number of
     * buffers allocated, this function has no effect.
     */
    void allocateBuffers();

    /* Sets the generation number on the IGraphicBufferProducer and updates the
     * generation number on any buffers attached to the Surface after this call.
     * See IGBP::setGenerationNumber for more information. */
    status_t setGenerationNumber(uint32_t generationNumber);

    // See IGraphicBufferProducer::getConsumerName
    String8 getConsumerName() const;

protected:
    virtual ~Surface();

private:
    // can't be copied
    Surface& operator = (const Surface& rhs);
    Surface(const Surface& rhs);

    // ANativeWindow hooks
    static int hook_cancelBuffer(ANativeWindow* window,
            ANativeWindowBuffer* buffer, int fenceFd);
    static int hook_dequeueBuffer(ANativeWindow* window,
            ANativeWindowBuffer** buffer, int* fenceFd);
    static int hook_perform(ANativeWindow* window, int operation, ...);
    static int hook_query(const ANativeWindow* window, int what, int* value);
    static int hook_queueBuffer(ANativeWindow* window,
            ANativeWindowBuffer* buffer, int fenceFd);
    static int hook_setSwapInterval(ANativeWindow* window, int interval);

    static int hook_cancelBuffer_DEPRECATED(ANativeWindow* window,
            ANativeWindowBuffer* buffer);
    static int hook_dequeueBuffer_DEPRECATED(ANativeWindow* window,
            ANativeWindowBuffer** buffer);
    static int hook_lockBuffer_DEPRECATED(ANativeWindow* window,
            ANativeWindowBuffer* buffer);
    static int hook_queueBuffer_DEPRECATED(ANativeWindow* window,
            ANativeWindowBuffer* buffer);

    int dispatchConnect(va_list args);
    int dispatchDisconnect(va_list args);
    int dispatchSetBufferCount(va_list args);
    int dispatchSetBuffersGeometry(va_list args);
    int dispatchSetBuffersDimensions(va_list args);
    int dispatchSetBuffersUserDimensions(va_list args);
    int dispatchSetBuffersFormat(va_list args);
    int dispatchSetScalingMode(va_list args);
    int dispatchSetBuffersTransform(va_list args);
    int dispatchSetBuffersStickyTransform(va_list args);
    int dispatchSetBuffersTimestamp(va_list args);
    int dispatchSetCrop(va_list args);
    int dispatchSetPostTransformCrop(va_list args);
    int dispatchSetUsage(va_list args);
    int dispatchLock(va_list args);
    int dispatchUnlockAndPost(va_list args);
    int dispatchSetSidebandStream(va_list args);
    int dispatchSetBuffersDataSpace(va_list args);
    int dispatchSetSurfaceDamage(va_list args);

protected:
    virtual int dequeueBuffer(ANativeWindowBuffer** buffer, int* fenceFd);
    virtual int cancelBuffer(ANativeWindowBuffer* buffer, int fenceFd);
    virtual int queueBuffer(ANativeWindowBuffer* buffer, int fenceFd);
    virtual int perform(int operation, va_list args);
    virtual int query(int what, int* value) const;
    virtual int setSwapInterval(int interval);

    virtual int lockBuffer_DEPRECATED(ANativeWindowBuffer* buffer);

    virtual int connect(int api);
    virtual int disconnect(int api);
    virtual int setBufferCount(int bufferCount);
    virtual int setBuffersDimensions(uint32_t width, uint32_t height);
    virtual int setBuffersUserDimensions(uint32_t width, uint32_t height);
    virtual int setBuffersFormat(PixelFormat format);
    virtual int setScalingMode(int mode);
    virtual int setBuffersTransform(uint32_t transform);
    virtual int setBuffersStickyTransform(uint32_t transform);
    virtual int setBuffersTimestamp(int64_t timestamp);
    virtual int setBuffersDataSpace(android_dataspace dataSpace);
    virtual int setCrop(Rect const* rect);
    virtual int setUsage(uint32_t reqUsage);
    virtual void setSurfaceDamage(android_native_rect_t* rects, size_t numRects);

public:
    virtual int lock(ANativeWindow_Buffer* outBuffer, ARect* inOutDirtyBounds);
    virtual int unlockAndPost();

    virtual int connect(int api, const sp<IProducerListener>& listener);
    virtual int detachNextBuffer(sp<GraphicBuffer>* outBuffer,
            sp<Fence>* outFence);
    virtual int attachBuffer(ANativeWindowBuffer*);

protected:
    enum { NUM_BUFFER_SLOTS = BufferQueue::NUM_BUFFER_SLOTS };
    enum { DEFAULT_FORMAT = PIXEL_FORMAT_RGBA_8888 };

private:
    void freeAllBuffers();
    int getSlotFromBufferLocked(android_native_buffer_t* buffer) const;

    struct BufferSlot {
        sp<GraphicBuffer> buffer;
        Region dirtyRegion;
    };

    // mSurfaceTexture is the interface to the surface texture server. All
    // operations on the surface texture client ultimately translate into
    // interactions with the server using this interface.
    // TODO: rename to mBufferProducer
    sp<IGraphicBufferProducer> mGraphicBufferProducer;

    // mSlots stores the buffers that have been allocated for each buffer slot.
    // It is initialized to null pointers, and gets filled in with the result of
    // IGraphicBufferProducer::requestBuffer when the client dequeues a buffer from a
    // slot that has not yet been used. The buffer allocated to a slot will also
    // be replaced if the requested buffer usage or geometry differs from that
    // of the buffer allocated to a slot.
    BufferSlot mSlots[NUM_BUFFER_SLOTS];

    // mReqWidth is the buffer width that will be requested at the next dequeue
    // operation. It is initialized to 1.
    uint32_t mReqWidth;

    // mReqHeight is the buffer height that will be requested at the next
    // dequeue operation. It is initialized to 1.
    uint32_t mReqHeight;

    // mReqFormat is the buffer pixel format that will be requested at the next
    // deuque operation. It is initialized to PIXEL_FORMAT_RGBA_8888.
    PixelFormat mReqFormat;

    // mReqUsage is the set of buffer usage flags that will be requested
    // at the next deuque operation. It is initialized to 0.
    uint32_t mReqUsage;

    // mTimestamp is the timestamp that will be used for the next buffer queue
    // operation. It defaults to NATIVE_WINDOW_TIMESTAMP_AUTO, which means that
    // a timestamp is auto-generated when queueBuffer is called.
    int64_t mTimestamp;

    // mDataSpace is the buffer dataSpace that will be used for the next buffer
    // queue operation. It defaults to HAL_DATASPACE_UNKNOWN, which
    // means that the buffer contains some type of color data.
    android_dataspace mDataSpace;

    // mCrop is the crop rectangle that will be used for the next buffer
    // that gets queued. It is set by calling setCrop.
    Rect mCrop;

    // mScalingMode is the scaling mode that will be used for the next
    // buffers that get queued. It is set by calling setScalingMode.
    int mScalingMode;

    // mTransform is the transform identifier that will be used for the next
    // buffer that gets queued. It is set by calling setTransform.
    uint32_t mTransform;

    // mStickyTransform is a transform that is applied on top of mTransform
    // in each buffer that is queued.  This is typically used to force the
    // compositor to apply a transform, and will prevent the transform hint
    // from being set by the compositor.
    uint32_t mStickyTransform;

    // mDefaultWidth is default width of the buffers, regardless of the
    // native_window_set_buffers_dimensions call.
    uint32_t mDefaultWidth;

    // mDefaultHeight is default height of the buffers, regardless of the
    // native_window_set_buffers_dimensions call.
    uint32_t mDefaultHeight;

    // mUserWidth, if non-zero, is an application-specified override
    // of mDefaultWidth.  This is lower priority than the width set by
    // native_window_set_buffers_dimensions.
    uint32_t mUserWidth;

    // mUserHeight, if non-zero, is an application-specified override
    // of mDefaultHeight.  This is lower priority than the height set
    // by native_window_set_buffers_dimensions.
    uint32_t mUserHeight;

    // mTransformHint is the transform probably applied to buffers of this
    // window. this is only a hint, actual transform may differ.
    uint32_t mTransformHint;

    // mProducerControlledByApp whether this buffer producer is controlled
    // by the application
    bool mProducerControlledByApp;

    // mSwapIntervalZero set if we should drop buffers at queue() time to
    // achieve an asynchronous swap interval
    bool mSwapIntervalZero;

    // mConsumerRunningBehind whether the consumer is running more than
    // one buffer behind the producer.
    mutable bool mConsumerRunningBehind;

    // mMutex is the mutex used to prevent concurrent access to the member
    // variables of Surface objects. It must be locked whenever the
    // member variables are accessed.
    mutable Mutex mMutex;

    // must be used from the lock/unlock thread
    sp<GraphicBuffer>           mLockedBuffer;
    sp<GraphicBuffer>           mPostedBuffer;
    bool                        mConnectedToCpu;

    // When a CPU producer is attached, this reflects the region that the
    // producer wished to update as well as whether the Surface was able to copy
    // the previous buffer back to allow a partial update.
    //
    // When a non-CPU producer is attached, this reflects the surface damage
    // (the change since the previous frame) passed in by the producer.
    Region mDirtyRegion;

    // Stores the current generation number. See setGenerationNumber and
    // IGraphicBufferProducer::setGenerationNumber for more information.
    uint32_t mGenerationNumber;
};

}; // namespace android

#endif  // ANDROID_GUI_SURFACE_H
