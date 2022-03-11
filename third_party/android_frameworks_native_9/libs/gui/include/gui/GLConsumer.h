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

#ifndef ANDROID_GUI_CONSUMER_H
#define ANDROID_GUI_CONSUMER_H

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <gui/BufferQueueDefs.h>
#include <gui/ConsumerBase.h>

#include <ui/FenceTime.h>
#include <ui/GraphicBuffer.h>

#include <utils/String8.h>
#include <utils/Vector.h>
#include <utils/threads.h>

namespace android {
// ----------------------------------------------------------------------------


class String8;

/*
 * GLConsumer consumes buffers of graphics data from a BufferQueue,
 * and makes them available to OpenGL as a texture.
 *
 * A typical usage pattern is to set up the GLConsumer with the
 * desired options, and call updateTexImage() when a new frame is desired.
 * If a new frame is available, the texture will be updated.  If not,
 * the previous contents are retained.
 *
 * By default, the texture is attached to the GL_TEXTURE_EXTERNAL_OES
 * texture target, in the EGL context of the first thread that calls
 * updateTexImage().
 *
 * This class was previously called SurfaceTexture.
 */
class GLConsumer : public ConsumerBase {
public:
    enum { TEXTURE_EXTERNAL = 0x8D65 }; // GL_TEXTURE_EXTERNAL_OES
    typedef ConsumerBase::FrameAvailableListener FrameAvailableListener;

    // GLConsumer constructs a new GLConsumer object. If the constructor with
    // the tex parameter is used, tex indicates the name of the OpenGL ES
    // texture to which images are to be streamed. texTarget specifies the
    // OpenGL ES texture target to which the texture will be bound in
    // updateTexImage. useFenceSync specifies whether fences should be used to
    // synchronize access to buffers if that behavior is enabled at
    // compile-time.
    //
    // A GLConsumer may be detached from one OpenGL ES context and then
    // attached to a different context using the detachFromContext and
    // attachToContext methods, respectively. The intention of these methods is
    // purely to allow a GLConsumer to be transferred from one consumer
    // context to another. If such a transfer is not needed there is no
    // requirement that either of these methods be called.
    //
    // If the constructor with the tex parameter is used, the GLConsumer is
    // created in a state where it is considered attached to an OpenGL ES
    // context for the purposes of the attachToContext and detachFromContext
    // methods. However, despite being considered "attached" to a context, the
    // specific OpenGL ES context doesn't get latched until the first call to
    // updateTexImage. After that point, all calls to updateTexImage must be
    // made with the same OpenGL ES context current.
    //
    // If the constructor without the tex parameter is used, the GLConsumer is
    // created in a detached state, and attachToContext must be called before
    // calls to updateTexImage.
    GLConsumer(const sp<IGraphicBufferConsumer>& bq,
            uint32_t tex, uint32_t texureTarget, bool useFenceSync,
            bool isControlledByApp);

    GLConsumer(const sp<IGraphicBufferConsumer>& bq, uint32_t texureTarget,
            bool useFenceSync, bool isControlledByApp);

    // updateTexImage acquires the most recently queued buffer, and sets the
    // image contents of the target texture to it.
    //
    // This call may only be made while the OpenGL ES context to which the
    // target texture belongs is bound to the calling thread.
    //
    // This calls doGLFenceWait to ensure proper synchronization.
    status_t updateTexImage();

    // releaseTexImage releases the texture acquired in updateTexImage().
    // This is intended to be used in single buffer mode.
    //
    // This call may only be made while the OpenGL ES context to which the
    // target texture belongs is bound to the calling thread.
    status_t releaseTexImage();

    // setReleaseFence stores a fence that will signal when the current buffer
    // is no longer being read. This fence will be returned to the producer
    // when the current buffer is released by updateTexImage(). Multiple
    // fences can be set for a given buffer; they will be merged into a single
    // union fence.
    virtual void setReleaseFence(const sp<Fence>& fence);

    // getTransformMatrix retrieves the 4x4 texture coordinate transform matrix
    // associated with the texture image set by the most recent call to
    // updateTexImage.
    //
    // This transform matrix maps 2D homogeneous texture coordinates of the form
    // (s, t, 0, 1) with s and t in the inclusive range [0, 1] to the texture
    // coordinate that should be used to sample that location from the texture.
    // Sampling the texture outside of the range of this transform is undefined.
    //
    // This transform is necessary to compensate for transforms that the stream
    // content producer may implicitly apply to the content. By forcing users of
    // a GLConsumer to apply this transform we avoid performing an extra
    // copy of the data that would be needed to hide the transform from the
    // user.
    //
    // The matrix is stored in column-major order so that it may be passed
    // directly to OpenGL ES via the glLoadMatrixf or glUniformMatrix4fv
    // functions.
    void getTransformMatrix(float mtx[16]);

    // Computes the transform matrix documented by getTransformMatrix
    // from the BufferItem sub parts.
    static void computeTransformMatrix(float outTransform[16],
            const sp<GraphicBuffer>& buf, const Rect& cropRect,
            uint32_t transform, bool filtering);

    // Scale the crop down horizontally or vertically such that it has the
    // same aspect ratio as the buffer does.
    static Rect scaleDownCrop(const Rect& crop, uint32_t bufferWidth, uint32_t bufferHeight);

    // getTimestamp retrieves the timestamp associated with the texture image
    // set by the most recent call to updateTexImage.
    //
    // The timestamp is in nanoseconds, and is monotonically increasing. Its
    // other semantics (zero point, etc) are source-dependent and should be
    // documented by the source.
    int64_t getTimestamp();

    // getDataSpace retrieves the DataSpace associated with the texture image
    // set by the most recent call to updateTexImage.
    android_dataspace getCurrentDataSpace();

    // getFrameNumber retrieves the frame number associated with the texture
    // image set by the most recent call to updateTexImage.
    //
    // The frame number is an incrementing counter set to 0 at the creation of
    // the BufferQueue associated with this consumer.
    uint64_t getFrameNumber();

    // setDefaultBufferSize is used to set the size of buffers returned by
    // requestBuffers when a with and height of zero is requested.
    // A call to setDefaultBufferSize() may trigger requestBuffers() to
    // be called from the client.
    // The width and height parameters must be no greater than the minimum of
    // GL_MAX_VIEWPORT_DIMS and GL_MAX_TEXTURE_SIZE (see: glGetIntegerv).
    // An error due to invalid dimensions might not be reported until
    // updateTexImage() is called.
    status_t setDefaultBufferSize(uint32_t width, uint32_t height);

    // setFilteringEnabled sets whether the transform matrix should be computed
    // for use with bilinear filtering.
    void setFilteringEnabled(bool enabled);

    // getCurrentBuffer returns the buffer associated with the current image.
    // When outSlot is not nullptr, the current buffer slot index is also
    // returned.
    sp<GraphicBuffer> getCurrentBuffer(int* outSlot = nullptr) const;

    // getCurrentTextureTarget returns the texture target of the current
    // texture as returned by updateTexImage().
    uint32_t getCurrentTextureTarget() const;

    // getCurrentCrop returns the cropping rectangle of the current buffer.
    Rect getCurrentCrop() const;

    // getCurrentTransform returns the transform of the current buffer.
    uint32_t getCurrentTransform() const;

    // getCurrentScalingMode returns the scaling mode of the current buffer.
    uint32_t getCurrentScalingMode() const;

    // getCurrentFence returns the fence indicating when the current buffer is
    // ready to be read from.
    sp<Fence> getCurrentFence() const;

    // getCurrentFence returns the FenceTime indicating when the current
    // buffer is ready to be read from.
    std::shared_ptr<FenceTime> getCurrentFenceTime() const;

    // setConsumerUsageBits overrides the ConsumerBase method to OR
    // DEFAULT_USAGE_FLAGS to usage.
    status_t setConsumerUsageBits(uint64_t usage);

    // detachFromContext detaches the GLConsumer from the calling thread's
    // current OpenGL ES context.  This context must be the same as the context
    // that was current for previous calls to updateTexImage.
    //
    // Detaching a GLConsumer from an OpenGL ES context will result in the
    // deletion of the OpenGL ES texture object into which the images were being
    // streamed.  After a GLConsumer has been detached from the OpenGL ES
    // context calls to updateTexImage will fail returning INVALID_OPERATION
    // until the GLConsumer is attached to a new OpenGL ES context using the
    // attachToContext method.
    status_t detachFromContext();

    // attachToContext attaches a GLConsumer that is currently in the
    // 'detached' state to the current OpenGL ES context.  A GLConsumer is
    // in the 'detached' state iff detachFromContext has successfully been
    // called and no calls to attachToContext have succeeded since the last
    // detachFromContext call.  Calls to attachToContext made on a
    // GLConsumer that is not in the 'detached' state will result in an
    // INVALID_OPERATION error.
    //
    // The tex argument specifies the OpenGL ES texture object name in the
    // new context into which the image contents will be streamed.  A successful
    // call to attachToContext will result in this texture object being bound to
    // the texture target and populated with the image contents that were
    // current at the time of the last call to detachFromContext.
    status_t attachToContext(uint32_t tex);

protected:

    // abandonLocked overrides the ConsumerBase method to clear
    // mCurrentTextureImage in addition to the ConsumerBase behavior.
    virtual void abandonLocked();

    // dumpLocked overrides the ConsumerBase method to dump GLConsumer-
    // specific info in addition to the ConsumerBase behavior.
    virtual void dumpLocked(String8& result, const char* prefix) const;

    // acquireBufferLocked overrides the ConsumerBase method to update the
    // mEglSlots array in addition to the ConsumerBase behavior.
    virtual status_t acquireBufferLocked(BufferItem *item, nsecs_t presentWhen,
            uint64_t maxFrameNumber = 0) override;

    // releaseBufferLocked overrides the ConsumerBase method to update the
    // mEglSlots array in addition to the ConsumerBase.
    virtual status_t releaseBufferLocked(int slot,
            const sp<GraphicBuffer> graphicBuffer,
            EGLDisplay display, EGLSyncKHR eglFence) override;

    status_t releaseBufferLocked(int slot,
            const sp<GraphicBuffer> graphicBuffer, EGLSyncKHR eglFence) {
        return releaseBufferLocked(slot, graphicBuffer, mEglDisplay, eglFence);
    }

    struct PendingRelease {
        PendingRelease() : isPending(false), currentTexture(-1),
                graphicBuffer(), display(nullptr), fence(nullptr) {}

        bool isPending;
        int currentTexture;
        sp<GraphicBuffer> graphicBuffer;
        EGLDisplay display;
        EGLSyncKHR fence;
    };

    // This releases the buffer in the slot referenced by mCurrentTexture,
    // then updates state to refer to the BufferItem, which must be a
    // newly-acquired buffer. If pendingRelease is not null, the parameters
    // which would have been passed to releaseBufferLocked upon the successful
    // completion of the method will instead be returned to the caller, so that
    // it may call releaseBufferLocked itself later.
    status_t updateAndReleaseLocked(const BufferItem& item,
            PendingRelease* pendingRelease = nullptr);

    // Binds mTexName and the current buffer to mTexTarget.  Uses
    // mCurrentTexture if it's set, mCurrentTextureImage if not.  If the
    // bind succeeds, this calls doGLFenceWait.
    status_t bindTextureImageLocked();

    // Gets the current EGLDisplay and EGLContext values, and compares them
    // to mEglDisplay and mEglContext.  If the fields have been previously
    // set, the values must match; if not, the fields are set to the current
    // values.
    // The contextCheck argument is used to ensure that a GL context is
    // properly set; when set to false, the check is not performed.
    status_t checkAndUpdateEglStateLocked(bool contextCheck = false);

private:
    // EglImage is a utility class for tracking and creating EGLImageKHRs. There
    // is primarily just one image per slot, but there is also special cases:
    //  - For releaseTexImage, we use a debug image (mReleasedTexImage)
    //  - After freeBuffer, we must still keep the current image/buffer
    // Reference counting EGLImages lets us handle all these cases easily while
    // also only creating new EGLImages from buffers when required.
    class EglImage : public LightRefBase<EglImage>  {
    public:
        EglImage(sp<GraphicBuffer> graphicBuffer);

        // createIfNeeded creates an EGLImage if required (we haven't created
        // one yet, or the EGLDisplay or crop-rect has changed).
        status_t createIfNeeded(EGLDisplay display,
                                const Rect& cropRect,
                                bool forceCreate = false);

        // This calls glEGLImageTargetTexture2DOES to bind the image to the
        // texture in the specified texture target.
        void bindToTextureTarget(uint32_t texTarget);

        const sp<GraphicBuffer>& graphicBuffer() { return mGraphicBuffer; }
        const native_handle* graphicBufferHandle() {
            return mGraphicBuffer == NULL ? NULL : mGraphicBuffer->handle;
        }

    private:
        // Only allow instantiation using ref counting.
        friend class LightRefBase<EglImage>;
        virtual ~EglImage();

        // createImage creates a new EGLImage from a GraphicBuffer.
        EGLImageKHR createImage(EGLDisplay dpy,
                const sp<GraphicBuffer>& graphicBuffer, const Rect& crop);

        // Disallow copying
        EglImage(const EglImage& rhs);
        void operator = (const EglImage& rhs);

        // mGraphicBuffer is the buffer that was used to create this image.
        sp<GraphicBuffer> mGraphicBuffer;

        // mEglImage is the EGLImage created from mGraphicBuffer.
        EGLImageKHR mEglImage;

        // mEGLDisplay is the EGLDisplay that was used to create mEglImage.
        EGLDisplay mEglDisplay;

        // mCropRect is the crop rectangle passed to EGL when mEglImage
        // was created.
        Rect mCropRect;
    };

    // freeBufferLocked frees up the given buffer slot. If the slot has been
    // initialized this will release the reference to the GraphicBuffer in that
    // slot and destroy the EGLImage in that slot.  Otherwise it has no effect.
    //
    // This method must be called with mMutex locked.
    virtual void freeBufferLocked(int slotIndex);

    // computeCurrentTransformMatrixLocked computes the transform matrix for the
    // current texture.  It uses mCurrentTransform and the current GraphicBuffer
    // to compute this matrix and stores it in mCurrentTransformMatrix.
    // mCurrentTextureImage must not be NULL.
    void computeCurrentTransformMatrixLocked();

    // doGLFenceWaitLocked inserts a wait command into the OpenGL ES command
    // stream to ensure that it is safe for future OpenGL ES commands to
    // access the current texture buffer.
    status_t doGLFenceWaitLocked() const;

    // syncForReleaseLocked performs the synchronization needed to release the
    // current slot from an OpenGL ES context.  If needed it will set the
    // current slot's fence to guard against a producer accessing the buffer
    // before the outstanding accesses have completed.
    status_t syncForReleaseLocked(EGLDisplay dpy);

    // returns a graphic buffer used when the texture image has been released
    static sp<GraphicBuffer> getDebugTexImageBuffer();

    // The default consumer usage flags that GLConsumer always sets on its
    // BufferQueue instance; these will be OR:d with any additional flags passed
    // from the GLConsumer user. In particular, GLConsumer will always
    // consume buffers as hardware textures.
    static const uint64_t DEFAULT_USAGE_FLAGS = GraphicBuffer::USAGE_HW_TEXTURE;

    // mCurrentTextureImage is the EglImage/buffer of the current texture. It's
    // possible that this buffer is not associated with any buffer slot, so we
    // must track it separately in order to support the getCurrentBuffer method.
    sp<EglImage> mCurrentTextureImage;

    // mCurrentCrop is the crop rectangle that applies to the current texture.
    // It gets set each time updateTexImage is called.
    Rect mCurrentCrop;

    // mCurrentTransform is the transform identifier for the current texture. It
    // gets set each time updateTexImage is called.
    uint32_t mCurrentTransform;

    // mCurrentScalingMode is the scaling mode for the current texture. It gets
    // set each time updateTexImage is called.
    uint32_t mCurrentScalingMode;

    // mCurrentFence is the fence received from BufferQueue in updateTexImage.
    sp<Fence> mCurrentFence;

    // The FenceTime wrapper around mCurrentFence.
    std::shared_ptr<FenceTime> mCurrentFenceTime{FenceTime::NO_FENCE};

    // mCurrentTransformMatrix is the transform matrix for the current texture.
    // It gets computed by computeTransformMatrix each time updateTexImage is
    // called.
    float mCurrentTransformMatrix[16];

    // mCurrentTimestamp is the timestamp for the current texture. It
    // gets set each time updateTexImage is called.
    int64_t mCurrentTimestamp;

    // mCurrentDataSpace is the dataspace for the current texture. It
    // gets set each time updateTexImage is called.
    android_dataspace mCurrentDataSpace;

    // mCurrentFrameNumber is the frame counter for the current texture.
    // It gets set each time updateTexImage is called.
    uint64_t mCurrentFrameNumber;

    uint32_t mDefaultWidth, mDefaultHeight;

    // mFilteringEnabled indicates whether the transform matrix is computed for
    // use with bilinear filtering. It defaults to true and is changed by
    // setFilteringEnabled().
    bool mFilteringEnabled;

    // mTexName is the name of the OpenGL texture to which streamed images will
    // be bound when updateTexImage is called. It is set at construction time
    // and can be changed with a call to attachToContext.
    uint32_t mTexName;

    // mUseFenceSync indicates whether creation of the EGL_KHR_fence_sync
    // extension should be used to prevent buffers from being dequeued before
    // it's safe for them to be written. It gets set at construction time and
    // never changes.
    const bool mUseFenceSync;

    // mTexTarget is the GL texture target with which the GL texture object is
    // associated.  It is set in the constructor and never changed.  It is
    // almost always GL_TEXTURE_EXTERNAL_OES except for one use case in Android
    // Browser.  In that case it is set to GL_TEXTURE_2D to allow
    // glCopyTexSubImage to read from the texture.  This is a hack to work
    // around a GL driver limitation on the number of FBO attachments, which the
    // browser's tile cache exceeds.
    const uint32_t mTexTarget;

    // EGLSlot contains the information and object references that
    // GLConsumer maintains about a BufferQueue buffer slot.
    struct EglSlot {
        EglSlot() : mEglFence(EGL_NO_SYNC_KHR) {}

        // mEglImage is the EGLImage created from mGraphicBuffer.
        sp<EglImage> mEglImage;

        // mFence is the EGL sync object that must signal before the buffer
        // associated with this buffer slot may be dequeued. It is initialized
        // to EGL_NO_SYNC_KHR when the buffer is created and (optionally, based
        // on a compile-time option) set to a new sync object in updateTexImage.
        EGLSyncKHR mEglFence;
    };

    // mEglDisplay is the EGLDisplay with which this GLConsumer is currently
    // associated.  It is intialized to EGL_NO_DISPLAY and gets set to the
    // current display when updateTexImage is called for the first time and when
    // attachToContext is called.
    EGLDisplay mEglDisplay;

    // mEglContext is the OpenGL ES context with which this GLConsumer is
    // currently associated.  It is initialized to EGL_NO_CONTEXT and gets set
    // to the current GL context when updateTexImage is called for the first
    // time and when attachToContext is called.
    EGLContext mEglContext;

    // mEGLSlots stores the buffers that have been allocated by the BufferQueue
    // for each buffer slot.  It is initialized to null pointers, and gets
    // filled in with the result of BufferQueue::acquire when the
    // client dequeues a buffer from a
    // slot that has not yet been used. The buffer allocated to a slot will also
    // be replaced if the requested buffer usage or geometry differs from that
    // of the buffer allocated to a slot.
    EglSlot mEglSlots[BufferQueueDefs::NUM_BUFFER_SLOTS];

    // mCurrentTexture is the buffer slot index of the buffer that is currently
    // bound to the OpenGL texture. It is initialized to INVALID_BUFFER_SLOT,
    // indicating that no buffer slot is currently bound to the texture. Note,
    // however, that a value of INVALID_BUFFER_SLOT does not necessarily mean
    // that no buffer is bound to the texture. A call to setBufferCount will
    // reset mCurrentTexture to INVALID_BUFFER_SLOT.
    int mCurrentTexture;

    // mAttached indicates whether the ConsumerBase is currently attached to
    // an OpenGL ES context.  For legacy reasons, this is initialized to true,
    // indicating that the ConsumerBase is considered to be attached to
    // whatever context is current at the time of the first updateTexImage call.
    // It is set to false by detachFromContext, and then set to true again by
    // attachToContext.
    bool mAttached;

    // protects static initialization
    static Mutex sStaticInitLock;

    // mReleasedTexImageBuffer is a dummy buffer used when in single buffer
    // mode and releaseTexImage() has been called
    static sp<GraphicBuffer> sReleasedTexImageBuffer;
    sp<EglImage> mReleasedTexImage;
};

// ----------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_GUI_CONSUMER_H
