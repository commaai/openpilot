/*
 * Copyright 2013 The Android Open Source Project
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

#ifndef ANDROID_SURFACE_TEXTURE_GL_THREAD_TO_GL_H
#define ANDROID_SURFACE_TEXTURE_GL_THREAD_TO_GL_H

#include "SurfaceTextureGLToGL.h"

namespace android {

/*
 * This test fixture is for testing GL -> GL texture streaming from one thread
 * to another.  It contains functionality to create a producer thread that will
 * perform GL rendering to an ANativeWindow that feeds frames to a
 * GLConsumer.  Additionally it supports interlocking the producer and
 * consumer threads so that a specific sequence of calls can be
 * deterministically created by the test.
 *
 * The intended usage is as follows:
 *
 * TEST_F(...) {
 *     class PT : public ProducerThread {
 *         virtual void render() {
 *             ...
 *             swapBuffers();
 *         }
 *     };
 *
 *     runProducerThread(new PT());
 *
 *     // The order of these calls will vary from test to test and may include
 *     // multiple frames and additional operations (e.g. GL rendering from the
 *     // texture).
 *     fc->waitForFrame();
 *     mST->updateTexImage();
 *     fc->finishFrame();
 * }
 *
 */
class SurfaceTextureGLThreadToGLTest : public SurfaceTextureGLToGLTest {
protected:

    // ProducerThread is an abstract base class to simplify the creation of
    // OpenGL ES frame producer threads.
    class ProducerThread : public Thread {
    public:
        virtual ~ProducerThread() {
        }

        void setEglObjects(EGLDisplay producerEglDisplay,
                EGLSurface producerEglSurface,
                EGLContext producerEglContext) {
            mProducerEglDisplay = producerEglDisplay;
            mProducerEglSurface = producerEglSurface;
            mProducerEglContext = producerEglContext;
        }

        virtual bool threadLoop() {
            eglMakeCurrent(mProducerEglDisplay, mProducerEglSurface,
                    mProducerEglSurface, mProducerEglContext);
            render();
            eglMakeCurrent(mProducerEglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE,
                    EGL_NO_CONTEXT);
            return false;
        }

    protected:
        virtual void render() = 0;

        void swapBuffers() {
            eglSwapBuffers(mProducerEglDisplay, mProducerEglSurface);
        }

        EGLDisplay mProducerEglDisplay;
        EGLSurface mProducerEglSurface;
        EGLContext mProducerEglContext;
    };

    // FrameCondition is a utility class for interlocking between the producer
    // and consumer threads.  The FrameCondition object should be created and
    // destroyed in the consumer thread only.  The consumer thread should set
    // the FrameCondition as the FrameAvailableListener of the GLConsumer,
    // and should call both waitForFrame and finishFrame once for each expected
    // frame.
    //
    // This interlocking relies on the fact that onFrameAvailable gets called
    // synchronously from GLConsumer::queueBuffer.
    class FrameCondition : public GLConsumer::FrameAvailableListener {
    public:
        FrameCondition():
                mFrameAvailable(false),
                mFrameFinished(false) {
        }

        // waitForFrame waits for the next frame to arrive.  This should be
        // called from the consumer thread once for every frame expected by the
        // test.
        void waitForFrame() {
            Mutex::Autolock lock(mMutex);
            ALOGV("+waitForFrame");
            while (!mFrameAvailable) {
                mFrameAvailableCondition.wait(mMutex);
            }
            mFrameAvailable = false;
            ALOGV("-waitForFrame");
        }

        // Allow the producer to return from its swapBuffers call and continue
        // on to produce the next frame.  This should be called by the consumer
        // thread once for every frame expected by the test.
        void finishFrame() {
            Mutex::Autolock lock(mMutex);
            ALOGV("+finishFrame");
            mFrameFinished = true;
            mFrameFinishCondition.signal();
            ALOGV("-finishFrame");
        }

        // This should be called by GLConsumer on the producer thread.
        virtual void onFrameAvailable(const BufferItem& /* item */) {
            Mutex::Autolock lock(mMutex);
            ALOGV("+onFrameAvailable");
            mFrameAvailable = true;
            mFrameAvailableCondition.signal();
            while (!mFrameFinished) {
                mFrameFinishCondition.wait(mMutex);
            }
            mFrameFinished = false;
            ALOGV("-onFrameAvailable");
        }

    protected:
        bool mFrameAvailable;
        bool mFrameFinished;

        Mutex mMutex;
        Condition mFrameAvailableCondition;
        Condition mFrameFinishCondition;
    };

    virtual void SetUp() {
        SurfaceTextureGLToGLTest::SetUp();
        mFC = new FrameCondition();
        mST->setFrameAvailableListener(mFC);
    }

    virtual void TearDown() {
        if (mProducerThread != NULL) {
            mProducerThread->requestExitAndWait();
        }
        mProducerThread.clear();
        mFC.clear();
        SurfaceTextureGLToGLTest::TearDown();
    }

    void runProducerThread(const sp<ProducerThread> producerThread) {
        ASSERT_TRUE(mProducerThread == NULL);
        mProducerThread = producerThread;
        producerThread->setEglObjects(mEglDisplay, mProducerEglSurface,
                mProducerEglContext);
        producerThread->run("ProducerThread");
    }

    sp<ProducerThread> mProducerThread;
    sp<FrameCondition> mFC;
};

} // namespace android

#endif
