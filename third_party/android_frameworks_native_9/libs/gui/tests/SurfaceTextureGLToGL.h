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

#ifndef ANDROID_SURFACE_TEXTURE_GL_TO_GL_H
#define ANDROID_SURFACE_TEXTURE_GL_TO_GL_H

#include "SurfaceTextureGL.h"

namespace android {

/*
 * This test fixture is for testing GL -> GL texture streaming.  It creates an
 * EGLSurface and an EGLContext for the image producer to use.
 */
class SurfaceTextureGLToGLTest : public SurfaceTextureGLTest {
protected:
    SurfaceTextureGLToGLTest():
            mProducerEglSurface(EGL_NO_SURFACE),
            mProducerEglContext(EGL_NO_CONTEXT) {
    }

    virtual void SetUp() {
        SurfaceTextureGLTest::SetUp();
    }

    void SetUpWindowAndContext() {
        mProducerEglSurface = eglCreateWindowSurface(mEglDisplay, mGlConfig,
                mANW.get(), NULL);
        ASSERT_EQ(EGL_SUCCESS, eglGetError());
        ASSERT_NE(EGL_NO_SURFACE, mProducerEglSurface);

        mProducerEglContext = eglCreateContext(mEglDisplay, mGlConfig,
                EGL_NO_CONTEXT, getContextAttribs());
        ASSERT_EQ(EGL_SUCCESS, eglGetError());
        ASSERT_NE(EGL_NO_CONTEXT, mProducerEglContext);
    }

    virtual void TearDown() {
        if (mProducerEglContext != EGL_NO_CONTEXT) {
            eglDestroyContext(mEglDisplay, mProducerEglContext);
        }
        if (mProducerEglSurface != EGL_NO_SURFACE) {
            eglDestroySurface(mEglDisplay, mProducerEglSurface);
        }
        SurfaceTextureGLTest::TearDown();
    }

    EGLSurface mProducerEglSurface;
    EGLContext mProducerEglContext;
};

} // namespace android

#endif
