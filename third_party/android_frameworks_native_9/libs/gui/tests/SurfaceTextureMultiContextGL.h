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

#ifndef ANDROID_SURFACE_TEXTURE_MULTI_CONTEXT_GL_H
#define ANDROID_SURFACE_TEXTURE_MULTI_CONTEXT_GL_H

#include "SurfaceTextureGL.h"

namespace android {

class SurfaceTextureMultiContextGLTest : public SurfaceTextureGLTest {
protected:
    enum { SECOND_TEX_ID = 123 };
    enum { THIRD_TEX_ID = 456 };

    SurfaceTextureMultiContextGLTest():
            mSecondEglContext(EGL_NO_CONTEXT),
            mThirdEglContext(EGL_NO_CONTEXT) {
    }

    virtual void SetUp() {
        SurfaceTextureGLTest::SetUp();

        // Set up the secondary context and texture renderer.
        mSecondEglContext = eglCreateContext(mEglDisplay, mGlConfig,
                EGL_NO_CONTEXT, getContextAttribs());
        ASSERT_EQ(EGL_SUCCESS, eglGetError());
        ASSERT_NE(EGL_NO_CONTEXT, mSecondEglContext);

        ASSERT_TRUE(eglMakeCurrent(mEglDisplay, mEglSurface, mEglSurface,
                mSecondEglContext));
        ASSERT_EQ(EGL_SUCCESS, eglGetError());
        mSecondTextureRenderer = new TextureRenderer(SECOND_TEX_ID, mST);
        ASSERT_NO_FATAL_FAILURE(mSecondTextureRenderer->SetUp());

        // Set up the tertiary context and texture renderer.
        mThirdEglContext = eglCreateContext(mEglDisplay, mGlConfig,
                EGL_NO_CONTEXT, getContextAttribs());
        ASSERT_EQ(EGL_SUCCESS, eglGetError());
        ASSERT_NE(EGL_NO_CONTEXT, mThirdEglContext);

        ASSERT_TRUE(eglMakeCurrent(mEglDisplay, mEglSurface, mEglSurface,
                mThirdEglContext));
        ASSERT_EQ(EGL_SUCCESS, eglGetError());
        mThirdTextureRenderer = new TextureRenderer(THIRD_TEX_ID, mST);
        ASSERT_NO_FATAL_FAILURE(mThirdTextureRenderer->SetUp());

        // Switch back to the primary context to start the tests.
        ASSERT_TRUE(eglMakeCurrent(mEglDisplay, mEglSurface, mEglSurface,
                mEglContext));
    }

    virtual void TearDown() {
        if (mThirdEglContext != EGL_NO_CONTEXT) {
            eglDestroyContext(mEglDisplay, mThirdEglContext);
        }
        if (mSecondEglContext != EGL_NO_CONTEXT) {
            eglDestroyContext(mEglDisplay, mSecondEglContext);
        }
        SurfaceTextureGLTest::TearDown();
    }

    EGLContext mSecondEglContext;
    sp<TextureRenderer> mSecondTextureRenderer;

    EGLContext mThirdEglContext;
    sp<TextureRenderer> mThirdTextureRenderer;
};

}

#endif
