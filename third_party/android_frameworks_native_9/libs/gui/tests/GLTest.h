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

#ifndef ANDROID_GL_TEST_H
#define ANDROID_GL_TEST_H

#include <gtest/gtest.h>

#include <gui/SurfaceComposerClient.h>

#include <EGL/egl.h>
#include <GLES/gl.h>

namespace android {

class GLTest : public ::testing::Test {
public:
    static void loadShader(GLenum shaderType, const char* pSource,
            GLuint* outShader);
    static void createProgram(const char* pVertexSource,
            const char* pFragmentSource, GLuint* outPgm);

protected:
    GLTest() :
            mDisplaySecs(0),
            mEglDisplay(EGL_NO_DISPLAY),
            mEglSurface(EGL_NO_SURFACE),
            mEglContext(EGL_NO_CONTEXT),
            mGlConfig(NULL) {
    }

    virtual void SetUp();
    virtual void TearDown();

    virtual EGLint const* getConfigAttribs();
    virtual EGLint const* getContextAttribs();
    virtual EGLint getSurfaceWidth();
    virtual EGLint getSurfaceHeight();
    virtual EGLSurface createWindowSurface(EGLDisplay display, EGLConfig config,
                                           sp<ANativeWindow>& window) const;

    ::testing::AssertionResult checkPixel(int x, int y,
            int r, int g, int b, int a, int tolerance = 2);
    ::testing::AssertionResult assertRectEq(const Rect &r1, const Rect &r2,
            int tolerance = 1);

    int mDisplaySecs;
    sp<SurfaceComposerClient> mComposerClient;
    sp<SurfaceControl> mSurfaceControl;

    EGLDisplay mEglDisplay;
    EGLSurface mEglSurface;
    EGLContext mEglContext;
    EGLConfig  mGlConfig;
};

} // namespace android

#endif
