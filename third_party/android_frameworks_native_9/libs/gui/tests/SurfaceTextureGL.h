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

#ifndef ANDROID_SURFACE_TEXTURE_GL_H
#define ANDROID_SURFACE_TEXTURE_GL_H

#include "GLTest.h"

#include "FrameWaiter.h"
#include "TextureRenderer.h"

#include <gui/GLConsumer.h>
#include <gui/Surface.h>

namespace android {

class FrameWaiter;
class GLConsumer;
class TextureRenderer;

class SurfaceTextureGLTest : public GLTest {
protected:
    enum { TEX_ID = 123 };

    void SetUp() {
        GLTest::SetUp();
        sp<IGraphicBufferProducer> producer;
        BufferQueue::createBufferQueue(&producer, &mConsumer);
        mST = new GLConsumer(mConsumer, TEX_ID, GLConsumer::TEXTURE_EXTERNAL,
                true, false);
        mSTC = new Surface(producer);
        mANW = mSTC;
        mTextureRenderer = new TextureRenderer(TEX_ID, mST);
        ASSERT_NO_FATAL_FAILURE(mTextureRenderer->SetUp());
        mFW = new FrameWaiter;
        mST->setFrameAvailableListener(mFW);
    }

    void TearDown() {
        mTextureRenderer.clear();
        mANW.clear();
        mSTC.clear();
        mST.clear();
        GLTest::TearDown();
    }

    void drawTexture() {
        mTextureRenderer->drawTexture();
    }

    sp<IGraphicBufferConsumer> mConsumer;
    sp<GLConsumer> mST;
    sp<Surface> mSTC;
    sp<ANativeWindow> mANW;
    sp<TextureRenderer> mTextureRenderer;
    sp<FrameWaiter> mFW;
};

} // namespace android

#endif
