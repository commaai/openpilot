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

#ifndef ANDROID_SURFACE_TEXTURE_FBO_H
#define ANDROID_SURFACE_TEXTURE_FBO_H

#include "SurfaceTextureGL.h"

#include <GLES2/gl2.h>

namespace android {

class SurfaceTextureFBOTest : public SurfaceTextureGLTest {
protected:
    virtual void SetUp() {
        SurfaceTextureGLTest::SetUp();

        glGenFramebuffers(1, &mFbo);
        ASSERT_EQ(GLenum(GL_NO_ERROR), glGetError());

        glGenTextures(1, &mFboTex);
        glBindTexture(GL_TEXTURE_2D, mFboTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, getSurfaceWidth(),
                getSurfaceHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        ASSERT_EQ(GLenum(GL_NO_ERROR), glGetError());

        glBindFramebuffer(GL_FRAMEBUFFER, mFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D, mFboTex, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        ASSERT_EQ(GLenum(GL_NO_ERROR), glGetError());
    }

    virtual void TearDown() {
        SurfaceTextureGLTest::TearDown();

        glDeleteTextures(1, &mFboTex);
        glDeleteFramebuffers(1, &mFbo);
    }

    GLuint mFbo;
    GLuint mFboTex;
};

void fillRGBA8BufferSolid(uint8_t* buf, int w, int h, int stride,
        uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    const size_t PIXEL_SIZE = 4;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            off_t offset = (y * stride + x) * PIXEL_SIZE;
            buf[offset + 0] = r;
            buf[offset + 1] = g;
            buf[offset + 2] = b;
            buf[offset + 3] = a;
        }
    }
}

} // namespace android

#endif
