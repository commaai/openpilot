#include "selfdrive/common/visionimg.h"

#include <cassert>

EGLImageTexture::EGLImageTexture(const VisionBuf *buf) {
  glGenTextures(1, &frame_tex);
  glBindTexture(GL_TEXTURE_2D, frame_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buf->width, buf->height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
  glGenerateMipmap(GL_TEXTURE_2D);
}

EGLImageTexture::~EGLImageTexture() {
  glDeleteTextures(1, &frame_tex);
}
