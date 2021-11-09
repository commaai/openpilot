#include "selfdrive/common/visionimg.h"

#include <cassert>

#ifdef QCOM
#include <gralloc_priv.h>
#include <system/graphics.h>
#include <ui/GraphicBuffer.h>
#include <ui/PixelFormat.h>
#define GL_GLEXT_PROTOTYPES
#include <GLES2/gl2ext.h>
using namespace android;

EGLImageTexture::EGLImageTexture(const VisionBuf *buf) {
  const int bpp = 3;
  assert((buf->len % buf->stride) == 0);
  assert((buf->stride % bpp) == 0);

  const int format = HAL_PIXEL_FORMAT_RGB_888;
  private_handle = new private_handle_t(buf->fd, buf->len,
                             private_handle_t::PRIV_FLAGS_USES_ION|private_handle_t::PRIV_FLAGS_FRAMEBUFFER,
                             0, format,
                             buf->stride/bpp, buf->len/buf->stride,
                             buf->width, buf->height);

  // GraphicBuffer is ref counted by EGLClientBuffer(ANativeWindowBuffer), no need and not possible to release.	
  GraphicBuffer* gb = new GraphicBuffer(buf->width, buf->height, (PixelFormat)format,
                                        GraphicBuffer::USAGE_HW_TEXTURE, buf->stride/bpp, (private_handle_t*)private_handle, false);

  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  assert(display != EGL_NO_DISPLAY);

  EGLint img_attrs[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
  img_khr = eglCreateImageKHR(display, EGL_NO_CONTEXT,
                              EGL_NATIVE_BUFFER_ANDROID, gb->getNativeBuffer(), img_attrs);
  assert(img_khr != EGL_NO_IMAGE_KHR);

  glGenTextures(1, &frame_tex);
  glBindTexture(GL_TEXTURE_2D, frame_tex);
  glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img_khr);
}

EGLImageTexture::~EGLImageTexture() {
  glDeleteTextures(1, &frame_tex);
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  assert(display != EGL_NO_DISPLAY);
  eglDestroyImageKHR(display, img_khr);
  delete (private_handle_t*)private_handle;
}

#else // ifdef QCOM

EGLImageTexture::EGLImageTexture(const VisionBuf *buf) {
  glGenTextures(1, &frame_tex);
  glBindTexture(GL_TEXTURE_2D, frame_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buf->width, buf->height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
  glGenerateMipmap(GL_TEXTURE_2D);

  glGenBuffers(1, &frame_buf);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, frame_buf);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, buf->len, nullptr, GL_DYNAMIC_DRAW);
  buffer = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, buf->len, GL_MAP_WRITE_BIT);
  assert(buffer);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

EGLImageTexture::~EGLImageTexture() {
  glDeleteTextures(1, &frame_tex);
  glDeleteBuffers(1, &frame_buf);
}
#endif // ifdef QCOM
