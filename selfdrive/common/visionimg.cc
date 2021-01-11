#include <cassert>

#ifdef QCOM

#include <system/graphics.h>
#include <ui/GraphicBuffer.h>
#include <ui/PixelFormat.h>
#include <gralloc_priv.h>

#include <GLES3/gl3.h>
#define GL_GLEXT_PROTOTYPES
#include <GLES2/gl2ext.h>

#include <EGL/egl.h>
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/eglext.h>

#else // ifdef QCOM

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GLES3/gl3.h>
#endif

#endif // ifdef QCOM

#include "common/visionimg.h"

#ifdef QCOM
using namespace android;

static EGLClientBuffer visionimg_to_egl(const VisionImg *img, void **pph) {
  assert((img->size % img->stride) == 0);
  assert((img->stride % img->bpp) == 0);

  int format = 0;
  if (img->format == VISIONIMG_FORMAT_RGB24) {
    format = HAL_PIXEL_FORMAT_RGB_888;
  } else {
    assert(false);
  }

  private_handle_t* hnd = new private_handle_t(img->fd, img->size,
                             private_handle_t::PRIV_FLAGS_USES_ION|private_handle_t::PRIV_FLAGS_FRAMEBUFFER,
                             0, format,
                             img->stride/img->bpp, img->size/img->stride,
                             img->width, img->height);

  GraphicBuffer* gb = new GraphicBuffer(img->width, img->height, (PixelFormat)format,
                                        GraphicBuffer::USAGE_HW_TEXTURE, img->stride/img->bpp, hnd, false);
  // GraphicBuffer is ref counted by EGLClientBuffer(ANativeWindowBuffer), no need and not possible to release.
  *pph = hnd;
  return (EGLClientBuffer) gb->getNativeBuffer();
}

GLuint visionimg_to_gl(const VisionImg *img, EGLImageKHR *pkhr, void **pph) {

  EGLClientBuffer buf = visionimg_to_egl(img, pph);

  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  assert(display != EGL_NO_DISPLAY);

  EGLint img_attrs[] = { EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE };
  EGLImageKHR image = eglCreateImageKHR(display, EGL_NO_CONTEXT,
                                        EGL_NATIVE_BUFFER_ANDROID, buf, img_attrs);
  assert(image != EGL_NO_IMAGE_KHR);

  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, image);
  *pkhr = image;
  return tex;
}

void visionimg_destroy_gl(EGLImageKHR khr, void *ph) {
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  assert(display != EGL_NO_DISPLAY);
  eglDestroyImageKHR(display, khr);
  delete (private_handle_t*)ph;
}

#else // ifdef QCOM

GLuint visionimg_to_gl(const VisionImg *img, EGLImageKHR *pkhr, void **pph) {
  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img->width, img->height, 0, GL_RGB, GL_UNSIGNED_BYTE, *pph);
  glGenerateMipmap(GL_TEXTURE_2D);
  *pkhr = (EGLImageKHR)1; // not NULL
  return texture;
}

void visionimg_destroy_gl(EGLImageKHR khr, void *ph) {
  // empty
}
#endif // ifdef QCOM
