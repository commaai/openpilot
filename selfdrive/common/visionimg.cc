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

#endif

#include "common/util.h"
#include "common/visionbuf.h"

#include "common/visionimg.h"

#ifdef QCOM

using namespace android;

// from libadreno_utils.so
extern "C" void compute_aligned_width_and_height(int width,
                                      int height,
                                      int bpp,
                                      int tile_mode,
                                      int raster_mode,
                                      int padding_threshold,
                                      int *aligned_w,
                                      int *aligned_h);
#endif

void visionimg_compute_aligned_width_and_height(int width, int height, int *aligned_w, int *aligned_h) {
#if defined(QCOM) && !defined(QCOM_REPLAY)
  compute_aligned_width_and_height(ALIGN(width, 32), ALIGN(height, 32), 3, 0, 0, 512, aligned_w, aligned_h);
#else
  *aligned_w = width; *aligned_h = height;
#endif
}

VisionImg visionimg_alloc_rgb24(int width, int height, VisionBuf *out_buf) {
  int aligned_w = 0, aligned_h = 0;
  visionimg_compute_aligned_width_and_height(width, height, &aligned_w, &aligned_h);

  int stride = aligned_w * 3;
  size_t size = (size_t) aligned_w * aligned_h * 3;

  VisionBuf buf = visionbuf_allocate(size);

  *out_buf = buf;

  return (VisionImg){
    .fd = buf.fd,
    .format = VISIONIMG_FORMAT_RGB24,
    .width = width,
    .height = height,
    .stride = stride,
    .size = size,
    .bpp = 3,
  };
}

void EGLTexture::init(const VisionImg &img, void *addr) {
  assert((img.size % img.stride) == 0);
  assert((img.stride % img.bpp) == 0);
#ifdef QCOM
  int format = HAL_PIXEL_FORMAT_RGB_888;
  private_handle = new private_handle_t(
      img.fd, img.size,
      private_handle_t::PRIV_FLAGS_USES_ION | private_handle_t::PRIV_FLAGS_FRAMEBUFFER,
      0, format,
      img.stride / img.bpp, img.size / img.stride,
      img.width, img.height);

  GraphicBuffer *buf = new GraphicBuffer(
      img.width, img.height, (PixelFormat)format,
      GraphicBuffer::USAGE_HW_TEXTURE, img.stride / img.bpp,
      (private_handle_t *)private_handle, false);

  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  assert(display != EGL_NO_DISPLAY);

  EGLint img_attrs[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
  img_khr = eglCreateImageKHR(
      display, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID,
      (EGLClientBuffer)buf->getNativeBuffer(), img_attrs);
  assert(img_khr != EGL_NO_IMAGE_KHR);

  glGenTextures(1, &frame_tex);
  glBindTexture(GL_TEXTURE_2D, frame_tex);
  glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img_khr);
#else
  glGenTextures(1, &frame_tex);
  glBindTexture(GL_TEXTURE_2D, frame_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, addr);
  glGenerateMipmap(GL_TEXTURE_2D);
#endif
  glBindTexture(GL_TEXTURE_2D, frame_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  // BGR
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
}

void EGLTexture::destroy() {
  if (frame_tex != 0) {
    glDeleteTextures(1, &frame_tex);
    frame_tex = 0;
  }
#ifdef QCOM
  if (img_khr != 0) {
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    assert(display != EGL_NO_DISPLAY);
    eglDestroyImageKHR(display, img_khr);
    img_khr = 0;
  }
  if (private_handle) {
    delete (private_handle_t *)private_handle;
    private_handle = nullptr;
  }
#endif
}
