#pragma once
#include "common/visionbuf.h"
#include "common/glutil.h"

#ifdef QCOM
#include <EGL/egl.h>
#include <EGL/eglext.h>
#undef Status
#endif

#define VISIONIMG_FORMAT_RGB24 1

typedef struct VisionImg {
  int fd;
  int format;
  int width, height, stride;
  int bpp;
  size_t size;
} VisionImg;

void visionimg_compute_aligned_width_and_height(int width, int height, int *aligned_w, int *aligned_h);
VisionImg visionimg_alloc_rgb24(int width, int height, VisionBuf *out_buf);

class EGLImageTexture {
 public:
  EGLImageTexture(const VisionImg &img, void *addr);
  ~EGLImageTexture();
  GLuint frame_tex = 0;
private:
 static void bindTexture(GLuint tex) {
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

   // BGR
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
 }
#ifdef QCOM
  void *private_handle = nullptr;
  EGLImageKHR img_khr = 0;
#endif
};
