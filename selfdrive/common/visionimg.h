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

#ifdef QCOM
 private:
  void *private_handle = nullptr;
  EGLImageKHR img_khr = 0;
#endif
};
