#ifndef VISIONIMG_H
#define VISIONIMG_H

#include "common/visionbuf.h"

#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#undef Status

#ifdef __cplusplus
extern "C" {
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

EGLClientBuffer visionimg_to_egl(const VisionImg *img, void **pph);
GLuint visionimg_to_gl(const VisionImg *img, EGLImageKHR *pkhr, void **pph);
void visionimg_destroy_gl(EGLImageKHR khr, void *ph);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
