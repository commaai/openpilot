#pragma once
#include "visionbuf.h"
#include "common/glutil.h"

#ifdef QCOM
#include <EGL/egl.h>
#include <EGL/eglext.h>
#undef Status
#else
typedef int EGLImageKHR;
typedef void *EGLClientBuffer;
#endif

#define VISIONIMG_FORMAT_RGB24 1

typedef struct VisionImg {
  int fd;
  int format;
  int width, height, stride;
  int bpp;
  size_t size;
} VisionImg;

GLuint visionimg_to_gl(const VisionImg *img, EGLImageKHR *pkhr, void **pph);
void visionimg_destroy_gl(EGLImageKHR khr, void *ph);
