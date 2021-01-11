#pragma once
#include "visionbuf.h"
#include "common/glutil.h"

#ifdef QCOM
#include <EGL/egl.h>
#include <EGL/eglext.h>
#undef Status
#endif

class EGLImageTexture {
 public:
  EGLImageTexture(const VisionBuf *buf);
  ~EGLImageTexture();
  GLuint frame_tex = 0;
#ifdef QCOM
  void *private_handle = nullptr;
  EGLImageKHR img_khr = 0;
#endif
};
