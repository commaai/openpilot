#pragma once

#include "visionbuf.h"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GLES3/gl3.h>
#endif

#if defined(QCOM) || defined(QCOM2)
#include <EGL/egl.h>
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/eglext.h>
#undef Bool
#undef CursorShape
#undef Expose
#undef KeyPress
#undef KeyRelease
#undef FocusIn
#undef FocusOut
#undef FontChange
#undef None
#undef Status
#undef Unsorted
#endif

class EGLImageTexture {
 public:
  EGLImageTexture(const VisionBuf *buf);
  ~EGLImageTexture();
  GLuint frame_tex = 0;
#ifdef QCOM
  void *private_handle = nullptr;
#endif
  EGLImageKHR img_khr = 0;
};
