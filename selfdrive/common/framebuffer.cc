#include "selfdrive/common/framebuffer.h"

#include <cstdio>
#include <cassert>

#include "selfdrive/common/util.h"

#include <ui/DisplayInfo.h>

#include <gui/ISurfaceComposer.h>
#include <gui/Surface.h>
#include <gui/SurfaceComposerClient.h>
#include <GLES2/gl2.h>
#include <EGL/eglext.h>

#define BACKLIGHT_LEVEL 205

using namespace android;

struct FramebufferState {
    sp<SurfaceComposerClient> session;
    sp<IBinder> dtoken;
    DisplayInfo dinfo;
    sp<SurfaceControl> control;

    sp<Surface> s;
    EGLDisplay display;

    EGLint egl_major, egl_minor;
    EGLConfig config;
    EGLSurface surface;
    EGLContext context;
};

void FrameBuffer::swap() {
  eglSwapBuffers(s->display, s->surface);
  assert(glGetError() == GL_NO_ERROR);
}

bool set_brightness(int brightness) {
  char bright[64];
  snprintf(bright, sizeof(bright), "%d", brightness);
  return 0 == util::write_file("/sys/class/leds/lcd-backlight/brightness", bright, strlen(bright));
}

void FrameBuffer::set_power(int mode) {
  SurfaceComposerClient::setDisplayPowerMode(s->dtoken, mode);
}

FrameBuffer::FrameBuffer(const char *name, uint32_t layer, int alpha, int *out_w, int *out_h) {
  s = new FramebufferState;

  s->session = new SurfaceComposerClient();
  assert(s->session != NULL);

  s->dtoken = SurfaceComposerClient::getBuiltInDisplay(
                ISurfaceComposer::eDisplayIdMain);
  assert(s->dtoken != NULL);

  status_t status = SurfaceComposerClient::getDisplayInfo(s->dtoken, &s->dinfo);
  assert(status == 0);

  //int orientation = 3; // rotate framebuffer 270 degrees
  int orientation = 1; // rotate framebuffer 90 degrees
  if(orientation == 1 || orientation == 3) {
      int temp = s->dinfo.h;
      s->dinfo.h = s->dinfo.w;
      s->dinfo.w = temp;
  }

  printf("dinfo %dx%d\n", s->dinfo.w, s->dinfo.h);

  Rect destRect(s->dinfo.w, s->dinfo.h);
  s->session->setDisplayProjection(s->dtoken, orientation, destRect, destRect);

  s->control = s->session->createSurface(String8(name),
                  s->dinfo.w, s->dinfo.h, PIXEL_FORMAT_RGBX_8888);
  assert(s->control != NULL);

  SurfaceComposerClient::openGlobalTransaction();
  status = s->control->setLayer(layer);
  SurfaceComposerClient::closeGlobalTransaction();
  assert(status == 0);

  s->s = s->control->getSurface();
  assert(s->s != NULL);

  // init opengl and egl
  const EGLint attribs[] = {
    EGL_RED_SIZE,     8,
    EGL_GREEN_SIZE,   8,
    EGL_BLUE_SIZE,    8,
    EGL_ALPHA_SIZE,   alpha ? 8 : 0,
    EGL_DEPTH_SIZE,   0,
    EGL_STENCIL_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
    // enable MSAA
    EGL_SAMPLE_BUFFERS, 1,
    EGL_SAMPLES, 4,
    EGL_NONE,
  };

  s->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  assert(s->display != EGL_NO_DISPLAY);

  int success = eglInitialize(s->display, &s->egl_major, &s->egl_minor);
  assert(success);

  printf("egl version %d.%d\n", s->egl_major, s->egl_minor);

  EGLint num_configs;
  success = eglChooseConfig(s->display, attribs, &s->config, 1, &num_configs);
  assert(success);

  s->surface = eglCreateWindowSurface(s->display, s->config, s->s.get(), NULL);
  assert(s->surface != EGL_NO_SURFACE);

  const EGLint context_attribs[] = {
    EGL_CONTEXT_CLIENT_VERSION, 3,
    EGL_NONE,
  };
  s->context = eglCreateContext(s->display, s->config, NULL, context_attribs);
  assert(s->context != EGL_NO_CONTEXT);

  EGLint w, h;
  eglQuerySurface(s->display, s->surface, EGL_WIDTH, &w);
  eglQuerySurface(s->display, s->surface, EGL_HEIGHT, &h);
  printf("egl w %d h %d\n", w, h);

  success = eglMakeCurrent(s->display, s->surface, s->surface, s->context);
  assert(success);

  printf("gl version %s\n", glGetString(GL_VERSION));

  set_brightness(BACKLIGHT_LEVEL);

  if (out_w) *out_w = w;
  if (out_h) *out_h = h;
}

FrameBuffer::~FrameBuffer() {
  eglDestroyContext(s->display, s->context);
  eglDestroySurface(s->display, s->surface);
  eglTerminate(s->display);
  delete s;
}
