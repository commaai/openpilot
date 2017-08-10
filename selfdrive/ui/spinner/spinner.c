#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

#include <GLES3/gl3.h>
#include <EGL/eglext.h>

#include "nanovg.h"
#define NANOVG_GLES3_IMPLEMENTATION
#include "nanovg_gl.h"
#include "nanovg_gl_utils.h"

#include "common/framebuffer.h"


int main(int argc, char** argv) {
  int err;

  const char* spintext = NULL;
  if (argc >= 2) {
    spintext = argv[1];
  }

  // spinner
  int fb_w, fb_h;
  EGLDisplay display;
  EGLSurface surface;
  FramebufferState *fb = framebuffer_init("spinner", 0x00001000, false,
                     &display, &surface, &fb_w, &fb_h);
  assert(fb);

  NVGcontext *vg = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
  assert(vg);

  int font = nvgCreateFont(vg, "Bold", "../../assets/courbd.ttf");
  assert(font >= 0);

  for (int cnt = 0; ; cnt++) {
    glClearColor(0.1, 0.1, 0.1, 1.0);
    glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    nvgBeginFrame(vg, fb_w, fb_h, 1.0f);


    for (int k=0; k<3; k++) {
      float ang = (2*M_PI * (float)cnt / 120.0) + (k / 3.0) * 2*M_PI;

      nvgBeginPath(vg);
        nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 255));
        nvgStrokeWidth(vg, 5);
        
        nvgMoveTo(vg, fb_w/2 + 50 * cosf(ang), fb_h/2 + 50 * sinf(ang));
        nvgLineTo(vg, fb_w/2 + 15 * cosf(ang), fb_h/2 + 15 * sinf(ang));
        nvgMoveTo(vg, fb_w/2 - 15 * cosf(ang), fb_h/2 - 15 * sinf(ang));
        nvgLineTo(vg, fb_w/2 - 50 * cosf(ang), fb_h/2 - 50 * sinf(ang));
      nvgStroke(vg);
    }

    if (spintext) {
      nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
      nvgFontSize(vg, 96.0f);
      nvgText(vg, fb_w / 2, fb_h*2/3, spintext, NULL);      
    }

    nvgEndFrame(vg);

    eglSwapBuffers(display, surface);
    assert(glGetError() == GL_NO_ERROR);
  }

  return 0;
}
