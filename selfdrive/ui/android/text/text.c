#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "nanovg.h"
#define NANOVG_GLES3_IMPLEMENTATION
#include "nanovg_gl.h"
#include "nanovg_gl_utils.h"

#include "common/framebuffer.h"
#include "common/touch.h"


#define COLOR_WHITE nvgRGBA(255, 255, 255, 255)
#define MAX_TEXT_SIZE 2048

extern const unsigned char _binary_opensans_regular_ttf_start[];
extern const unsigned char _binary_opensans_regular_ttf_end[];

int main(int argc, char** argv) {
  int err;

  // spinner
  int fb_w, fb_h;
  FramebufferState *fb = framebuffer_init("text", 0x00001000, false,
                                          &fb_w, &fb_h);
  assert(fb);

  NVGcontext *vg = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
  assert(vg);

  int font = nvgCreateFontMem(vg, "regular", (unsigned char*)_binary_opensans_regular_ttf_start, _binary_opensans_regular_ttf_end-_binary_opensans_regular_ttf_start, 0);
assert(font >= 0);

  // Awake
  framebuffer_set_power(fb, HWC_POWER_MODE_NORMAL);
  set_brightness(255);

  glClearColor(0.1, 0.1, 0.1, 1.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  nvgBeginFrame(vg, fb_w, fb_h, 1.0f);

  // background
  nvgBeginPath(vg);
  NVGpaint bg = nvgLinearGradient(vg, fb_w, 0, fb_w, fb_h,
  nvgRGBA(0, 0, 0, 175), nvgRGBA(0, 0, 0, 255));
  nvgFillPaint(vg, bg);
  nvgRect(vg, 0, 0, fb_w, fb_h);
  nvgFill(vg);


  // Text
  nvgFillColor(vg, COLOR_WHITE);
  nvgFontSize(vg, 75.0f);

  if (argc >= 2) {
    float x = 150;
    float y = 150;

    // Copy text
    char * text = malloc(MAX_TEXT_SIZE);
    strncpy(text, argv[1], MAX_TEXT_SIZE);

    float lineh;
    nvgTextMetrics(vg, NULL, NULL, &lineh);

    // nvgTextBox strips leading whitespace. We have to reimplement
    char * next = strtok(text, "\n");
    while (next != NULL){
      nvgText(vg, x, y, next, NULL);
      y += lineh;
      next = strtok(NULL, "\n");
    }
  }

  // Button
  int b_x = 1500;
  int b_y = 800;
  int b_w = 300;
  int b_h = 150;

  nvgBeginPath(vg);
  nvgFillColor(vg, nvgRGBA(8, 8, 8, 255));
  nvgRoundedRect(vg, b_x, b_y, b_w, b_h, 20);
  nvgFill(vg);

  nvgFillColor(vg, nvgRGBA(255, 255, 255, 255));
  nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
  nvgText(vg, b_x+b_w/2, b_y+b_h/2, "Exit", NULL);

  nvgBeginPath(vg);
  nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 50));
  nvgStrokeWidth(vg, 5);
  nvgRoundedRect(vg, b_x, b_y, b_w, b_h, 20);
  nvgStroke(vg);

  // Draw to screen
  nvgEndFrame(vg);
  framebuffer_swap(fb);
  assert(glGetError() == GL_NO_ERROR);


  // Wait for button
  TouchState touch;
  touch_init(&touch);

  while (true){
    int touch_x = -1, touch_y = -1;
    int res = touch_poll(&touch, &touch_x, &touch_y, 0);
    if (res){

      if (touch_x > b_x && touch_x < b_x + b_w){
        if (touch_y > b_y && touch_y < b_y + b_h){
          return 1;
        }
      }
    }

    usleep(1000000 / 60);
  }

  return 0;
}
