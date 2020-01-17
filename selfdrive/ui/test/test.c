#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "common/framebuffer.h"
#include "common/touch.h"

typedef struct UIState {
  FramebufferState *fb;
  int fb_w, fb_h;
  EGLDisplay display;
  EGLSurface surface;
} UIState;

TouchState touch = {0};

void wait_for_touch() {
  int touch_x = -1, touch_y = -1;
  while (1) {
    int touched = touch_poll(&touch, &touch_x, &touch_y, 0);
    if (touched == 1) { break; }
  }
}

int main() {
  UIState uistate;
  UIState *s = &uistate;

  memset(s, 0, sizeof(UIState));
  s->fb = framebuffer_init("ui", 0x00010000, true,
                           &s->display, &s->surface, &s->fb_w, &s->fb_h);

  touch_init(&touch);

  printf("waiting for touch with screen on\n");
  framebuffer_set_power(s->fb, HWC_POWER_MODE_NORMAL);
  wait_for_touch();

  printf("waiting for touch with screen off\n");
  framebuffer_set_power(s->fb, HWC_POWER_MODE_OFF);
  wait_for_touch();
  printf("done\n");
}

