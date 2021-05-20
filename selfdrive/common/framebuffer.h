#pragma once

#include <cstdlib>

#include "hardware/hwcomposer_defs.h"

bool set_brightness(int brightness);

struct FramebufferState;
class FrameBuffer {
 public:
  FrameBuffer(const char *name, uint32_t layer, int alpha, int *out_w, int *out_h);
  ~FrameBuffer();
  void set_power(int mode);
  void swap();
private:
  FramebufferState *s;
};
