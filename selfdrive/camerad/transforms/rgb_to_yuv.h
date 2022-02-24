#pragma once

#include "selfdrive/common/clutil.h"

class Rgb2Yuv {
public:
  Rgb2Yuv(cl_context ctx, cl_device_id device_id, int width, int height, int rgb_stride);
  ~Rgb2Yuv();
  void queue(cl_command_queue q, cl_mem rgb_cl, cl_mem yuv_cl);
private:
  size_t work_size[2];
  cl_kernel krnl;
};

