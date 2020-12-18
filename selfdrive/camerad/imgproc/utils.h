#pragma once

#include <stdint.h>
#include <stddef.h>

#include <vector>

#include "clutil.h"

const int ROI_X_MIN = 1;
const int ROI_X_MAX = 6;
const int ROI_Y_MIN = 2;
const int ROI_Y_MAX = 3;

class Rgb2Gray {
public:
  Rgb2Gray(cl_device_id device_id, cl_context ctx, int rgb_width, int rgb_height, int filter_size);
  ~Rgb2Gray();
  uint16_t Update(cl_command_queue q, const uint8_t *rgb_buf, const int roi_id);

private:
  cl_mem roi_cl, result_cl, filter_cl;
  cl_program prg;
  cl_kernel krnl;
  const int width, height;
  std::vector<uint8_t> roi_buf;
  std::vector<int16_t> result_buf;
};

bool is_blur(const uint16_t *lapmap, const size_t size);
