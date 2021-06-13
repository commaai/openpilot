#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "selfdrive/common/clutil.h"

#define NUM_SEGMENTS_X 8
#define NUM_SEGMENTS_Y 6

#define ROI_X_MIN 1
#define ROI_X_MAX 6
#define ROI_Y_MIN 2
#define ROI_Y_MAX 3

#define LM_THRESH 120
#define LM_PREC_THRESH 0.9 // 90 perc is blur

// only apply to QCOM
#define FULL_STRIDE_X 1280
#define FULL_STRIDE_Y 896

#define CONV_LOCAL_WORKSIZE 16

class LapConv {
public:
  LapConv(cl_device_id device_id, cl_context ctx, int rgb_width, int rgb_height, int filter_size);
  ~LapConv();
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
