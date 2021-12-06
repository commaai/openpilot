#pragma once

#include <assert.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>

#include <memory>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "selfdrive/common/mat.h"
#include "selfdrive/modeld/transforms/loadyuv.h"
#include "selfdrive/modeld/transforms/transform.h"

constexpr int MODEL_WIDTH = 512;
constexpr int MODEL_HEIGHT = 256;
constexpr int MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 / 2;

const bool send_raw_pred = getenv("SEND_RAW_PRED") != NULL;

template<size_t input_size, size_t output_size>
void softmax(const std::array<float, input_size> &input, std::array<float, output_size> &output, const int output_offset=0) {
  static_assert(input_size <= output_size);
  assert(output_offset + input_size <= output_size);

  const float max_val = *std::max_element(input.data(), input.end());
  float denominator = 0;
  for(int i = 0; i < input_size; i++) {
    float const v_exp = expf(input[i] - max_val);
    denominator += v_exp;
    output[output_offset + i] = v_exp;
  }

  const float inv_denominator = 1. / denominator;
  for(int i = 0; i < input_size; i++) {
    output[output_offset + i] *= inv_denominator;
  }
}

float softplus(float input);
float sigmoid(float input);

class ModelFrame {
 public:
  ModelFrame(cl_device_id device_id, cl_context context);
  ~ModelFrame();
  float* prepare(cl_mem yuv_cl, int width, int height, const mat3& transform, cl_mem *output);

  const int buf_size = MODEL_FRAME_SIZE * 2;

 private:
  Transform transform;
  LoadYUVState loadyuv;
  cl_command_queue q;
  cl_mem y_cl, u_cl, v_cl, net_input_cl;
  std::unique_ptr<float[]> input_frames;
};
