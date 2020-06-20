#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <assert.h>
#include <czmq.h>

#include <CL/cl2.hpp>

#include "common/mat.h"

constexpr int MODEL_WIDTH = 512;
constexpr int MODEL_HEIGHT = 256;
constexpr int MODEL_FRAME_SIZE = (MODEL_WIDTH * MODEL_HEIGHT * 3 / 2);
class ModelFrame {
 public:
  ModelFrame() {}
  ~ModelFrame();
  void init(cl::Context &ctx, cl::Device &device);
  void prepare(cl::Buffer &yuv_cl, int in_width, int in_height, mat3 transform);
  inline float *getFrame() const { return input_frames_; }
  inline size_t getFrameSize() const { return MODEL_FRAME_SIZE * 2; }

 private:
  void transform(cl::Buffer &in_yuv, int in_width, int in_height, mat3 projection);
  void loadyuv();
  float *input_frames_ = nullptr;
  cl::CommandQueue q_;
  cl::Buffer transformed_y_cl_, transformed_u_cl_, transformed_v_cl_;
  cl::Buffer m_y_cl, m_uv_cl_, net_input_;
  cl::Kernel transform_krnl_, loadys_krnl_, loaduv_krnl_;
};

inline float sigmoid(float input) { return 1 / (1 + expf(-input)); }
inline float softplus(float input) { return log1p(expf(input)); }
