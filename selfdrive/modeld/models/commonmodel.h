#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <assert.h>
#include <czmq.h>
#include "common/mat.h"

class ModelFrame {
 public:
  ModelFrame() {}
  void init(cl::Context &ctx, cl::Device &device, int width, int height);
  float *prepare(cl::Buffer &yuv_cl, int in_width, int in_height, mat3 &transform);
  void unmap(void *buf) { q_.enqueueUnmapMemObject(net_input_, (void *)buf); }

 private:
  void warpPerspectiveQueue(cl::Buffer &in_yuv, int in_width, int in_height, mat3 &projection);
  void yuvQueue();

  cl::CommandQueue q_;
  cl::Buffer transformed_y_cl_, transformed_u_cl_, transformed_v_cl_;
  cl::Buffer m_y_cl, m_uv_cl_, net_input_;
  size_t net_input_size_;
  cl::Kernel kernel_, loadys_krnl_, loaduv_krnl_;
  int width_, height_;
};

inline float sigmoid(float input) { return 1 / (1 + expf(-input)); }
inline float softplus(float input) { return log1p(expf(input)); }
