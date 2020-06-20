#include "commonmodel.h"

void ModelFrame::init(cl::Context &ctx, cl::Device &device, int width, int height) {
  q_ = cl::CommandQueue(ctx, device);
  width_ = width;
  height_ = height;

  transformed_y_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, width_ * height_);
  transformed_u_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, (width_ / 2) * (height_ / 2));
  transformed_v_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, (width_ / 2) * (height_ / 2));
  m_y_cl = cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * 3 * sizeof(float));
  m_uv_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * 3 * sizeof(float));

  net_input_size_ = ((width_ * height_ * 3) / 2) * sizeof(float);
  net_input_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, net_input_size_);

  cl_program program = CLU_LOAD_FROM_FILE(ctx.get(), device.get(), "transforms/transform.cl", "");
  cl::Program prog(program, true);
  kernel_ = cl::Kernel(prog, "warpPerspective");

  char args[1024];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
           "-DTRANSFORMED_WIDTH=%d -DTRANSFORMED_HEIGHT=%d",
           width, height);
  program = CLU_LOAD_FROM_FILE(ctx.get(), device.get(), "transforms/loadyuv.cl", args);
  cl::Program prog2(program, true);
  loadys_krnl_ = cl::Kernel(prog2, "loadys");
  loaduv_krnl_ = cl::Kernel(prog2, "loaduv");
}

float *ModelFrame::prepare(cl::Buffer &yuv_cl, int in_width, int in_height, mat3 &transform) {
  warpPerspectiveQueue(yuv_cl, in_width, in_height, transform);
  yuvQueue();
  float *net_input_buf = (float *)q_.enqueueMapBuffer(net_input_, CL_TRUE, CL_MAP_READ, 0, net_input_size_);
  q_.finish();
  return net_input_buf;
}

void ModelFrame::warpPerspectiveQueue(cl::Buffer &in_yuv, int in_width, int in_height, mat3 &projection) {
  const int zero = 0;

  // sampled using pixel center origin
  // (because thats how fastcv and opencv does it)

  mat3 projection_y = projection;

  // in and out uv is half the size of y.
  mat3 projection_uv = transform_scale_buffer(projection, 0.5);
  q_.enqueueWriteBuffer(m_y_cl, CL_TRUE, 0, 3 * 3 * sizeof(float), (void *)projection_y.v);
  q_.enqueueWriteBuffer(m_uv_cl_, CL_TRUE, 0, 3 * 3 * sizeof(float), (void *)projection_uv.v);

  const int in_y_width = in_width;
  const int in_y_height = in_height;
  const int in_uv_width = in_width / 2;
  const int in_uv_height = in_height / 2;
  const int in_y_offset = 0;
  const int in_u_offset = in_y_offset + in_y_width * in_y_height;
  const int in_v_offset = in_u_offset + in_uv_width * in_uv_height;

  const int out_y_width = width_;
  const int out_y_height = height_;
  const int out_uv_width = width_ / 2;
  const int out_uv_height = height_ / 2;
  kernel_.setArg(0, in_yuv);
  kernel_.setArg(1, in_y_width);
  kernel_.setArg(2, in_y_offset);
  kernel_.setArg(3, in_y_height);
  kernel_.setArg(4, in_y_width);
  kernel_.setArg(5, transformed_y_cl_);
  kernel_.setArg(6, out_y_width);
  kernel_.setArg(7, zero);
  kernel_.setArg(8, out_y_height);
  kernel_.setArg(9, out_y_width);
  kernel_.setArg(10, m_y_cl);

  q_.enqueueNDRangeKernel(kernel_, cl::NullRange, cl::NDRange(out_y_width, out_y_height));

  kernel_.setArg(1, in_uv_width);
  kernel_.setArg(2, in_u_offset);
  kernel_.setArg(3, in_uv_height);
  kernel_.setArg(4, in_uv_width);
  kernel_.setArg(5, transformed_u_cl_);
  kernel_.setArg(6, out_uv_width);
  kernel_.setArg(7, zero);
  kernel_.setArg(8, out_uv_height);
  kernel_.setArg(9, out_uv_width);
  kernel_.setArg(10, m_uv_cl_);

  q_.enqueueNDRangeKernel(kernel_, cl::NullRange, cl::NDRange(out_uv_width, out_uv_height));

  kernel_.setArg(2, in_v_offset);
  kernel_.setArg(5, transformed_v_cl_);

  q_.enqueueNDRangeKernel(kernel_, cl::NullRange, cl::NDRange(out_uv_width, out_uv_height));
}

void ModelFrame::yuvQueue() {
  loadys_krnl_.setArg(0, transformed_y_cl_);
  loadys_krnl_.setArg(1, net_input_);

  const size_t loadys_work_size = (width_ * height_) / 8;
  q_.enqueueNDRangeKernel(loadys_krnl_, cl::NullRange, cl::NDRange(loadys_work_size));
  const size_t loaduv_work_size = ((width_ / 2) * (height_ / 2)) / 8;
  cl_int loaduv_out_off = (width_ * height_);

  loaduv_krnl_.setArg(0, transformed_u_cl_);
  loaduv_krnl_.setArg(1, net_input_);
  loaduv_krnl_.setArg(2, loaduv_out_off);

  q_.enqueueNDRangeKernel(loaduv_krnl_, cl::NullRange, cl::NDRange(loaduv_work_size));
  loaduv_out_off += (width_ / 2) * (height_ / 2);

  loaduv_krnl_.setArg(0, transformed_v_cl_);
  loaduv_krnl_.setArg(1, net_input_);
  loaduv_krnl_.setArg(2, loaduv_out_off);

  q_.enqueueNDRangeKernel(loaduv_krnl_, cl::NullRange, cl::NDRange(loaduv_work_size));
}
