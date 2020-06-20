#include "commonmodel.h"

#include "clutil.h"
void ModelFrame::init(cl::Context &ctx, cl::Device &device) {
  input_frames_ = (float *)calloc(MODEL_FRAME_SIZE * 2, sizeof(float));

  q_ = cl::CommandQueue(ctx, device);

  transformed_y_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, MODEL_WIDTH * MODEL_HEIGHT);
  transformed_u_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2));
  transformed_v_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2));
  m_y_cl = cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * 3 * sizeof(float));
  m_uv_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, 3 * 3 * sizeof(float));
  net_input_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, MODEL_FRAME_SIZE * sizeof(float));

  cl::Program program = cl::Program(CLU_LOAD_FROM_FILE(ctx.get(), device.get(), "transforms/transform.cl", ""));
  transform_krnl_ = cl::Kernel(program, "warpPerspective");

  char args[1024];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
           "-DTRANSFORMED_WIDTH=%d -DTRANSFORMED_HEIGHT=%d",
           MODEL_WIDTH, MODEL_HEIGHT);
  program = cl::Program(CLU_LOAD_FROM_FILE(ctx.get(), device.get(), "transforms/loadyuv.cl", args));
  loadys_krnl_ = cl::Kernel(program, "loadys");
  loaduv_krnl_ = cl::Kernel(program, "loaduv");
}

void ModelFrame::prepare(cl::Buffer &yuv_cl, int in_width, int in_height, mat3 tf) {
  transform(yuv_cl, in_width, in_height, tf);
  loadyuv();
  q_.finish();
  memmove(input_frames_, &input_frames_[MODEL_FRAME_SIZE], MODEL_FRAME_SIZE * sizeof(float));
  q_.enqueueReadBuffer(net_input_, CL_TRUE, 0, MODEL_FRAME_SIZE * sizeof(float), &input_frames_[MODEL_FRAME_SIZE]);
}

void ModelFrame::transform(cl::Buffer &in_yuv, int in_width, int in_height, mat3 projection) {
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

  const int out_y_width = MODEL_WIDTH;
  const int out_y_height = MODEL_HEIGHT;
  const int out_uv_width = MODEL_WIDTH / 2;
  const int out_uv_height = MODEL_HEIGHT / 2;
  transform_krnl_.setArg(0, in_yuv);
  transform_krnl_.setArg(1, in_y_width);
  transform_krnl_.setArg(2, in_y_offset);
  transform_krnl_.setArg(3, in_y_height);
  transform_krnl_.setArg(4, in_y_width);
  transform_krnl_.setArg(5, transformed_y_cl_);
  transform_krnl_.setArg(6, out_y_width);
  transform_krnl_.setArg(7, zero);
  transform_krnl_.setArg(8, out_y_height);
  transform_krnl_.setArg(9, out_y_width);
  transform_krnl_.setArg(10, m_y_cl);

  q_.enqueueNDRangeKernel(transform_krnl_, cl::NullRange, cl::NDRange(out_y_width, out_y_height));

  transform_krnl_.setArg(1, in_uv_width);
  transform_krnl_.setArg(2, in_u_offset);
  transform_krnl_.setArg(3, in_uv_height);
  transform_krnl_.setArg(4, in_uv_width);
  transform_krnl_.setArg(5, transformed_u_cl_);
  transform_krnl_.setArg(6, out_uv_width);
  transform_krnl_.setArg(7, zero);
  transform_krnl_.setArg(8, out_uv_height);
  transform_krnl_.setArg(9, out_uv_width);
  transform_krnl_.setArg(10, m_uv_cl_);

  q_.enqueueNDRangeKernel(transform_krnl_, cl::NullRange, cl::NDRange(out_uv_width, out_uv_height));

  transform_krnl_.setArg(2, in_v_offset);
  transform_krnl_.setArg(5, transformed_v_cl_);

  q_.enqueueNDRangeKernel(transform_krnl_, cl::NullRange, cl::NDRange(out_uv_width, out_uv_height));
}

void ModelFrame::loadyuv() {
  loadys_krnl_.setArg(0, transformed_y_cl_);
  loadys_krnl_.setArg(1, net_input_);

  const size_t loadys_work_size = (MODEL_WIDTH * MODEL_HEIGHT) / 8;
  q_.enqueueNDRangeKernel(loadys_krnl_, cl::NullRange, cl::NDRange(loadys_work_size));
  const size_t loaduv_work_size = ((MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2)) / 8;
  cl_int loaduv_out_off = (MODEL_WIDTH * MODEL_HEIGHT);

  loaduv_krnl_.setArg(0, transformed_u_cl_);
  loaduv_krnl_.setArg(1, net_input_);
  loaduv_krnl_.setArg(2, loaduv_out_off);

  q_.enqueueNDRangeKernel(loaduv_krnl_, cl::NullRange, cl::NDRange(loaduv_work_size));
  loaduv_out_off += (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2);

  loaduv_krnl_.setArg(0, transformed_v_cl_);
  loaduv_krnl_.setArg(1, net_input_);
  loaduv_krnl_.setArg(2, loaduv_out_off);

  q_.enqueueNDRangeKernel(loaduv_krnl_, cl::NullRange, cl::NDRange(loaduv_work_size));
}

ModelFrame::~ModelFrame() {
  free(input_frames_);
}
