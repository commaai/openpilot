#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cmath>

const int16_t lapl_conv_krnl[9] = {0, 1, 0,
                                   1, -4, 1,
                                   0, 1, 0};

// calculate score based on laplacians in one area
uint16_t get_lapmap_one(const int16_t *lap, int x_pitch, int y_pitch) {
  const int size = x_pitch * y_pitch;
  // avg and max of roi
  int16_t max = 0;
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    const int16_t v = lap[i % x_pitch + (i / x_pitch) * x_pitch];
    sum += v;
    if (v > max) max = v;
  }

  const int16_t mean = sum / size;

  // var of roi
  int var = 0;
  for (int i = 0; i < size; ++i) {
    var += std::pow(lap[i % x_pitch + (i / x_pitch) * x_pitch] - mean, 2);
  }

  const float fvar = (float)var / size;
  return std::min(5 * fvar + max, (float)65535);
}

bool is_blur(const uint16_t *lapmap, const size_t size) {
  float bad_sum = 0;
  for (int i = 0; i < size; i++) {
    if (lapmap[i] < LM_THRESH) {
      bad_sum += 1 / (float)size;
    }
  }
  return (bad_sum > LM_PREC_THRESH);
}

static cl_program build_conv_program(cl_device_id device_id, cl_context context, int image_w, int image_h, int filter_size) {
  char args[4096];
  snprintf(args, sizeof(args),
          "-cl-fast-relaxed-math -cl-denorms-are-zero "
          "-DIMAGE_W=%d -DIMAGE_H=%d -DFLIP_RB=%d "
          "-DFILTER_SIZE=%d -DHALF_FILTER_SIZE=%d -DTWICE_HALF_FILTER_SIZE=%d -DHALF_FILTER_SIZE_IMAGE_W=%d",
          image_w, image_h, 1,
          filter_size, filter_size/2, (filter_size/2)*2, (filter_size/2)*image_w);
  return cl_program_from_file(context, device_id, "imgproc/conv.cl", args);
}

LapConv::LapConv(cl_device_id device_id, cl_context ctx, int rgb_width, int rgb_height, int filter_size)
    : width(rgb_width / NUM_SEGMENTS_X), height(rgb_height / NUM_SEGMENTS_Y), 
      roi_buf(width * height * 3), result_buf(width * height) {

  prg = build_conv_program(device_id, ctx, width, height, filter_size);
  krnl = CL_CHECK_ERR(clCreateKernel(prg, "rgb2gray_conv2d", &err));
  // TODO: Removed CL_MEM_SVM_FINE_GRAIN_BUFFER, confirm it doesn't matter
  roi_cl = CL_CHECK_ERR(clCreateBuffer(ctx, CL_MEM_READ_WRITE, roi_buf.size() * sizeof(roi_buf[0]), NULL, &err));
  result_cl = CL_CHECK_ERR(clCreateBuffer(ctx, CL_MEM_READ_WRITE, result_buf.size() * sizeof(result_buf[0]), NULL, &err));
  filter_cl = CL_CHECK_ERR(clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          9 * sizeof(int16_t), (void *)&lapl_conv_krnl, &err));
}

LapConv::~LapConv() {
  CL_CHECK(clReleaseMemObject(roi_cl));
  CL_CHECK(clReleaseMemObject(result_cl));
  CL_CHECK(clReleaseMemObject(filter_cl));
  CL_CHECK(clReleaseKernel(krnl));
  CL_CHECK(clReleaseProgram(prg));
}

uint16_t LapConv::Update(cl_command_queue q, const uint8_t *rgb_buf, const int roi_id) {
  // sharpness scores
  const int x_offset = ROI_X_MIN + roi_id % (ROI_X_MAX - ROI_X_MIN + 1);
  const int y_offset = ROI_Y_MIN + roi_id / (ROI_X_MAX - ROI_X_MIN + 1);

  const uint8_t *rgb_offset = rgb_buf + y_offset * height * FULL_STRIDE_X * 3 + x_offset * width * 3;
  for (int i = 0; i < height; ++i) {
    memcpy(&roi_buf[i * width * 3], &rgb_offset[i * FULL_STRIDE_X * 3], width * 3);
  }

  constexpr int local_mem_size = (CONV_LOCAL_WORKSIZE + 2 * (3 / 2)) * (CONV_LOCAL_WORKSIZE + 2 * (3 / 2)) * (3 * sizeof(uint8_t));
  const size_t global_work_size[] = {(size_t)width, (size_t)height};
  const size_t local_work_size[] = {CONV_LOCAL_WORKSIZE, CONV_LOCAL_WORKSIZE};

  CL_CHECK(clEnqueueWriteBuffer(q, roi_cl, true, 0, roi_buf.size() * sizeof(roi_buf[0]), roi_buf.data(), 0, 0, 0));
  CL_CHECK(clSetKernelArg(krnl, 0, sizeof(cl_mem), (void *)&roi_cl));
  CL_CHECK(clSetKernelArg(krnl, 1, sizeof(cl_mem), (void *)&result_cl));
  CL_CHECK(clSetKernelArg(krnl, 2, sizeof(cl_mem), (void *)&filter_cl));
  CL_CHECK(clSetKernelArg(krnl, 3, local_mem_size, 0));
  cl_event conv_event;
  CL_CHECK(clEnqueueNDRangeKernel(q, krnl, 2, NULL, global_work_size, local_work_size, 0, 0, &conv_event));
  CL_CHECK(clWaitForEvents(1, &conv_event));
  CL_CHECK(clReleaseEvent(conv_event));
  CL_CHECK(clEnqueueReadBuffer(q, result_cl, true, 0,
                               result_buf.size() * sizeof(result_buf[0]), result_buf.data(), 0, 0, 0));

  return get_lapmap_one(result_buf.data(), width, height);
}
