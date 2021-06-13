#include "selfdrive/modeld/models/commonmodel.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/timing.h"

void frame_init(ModelFrame* frame, int width, int height,
                      cl_device_id device_id, cl_context context) {
  transform_init(&frame->transform, context, device_id);
  frame->width = width;
  frame->height = height;

  frame->y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)width*height, NULL, &err));
  frame->u_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)(width/2)*(height/2), NULL, &err));
  frame->v_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)(width/2)*(height/2), NULL, &err));
  frame->net_input_size = ((width*height*3)/2)*sizeof(float);
  frame->net_input = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                frame->net_input_size, (void*)NULL, &err));
  loadyuv_init(&frame->loadyuv, context, device_id, width, height);
}

float *frame_prepare(ModelFrame* frame, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           const mat3 &transform) {
  transform_queue(&frame->transform, q,
                  yuv_cl, width, height,
                  frame->y_cl, frame->u_cl, frame->v_cl,
                  frame->width, frame->height,
                  transform);
  loadyuv_queue(&frame->loadyuv, q,
                frame->y_cl, frame->u_cl, frame->v_cl,
                frame->net_input);
  float *net_input_buf = (float *)CL_CHECK_ERR(clEnqueueMapBuffer(q, frame->net_input, CL_TRUE,
                                            CL_MAP_READ, 0, frame->net_input_size,
                                            0, NULL, NULL, &err));
  clFinish(q);
  return net_input_buf;
}

void frame_free(ModelFrame* frame) {
  transform_destroy(&frame->transform);
  loadyuv_destroy(&frame->loadyuv);
  CL_CHECK(clReleaseMemObject(frame->net_input));
  CL_CHECK(clReleaseMemObject(frame->v_cl));
  CL_CHECK(clReleaseMemObject(frame->u_cl));
  CL_CHECK(clReleaseMemObject(frame->y_cl));
}

void softmax(const float* input, float* output, size_t len) {
  const float max_val = *std::max_element(input, input + len);
  float denominator = 0;
  for(int i = 0; i < len; i++) {
    float const v_exp = expf(input[i] - max_val);
    denominator += v_exp;
    output[i] = v_exp;
  }

  const float inv_denominator = 1. / denominator;
  for(int i = 0; i < len; i++) {
    output[i] *= inv_denominator;
  }
}

float sigmoid(float input) {
  return 1 / (1 + expf(-input));
}

float softplus(float input) {
  return log1p(expf(input));
}
