#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <float.h>
#include <stdlib.h>
#include "common/mat.h"
#include "transforms/transform.h"
#include "transforms/loadyuv.h"

const bool send_raw_pred = getenv("SEND_RAW_PRED") != NULL;

void softmax(const float* input, float* output, size_t len);
float softplus(float input);
float sigmoid(float input);

typedef struct ModelFrame {
  Transform transform;
  int width, height;
  cl_mem y_cl, u_cl, v_cl;
  LoadYUVState loadyuv;
  cl_mem net_input;
  size_t net_input_size;
} ModelFrame;

void frame_init(ModelFrame* frame, int width, int height,
                      cl_device_id device_id, cl_context context);
float *frame_prepare(ModelFrame* frame, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           const mat3 &transform);
void frame_free(ModelFrame* frame);
