#ifndef COMMONMODEL_H
#define COMMONMODEL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "common/mat.h"
#include "transforms/transform.h"
#include "transforms/loadyuv.h"

#ifdef __cplusplus
extern "C" {
#endif

float softplus(float input);
float sigmoid(float input);

typedef struct ModelFrame {
  cl_device_id device_id;
  cl_context context;

  // input
  Transform transform;
  int transformed_width, transformed_height;
  cl_mem transformed_y_cl, transformed_u_cl, transformed_v_cl;
  LoadYUVState loadyuv;
  cl_mem net_input;
  size_t net_input_size;
} ModelFrame;

void frame_init(ModelFrame* frame, int width, int height,
                      cl_device_id device_id, cl_context context);
float *frame_prepare(ModelFrame* frame, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform);
void frame_free(ModelFrame* frame);

#ifdef __cplusplus
}
#endif

#endif

