#ifndef COMMONMODEL_H
#define COMMONMODEL_H

#include <CL/cl.h>

#include "common/mat.h"
#include "common/modeldata.h"
#include "transforms/transform.h"
#include "transforms/loadyuv.h"

#ifdef __cplusplus
extern "C" {
#endif

void softmax(const float* input, float* output, size_t len);
float softplus(float input);
float sigmoid(float input);

typedef struct ModelInput {
  cl_device_id device_id;
  cl_context context;

  // input
  Transform transform;
  int transformed_width, transformed_height;
  cl_mem transformed_y_cl, transformed_u_cl, transformed_v_cl;
  LoadYUVState loadyuv;
  cl_mem net_input;
  size_t net_input_size;
} ModelInput;

void model_input_init(ModelInput* s, int width, int height,
                      cl_device_id device_id, cl_context context);
float *model_input_prepare(ModelInput* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform);
void model_input_free(ModelInput* s);

void model_publish(void* sock, uint32_t frame_id,
                   const mat3 transform, const ModelData data);

#ifdef __cplusplus
}
#endif

#endif

