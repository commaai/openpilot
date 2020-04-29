#include "commonmodel.h"

#include <czmq.h>
#include <math.h>
#include <stdio.h>
#include "cereal/gen/c/log.capnp.h"
#include "common/mat.h"
#include "common/timing.h"

void frame_init(ModelFrame* frame, int width, int height,
                      cl_device_id device_id, cl_context context) {
  int err;
  frame->device_id = device_id;
  frame->context = context;

  transform_init(&frame->transform, context, device_id);
  frame->transformed_width = width;
  frame->transformed_height = height;

  frame->transformed_y_cl = clCreateBuffer(frame->context, CL_MEM_READ_WRITE,
                                       frame->transformed_width*frame->transformed_height, NULL, &err);
  assert(err == 0);
  frame->transformed_u_cl = clCreateBuffer(frame->context, CL_MEM_READ_WRITE,
                                       (frame->transformed_width/2)*(frame->transformed_height/2), NULL, &err);
  assert(err == 0);
  frame->transformed_v_cl = clCreateBuffer(frame->context, CL_MEM_READ_WRITE,
                                       (frame->transformed_width/2)*(frame->transformed_height/2), NULL, &err);
  assert(err == 0);

  frame->net_input_size = ((width*height*3)/2)*sizeof(float);
  frame->net_input = clCreateBuffer(frame->context, CL_MEM_READ_WRITE,
                                frame->net_input_size, (void*)NULL, &err);
  assert(err == 0);

  loadyuv_init(&frame->loadyuv, context, device_id, frame->transformed_width, frame->transformed_height);
}

float *frame_prepare(ModelFrame* frame, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform) {
  int err;
  int i = 0;
  transform_queue(&frame->transform, q,
                  yuv_cl, width, height,
                  frame->transformed_y_cl, frame->transformed_u_cl, frame->transformed_v_cl,
                  frame->transformed_width, frame->transformed_height,
                  transform);
  loadyuv_queue(&frame->loadyuv, q,
                frame->transformed_y_cl, frame->transformed_u_cl, frame->transformed_v_cl,
                frame->net_input);
  float *net_input_buf = (float *)clEnqueueMapBuffer(q, frame->net_input, CL_TRUE,
                                            CL_MAP_READ, 0, frame->net_input_size,
                                            0, NULL, NULL, &err);
  clFinish(q);
  return net_input_buf;
}

void frame_free(ModelFrame* frame) {
  transform_destroy(&frame->transform);
  loadyuv_destroy(&frame->loadyuv);
}


float sigmoid(float input) {
  return 1 / (1 + expf(-input));
}

float softplus(float input) {
  return log1p(expf(input));
}

bool allclose(float *arr1, float *arr2, int len, float rtol, float atol) {
  float max_err = 0;
  float max_rerr = 0;
  for (int i=0; i<len; i++) {
    max_err = fabs(arr1[i] - arr2[i])>max_err ? fabs(arr1[i] - arr2[i]):max_err;
    max_rerr = fabs((arr1[i] - arr2[i])/(arr1[i] + 1e-6))>max_rerr ? fabs((arr1[i] - arr2[i])/(arr1[i] + 1e-6)):max_rerr;
  }
  return (bool)(max_err < atol && max_rerr < rtol);
}

void f32_fromfile(const char* filepath, float *buf, int len) {
  FILE *stream;
  stream = fopen(filepath, "r");
  for (int i=0; i<len; i++) {
    fscanf(stream, "%f", &buf[i]);
  }
  fclose(stream);
}
