#include "commonmodel.h"

#include <czmq.h>
#include "cereal/gen/c/log.capnp.h"
#include "common/mat.h"
#include "common/timing.h"

void model_input_init(ModelInput* s, int width, int height,
                      cl_device_id device_id, cl_context context) {
  int err;
  s->device_id = device_id;
  s->context = context;

  transform_init(&s->transform, context, device_id);
  s->transformed_width = width;
  s->transformed_height = height;

  s->transformed_y_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       s->transformed_width*s->transformed_height, NULL, &err);
  assert(err == 0);
  s->transformed_u_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       (s->transformed_width/2)*(s->transformed_height/2), NULL, &err);
  assert(err == 0);
  s->transformed_v_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       (s->transformed_width/2)*(s->transformed_height/2), NULL, &err);
  assert(err == 0);

  s->net_input_size = ((width*height*3)/2)*sizeof(float);
  s->net_input = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                s->net_input_size, (void*)NULL, &err);
  assert(err == 0);

  loadyuv_init(&s->loadyuv, context, device_id, s->transformed_width, s->transformed_height);
}

float *model_input_prepare(ModelInput* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform) {
  int err;
  int i = 0;
  transform_queue(&s->transform, q,
                  yuv_cl, width, height,
                  s->transformed_y_cl, s->transformed_u_cl, s->transformed_v_cl,
                  s->transformed_width, s->transformed_height,
                  transform);
  loadyuv_queue(&s->loadyuv, q,
                s->transformed_y_cl, s->transformed_u_cl, s->transformed_v_cl,
                s->net_input);
  float *net_input_buf = (float *)clEnqueueMapBuffer(q, s->net_input, CL_TRUE,
                                            CL_MAP_READ, 0, s->net_input_size,
                                            0, NULL, NULL, &err);
  clFinish(q);
  return net_input_buf;
}

void model_input_free(ModelInput* s) {
  transform_destroy(&s->transform);
  loadyuv_destroy(&s->loadyuv);
}


float sigmoid(float input) {
  return 1 / (1 + expf(-input));
}

float softplus(float input) {
  return log1p(expf(input));
}
