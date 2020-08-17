#include <assert.h>
#include <stdio.h>
#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "clutil.h"
#include "transforms/transform.h"

typedef struct {
  int disable_model;
  Transform transform;

  int in_width;
  int in_height;
  int out_width;
  int out_height;

  cl_context context;
  cl_command_queue command_queue;
  cl_device_id device_id;

  size_t in_yuv_size;
  cl_mem in_yuv_cl;

  cl_mem out_y_cl, out_u_cl, out_v_cl;
} VisionTest;

void initialize_opencl(VisionTest* visiontest) {
  // init cl
  int err;
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_CPU);
  visiontest->context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  assert(err == 0);

  visiontest->device_id = device_id;
}

VisionTest* visiontest_create(int temporal_model, int disable_model,
                              int input_width, int input_height,
                              int model_input_width, int model_input_height) {
  int err = 0;

  VisionTest* const vt = calloc(1, sizeof(*vt));
  assert(vt);

  vt->disable_model = disable_model;
  vt->in_width = input_width;
  vt->in_height = input_height;
  vt->out_width = model_input_width;
  vt->out_height = model_input_height;

  initialize_opencl(vt);

  transform_init(&vt->transform, vt->context, vt->device_id);


  assert((vt->in_width%2) == 0 && (vt->in_height%2) == 0);
  vt->in_yuv_size = vt->in_width*vt->in_height*3/2;
  vt->in_yuv_cl = clCreateBuffer(vt->context, CL_MEM_READ_WRITE,
                                 vt->in_yuv_size, NULL, &err);
  assert(err == 0);

  vt->out_y_cl = clCreateBuffer(vt->context, CL_MEM_READ_WRITE,
                                vt->out_width*vt->out_width, NULL, &err);
  assert(err == 0);
  vt->out_u_cl = clCreateBuffer(vt->context, CL_MEM_READ_WRITE,
                                 vt->out_width*vt->out_width/4, NULL, &err);
  assert(err == 0);
  vt->out_v_cl = clCreateBuffer(vt->context, CL_MEM_READ_WRITE,
                                 vt->out_width*vt->out_width/4, NULL, &err);
  assert(err == 0);

  vt->command_queue = clCreateCommandQueue(vt->context, vt->device_id, 0, &err);
  assert(err == 0);

  return vt;
}

void visiontest_destroy(VisionTest* vt) {
  transform_destroy(&vt->transform);

  int err = 0;

  err = clReleaseMemObject(vt->in_yuv_cl);
  assert(err == 0);
  err = clReleaseMemObject(vt->out_y_cl);
  assert(err == 0);
  err = clReleaseMemObject(vt->out_u_cl);
  assert(err == 0);
  err = clReleaseMemObject(vt->out_v_cl);
  assert(err == 0);

  err = clReleaseCommandQueue(vt->command_queue);
  assert(err == 0);

  err = clReleaseContext(vt->context);
  assert(err == 0);

  free(vt);
}

void visiontest_transform(VisionTest* vt, const uint8_t* yuv_data,
                          uint8_t* out_y, uint8_t* out_u, uint8_t* out_v,
                          const float* transform) {
  int err = 0;

  err = clEnqueueWriteBuffer(vt->command_queue, vt->in_yuv_cl, CL_FALSE,
                             0, vt->in_yuv_size, yuv_data, 0, NULL, NULL);
  assert(err == 0);

  mat3 transform_m = *(const mat3*)transform;

  transform_queue(&vt->transform, vt->command_queue,
                  vt->in_yuv_cl, vt->in_width, vt->in_height,
                  vt->out_y_cl, vt->out_u_cl, vt->out_v_cl,
                  vt->out_width, vt->out_height,
                  transform_m);

  err = clEnqueueReadBuffer(vt->command_queue, vt->out_y_cl, CL_FALSE,
                            0, vt->out_width*vt->out_height, out_y,
                            0, NULL, NULL);
  assert(err == 0);
  err = clEnqueueReadBuffer(vt->command_queue, vt->out_u_cl, CL_FALSE,
                            0, vt->out_width*vt->out_height/4, out_u,
                            0, NULL, NULL);
  assert(err == 0);
  err = clEnqueueReadBuffer(vt->command_queue, vt->out_v_cl, CL_FALSE,
                            0, vt->out_width*vt->out_height/4, out_v,
                            0, NULL, NULL);
  assert(err == 0);

  clFinish(vt->command_queue);
}

