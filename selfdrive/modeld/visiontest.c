#include <assert.h>
#include <stdio.h>
#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "selfdrive/common/clutil.h"
#include "selfdrive/modeld/transforms/transform.h"

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
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_CPU);
  visiontest->context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
  visiontest->device_id = device_id;
}

VisionTest* visiontest_create(int temporal_model, int disable_model,
                              int input_width, int input_height,
                              int model_input_width, int model_input_height) {
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
  vt->in_yuv_cl = CL_CHECK_ERR(clCreateBuffer(vt->context, CL_MEM_READ_WRITE,
                                 vt->in_yuv_size, NULL, &err));
  vt->out_y_cl = CL_CHECK_ERR(clCreateBuffer(vt->context, CL_MEM_READ_WRITE,
                                vt->out_width*vt->out_width, NULL, &err));
  vt->out_u_cl = CL_CHECK_ERR(clCreateBuffer(vt->context, CL_MEM_READ_WRITE,
                                 vt->out_width*vt->out_width/4, NULL, &err));
  vt->out_v_cl = CL_CHECK_ERR(clCreateBuffer(vt->context, CL_MEM_READ_WRITE,
                                 vt->out_width*vt->out_width/4, NULL, &err));
  vt->command_queue = CL_CHECK_ERR(clCreateCommandQueue(vt->context, vt->device_id, 0, &err));
  return vt;
}

void visiontest_destroy(VisionTest* vt) {
  transform_destroy(&vt->transform);

  CL_CHECK(clReleaseMemObject(vt->in_yuv_cl));
  CL_CHECK(clReleaseMemObject(vt->out_y_cl));
  CL_CHECK(clReleaseMemObject(vt->out_u_cl));
  CL_CHECK(clReleaseMemObject(vt->out_v_cl));
  CL_CHECK(clReleaseCommandQueue(vt->command_queue));
  CL_CHECK(clReleaseContext(vt->context));

  free(vt);
}

void visiontest_transform(VisionTest* vt, const uint8_t* yuv_data,
                          uint8_t* out_y, uint8_t* out_u, uint8_t* out_v,
                          const float* transform) {
  CL_CHECK(clEnqueueWriteBuffer(vt->command_queue, vt->in_yuv_cl, CL_FALSE,
                             0, vt->in_yuv_size, yuv_data, 0, NULL, NULL));

  mat3 transform_m = *(const mat3*)transform;

  transform_queue(&vt->transform, vt->command_queue,
                  vt->in_yuv_cl, vt->in_width, vt->in_height,
                  vt->out_y_cl, vt->out_u_cl, vt->out_v_cl,
                  vt->out_width, vt->out_height,
                  transform_m);

  CL_CHECK(clEnqueueReadBuffer(vt->command_queue, vt->out_y_cl, CL_FALSE,
                            0, vt->out_width*vt->out_height, out_y,
                            0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(vt->command_queue, vt->out_u_cl, CL_FALSE,
                            0, vt->out_width*vt->out_height/4, out_u,
                            0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(vt->command_queue, vt->out_v_cl, CL_FALSE,
                            0, vt->out_width*vt->out_height/4, out_v,
                            0, NULL, NULL));

  clFinish(vt->command_queue);
}

