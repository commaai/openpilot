#include "selfdrive/modeld/models/commonmodel.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include "common/clutil.h"

ModelFrame::ModelFrame(cl_device_id device_id, cl_context context) {
  frame = std::make_unique<float[]>(MODEL_FRAME_SIZE);

  q = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &err));
  y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, MODEL_WIDTH * MODEL_HEIGHT, NULL, &err));
  u_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
  v_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
  net_input_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, MODEL_FRAME_SIZE * sizeof(float), NULL, &err));

  transform_init(&transform, context, device_id);
  loadyuv_init(&loadyuv, context, device_id, MODEL_WIDTH, MODEL_HEIGHT);
}

float* ModelFrame::prepare(cl_mem yuv_cl, int frame_width, int frame_height, int frame_stride, int frame_uv_offset, const mat3 &projection, cl_mem *output) {
  transform_queue(&this->transform, q,
                  yuv_cl, frame_width, frame_height, frame_stride, frame_uv_offset,
                  y_cl, u_cl, v_cl, MODEL_WIDTH, MODEL_HEIGHT, projection);
  loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, net_input_cl);
  CL_CHECK(clEnqueueReadBuffer(q, net_input_cl, CL_TRUE, 0, MODEL_FRAME_SIZE * sizeof(float), &frame[0], 0, nullptr, nullptr));
  clFinish(q);
  return &frame[0];
}

ModelFrame::~ModelFrame() {
  transform_destroy(&transform);
  loadyuv_destroy(&loadyuv);
  CL_CHECK(clReleaseMemObject(net_input_cl));
  CL_CHECK(clReleaseMemObject(v_cl));
  CL_CHECK(clReleaseMemObject(u_cl));
  CL_CHECK(clReleaseMemObject(y_cl));
  CL_CHECK(clReleaseCommandQueue(q));
}

float sigmoid(float input) {
  return 1 / (1 + expf(-input));
}
