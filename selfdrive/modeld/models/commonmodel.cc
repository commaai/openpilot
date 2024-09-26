#include "selfdrive/modeld/models/commonmodel.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include "common/clutil.h"

ModelFrame::ModelFrame(cl_device_id device_id, cl_context context) {
  input_frames = std::make_unique<uint8_t[]>(buf_size);
  input_frames_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err));

  q = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &err));
  y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, MODEL_WIDTH * MODEL_HEIGHT, NULL, &err));
  u_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
  v_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));

  cl_buffer_region region;
  region.origin = buf_size / 2;  // half of the buffer, in bytes
  region.size = buf_size / 2;  // the second half of the buffer, in bytes
  net_input_cl = CL_CHECK_ERR(clCreateSubBuffer(input_frames_cl, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err));

  transform_init(&transform, context, device_id);
  loadyuv_init(&loadyuv, context, device_id, MODEL_WIDTH, MODEL_HEIGHT);
}

cl_mem* ModelFrame::prepare(cl_mem yuv_cl, int frame_width, int frame_height, int frame_stride, int frame_uv_offset, const mat3 &projection, cl_mem *output) {
  transform_queue(&this->transform, q,
                  yuv_cl, frame_width, frame_height, frame_stride, frame_uv_offset,
                  y_cl, u_cl, v_cl, MODEL_WIDTH, MODEL_HEIGHT, projection);
 CL_CHECK(clEnqueueCopyBuffer(q, input_frames_cl, input_frames_cl, MODEL_FRAME_SIZE * sizeof(uint8_t), 0, MODEL_FRAME_SIZE * sizeof(uint8_t), 0, nullptr, nullptr));
  loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, net_input_cl);

  // NOTE: Since thneed is using a different command queue, this clFinish is needed to ensure the image is ready.
  clFinish(q);
  return &input_frames_cl;
}

uint8_t* ModelFrame::buffer_from_cl(cl_mem *in_frames) {
  CL_CHECK(clEnqueueReadBuffer(q, *in_frames, CL_TRUE, 0, MODEL_FRAME_SIZE * 2 * sizeof(uint8_t), &input_frames[0], 0, nullptr, nullptr));
  clFinish(q);
  return &input_frames[0];
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