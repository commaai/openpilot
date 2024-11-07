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
  img_buffer_20hz_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 5*frame_size_bytes, NULL, &err));
  region.origin = 4 * frame_size_bytes;
  region.size = frame_size_bytes;
  last_img_cl = CL_CHECK_ERR(clCreateSubBuffer(img_buffer_20hz_cl, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err));

  transform_init(&transform, context, device_id);
  loadyuv_init(&loadyuv, context, device_id, MODEL_WIDTH, MODEL_HEIGHT);
}

cl_mem* ModelFrame::prepare(cl_mem yuv_cl, int frame_width, int frame_height, int frame_stride, int frame_uv_offset, const mat3 &projection) {
  transform_queue(&this->transform, q,
                yuv_cl, frame_width, frame_height, frame_stride, frame_uv_offset,
                y_cl, u_cl, v_cl, MODEL_WIDTH, MODEL_HEIGHT, projection);

  for (int i = 0; i < 4; i++) {
    CL_CHECK(clEnqueueCopyBuffer(q, img_buffer_20hz_cl, img_buffer_20hz_cl, (i+1)*frame_size_bytes, i*frame_size_bytes, frame_size_bytes, 0, nullptr, nullptr));
  }
  loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, last_img_cl);

  copy_queue(&loadyuv, q, img_buffer_20hz_cl, input_frames_cl, 0, 0, frame_size_bytes);
  copy_queue(&loadyuv, q, last_img_cl, input_frames_cl, 0, frame_size_bytes, frame_size_bytes);

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
  CL_CHECK(clReleaseMemObject(img_buffer_20hz_cl));
  CL_CHECK(clReleaseMemObject(last_img_cl));
  CL_CHECK(clReleaseMemObject(v_cl));
  CL_CHECK(clReleaseMemObject(u_cl));
  CL_CHECK(clReleaseMemObject(y_cl));
  CL_CHECK(clReleaseCommandQueue(q));
}