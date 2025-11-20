#include "selfdrive/modeld/models/commonmodel.h"

#include <cmath>
#include <cstring>

#include "common/clutil.h"


ModelFrame::ModelFrame(cl_device_id device_id, cl_context context) {
  q = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &err));
  full_input_frame = std::make_unique<uint8_t[]>(full_img_size);
  single_frame_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, full_img_size, NULL, &err));
}

ModelFrame::~ModelFrame() {
  CL_CHECK(clReleaseMemObject(single_frame_cl));
  CL_CHECK(clReleaseCommandQueue(q));
}

uint8_t* ModelFrame::array_from_vision_buf(cl_mem *vision_buf) {
  CL_CHECK(clEnqueueReadBuffer(q, *vision_buf, CL_TRUE, 0, full_img_size * sizeof(uint8_t), &full_input_frame[0], 0, nullptr, nullptr));
  clFinish(q);
  return &full_input_frame[0];
}

cl_mem* ModelFrame::cl_from_vision_buf(cl_mem *vision_buf) {
  CL_CHECK(clEnqueueCopyBuffer(q, *vision_buf, single_frame_cl,  0, 0, full_img_size * sizeof(uint8_t), 0, nullptr, nullptr));
  clFinish(q);
  return &single_frame_cl;
}
  
DrivingModelFrame::DrivingModelFrame(cl_device_id device_id, cl_context context, int _temporal_skip) : ModelFrame(device_id, context) {
}

DrivingModelFrame::~DrivingModelFrame() {
}

MonitoringModelFrame::MonitoringModelFrame(cl_device_id device_id, cl_context context) : ModelFrame(device_id, context) {
}


MonitoringModelFrame::~MonitoringModelFrame() {
}
