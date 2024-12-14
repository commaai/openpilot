#pragma once

#include <cfloat>
#include <cstdlib>
#include <cassert>

#include <memory>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "common/mat.h"
#include "selfdrive/modeld/transforms/loadyuv.h"
#include "selfdrive/modeld/transforms/transform.h"

class ModelFrame {
public:
  ModelFrame(cl_device_id device_id, cl_context context) {
    init_transform(device_id, context);
    q = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &err));
  }
  virtual ~ModelFrame() {}
  virtual cl_mem* prepare(cl_mem yuv_cl, int frame_width, int frame_height, int frame_stride, int frame_uv_offset, const mat3& projection) { return NULL; }
  virtual uint8_t* buffer_from_cl(cl_mem *in_frames) { return NULL; }

  int MODEL_WIDTH;
  int MODEL_HEIGHT;
  int MODEL_FRAME_SIZE;

protected:
  cl_mem y_cl, u_cl, v_cl;
  Transform transform;
  cl_command_queue q;

  void init_transform(cl_device_id device_id, cl_context context) {
    y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, MODEL_WIDTH * MODEL_HEIGHT, NULL, &err));
    u_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
    v_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
    transform_init(&transform, context, device_id);
  }

  void deinit_transform() {
    transform_destroy(&transform);
    CL_CHECK(clReleaseMemObject(v_cl));
    CL_CHECK(clReleaseMemObject(u_cl));
    CL_CHECK(clReleaseMemObject(y_cl));
  }

  void run_transform(cl_mem yuv_cl, int frame_width, int frame_height, int frame_stride, int frame_uv_offset, const mat3& projection) {
    transform_queue(&transform, q,
        yuv_cl, frame_width, frame_height, frame_stride, frame_uv_offset,
        y_cl, u_cl, v_cl, MODEL_WIDTH, MODEL_HEIGHT, projection);
  }
};

class DrivingModelFrame : public ModelFrame {
public:
  DrivingModelFrame(cl_device_id device_id, cl_context context);
  ~DrivingModelFrame();
  cl_mem* prepare(cl_mem yuv_cl, int frame_width, int frame_height, int frame_stride, int frame_uv_offset, const mat3& projection);
  uint8_t* buffer_from_cl(cl_mem *in_frames);

  const int MODEL_WIDTH = 512;
  const int MODEL_HEIGHT = 256;
  const int MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 / 2;
  const int buf_size = MODEL_FRAME_SIZE * 2;
  const size_t frame_size_bytes = MODEL_FRAME_SIZE * sizeof(uint8_t);

private:
  LoadYUVState loadyuv;
  cl_mem img_buffer_20hz_cl, last_img_cl, input_frames_cl;
  cl_buffer_region region;
  std::unique_ptr<uint8_t[]> input_frames;
};

class MonitoringModelFrame : public ModelFrame {
public:
  MonitoringModelFrame(cl_device_id device_id, cl_context context);
  ~MonitoringModelFrame();
  cl_mem* prepare(cl_mem yuv_cl, int frame_width, int frame_height, int frame_stride, int frame_uv_offset, const mat3& projection);
  uint8_t* buffer_from_cl(cl_mem *in_frames);

  const int MODEL_WIDTH = 1440;
  const int MODEL_HEIGHT = 960;
  const int MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT;

private:
  cl_mem input_frame_cl;
  std::unique_ptr<uint8_t[]> input_frame;
};
