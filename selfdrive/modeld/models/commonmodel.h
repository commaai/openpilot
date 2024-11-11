#pragma once

#include <cfloat>
#include <cstdlib>

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
  ModelFrame(cl_device_id device_id, cl_context context);
  ~ModelFrame();
  cl_mem* prepare(cl_mem yuv_cl, int width, int height, int frame_stride, int frame_uv_offset, const mat3& transform);
  uint8_t* buffer_from_cl(cl_mem *in_frames);

  const int MODEL_WIDTH = 512;
  const int MODEL_HEIGHT = 256;
  const int MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 / 2;
  const int buf_size = MODEL_FRAME_SIZE * 2;
  const size_t frame_size_bytes = MODEL_FRAME_SIZE * sizeof(uint8_t);

private:
  Transform transform;
  LoadYUVState loadyuv;
  cl_command_queue q;
  cl_mem y_cl, u_cl, v_cl, img_buffer_20hz_cl, last_img_cl, input_frames_cl;
  cl_buffer_region region;
  std::unique_ptr<uint8_t[]> input_frames;
};
