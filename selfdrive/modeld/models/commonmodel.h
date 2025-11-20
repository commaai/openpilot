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
#include "common/clutil.h"


class ModelFrame {
public:
  ModelFrame(cl_device_id device_id, cl_context context);
  ~ModelFrame();

  int MODEL_WIDTH;
  int MODEL_HEIGHT;
  int MODEL_FRAME_SIZE;
  int buf_size;
  uint8_t* array_from_vision_buf(cl_mem *vision_buf);
  cl_mem* cl_from_vision_buf(cl_mem *vision_buf);

  // DONT HARDCODE THIS
  const int RAW_IMG_HEIGHT = 1208;
  const int RAW_IMG_WIDTH = 1928;
  const int full_img_size = RAW_IMG_HEIGHT * RAW_IMG_WIDTH * 3 / 2;

protected:
  cl_command_queue q;
  cl_mem single_frame_cl;
  std::unique_ptr<uint8_t[]> full_input_frame;
};
