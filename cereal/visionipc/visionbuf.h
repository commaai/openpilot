#pragma once
#include "visionipc.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define VISIONBUF_SYNC_FROM_DEVICE 0
#define VISIONBUF_SYNC_TO_DEVICE 1

enum VisionStreamType {
  VISION_STREAM_RGB_BACK,
  VISION_STREAM_RGB_FRONT,
  VISION_STREAM_RGB_WIDE,
  VISION_STREAM_YUV_BACK,
  VISION_STREAM_YUV_FRONT,
  VISION_STREAM_YUV_WIDE,
  VISION_STREAM_MAX,
};

class VisionBuf {
 public:
  size_t len = 0;
  size_t mmap_len = 0;
  void * addr = nullptr;
  int fd = 0;

  bool rgb = false;
  size_t width = 0;
  size_t height = 0;
  size_t stride = 0;

  // YUV
  uint8_t * y = nullptr;
  uint8_t * u = nullptr;
  uint8_t * v = nullptr;

  // Visionipc
  uint64_t server_id = 0;
  size_t idx = 0;
  VisionStreamType type;

  // OpenCL
  cl_mem buf_cl = nullptr;
  cl_command_queue copy_q = nullptr;

  // ion
  int handle = 0;

  void allocate(size_t len);
  void import();
  void init_cl(cl_device_id device_id, cl_context ctx);
  void init_rgb(size_t width, size_t height, size_t stride);
  void init_yuv(size_t width, size_t height);
  int sync(int dir);
  int free();
};

void visionbuf_compute_aligned_width_and_height(int width, int height, int *aligned_w, int *aligned_h);
