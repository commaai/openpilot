#pragma once

#include "msgq/visionipc/visionipc.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define VISIONBUF_SYNC_FROM_DEVICE 0
#define VISIONBUF_SYNC_TO_DEVICE 1

enum VisionStreamType {
  VISION_STREAM_ROAD,
  VISION_STREAM_DRIVER,
  VISION_STREAM_WIDE_ROAD,

  VISION_STREAM_MAP,
  VISION_STREAM_MAX,
};

class VisionBuf {
 public:
  size_t len = 0;
  size_t mmap_len = 0;
  void * addr = nullptr;
  uint64_t *frame_id;
  int fd = 0;

  bool rgb = false;
  size_t width = 0;
  size_t height = 0;
  size_t stride = 0;
  size_t uv_offset = 0;

  // YUV
  uint8_t * y = nullptr;
  uint8_t * uv = nullptr;

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
  void init_yuv(size_t width, size_t height, size_t stride, size_t uv_offset);
  int sync(int dir);
  int free();

  void set_frame_id(uint64_t id);
  uint64_t get_frame_id();
};

void visionbuf_compute_aligned_width_and_height(int width, int height, int *aligned_w, int *aligned_h);
