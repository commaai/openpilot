#pragma once

#include "msgq/visionipc/visionipc.h"

#define VISIONBUF_SYNC_FROM_DEVICE 0
#define VISIONBUF_SYNC_TO_DEVICE 1

// Stream ids are opaque to visionipc. Producers/consumers pick an id when
// creating a stream; the sentinel below is reserved to request the list of
// available streams from a server.
typedef uint32_t VisionStreamType;
constexpr VisionStreamType VISION_STREAM_LIST = 0xffffffff;
constexpr size_t VISIONIPC_MAX_STREAMS = 64;

class VisionBuf {
 public:
  size_t len = 0;
  size_t mmap_len = 0;
  void * addr = nullptr;
  uint64_t *frame_id;
  int fd = 0;

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

  // ion
  int handle = 0;

  void allocate(size_t len);
  void import();
  void init_yuv(size_t width, size_t height, size_t stride, size_t uv_offset);
  int sync(int dir);
  int free();

  void set_frame_id(uint64_t id);
  uint64_t get_frame_id();
};
