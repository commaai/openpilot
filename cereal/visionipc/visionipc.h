#pragma once

#include <cstdint>
#include <cstddef>

constexpr int VISIONIPC_MAX_FDS = 128;

struct VisionIpcBufExtra {
  uint32_t frame_id;
  uint64_t timestamp_sof;
  uint64_t timestamp_eof;
  bool valid;
};

struct VisionIpcPacket {
  uint64_t server_id;
  size_t idx;
  struct VisionIpcBufExtra extra;
};
