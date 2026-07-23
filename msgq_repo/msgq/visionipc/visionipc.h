#pragma once

#include <cstdint>
#include <cstddef>


int ipc_connect(const char* socket_path);
int ipc_bind(const char* socket_path);
int ipc_sendrecv_with_fds(bool send, int fd, void *buf, size_t buf_size, int* fds, int num_fds,
                          int *out_num_fds);

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
