#ifndef VISIONIPC_H
#define VISIONIPC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define VIPC_SOCKET_PATH "/tmp/vision_socket"
#define VIPC_MAX_FDS 64

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VisionIPCPacketType {
  VIPC_INVALID = 0,
  VIPC_STREAM_SUBSCRIBE,
  VIPC_STREAM_BUFS,
  VIPC_STREAM_ACQUIRE,
  VIPC_STREAM_RELEASE,
} VisionIPCPacketType;

typedef enum VisionStreamType {
  VISION_STREAM_RGB_BACK,
  VISION_STREAM_RGB_FRONT,
  VISION_STREAM_RGB_WIDE,
  VISION_STREAM_YUV,
  VISION_STREAM_YUV_FRONT,
  VISION_STREAM_YUV_WIDE,
  VISION_STREAM_MAX,
} VisionStreamType;

typedef struct VisionUIInfo {
  int big_box_x, big_box_y;
  int big_box_width, big_box_height;
  int transformed_width, transformed_height;

  int front_box_x, front_box_y;
  int front_box_width, front_box_height;

  int wide_box_x, wide_box_y;
  int wide_box_width, wide_box_height;

} VisionUIInfo;

typedef struct VisionStreamBufs {
  VisionStreamType type;

  int width, height, stride;
  size_t buf_len;

  union {
    VisionUIInfo ui_info;
  } buf_info;
} VisionStreamBufs;

typedef struct VIPCBufExtra {
  // only for yuv
  uint32_t frame_id;
  uint64_t timestamp_eof;
} VIPCBufExtra;

typedef union VisionPacketData {
  struct {
    VisionStreamType type;
    bool tbuffer;
  } stream_sub;
  VisionStreamBufs stream_bufs;
  struct {
    VisionStreamType type;
    int idx;
    VIPCBufExtra extra;
  } stream_acq;
  struct {
    VisionStreamType type;
    int idx;
  } stream_rel;
} VisionPacketData;

typedef struct VisionPacket {
  int type;
  VisionPacketData d;
  int num_fds;
  int fds[VIPC_MAX_FDS];
} VisionPacket;

int vipc_connect(void);
int vipc_recv(int fd, VisionPacket *out_p);
int vipc_send(int fd, const VisionPacket *p);

typedef struct VIPCBuf {
  int fd;
  size_t len;
  void* addr;
} VIPCBuf;
void vipc_bufs_load(VIPCBuf *bufs, const VisionStreamBufs *stream_bufs,
                     int num_fds, const int* fds);



typedef struct VisionStream {
  int ipc_fd;
  int last_idx;
  int last_type;
  int num_bufs;
  VisionStreamBufs bufs_info;
  VIPCBuf *bufs;
} VisionStream;

int visionstream_init(VisionStream *s, VisionStreamType type, bool tbuffer, VisionStreamBufs *out_bufs_info);
void visionstream_release(VisionStream *s);
VIPCBuf* visionstream_get(VisionStream *s, VIPCBufExtra *out_extra);
void visionstream_destroy(VisionStream *s);

#ifdef __cplusplus
}
#endif

#endif
