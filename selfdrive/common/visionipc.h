#ifndef VISIONIPC_H
#define VISIONIPC_H

#define VIPC_SOCKET_PATH "/tmp/vision_socket"
#define VIPC_MAX_FDS 64


typedef enum VisionIPCPacketType {
  VIPC_INVALID = 0,
  VIPC_STREAM_SUBSCRIBE,
  VIPC_STREAM_BUFS,
  VIPC_STREAM_ACQUIRE,
  VIPC_STREAM_RELEASE,
} VisionIPCPacketType;

typedef enum VisionStreamType {
  VISION_STREAM_UI_BACK,
  VISION_STREAM_UI_FRONT,
  VISION_STREAM_YUV,
  VISION_STREAM_MAX,
} VisionStreamType;

typedef struct VisionUIInfo {
  int big_box_x, big_box_y;
  int big_box_width, big_box_height;
  int transformed_width, transformed_height;

  int front_box_x, front_box_y;
  int front_box_width, front_box_height;
} VisionUIInfo;

typedef struct VisionStreamBufs {
  VisionStreamType type;

  int width, height, stride;
  size_t buf_len;

  union {
    VisionUIInfo ui_info;
  } buf_info;
} VisionStreamBufs;

typedef union VisionPacketData {
  struct {
    VisionStreamType type;
  } stream_sub;
  VisionStreamBufs stream_bufs;
  struct {
    VisionStreamType type;
    int idx;
  } stream_acq, stream_rel;
} VisionPacketData;

typedef struct VisionPacket {
  int type;
  VisionPacketData d;
  int num_fds;
  int fds[VIPC_MAX_FDS];
} VisionPacket;

int vipc_connect();
int vipc_recv(int fd, VisionPacket *out_p);
int vipc_send(int fd, const VisionPacket p);
int vipc_send_p(int fd, const VisionPacket *p); // for cffi

typedef struct VisionBuf {
  int fd;
  size_t len;
  void* addr;
} VisionBuf;
void visionbufs_load(VisionBuf *bufs, const VisionStreamBufs stream_bufs,
                     int num_fds, const int* fds);
void visionbufs_load_p(VisionBuf *bufs, const VisionStreamBufs *stream_bufs,
                     int num_fds, const int* fds); // for cffi

#endif
