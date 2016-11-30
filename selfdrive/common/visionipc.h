#ifndef VISIONIPC_H
#define VISIONIPC_H

#define VIPC_SOCKET_PATH "/tmp/vision_socket"
#define VIPC_MAX_FDS 64


#define VISION_INVALID 0
#define VISION_UI_SUBSCRIBE 1
#define VISION_UI_BUFS 2
#define VISION_UI_ACQUIRE 3
#define VISION_UI_RELEASE 4

typedef struct VisionUIBufs {
  int width, height, stride;
  int front_width, front_height, front_stride;

  int big_box_x, big_box_y;
  int big_box_width, big_box_height;
  int transformed_width, transformed_height;

  int front_box_x, front_box_y;
  int front_box_width, front_box_height;

  size_t buf_len;
  int num_bufs;
  size_t front_buf_len;
  int num_front_bufs;
} VisionUIBufs;

typedef union VisionPacketData {
  VisionUIBufs ui_bufs;
  struct {
    bool front;
    int idx;
  } ui_acq, ui_rel;
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

#endif
