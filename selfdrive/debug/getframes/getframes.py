#!/usr/bin/env python
import os
import subprocess
from cffi import FFI

import numpy as np

gf_dir = os.path.dirname(os.path.abspath(__file__))

subprocess.check_call(["make"], cwd=gf_dir)


ffi = FFI()
ffi.cdef("""

typedef enum VisionStreamType {
  VISION_STREAM_RGB_BACK,
  VISION_STREAM_RGB_FRONT,
  VISION_STREAM_YUV,
  VISION_STREAM_YUV_FRONT,
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

typedef struct VIPCBuf {
  int fd;
  size_t len;
  void* addr;
} VIPCBuf;

typedef struct VIPCBufExtra {
  // only for yuv
  uint32_t frame_id;
  uint64_t timestamp_eof;
} VIPCBufExtra;

typedef struct VisionStream {
  int ipc_fd;
  int last_idx;
  int last_type;
  int num_bufs;
  VisionStreamBufs bufs_info;
  VIPCBuf *bufs;
} VisionStream;

int visionstream_init(VisionStream *s, VisionStreamType type, bool tbuffer, VisionStreamBufs *out_bufs_info);
VIPCBuf* visionstream_get(VisionStream *s, VIPCBufExtra *out_extra);
void visionstream_destroy(VisionStream *s);

"""
)

clib = ffi.dlopen(os.path.join(gf_dir, "libvisionipc.so"))


def getframes(front=False):
  s = ffi.new("VisionStream*")
  buf_info = ffi.new("VisionStreamBufs*")

  if front:
   stream_type = clib.VISION_STREAM_RGB_FRONT
  else:
   stream_type = clib.VISION_STREAM_RGB_BACK

  err = clib.visionstream_init(s, stream_type, True, buf_info)
  assert err == 0

  w = buf_info.width
  h = buf_info.height
  assert buf_info.stride == w*3
  assert buf_info.buf_len == w*h*3

  while True:
    buf = clib.visionstream_get(s, ffi.NULL)

    pbuf = ffi.buffer(buf.addr, buf.len)
    yield np.frombuffer(pbuf, dtype=np.uint8).reshape((h, w, 3))


if __name__ == "__main__":
  for buf in getframes():
    print buf.shape, buf[101, 101]
