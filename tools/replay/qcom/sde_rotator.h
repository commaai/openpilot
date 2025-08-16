#pragma once

#include <linux/videodev2.h>
#include <poll.h>

#include "msgq/visionipc/visionbuf.h"

#ifndef V4L2_PIX_FMT_NV12_UBWC
#define V4L2_PIX_FMT_NV12_UBWC v4l2_fourcc('Q', '1', '2', '8')
#endif

#define ROTATOR_DEVICE "/dev/video2"

class SdeRotator {
public:
  SdeRotator() = default;
  ~SdeRotator();
  bool init(const char *dev = ROTATOR_DEVICE);
  int configUBWCtoNV12(int width, int height);
  int putFrame(VisionBuf *ubwc);
  VisionBuf* getFrame(int timeout_ms);
  void convertStride(VisionBuf *rotated_buf, VisionBuf *user_buf);
  bool queued = false;

private:
  int fd;
  struct v4l2_format fmt_cap = {0}, fmt_out = {0};
  VisionBuf cap_buf;
  struct pollfd pfd;
  struct v4l2_buffer cap_desc{};  // cached QUERYBUF result
};
