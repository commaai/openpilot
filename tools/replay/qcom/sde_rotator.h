#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <poll.h>
#include "common/util.h"
#include "common/swaglog.h"
#include "msgq/visionipc/visionbuf.h"

#define ROTATOR_DEVICE "/dev/video2"


class SdeRotator {
public:
  SdeRotator();
  ~SdeRotator();
  int configUBWCtoNV12(int width, int height);
  int configUBWCtoNV12WithOutputBuf(int width, int height, VisionBuf *output_buf);
  int putFrame(VisionBuf *ubwc);
  int putFrameWithOutputBuf(VisionBuf *ubwc, VisionBuf *output_buf);
  VisionBuf* getFrame(int timeout_ms);
  VisionBuf* getFrameExternal(int timeout_ms);
  bool queued = false;
  bool init(const char *dev = ROTATOR_DEVICE);
  int cleanup();
  int saveFrame(const char* filename, bool append);
  int convert_ubwc_to_linear(int out_buf_fd,
                          int width, int height,
                          unsigned char **linear_data, size_t *linear_size);
  void convertStride(VisionBuf *rotated_buf, VisionBuf *user_buf);


private:
  int fd;
  struct v4l2_format fmt_cap = {0}, fmt_out = {0};
  VisionBuf cap_buf;
  struct pollfd pfd;
  struct v4l2_buffer cap_desc{};       // cached QUERYBUF result
  VisionBuf *external_output_buf;
};