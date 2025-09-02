#include "sde_rotator.h"

#include <assert.h>
#include <cstdio>
#include <errno.h>
#include <fcntl.h>
#include <linux/ion.h>
#include <msm_ion.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "common/util.h"
#include "common/swaglog.h"
#include "third_party/linux/include/msm_media_info.h"


static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR
  };
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
}

bool SdeRotator::init(const char *dev) {
  LOGD("Initializing sde_rot device %s", dev);
  fd = open(dev, O_RDWR, 0);
  if (fd < 0) {
    LOGE("failed to open rotator device");
    return false;
  }
  fmt_cap = {}, fmt_out = {}, cap_desc = {};
  pfd = { .fd = fd, .events = POLLIN | POLLRDNORM, .revents = 0 };
  return true;
}

SdeRotator::~SdeRotator() {
  if (fd >= 0) {
    close(fd);
    fd = -1;
  }
  cap_buf.free();
  cap_buf.~VisionBuf();
  new (&cap_buf) VisionBuf();
  queued = false;
}

int SdeRotator::configUBWCtoNV12(int width, int height) {
  // stop streaming if already started
  enum v4l2_buf_type t;
  t = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  util::safe_ioctl(fd, VIDIOC_STREAMOFF, &t, "VIDIOC_STREAMOFF OUTPUT failed");
  t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  util::safe_ioctl(fd, VIDIOC_STREAMOFF, &t, "VIDIOC_STREAMOFF CAPTURE failed");
  LOGD("Configuring rotator for width=%d height=%d", width, height);
  queued = false;
  fmt_out.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt_out.fmt.pix.width       = width;
  fmt_out.fmt.pix.height      = height;
  fmt_out.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12_UBWC;
  fmt_out.fmt.pix.field       = V4L2_FIELD_NONE;
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_out, "VIDIOC_S_FMT OUTPUT failed");

  fmt_cap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt_cap.fmt.pix.width       = width;
  fmt_cap.fmt.pix.height      = height;
  fmt_cap.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
  fmt_cap.fmt.pix.field       = V4L2_FIELD_NONE;
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_cap, "VIDIOC_S_FMT CAPTURE failed");

  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT, 1);
  if (cap_buf.fd >= 0) {
    munmap(cap_buf.addr, cap_buf.len);
    cap_buf.free();
  }
  cap_buf = VisionBuf();
  cap_buf.allocate(fmt_cap.fmt.pix.sizeimage);
  cap_buf.addr = mmap(nullptr,
                      fmt_cap.fmt.pix.sizeimage,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED,
                      cap_buf.fd,
                      0);
  assert(cap_buf.addr != MAP_FAILED);
  cap_buf.init_yuv(fmt_cap.fmt.pix.width, fmt_cap.fmt.pix.height,
                    fmt_cap.fmt.pix.bytesperline, fmt_cap.fmt.pix.bytesperline * fmt_cap.fmt.pix.height);

  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, 1);
  memset(&cap_desc, 0, sizeof(cap_desc));
  cap_desc.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  cap_desc.memory = V4L2_MEMORY_USERPTR;
  cap_desc.index  = 0;
  util::safe_ioctl(fd, VIDIOC_QUERYBUF, &cap_desc, "VIDIOC_QUERYBUF CAPTURE failed");

  t = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &t, "VIDIOC_STREAMON OUTPUT failed");
  t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &t, "VIDIOC_STREAMON CAPTURE failed");

  return 0;
}

int SdeRotator::putFrame(VisionBuf *ubwc)
{
  if (ubwc->width != cap_buf.width || ubwc->height != cap_buf.height)
    configUBWCtoNV12(ubwc->width, ubwc->height);
  /* OUTPUT (UBWC) */
  struct v4l2_buffer out = {};
  out.type      = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  out.memory    = V4L2_MEMORY_USERPTR;
  out.index     = 0;
  out.m.userptr = static_cast<unsigned long>(ubwc->fd);
  out.length    = fmt_out.fmt.pix.sizeimage;
  util::safe_ioctl(fd, VIDIOC_QBUF, &out, "VIDIOC_QBUF OUTPUT failed");
  /* CAPTURE (linear NV12) – use previously cached cap_desc */
  struct v4l2_buffer cap = cap_desc;
  cap.m.userptr = static_cast<unsigned long>(cap_buf.fd);
  util::safe_ioctl(fd, VIDIOC_QBUF, &cap, "VIDIOC_QBUF CAPTURE failed");

  queued = true;
  return 0;
}

VisionBuf* SdeRotator::getFrame(int timeout_ms) {
  if (poll(&pfd, 1, timeout_ms) <= 0 || !queued) { // timeout or in flight
    return nullptr;
  }
  /* dequeue CAPTURE */
  struct v4l2_buffer cap = {};
  cap.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  cap.memory = V4L2_MEMORY_USERPTR;
  util::safe_ioctl(fd, VIDIOC_DQBUF, &cap, "VIDIOC_DQBUF CAPTURE failed");
  /* dequeue OUTPUT (frees the slot) */
  struct v4l2_buffer out = {};
  out.type   = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  out.memory = V4L2_MEMORY_USERPTR;
  util::safe_ioctl(fd, VIDIOC_DQBUF, &out, "VIDIOC_DQBUF OUTPUT failed");

  queued = false;
  return &cap_buf;
}

void SdeRotator::convertStride(VisionBuf *rotated_buf, VisionBuf *user_buf) {
  // Copy Y plane row by row
  for (int y = 0; y < user_buf->height; y++) {
    uint8_t *src_row = (uint8_t*)rotated_buf->y + y * rotated_buf->stride;
    uint8_t *dst_row = (uint8_t*)user_buf->y + y * user_buf->stride;
    memcpy(dst_row, src_row, user_buf->width);
  }
  // Copy UV plane row by row (NV12: height/2)
  for (int y = 0; y < user_buf->height / 2; y++) {
    uint8_t *src_row = (uint8_t*)rotated_buf->uv + y * rotated_buf->stride;
    uint8_t *dst_row = (uint8_t*)user_buf->uv + y * user_buf->stride;
    memcpy(dst_row, src_row, user_buf->width);
  }
}
