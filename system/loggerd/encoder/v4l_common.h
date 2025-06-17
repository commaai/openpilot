#pragma once

#include <cerrno>
#include <cassert>
#include <common/util.h>
#include <common/swaglog.h>
#include <msgq/visionipc/visionbuf.h>

// has to be in this order
#include "third_party/linux/include/v4l2-controls.h"
#include <linux/videodev2.h>
#define V4L2_QCOM_BUF_FLAG_CODECCONFIG 0x00020000
#define V4L2_QCOM_BUF_FLAG_EOS 0x02000000


static void checked_ioctl(int fd, unsigned long request, void *argp) {
  int ret = util::safe_ioctl(fd, request, argp);
  if (ret != 0) {
    LOGE("checked_ioctl failed with error %d (%d %lx %p)", errno, fd, request, argp);
    assert(0);
  }
}

static void dequeue_buffer(int fd, v4l2_buf_type buf_type, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL) {
  v4l2_plane plane = {0};
  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
    .m = { .planes = &plane, },
    .length = 1,
  };
  checked_ioctl(fd, VIDIOC_DQBUF, &v4l_buf);

  if (index) *index = v4l_buf.index;
  if (bytesused) *bytesused = v4l_buf.m.planes[0].bytesused;
  if (flags) *flags = v4l_buf.flags;
  if (timestamp) *timestamp = v4l_buf.timestamp;
  assert(v4l_buf.m.planes[0].data_offset == 0);
}

static void queue_buffer(int fd, v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, struct timeval timestamp={}) {
  v4l2_plane plane = {
    .bytesused = (uint32_t)buf->len,
    .length = (unsigned int)buf->len,
    .m = { .userptr = (unsigned long)buf->addr, },
    .reserved = {(unsigned int)buf->fd}
  };

  v4l2_buffer v4l_buf = {
    .index = index,
    .type = buf_type,
    .flags = V4L2_BUF_FLAG_TIMESTAMP_COPY,
    .timestamp = timestamp,
    .memory = V4L2_MEMORY_USERPTR,
    .m = { .planes = &plane, },
    .length = 1,
  };

  checked_ioctl(fd, VIDIOC_QBUF, &v4l_buf);
}

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR
  };
  checked_ioctl(fd, VIDIOC_REQBUFS, &reqbuf);
}