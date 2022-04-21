#include <cassert>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include "selfdrive/loggerd/v4l_encoder.h"
#include "selfdrive/common/util.h"

V4LEncoder::V4LEncoder(
  const char* filename, CameraType type, int in_width, int in_height,
  int fps, int bitrate, bool h265, int out_width, int out_height, bool write)
  : type(type), in_width_(in_width), in_height_(in_height),
    width(out_width), height(out_height), fps(fps),
    filename(filename), remuxing(!h265), write(write) {
  fd = open("/dev/video33", O_RDWR|O_NONBLOCK);
  assert(fd >= 0);

  struct v4l2_capability cap;
  assert(ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0);
  LOGD("opened encoder device %s %s = %d", cap.driver, cap.card, fd);

  struct v4l2_format fmt_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .fmt {
      .pix_mp = {
        .width = width,
        .height = height,
        .pixelformat = V4L2_PIX_FMT_HEVC,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_DEFAULT,
      }
    }
  };
  assert(ioctl(fd, VIDIOC_S_FMT, &fmt_out) == 0);

  v4l2_streamparm streamparm = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .parm {
      .output {
        // TODO: frame time and stuff
      }
    }
  };
  assert(ioctl(fd, VIDIOC_S_PARM, &streamparm));

  struct v4l2_format fmt_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .fmt {
      .pix_mp = {
        .width = width,
        .height = height,
        .pixelformat = V4L2_PIX_FMT_NV12,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_470_SYSTEM_BG,
      }
    }
  };
  assert(ioctl(fd, VIDIOC_S_FMT, &fmt_in) == 0);

  struct v4l2_control ctrls[] = {
    { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = 10000000},
    // TODO: find what these are
    { .id = V4L2_CTRL_CLASS_MPEG+0x2009, .value = 2},
    { .id = V4L2_CTRL_CLASS_MPEG+0x2028, .value = 0},
    { .id = V4L2_CTRL_CLASS_MPEG+0x2029, .value = 15},
    { .id = V4L2_CTRL_CLASS_MPEG+0x2034, .value = 1},
    { .id = V4L2_CTRL_CLASS_MPEG+0x2006, .value = 29},
    { .id = V4L2_CTRL_CLASS_MPEG+0x2007, .value = 0},
    { .id = V4L2_CTRL_CLASS_MPEG+0x2005, .value = 1},
  };

  for (auto ctrl : ctrls) {
    assert(ioctl(fd, VIDIOC_S_CTRL, &ctrl) == 0);
  }

  struct v4l2_requestbuffers reqbuf_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .memory = V4L2_MEMORY_USERPTR,
    .count = 7
  };
  assert(ioctl(fd, VIDIOC_REQBUFS, &reqbuf_in) == 0);

  struct v4l2_requestbuffers reqbuf_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .memory = V4L2_MEMORY_USERPTR,
    .count = 6
  };
  assert(ioctl(fd, VIDIOC_REQBUFS, &reqbuf_out) == 0);

  // publish
  service_name = this->type == DriverCam ? "driverEncodeData" :
    (this->type == WideRoadCam ? "wideRoadEncodeData" :
    (this->remuxing ? "qRoadEncodeData" : "roadEncodeData"));
  pm.reset(new PubMaster({service_name}));
}

void V4LEncoder::encoder_open(const char* path) {
  if (this->write) {
    writer.reset(new VideoWriter(path, this->filename, this->remuxing, this->width, this->height, this->fps, !this->remuxing, false));
  }
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  assert(ioctl(fd, VIDIOC_STREAMON, &buf_type) == 0);

  // TODO: queue up 6 V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
}

int V4LEncoder::encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                  int in_width, int in_height, uint64_t ts) {
  assert(in_width == in_width_);
  assert(in_height == in_height_);

  return 0;
}

void V4LEncoder::encoder_close() {
  /*v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  assert(ioctl(fd, VIDIOC_STREAMOFF, &buf_type));
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  assert(ioctl(fd, VIDIOC_STREAMOFF, &buf_type));
  writer.reset();*/
}

V4LEncoder::~V4LEncoder() {
  close(fd);
}