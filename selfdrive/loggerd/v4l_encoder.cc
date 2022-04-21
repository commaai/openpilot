#include <cassert>
#include <sys/ioctl.h>

#include "selfdrive/loggerd/v4l_encoder.h"
#include "selfdrive/common/util.h"

#include "libyuv.h"
#include "selfdrive/loggerd/include/msm_media_info.h"

// echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level

void V4LEncoder::dequeue_in_handler(V4LEncoder *e) {
  bool exit = false;
  while (!exit) {
    if (e->buffer_queued == e->buffer_in) { usleep(10*1000); continue; }
    int ret = e->dequeue_buffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, e->buffer_queued, &e->buf_in[e->buffer_queued], NULL);
    if (ret == -1) { usleep(10*1000); continue; }
    e->buffer_queued = (e->buffer_queued + 1) % 7;
  }
}

void V4LEncoder::dequeue_out_handler(V4LEncoder *e) {
  bool exit = false;
  while (!exit) {
    unsigned int bytesused;
    int ret = e->dequeue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, e->buffer_out, &e->buf_out[e->buffer_out], &bytesused);
    if (ret == -1) { usleep(10*1000); continue; }

    printf("got %d bytes\n", bytesused);

    // TODO: process

    e->queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, e->buffer_out, &e->buf_out[e->buffer_out]);
    e->buffer_out = (e->buffer_out + 1) % 6;
  }
}

V4LEncoder::V4LEncoder(
  const char* filename, CameraType type, int in_width, int in_height,
  int fps, int bitrate, bool h265, int out_width, int out_height, bool write)
  : type(type), in_width_(in_width), in_height_(in_height),
    width(out_width), height(out_height), fps(fps),
    filename(filename), remuxing(!h265), write(write) {
  fd = open("/dev/video33", O_RDWR);
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
        // TODO: more stuff here? we don't know
        .timeperframe { .numerator = 1,
          .denominator = 20
        }
      }
    }
  };
  assert(ioctl(fd, VIDIOC_S_PARM, &streamparm) == 0);

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

  LOGD("in buffer size %d, out buffer size %d",
    fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage,
    fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);

  struct v4l2_control ctrls[] = {
    { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = 10000000},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL, .value = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = 29},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = 0},
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD, .value = 1},
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
  for (int i = 0; i < 7; i++) {
    buf_in[i].allocate(fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage);
  }

  struct v4l2_requestbuffers reqbuf_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .memory = V4L2_MEMORY_USERPTR,
    .count = 6
  };
  assert(ioctl(fd, VIDIOC_REQBUFS, &reqbuf_out) == 0);
  for (int i = 0; i < 6; i++) {
    buf_out[i].allocate(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);
  }

  // publish
  service_name = this->type == DriverCam ? "driverEncodeData" :
    (this->type == WideRoadCam ? "wideRoadEncodeData" :
    (this->remuxing ? "qRoadEncodeData" : "roadEncodeData"));
  pm.reset(new PubMaster({service_name}));
}

int V4LEncoder::dequeue_buffer(v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, unsigned int *bytesused) {
  v4l2_plane plane = {
    .bytesused = 0,
    .length = (unsigned int)buf->len,
    .m {
      .userptr = (unsigned long)buf->addr,
    },
    .reserved = {(unsigned int)buf->fd}
  };

  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .index = index,
    .memory = V4L2_MEMORY_USERPTR,
    .m {
      .planes = &plane,
    },
    .length = 1,
    .bytesused = 0,
    .flags = V4L2_BUF_FLAG_QUEUED|V4L2_BUF_FLAG_TIMESTAMP_COPY
  };
  int ret = ioctl(fd, VIDIOC_DQBUF, &v4l_buf);
  if (ret == -1 && errno == 11) return ret;
  
  //printf("dequeue %d buffer %d: %d errno %d\n", buf_type, index, ret, ret==0 ? 0 : errno);
  if (bytesused) *bytesused = v4l_buf.m.planes[0].bytesused;

  return ret;
}

int V4LEncoder::queue_buffer(v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf) {
  v4l2_plane plane = {
    .length = (unsigned int)buf->len,
    .m {
      .userptr = (unsigned long)buf->addr,
    },
    .reserved = {(unsigned int)buf->fd}
  };

  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .index = index,
    .memory = V4L2_MEMORY_USERPTR,
    .m {
      .planes = &plane,
    },
    .length = 1,
    .bytesused = 0,
    .flags = V4L2_BUF_FLAG_QUEUED|V4L2_BUF_FLAG_TIMESTAMP_COPY
  };

  int ret = ioctl(fd, VIDIOC_QBUF, &v4l_buf);
  //printf("queue %d buffer %d: %d errno %d\n", buf_type, index, ret, ret==0 ? 0 : errno);
  return ret;
}

void V4LEncoder::encoder_open(const char* path) {
  if (this->write) {
    writer.reset(new VideoWriter(path, this->filename, this->remuxing, this->width, this->height, this->fps, !this->remuxing, false));
  }
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  assert(ioctl(fd, VIDIOC_STREAMON, &buf_type) == 0);

  for (unsigned int i = 0; i < 6; i++) {
    queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, &buf_out[i]);
  }

  buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  assert(ioctl(fd, VIDIOC_STREAMON, &buf_type) == 0);

  dequeue_out_thread = std::thread(V4LEncoder::dequeue_out_handler, this);
  dequeue_in_thread = std::thread(V4LEncoder::dequeue_in_handler, this);
  this->is_open = true;
}

int V4LEncoder::encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                  int in_width, int in_height, uint64_t ts) {
  assert(in_width == in_width_);
  assert(in_height == in_height_);

  uint8_t *in_y_ptr = (uint8_t*)buf_in[buffer_in].addr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, this->width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, this->width);
  uint8_t *in_uv_ptr = in_y_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, this->height));

  // GRRR COPY
  int err = libyuv::I420ToNV12(y_ptr, this->width,
                   u_ptr, this->width/2,
                   v_ptr, this->width/2,
                   in_y_ptr, in_y_stride,
                   in_uv_ptr, in_uv_stride,
                   this->width, this->height);
  assert(err == 0);

  // TODO: detect wraparound and block
  buf_in[buffer_in].sync(VISIONBUF_SYNC_TO_DEVICE);
  int ret = queue_buffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buffer_in, &buf_in[buffer_in]);
  assert(ret == 0);
  buffer_in = (buffer_in + 1) % 7;

  return 0;
}

void V4LEncoder::encoder_close() {
  if (this->is_open) {
    v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    assert(ioctl(fd, VIDIOC_STREAMOFF, &buf_type) == 0);
    buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    assert(ioctl(fd, VIDIOC_STREAMOFF, &buf_type) == 0);
    writer.reset();
  }
  this->is_open = false;
}

V4LEncoder::~V4LEncoder() {
  close(fd);
}
