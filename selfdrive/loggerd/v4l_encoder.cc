#include <cassert>
#include <sys/ioctl.h>
#include <poll.h>

#include "selfdrive/loggerd/v4l_encoder.h"
#include "selfdrive/common/util.h"

#include "libyuv.h"
#include "selfdrive/loggerd/include/msm_media_info.h"

// echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level

#define V4L_BUF_FLAG_FRAME (V4L2_BUF_FLAG_KEYFRAME | V4L2_BUF_FLAG_PFRAME | V4L2_BUF_FLAG_BFRAME)

void V4LEncoder::dequeue_handler(V4LEncoder *e) {
  e->segment_num++;
  uint32_t idx = 0;
  bool exit = false;

  // POLLIN is capture, POLLOUT is frame
  struct pollfd pfd;
  pfd.events = POLLIN | POLLOUT;
  pfd.fd = e->fd;

  while (!exit || e->buffer_in_outstanding > 0) {
    int rc = poll(&pfd, 1, 1000);
    if (!rc) { printf("poll timeout"); continue; }

    /*struct timespec t;
    clock_gettime(CLOCK_BOOTTIME, &t);
    uint64_t current_time = t.tv_sec * 1000000000ULL + t.tv_nsec;
    printf("poll %x at %f ms\n", pfd.revents, current_time/1e6);*/

    if (pfd.revents & POLLIN) {
      unsigned int bytesused, flags, index;
      struct timeval timestamp;
      int ret = e->dequeue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &index, &bytesused, &flags, &timestamp);
      assert(ret == 0);

      // copy out the data and hand buffer back
      e->buf_out[index].sync(VISIONBUF_SYNC_FROM_DEVICE);
      auto dat = kj::heapArray<capnp::byte>((uint8_t*)e->buf_out[index].addr, bytesused);
      ret = e->queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, &e->buf_out[index]);
      assert(ret == 0);

      // eof packet, we exit
      if (!(flags & V4L_BUF_FLAG_FRAME)) exit = true;
      uint64_t ts = timestamp.tv_sec * 1000000 + timestamp.tv_usec;

      // broadcast packet
      MessageBuilder msg;
      auto event = msg.initEvent(true);
      auto edata = (e->type == DriverCam) ? event.initDriverEncodeData() :
        ((e->type == WideRoadCam) ? event.initWideRoadEncodeData() :
        (e->remuxing ? event.initQRoadEncodeData() : event.initRoadEncodeData()));
      edata.setData(dat);
      edata.setTimestampEof(ts);
      edata.setIdx(idx++);
      edata.setSegmentNum(e->segment_num);
      edata.setFlags(flags);
      e->pm->send(e->service_name, msg);

      /*uint64_t current_tm = event.getLogMonoTime() / 1000;
      printf("%20s got %6d bytes in buffer %d with flags %8x and ts %lu lat %.2f ms\n", e->filename, bytesused, index, flags, ts, (current_tm-ts)/1000.);*/

      // TODO: writing should be moved to loggerd
      if (e->write && bytesused != 0) {
        e->writer->write(dat.begin(), bytesused, ts, false, flags & V4L2_BUF_FLAG_KEYFRAME);
      }
    }

    if (pfd.revents & POLLOUT) {
      int ret = e->dequeue_buffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
      assert(ret == 0);
      e->buffer_in_outstanding--;
    }
  }
}

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
  assert(strcmp((const char *)cap.driver, "msm_vidc_driver") == 0);
  assert(strcmp((const char *)cap.card, "msm_vidc_venc") == 0);

  struct v4l2_format fmt_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .fmt {
      .pix_mp = {
        // downscales are free with v4l
        .width = (unsigned int)out_width,
        .height = (unsigned int)out_height,
        .pixelformat = h265 ? V4L2_PIX_FMT_HEVC : V4L2_PIX_FMT_H264,
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
        .width = (unsigned int)in_width,
        .height = (unsigned int)in_height,
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

  if (h265) {
    struct v4l2_control ctrls[] = {
      { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = bitrate},
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
  } else {
    struct v4l2_control ctrls[] = {
      { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = bitrate},
      { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = 15},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, .value = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE, .value = 0},
      // after buffer allocation in OMX. why do P/B frame counts change?
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = 7},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = 1},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD, .value = 1},
    };
    for (auto ctrl : ctrls) {
      assert(ioctl(fd, VIDIOC_S_CTRL, &ctrl) == 0);
    }
  }

  struct v4l2_requestbuffers reqbuf_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .memory = V4L2_MEMORY_USERPTR,
    .count = BUF_OUT_COUNT
  };
  assert(ioctl(fd, VIDIOC_REQBUFS, &reqbuf_out) == 0);
  for (int i = 0; i < BUF_OUT_COUNT; i++) {
    buf_out[i].allocate(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);
  }

  struct v4l2_requestbuffers reqbuf_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .memory = V4L2_MEMORY_USERPTR,
    .count = BUF_IN_COUNT
  };
  assert(ioctl(fd, VIDIOC_REQBUFS, &reqbuf_in) == 0);
  for (int i = 0; i < BUF_IN_COUNT; i++) {
    buf_in[i].allocate(fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage);
  }

  // publish
  service_name = this->type == DriverCam ? "driverEncodeData" :
    (this->type == WideRoadCam ? "wideRoadEncodeData" :
    (this->remuxing ? "qRoadEncodeData" : "roadEncodeData"));
  pm.reset(new PubMaster({service_name}));
}

int V4LEncoder::dequeue_buffer(v4l2_buf_type buf_type, unsigned int *index, unsigned int *bytesused, unsigned int *flags, struct timeval *timestamp) {
  v4l2_plane plane = {0};
  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
    .m {
      .planes = &plane,
    },
    .length = 1,
  };
  int ret = ioctl(fd, VIDIOC_DQBUF, &v4l_buf);
  if (ret == -1 && errno == 11) return ret;

  if (ret != 0 && errno != 11) {
    printf("dequeue %d buffer %d: %d errno %d\n", buf_type, v4l_buf.index, ret, ret==0 ? 0 : errno);
    return ret;
  }
  
  if (index) *index = v4l_buf.index;
  if (bytesused) *bytesused = v4l_buf.m.planes[0].bytesused;
  if (flags) *flags = v4l_buf.flags;
  if (timestamp) *timestamp = v4l_buf.timestamp;
  assert(v4l_buf.m.planes[0].data_offset == 0);

  return ret;
}

int V4LEncoder::queue_buffer(v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, unsigned int bytesused, struct timeval timestamp) {
  v4l2_plane plane = {
    .bytesused = bytesused,
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
    .flags = V4L2_BUF_FLAG_TIMESTAMP_COPY,
    .timestamp = timestamp
  };

  int ret = ioctl(fd, VIDIOC_QBUF, &v4l_buf);
  if (ret != 0) {
    printf("queue %d buffer %d: %d errno %d\n", buf_type, index, ret, ret==0 ? 0 : errno);
  }
  return ret;
}

void V4LEncoder::encoder_open(const char* path) {
  if (this->write) {
    writer.reset(new VideoWriter(path, this->filename, this->remuxing, this->width, this->height, this->fps, !this->remuxing, false));
    // this is "codecconfig", the default seems okay without extradata
    writer->write(NULL, 0, 0, true, false);
  }

  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  assert(ioctl(fd, VIDIOC_STREAMON, &buf_type) == 0);

  for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
    queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, &buf_out[i]);
  }

  buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  assert(ioctl(fd, VIDIOC_STREAMON, &buf_type) == 0);

  dequeue_thread = std::thread(V4LEncoder::dequeue_handler, this);
  this->is_open = true;
}

int V4LEncoder::encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                  int in_width, int in_height, uint64_t ts) {
  assert(in_width == in_width_);
  assert(in_height == in_height_);

  uint8_t *in_y_ptr = (uint8_t*)buf_in[buffer_in].addr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, in_width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, in_width);
  uint8_t *in_uv_ptr = in_y_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, in_height));

  // GRRR COPY
  int err = libyuv::I420ToNV12(y_ptr, in_width,
                   u_ptr, in_width/2,
                   v_ptr, in_width/2,
                   in_y_ptr, in_y_stride,
                   in_uv_ptr, in_uv_stride,
                   in_width, in_height);
  assert(err == 0);

  struct timeval timestamp {
    .tv_sec = (long)(ts/1000000000),
    .tv_usec = (long)((ts/1000) % 1000000),
  };

  // reserve buffer
  assert(buffer_in_outstanding < BUF_IN_COUNT);
  buffer_in_outstanding++;

  // push buffer
  buf_in[buffer_in].sync(VISIONBUF_SYNC_TO_DEVICE);
  int ret = queue_buffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buffer_in, &buf_in[buffer_in],
    VENUS_BUFFER_SIZE(COLOR_FMT_NV12, in_width, in_height),
    timestamp
  );
  assert(ret == 0);
  buffer_in = (buffer_in + 1) % BUF_IN_COUNT;

  return 0;
}

void V4LEncoder::encoder_close() {
  if (this->is_open) {
    {
      v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
      assert(ioctl(fd, VIDIOC_STREAMOFF, &buf_type) == 0);
    }
    dequeue_thread.join();
    {
      v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      assert(ioctl(fd, VIDIOC_STREAMOFF, &buf_type) == 0);
    }
    writer.reset();
  }
  this->is_open = false;
}

V4LEncoder::~V4LEncoder() {
  close(fd);
}
