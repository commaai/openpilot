#include <cassert>
#include <sys/ioctl.h>
#include <poll.h>

#include "selfdrive/loggerd/v4l_encoder.h"
#include "selfdrive/common/util.h"

#include "libyuv.h"
#include "selfdrive/loggerd/include/msm_media_info.h"

// has to be in this order
#include "selfdrive/loggerd/include/v4l2-controls.h"
#include <linux/videodev2.h>
#define V4L2_QCOM_BUF_FLAG_EOS 0x02000000
// echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level

const bool env_debug_encoder = getenv("DEBUG_ENCODER") != NULL;

#define checked_ioctl(x,y,z) { int _ret = HANDLE_EINTR(ioctl(x,y,z)); assert(_ret==0); }

static int dequeue_buffer(int fd, v4l2_buf_type buf_type, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL) {
  v4l2_plane plane = {0};
  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
    .m = { .planes = &plane, },
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

static int queue_buffer(int fd, v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, struct timeval timestamp={0}) {
  v4l2_plane plane = {
    .length = (unsigned int)buf->len,
    .m = { .userptr = (unsigned long)buf->addr, },
    .reserved = {(unsigned int)buf->fd}
  };

  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .index = index,
    .memory = V4L2_MEMORY_USERPTR,
    .m = { .planes = &plane, },
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

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
    .count = count
  };
  checked_ioctl(fd, VIDIOC_REQBUFS, &reqbuf);
}

// TODO: writing should be moved to loggerd
void V4LEncoder::write_handler(V4LEncoder *e, const char *path) {
  VideoWriter writer(path, e->filename, e->remuxing, e->width, e->height, e->fps, !e->remuxing, false);

  // this is "codecconfig", the default seems okay without extradata
  // TODO: raw_logger does this header write also, refactor to remove from VideoWriter
  writer.write(NULL, 0, 0, true, false);

  while (1) {
    auto out_buf = e->to_write.pop();

    capnp::FlatArrayMessageReader cmsg(*out_buf);
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    auto edata = (e->type == DriverCam) ? event.getDriverEncodeData() :
      ((e->type == WideRoadCam) ? event.getWideRoadEncodeData() :
      (e->remuxing ? event.getQRoadEncodeData() : event.getRoadEncodeData()));
    auto flags = edata.getFlags();
    if (flags & V4L2_QCOM_BUF_FLAG_EOS) break;

    // dangerous cast from const, but should be fine
    auto data = edata.getData();
    writer.write((uint8_t *)data.begin(), data.size(), edata.getTimestampEof(), false, flags & V4L2_BUF_FLAG_KEYFRAME);

    // free the data
    delete out_buf;
  }

  // VideoWriter is freed on out of scope
}

void V4LEncoder::dequeue_handler(V4LEncoder *e) {
  e->segment_num++;
  uint32_t idx = 0;
  bool exit = false;

  // POLLIN is capture, POLLOUT is frame
  struct pollfd pfd;
  pfd.events = POLLIN | POLLOUT;
  pfd.fd = e->fd;

  while (!exit) {
    int rc = poll(&pfd, 1, 1000);
    if (!rc) { printf("poll timeout"); continue; }

    /*struct timespec t;
    clock_gettime(CLOCK_BOOTTIME, &t);
    uint64_t current_time = t.tv_sec * 1000000000ULL + t.tv_nsec;
    printf("poll %x at %f ms\n", pfd.revents, current_time/1e6);*/

    if (pfd.revents & POLLIN) {
      unsigned int bytesused, flags, index;
      struct timeval timestamp;
      int ret = dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &index, &bytesused, &flags, &timestamp);
      assert(ret == 0);
      e->buf_out[index].sync(VISIONBUF_SYNC_FROM_DEVICE);
      uint8_t *buf = (uint8_t*)e->buf_out[index].addr;

      // eof packet, we exit
      if (flags & V4L2_QCOM_BUF_FLAG_EOS) exit = true;
      uint64_t ts = timestamp.tv_sec * 1000000 + timestamp.tv_usec;

      // broadcast packet
      MessageBuilder msg;
      auto event = msg.initEvent(true);
      auto edata = (e->type == DriverCam) ? event.initDriverEncodeData() :
        ((e->type == WideRoadCam) ? event.initWideRoadEncodeData() :
        (e->remuxing ? event.initQRoadEncodeData() : event.initRoadEncodeData()));
      edata.setData(kj::arrayPtr<capnp::byte>(buf, bytesused));
      edata.setTimestampEof(ts);
      edata.setIdx(idx++);
      edata.setSegmentNum(e->segment_num);
      edata.setFlags(flags);

      auto words = new kj::Array<capnp::word>(capnp::messageToFlatArray(msg));
      auto bytes = words->asBytes();
      e->pm->send(e->service_name, bytes.begin(), bytes.size());
      if (e->write) {
        e->to_write.push(words);
      } else {
        delete words;
      }

      if (env_debug_encoder) {
        printf("%20s got %6d bytes in buffer %d with flags %8x and ts %lu lat %.2f ms\n", e->filename, bytesused, index, flags, ts, ((event.getLogMonoTime() / 1000)-ts)/1000.);
      }

      // requeue the buffer
      ret = queue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, &e->buf_out[index]);
      assert(ret == 0);
    }

    if (pfd.revents & POLLOUT) {
      int ret = dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
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
  checked_ioctl(fd, VIDIOC_QUERYCAP, &cap);
  LOGD("opened encoder device %s %s = %d", cap.driver, cap.card, fd);
  assert(strcmp((const char *)cap.driver, "msm_vidc_driver") == 0);
  assert(strcmp((const char *)cap.card, "msm_vidc_venc") == 0);

  struct v4l2_format fmt_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .fmt = {
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
  checked_ioctl(fd, VIDIOC_S_FMT, &fmt_out);

  v4l2_streamparm streamparm = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .parm = {
      .output = {
        // TODO: more stuff here? we don't know
        .timeperframe = {
          .numerator = 1,
          .denominator = 20
        }
      }
    }
  };
  checked_ioctl(fd, VIDIOC_S_PARM, &streamparm);

  struct v4l2_format fmt_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .fmt = {
      .pix_mp = {
        .width = (unsigned int)in_width,
        .height = (unsigned int)in_height,
        .pixelformat = V4L2_PIX_FMT_NV12,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_470_SYSTEM_BG,
      }
    }
  };
  checked_ioctl(fd, VIDIOC_S_FMT, &fmt_in);

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
      checked_ioctl(fd, VIDIOC_S_CTRL, &ctrl);
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
      checked_ioctl(fd, VIDIOC_S_CTRL, &ctrl);
    }
  }

  // allocate buffers
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, BUF_OUT_COUNT);
  for (int i = 0; i < BUF_OUT_COUNT; i++) {
    buf_out[i].allocate(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);
  }

  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, BUF_IN_COUNT);
  for (int i = 0; i < BUF_IN_COUNT; i++) {
    buf_in[i].allocate(fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage);
  }

  // start encoder
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  checked_ioctl(fd, VIDIOC_STREAMON, &buf_type);
  buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  checked_ioctl(fd, VIDIOC_STREAMON, &buf_type);

  // queue up output buffers
  for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
    int ret = queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, &buf_out[i]);
    assert(ret == 0);
  }

  // publish
  service_name = this->type == DriverCam ? "driverEncodeData" :
    (this->type == WideRoadCam ? "wideRoadEncodeData" :
    (this->remuxing ? "qRoadEncodeData" : "roadEncodeData"));
  pm.reset(new PubMaster({service_name}));
}


void V4LEncoder::encoder_open(const char* path) {
  dequeue_handler_thread = std::thread(V4LEncoder::dequeue_handler, this);
  if (this->write) write_handler_thread = std::thread(V4LEncoder::write_handler, this, path);
  this->is_open = true;
}

int V4LEncoder::encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                  int in_width, int in_height, uint64_t ts) {
  assert(in_width == in_width_);
  assert(in_height == in_height_);
  assert(is_open);

  // reserve buffer
  if (buffer_in_outstanding >= BUF_IN_COUNT) {
    LOGE("ENCODER FRAME DROPPED")
    return -1;
  }
  buffer_in_outstanding++;

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

  // push buffer
  buf_in[buffer_in].sync(VISIONBUF_SYNC_TO_DEVICE);
  int ret = queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buffer_in, &buf_in[buffer_in], timestamp);
  assert(ret == 0);
  buffer_in = (buffer_in + 1) % BUF_IN_COUNT;

  return 0;
}

void V4LEncoder::encoder_close() {
  if (this->is_open) {
    struct v4l2_encoder_cmd encoder_cmd = { .cmd = V4L2_ENC_CMD_STOP };
    checked_ioctl(fd, VIDIOC_ENCODER_CMD, &encoder_cmd);
    // join waits for V4L2_QCOM_BUF_FLAG_EOS
    dequeue_handler_thread.join();
    if (this->write) write_handler_thread.join();
  }
  this->is_open = false;
}

V4LEncoder::~V4LEncoder() {
  encoder_close();
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  checked_ioctl(fd, VIDIOC_STREAMOFF, &buf_type);
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, 0);
  buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  checked_ioctl(fd, VIDIOC_STREAMOFF, &buf_type);
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, 0);
  close(fd);
}
