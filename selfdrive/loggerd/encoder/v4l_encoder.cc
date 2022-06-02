#include <cassert>
#include <sys/ioctl.h>
#include <poll.h>

#include "selfdrive/loggerd/encoder/v4l_encoder.h"
#include "common/util.h"
#include "common/timing.h"

#include "libyuv.h"
#include "msm_media_info.h"

// has to be in this order
#include "v4l2-controls.h"
#include <linux/videodev2.h>
#define V4L2_QCOM_BUF_FLAG_CODECCONFIG 0x00020000
#define V4L2_QCOM_BUF_FLAG_EOS 0x02000000

// echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level
const int env_debug_encoder = (getenv("DEBUG_ENCODER") != NULL) ? atoi(getenv("DEBUG_ENCODER")) : 0;

#define checked_ioctl(x,y,z) { int _ret = HANDLE_EINTR(ioctl(x,y,z)); if (_ret!=0) { LOGE("checked_ioctl failed %d %lx %p", x, y, z); } assert(_ret==0); }

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

static void queue_buffer(int fd, v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, struct timeval timestamp={0}, unsigned int bytesused=0) {
  v4l2_plane plane = {
    .length = (unsigned int)buf->len,
    .m = { .userptr = (unsigned long)buf->addr, },
    .bytesused = bytesused,
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

  checked_ioctl(fd, VIDIOC_QBUF, &v4l_buf);
}

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
    .count = count
  };
  checked_ioctl(fd, VIDIOC_REQBUFS, &reqbuf);
}

void V4LEncoder::dequeue_handler(V4LEncoder *e) {
  std::string dequeue_thread_name = "dq-"+std::string(e->filename);
  util::set_thread_name(dequeue_thread_name.c_str());

  e->segment_num++;
  uint32_t idx = -1;
  bool exit = false;

  // POLLIN is capture, POLLOUT is frame
  struct pollfd pfd;
  pfd.events = POLLIN | POLLOUT;
  pfd.fd = e->fd;

  // save the header
  kj::Array<capnp::byte> header;

  while (!exit) {
    int rc = poll(&pfd, 1, 1000);
    if (!rc) { LOGE("encoder dequeue poll timeout"); continue; }

    if (env_debug_encoder >= 2) {
      printf("%20s poll %x at %.2f ms\n", e->filename, pfd.revents, millis_since_boot());
    }

    int frame_id = -1;
    if (pfd.revents & POLLIN) {
      unsigned int bytesused, flags, index;
      struct timeval timestamp;
      dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &index, &bytesused, &flags, &timestamp);
      e->buf_out[index].sync(VISIONBUF_SYNC_FROM_DEVICE);
      uint8_t *buf = (uint8_t*)e->buf_out[index].addr;
      int64_t ts = timestamp.tv_sec * 1000000 + timestamp.tv_usec;

      // eof packet, we exit
      if (flags & V4L2_QCOM_BUF_FLAG_EOS) {
        exit = true;
      } else if (flags & V4L2_QCOM_BUF_FLAG_CODECCONFIG) {
        // save header
        header = kj::heapArray<capnp::byte>(buf, bytesused);
      } else {
        VisionIpcBufExtra extra = e->extras.pop();
        assert(extra.timestamp_eof/1000 == ts); // stay in sync
        frame_id = extra.frame_id;
        ++idx;
        e->publisher_publish(e, e->segment_num, idx, extra, flags, header, kj::arrayPtr<capnp::byte>(buf, bytesused));
      }

      if (env_debug_encoder) {
        printf("%20s got(%d) %6d bytes flags %8x idx %3d/%4d id %8d ts %ld lat %.2f ms (%lu frames free)\n",
          e->filename, index, bytesused, flags, e->segment_num, idx, frame_id, ts, millis_since_boot()-(ts/1000.), e->free_buf_in.size());
      }

      // requeue the buffer
      queue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, &e->buf_out[index]);
    }

    if (pfd.revents & POLLOUT) {
      unsigned int index;
      dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, &index);
      e->free_buf_in.push(index);
    }
  }
}

void V4LEncoder::encoder_init() {
  fd = open("/dev/v4l/by-path/platform-aa00000.qcom_vidc-video-index1", O_RDWR|O_NONBLOCK);
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
        .pixelformat = (codec == cereal::EncodeIndex::Type::FULL_H_E_V_C) ? V4L2_PIX_FMT_HEVC : V4L2_PIX_FMT_H264,
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

  // shared ctrls
  {
    struct v4l2_control ctrls[] = {
      { .id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE},
      { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = bitrate},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL, .value = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD, .value = 1},
    };
    for (auto ctrl : ctrls) {
      checked_ioctl(fd, VIDIOC_S_CTRL, &ctrl);
    }
  }

  if (codec == cereal::EncodeIndex::Type::FULL_H_E_V_C) {
    struct v4l2_control ctrls[] = {
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = 29},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = 0},
    };
    for (auto ctrl : ctrls) {
      checked_ioctl(fd, VIDIOC_S_CTRL, &ctrl);
    }
  } else {
    struct v4l2_control ctrls[] = {
      { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = 14},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, .value = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE, .value = 0},
    };
    for (auto ctrl : ctrls) {
      checked_ioctl(fd, VIDIOC_S_CTRL, &ctrl);
    }
  }

  // allocate buffers
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, BUF_OUT_COUNT);
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, BUF_IN_COUNT);

  // start encoder
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  checked_ioctl(fd, VIDIOC_STREAMON, &buf_type);
  buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  checked_ioctl(fd, VIDIOC_STREAMON, &buf_type);

  // queue up output buffers
  for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
    buf_out[i].allocate(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);
    queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, &buf_out[i]);
  }
  // queue up input buffers
  for (unsigned int i = 0; i < BUF_IN_COUNT; i++) {
    free_buf_in.push(i);
  }

  publisher_init();
}

void V4LEncoder::encoder_open(const char* path) {
  dequeue_handler_thread = std::thread(V4LEncoder::dequeue_handler, this);
  writer_open(path);
  this->is_open = true;
  this->counter = 0;
}

int V4LEncoder::encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) {
  struct timeval timestamp {
    .tv_sec = (long)(extra->timestamp_eof/1000000000),
    .tv_usec = (long)((extra->timestamp_eof/1000) % 1000000),
  };

  // reserve buffer
  int buffer_in = free_buf_in.pop();

  // push buffer
  extras.push(*extra);
  //buf->sync(VISIONBUF_SYNC_TO_DEVICE);
  queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buffer_in, buf, timestamp);

  return this->counter++;
}

void V4LEncoder::encoder_close() {
  if (this->is_open) {
    // pop all the frames before closing, then put the buffers back
    for (int i = 0; i < BUF_IN_COUNT; i++) free_buf_in.pop();
    for (int i = 0; i < BUF_IN_COUNT; i++) free_buf_in.push(i);
    // no frames, stop the encoder
    struct v4l2_encoder_cmd encoder_cmd = { .cmd = V4L2_ENC_CMD_STOP };
    checked_ioctl(fd, VIDIOC_ENCODER_CMD, &encoder_cmd);
    // join waits for V4L2_QCOM_BUF_FLAG_EOS
    dequeue_handler_thread.join();
    assert(extras.empty());
    writer_close();
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
