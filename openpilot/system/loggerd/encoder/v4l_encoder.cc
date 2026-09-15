#include <cassert>
#include <string>
#include <sys/ioctl.h>
#include <poll.h>
#include <utility>

#include "system/loggerd/encoder/v4l_encoder.h"
#include "common/util.h"
#include "common/timing.h"

#include <media/msm_media_info.h>

// has to be in this order
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#define V4L2_QCOM_BUF_FLAG_CODECCONFIG 0x00020000
#define V4L2_QCOM_BUF_FLAG_EOS 0x02000000

/*
  kernel debugging:
  echo 0xff > /sys/module/videobuf2_core/parameters/debug
  echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level
  echo 0xff > /sys/devices/platform/soc/aa00000.qcom,vidc/video4linux/video33/dev_debug
*/
const int env_debug_encoder = (getenv("DEBUG_ENCODER") != NULL) ? atoi(getenv("DEBUG_ENCODER")) : 0;

static void dequeue_buffer(int fd, v4l2_buf_type buf_type, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL) {
  v4l2_plane plane = {0};
  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
    .m = { .planes = &plane, },
    .length = 1,
  };
  util::safe_ioctl(fd, VIDIOC_DQBUF, &v4l_buf, "VIDIOC_DQBUF failed");

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
  util::safe_ioctl(fd, VIDIOC_QBUF, &v4l_buf, "VIDIOC_QBUF failed");
}

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
  };
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
}

void V4LEncoder::dequeue_handler(V4LEncoder *e) {
  std::string dequeue_thread_name = "dq-"+std::string(e->encoder_info.publish_name);
  util::set_thread_name(dequeue_thread_name.c_str());

  e->segment_num++;
  uint32_t idx = -1;
  bool exit = false;

  // POLLIN is capture, POLLOUT is frame. Qualcomm's reference client also
  // requests the corresponding normal-data bits.
  struct pollfd pfd;
  pfd.events = POLLIN | POLLRDNORM | POLLOUT | POLLWRNORM;
  pfd.fd = e->fd;

  // save the header
  kj::Array<capnp::byte> header;

  while (!exit) {
    int rc = poll(&pfd, 1, 1000);
    if (rc < 0) {
      if (errno != EINTR) {
        // TODO: exit encoder?
        // ignore the error and keep going
        LOGE("poll failed (%d - %d)", rc, errno);
      }
      continue;
    } else if (rc == 0) {
      LOGE("encoder dequeue poll timeout");
      continue;
    }

    if (env_debug_encoder >= 2) {
      printf("%20s poll %x at %.2f ms\n", e->encoder_info.publish_name, pfd.revents, millis_since_boot());
    }

    int frame_id = -1;
    if (pfd.revents & (POLLIN | POLLRDNORM)) {
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
        if (e->packet_callback) e->packet_callback(header.begin(), header.size(), ts, true, false);
      } else {
        VisionIpcBufExtra extra = e->extras.pop();
        assert(extra.timestamp_eof/1000 == ts); // stay in sync
        frame_id = extra.frame_id;
        ++idx;
        if (e->packet_callback) {
          e->packet_callback(buf, bytesused, ts, false, flags & V4L2_BUF_FLAG_KEYFRAME);
        } else {
          e->publisher_publish(e->segment_num, idx, extra, flags, header, kj::arrayPtr<capnp::byte>(buf, bytesused));
        }
      }

      if (env_debug_encoder) {
        printf("%20s got(%d) %6d bytes flags %8x idx %3d/%4d id %8d ts %ld lat %.2f ms (%lu frames free)\n",
          e->encoder_info.publish_name, index, bytesused, flags, e->segment_num, idx, frame_id, ts, millis_since_boot()-(ts/1000.), e->free_buf_in.size());
      }

      // requeue the buffer
      queue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, &e->buf_out[index]);
    }

    if (pfd.revents & (POLLOUT | POLLWRNORM)) {
      unsigned int index;
      dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, &index);
      VisionBuf *input_buf = e->input_bufs[index].exchange(nullptr);
      if (input_buf && e->input_done_callback) e->input_done_callback(input_buf);
      e->free_buf_in.push(index);
    }
  }
}

V4LEncoder::V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height)
    : V4LEncoder(encoder_info, in_width, in_height, Options{}) {}

V4LEncoder::V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options)
    : VideoEncoder(encoder_info, in_width, in_height), packet_callback(std::move(options.packet_callback)),
      input_done_callback(std::move(options.input_done_callback)) {
  fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-aa00000.qcom_vidc-video-index1", O_RDWR|O_NONBLOCK));
  assert(fd >= 0);

  struct v4l2_capability cap;
  util::safe_ioctl(fd, VIDIOC_QUERYCAP, &cap, "VIDIOC_QUERYCAP failed");
  LOGD("opened encoder device %s %s = %d", cap.driver, cap.card, fd);
  assert(strcmp((const char *)cap.driver, "msm_vidc_driver") == 0);
  assert(strcmp((const char *)cap.card, "msm_vidc_venc") == 0);

  EncoderSettings encoder_settings = encoder_info.get_settings(in_width);
  current_bitrate = encoder_settings.bitrate;
  bool is_h265 = encoder_settings.encode_type == cereal::EncodeIndex::Type::FULL_H_E_V_C;

  struct v4l2_format fmt_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .fmt = {
      .pix_mp = {
        // downscales are free with v4l
        .width = (unsigned int)(out_width),
        .height = (unsigned int)(out_height),
        .pixelformat = is_h265 ? V4L2_PIX_FMT_HEVC : V4L2_PIX_FMT_H264,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_DEFAULT,
      }
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_out, "VIDIOC_S_FMT failed");

  v4l2_streamparm streamparm = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .parm = {
      .output = {
        // TODO: more stuff here? we don't know
        .timeperframe = {
          .numerator = 1,
          .denominator = (unsigned int)encoder_info.fps
        }
      }
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_PARM, &streamparm, "VIDIOC_S_PARM failed");

  struct v4l2_format fmt_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .fmt = {
      .pix_mp = {
        .width = (unsigned int)in_width,
        .height = (unsigned int)in_height,
        .pixelformat = options.input_format,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_470_SYSTEM_BG,
      }
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_in, "VIDIOC_S_FMT failed");

  LOGD("in buffer size %d, out buffer size %d",
    fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage,
    fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);

  // shared ctrls
  {
    struct v4l2_control ctrls[] = {
      { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = encoder_settings.bitrate},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = encoder_settings.gop_size - encoder_settings.b_frames - 1},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = encoder_settings.b_frames},
      { .id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL, .value = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD, .value = 1},
    };
    for (auto ctrl : ctrls) {
      util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
    }
  }
  if (options.max_performance) {
    struct v4l2_control ctrl = {
      .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY,
      .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_ENABLE,
    };
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL offline encode failed");
  }

  if (is_h265) {
    struct v4l2_control ctrls[] = {
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5},
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO, .value = V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED},
    };
    for (auto ctrl : ctrls) {
      util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
    }
  } else {
    if (encoder_info.is_live) {
      struct v4l2_control ctrls[] = {
        { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
        { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = V4L2_MPEG_VIDEO_H264_LEVEL_3_1},
        { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, .value = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0},
      };
      for (auto ctrl : ctrls) {
        util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
      }
    } else {
      struct v4l2_control ctrls[] = {
        { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
        { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN},
        { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, .value = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0},
      };
      for (auto ctrl : ctrls) {
        util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
      }
    }

    struct v4l2_control ctrls[] = {
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, .value = V4L2_MPEG_VIDEO_H264_LOOP_FILTER_MODE_ENABLED},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA, .value = 0},
      { .id = V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE, .value = 0},
    };
    for (auto ctrl : ctrls) {
      util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
    }
  }

  // allocate buffers
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, BUF_OUT_COUNT);
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, BUF_IN_COUNT);

  // start encoder
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &buf_type, "VIDIOC_STREAMON failed");
  buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &buf_type, "VIDIOC_STREAMON failed");

  // queue up output buffers
  for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
    buf_out[i].allocate(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);
    queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, &buf_out[i]);
  }
  // queue up input buffers
  for (unsigned int i = 0; i < BUF_IN_COUNT; i++) {
    free_buf_in.push(i);
  }
}

void V4LEncoder::encoder_open() {
  dequeue_handler_thread = std::thread(V4LEncoder::dequeue_handler, this);
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
  input_bufs[buffer_in].store(buf);

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
    util::safe_ioctl(fd, VIDIOC_ENCODER_CMD, &encoder_cmd, "VIDIOC_ENCODER_CMD failed");
    // join waits for V4L2_QCOM_BUF_FLAG_EOS
    dequeue_handler_thread.join();
    assert(extras.empty());
  }
  this->is_open = false;
}

void V4LEncoder::set_bitrate(int bitrate) {
  if (bitrate == current_bitrate) return;
  if (bitrate <= 0) {
    LOGE("invalid livestream encoder bitrate %d", bitrate);
    return;
  }

  struct v4l2_control ctrl = {
    .id = V4L2_CID_MPEG_VIDEO_BITRATE,
    .value = bitrate,
  };

  if (util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl) == -1) {
    LOGE("failed to update %s bitrate to %d", encoder_info.publish_name, bitrate);
    return;
  }
  current_bitrate = bitrate;
}

void V4LEncoder::request_keyframe() {
  struct v4l2_control ctrl = {
    .id = V4L2_CID_MPEG_VIDC_VIDEO_REQUEST_IFRAME,
    .value = 1,
  };

  if (util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl) == -1) {
    LOGE("failed to request keyframe for %s", encoder_info.publish_name);
  }
}

V4LEncoder::~V4LEncoder() {
  encoder_close();
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMOFF, &buf_type, "VIDIOC_STREAMOFF failed");
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, 0);
  buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMOFF, &buf_type, "VIDIOC_STREAMOFF failed");
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, 0);
  close(fd);

  for (int i = 0; i < BUF_OUT_COUNT; i++) {
    if (buf_out[i].free() != 0) {
      LOGE("Failed to free buffer");
    }
  }
}
