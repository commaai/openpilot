#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <poll.h>
#include <utility>

#include "system/loggerd/encoder/v4l_encoder.h"
#include "system/loggerd/encoder/v4l_device.h"
#include "common/util.h"
#include "common/timing.h"

// has to be in this order
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>

/*
  kernel debugging:
  echo 0xff > /sys/module/videobuf2_common/parameters/debug
  echo 0xff > /sys/class/video4linux/videoN/dev_debug
*/
const int env_debug_encoder = (getenv("DEBUG_ENCODER") != NULL) ? atoi(getenv("DEBUG_ENCODER")) : 0;

// Venus puts the codec header in front of the first slice of every IDR frame.
// returns the header's length, i.e. the offset of the first VCL NAL unit's start code, or 0 if there's no header
static size_t header_size(const uint8_t *dat, size_t len, bool h265) {
  for (size_t i = 0; i + 3 < len; i++) {
    if (dat[i] == 0 && dat[i+1] == 0 && dat[i+2] == 1) {
      int nal_type = h265 ? (dat[i+3] >> 1) & 0x3f : dat[i+3] & 0x1f;
      bool vcl = h265 ? nal_type < 32 : (nal_type >= 1 && nal_type <= 5);
      if (vcl) return (i > 0 && dat[i-1] == 0) ? i-1 : i;
    }
  }
  return 0;
}

// the lowest H.264 level (H.264 Table A-1) whose frame size and macroblock rate fit, up to Venus' maximum of 5.1.
// live streams don't go below 3.1
static v4l2_mpeg_video_h264_level h264_level(int width, int height, int fps, bool live) {
  struct Level { v4l2_mpeg_video_h264_level level; int64_t max_mbps, max_fs; };
  static const Level levels[] = {
    {V4L2_MPEG_VIDEO_H264_LEVEL_1_0, 1485, 99},
    {V4L2_MPEG_VIDEO_H264_LEVEL_1_1, 3000, 396},
    {V4L2_MPEG_VIDEO_H264_LEVEL_1_2, 6000, 396},
    {V4L2_MPEG_VIDEO_H264_LEVEL_1_3, 11880, 396},
    {V4L2_MPEG_VIDEO_H264_LEVEL_2_0, 11880, 396},
    {V4L2_MPEG_VIDEO_H264_LEVEL_2_1, 19800, 792},
    {V4L2_MPEG_VIDEO_H264_LEVEL_2_2, 20250, 1620},
    {V4L2_MPEG_VIDEO_H264_LEVEL_3_0, 40500, 1620},
    {V4L2_MPEG_VIDEO_H264_LEVEL_3_1, 108000, 3600},
    {V4L2_MPEG_VIDEO_H264_LEVEL_3_2, 216000, 5120},
    {V4L2_MPEG_VIDEO_H264_LEVEL_4_0, 245760, 8192},
    {V4L2_MPEG_VIDEO_H264_LEVEL_4_1, 245760, 8192},
    {V4L2_MPEG_VIDEO_H264_LEVEL_4_2, 522240, 8704},
    {V4L2_MPEG_VIDEO_H264_LEVEL_5_0, 589824, 22080},
    {V4L2_MPEG_VIDEO_H264_LEVEL_5_1, 983040, 36864},
  };
  const int64_t w = (width + 15) / 16, h = (height + 15) / 16;
  const int64_t fs = w * h;
  for (const auto &l : levels) {
    if (live && l.level < V4L2_MPEG_VIDEO_H264_LEVEL_3_1) continue;
    // neither dimension may exceed sqrt(8 * MaxFS) macroblocks
    if (fs <= l.max_fs && fs * fps <= l.max_mbps && w * w <= 8 * l.max_fs && h * h <= 8 * l.max_fs) return l.level;
  }
  return V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
}

static void dequeue_buffer(int fd, v4l2_buf_type buf_type, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL, unsigned int *data_offset=NULL) {
  v4l2_plane plane = {0};
  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .memory = V4L2_MEMORY_DMABUF,
    .m = { .planes = &plane, },
    .length = 1,
  };
  util::safe_ioctl(fd, VIDIOC_DQBUF, &v4l_buf, "VIDIOC_DQBUF failed");

  if (index) *index = v4l_buf.index;
  if (bytesused) *bytesused = v4l_buf.m.planes[0].bytesused;
  if (flags) *flags = v4l_buf.flags;
  if (timestamp) *timestamp = v4l_buf.timestamp;
  if (data_offset) *data_offset = v4l_buf.m.planes[0].data_offset;
}

static void queue_buffer(int fd, v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, struct timeval timestamp={}) {
  v4l2_plane plane = {
    .bytesused = (uint32_t)buf->len,
    .length = (unsigned int)buf->len,
    .m = { .fd = buf->fd, },
  };

  v4l2_buffer v4l_buf = {
    .index = index,
    .type = buf_type,
    .flags = V4L2_BUF_FLAG_TIMESTAMP_COPY,
    .timestamp = timestamp,
    .memory = V4L2_MEMORY_DMABUF,
    .m = { .planes = &plane, },
    .length = 1,
  };
  util::safe_ioctl(fd, VIDIOC_QBUF, &v4l_buf, "VIDIOC_QBUF failed");
}

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = V4L2_MEMORY_DMABUF,
  };
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
  // the driver may give more, but every index that's queued has to exist
  if (reqbuf.count < count) {
    LOGE("VIDIOC_REQBUFS gave %u buffers, %u are needed", reqbuf.count, count);
    throw std::runtime_error("VIDIOC_REQBUFS gave too few buffers");
  }
}

static void set_controls(int fd, std::initializer_list<v4l2_control> ctrls) {
  for (auto ctrl : ctrls) {
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
  }
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
      if (errno == EINTR) continue;
      LOGE("%s: poll failed (%d)", e->encoder_info.publish_name, errno);
      throw std::runtime_error("encoder poll failed");
    } else if (rc == 0) {
      LOGE("encoder dequeue poll timeout");
      continue;
    }

    if (env_debug_encoder >= 2) {
      printf("%20s poll %x at %.2f ms\n", e->encoder_info.publish_name, pfd.revents, millis_since_boot());
    }

    // the encoder failed: Venus puts both queues in an error state after a session error or a firmware crash (the
    // bitstream queue always has buffers queued, so this isn't mem2mem's POLLERR for idle queues). nothing is encoded
    // after that, so the process exits (the uncaught exception aborts it): selfdrived reports processNotRunning and the
    // next onroad cycle starts it again
    if (pfd.revents & POLLERR) {
      LOGE("%s: encoder failed (poll revents %#x)", e->encoder_info.publish_name, pfd.revents);
      throw std::runtime_error("encoder failed");
    }

    int frame_id = -1;
    if (pfd.revents & (POLLIN | POLLRDNORM)) {
      unsigned int bytesused, flags, index, data_offset;
      struct timeval timestamp;
      dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &index, &bytesused, &flags, &timestamp, &data_offset);
      VisionBuf &out = e->buf_out[index];
      bytesused -= data_offset;
      int err = out.begin_cpu_read(data_offset, bytesused);
      assert(err == 0);
      uint8_t *buf = (uint8_t*)out.addr + data_offset;
      int64_t ts = timestamp.tv_sec * 1000000 + timestamp.tv_usec;

      // the header is in front of the first slice of every IDR frame. it's announced when it changes, and published
      // with every keyframe
      const size_t header_len = (flags & V4L2_BUF_FLAG_KEYFRAME) ? header_size(buf, bytesused, e->is_h265) : 0;
      if (header_len > 0) {
        if (header.size() != header_len || memcmp(header.begin(), buf, header_len) != 0) {
          header = kj::heapArray<capnp::byte>(buf, header_len);
          if (e->packet_callback) e->packet_callback(header.begin(), header.size(), ts, true, false);
        }
        buf += header_len;
        bytesused -= header_len;
      }
      // the drain ends with a LAST buffer, which may hold the last frame
      if (bytesused > 0) {
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
        printf("%20s got(%d) %6d bytes flags %8x idx %3d/%4d id %8d ts %ld lat %.2f ms\n",
          e->encoder_info.publish_name, index, bytesused, flags, e->segment_num, idx, frame_id, ts, millis_since_boot()-(ts/1000.));
      }

      // requeue the buffer
      err = out.end_cpu_access(false);
      assert(err == 0);
      queue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, &out);
      exit = flags & V4L2_BUF_FLAG_LAST;
    }

    if (pfd.revents & (POLLOUT | POLLWRNORM)) {
      unsigned int index;
      dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, &index);
      VisionBuf *input_buf;
      {
        std::lock_guard lk(e->input_lock);
        input_buf = std::exchange(e->input_bufs[index], nullptr);
      }
      e->input_cv.notify_all();
      if (input_buf && e->input_done_callback) e->input_done_callback(input_buf);
    }
  }
}

V4LEncoder::V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height)
    : V4LEncoder(encoder_info, in_width, in_height, Options{}) {}

V4LEncoder::V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options)
    : VideoEncoder(encoder_info, in_width, in_height), packet_callback(std::move(options.packet_callback)),
      input_done_callback(std::move(options.input_done_callback)) {
  fd = open_video_device("qcom-venus-encoder");
  assert(fd >= 0);

  struct v4l2_capability cap;
  util::safe_ioctl(fd, VIDIOC_QUERYCAP, &cap, "VIDIOC_QUERYCAP failed");
  LOGD("opened encoder device %s %s = %d", cap.driver, cap.card, fd);
  assert(strcmp((const char *)cap.driver, "qcom-venus") == 0);

  EncoderSettings encoder_settings = encoder_info.get_settings(in_width);
  current_bitrate = encoder_settings.bitrate;
  is_h265 = encoder_settings.encode_type == cereal::EncodeIndex::Type::FULL_H_E_V_C;

  struct v4l2_format fmt_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .fmt = {
      .pix_mp = {
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
        .pixelformat = V4L2_PIX_FMT_NV12,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_470_SYSTEM_BG,
      }
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_in, "VIDIOC_S_FMT failed");

  // Venus aligns the frame size to 128x32 for its buffers, and encodes all of it unless it's cropped to the frame
  struct v4l2_selection sel = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT,
    .target = V4L2_SEL_TGT_CROP,
    .r = {.left = 0, .top = 0, .width = (uint32_t)in_width, .height = (uint32_t)in_height},
  };
  util::safe_ioctl(fd, VIDIOC_S_SELECTION, &sel, "VIDIOC_S_SELECTION failed");
  // the source format and crop reset the bitstream size to the crop. a smaller one is a scaled encode, so it's set
  // again. the stream's header has the exact size (H.264 frame cropping, the HEVC conformance window)
  if (out_width != in_width || out_height != in_height) {
    util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt_out, "VIDIOC_G_FMT failed");
    fmt_out.fmt.pix_mp.width = out_width;
    fmt_out.fmt.pix_mp.height = out_height;
    util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_out, "VIDIOC_S_FMT failed");
    // a kernel without the scaling patch crops the source to that size instead
    util::safe_ioctl(fd, VIDIOC_G_SELECTION, &sel, "VIDIOC_G_SELECTION failed");
    assert(sel.r.width == (uint32_t)in_width && sel.r.height == (uint32_t)in_height);
  }

  set_controls(fd, {
    { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = encoder_settings.bitrate},
    // VBR without frame skipping is the firmware's VBR_CFR
    { .id = V4L2_CID_MPEG_VIDEO_BITRATE_MODE, .value = V4L2_MPEG_VIDEO_BITRATE_MODE_VBR},
    { .id = V4L2_CID_MPEG_VIDEO_GOP_SIZE, .value = encoder_settings.gop_size},
    { .id = V4L2_CID_MPEG_VIDEO_B_FRAMES, .value = encoder_settings.b_frames},
    { .id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME},
  });
  if (is_h265) {
    // Venus has no tier control, so this is Main tier. at level 5 its 25 Mbps limit is above the road cameras' 10 Mbps
    set_controls(fd, {
      { .id = V4L2_CID_MPEG_VIDEO_HEVC_PROFILE, .value = V4L2_MPEG_VIDEO_HEVC_PROFILE_MAIN},
      { .id = V4L2_CID_MPEG_VIDEO_HEVC_LEVEL, .value = V4L2_MPEG_VIDEO_HEVC_LEVEL_5},
    });
  } else {
    set_controls(fd, {
      { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = h264_level(out_width, out_height, encoder_info.fps, encoder_info.is_live)},
      { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
      { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, .value = V4L2_MPEG_VIDEO_H264_LOOP_FILTER_MODE_ENABLED},
    });
  }

  // allocate buffers
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, BUF_OUT_COUNT);
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, BUF_IN_COUNT);
  // the bitstream buffer size comes from the firmware once the buffers are requested
  util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt_out, "VIDIOC_G_FMT failed");

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
}

void V4LEncoder::encoder_open() {
  if (segment_num >= 0) {
    // the drain in encoder_close stopped the encoder, restart it with an IDR frame for the new segment
    struct v4l2_encoder_cmd encoder_cmd = { .cmd = V4L2_ENC_CMD_START };
    util::safe_ioctl(fd, VIDIOC_ENCODER_CMD, &encoder_cmd, "VIDIOC_ENCODER_CMD failed");
    request_keyframe();
  }
  dequeue_handler_thread = std::thread(V4LEncoder::dequeue_handler, this);
  this->is_open = true;
  this->counter = 0;
}

int V4LEncoder::encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) {
  struct timeval timestamp {
    .tv_sec = (long)(extra->timestamp_eof/1000000000),
    .tv_usec = (long)((extra->timestamp_eof/1000) % 1000000),
  };

  // reserve the buffer's index
  const int buffer_in = buf->idx % BUF_IN_COUNT;
  {
    std::unique_lock lk(input_lock);
    input_cv.wait(lk, [&] { return input_bufs[buffer_in] == nullptr; });
    input_bufs[buffer_in] = buf;
  }

  // push buffer. the dequeue thread waits for its extra, so it's pushed after the frame is queued
  try {
    queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buffer_in, buf, timestamp);
  } catch (...) {
    // free the index, else encoder_close waits for it forever
    std::lock_guard lk(input_lock);
    input_bufs[buffer_in] = nullptr;
    throw;
  }
  extras.push(*extra);

  return this->counter++;
}

void V4LEncoder::encoder_close() {
  if (this->is_open) {
    // wait for all the frames to be encoded
    {
      std::unique_lock lk(input_lock);
      input_cv.wait(lk, [&] { return std::all_of(std::begin(input_bufs), std::end(input_bufs), [](VisionBuf *b) { return !b; }); });
    }
    // no frames, stop the encoder
    struct v4l2_encoder_cmd encoder_cmd = { .cmd = V4L2_ENC_CMD_STOP };
    util::safe_ioctl(fd, VIDIOC_ENCODER_CMD, &encoder_cmd, "VIDIOC_ENCODER_CMD failed");
    // join waits for the LAST buffer
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
    .id = V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME,
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
