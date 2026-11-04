#include "system/loggerd/encoder/v4l_encoder.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <poll.h>
#include <utility>

#include "msgq/visionipc/visionbuf.h"
#include "system/loggerd/encoder/msm_vidc_encoder.h"
#include "system/loggerd/encoder/venus_encoder.h"
#include "common/util.h"
#include "common/timing.h"

/*
  kernel debugging:
  echo 0xff > /sys/module/videobuf2_core/parameters/debug  (videobuf2_common upstream)
  echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level
  echo 0xff > /sys/class/video4linux/videoN/dev_debug
*/
const int env_debug_encoder = (getenv("DEBUG_ENCODER") != NULL) ? atoi(getenv("DEBUG_ENCODER")) : 0;

std::unique_ptr<V4LEncoder> V4LEncoder::create(const EncoderInfo &encoder_info, int in_width, int in_height, Options options) {
  // VisionIPC uses DMA-BUF heaps on the upstream kernel, ION on the downstream one
  if (strcmp(visionbuf_backend_name(), "dma_heap") == 0) {
    return std::make_unique<VenusEncoder>(encoder_info, in_width, in_height, std::move(options));
  }
  return std::make_unique<MsmVidcEncoder>(encoder_info, in_width, in_height, std::move(options));
}

static void dequeue_buffer(int fd, v4l2_memory memory, v4l2_buf_type buf_type, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL, unsigned int *data_offset=NULL) {
  v4l2_plane plane = {0};
  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .memory = memory,
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

void V4LEncoder::queue_buffer(v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, struct timeval timestamp) {
  v4l2_plane plane = {
    .bytesused = (uint32_t)buf->len,
    .length = (unsigned int)buf->len,
  };
  set_plane(&plane, buf);

  v4l2_buffer v4l_buf = {
    .index = index,
    .type = buf_type,
    .flags = V4L2_BUF_FLAG_TIMESTAMP_COPY,
    .timestamp = timestamp,
    .memory = memory,
    .m = { .planes = &plane, },
    .length = 1,
  };
  util::safe_ioctl(fd, VIDIOC_QBUF, &v4l_buf, "VIDIOC_QBUF failed");
}

void V4LEncoder::request_buffers(v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = memory,
  };
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
  // the driver may give more, but every index that's queued has to exist
  if (reqbuf.count < count) {
    LOGE("VIDIOC_REQBUFS gave %u buffers, %u are needed", reqbuf.count, count);
    throw std::runtime_error("VIDIOC_REQBUFS gave too few buffers");
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

    // the encoder failed: upstream Venus puts both queues in an error state after a session error or a firmware crash
    // (the bitstream queue always has buffers queued, so this isn't mem2mem's POLLERR for idle queues). nothing is
    // encoded after that, so the process exits (the uncaught exception aborts it): selfdrived reports processNotRunning
    // and the next onroad cycle starts it again. msm_vidc never reports POLLERR
    if (pfd.revents & POLLERR) {
      LOGE("%s: encoder failed (poll revents %#x)", e->encoder_info.publish_name, pfd.revents);
      throw std::runtime_error("encoder failed");
    }

    int frame_id = -1;
    if (pfd.revents & (POLLIN | POLLRDNORM)) {
      unsigned int bytesused, flags, index, data_offset;
      struct timeval timestamp;
      dequeue_buffer(e->fd, e->memory, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &index, &bytesused, &flags, &timestamp, &data_offset);
      VisionBuf &out = e->buf_out[index];
      // the hardware wrote it (ION invalidates here), and the CPU reads the payload until it's requeued
      out.sync(VISIONBUF_SYNC_FROM_DEVICE);
      bytesused -= data_offset;
      int err = out.begin_cpu_read(data_offset, bytesused);
      assert(err == 0);
      uint8_t *buf = (uint8_t*)out.addr + data_offset;
      int64_t ts = timestamp.tv_sec * 1000000 + timestamp.tv_usec;

      const Bitstream bs = e->parse_bitstream(flags, buf, bytesused);
      if (bs.header_len > 0) {
        // the header is announced when it changes, and published with every keyframe
        if (header.size() != bs.header_len || memcmp(header.begin(), buf, bs.header_len) != 0) {
          header = kj::heapArray<capnp::byte>(buf, bs.header_len);
          if (e->packet_callback) e->packet_callback(header.begin(), header.size(), ts, true, false);
        }
        buf += bs.header_len;
        bytesused -= bs.header_len;
      }
      if (bs.frame) {
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
          e->encoder_info.publish_name, index, bytesused, flags, e->segment_num, idx, frame_id, ts, millis_since_boot()-(ts/1000.), e->in_slots.free_count());
      }

      err = out.end_cpu_access(false);
      assert(err == 0);
      e->queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, &out);
      exit = bs.eos;
    }

    if (pfd.revents & (POLLOUT | POLLWRNORM)) {
      unsigned int index;
      dequeue_buffer(e->fd, e->memory, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, &index);
      VisionBuf *input_buf = e->input_bufs[index].exchange(nullptr);
      if (input_buf && e->input_done_callback) e->input_done_callback(input_buf);
      e->in_slots.release(index);
    }
  }
}

V4LEncoder::V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options, int fd,
                       v4l2_memory memory, int input_count)
    : VideoEncoder(encoder_info, in_width, in_height), fd(fd), memory(memory), in_slots(input_count),
      packet_callback(std::move(options.packet_callback)), input_done_callback(std::move(options.input_done_callback)),
      input_bufs(input_count) {
  assert(fd >= 0);
  EncoderSettings encoder_settings = encoder_info.get_settings(in_width);
  current_bitrate = encoder_settings.bitrate;
  is_h265 = encoder_settings.encode_type == cereal::EncodeIndex::Type::FULL_H_E_V_C;
}

void V4LEncoder::set_formats(uint32_t input_format) {
  fmt_out = {
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

  fmt_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .fmt = {
      .pix_mp = {
        .width = (unsigned int)in_width,
        .height = (unsigned int)in_height,
        .pixelformat = input_format,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_470_SYSTEM_BG,
      }
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_in, "VIDIOC_S_FMT failed");

  LOGD("in buffer size %d, out buffer size %d",
    fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage,
    fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);
}

void V4LEncoder::set_controls(std::initializer_list<v4l2_control> ctrls) {
  for (auto ctrl : ctrls) {
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
  }
}

void V4LEncoder::start(bool requery_bitstream_size) {
  request_buffers(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, BUF_OUT_COUNT);
  request_buffers(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, in_slots.size());
  if (requery_bitstream_size) {
    util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt_out, "VIDIOC_G_FMT failed");
  }

  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &buf_type, "VIDIOC_STREAMON failed");
  buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &buf_type, "VIDIOC_STREAMON failed");

  // queue up the bitstream buffers
  for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
    buf_out[i].allocate(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);
    queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, &buf_out[i]);
  }
}

void V4LEncoder::encoder_open() {
  if (segment_num >= 0) resume();
  dequeue_handler_thread = std::thread(V4LEncoder::dequeue_handler, this);
  this->is_open = true;
  this->counter = 0;
}

int V4LEncoder::encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) {
  struct timeval timestamp {
    .tv_sec = (long)(extra->timestamp_eof/1000000000),
    .tv_usec = (long)((extra->timestamp_eof/1000) % 1000000),
  };

  // reserve an input buffer
  int buffer_in = acquire_input(buf);
  input_bufs[buffer_in].store(buf);

  // push buffer
  extras.push(*extra);
  queue_buffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buffer_in, buf, timestamp);

  return this->counter++;
}

void V4LEncoder::encoder_close() {
  if (this->is_open) {
    // wait for all the frames to be encoded
    in_slots.wait_idle();
    // no frames, stop the encoder
    struct v4l2_encoder_cmd encoder_cmd = { .cmd = V4L2_ENC_CMD_STOP };
    util::safe_ioctl(fd, VIDIOC_ENCODER_CMD, &encoder_cmd, "VIDIOC_ENCODER_CMD failed");
    // join waits for the end of stream
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
    .id = keyframe_control(),
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
  request_buffers(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, 0);
  buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMOFF, &buf_type, "VIDIOC_STREAMOFF failed");
  request_buffers(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, 0);
  close(fd);

  for (int i = 0; i < BUF_OUT_COUNT; i++) {
    if (buf_out[i].free() != 0) {
      LOGE("Failed to free buffer");
    }
  }
}
