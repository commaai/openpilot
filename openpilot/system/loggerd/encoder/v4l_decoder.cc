#include "system/loggerd/encoder/v4l_decoder.h"

#include <assert.h>
#include <cerrno>
#include <climits>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>


#include "common/swaglog.h"
#include "common/util.h"

constexpr int OFFLINE_CORE_PLACEMENT_RATE = 80 << 16;

// echo "0xFFFF" > /sys/kernel/debug/msm_vidc/debug_level

static void copyBuffer(VisionBuf *src_buf, VisionBuf *dst_buf) {
  // Copy Y plane
  memcpy(dst_buf->y, src_buf->y, src_buf->height * src_buf->stride);
  // Copy UV plane
  memcpy(dst_buf->uv, src_buf->uv, src_buf->height / 2 * src_buf->stride);
}

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR
  };
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
}

V4LDecoder::~V4LDecoder() {
  if (fd > 0) {
    close(fd);
  }
}

bool V4LDecoder::init(const char* dev, size_t width, size_t height, uint64_t codec,
                   bool direct_mode, uint32_t capture_fourcc) {
  LOG("Initializing msm_vidc device %s", dev);
  this->w = width;
  this->h = height;
  this->direct = direct_mode;
  this->capture_format = capture_fourcc;
  this->fd = open(dev, O_RDWR | O_NONBLOCK, 0);
  if (fd < 0) {
    LOGE("failed to open video device %s", dev);
    return false;
  }
  subscribeEvents();
  v4l2_buf_type out_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  setPlaneFormat(out_type, codec); // Also allocates the output buffers
  setFPS(FPS);
  if (direct) {
    struct v4l2_control ctrls[] = {
      // A finite real-time load lets the driver place decode and encode on separate cores.
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE, .value = OFFLINE_CORE_PLACEMENT_RATE },
      { .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_ENABLE },
    };
    for (auto ctrl : ctrls) {
      util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL offline decode failed");
    }
  }
  request_buffers(fd, out_type, OUTPUT_BUFFER_COUNT);
  util::safe_ioctl(fd, VIDIOC_STREAMON, &out_type, "VIDIOC_STREAMON OUTPUT failed");
  restartCapture();
  pfd = {fd, POLLIN | POLLOUT | POLLWRNORM | POLLRDNORM | POLLPRI, 0};

  this->initialized = true;
  return true;
}

VisionBuf* V4LDecoder::decodeFrame(AVPacket *pkt, VisionBuf *buf) {
  assert(initialized && !direct && pkt != nullptr && buf != nullptr);
  bool queued = false;
  while (true) {
    if (!queued) queued = queuePacket(pkt, 0);
    V4LDecodedFrame frame;
    if (!pump(frame, -1)) return nullptr;
    if (!frame.buf) continue;

    VisionBuf *decoded = frame.buf;
    copyBuffer(decoded, buf);
    releaseFrame(decoded);
    return buf;
  }
}

void V4LDecoder::releaseFrame(VisionBuf *buf) {
  assert(buf >= cap_bufs && buf < cap_bufs + CAPTURE_BUFFER_COUNT);
  queueCaptureBuffer(buf - cap_bufs);
}

bool V4LDecoder::queuePacket(const AVPacket *pkt, uint64_t token) {
  int buf_index = getBufferUnlocked();
  return buf_index >= 0 && sendPacket(buf_index, pkt, token);
}

bool V4LDecoder::pump(V4LDecodedFrame &frame, int timeout_ms) {
  frame = {};
  int rc;
  while (true) {
    rc = poll(&pfd, 1, timeout_ms);
    if (rc < 0) {
      if (errno == EINTR) continue;
      LOGE("poll() error: %d", errno);
      return false;
    }
    break;
  }

  if (rc == 0) return true;

  int result;

  // Port changes must be handled before capture DQ so no old-format surface is
  // handed to a client after the driver has requested a capture flush.
  while ((result = handleEvent()) > 0) {}
  if (result < 0) return false;

  while ((result = handleOutput()) > 0) {}
  if (result < 0) return false;

  result = handleCapture(&frame);
  return result >= 0;
}

int V4LDecoder::handleCapture(V4LDecodedFrame *frame) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1] = {0};
  buf.type          = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory        = V4L2_MEMORY_USERPTR;
  buf.m.planes      = planes;
  buf.length        = 1;
  int err = HANDLE_EINTR(ioctl(this->fd, VIDIOC_DQBUF, &buf));
  if (err < 0 && errno == EAGAIN) return 0;
  if (err < 0) {
    LOGE("VIDIOC_DQBUF CAPTURE failed: %d", errno);
    return -1;
  }

  const bool has_payload = buf.m.planes[0].bytesused != 0;
  const bool eos = (buf.flags & V4L2_QCOM_BUF_FLAG_EOS) != 0;

  frame->buf = nullptr;
  if (!reconfigure_pending && has_payload) {
    frame->buf = &cap_bufs[buf.index];
    frame->token = (uint64_t)buf.timestamp.tv_sec * 1000000ULL + buf.timestamp.tv_usec;
  } else if (!reconfigure_pending && !eos) {
    queueCaptureBuffer(buf.index);
  }

  return 1;
}

bool V4LDecoder::subscribeEvents() {
  for (uint32_t event : subscriptions) {
    struct v4l2_event_subscription sub = { .type = event};
    util::safe_ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub, "VIDIOC_SUBSCRIBE_EVENT failed");
  }
  return true;
}

bool V4LDecoder::setPlaneFormat(enum v4l2_buf_type type, uint32_t fourcc) {
  struct v4l2_format fmt = {.type = type};
  struct v4l2_pix_format_mplane *pix = &fmt.fmt.pix_mp;
  *pix = {
    .width = (__u32)this->w,
    .height = (__u32)this->h,
    .pixelformat = fourcc
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT failed");
  if (type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) {
    this->out_buf_size = pix->plane_fmt[0].sizeimage;
    for (int i = 0; i < OUTPUT_BUFFER_COUNT; i++) {
      this->out_bufs[i].allocate(this->out_buf_size);
      this->out_buf_flag[i] = false;
    }
    LOGD("Set output buffer size to %d, count %d, addr %p", this->out_buf_size, OUTPUT_BUFFER_COUNT, this->out_bufs[0].addr);
  } else if (type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    request_buffers(this->fd, type, CAPTURE_BUFFER_COUNT);
    util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt, "VIDIOC_G_FMT failed");
    const __u32 y_size    = pix->plane_fmt[0].sizeimage;
    const __u32 y_stride  = pix->plane_fmt[0].bytesperline;
    for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; i++) {
      size_t uv_offset = (size_t)y_stride * pix->height;
      size_t required = uv_offset + (y_stride * pix->height / 2); // enough for Y + UV. For linear NV12, UV plane starts at y_stride * height.
      size_t alloc_size = std::max<size_t>(y_size, required);
      this->cap_bufs[i].allocate(alloc_size);
      this->cap_bufs[i].init_yuv(pix->width, pix->height, y_stride, uv_offset);
    }
    LOGD("Set capture buffer size to %d, count %d, addr %p, extradata size %d",
      pix->plane_fmt[0].sizeimage, CAPTURE_BUFFER_COUNT, this->cap_bufs[0].addr, pix->plane_fmt[1].sizeimage);
  }
  return true;
}

bool V4LDecoder::setFPS(uint32_t fps) {
  struct v4l2_streamparm streamparam = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
  };
  streamparam.parm.output.timeperframe = {1, fps};
  util::safe_ioctl(fd, VIDIOC_S_PARM, &streamparam, "VIDIOC_S_PARM failed");
  return true;
}

bool V4LDecoder::restartCapture() {
  // stop if already initialized
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (this->initialized) {
    LOGD("Restarting capture, flushing buffers...");
    util::safe_ioctl(this->fd, VIDIOC_STREAMOFF, &type, "VIDIOC_STREAMOFF CAPTURE failed");
    struct v4l2_requestbuffers reqbuf = {.type = type, .memory = V4L2_MEMORY_USERPTR};
    util::safe_ioctl(this->fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
    for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; ++i) {
      this->cap_bufs[i].free();
      cap_bufs[i].~VisionBuf();
      new (&cap_bufs[i]) VisionBuf();
    }
  }
  // setup, start and queue capture buffers
  setDBP();
  setPlaneFormat(type, capture_format);
  if (direct) {
    struct v4l2_control ctrl = {
      .id = V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE,
      .value = OFFLINE_CORE_PLACEMENT_RATE,
    };
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL placement decode failed");
  }
  util::safe_ioctl(this->fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON CAPTURE failed");
  for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; ++i) {
    queueCaptureBuffer(i);
  }
  if (direct) {
    struct v4l2_control ctrl = {
      .id = V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE,
      .value = INT_MAX,
    };
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL turbo decode failed");
  }

  return true;
}

bool V4LDecoder::queueCaptureBuffer(int i) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1] = {0};

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory = V4L2_MEMORY_USERPTR;
  buf.index = i;
  buf.m.planes = planes;
  buf.length = 1;
  // decoded frame plane
  planes[0].m.userptr     = (unsigned long)this->cap_bufs[i].addr; // no security
  planes[0].length        = this->cap_bufs[i].len;
  planes[0].reserved[0]   = this->cap_bufs[i].fd; // ION fd
  planes[0].reserved[1]   = 0;
  planes[0].bytesused     = this->cap_bufs[i].len;
  planes[0].data_offset   = 0;
  util::safe_ioctl(this->fd, VIDIOC_QBUF, &buf, "VIDIOC_QBUF failed");
  return true;
}

bool V4LDecoder::queueOutputBuffer(int i, size_t size, uint64_t token) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1] = {0};

  buf.type                = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buf.memory              = V4L2_MEMORY_USERPTR;
  buf.index               = i;
  buf.flags               = V4L2_BUF_FLAG_TIMESTAMP_COPY;
  buf.timestamp.tv_sec    = token / 1000000ULL;
  buf.timestamp.tv_usec   = token % 1000000ULL;
  buf.m.planes            = planes;
  buf.length              = 1;
  // decoded frame plane
  planes[0].m.userptr     = (unsigned long)this->out_bufs[i].addr;
  planes[0].length        = this->out_buf_size;
  planes[0].reserved[0]   = this->out_bufs[i].fd; // ION fd
  planes[0].reserved[1]   = 0;
  planes[0].bytesused     = size;
  planes[0].data_offset   = 0;
  assert(this->out_buf_size % 4096 == 0);               // ditto for size

  util::safe_ioctl(this->fd, VIDIOC_QBUF, &buf, "VIDIOC_QBUF failed");
  this->out_buf_flag[i] = true; // mark as queued
  return true;
}

bool V4LDecoder::setDBP() {
  struct v4l2_ext_control control[2] = {0};
  struct v4l2_ext_controls controls = {0};
  control[0].id           = V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_MODE;
  control[0].value        = 1; // V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_SECONDARY
  control[1].id           = V4L2_CID_MPEG_VIDC_VIDEO_DPB_COLOR_FORMAT;
  control[1].value        = 0; // V4L2_MPEG_VIDC_VIDEO_DPB_COLOR_FMT_NONE
  controls.count          = 2;
  controls.ctrl_class     = V4L2_CTRL_CLASS_MPEG;
  controls.controls       = control;
  util::safe_ioctl(fd, VIDIOC_S_EXT_CTRLS, &controls, "VIDIOC_S_EXT_CTRLS failed");
  return true;
}

bool V4LDecoder::sendPacket(int buf_index, const AVPacket *pkt, uint64_t token) {
  assert(buf_index >= 0 && buf_index < OUTPUT_BUFFER_COUNT);
  assert(pkt != nullptr && pkt->data != nullptr && pkt->size > 0);
  assert((size_t)pkt->size <= (size_t)this->out_buf_size);
  // Prepare output buffer
  uint8_t * data = (uint8_t *)this->out_bufs[buf_index].addr;
  memcpy(data, pkt->data, pkt->size);
  queueOutputBuffer(buf_index, pkt->size, token);
  return true;
}

int V4LDecoder::getBufferUnlocked() {
  for (int i = 0; i < OUTPUT_BUFFER_COUNT; i++) {
    if (!out_buf_flag[i]) {
      return i;
    }
  }
  return -1;
}


int V4LDecoder::handleOutput() {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1];
  buf.type      = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buf.memory    = V4L2_MEMORY_USERPTR;
  buf.m.planes  = planes;
  buf.length    = 1;
  int err = HANDLE_EINTR(ioctl(this->fd, VIDIOC_DQBUF, &buf));
  if (err < 0 && errno == EAGAIN) return 0;
  if (err < 0) {
    LOGE("VIDIOC_DQBUF OUTPUT failed: %d", errno);
    return -1;
  }
  this->out_buf_flag[buf.index] = false; // mark as not queued
  return 1;
}

int V4LDecoder::handleEvent() {
  // dequeue event
  struct v4l2_event event = {0};
  int err = HANDLE_EINTR(ioctl(this->fd, VIDIOC_DQEVENT, &event));
  if (err < 0 && (errno == EAGAIN || errno == ENOENT)) return 0;
  if (err < 0) {
    LOGE("VIDIOC_DQEVENT failed: %d", errno);
    return -1;
  }
  switch (event.type) {
    case V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_INSUFFICIENT: {
      unsigned int *ptr     = (unsigned int *)event.u.data;
      unsigned int height   = ptr[0];
      unsigned int width    = ptr[1];
      this->w               = width;
      this->h               = height;
      LOGD("Port Reconfig received insufficient, new size %ux%u, flushing capture bufs...", width, height); // This is normal
      struct v4l2_decoder_cmd dec;
      dec.flags = V4L2_QCOM_CMD_FLUSH_CAPTURE;
      dec.cmd = V4L2_QCOM_CMD_FLUSH;
      util::safe_ioctl(this->fd, VIDIOC_DECODER_CMD, &dec, "VIDIOC_DECODER_CMD FLUSH_CAPTURE failed");
      this->reconfigure_pending = true;
      LOGD("Waiting for flush done event to reconfigure capture queue");
      break;
    }

    case V4L2_EVENT_MSM_VIDC_FLUSH_DONE: {
      unsigned int *ptr   = (unsigned int *)event.u.data;
      unsigned int flags  = ptr[0];
      if (flags & V4L2_QCOM_CMD_FLUSH_CAPTURE) {
        if (this->reconfigure_pending) {
          this->restartCapture();
          this->reconfigure_pending = false;
        }
      }
      break;
    }
    default:
      break;
  }
  return 1;
}

void V4LDecoder::sendEOS() {
  struct v4l2_decoder_cmd command = { .cmd = V4L2_DEC_CMD_STOP };
  util::safe_ioctl(fd, VIDIOC_DECODER_CMD, &command, "VIDIOC_DECODER_CMD STOP failed");
}
