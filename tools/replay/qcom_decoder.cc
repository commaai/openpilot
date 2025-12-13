#include "qcom_decoder.h"

#include <assert.h>
#include "third_party/linux/include/v4l2-controls.h"
#include <linux/videodev2.h>


#include "common/swaglog.h"
#include "common/util.h"

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

MsmVidc::~MsmVidc() {
  if (fd > 0) {
    close(fd);
  }
}

bool MsmVidc::init(const char* dev, size_t width, size_t height, uint64_t codec) {
  LOG("Initializing msm_vidc device %s", dev);
  this->w = width;
  this->h = height;
  this->fd = open(dev, O_RDWR, 0);
  if (fd < 0) {
    LOGE("failed to open video device %s", dev);
    return false;
  }
  subscribeEvents();
  v4l2_buf_type out_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  setPlaneFormat(out_type, V4L2_PIX_FMT_HEVC); // Also allocates the output buffer
  setFPS(FPS);
  request_buffers(fd, out_type, OUTPUT_BUFFER_COUNT);
  util::safe_ioctl(fd, VIDIOC_STREAMON, &out_type, "VIDIOC_STREAMON OUTPUT failed");
  restartCapture();
  setupPolling();

  this->initialized = true;
  return true;
}

VisionBuf* MsmVidc::decodeFrame(AVPacket *pkt, VisionBuf *buf) {
  assert(initialized && (pkt != nullptr) && (buf != nullptr));

  this->frame_ready = false;
  this->current_output_buf = buf;
  bool sent_packet = false;

  while (!this->frame_ready) {
    if (!sent_packet) {
      int buf_index = getBufferUnlocked();
      if (buf_index >= 0) {
        assert(buf_index < out_buf_cnt);
        sendPacket(buf_index, pkt);
        sent_packet = true;
      }
    }

    if (poll(pfd, nfds, -1) < 0) {
      LOGE("poll() error: %d", errno);
      return nullptr;
    }

    if (VisionBuf* result = processEvents()) {
      return result;
    }
  }

  return buf;
}

VisionBuf* MsmVidc::processEvents() {
  for (int idx = 0; idx < nfds; idx++) {
    short revents = pfd[idx].revents;
    if (!revents) continue;

    if (idx == ev[EV_VIDEO]) {
      if (revents & (POLLIN | POLLRDNORM)) {
        VisionBuf *result = handleCapture();
        if (result == this->current_output_buf) {
          this->frame_ready = true;
        }
      }
      if (revents & (POLLOUT | POLLWRNORM)) {
        handleOutput();
      }
      if (revents & POLLPRI) {
        handleEvent();
      }
    } else {
      LOGE("Unexpected event on fd %d", pfd[idx].fd);
    }
  }
  return nullptr;
}

VisionBuf* MsmVidc::handleCapture() {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1] = {0};
  buf.type          = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory        = V4L2_MEMORY_USERPTR;
  buf.m.planes      = planes;
  buf.length        = 1;
  util::safe_ioctl(this->fd, VIDIOC_DQBUF, &buf, "VIDIOC_DQBUF CAPTURE failed");

  if (this->reconfigure_pending || buf.m.planes[0].bytesused == 0) {
    return nullptr;
  }

  copyBuffer(&cap_bufs[buf.index], this->current_output_buf);
  queueCaptureBuffer(buf.index);
  return this->current_output_buf;
}

bool MsmVidc::subscribeEvents() {
  for (uint32_t event : subscriptions) {
    struct v4l2_event_subscription sub = { .type = event};
    util::safe_ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub, "VIDIOC_SUBSCRIBE_EVENT failed");
  }
  return true;
}

bool MsmVidc::setPlaneFormat(enum v4l2_buf_type type, uint32_t fourcc) {
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
    int ion_size = this->out_buf_size * OUTPUT_BUFFER_COUNT; // Output (input) buffers are ION buffer.
    this->out_buf.allocate(ion_size); // mmap rw
    for (int i = 0; i < OUTPUT_BUFFER_COUNT; i++) {
      this->out_buf_off[i] = i * this->out_buf_size;
      this->out_buf_addr[i] = (char *)this->out_buf.addr + this->out_buf_off[i];
      this->out_buf_flag[i] = false;
    }
    LOGD("Set output buffer size to %d, count %d, addr %p", this->out_buf_size, OUTPUT_BUFFER_COUNT, this->out_buf.addr);
  } else if (type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    request_buffers(this->fd, type, CAPTURE_BUFFER_COUNT);
    util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt, "VIDIOC_G_FMT failed");
    const __u32 y_size    = pix->plane_fmt[0].sizeimage;
    const __u32 y_stride  = pix->plane_fmt[0].bytesperline;
    for (int i = 0; i < CAPTURE_BUFFER_COUNT; i++) {
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

bool MsmVidc::setFPS(uint32_t fps) {
  struct v4l2_streamparm streamparam = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .parm.output.timeperframe = {1, fps}
  };
  util::safe_ioctl(fd, VIDIOC_S_PARM, &streamparam, "VIDIOC_S_PARM failed");
  return true;
}

bool MsmVidc::restartCapture() {
  // stop if already initialized
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (this->initialized) {
    LOGD("Restarting capture, flushing buffers...");
    util::safe_ioctl(this->fd, VIDIOC_STREAMOFF, &type, "VIDIOC_STREAMOFF CAPTURE failed");
    struct v4l2_requestbuffers reqbuf = {.type = type, .memory = V4L2_MEMORY_USERPTR};
    util::safe_ioctl(this->fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
    for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; ++i) {
      this->cap_bufs[i].free();
      this->cap_buf_flag[i] = false; // mark as not queued
      cap_bufs[i].~VisionBuf();
      new (&cap_bufs[i]) VisionBuf();
    }
  }
  // setup, start and queue capture buffers
  setDBP();
  setPlaneFormat(type, V4L2_PIX_FMT_NV12);
  util::safe_ioctl(this->fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON CAPTURE failed");
  for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; ++i) {
    queueCaptureBuffer(i);
  }

  return true;
}

bool MsmVidc::queueCaptureBuffer(int i) {
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
  this->cap_buf_flag[i] = true; // mark as queued
  return true;
}

bool MsmVidc::queueOutputBuffer(int i, size_t size) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1] = {0};

  buf.type                = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buf.memory              = V4L2_MEMORY_USERPTR;
  buf.index               = i;
  buf.m.planes            = planes;
  buf.length              = 1;
  // decoded frame plane
  planes[0].m.userptr     = (unsigned long)this->out_buf_off[i]; // check this
  planes[0].length        = this->out_buf_size;
  planes[0].reserved[0]   = this->out_buf.fd; // ION fd
  planes[0].reserved[1]   = 0;
  planes[0].bytesused     = size;
  planes[0].data_offset   = 0;
  assert((this->out_buf_off[i] & 0xfff) == 0);          // must be 4 KiB aligned
  assert(this->out_buf_size % 4096 == 0);               // ditto for size

  util::safe_ioctl(this->fd, VIDIOC_QBUF, &buf, "VIDIOC_QBUF failed");
  this->out_buf_flag[i] = true; // mark as queued
  return true;
}

bool MsmVidc::setDBP() {
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

bool MsmVidc::setupPolling() {
  // Initialize poll array
  pfd[EV_VIDEO] = {fd, POLLIN | POLLOUT | POLLWRNORM | POLLRDNORM | POLLPRI, 0};
  ev[EV_VIDEO] = EV_VIDEO;
  nfds = 1;
  return true;
}

bool MsmVidc::sendPacket(int buf_index, AVPacket *pkt) {
  assert(buf_index >= 0 && buf_index < out_buf_cnt);
  assert(pkt != nullptr && pkt->data != nullptr && pkt->size > 0);
  // Prepare output buffer
  memset(this->out_buf_addr[buf_index], 0, this->out_buf_size);
  uint8_t * data = (uint8_t *)this->out_buf_addr[buf_index];
  memcpy(data, pkt->data, pkt->size);
  queueOutputBuffer(buf_index, pkt->size);
  return true;
}

int MsmVidc::getBufferUnlocked() {
  for (int i = 0; i < this->out_buf_cnt; i++) {
    if (!out_buf_flag[i]) {
      return i;
    }
  }
  return -1;
}


bool MsmVidc::handleOutput() {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1];
  buf.type      = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buf.memory    = V4L2_MEMORY_USERPTR;
  buf.m.planes  = planes;
  buf.length    = 1;
  util::safe_ioctl(this->fd, VIDIOC_DQBUF, &buf, "VIDIOC_DQBUF OUTPUT failed");
  this->out_buf_flag[buf.index] = false; // mark as not queued
  return true;
}

bool MsmVidc::handleEvent() {
  // dequeue event
  struct v4l2_event event = {0};
  util::safe_ioctl(this->fd, VIDIOC_DQEVENT, &event, "VIDIOC_DQEVENT failed");
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
  return true;
}