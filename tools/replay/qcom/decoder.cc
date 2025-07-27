#include <sys/signalfd.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <deque>

#include <assert.h>
#include "decoder.h"
#include "common/swaglog.h"
#include "common/util.h"
#include <cstdio>


static void checked_ioctl(int fd, unsigned long request, void *argp) {
  int ret = util::safe_ioctl(fd, request, argp);
  if (ret != 0) {
    LOGE("checked_ioctl failed with error %d (%d %lx %p)", errno, fd, request, argp);
    assert(0);
  }
}

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
    .count = count
  };
  checked_ioctl(fd, VIDIOC_REQBUFS, &reqbuf);
}

MsmVidc::~MsmVidc() {
  if (fd > 0) {
    close(fd);
  }
  if (sigfd > 0) {
    close(sigfd);
  }
}

bool MsmVidc::init(const char* dev,
                   size_t width, size_t height,
                   uint64_t codec) {
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
  setControls();
  checked_ioctl(fd, VIDIOC_STREAMON, &out_type);
  restartCapture();
  setupPolling();
  rotator.init();
  rotator.configUBWCtoNV12(width, height);
  this->initialized = true;
  return true;
}

VisionBuf* MsmVidc::decodeFrame(AVPacket *pkt, VisionBuf *buf) {
  // Decode a single frame, return the output buffer with the decoded frame.
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
      break;
    }

    // Handle events
    for (int idx = 0; idx < nfds; idx++) {
      short revents = pfd[idx].revents;
      if (!revents) {
        continue; // no events for this fd
      }

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

      } else if (idx == ev[EV_SIGNAL]) {
        handleSignal();
        return nullptr;
      } else {
        LOGE("Unexpected event on fd %d", pfd[idx].fd);
        continue;
      }
    }
  }
  return this->frame_ready ? buf : nullptr;
}

bool MsmVidc::subscribeEvents() {
  for (uint32_t event : subscriptions) {
    struct v4l2_event_subscription sub = { .type = event};
    checked_ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
  }
  return true;
}

bool MsmVidc::setPlaneFormat(enum v4l2_buf_type type, uint32_t fourcc) {
  struct v4l2_format fmt = {.type = type};
  struct v4l2_pix_format_mplane *pix = &fmt.fmt.pix_mp;
  *pix = { .width = (__u32)this->w, .height = (__u32)this->h, .pixelformat = fourcc };
  checked_ioctl(fd, VIDIOC_S_FMT, &fmt);
  if (type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) {
    this->out_buf_size = pix->plane_fmt[0].sizeimage;
    int ion_size = this->out_buf_size * OUTPUT_BUFFER_COUNT; // Output (input) buffers are ION buffer.
    this->out_buf.allocate_no_cache(ion_size); // mmap rw
    for (int i = 0; i < OUTPUT_BUFFER_COUNT; i++) {
      this->out_buf_off[i] = i * this->out_buf_size;
      this->out_buf_addr[i] = (char *)this->out_buf.addr + this->out_buf_off[i];
      this->out_buf_flag[i] = false;
    }
    LOGD("Set output buffer size to %d, count %d, addr %p", this->out_buf_size, OUTPUT_BUFFER_COUNT, this->out_buf.addr);
  } else if (type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    request_buffers(this->fd, type, CAPTURE_BUFFER_COUNT);
    checked_ioctl(fd, VIDIOC_G_FMT, &fmt);
    for (int i = 0; i < CAPTURE_BUFFER_COUNT; i++) {
      this->cap_bufs[i].allocate_no_cache(pix->plane_fmt[0].sizeimage);
      this->cap_bufs[i].init_yuv(pix->width, pix->height, pix->plane_fmt[0].bytesperline, 0);
    }
    this->cap_buf_format = pix->pixelformat;
    this->ext_buf.allocate_no_cache(pix->plane_fmt[1].sizeimage * CAPTURE_BUFFER_COUNT);
    for (int i = 0; i < CAPTURE_BUFFER_COUNT; i++) {
      size_t offset = i * pix->plane_fmt[1].sizeimage;
      this->ext_buf_off[i] = offset;
      this->ext_buf_addr[i] = (char *)this->ext_buf.addr + offset;
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
  checked_ioctl(fd, VIDIOC_S_PARM, &streamparam);
  return true;
}

bool MsmVidc::setControls() {
  struct v4l2_control control = {0};
	control.id = V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE;
	control.value = INT_MAX;
  checked_ioctl(fd, VIDIOC_S_CTRL, &control);
	control.id = V4L2_CID_MPEG_VIDC_VIDEO_CONCEAL_COLOR_8BIT;
	control.value = 0x000000ff;
  checked_ioctl(fd, VIDIOC_S_CTRL, &control);
	control.id = V4L2_CID_MPEG_VIDC_VIDEO_EXTRADATA;
	control.value = V4L2_MPEG_VIDC_EXTRADATA_INTERLACE_VIDEO;
  checked_ioctl(fd, VIDIOC_S_CTRL, &control);
	control.id = V4L2_CID_MPEG_VIDC_VIDEO_EXTRADATA;
	control.value = V4L2_MPEG_VIDC_EXTRADATA_OUTPUT_CROP;
  checked_ioctl(fd, VIDIOC_S_CTRL, &control);
	control.id = V4L2_CID_MPEG_VIDC_VIDEO_EXTRADATA;
	control.value = V4L2_MPEG_VIDC_EXTRADATA_ASPECT_RATIO;
  checked_ioctl(fd, VIDIOC_S_CTRL, &control);
	control.id = V4L2_CID_MPEG_VIDC_VIDEO_EXTRADATA;
	control.value = V4L2_MPEG_VIDC_EXTRADATA_FRAME_RATE;
  checked_ioctl(fd, VIDIOC_S_CTRL, &control);
  LOGD("Set controls: operating rate %d, conceal color 0x%08x, extradata interlace %d, output crop %d, aspect ratio %d, frame rate %d",
    control.value, control.value, control.value, control.value, control.value, control.value);
  return true;
}

bool MsmVidc::restartCapture() {
  // stop if already initialized
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (this->initialized) {
    LOGD("Restarting capture, flushing buffers...");
    checked_ioctl(this->fd, VIDIOC_STREAMOFF, &type);
    struct v4l2_requestbuffers reqbuf = {.memory = V4L2_MEMORY_USERPTR, .type = type};
    checked_ioctl(this->fd, VIDIOC_REQBUFS, &reqbuf);
    for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; ++i) {
      this->cap_bufs[i].free();
      this->cap_buf_flag[i] = false; // mark as not queued
      cap_bufs[i].~VisionBuf();
      new (&cap_bufs[i]) VisionBuf();
    }
  }
  // setup, start and queue capture buffers
  setDBP(V4L2_MPEG_VIDC_VIDEO_DPB_COLOR_FMT_NONE);
  setPlaneFormat(type, v4l2_fourcc('Q', '1', '2', '8'));
  checked_ioctl(this->fd, VIDIOC_STREAMON, &type);
  for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; ++i) {
    queueCaptureBuffer(i);
  }

  return true;
}

bool MsmVidc::queueCaptureBuffer(int i) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[CAP_PLANES] = {0};

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory = V4L2_MEMORY_USERPTR;
  buf.index = i;
  buf.m.planes = planes;
  buf.length = 2;
  // decoded frame plane
  planes[0].m.userptr = (unsigned long)this->cap_bufs[i].addr; // no security
  planes[0].length = this->cap_bufs[i].len;
  planes[0].reserved[0] = this->cap_bufs[i].fd; // ION fd
  planes[0].reserved[1] = 0;
  planes[0].bytesused = this->cap_bufs[i].len;
  planes[0].data_offset = 0;
  // extradata plane
  planes[1].m.userptr = (unsigned long)this->ext_buf_addr[i];
  planes[1].length = this->ext_buf.len;
  planes[1].reserved[0] = this->ext_buf.fd; // ION fd
  planes[1].reserved[1] = this->ext_buf_off[i]; // offset in the buffer
  planes[1].bytesused = 0;
  planes[1].data_offset = 0;
  checked_ioctl(this->fd, VIDIOC_QBUF, &buf);
  this->cap_buf_flag[i] = true; // mark as queued
  return true;
}

bool MsmVidc::queueOutputBuffer(int i, size_t size) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[OUT_PLANES] = {0};

  buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buf.memory = V4L2_MEMORY_USERPTR;
  buf.index = i;
  buf.m.planes = planes;
  buf.length = 1;
  // decoded frame plane
  planes[0].m.userptr = (unsigned long)this->out_buf_off[i]; // check this
  planes[0].length = this->out_buf_size;
  planes[0].reserved[0] = this->out_buf.fd; // ION fd
  planes[0].reserved[1] = 0;
  planes[0].bytesused = size;
  planes[0].data_offset = 0;
  assert((this->out_buf_off[i] & 0xfff) == 0);          // must be 4 KiB aligned
  assert(this->out_buf_size % 4096 == 0);               // ditto for size

  checked_ioctl(this->fd, VIDIOC_QBUF, &buf);
  this->out_buf_flag[i] = true; // mark as queued
  return true;
}

bool MsmVidc::setDBP(v4l2_mpeg_vidc_video_dpb_color_format format) {
  struct v4l2_ext_control control[2] = {0};
  struct v4l2_ext_controls controls = {0};

  control[0].id = V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_MODE;
  control[0].value = V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_PRIMARY;
  control[1].id = V4L2_CID_MPEG_VIDC_VIDEO_DPB_COLOR_FORMAT;
  control[1].value = format;
  controls.count = 2;
  controls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  controls.controls = control;

  checked_ioctl(fd, VIDIOC_S_EXT_CTRLS, &controls);
  LOGD("Set DBP format to %d", format);
  return true;
}

bool MsmVidc::setupPolling() {
  // setup polling
  sigset_t sigmask;
  sigemptyset(&sigmask);
	sigaddset(&sigmask, SIGINT);
	sigaddset(&sigmask, SIGTERM);
  this->sigfd = signalfd(-1, &sigmask, SFD_CLOEXEC);
  assert(sigfd > 0);
  sigprocmask(SIG_BLOCK, &sigmask, nullptr);
  nfds = 0;
  assert(this->fd > 0);
  pfd[nfds].fd = this->fd;
  pfd[nfds].events = POLLOUT | POLLWRNORM | POLLPRI;
  ev[EV_VIDEO] = nfds++;
  pfd[nfds].fd = this->sigfd;
  pfd[nfds].events = POLLIN;
  ev[EV_SIGNAL] = nfds++;
  pfd[ev[EV_VIDEO]].events |= POLLIN | POLLRDNORM;
  LOGD("Setup polling with %d fds", nfds);
  for (int i = 0; i < nfds; i++) {
    LOGD("Poll fd %d, events %d", pfd[i].fd, pfd[i].events);
  }
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

int MsmVidc::handleSignal() {
	struct signalfd_siginfo siginfo;
	sigset_t sigmask;
  if (read(this->sigfd, &siginfo, sizeof (siginfo)) < 0) {
		return -1;
	}
  sigemptyset(&sigmask);
	sigaddset(&sigmask, siginfo.ssi_signo);
	sigprocmask(SIG_UNBLOCK, &sigmask, NULL);
  // clean up
  LOGD("Received signal %d, cleaning up", siginfo.ssi_signo);
  if (fd > 0) {
    close(fd);
  }
  if (sigfd > 0) {
    close(sigfd);
  }
  return 0;
}

VisionBuf* MsmVidc::handleCapture() {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[CAP_PLANES] = {0};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory = V4L2_MEMORY_USERPTR;
  buf.m.planes = planes;
  buf.length = CAP_PLANES;
  checked_ioctl(this->fd, VIDIOC_DQBUF, &buf);

  if (buf.m.planes[0].bytesused) {
    static size_t cap_cnt = 0;
    cap_cnt++;
    if (cap_cnt % 240 == 0) {
      LOGD("Dequeued %zu capture buffers", cap_cnt);
    }
    if (!this->reconfigure_pending) {
      rotator.putFrame(&cap_bufs[buf.index]);
      VisionBuf *rotated = rotator.getFrame(100);
      queueCaptureBuffer(buf.index);
      if (rotated) {
        rotator.convertStride(rotated, this->current_output_buf);
        return this->current_output_buf;
      }
    }
  } else {
    LOGE("Dequeued empty capture buffer %d", buf.index);
  }
  return nullptr;
}

bool MsmVidc::handleOutput() {
  struct v4l2_buffer buf = {0};
	struct v4l2_plane planes[OUT_PLANES];

	buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
	buf.memory = V4L2_MEMORY_USERPTR;
	buf.m.planes = planes;
	buf.length = OUT_PLANES;

  checked_ioctl(this->fd, VIDIOC_DQBUF, &buf);
  this->out_buf_flag[buf.index] = false; // mark as not queued
  return true;
}

bool MsmVidc::handleEvent() {
  // dequeue event
  struct v4l2_event event = {0};
  checked_ioctl(this->fd, VIDIOC_DQEVENT, &event);
  switch (event.type) {
    case V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_INSUFFICIENT: {
      unsigned int *ptr = (unsigned int *)event.u.data;
      unsigned int height = ptr[0];
      unsigned int width = ptr[1];
      this->w = width;
      this->h = height;
      LOGD("Port Reconfig received insufficient, new size %ux%u, flushing capture bufs...", width, height); // This is normal
      struct v4l2_decoder_cmd dec;
      dec.flags = V4L2_QCOM_CMD_FLUSH_CAPTURE;
      dec.cmd = V4L2_QCOM_CMD_FLUSH;
      checked_ioctl(this->fd, VIDIOC_DECODER_CMD, &dec);
      this->reconfigure_pending = true;
      LOGD("Waiting for flush done event to reconfigure capture queue");
      break;
    }

    case V4L2_EVENT_MSM_VIDC_FLUSH_DONE: {
      unsigned int *ptr = (unsigned int *)event.u.data;
      unsigned int flags = ptr[0];
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