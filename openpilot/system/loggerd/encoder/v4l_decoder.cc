#include "system/loggerd/encoder/v4l_decoder.h"

#include <assert.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "system/loggerd/encoder/v4l_device.h"
#include "common/swaglog.h"
#include "common/util.h"

static void copyBuffer(VisionBuf *src_buf, VisionBuf *dst_buf) {
  int err = src_buf->begin_cpu_access(false);
  assert(err == 0);
  // Copy Y plane
  memcpy(dst_buf->y, src_buf->y, src_buf->height * src_buf->stride);
  // Copy UV plane
  memcpy(dst_buf->uv, src_buf->uv, src_buf->height / 2 * src_buf->stride);
  err = src_buf->end_cpu_access(false);
  assert(err == 0);
}

static void request_buffers(int fd, v4l2_buf_type buf_type, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = V4L2_MEMORY_DMABUF
  };
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
  // the driver may give more, but every index that's queued has to exist
  if (reqbuf.count < count) {
    LOGE("VIDIOC_REQBUFS gave %u buffers, %u are needed", reqbuf.count, count);
    throw std::runtime_error("VIDIOC_REQBUFS gave too few buffers");
  }
}

V4LDecoder::V4LDecoder() {
  wake_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (wake_fd < 0) throw std::runtime_error("eventfd failed");
}

V4LDecoder::~V4LDecoder() {
  if (fd >= 0) {
    close(fd);
  }
  close(wake_fd);
  // after close, the driver has released them
  for (auto &buf : out_bufs) {
    if (buf.addr) buf.free();
  }
  for (auto &buf : cap_bufs) {
    if (buf.addr) buf.free();
  }
}

bool V4LDecoder::init(size_t width, size_t height, uint64_t codec, bool direct_mode) {
  this->w = width;
  this->h = height;
  this->direct = direct_mode;
  this->fd = open_video_device("qcom-venus-decoder");
  if (fd < 0) {
    LOGE("failed to open the video decoder");
    return false;
  }
  struct v4l2_event_subscription sub = { .type = V4L2_EVENT_SOURCE_CHANGE };
  util::safe_ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub, "VIDIOC_SUBSCRIBE_EVENT failed");
  setOutputFormat(codec); // Also allocates the output buffers
  setFPS(FPS);
  v4l2_buf_type out_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  request_buffers(fd, out_type, OUTPUT_BUFFER_COUNT);
  util::safe_ioctl(fd, VIDIOC_STREAMON, &out_type, "VIDIOC_STREAMON OUTPUT failed");
  // the capture queue is set up once the header is decoded
  pfds[0] = {fd, 0, 0};
  pfds[1] = {wake_fd, POLLIN, 0};

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
  // a drained capture queue is freed once the last of its frames is released, they aren't queued again
  if (!drained) queueCaptureBuffer(buf - cap_bufs);
  --held;
  // pump() may be waiting for a capture buffer, or for the drained queue's last frame
  uint64_t one = 1;
  if (write(wake_fd, &one, sizeof(one)) < 0) LOGE("eventfd write failed: %d", errno);
}

bool V4LDecoder::queuePacket(const AVPacket *pkt, uint64_t token) {
  int buf_index = getBufferUnlocked();
  return buf_index >= 0 && sendPacket(buf_index, pkt, token);
}

bool V4LDecoder::pump(V4LDecodedFrame &frame, int timeout_ms) {
  frame = {};
  if (drained && held == 0) {
    drained = false;
    reconfigure_pending = false;
    restartCapture();
  }
  // the drained capture queue reports POLLIN until it restarts
  pfds[0].events = drained ? (POLLOUT | POLLWRNORM | POLLPRI) : (POLLIN | POLLOUT | POLLWRNORM | POLLRDNORM | POLLPRI);
  int rc;
  while (true) {
    rc = poll(pfds, 2, timeout_ms);
    if (rc < 0) {
      if (errno == EINTR) continue;
      LOGE("poll() error: %d", errno);
      return false;
    }
    // a mem2mem device returns POLLERR at once while neither queue has a buffer queued. then nothing completes until
    // releaseFrame() queues one from another thread, which writes the wake fd, so wait for that or an event
    if (pfds[0].revents == POLLERR && pfds[0].events != POLLPRI) {
      pfds[0].events = POLLPRI;
      continue;
    }
    break;
  }

  if (rc == 0) return true;
  if (pfds[1].revents & POLLIN) {
    uint64_t n;
    if (read(wake_fd, &n, sizeof(n)) < 0) LOGE("eventfd read failed: %d", errno);
  }

  // POLLERR alone isn't a failure: a mem2mem device reports it while neither queue has buffers. a failed decoder (Venus
  // puts the queues in an error state after a session error or a firmware crash) fails the DQBUFs with EIO, so the
  // caller sees pump() fail
  int result;

  // Port changes must be handled before capture DQ so no old-format surface is
  // handed to a client after the driver has requested a capture flush.
  while ((result = handleEvent()) > 0) {}
  if (result < 0) return false;

  while ((result = handleOutput()) > 0) {}
  if (result < 0) return false;

  result = capture_started ? handleCapture(&frame) : 0;
  return result >= 0;
}

int V4LDecoder::handleCapture(V4LDecodedFrame *frame) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1] = {0};
  buf.type          = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory        = V4L2_MEMORY_DMABUF;
  buf.m.planes      = planes;
  buf.length        = 1;
  int err = HANDLE_EINTR(ioctl(this->fd, VIDIOC_DQBUF, &buf));
  // EPIPE: the LAST buffer was already dequeued
  if (err < 0 && (errno == EAGAIN || errno == EPIPE)) return 0;
  if (err < 0) {
    LOGE("VIDIOC_DQBUF CAPTURE failed: %d", errno);
    return -1;
  }

  // every buffer with a payload is a frame, including those between a resolution change and its LAST buffer
  if (buf.m.planes[0].bytesused != 0) {
    frame->buf = &cap_bufs[buf.index];
    frame->token = (uint64_t)buf.timestamp.tv_sec * 1000000ULL + buf.timestamp.tv_usec;
    ++held;
  } else if (!(buf.flags & V4L2_BUF_FLAG_LAST)) {
    queueCaptureBuffer(buf.index);
  }
  // LAST ends the old format's frames after a resolution change, else it's the end of the stream
  if ((buf.flags & V4L2_BUF_FLAG_LAST) && reconfigure_pending) {
    drained = true;
  }
  return 1;
}

void V4LDecoder::setOutputFormat(uint32_t fourcc) {
  struct v4l2_format fmt = {.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE};
  fmt.fmt.pix_mp = {
    .width = (__u32)this->w,
    .height = (__u32)this->h,
    .pixelformat = fourcc
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT failed");
  this->out_buf_size = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
  for (int i = 0; i < OUTPUT_BUFFER_COUNT; i++) {
    this->out_bufs[i].allocate(this->out_buf_size);
    this->out_buf_flag[i] = false;
  }
  LOGD("Set output buffer size to %d, count %d, addr %p", this->out_buf_size, OUTPUT_BUFFER_COUNT, this->out_bufs[0].addr);
}

bool V4LDecoder::setFPS(uint32_t fps) {
  struct v4l2_streamparm streamparam = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
  };
  streamparam.parm.output.timeperframe = {1, fps};
  util::safe_ioctl(fd, VIDIOC_S_PARM, &streamparam, "VIDIOC_S_PARM failed");
  return true;
}

void V4LDecoder::restartCapture() {
  if (capture_started) {
    LOGD("Restarting capture, flushing buffers...");
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    util::safe_ioctl(this->fd, VIDIOC_STREAMOFF, &type, "VIDIOC_STREAMOFF CAPTURE failed");
    struct v4l2_requestbuffers reqbuf = {.type = type, .memory = V4L2_MEMORY_DMABUF};
    util::safe_ioctl(this->fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
    for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; ++i) {
      this->cap_bufs[i].free();
      cap_bufs[i].~VisionBuf();
      new (&cap_bufs[i]) VisionBuf();
    }
    this->capture_started = false;
  }
  // the coded size is aligned, the visible size is the compose rectangle. it's read before S_FMT, which resets it to
  // the coded size
  struct v4l2_selection sel = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .target = V4L2_SEL_TGT_COMPOSE };
  util::safe_ioctl(fd, VIDIOC_G_SELECTION, &sel, "VIDIOC_G_SELECTION failed");
  this->w = sel.r.width;
  this->h = sel.r.height;
  // keep the decoded size from the stream header
  struct v4l2_format fmt = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE};
  util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt, "VIDIOC_G_FMT failed");
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT failed");

  struct v4l2_pix_format_mplane *pix = &fmt.fmt.pix_mp;
  request_buffers(this->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, CAPTURE_BUFFER_COUNT);
  util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt, "VIDIOC_G_FMT failed");
  const __u32 y_size    = pix->plane_fmt[0].sizeimage;
  const __u32 y_stride  = pix->plane_fmt[0].bytesperline;
  for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; i++) {
    size_t uv_offset = (size_t)y_stride * pix->height;
    size_t required = uv_offset + (y_stride * pix->height / 2); // enough for Y + UV. For linear NV12, UV plane starts at y_stride * height.
    size_t alloc_size = std::max<size_t>(y_size, required);
    this->cap_bufs[i].allocate(alloc_size);
    this->cap_bufs[i].init_yuv(w, h, y_stride, uv_offset);
    // the encoder queues a decoded frame at its buffer index
    this->cap_bufs[i].idx = i;
  }
  LOGD("Set capture buffer size to %d, count %d, addr %p", pix->plane_fmt[0].sizeimage, CAPTURE_BUFFER_COUNT, this->cap_bufs[0].addr);

  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  util::safe_ioctl(this->fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON CAPTURE failed");
  for (size_t i = 0; i < CAPTURE_BUFFER_COUNT; ++i) {
    queueCaptureBuffer(i);
  }
  this->capture_started = true;
  // the caller asked for the end of the stream while there was no capture queue to end
  if (eos_pending) sendEOS();
}

bool V4LDecoder::queueCaptureBuffer(int i) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1] = {0};

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory = V4L2_MEMORY_DMABUF;
  buf.index = i;
  buf.m.planes = planes;
  buf.length = 1;
  // decoded frame plane
  planes[0].m.fd          = this->cap_bufs[i].fd;
  planes[0].length        = this->cap_bufs[i].len;
  planes[0].bytesused     = this->cap_bufs[i].len;
  planes[0].data_offset   = 0;
  util::safe_ioctl(this->fd, VIDIOC_QBUF, &buf, "VIDIOC_QBUF failed");
  return true;
}

bool V4LDecoder::queueOutputBuffer(int i, size_t size, uint64_t token) {
  struct v4l2_buffer buf = {0};
  struct v4l2_plane planes[1] = {0};

  buf.type                = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buf.memory              = V4L2_MEMORY_DMABUF;
  buf.index               = i;
  buf.flags               = V4L2_BUF_FLAG_TIMESTAMP_COPY;
  buf.timestamp.tv_sec    = token / 1000000ULL;
  buf.timestamp.tv_usec   = token % 1000000ULL;
  buf.m.planes            = planes;
  buf.length              = 1;
  // decoded frame plane
  planes[0].m.fd          = this->out_bufs[i].fd;
  planes[0].length        = this->out_buf_size;
  planes[0].bytesused     = size;
  planes[0].data_offset   = 0;
  assert(this->out_buf_size % 4096 == 0);               // ditto for size

  util::safe_ioctl(this->fd, VIDIOC_QBUF, &buf, "VIDIOC_QBUF failed");
  this->out_buf_flag[i] = true; // mark as queued
  return true;
}

bool V4LDecoder::sendPacket(int buf_index, const AVPacket *pkt, uint64_t token) {
  assert(buf_index >= 0 && buf_index < OUTPUT_BUFFER_COUNT);
  assert(pkt != nullptr && pkt->data != nullptr && pkt->size > 0);
  assert((size_t)pkt->size <= (size_t)this->out_buf_size);
  // Prepare output buffer
  VisionBuf *buf = &this->out_bufs[buf_index];
  int err = buf->begin_cpu_access(true);
  assert(err == 0);
  memcpy(buf->addr, pkt->data, pkt->size);
  err = buf->end_cpu_access(true);
  assert(err == 0);
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
  buf.memory    = V4L2_MEMORY_DMABUF;
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
  struct v4l2_event event = {0};
  int err = HANDLE_EINTR(ioctl(this->fd, VIDIOC_DQEVENT, &event));
  if (err < 0 && (errno == EAGAIN || errno == ENOENT)) return 0;
  if (err < 0) {
    LOGE("VIDIOC_DQEVENT failed: %d", errno);
    return -1;
  }
  if (event.type == V4L2_EVENT_SOURCE_CHANGE && (event.u.src_change.changes & V4L2_EVENT_SRC_CH_RESOLUTION)) {
    if (!capture_started) {
      restartCapture();
    } else {
      // the decoder returns the remaining frames, then a LAST buffer
      LOGD("Source change, draining capture bufs...");
      reconfigure_pending = true;
    }
  }
  return 1;
}

void V4LDecoder::sendEOS() {
  // the decoder can't stop before the capture queue is set up
  eos_pending = !capture_started || reconfigure_pending;
  if (eos_pending) return;
  struct v4l2_decoder_cmd command = { .cmd = V4L2_DEC_CMD_STOP };
  util::safe_ioctl(fd, VIDIOC_DECODER_CMD, &command, "VIDIOC_DECODER_CMD STOP failed");
}
