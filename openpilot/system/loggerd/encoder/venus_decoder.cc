#include "system/loggerd/encoder/venus_decoder.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "system/loggerd/encoder/v4l_device.h"
#include "common/swaglog.h"
#include "common/util.h"

VenusDecoder::VenusDecoder() : V4LDecoder(V4L2_MEMORY_DMABUF) {
  wake_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (wake_fd < 0) throw std::runtime_error("eventfd failed");
}

VenusDecoder::~VenusDecoder() {
  close(wake_fd);
}

int VenusDecoder::openDevice() {
  return open_video_device("qcom-venus-decoder");
}

void VenusDecoder::subscribeEvents() {
  struct v4l2_event_subscription sub = { .type = V4L2_EVENT_SOURCE_CHANGE };
  util::safe_ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub, "VIDIOC_SUBSCRIBE_EVENT failed");
}

void VenusDecoder::restartCapture() {
  if (capture_started) stopCapture();
  // the coded size is aligned, the visible size is the compose rectangle. it's read before S_FMT, which resets it to
  // the coded size
  struct v4l2_selection sel = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .target = V4L2_SEL_TGT_COMPOSE };
  util::safe_ioctl(fd, VIDIOC_G_SELECTION, &sel, "VIDIOC_G_SELECTION failed");
  this->w = sel.r.width;
  this->h = sel.r.height;
  // keep the decoded size from the stream header
  struct v4l2_format fmt = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE};
  util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt, "VIDIOC_G_FMT failed");
  fmt.fmt.pix_mp.pixelformat = capture_format;
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT failed");
  allocateCapture(fmt, w, h);
  startCapture();
  // the caller asked for the end of the stream while there was no capture queue to end
  if (eos_pending) sendEOS();
}

void VenusDecoder::handleCapture(const v4l2_buffer &buf, V4LDecodedFrame *frame) {
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
}

void VenusDecoder::releaseFrame(VisionBuf *buf) {
  assert(buf >= cap_bufs && buf < cap_bufs + CAPTURE_BUFFER_COUNT);
  // a drained capture queue is freed once the last of its frames is released, they aren't queued again
  if (!drained) queueCaptureBuffer(buf - cap_bufs);
  --held;
  // pump() may be waiting for a capture buffer, or for the drained queue's last frame
  uint64_t one = 1;
  if (write(wake_fd, &one, sizeof(one)) < 0) LOGE("eventfd write failed: %d", errno);
}

void VenusDecoder::beforePoll() {
  if (drained && held == 0) {
    drained = false;
    reconfigure_pending = false;
    restartCapture();
  }
}

short VenusDecoder::pollEvents() const {
  // the drained capture queue reports POLLIN until it restarts
  return drained ? (POLLOUT | POLLWRNORM | POLLPRI) : V4LDecoder::pollEvents();
}

int VenusDecoder::handleEvent() {
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

void VenusDecoder::sendEOS() {
  // the decoder can't stop before the capture queue is set up
  eos_pending = !capture_started || reconfigure_pending;
  if (eos_pending) return;
  V4LDecoder::sendEOS();
}

void VenusDecoder::writePacket(VisionBuf *buf, const AVPacket *pkt) {
  int err = buf->begin_cpu_access(true);
  assert(err == 0);
  memcpy(buf->addr, pkt->data, pkt->size);
  err = buf->end_cpu_access(true);
  assert(err == 0);
}

void VenusDecoder::setPlane(struct v4l2_plane *plane, const VisionBuf *buf) {
  plane->m.fd = buf->fd;
}
