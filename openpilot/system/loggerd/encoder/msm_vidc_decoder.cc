#include "system/loggerd/encoder/msm_vidc_decoder.h"

#include <climits>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "system/loggerd/encoder/msm_vidc.h"
#include "common/swaglog.h"
#include "common/util.h"

constexpr int OFFLINE_CORE_PLACEMENT_RATE = 80 << 16;

// echo "0xFFFF" > /sys/kernel/debug/msm_vidc/debug_level

uint32_t MsmVidcDecoder::direct_format() const {
  return V4L2_PIX_FMT_NV12_UBWC;
}

int MsmVidcDecoder::openDevice() {
  return open("/dev/video32", O_RDWR | O_NONBLOCK, 0);
}

void MsmVidcDecoder::subscribeEvents() {
  for (uint32_t event : {V4L2_EVENT_MSM_VIDC_FLUSH_DONE, V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_INSUFFICIENT}) {
    struct v4l2_event_subscription sub = { .type = event};
    util::safe_ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub, "VIDIOC_SUBSCRIBE_EVENT failed");
  }
}

void MsmVidcDecoder::setControls() {
  if (!direct) return;
  struct v4l2_control ctrls[] = {
    // A finite real-time load lets the driver place decode and encode on separate cores.
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE, .value = OFFLINE_CORE_PLACEMENT_RATE },
    { .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_ENABLE },
  };
  for (auto ctrl : ctrls) {
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL offline decode failed");
  }
}

void MsmVidcDecoder::restartCapture() {
  // stop if already started
  if (capture_started) stopCapture();
  // setup, start and queue capture buffers
  setDBP();
  struct v4l2_format fmt = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE};
  fmt.fmt.pix_mp = {
    .width = (__u32)this->w,
    .height = (__u32)this->h,
    .pixelformat = capture_format,
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT failed");
  allocateCapture(fmt);
  if (direct) {
    struct v4l2_control ctrl = {
      .id = V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE,
      .value = OFFLINE_CORE_PLACEMENT_RATE,
    };
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL placement decode failed");
  }
  startCapture();
  if (direct) {
    struct v4l2_control ctrl = {
      .id = V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE,
      .value = INT_MAX,
    };
    util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL turbo decode failed");
  }
}

void MsmVidcDecoder::handleCapture(const v4l2_buffer &buf, V4LDecodedFrame *frame) {
  const bool has_payload = buf.m.planes[0].bytesused != 0;
  const bool eos = (buf.flags & V4L2_QCOM_BUF_FLAG_EOS) != 0;
  // FLUSH_CAPTURE returns the old capture buffers empty, they're freed when the capture queue restarts
  if (!reconfigure_pending && has_payload) {
    frame->buf = &cap_bufs[buf.index];
    frame->token = (uint64_t)buf.timestamp.tv_sec * 1000000ULL + buf.timestamp.tv_usec;
  } else if (!reconfigure_pending && !eos) {
    queueCaptureBuffer(buf.index);
  }
}

int MsmVidcDecoder::handleEvent() {
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

void MsmVidcDecoder::writePacket(VisionBuf *buf, const AVPacket *pkt) {
  // msm_vidc does the cache maintenance of the ION buffers it's queued itself
  memcpy(buf->addr, pkt->data, pkt->size);
}

void MsmVidcDecoder::setPlane(struct v4l2_plane *plane, const VisionBuf *buf) {
  plane->m.userptr = (unsigned long)buf->addr; // no security
  plane->reserved[0] = buf->fd; // ION fd
  plane->reserved[1] = 0;
}

void MsmVidcDecoder::setDBP() {
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
}
