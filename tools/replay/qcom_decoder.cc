#include "qcom_decoder.h"

#include <assert.h>
#include "third_party/linux/include/v4l2-controls.h"
#include <linux/videodev2.h>

#include "common/swaglog.h"
#include "common/util.h"

MsmVidc::~MsmVidc() {
  if (fd > 0) {
    close(fd);
  }
}

bool MsmVidc::init(const char* dev, size_t width, size_t height, uint64_t codec) {
  LOG("Initializing msm_vidc device %s", dev);
  w = width;
  h = height;
  
  fd = open(dev, O_RDWR, 0);
  if (fd < 0) {
    LOGE("failed to open video device %s", dev);
    return false;
  }

  // Setup output (input) format
  struct v4l2_format fmt = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .fmt.pix_mp = {
      .width = (__u32)width,
      .height = (__u32)height,
      .pixelformat = V4L2_PIX_FMT_HEVC
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT OUTPUT failed");

  // Setup capture (output) format  
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT CAPTURE failed");

  // Set decoder controls
  struct v4l2_ext_control controls[2] = {
    {.id = V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_MODE, .value = 1},
    {.id = V4L2_CID_MPEG_VIDC_VIDEO_DPB_COLOR_FORMAT, .value = 0}
  };
  struct v4l2_ext_controls ext_controls = {
    .count = 2,
    .ctrl_class = V4L2_CTRL_CLASS_MPEG,
    .controls = controls
  };
  util::safe_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ext_controls, "VIDIOC_S_EXT_CTRLS failed");

  // Allocate buffers
  struct v4l2_requestbuffers reqbuf = {
    .count = 1,
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .memory = V4L2_MEMORY_USERPTR
  };
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS OUTPUT failed");

  reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  reqbuf.count = 1;
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS CAPTURE failed");

  // Start streams
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON OUTPUT failed");
  
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON CAPTURE failed");

  // Allocate input buffer
  input_buf.allocate(1024 * 1024); // 1MB should be enough for most frames

  initialized = true;
  return true;
}

bool MsmVidc::decodeFrame(AVPacket *pkt, VisionBuf *output_buf) {
  assert(initialized && pkt && output_buf);
  
  if (pkt->size > input_buf.len) {
    LOGE("Packet too large: %d > %zu", pkt->size, input_buf.len);
    return false;
  }

  // Copy packet data to input buffer
  memcpy(input_buf.addr, pkt->data, pkt->size);

  // Queue input buffer
  struct v4l2_plane input_plane = {
    .bytesused = (__u32)pkt->size,
    .length = (__u32)input_buf.len,
    .m.userptr = reinterpret_cast<unsigned long>(input_buf.addr),
    .reserved = {(__u32)input_buf.fd, 0}
  };
  
  struct v4l2_buffer input_buffer = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .memory = V4L2_MEMORY_USERPTR,
    .index = 0,
    .length = 1,
    .m.planes = &input_plane
  };
  
  util::safe_ioctl(fd, VIDIOC_QBUF, &input_buffer, "VIDIOC_QBUF INPUT failed");

  // Queue output buffer
  struct v4l2_plane output_plane = {
    .length = (__u32)output_buf->len,
    .m.userptr = reinterpret_cast<unsigned long>(output_buf->addr),
    .reserved = {(__u32)output_buf->fd, 0}
  };
  
  struct v4l2_buffer output_buffer = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .memory = V4L2_MEMORY_USERPTR,
    .index = 0,
    .length = 1,
    .m.planes = &output_plane
  };
  
  util::safe_ioctl(fd, VIDIOC_QBUF, &output_buffer, "VIDIOC_QBUF OUTPUT failed");

  // Wait for completion and dequeue
  util::safe_ioctl(fd, VIDIOC_DQBUF, &input_buffer, "VIDIOC_DQBUF INPUT failed");
  util::safe_ioctl(fd, VIDIOC_DQBUF, &output_buffer, "VIDIOC_DQBUF OUTPUT failed");

  return output_buffer.m.planes[0].bytesused > 0;
}