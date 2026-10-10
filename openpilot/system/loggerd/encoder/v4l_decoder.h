#pragma once

#include <linux/videodev2.h>
#include <poll.h>

#include "msgq/visionipc/visionbuf.h"

extern "C" {
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
}

#define V4L2_EVENT_MSM_VIDC_START (V4L2_EVENT_PRIVATE_START + 0x00001000)
#define V4L2_EVENT_MSM_VIDC_FLUSH_DONE (V4L2_EVENT_MSM_VIDC_START + 1)
#define V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_INSUFFICIENT (V4L2_EVENT_MSM_VIDC_START + 3)
#ifndef V4L2_CID_MPEG_MSM_VIDC_BASE
#define V4L2_CID_MPEG_MSM_VIDC_BASE 0x00992000
#endif
#ifndef V4L2_CID_MPEG_VIDC_VIDEO_DPB_COLOR_FORMAT
#define V4L2_CID_MPEG_VIDC_VIDEO_DPB_COLOR_FORMAT (V4L2_CID_MPEG_MSM_VIDC_BASE + 44)
#endif
#ifndef V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_MODE
#define V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_MODE (V4L2_CID_MPEG_MSM_VIDC_BASE + 22)
#endif
#ifndef V4L2_PIX_FMT_NV12_UBWC
#define V4L2_PIX_FMT_NV12_UBWC v4l2_fourcc('Q', '1', '2', '8')
#endif
#ifndef V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY
#define V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY (V4L2_CID_MPEG_MSM_VIDC_BASE + 52)
#define V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_ENABLE 0
#define V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE 1
#endif
#ifndef V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE
#define V4L2_CID_MPEG_VIDC_VIDEO_OPERATING_RATE (V4L2_CID_MPEG_MSM_VIDC_BASE + 53)
#endif
#define V4L2_QCOM_CMD_FLUSH_CAPTURE (1 << 1)
#define V4L2_QCOM_CMD_FLUSH (4)
#ifndef V4L2_QCOM_BUF_FLAG_EOS
#define V4L2_QCOM_BUF_FLAG_EOS 0x02000000
#endif

#define OUTPUT_BUFFER_COUNT 	8
#define CAPTURE_BUFFER_COUNT 	16
#define FPS 									20

struct V4LDecodedFrame {
  VisionBuf *buf = nullptr;
  uint64_t token = 0;
};

class V4LDecoder {
public:
  static constexpr const char *DEVICE = "/dev/video32";

  V4LDecoder() = default;
  ~V4LDecoder();

  bool init(const char* dev, size_t width, size_t height, uint64_t codec,
            bool direct_mode = false, uint32_t capture_fourcc = V4L2_PIX_FMT_NV12);
  VisionBuf* decodeFrame(AVPacket* pkt, VisionBuf* buf);
  // queuePacket() and pump() are single-threaded. releaseFrame() may be called
  // from a consumer thread after a direct capture surface is no longer needed.
  bool queuePacket(const AVPacket *pkt, uint64_t token);
  bool pump(V4LDecodedFrame &frame, int timeout_ms);
  void releaseFrame(VisionBuf *buf);
  void sendEOS();
  size_t maxPacketSize() const { return out_buf_size; }

  AVFormatContext* avctx = nullptr;
  int fd = 0;

private:
  bool initialized = false;
  bool reconfigure_pending = false;
  bool direct = false;
  uint32_t capture_format = V4L2_PIX_FMT_NV12;

  VisionBuf out_bufs[OUTPUT_BUFFER_COUNT];    // Distinct dma-buf per in-flight packet
  VisionBuf cap_bufs[CAPTURE_BUFFER_COUNT];   // Capture (output) buffers

  size_t w = 0, h = 0;
  int out_buf_size = 0;

  bool out_buf_flag[OUTPUT_BUFFER_COUNT] = {false};

  const int subscriptions[2] = {
    V4L2_EVENT_MSM_VIDC_FLUSH_DONE,
    V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_INSUFFICIENT
  };

  struct pollfd pfd = {};

  bool subscribeEvents();
  bool setPlaneFormat(v4l2_buf_type type, uint32_t fourcc);
  bool setFPS(uint32_t fps);
  bool restartCapture();
  bool queueCaptureBuffer(int i);
  bool queueOutputBuffer(int i, size_t size, uint64_t token);
  bool setDBP();
  bool sendPacket(int buf_index, const AVPacket* pkt, uint64_t token);
  int getBufferUnlocked();
  int handleCapture(V4LDecodedFrame *frame);
  int handleOutput();
  int handleEvent();
};
