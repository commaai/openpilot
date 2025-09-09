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
#define V4L2_CID_MPEG_MSM_VIDC_BASE 0x00992000
#define V4L2_CID_MPEG_VIDC_VIDEO_DPB_COLOR_FORMAT (V4L2_CID_MPEG_MSM_VIDC_BASE + 44)
#define V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_MODE (V4L2_CID_MPEG_MSM_VIDC_BASE + 22)
#define V4L2_QCOM_CMD_FLUSH_CAPTURE (1 << 1)
#define V4L2_QCOM_CMD_FLUSH (4)

#define VIDEO_DEVICE "/dev/video32"
#define OUTPUT_BUFFER_COUNT 	8
#define CAPTURE_BUFFER_COUNT 	8
#define FPS 									20


class MsmVidc {
public:
  MsmVidc() = default;
  ~MsmVidc();

  bool init(const char* dev, size_t width, size_t height, uint64_t codec);
  VisionBuf* decodeFrame(AVPacket* pkt, VisionBuf* buf);

  AVFormatContext* avctx = nullptr;
  int fd = 0;

private:
  bool initialized = false;
  bool reconfigure_pending = false;
  bool frame_ready = false;

  VisionBuf* current_output_buf = nullptr;
  VisionBuf out_buf;                          // Single input buffer
  VisionBuf cap_bufs[CAPTURE_BUFFER_COUNT];   // Capture (output) buffers

  size_t w = 1928, h = 1208;
  size_t cap_height = 0, cap_width = 0;

  int cap_buf_size = 0;
  int out_buf_size = 0;

  size_t cap_plane_off[CAPTURE_BUFFER_COUNT] = {0};
  size_t cap_plane_stride[CAPTURE_BUFFER_COUNT] = {0};
  bool cap_buf_flag[CAPTURE_BUFFER_COUNT] = {false};

  size_t out_buf_off[OUTPUT_BUFFER_COUNT] = {0};
  void* out_buf_addr[OUTPUT_BUFFER_COUNT] = {0};
  bool out_buf_flag[OUTPUT_BUFFER_COUNT] = {false};
  const int out_buf_cnt = OUTPUT_BUFFER_COUNT;

  const int subscriptions[2] = {
    V4L2_EVENT_MSM_VIDC_FLUSH_DONE,
    V4L2_EVENT_MSM_VIDC_PORT_SETTINGS_CHANGED_INSUFFICIENT
  };

  enum { EV_VIDEO, EV_COUNT };
  struct pollfd pfd[EV_COUNT] = {0};
  int ev[EV_COUNT] = {-1};
  int nfds = 0;

  VisionBuf* processEvents();
  bool setupOutput();
  bool subscribeEvents();
  bool setPlaneFormat(v4l2_buf_type type, uint32_t fourcc);
  bool setFPS(uint32_t fps);
  bool restartCapture();
  bool queueCaptureBuffer(int i);
  bool queueOutputBuffer(int i, size_t size);
  bool setDBP();
  bool setupPolling();
  bool sendPacket(int buf_index, AVPacket* pkt);
  int getBufferUnlocked();
  VisionBuf* handleCapture();
  bool handleOutput();
  bool handleEvent();
};
