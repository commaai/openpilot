#pragma once

#include <linux/videodev2.h>
#include <poll.h>

#include <atomic>

#include "msgq/visionipc/visionbuf.h"

extern "C" {
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
}

#define OUTPUT_BUFFER_COUNT 	8
#define CAPTURE_BUFFER_COUNT 	16
#define FPS 									20

struct V4LDecodedFrame {
  VisionBuf *buf = nullptr;
  uint64_t token = 0;
};

// the Venus decoder. it decodes to linear NV12, and the capture queue is set up from the stream's header
// (V4L2_EVENT_SOURCE_CHANGE), as the V4L2 stateful decoder interface says
class V4LDecoder {
public:
  V4LDecoder();
  ~V4LDecoder();

  // direct_mode decodes to buffers the caller passes on (pump() and releaseFrame()), which the encoder reads.
  // otherwise frames are copied out by decodeFrame()
  bool init(size_t width, size_t height, uint64_t codec, bool direct_mode = false);
  VisionBuf* decodeFrame(AVPacket* pkt, VisionBuf* buf);
  // queuePacket() and pump() are single-threaded. releaseFrame() may be called
  // from a consumer thread after a direct capture surface is no longer needed.
  bool queuePacket(const AVPacket *pkt, uint64_t token);
  bool pump(V4LDecodedFrame &frame, int timeout_ms);
  void releaseFrame(VisionBuf *buf);
  void sendEOS();
  size_t maxPacketSize() const { return out_buf_size; }

  AVFormatContext* avctx = nullptr;
  int fd = -1;

private:
  bool initialized = false;
  bool reconfigure_pending = false;
  bool direct = false;
  bool capture_started = false;
  bool eos_pending = false;
  // after a resolution change, the old capture queue's frames end with a LAST buffer. it's restarted in the new format
  // once the caller has released all of them. every release ends a poll through wake_fd
  std::atomic<bool> drained = false;
  std::atomic<int> held = 0;  // frames handed to the caller and not released
  int wake_fd = -1;

  VisionBuf out_bufs[OUTPUT_BUFFER_COUNT];    // Distinct dma-buf per in-flight packet
  VisionBuf cap_bufs[CAPTURE_BUFFER_COUNT];   // Capture (output) buffers

  size_t w = 0, h = 0;
  int out_buf_size = 0;

  bool out_buf_flag[OUTPUT_BUFFER_COUNT] = {false};

  struct pollfd pfds[2] = {};

  void setOutputFormat(uint32_t fourcc);
  bool setFPS(uint32_t fps);
  void restartCapture();
  bool queueCaptureBuffer(int i);
  bool queueOutputBuffer(int i, size_t size, uint64_t token);
  bool sendPacket(int buf_index, const AVPacket* pkt, uint64_t token);
  int getBufferUnlocked();
  int handleCapture(V4LDecodedFrame *frame);
  int handleOutput();
  int handleEvent();
};
