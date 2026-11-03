#pragma once

#include <linux/videodev2.h>
#include <poll.h>

#include <memory>

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

// a V4L2 stateful decoder. the queues and buffers are here, and what differs between the kernels' drivers is in
// VenusDecoder (the upstream kernel's Venus driver) and MsmVidcDecoder (the downstream kernel's msm_vidc), picked by
// create(). above all, how a change of the stream's resolution reconfigures the capture queue
class V4LDecoder {
public:
  // the decoder for this device's kernel
  static std::unique_ptr<V4LDecoder> create();
  virtual ~V4LDecoder();

  // direct_mode decodes to buffers the caller passes on (pump() and releaseFrame()), in direct_format(), which is what
  // the encoder of the same kernel reads. otherwise frames are decoded to NV12 and copied out by decodeFrame()
  bool init(size_t width, size_t height, uint64_t codec, bool direct_mode = false);
  virtual uint32_t direct_format() const = 0;
  VisionBuf* decodeFrame(AVPacket* pkt, VisionBuf* buf);
  // queuePacket() and pump() are single-threaded. releaseFrame() may be called
  // from a consumer thread after a direct capture surface is no longer needed.
  bool queuePacket(const AVPacket *pkt, uint64_t token);
  bool pump(V4LDecodedFrame &frame, int timeout_ms);
  virtual void releaseFrame(VisionBuf *buf);
  virtual void sendEOS();
  size_t maxPacketSize() const { return out_buf_size; }

  AVFormatContext* avctx = nullptr;
  int fd = -1;

protected:
  explicit V4LDecoder(v4l2_memory memory) : memory(memory) {}

  // the setup in init: the device, then the events, then (after the output format) the controls, then once the output
  // queue streams, the capture queue if it can be set up already
  virtual int openDevice() = 0;
  virtual void subscribeEvents() = 0;
  virtual void setControls() {}
  virtual void outputStarted() {}
  // a dequeued CAPTURE buffer: sets frame->buf if it's a frame for the caller
  virtual void handleCapture(const v4l2_buffer &buf, V4LDecodedFrame *frame) = 0;
  // 1 if there was an event, 0 if there was none, -1 on an error
  virtual int handleEvent() = 0;
  // before pump() polls, the events it polls the device for, and an eventfd it polls too, which another thread can
  // write to end the poll
  virtual void beforePoll() {}
  virtual short pollEvents() const { return POLLIN | POLLOUT | POLLWRNORM | POLLRDNORM | POLLPRI; }
  virtual int wakeFd() const { return -1; }
  // copies a packet into an OUTPUT buffer
  virtual void writePacket(VisionBuf *buf, const AVPacket *pkt) = 0;
  // points a V4L2 plane at a buffer
  virtual void setPlane(struct v4l2_plane *plane, const VisionBuf *buf) = 0;

  // the capture queue: requests and allocates its buffers in the format fmt was set to, with the visible size
  // width x height (the format's if they're 0), starts it, and stops it and frees them
  void allocateCapture(v4l2_format &fmt, size_t width = 0, size_t height = 0);
  void startCapture();
  void stopCapture();
  bool queueCaptureBuffer(int i);

  const v4l2_memory memory;
  bool direct = false;
  bool capture_started = false;
  bool reconfigure_pending = false;
  uint32_t capture_format = V4L2_PIX_FMT_NV12;
  size_t w = 0, h = 0;

  VisionBuf out_bufs[OUTPUT_BUFFER_COUNT];    // Distinct dma-buf per in-flight packet
  VisionBuf cap_bufs[CAPTURE_BUFFER_COUNT];   // Capture (output) buffers

private:
  bool initialized = false;
  int out_buf_size = 0;
  bool out_buf_flag[OUTPUT_BUFFER_COUNT] = {false};
  struct pollfd pfds[2] = {};

  void setOutputFormat(uint32_t fourcc);
  bool setFPS(uint32_t fps);
  bool queueOutputBuffer(int i, size_t size, uint64_t token);
  bool sendPacket(int buf_index, const AVPacket* pkt, uint64_t token);
  int getBufferUnlocked();
  int dequeueCapture(V4LDecodedFrame *frame);
  int handleOutput();
};
