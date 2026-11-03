#pragma once

#include <atomic>

#include "system/loggerd/encoder/v4l_decoder.h"

// the upstream kernel's Venus decoder (qcom-venus). it takes DMA-BUFs and decodes to linear NV12, and the capture
// queue is set up from the stream's header (V4L2_EVENT_SOURCE_CHANGE), as the V4L2 stateful decoder interface says
class VenusDecoder : public V4LDecoder {
public:
  VenusDecoder();
  ~VenusDecoder() override;
  uint32_t direct_format() const override { return V4L2_PIX_FMT_NV12; }
  void releaseFrame(VisionBuf *buf) override;
  void sendEOS() override;

protected:
  int openDevice() override;
  void subscribeEvents() override;
  void handleCapture(const v4l2_buffer &buf, V4LDecodedFrame *frame) override;
  int handleEvent() override;
  void beforePoll() override;
  short pollEvents() const override;
  int wakeFd() const override { return wake_fd; }
  void writePacket(VisionBuf *buf, const AVPacket *pkt) override;
  void setPlane(struct v4l2_plane *plane, const VisionBuf *buf) override;

private:
  void restartCapture();

  bool eos_pending = false;
  // after a resolution change, the old capture queue's frames end with a LAST buffer. it's restarted in the new format
  // once the caller has released all of them. every release ends a poll through wake_fd
  std::atomic<bool> drained = false;
  std::atomic<int> held = 0;  // frames handed to the caller and not released
  int wake_fd = -1;
};
