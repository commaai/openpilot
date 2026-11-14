#pragma once

#include "system/loggerd/encoder/v4l_decoder.h"

// the downstream kernel's msm_vidc decoder. it takes ION buffers as pointers, and decodes straight to the UBWC its
// encoder reads
class MsmVidcDecoder : public V4LDecoder {
public:
  MsmVidcDecoder() : V4LDecoder(V4L2_MEMORY_USERPTR) {}
  uint32_t direct_format() const override;

protected:
  int openDevice() override;
  void subscribeEvents() override;
  void setControls() override;
  void outputStarted() override { restartCapture(); }
  void handleCapture(const v4l2_buffer &buf, V4LDecodedFrame *frame) override;
  int handleEvent() override;
  void writePacket(VisionBuf *buf, const AVPacket *pkt) override;
  void setPlane(struct v4l2_plane *plane, const VisionBuf *buf) override;

private:
  void restartCapture();
  void setDBP();
};
