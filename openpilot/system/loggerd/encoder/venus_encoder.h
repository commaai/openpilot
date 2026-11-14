#pragma once

#include "system/loggerd/encoder/v4l_encoder.h"

// the upstream kernel's Venus encoder (qcom-venus). it relies on four AGNOS kernel patches that aren't upstream yet:
// - "media: venus: signal EOS with a real buffer": without it the drain in encoder_close never ends
// - "media: venus: keep capture buffers across encoder drain": without it there are empty buffers after a restart
// - "media: venus: venc: make every I-frame an IDR frame": without it a requested keyframe has no header
// - "media: venus: venc: scale the source to the capture format size": only for the scaled streams (qcamera and the
//   livestreams). without it a smaller encode is a crop, which the constructor asserts against
class VenusEncoder : public V4LEncoder {
public:
  VenusEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options);
  ~VenusEncoder() override;

protected:
  Bitstream parse_bitstream(uint32_t flags, const uint8_t *dat, size_t len) override;
  int acquire_input(const VisionBuf *buf) override;
  void set_plane(v4l2_plane *plane, const VisionBuf *buf) override;
  uint32_t keyframe_control() const override { return V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME; }
  void resume() override;

private:
  void set_codec_controls(const EncoderSettings &encoder_settings);
};
