#pragma once

#include "system/loggerd/encoder/v4l_encoder.h"

// the downstream kernel's msm_vidc encoder. it scales in hardware, and takes ION buffers as pointers
class MsmVidcEncoder : public V4LEncoder {
public:
  MsmVidcEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options);
  ~MsmVidcEncoder() override;

protected:
  Bitstream parse_bitstream(uint32_t flags, const uint8_t *dat, size_t len) override;
  int acquire_input(const VisionBuf *buf) override;
  void set_plane(v4l2_plane *plane, const VisionBuf *buf) override;
  uint32_t keyframe_control() const override;

private:
  void set_codec_controls(const EncoderSettings &encoder_settings, bool max_performance);
};
