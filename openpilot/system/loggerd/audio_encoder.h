#pragma once

#include <deque>
#include <functional>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
}

struct AudioPacketDeleter {
  void operator()(AVPacket *packet) const { av_packet_free(&packet); }
};
using AudioPacket = std::unique_ptr<AVPacket, AudioPacketDeleter>;

class AudioEncoder {
public:
  using Callback = std::function<void(const AVPacket *, const AVCodecContext *, int, int)>;
  AudioEncoder(int sample_rate, Callback callback);
  ~AudioEncoder();
  void write(const uint8_t *data, int len, uint64_t timestamp);
  const AVCodecContext *context() const { return codec_ctx; }

private:
  void encode(AVFrame *frame);
  AVCodecContext *codec_ctx = nullptr;
  AVFrame *frame = nullptr;
  std::deque<float> buffer;
  Callback callback;
  int64_t start_pts = 0, next_pts = 0, sample_count = 0;
};
