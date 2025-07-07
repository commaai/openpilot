#pragma once

#include <string>
#include <deque>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include "cereal/messaging/messaging.h"

class VideoWriter {
public:
  VideoWriter(const char *path, const char *filename, bool remuxing, int width, int height, int fps, cereal::EncodeIndex::Type codec);
  void write(uint8_t *data, int len, long long timestamp, bool codecconfig, bool keyframe);
  void write_audio(uint8_t *data, int len, long long timestamp, int sample_rate);

  ~VideoWriter();

private:
  void initialize_audio(int sample_rate);
  void encode_and_write_audio_frame(AVFrame* frame);

  std::string vid_path, lock_path;
  FILE *of = nullptr;

  AVCodecContext *codec_ctx;
  AVFormatContext *ofmt_ctx;
  AVStream *out_stream;

  bool audio_initialized = false;
  bool header_written = false;
  AVStream *audio_stream = nullptr;
  AVCodecContext *audio_codec_ctx = nullptr;
  AVFrame *audio_frame = nullptr;
  uint64_t audio_pts = 0;
  std::deque<float> audio_buffer;

  bool remuxing;
};
