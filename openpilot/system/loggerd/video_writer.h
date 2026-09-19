#pragma once

#include <string>
#include <deque>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include "openpilot/cereal/messaging/messaging.h"
#include "system/loggerd/audio_encoder.h"

class VideoWriter {
public:
  VideoWriter(const char *path, const char *filename, bool remuxing, int width, int height, int fps, cereal::EncodeIndex::Type codec);
  void set_metadata(const char *key, const char *value);
  void write(uint8_t *data, int len, long long timestamp, bool codecconfig, bool keyframe);
  void write_audio(const AVPacket *packet, const AVCodecContext *codec);

  ~VideoWriter();

private:
  void flush_audio();

  std::string vid_path, lock_path;
  FILE *of = nullptr;

  AVCodecContext *codec_ctx;
  AVFormatContext *ofmt_ctx;
  AVStream *out_stream;

  bool header_written = false;
  AVStream *audio_stream = nullptr;
  std::deque<AudioPacket> audio_packets;
  AVRational audio_time_base;

  bool remuxing;
};
