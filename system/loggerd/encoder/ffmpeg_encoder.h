#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

#include "system/loggerd/encoder/encoder.h"
#include "system/loggerd/loggerd.h"

class FfmpegEncoder : public VideoEncoder {
 public:
  FfmpegEncoder(const char* filename, CameraType type, int in_width, int in_height, int fps,
                int bitrate, cereal::EncodeIndex::Type codec, int out_width, int out_height) :
                VideoEncoder(filename, type, in_width, in_height, fps, bitrate, cereal::EncodeIndex::Type::BIG_BOX_LOSSLESS, out_width, out_height) { encoder_init(); }
  ~FfmpegEncoder();
  void encoder_init();
  int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra);
  void encoder_open(const char* path);
  void encoder_close();

private:
  int segment_num = -1;
  int counter = 0;
  bool is_open = false;

  AVCodecContext *codec_ctx;
  AVFrame *frame = NULL;
  std::vector<uint8_t> convert_buf;
  std::vector<uint8_t> downscale_buf;
};
