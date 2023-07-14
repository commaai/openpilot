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
  FfmpegEncoder(const EncoderInfo &encoder_info, int in_width, int in_height)
      : VideoEncoder(encoder_info, in_width, in_height) { encoder_init(); }
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
