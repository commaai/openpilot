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

#include "frame_logger.h"

class RawLogger : public FrameLogger {
 public:
  RawLogger(const std::string &filename, int awidth, int aheight, int afps);
  virtual ~RawLogger();

 private:
  bool ProcessFrame(uint64_t cnt, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                    int in_width, int in_height, const VIPCBufExtra &extra);
  void Open(const std::string &path);
  void Close();

  AVCodec *codec = NULL;
  AVCodecContext *codec_ctx = NULL;

  AVStream *stream = NULL;
  AVFormatContext *format_ctx = NULL;

  AVFrame *frame = NULL;

  double rawlogger_start_time = 0;
  int rawlogger_clip_cnt = 0;
};
