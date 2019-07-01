#ifndef FFV1LOGGER_H
#define FFV1LOGGER_H

#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "frame_logger.h"

class RawLogger : public FrameLogger {
public:
  RawLogger(const std::string &filename, int awidth, int aheight, int afps);
  ~RawLogger();

  int ProcessFrame(uint64_t ts, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr);
  void Open(const std::string &path);
  void Close();

private:
  std::string filename;
  int width, height, fps;
  int counter = 0;

  AVCodec *codec = NULL;
  AVCodecContext *codec_ctx = NULL;

  AVStream *stream = NULL;
  AVFormatContext *format_ctx = NULL;

  AVFrame *frame = NULL;
};

#endif
