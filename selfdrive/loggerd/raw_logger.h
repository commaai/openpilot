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

#include "selfdrive/loggerd/encoder.h"
#include "selfdrive/loggerd/video_writer.h"

class RawLogger : public VideoEncoder {
 public:
  RawLogger(const char* filename, CameraType type, int in_width, int in_height, int fps,
            int bitrate, bool h265, int out_width, int out_height, bool write = true);
  ~RawLogger();
  int encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                   int in_width, int in_height, uint64_t ts);
  void encoder_open(const char* path);
  void encoder_close();

private:
  const char* filename;
  //bool write;
  int fps;
  int counter = 0;
  bool is_open = false;

  int in_width_, in_height_;

  AVFrame *frame = NULL;
  std::vector<uint8_t> downscale_buf;

  VideoWriter *writer = NULL;
};
