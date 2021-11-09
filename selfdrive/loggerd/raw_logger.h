#pragma once

#include <string>

#include "selfdrive/loggerd/encoder.h"

class RawLogger : protected FFmpegEncoder {
public:
  RawLogger(const char* filename, int width, int height, int fps,
            int bitrate, bool h265, bool downscale, bool write = true);
  ~RawLogger();
  void encoder_open(const char* path);
  void encoder_close();
  int encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, int in_width, int in_height, uint64_t ts);

private:
  const char* filename;
  int counter = 0;
  bool is_open = false;
  AVFrame *frame = nullptr;
  std::string vid_path;
};
