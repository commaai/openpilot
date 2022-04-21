#pragma once

#include "selfdrive/loggerd/encoder.h"
#include "selfdrive/loggerd/loggerd.h"
#include "selfdrive/loggerd/video_writer.h"

class V4LEncoder : public VideoEncoder {
public:
  V4LEncoder(const char* filename, CameraType type, int width, int height, int fps, int bitrate, bool h265, int out_width, int out_height, bool write = true);
  ~V4LEncoder();
  int encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                   int in_width, int in_height, uint64_t ts);
  void encoder_open(const char* path);
  void encoder_close();
private:
  const char* filename;
  CameraType type;
  unsigned int in_width_, in_height_;
  unsigned int width, height, fps;
  bool remuxing, write;
  bool is_open = false;

  std::unique_ptr<VideoWriter> writer;
  int fd;

  std::unique_ptr<PubMaster> pm;
  const char *service_name;
};
