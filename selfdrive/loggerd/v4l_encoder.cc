#include "selfdrive/loggerd/v4l_encoder.h"

V4LEncoder::V4LEncoder(
  const char* filename, CameraType type, int in_width, int in_height,
  int fps, int bitrate, bool h265, int out_width, int out_height, bool write)
  : in_width_(in_width), in_height_(in_height),
    width(out_width), height(out_height), fps(fps),
    filename(filename), type(type), remuxing(!h265), write(write) {
}

void V4LEncoder::encoder_open(const char* path) {
  if (this->write) {
    writer.reset(new VideoWriter(path, this->filename, this->remuxing, this->width, this->height, this->fps, !this->remuxing, false));
  }
}