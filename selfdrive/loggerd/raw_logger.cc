#include "selfdrive/loggerd/raw_logger.h"

#include <cassert>

#include "selfdrive/common/util.h"

RawLogger::RawLogger(const char* filename, int width, int height, int fps,
                     int bitrate, bool h265, bool downscale, bool write) : FFmpegEncoder(AV_CODEC_ID_FFVHUFF, width, height, fps) {
  this->filename = filename;
  frame = av_frame_alloc();
  frame->format = codec_ctx->pix_fmt;
  frame->width = width;
  frame->height = height;
  frame->linesize[0] = width;
  frame->linesize[1] = width/2;
  frame->linesize[2] = width/2;
}

RawLogger::~RawLogger() {
  av_frame_free(&frame);
}

void RawLogger::encoder_open(const char* path) {
  vid_path = util::string_format("%s/%s.mkv", path, filename);
  writeHeader();
  is_open = true;
  counter = 0;
}

void RawLogger::encoder_close() {
  if (is_open) return;

  close();
  is_open = false;
}
