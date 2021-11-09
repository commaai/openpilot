#include "selfdrive/loggerd/raw_logger.h"

#include <cassert>

#include "selfdrive/common/util.h"

RawLogger::RawLogger(const char *filename, int width, int height, int fps,
                     int bitrate, bool h265, bool downscale, bool write) : FFmpegEncoder(AV_CODEC_ID_FFVHUFF, width, height, fps) {
  this->filename = filename;
  frame = av_frame_alloc();
  frame->format = codec_ctx->pix_fmt;
  printf("%d\n", codec_ctx->pix_fmt);
  frame->width = width;
  frame->height = height;
  frame->linesize[0] = width;
  frame->linesize[1] = width / 2;
  frame->linesize[2] = width / 2;
}

RawLogger::~RawLogger() {
  av_frame_free(&frame);
}

void RawLogger::encoder_open(const char *path) {
  vid_path = util::string_format("%s/%s.mkv", path, filename);
  open(vid_path.c_str());
  writeHeader();
  is_open = true;
  counter = 0;
}

void RawLogger::encoder_close() {
  if (is_open) return;

  close();
  is_open = false;
}

int RawLogger::encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, int in_width, int in_height, uint64_t ts) {
  frame->data[0] = (uint8_t *)y_ptr;
  frame->data[1] = (uint8_t *)u_ptr;
  frame->data[2] = (uint8_t *)v_ptr;
  frame->pts = counter;
  return encode(frame) ? counter++ : -1;
}