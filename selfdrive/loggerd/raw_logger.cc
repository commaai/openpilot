#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "selfdrive/loggerd/raw_logger.h"

#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#define __STDC_CONSTANT_MACROS

#include "libyuv.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

RawLogger::RawLogger(const char* filename, CameraType type, int in_width, int in_height, int fps,
                     int bitrate, bool h265, int out_width, int out_height, bool write)
  : in_width_(in_width), in_height_(in_height), filename(filename), fps(fps) {
  // TODO: respect write arg
  frame = av_frame_alloc();
  assert(frame);
  frame->format = AV_PIX_FMT_YUV420P;
  frame->width = out_width;
  frame->height = out_height;
  frame->linesize[0] = out_width;
  frame->linesize[1] = out_width/2;
  frame->linesize[2] = out_width/2;

  if (in_width != out_width || in_height != out_height) {
    downscale_buf.resize(out_width * out_height * 3 / 2);
  }
}

RawLogger::~RawLogger() {
  encoder_close();
  av_frame_free(&frame);
}

void RawLogger::encoder_open(const char* path) {
  writer = new VideoWriter(path, this->filename, true, frame->width, frame->height, this->fps, false, true);
  // write the header
  writer->write(NULL, 0, 0, true, false);
  is_open = true;
}

void RawLogger::encoder_close() {
  if (!is_open) return;
  delete writer;
  is_open = false;
}

int RawLogger::encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                            int in_width, int in_height, uint64_t ts) {
  assert(in_width == this->in_width_);
  assert(in_height == this->in_height_);

  if (downscale_buf.size() > 0) {
    uint8_t *out_y = downscale_buf.data();
    uint8_t *out_u = out_y + frame->width * frame->height;
    uint8_t *out_v = out_u + (frame->width / 2) * (frame->height / 2);
    libyuv::I420Scale(y_ptr, in_width,
                      u_ptr, in_width/2,
                      v_ptr, in_width/2,
                      in_width, in_height,
                      out_y, frame->width,
                      out_u, frame->width/2,
                      out_v, frame->width/2,
                      frame->width, frame->height,
                      libyuv::kFilterNone);
    frame->data[0] = out_y;
    frame->data[1] = out_u;
    frame->data[2] = out_v;
  } else {
    frame->data[0] = (uint8_t*)y_ptr;
    frame->data[1] = (uint8_t*)u_ptr;
    frame->data[2] = (uint8_t*)v_ptr;
  }
  frame->pts = counter*50*1000; // 50ms per frame

  int ret = counter;

  int err = avcodec_send_frame(writer->codec_ctx, frame);
  if (err < 0) {
    LOGE("avcodec_send_frame error %d", err);
    ret = -1;
  }

  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;
  while (ret >= 0) {
    err = avcodec_receive_packet(writer->codec_ctx, &pkt);
    if (err == AVERROR_EOF) {
      break;
    } else if (err == AVERROR(EAGAIN)) {
      // Encoder might need a few frames on startup to get started. Keep going
      ret = 0;
      break;
    } else if (err < 0) {
      LOGE("avcodec_receive_packet error %d", err);
      ret = -1;
      break;
    }

    writer->write(pkt.data, pkt.size, pkt.pts, false, pkt.flags & AV_PKT_FLAG_KEY);
    counter++;
  }
  av_packet_unref(&pkt);
  return ret;
}
