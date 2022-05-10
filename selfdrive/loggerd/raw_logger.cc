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

const int env_debug_encoder = (getenv("DEBUG_ENCODER") != NULL) ? atoi(getenv("DEBUG_ENCODER")) : 0;

RawLogger::RawLogger(const char* filename, CameraType type, int in_width, int in_height, int fps,
                     int bitrate, bool h265, int out_width, int out_height, bool write)
  : in_width_(in_width), in_height_(in_height) {

  // TODO: this are on the VideoEncoder class
  this->write = write;
  this->filename = filename;
  this->fps = fps;
  this->width = out_width;
  this->height = out_height;
  this->codec = RAW;
  this->type = type;

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

  publisher_init();
}

RawLogger::~RawLogger() {
  encoder_close();
  av_frame_free(&frame);
}

void RawLogger::encoder_open(const char* path) {
  AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_FFVHUFF);

  this->codec_ctx = avcodec_alloc_context3(codec);
  assert(this->codec_ctx);
  this->codec_ctx->width = frame->width;
  this->codec_ctx->height = frame->height;
  this->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  this->codec_ctx->time_base = (AVRational){ 1, fps };
  int err = avcodec_open2(this->codec_ctx, codec, NULL);
  assert(err >= 0);

  writer_open(path);
  is_open = true;
}

void RawLogger::encoder_close() {
  if (!is_open) return;
  writer_close();
  is_open = false;
}

int RawLogger::encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                            int in_width, int in_height, VisionIpcBufExtra *extra) {
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

  int err = avcodec_send_frame(this->codec_ctx, frame);
  if (err < 0) {
    LOGE("avcodec_send_frame error %d", err);
    ret = -1;
  }

  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;
  while (ret >= 0) {
    err = avcodec_receive_packet(this->codec_ctx, &pkt);
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

    if (env_debug_encoder) {
      printf("%20s got %8d bytes flags %8x\n", this->filename, pkt.size, pkt.flags);
    }

    publisher_publish(this, 0, counter, *extra,
      (pkt.flags & AV_PKT_FLAG_KEY) ? V4L2_BUF_FLAG_KEYFRAME : 0,
      kj::arrayPtr<capnp::byte>(pkt.data, (size_t)0), // TODO: get the header
      kj::arrayPtr<capnp::byte>(pkt.data, pkt.size));

    counter++;
  }
  av_packet_unref(&pkt);
  return ret;
}
