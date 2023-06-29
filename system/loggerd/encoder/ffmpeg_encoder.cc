#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "system/loggerd/encoder/ffmpeg_encoder.h"

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

#include "common/swaglog.h"
#include "common/util.h"

const int env_debug_encoder = (getenv("DEBUG_ENCODER") != NULL) ? atoi(getenv("DEBUG_ENCODER")) : 0;

void FfmpegEncoder::encoder_init() {
  frame = av_frame_alloc();
  assert(frame);
  frame->format = AV_PIX_FMT_YUV420P;
  frame->width = out_width;
  frame->height = out_height;
  frame->linesize[0] = out_width;
  frame->linesize[1] = out_width/2;
  frame->linesize[2] = out_width/2;

  convert_buf.resize(in_width * in_height * 3 / 2);

  if (in_width != out_width || in_height != out_height) {
    downscale_buf.resize(out_width * out_height * 3 / 2);
  }

  publisher_init();
}

FfmpegEncoder::~FfmpegEncoder() {
  encoder_close();
  av_frame_free(&frame);
}

void FfmpegEncoder::encoder_open(const char* path) {
  const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_FFVHUFF);

  this->codec_ctx = avcodec_alloc_context3(codec);
  assert(this->codec_ctx);
  this->codec_ctx->width = frame->width;
  this->codec_ctx->height = frame->height;
  this->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  this->codec_ctx->time_base = (AVRational){ 1, fps };
  int err = avcodec_open2(this->codec_ctx, codec, NULL);
  assert(err >= 0);

  is_open = true;
  segment_num++;
  counter = 0;
}

void FfmpegEncoder::encoder_close() {
  if (!is_open) return;

  avcodec_free_context(&codec_ctx);
  is_open = false;
}

int FfmpegEncoder::encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) {
  assert(buf->width == this->in_width);
  assert(buf->height == this->in_height);

  uint8_t *cy = convert_buf.data();
  uint8_t *cu = cy + in_width * in_height;
  uint8_t *cv = cu + (in_width / 2) * (in_height / 2);
  libyuv::NV12ToI420(buf->y, buf->stride,
                     buf->uv, buf->stride,
                     cy, in_width,
                     cu, in_width/2,
                     cv, in_width/2,
                     in_width, in_height);

  if (downscale_buf.size() > 0) {
    uint8_t *out_y = downscale_buf.data();
    uint8_t *out_u = out_y + frame->width * frame->height;
    uint8_t *out_v = out_u + (frame->width / 2) * (frame->height / 2);
    libyuv::I420Scale(cy, in_width,
                      cu, in_width/2,
                      cv, in_width/2,
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
    frame->data[0] = cy;
    frame->data[1] = cu;
    frame->data[2] = cv;
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
      printf("%20s got %8d bytes flags %8x idx %4d id %8d\n", this->filename, pkt.size, pkt.flags, counter, extra->frame_id);
    }

    publisher_publish(this, segment_num, counter, *extra,
      (pkt.flags & AV_PKT_FLAG_KEY) ? V4L2_BUF_FLAG_KEYFRAME : 0,
      kj::arrayPtr<capnp::byte>(pkt.data, (size_t)0), // TODO: get the header
      kj::arrayPtr<capnp::byte>(pkt.data, pkt.size));

    counter++;
  }
  av_packet_unref(&pkt);
  return ret;
}
