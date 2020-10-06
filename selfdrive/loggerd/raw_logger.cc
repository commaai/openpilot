#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <fcntl.h>
#include <math.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#define __STDC_CONSTANT_MACROS

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

#include "common/swaglog.h"
#include "common/utilpp.h"
#include "raw_logger.h"

FFmpegEncoder::FFmpegEncoder(std::string filename, AVCodecID codec_id, int bitrate, int width, int height, int fps)
    : FrameLogger(filename, width, height, fps) {
  int err = 0;

  av_register_all();
  codec = avcodec_find_encoder(codec_id);

  assert(codec);

  codec_ctx = avcodec_alloc_context3(codec);
  assert(codec_ctx);
  if (bitrate > 0) {
    codec_ctx->bit_rate = bitrate;
  }
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->time_base = (AVRational){1, fps};
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

  AVDictionary *pDic = 0;
  if (codec_id == AV_CODEC_ID_H265) {
    av_dict_set(&pDic, "x265-params", "qp=20", 0);
    av_dict_set(&pDic, "preset", "ultrafast", 0);
    av_dict_set(&pDic, "tune", "zero-latency", 0);
  }

  err = avcodec_open2(codec_ctx, codec, &pDic);
  // err = avcodec_open2(codec_ctx, codec, nullptr);
  assert(err >= 0);

  frame = av_frame_alloc();
  assert(frame);
  frame->format = codec_ctx->pix_fmt;
  frame->width = width;
  frame->height = height;
  frame->linesize[0] = width;
  frame->linesize[1] = width / 2;
  frame->linesize[2] = width / 2;
}

FFmpegEncoder::~FFmpegEncoder() {
  av_frame_free(&frame);
  avcodec_close(codec_ctx);
  av_free(codec_ctx);
}

bool FFmpegEncoder::Open(const std::string path, int segment) {
  format_ctx = NULL;
  avformat_alloc_output_context2(&format_ctx, NULL, NULL, path.c_str());
  assert(format_ctx);

  stream = avformat_new_stream(format_ctx, codec);
  assert(stream);
  stream->id = 0;
  stream->time_base = (AVRational){1, fps};

  int err = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
  assert(err >= 0);

  err = avio_open(&format_ctx->pb, path.c_str(), AVIO_FLAG_WRITE);
  assert(err >= 0);

  err = avformat_write_header(format_ctx, NULL);
  assert(err >= 0);
  return true;
}

void FFmpegEncoder::Close() {
  int err = av_write_trailer(format_ctx);
  assert(err == 0);

  avcodec_close(stream->codec);

  err = avio_closep(&format_ctx->pb);
  assert(err == 0);

  avformat_free_context(format_ctx);
  format_ctx = NULL;
}

bool FFmpegEncoder::ProcessFrame(uint64_t cnt, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                                 int in_width, int in_height, const VIPCBufExtra &extra) {
  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  frame->data[0] = (uint8_t*)y_ptr;
  frame->data[1] = (uint8_t*)u_ptr;
  frame->data[2] = (uint8_t*)v_ptr;
  frame->pts = cnt;

  bool ret = true;

  int got_output = 0;
  int err = avcodec_encode_video2(codec_ctx, &pkt, frame, &got_output);
  if (err) {
    LOGE("encoding error\n");
    ret = false;
  } else if (got_output) {

    av_packet_rescale_ts(&pkt, codec_ctx->time_base, stream->time_base);
    pkt.stream_index = 0;

    err = av_interleaved_write_frame(format_ctx, &pkt);
    if (err < 0) {
      LOGE("encoder writer error\n");
      ret = false;
    }
  }

  return ret;
}
