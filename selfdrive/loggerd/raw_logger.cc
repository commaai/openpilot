#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <fcntl.h>
#include <unistd.h>

#define __STDC_CONSTANT_MACROS

extern "C" {
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "common/swaglog.h"
#include "common/utilpp.h"

#include "raw_logger.h"

RawLogger::RawLogger(const std::string &afilename, int awidth, int aheight, int afps)
  : filename(afilename),
    width(awidth),
    height(aheight),
    fps(afps) {

  int err = 0;

  av_register_all();
  codec = avcodec_find_encoder(AV_CODEC_ID_FFVHUFF);
  // codec = avcodec_find_encoder(AV_CODEC_ID_FFV1);
  assert(codec);

  codec_ctx = avcodec_alloc_context3(codec);
  assert(codec_ctx);
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

  // codec_ctx->thread_count = 2;

  // ffv1enc doesn't respect AV_PICTURE_TYPE_I. make every frame a key frame for now.
  // codec_ctx->gop_size = 0;

  codec_ctx->time_base = (AVRational){ 1, fps };

  err = avcodec_open2(codec_ctx, codec, NULL);
  assert(err >= 0);

  frame = av_frame_alloc();
  assert(frame);
  frame->format = codec_ctx->pix_fmt;
  frame->width = width;
  frame->height = height;
  frame->linesize[0] = width;
  frame->linesize[1] = width/2;
  frame->linesize[2] = width/2;
}

RawLogger::~RawLogger() {
  av_frame_free(&frame);
  avcodec_close(codec_ctx);
  av_free(codec_ctx);
}

void RawLogger::Open(const std::string &path) {
  int err = 0;

  std::lock_guard<std::recursive_mutex> guard(lock);

  vid_path = util::string_format("%s/%s.mkv", path.c_str(), filename.c_str());

  // create camera lock file
  lock_path = util::string_format("%s/%s.lock", path.c_str(), filename.c_str());

  LOG("open %s\n", lock_path.c_str());

  int lock_fd = open(lock_path.c_str(), O_RDWR | O_CREAT, 0777);
  assert(lock_fd >= 0);
  close(lock_fd);

  format_ctx = NULL;
  avformat_alloc_output_context2(&format_ctx, NULL, NULL, vid_path.c_str());
  assert(format_ctx);

  stream = avformat_new_stream(format_ctx, codec);
  // AVStream *stream = avformat_new_stream(format_ctx, NULL);
  assert(stream);
  stream->id = 0;
  stream->time_base = (AVRational){ 1, fps };
  // codec_ctx->time_base = stream->time_base;

  err = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
  assert(err >= 0);

  err = avio_open(&format_ctx->pb, vid_path.c_str(), AVIO_FLAG_WRITE);
  assert(err >= 0);

  err = avformat_write_header(format_ctx, NULL);
  assert(err >= 0);

  is_open = true;
  counter = 0;
}

void RawLogger::Close() {
  int err = 0;

  std::lock_guard<std::recursive_mutex> guard(lock);

  if (!is_open) return;

  err = av_write_trailer(format_ctx);
  assert(err == 0);

  avcodec_close(stream->codec);

  err = avio_closep(&format_ctx->pb);
  assert(err == 0);

  avformat_free_context(format_ctx);
  format_ctx = NULL;

  unlink(lock_path.c_str());
  is_open = false;
}

int RawLogger::ProcessFrame(uint64_t ts, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr) {
  int err = 0;

  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  frame->data[0] = (uint8_t*)y_ptr;
  frame->data[1] = (uint8_t*)u_ptr;
  frame->data[2] = (uint8_t*)v_ptr;
  frame->pts = ts;

  int ret = counter;

  int got_output = 0;
  err = avcodec_encode_video2(codec_ctx, &pkt, frame, &got_output);
  if (err) {
    LOGE("encoding error\n");
    ret = -1;
  } else if (got_output) {

    av_packet_rescale_ts(&pkt, codec_ctx->time_base, stream->time_base);
    pkt.stream_index = 0;

    err = av_interleaved_write_frame(format_ctx, &pkt);
    if (err < 0) {
      LOGE("encoder writer error\n");
      ret = -1;
    } else {
      counter++;
    }
  }

  return ret;
}
