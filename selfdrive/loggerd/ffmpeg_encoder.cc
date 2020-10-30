#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <fcntl.h>
#include <math.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <libyuv.h>
#define __STDC_CONSTANT_MACROS

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

#include "common/swaglog.h"
#include "common/utilpp.h"
#include "ffmpeg_encoder.h"

FFmpegEncoder::FFmpegEncoder(std::string filename, AVCodecID codec_id, int bitrate,
 int in_width, int in_height, int out_width, int out_height,  int fps)
    : filename(filename), in_width(in_width), in_height(in_height),
    out_width(out_width), out_height(out_height), bitrate(bitrate), fps(fps) {
  av_register_all();
  codec = avcodec_find_encoder(codec_id);
  assert(codec);
  if (in_width != out_width || in_height != out_height) {
    y_ptr2 = std::make_unique<uint8_t[]>(out_width * out_height);
    u_ptr2 = std::make_unique<uint8_t[]>(out_width * out_height / 4);
    v_ptr2 = std::make_unique<uint8_t[]>(out_width * out_height / 4);
  }
}

FFmpegEncoder::~FFmpegEncoder() {
  Close();
}

void FFmpegEncoder::Open(const std::string &path) {
  assert(codec_ctx == nullptr);

  // create camera lock file
  lock_path = util::string_format("%s/%s.lock", path.c_str(), filename.c_str());
  LOG("open %s\n", lock_path.c_str());
  int lock_fd = open(lock_path.c_str(), O_RDWR | O_CREAT, 0777);
  assert(lock_fd >= 0);
  close(lock_fd);

  codec_ctx = avcodec_alloc_context3(codec);
  assert(codec_ctx);
  
  if (bitrate > 0) {
    codec_ctx->bit_rate = bitrate;
  }
  codec_ctx->width = out_width;
  codec_ctx->height = out_height;
  codec_ctx->time_base = (AVRational){1, fps};
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

  int err = avcodec_open2(codec_ctx, codec, nullptr);
  assert(err >= 0);

  frame = av_frame_alloc();
  assert(frame);
  frame->format = codec_ctx->pix_fmt;
  frame->width = out_width;
  frame->height = out_height;
  frame->linesize[0] = out_width;
  frame->linesize[1] = out_width / 2;
  frame->linesize[2] = out_width / 2;
 
  std::string file = util::string_format("%s/%s", path.c_str(), filename.c_str());
  avformat_alloc_output_context2(&format_ctx, nullptr, nullptr, file.c_str());
  assert(format_ctx);

  stream = avformat_new_stream(format_ctx, nullptr);
  assert(stream);
  stream->id = 0;
  stream->time_base = (AVRational){1, fps};

  err = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
  assert(err >= 0);

  err = avio_open(&format_ctx->pb, file.c_str(), AVIO_FLAG_WRITE);
  assert(err >= 0);

  err = avformat_write_header(format_ctx, nullptr);
  assert(err >= 0);
  is_open = true;
  counter = 0;
}

void FFmpegEncoder::Close() {
  if (codec_ctx) {
    int err = av_write_trailer(format_ctx);
    assert(err == 0);

    // avcodec_close(stream->codec);

    err = avio_closep(&format_ctx->pb);
    assert(err == 0);

    avformat_free_context(format_ctx);
    format_ctx = nullptr;

    av_frame_free(&frame);
    frame = nullptr;
   
    avcodec_close(codec_ctx);
    av_free(codec_ctx);
    codec_ctx = nullptr;

    unlink(lock_path.c_str());
    is_open = false;
  }
}

int FFmpegEncoder::ProcessFrame(uint64_t cnt, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr) {
  if (in_width != out_width || in_height != out_height) {
    
    libyuv::I420Scale(y_ptr, in_width,
              u_ptr, in_width / 2,
              v_ptr, in_width / 2,
              in_width, in_height,
              y_ptr2.get(), out_width,
              u_ptr2.get(), out_width / 2,
              v_ptr2.get(), out_width / 2,
              out_width, out_height,
              libyuv::kFilterNone);
    y_ptr = y_ptr2.get();
    u_ptr = u_ptr2.get();
    v_ptr = v_ptr2.get();
  }

  frame->data[0] = (uint8_t*)y_ptr;
  frame->data[1] = (uint8_t*)u_ptr;
  frame->data[2] = (uint8_t*)v_ptr;
  frame->pts = cnt;

  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;
  
  int ret = counter;
  int got_output = 0;
  int err = avcodec_encode_video2(codec_ctx, &pkt, frame, &got_output);
  if (err) {
    LOGW("encoding error\n");
    ret = -1;
  } else if (got_output) {
    av_packet_rescale_ts(&pkt, codec_ctx->time_base, stream->time_base);
    pkt.stream_index = 0;

    err = av_interleaved_write_frame(format_ctx, &pkt);
    if (err < 0) {
      LOGW("encoder writer error\n");
      ret = -1;
    } else {
      counter++;
    }
  }
  av_free_packet(&pkt);
  return ret;
}
