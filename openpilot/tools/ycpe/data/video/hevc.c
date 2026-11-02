// HEVC decoding for the browser (FFmpeg): push access units, then pull pictures
#include <stdint.h>
#include <string.h>
#include "libavcodec/avcodec.h"

#define EXPORT __attribute__((used, visibility("default")))

static AVCodecContext *ctx;
static AVPacket *pkt;
static AVFrame *frame;  // the current picture while frame->data[0] is set

// pts rides through the decoder as raw bits, so any double survives unchanged
static int64_t pts_bits(double d) { int64_t i; memcpy(&i, &d, 8); return i; }
static double pts_double(int64_t i) { double d; memcpy(&d, &i, 8); return d; }

EXPORT int hevc_init(void) {
  const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
  ctx = avcodec_alloc_context3(codec);
  pkt = av_packet_alloc();
  frame = av_frame_alloc();
  if (!ctx || !pkt || !frame) return -1;
  return avcodec_open2(ctx, codec, NULL);
}

EXPORT int hevc_push(const uint8_t *data, int len, double pts) {
  int err = av_new_packet(pkt, len);  // the decoder wants zeroed padding past the end
  if (err) return err;
  memcpy(pkt->data, data, len);
  pkt->pts = pts_bits(pts);
  err = avcodec_send_packet(ctx, pkt);
  av_packet_unref(pkt);
  return err;
}

EXPORT int hevc_flush(void) {
  return avcodec_send_packet(ctx, NULL);
}

EXPORT void hevc_release(void) {
  av_frame_unref(frame);
}

EXPORT void hevc_reset(void) {
  av_frame_unref(frame);
  avcodec_flush_buffers(ctx);
}

EXPORT int hevc_frame(void) {
  // once drained after a flush, the decoder takes a new stream
  if (!frame->data[0] && avcodec_receive_frame(ctx, frame) == AVERROR_EOF) avcodec_flush_buffers(ctx);
  return frame->data[0] != NULL;
}

EXPORT int hevc_width(void) { return frame->width; }
EXPORT int hevc_height(void) { return frame->height; }
EXPORT double hevc_pts(void) { return pts_double(frame->pts); }
EXPORT const uint8_t *hevc_plane(int c) { return frame->data[c]; }
EXPORT int hevc_stride(int c) { return frame->linesize[c]; }
