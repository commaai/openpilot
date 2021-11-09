#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "selfdrive/loggerd/encoder.h"

#include <cassert>
#include <mutex>

#include "selfdrive/common/swaglog.h"

#define CHECK_ERR(_expr)  assert((_expr) >= 0)

FFmpegEncoder::FFmpegEncoder(AVCodecID codec_id, int width, int height, int fps) : fps(fps) {
  codec = avcodec_find_encoder(codec_id);
  assert(codec);
  codec_ctx = avcodec_alloc_context3(codec);
  assert(codec_ctx);
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->time_base = (AVRational){1, fps};
  CHECK_ERR(avcodec_open2(codec_ctx, codec, NULL));
}

FFmpegEncoder::~FFmpegEncoder() {
  avcodec_close(codec_ctx);
  av_free(codec_ctx);
}

void FFmpegEncoder::open(const char *vid_path) {
  assert(format_ctx == nullptr);
  CHECK_ERR(avformat_alloc_output_context2(&format_ctx, NULL, NULL, vid_path));
  stream = avformat_new_stream(format_ctx, codec);
  assert(stream);
  stream->id = 0;
  stream->time_base = (AVRational){1, fps};
  CHECK_ERR(avio_open(&format_ctx->pb, vid_path, AVIO_FLAG_WRITE));
}

void FFmpegEncoder::close() {
  assert(format_ctx != nullptr);
  CHECK_ERR(av_write_trailer(format_ctx));
  CHECK_ERR(avcodec_close(stream->codec));
  CHECK_ERR(avio_closep(&format_ctx->pb));
  avformat_free_context(format_ctx);
  format_ctx = nullptr;
}

void FFmpegEncoder::writeHeader(const std::vector<uint8_t> &header) {
  if (!header.empty()) {
    // extradata will be freed by av_free() in avcodec_free_context()
    codec_ctx->extradata = (uint8_t *)av_mallocz(header.size() + AV_INPUT_BUFFER_PADDING_SIZE);
    codec_ctx->extradata_size = header.size();
    memcpy(codec_ctx->extradata, header.data(), header.size());
  }
  CHECK_ERR(avcodec_parameters_from_context(stream->codecpar, codec_ctx));
  CHECK_ERR(avformat_write_header(format_ctx, NULL));
}

void FFmpegEncoder::remux(OMX_BUFFERHEADERTYPE *out_buf) {
  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = out_buf->pBuffer + out_buf->nOffset;
  pkt.size = out_buf->nFilledLen;

  // input timestamps are in microseconds
  AVRational in_timebase = {1, 1000000};
  enum AVRounding rnd = static_cast<enum AVRounding>(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
  pkt.pts = pkt.dts = av_rescale_q_rnd(out_buf->nTimeStamp, in_timebase, format_ctx->streams[0]->time_base, rnd);
  pkt.duration = av_rescale_q(50 * 1000, in_timebase, format_ctx->streams[0]->time_base);
  if (out_buf->nFlags & OMX_BUFFERFLAG_SYNCFRAME) {
    pkt.flags |= AV_PKT_FLAG_KEY;
  }

  int err = av_write_frame(format_ctx, &pkt);
  if (err < 0) {
    LOGW("ts encoder write issue %d", err);
  }
  av_packet_unref(&pkt);
}

bool FFmpegEncoder::encode(AVFrame *frame) {
  int got_output = 0;
  AVPacket pkt = {};
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  int err = avcodec_encode_video2(codec_ctx, &pkt, frame, &got_output);
  if (err != 0 || !got_output) return false;

  av_packet_rescale_ts(&pkt, codec_ctx->time_base, stream->time_base);
  pkt.stream_index = 0;
  int ret = av_interleaved_write_frame(format_ctx, &pkt);
  av_packet_unref(&pkt);
  return ret >= 0;
}
