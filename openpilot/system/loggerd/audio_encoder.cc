#include "system/loggerd/audio_encoder.h"

#include <algorithm>
#include <cassert>
#include <utility>

AudioEncoder::AudioEncoder(int sample_rate, Callback callback) : callback(std::move(callback)) {
  const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
  assert(codec);
  codec_ctx = avcodec_alloc_context3(codec);
  assert(codec_ctx);
  codec_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;
  codec_ctx->sample_rate = sample_rate;
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)
  av_channel_layout_default(&codec_ctx->ch_layout, 1);
#else
  codec_ctx->channel_layout = AV_CH_LAYOUT_MONO;
#endif
  codec_ctx->bit_rate = 48000;
  codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  codec_ctx->time_base = (AVRational){1, sample_rate};
  int err = avcodec_open2(codec_ctx, codec, nullptr);
  assert(err >= 0);
  av_log_set_level(AV_LOG_WARNING);

  frame = av_frame_alloc();
  assert(frame);
  frame->format = codec_ctx->sample_fmt;
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)
  av_channel_layout_copy(&frame->ch_layout, &codec_ctx->ch_layout);
#else
  frame->channel_layout = codec_ctx->channel_layout;
#endif
  frame->sample_rate = sample_rate;
  frame->nb_samples = codec_ctx->frame_size;
  err = av_frame_get_buffer(frame, 0);
  assert(err >= 0);
}

void AudioEncoder::write(const uint8_t *data, int len, uint64_t timestamp) {
  assert(len % sizeof(int16_t) == 0);
  if (sample_count == 0) {
    start_pts = next_pts = av_rescale_q(timestamp, (AVRational){1, 1000000000}, codec_ctx->time_base);
  }
  const int16_t *samples = reinterpret_cast<const int16_t *>(data);
  const int count = len / sizeof(int16_t);
  sample_count += count;
  for (int i = 0; i < count; ++i) {
    buffer.push_back(samples[i] / 32768.0f);
  }
  while (buffer.size() >= codec_ctx->frame_size) {
    int err = av_frame_make_writable(frame);
    assert(err >= 0);
    std::copy_n(buffer.begin(), frame->nb_samples, reinterpret_cast<float *>(frame->data[0]));
    buffer.erase(buffer.begin(), buffer.begin() + frame->nb_samples);
    frame->pts = next_pts;
    next_pts += frame->nb_samples;
    encode(frame);
  }
}

void AudioEncoder::encode(AVFrame *input) {
  int err = avcodec_send_frame(codec_ctx, input);
  assert(err >= 0);
  AudioPacket packet(av_packet_alloc());
  assert(packet);
  while ((err = avcodec_receive_packet(codec_ctx, packet.get())) == 0) {
    int discard_start = std::clamp<int64_t>(start_pts - packet->pts, 0, packet->duration);
    int discard_end = std::clamp<int64_t>(packet->pts + packet->duration - (start_pts + sample_count), 0, packet->duration - discard_start);
    callback(packet.get(), codec_ctx, discard_start, discard_end);
    av_packet_unref(packet.get());
  }
  assert(err == AVERROR(EAGAIN) || err == AVERROR_EOF);
}

AudioEncoder::~AudioEncoder() {
  if (sample_count > 0) {
    if (!buffer.empty()) {
      int err = av_frame_make_writable(frame);
      assert(err >= 0);
      buffer.resize(frame->nb_samples, 0.0f);
      std::copy(buffer.begin(), buffer.end(), reinterpret_cast<float *>(frame->data[0]));
      frame->pts = next_pts;
      encode(frame);
    }
    encode(nullptr);
  }
  av_frame_free(&frame);
  avcodec_free_context(&codec_ctx);
}
