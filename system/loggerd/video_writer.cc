#include <cassert>

#include "system/loggerd/video_writer.h"
#include "common/swaglog.h"
#include "common/util.h"

VideoWriter::VideoWriter(const char *path, const char *filename, bool remuxing, int width, int height, int fps, cereal::EncodeIndex::Type codec)
  : remuxing(remuxing) {
  vid_path = util::string_format("%s/%s", path, filename);
  lock_path = util::string_format("%s/%s.lock", path, filename);

  int lock_fd = HANDLE_EINTR(open(lock_path.c_str(), O_RDWR | O_CREAT, 0664));
  assert(lock_fd >= 0);
  close(lock_fd);

  LOGD("encoder_open %s remuxing:%d", this->vid_path.c_str(), this->remuxing);
  if (this->remuxing) {
    bool raw = (codec == cereal::EncodeIndex::Type::BIG_BOX_LOSSLESS);
    avformat_alloc_output_context2(&this->ofmt_ctx, NULL, raw ? "matroska" : NULL, this->vid_path.c_str());
    assert(this->ofmt_ctx);

    // set codec correctly. needed?
    assert(codec != cereal::EncodeIndex::Type::FULL_H_E_V_C);
    const AVCodec *avcodec = avcodec_find_encoder(raw ? AV_CODEC_ID_FFVHUFF : AV_CODEC_ID_H264);
    assert(avcodec);

    this->codec_ctx = avcodec_alloc_context3(avcodec);
    assert(this->codec_ctx);
    this->codec_ctx->width = width;
    this->codec_ctx->height = height;
    this->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    this->codec_ctx->time_base = (AVRational){ 1, fps };

    if (codec == cereal::EncodeIndex::Type::BIG_BOX_LOSSLESS) {
      // without this, there's just noise
      int err = avcodec_open2(this->codec_ctx, avcodec, NULL);
      assert(err >= 0);
    }

    this->out_stream = avformat_new_stream(this->ofmt_ctx, raw ? avcodec : NULL);
    assert(this->out_stream);

    int err = avio_open(&this->ofmt_ctx->pb, this->vid_path.c_str(), AVIO_FLAG_WRITE);
    assert(err >= 0);

  } else {
    this->of = util::safe_fopen(this->vid_path.c_str(), "wb");
    assert(this->of);
  }
}

void VideoWriter::initialize_audio(int sample_rate) {
  assert(this->ofmt_ctx->oformat->audio_codec != AV_CODEC_ID_NONE); // check output format supports audio streams
  const AVCodec *audio_avcodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
  assert(audio_avcodec);
  this->audio_codec_ctx = avcodec_alloc_context3(audio_avcodec);
  assert(this->audio_codec_ctx);
  this->audio_codec_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;
  this->audio_codec_ctx->sample_rate = sample_rate;
  #if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)  // FFmpeg 5.1+
  av_channel_layout_default(&this->audio_codec_ctx->ch_layout, 1);
  #else
  this->audio_codec_ctx->channel_layout = AV_CH_LAYOUT_MONO;
  #endif
  this->audio_codec_ctx->bit_rate = 32000;
  this->audio_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  this->audio_codec_ctx->time_base = (AVRational){1, audio_codec_ctx->sample_rate};
  int err = avcodec_open2(this->audio_codec_ctx, audio_avcodec, NULL);
  assert(err >= 0);
  av_log_set_level(AV_LOG_WARNING); // hide "QAvg" info msgs at the end of every segment

  this->audio_stream = avformat_new_stream(this->ofmt_ctx, NULL);
  assert(this->audio_stream);
  err = avcodec_parameters_from_context(this->audio_stream->codecpar, this->audio_codec_ctx);
  assert(err >= 0);

  this->audio_frame = av_frame_alloc();
  assert(this->audio_frame);
  this->audio_frame->format = this->audio_codec_ctx->sample_fmt;
  #if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)  // FFmpeg 5.1+
  av_channel_layout_copy(&this->audio_frame->ch_layout, &this->audio_codec_ctx->ch_layout);
  #else
  this->audio_frame->channel_layout = this->audio_codec_ctx->channel_layout;
  #endif
  this->audio_frame->sample_rate = this->audio_codec_ctx->sample_rate;
  this->audio_frame->nb_samples = this->audio_codec_ctx->frame_size;
  err = av_frame_get_buffer(this->audio_frame, 0);
  assert(err >= 0);
}

void VideoWriter::write(uint8_t *data, int len, long long timestamp, bool codecconfig, bool keyframe) {
  if (of && data) {
    size_t written = util::safe_fwrite(data, 1, len, of);
    if (written != len) {
      LOGE("failed to write file.errno=%d", errno);
    }
  }

  if (remuxing) {
    if (codecconfig) {
      if (len > 0) {
        codec_ctx->extradata = (uint8_t*)av_mallocz(len + AV_INPUT_BUFFER_PADDING_SIZE);
        codec_ctx->extradata_size = len;
        memcpy(codec_ctx->extradata, data, len);
      }
      int err = avcodec_parameters_from_context(out_stream->codecpar, codec_ctx);
      assert(err >= 0);
      // if there is an audio stream, it must be initialized before this point
      err = avformat_write_header(ofmt_ctx, NULL);
      assert(err >= 0);
      header_written = true;
    } else {
      // input timestamps are in microseconds
      AVRational in_timebase = {1, 1000000};

      AVPacket pkt = {};
      pkt.data = data;
      pkt.size = len;
      pkt.stream_index = this->out_stream->index;

      enum AVRounding rnd = static_cast<enum AVRounding>(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
      pkt.pts = pkt.dts = av_rescale_q_rnd(timestamp, in_timebase, ofmt_ctx->streams[0]->time_base, rnd);
      pkt.duration = av_rescale_q(50*1000, in_timebase, ofmt_ctx->streams[0]->time_base);

      if (keyframe) {
        pkt.flags |= AV_PKT_FLAG_KEY;
      }

      // TODO: can use av_write_frame for non raw?
      int err = av_interleaved_write_frame(ofmt_ctx, &pkt);
      if (err < 0) { LOGW("ts encoder write issue len: %d ts: %lld", len, timestamp); }

      av_packet_unref(&pkt);
    }
  }
}

void VideoWriter::write_audio(uint8_t *data, int len, long long timestamp, int sample_rate) {
  if (!remuxing) return;
  if (!audio_initialized) {
    initialize_audio(sample_rate);
    audio_initialized = true;
  }
  if (!audio_codec_ctx) return;
  // sync logMonoTime of first audio packet with the timestampEof of first video packet
  if (audio_pts == 0) {
    audio_pts = (timestamp * audio_codec_ctx->sample_rate) / 1000000ULL;
  }

  // convert s16le samples to fltp and add to buffer
  const int16_t *raw_samples = reinterpret_cast<const int16_t*>(data);
  int sample_count = len / sizeof(int16_t);
  constexpr float normalizer = 1.0f / 32768.0f;

  const size_t max_buffer_size = sample_rate * 10; // 10 seconds
  if (audio_buffer.size() + sample_count > max_buffer_size) {
    size_t samples_to_drop = (audio_buffer.size() + sample_count) - max_buffer_size;
    LOGE("Audio buffer overflow, dropping %zu oldest samples", samples_to_drop);
    audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + samples_to_drop);
    audio_pts += samples_to_drop;
  }

  // Add new samples to the buffer
  const size_t original_size = audio_buffer.size();
  audio_buffer.resize(original_size + sample_count);
  std::transform(raw_samples, raw_samples + sample_count, audio_buffer.begin() + original_size,
                [](int16_t sample) { return sample * normalizer; });

  if (!header_written) return; // header not written yet, process audio frame after header is written
  while (audio_buffer.size() >= audio_codec_ctx->frame_size) {
    audio_frame->pts = audio_pts;
    float *f_samples = reinterpret_cast<float*>(audio_frame->data[0]);
    std::copy(audio_buffer.begin(), audio_buffer.begin() + audio_codec_ctx->frame_size, f_samples);
    audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + audio_codec_ctx->frame_size);
    encode_and_write_audio_frame(audio_frame);
  }
}

void VideoWriter::encode_and_write_audio_frame(AVFrame* frame) {
  if (!remuxing || !audio_codec_ctx) return;
  int send_result = avcodec_send_frame(audio_codec_ctx, frame); // encode frame
  if (send_result >= 0) {
    AVPacket *pkt = av_packet_alloc();
    while (avcodec_receive_packet(audio_codec_ctx, pkt) == 0) {
      av_packet_rescale_ts(pkt, audio_codec_ctx->time_base, audio_stream->time_base);
      pkt->stream_index = audio_stream->index;

      int err = av_interleaved_write_frame(ofmt_ctx, pkt); // write encoded frame
      if (err < 0) {
        LOGW("AUDIO: Write frame failed - error: %d", err);
      }
      av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
  } else {
    LOGW("AUDIO: Failed to send audio frame to encoder: %d", send_result);
  }
  audio_pts += audio_codec_ctx->frame_size;
}

void VideoWriter::process_remaining_audio() {
  // Process remaining audio samples by padding with silence
  if (audio_buffer.size() > 0 && audio_buffer.size() < audio_codec_ctx->frame_size) {
    audio_buffer.resize(audio_codec_ctx->frame_size, 0.0f);

    // Encode final frame
    audio_frame->pts = audio_pts;
    float *f_samples = reinterpret_cast<float *>(audio_frame->data[0]);
    std::copy(audio_buffer.begin(), audio_buffer.end(), f_samples);
    encode_and_write_audio_frame(audio_frame);
  }
}

VideoWriter::~VideoWriter() {
  if (this->remuxing) {
    if (this->audio_codec_ctx) {
      process_remaining_audio();
      encode_and_write_audio_frame(NULL); // flush encoder
      avcodec_free_context(&this->audio_codec_ctx);
    }
    int err = av_write_trailer(this->ofmt_ctx);
    if (err != 0) LOGE("av_write_trailer failed %d", err);
    avcodec_free_context(&this->codec_ctx);
    if (this->audio_frame) av_frame_free(&this->audio_frame);
    err = avio_closep(&this->ofmt_ctx->pb);
    if (err != 0) LOGE("avio_closep failed %d", err);
    avformat_free_context(this->ofmt_ctx);
  } else {
    util::safe_fflush(this->of);
    fclose(this->of);
    this->of = nullptr;
  }
  unlink(this->lock_path.c_str());
}
