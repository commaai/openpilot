#include "tools/replay/clip/recorder/ffmpeg.h"

#include <QDebug>

FFmpegEncoder::FFmpegEncoder(const QString& outputFile, int width, int height, int fps) {
  // Enable FFmpeg logging to stderr
  av_log_set_level(AV_LOG_ERROR);
  av_log_set_callback([](void* ptr, int level, const char* fmt, va_list vargs) {
    if (level <= av_log_get_level()) {
      vfprintf(stderr, fmt, vargs);
    }
  });

  // Allocate output context
  avformat_alloc_output_context2(&format_ctx, nullptr, nullptr, outputFile.toStdString().c_str());
  if (!format_ctx) {
    return;
  }

  // Find the H264 encoder
  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
  if (!codec) {
    return;
  }

  // Create video stream
  stream = avformat_new_stream(format_ctx, codec);
  if (!stream) {
    return;
  }

  // Create codec context
  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    return;
  }

  // Set codec parameters
  codec_ctx->codec_id = AV_CODEC_ID_H264;
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->time_base = (AVRational){1, fps};
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  codec_ctx->gop_size = 24;
  codec_ctx->max_b_frames = 0;
  codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

  // Set encoding parameters using AVDictionary
  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "preset", "ultrafast", 0);
  av_dict_set(&opts, "profile", "baseline", 0);
  av_dict_set(&opts, "crf", "28", 0);

  // Open codec with options
  if (avcodec_open2(codec_ctx, codec, &opts) < 0) {
    av_dict_free(&opts);
    return;
  }

  // Free options dictionary
  av_dict_free(&opts);

  // Copy codec parameters to stream
  if (avcodec_parameters_from_context(stream->codecpar, codec_ctx) < 0) {
    return;
  }

  // Set stream time base
  stream->time_base = codec_ctx->time_base;

  // Open output file
  if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&format_ctx->pb, outputFile.toStdString().c_str(), AVIO_FLAG_WRITE) < 0) {
      return;
    }
  }

  // Allocate frame
  frame = av_frame_alloc();
  if (!frame) {
    return;
  }

  frame->format = codec_ctx->pix_fmt;
  frame->width = width;
  frame->height = height;

  // Allocate frame buffer
  int ret = av_image_alloc(frame->data, frame->linesize,
                         width, height, codec_ctx->pix_fmt, 1);
  if (ret < 0) {
    return;
  }

  // Create scaling context
  sws_ctx = sws_getContext(width, height, AV_PIX_FMT_BGRA,
                         width, height, codec_ctx->pix_fmt,
                         SWS_BILINEAR, nullptr, nullptr, nullptr);

  // Allocate packet
  packet = av_packet_alloc();
  if (!packet) {
    return;
  }

  initialized = true;
}

FFmpegEncoder::~FFmpegEncoder() {
  if (initialized) {
    // Write trailer
    av_write_trailer(format_ctx);

    // Close output file
    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
      avio_closep(&format_ctx->pb);
    }

    // Free resources
    avcodec_free_context(&codec_ctx);
    avformat_free_context(format_ctx);
    av_frame_free(&frame);
    av_packet_free(&packet);
    sws_freeContext(sws_ctx);
  }
}

bool FFmpegEncoder::startRecording() {
  if (!initialized) return false;

  // Write header
  if (avformat_write_header(format_ctx, nullptr) < 0) {
    return false;
  }

  return true;
}

bool FFmpegEncoder::writeFrame(const QImage& image) {
  if (!initialized) return false;

  // Convert BGRA to YUV420P
  uint8_t* inData[4] = { (uint8_t*)image.bits(), nullptr, nullptr, nullptr };
  int inLinesize[4] = { image.bytesPerLine(), 0, 0, 0 };
  sws_scale(sws_ctx, inData, inLinesize, 0, image.height(),
            frame->data, frame->linesize);

  // Set frame timestamp and duration
  frame->pts = frame_count;
  frame->duration = 1;  // Each frame has duration of 1 in the time base units

  // Send frame to encoder
  int ret = avcodec_send_frame(codec_ctx, frame);
  if (ret < 0) {
    return false;
  }

  // Read encoded packets
  while (ret >= 0) {
    ret = avcodec_receive_packet(codec_ctx, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    } else if (ret < 0) {
      return false;
    }

    // Set packet timestamp and duration
    packet->pts = av_rescale_q(frame_count, codec_ctx->time_base, stream->time_base);
    packet->dts = packet->pts;
    packet->duration = av_rescale_q(1, codec_ctx->time_base, stream->time_base);

    // Write packet to output file
    packet->stream_index = stream->index;
    ret = av_interleaved_write_frame(format_ctx, packet);
    if (ret < 0) {
      return false;
    }

    av_packet_unref(packet);
  }

  frame_count++;
  return true;
}