#include "tools/clip/recorder/ffmpeg.h"
#include <QDebug>

FFmpegEncoder::FFmpegEncoder(const std::string& outputFile, int width, int height, int fps) {
  // Allocate output context
  if (avformat_alloc_output_context2(&format_ctx, nullptr, nullptr, outputFile.c_str()) < 0) {
    return;
  }

  // Find H.264 encoder
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
  codec_ctx->time_base = {1, fps};
  codec_ctx->framerate = {fps, 1};
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->gop_size = 12;
  codec_ctx->max_b_frames = 0;
  codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

  stream->time_base = codec_ctx->time_base;

  // Set encoding options
  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "preset", "ultrafast", 0);
  av_dict_set(&opts, "tune", "zerolatency", 0);
  av_dict_set(&opts, "crf", "28", 0);

  // Open codec
  if (avcodec_open2(codec_ctx, codec, &opts) < 0) {
    av_dict_free(&opts);
    return;
  }
  av_dict_free(&opts);

  // Copy codec parameters to stream
  if (avcodec_parameters_from_context(stream->codecpar, codec_ctx) < 0) {
    return;
  }

  stream->time_base = codec_ctx->time_base;

  // Open output file
  if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&format_ctx->pb, outputFile.c_str(), AVIO_FLAG_WRITE) < 0) {
      return;
    }
  }

  // Write header
  if (avformat_write_header(format_ctx, nullptr) < 0) {
    return;
  }

  // Allocate frame
  frame = av_frame_alloc();
  if (!frame) {
    return;
  }
  frame->format = AV_PIX_FMT_YUV420P;
  frame->width = width;
  frame->height = height;
  if (av_frame_get_buffer(frame, 0) < 0) {
    return;
  }

  // Create scaling context
  sws_ctx = sws_getContext(width, height, AV_PIX_FMT_BGRA,
                           width, height, AV_PIX_FMT_YUV420P,
                           SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (!sws_ctx) {
    return;
  }

  // Allocate packet
  packet = av_packet_alloc();
  if (!packet) {
    return;
  }

  initialized = true;
}

FFmpegEncoder::~FFmpegEncoder() {
  if (initialized) {
    encodeFrame(nullptr);
    av_write_trailer(format_ctx);
    if (!(format_ctx->oformat->flags & AVFMT_NOFILE) && format_ctx->pb) {
      avio_closep(&format_ctx->pb);
    }
  }

  av_frame_free(&frame);
  av_packet_free(&packet);
  sws_freeContext(sws_ctx);
  avcodec_free_context(&codec_ctx);
  avformat_free_context(format_ctx);
}

bool FFmpegEncoder::writeFrame(const QImage& image) {
  if (!initialized) return false;

  // Convert BGRA to YUV420P
  uint8_t* inData[1] = { (uint8_t*)image.bits() };
  int inLinesize[1] = { image.bytesPerLine() };
  sws_scale(sws_ctx, inData, inLinesize, 0, image.height(),
            frame->data, frame->linesize);

  frame->pts = frame_count;  // PTS in codec_ctx->time_base units
  return encodeFrame(frame);
}

bool FFmpegEncoder::encodeFrame(AVFrame* input_frame) {
  int ret = avcodec_send_frame(codec_ctx, input_frame);
  if (ret < 0) {
    fprintf(stderr, "Failed to send frame: %d\n", ret);
    return false;
  }

  while (true) {
    ret = avcodec_receive_packet(codec_ctx, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    }
    if (ret < 0) {
      fprintf(stderr, "Error receiving packet: %d\n", ret);
      return false;
    }

    // Set packet timestamps and duration, rescaling if necessary
    packet->pts = av_rescale_q(frame_count, codec_ctx->time_base, stream->time_base);
    packet->dts = packet->pts;  // No B-frames, so DTS = PTS
    packet->duration = av_rescale_q(1, codec_ctx->time_base, stream->time_base);
    packet->stream_index = stream->index;

    ret = av_interleaved_write_frame(format_ctx, packet);
    av_packet_unref(packet);

    if (ret < 0) {
      fprintf(stderr, "Failed to write packet: %d\n", ret);
      return false;
    }

    if (input_frame) {
      frame_count++;  // Only increment on actual frames, not flushing
    }
  }
  return true;
}