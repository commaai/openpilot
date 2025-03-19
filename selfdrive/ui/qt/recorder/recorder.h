#pragma once

#include <QImage>
#include <QString>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/dict.h>
#include <libswscale/swscale.h>
}

class FFmpegEncoder {
public:
  FFmpegEncoder(const QString& outputFile, int width, int height, int fps);
  ~FFmpegEncoder();

  bool writeFrame(const QImage& image);
  bool startRecording();

private:
  bool initialized = false;
  AVFormatContext* format_ctx = nullptr;
  AVCodecContext* codec_ctx = nullptr;
  AVStream* stream = nullptr;
  AVFrame* frame = nullptr;
  AVPacket* packet = nullptr;
  SwsContext* sws_ctx = nullptr;
  int64_t frame_count = 0;
};