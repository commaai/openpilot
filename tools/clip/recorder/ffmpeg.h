#ifndef FFMPEG_ENCODER_H
#define FFMPEG_ENCODER_H

#include <QImage>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

class FFmpegEncoder {
public:
  FFmpegEncoder(const std::string& outputFile, int width, int height, int fps);
  ~FFmpegEncoder();
  bool writeFrame(const QImage& image);

private:
  bool initialized = false;
  int64_t frame_count = 0;
  AVFormatContext* format_ctx = nullptr;
  AVStream* stream = nullptr;
  AVCodecContext* codec_ctx = nullptr;
  AVFrame* frame = nullptr;
  AVPacket* packet = nullptr;
  SwsContext* sws_ctx = nullptr;

  bool encodeFrame(AVFrame* input_frame);
};

#endif // FFMPEG_ENCODER_H