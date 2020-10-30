#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}
#include "camerad/cameras/camera_common.h"

class FFmpegEncoder{
public:
  FFmpegEncoder(std::string filename, AVCodecID codec_id, int bitrate, 
  int in_width, int in_height, int out_width, int out_height,  int fps);
  virtual ~FFmpegEncoder();
  void Rotate(const std::string new_path);
  int EncodeFrame(uint64_t cnt, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr);
  bool Open(const std::string path);
  void Close();
protected:
  int in_width, in_height, out_width, out_height, fps, bitrate, count;
  std::string filename, lock_path;

  AVCodec *codec = nullptr;
  AVCodecContext *codec_ctx = nullptr;

  AVStream *stream = nullptr;
  AVFormatContext *format_ctx = nullptr;

  AVFrame *frame = nullptr;
  std::unique_ptr<uint8_t[]> y_ptr2;
  std::unique_ptr<uint8_t[]> u_ptr2;
  std::unique_ptr<uint8_t[]> v_ptr2;
};

class RawEncoder : public FFmpegEncoder {
public:
  RawEncoder(std::string filename, int width, int height, int fps)
      : FFmpegEncoder(filename, AV_CODEC_ID_FFVHUFF, -1, width, height, width, height, fps) {}
};

class EncoderState : public FFmpegEncoder {
public:
  EncoderState(const LogCameraInfo& info, int width, int height)
      : FFmpegEncoder(info.filename, AV_CODEC_ID_H265, info.bitrate, width, height, width, height, info.fps) {}
  EncoderState(const LogCameraInfo& info, int in_width, int in_height, int out_width, int out_height)
      : FFmpegEncoder(info.filename, AV_CODEC_ID_H264, info.bitrate, in_width, in_height, out_width, out_height, info.fps) {}
};
