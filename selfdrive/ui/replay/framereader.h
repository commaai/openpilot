#pragma once

#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

class FrameReader {
public:
  FrameReader();
  ~FrameReader();
  bool load(const std::string &url);
  bool get(int idx, uint8_t *rgb, uint8_t *yuv);
  int getRGBSize() const { return width * height * 3; }
  int getYUVSize() const { return width * height * 3 / 2; }
  size_t getFrameCount() const { return frames_.size(); }
  bool valid() const { return valid_; }

  int width = 0, height = 0;

private:
  bool decode(int idx, uint8_t *rgb, uint8_t *yuv);
  bool decodeFrame(AVFrame *f, uint8_t *rgb, uint8_t *yuv);

  struct Frame {
    AVPacket pkt = {};
    int decoded = false;
    bool failed = false;
  };
  std::vector<Frame> frames_;
  AVFrame *av_frame_ = nullptr;
  AVFormatContext *pFormatCtx_ = nullptr;
  AVCodecContext *pCodecCtx_ = nullptr;
  int key_frames_count_ = 0;
  std::vector<uint8_t> yuv_buf_;
  bool valid_ = false;
};
