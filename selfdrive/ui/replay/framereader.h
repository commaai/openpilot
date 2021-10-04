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
  bool get(int idx, uint8_t* rgb_dat, uint8_t *yuv_dat = nullptr);
  int getRGBSize() const { return width * height * 3; }
  int getYUVSize() const { return width * height * 3 / 2; }
  size_t getFrameCount() const { return packets_.size(); }
  bool valid() const { return valid_; }

  int width = 0, height = 0;

private:
  bool decodeFrame(AVPacket *pkt, uint8_t *rgb_dat, uint8_t *yuv_dat = nullptr);

  std::vector<AVPacket> packets_;
  AVFormatContext *pFormatCtx_ = nullptr;
  AVCodecContext *pCodecCtx_ = nullptr;
  AVFrame *frmRgb_ = nullptr;
  struct SwsContext *sws_ctx_ = nullptr;
  bool valid_ = false;
};
