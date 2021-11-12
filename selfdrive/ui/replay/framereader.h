#pragma once

#include <string>
#include <vector>

#include "selfdrive/ui/replay/filereader.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

class FrameReader : protected FileReader {
public:
  FrameReader(bool local_cache = false, int chunk_size = -1, int retries = 0);
  ~FrameReader();
  bool load(const std::string &url, std::atomic<bool> *abort = nullptr);
  bool get(int idx, uint8_t *rgb, uint8_t *yuv);
  int getRGBSize() const { return width * height * 3; }
  int getYUVSize() const { return width * height * 3 / 2; }
  size_t getFrameCount() const { return frames_.size(); }
  bool valid() const { return valid_; }

  int width = 0, height = 0;

private:
  bool decode(int idx, uint8_t *rgb, uint8_t *yuv);
  bool decodeFrame(AVPacket *pkt);
  bool copyBuffers(AVFrame *f, uint8_t *rgb, uint8_t *yuv);

  struct Frame {
    AVPacket pkt = {};
    int decoded = false;
    bool failed = false;
  };
  std::vector<Frame> frames_;
  SwsContext *rgb_sws_ctx_ = nullptr, *yuv_sws_ctx_ = nullptr;
  AVFrame *av_frame_, *rgb_frame_, *yuv_frame_ = nullptr;
  AVFormatContext *pFormatCtx_ = nullptr;
  AVCodecContext *pCodecCtx_ = nullptr;
  int key_frames_count_ = 0;
  bool valid_ = false;
  AVIOContext *avio_ctx_ = nullptr;
};
