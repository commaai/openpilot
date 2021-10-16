#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// independent of QT, needs ffmpeg
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
  std::optional<std::pair<uint8_t *, uint8_t*>> get(int idx);
  int getRGBSize() const { return width * height * 3; }
  int getYUVSize() const { return width * height * 3 / 2; }
  size_t getFrameCount() const { return frames_.size(); }
  bool valid() const { return valid_; }

  int width = 0, height = 0;

private:
  void decodeThread();
  std::pair<uint8_t *, uint8_t *> decodeFrame(AVPacket *pkt);
  struct Frame {
    AVPacket pkt = {};
    std::unique_ptr<uint8_t[]> rgb_data = nullptr;
    std::unique_ptr<uint8_t[]> yuv_data = nullptr;
    bool failed = false;
  };
  std::vector<Frame> frames_;

  AVFormatContext *pFormatCtx_ = nullptr;
  AVCodecContext *pCodecCtx_ = nullptr;
  struct SwsContext *sws_ctx_ = nullptr;

  std::mutex mutex_;
  std::condition_variable cv_decode_;
  std::condition_variable cv_frame_;
  int decode_idx_ = 0;
  std::atomic<bool> exit_ = false;
  bool valid_ = false;
  std::thread decode_thread_;
};
