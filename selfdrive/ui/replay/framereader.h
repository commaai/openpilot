#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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
  std::optional<std::pair<uint8_t *, uint8_t *>> get(int idx);
  int getRGBSize() const { return width * height * 3; }
  int getYUVSize() const { return width * height * 3 / 2; }
  size_t getFrameCount() const { return frames_.size(); }
  bool valid() const { return valid_; }

  int width = 0, height = 0;

private:
  struct Buffer {
    Buffer(int rgb_size, int yuv_size) {
      rgb = std::make_unique<uint8_t[]>(rgb_size);
      yuv = std::make_unique<uint8_t[]>(yuv_size);
    }
    std::unique_ptr<uint8_t[]> rgb;
    std::unique_ptr<uint8_t[]> yuv;
  };

  struct Frame {
    AVPacket pkt = {};
    int decoded = false;
    bool failed = false;
    Buffer *buf = nullptr;

  };

  void decodeThread();
  Buffer *decodeFrame(AVFrame *f);

  std::vector<Frame> frames_;
  AVFormatContext *pFormatCtx_ = nullptr;
  AVCodecContext *pCodecCtx_ = nullptr;
  std::mutex mutex_;
  std::condition_variable cv_decode_;
  int key_frames_count_ = 0;
  std::atomic<int> decode_idx_ = -1;
  std::atomic<bool> exit_ = false;
  std::vector<Buffer *> buffers_;
  bool valid_ = false;
  std::thread decode_thread_;
};
