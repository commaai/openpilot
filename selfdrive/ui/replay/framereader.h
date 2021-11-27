#pragma once

#include <memory>
#include <string>
#include <vector>

#include "selfdrive/ui/replay/filereader.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

struct AVFrameDeleter {
  void operator()(AVFrame* frame) const { av_frame_free(&frame); }
};

class FrameReader {
public:
  FrameReader();
  ~FrameReader();
  bool load(const std::string &url, bool no_cuda = false, std::atomic<bool> *abort = nullptr, bool local_cache = false, int chunk_size = -1, int retries = 0);
  bool load(const std::byte *data, size_t size, bool no_cuda = false, std::atomic<bool> *abort = nullptr);
  bool get(int idx, uint8_t *rgb, uint8_t *yuv);
  int getRGBSize() const { return width * height * 3; }
  int getYUVSize() const { return width * height * 3 / 2; }
  size_t getFrameCount() const { return frames_.size(); }
  bool valid() const { return valid_; }

  int width = 0, height = 0;

private:
  bool initHardwareDecoder(AVHWDeviceType hw_device_type);
  bool decode(int idx, uint8_t *rgb, uint8_t *yuv);
  AVFrame * decodeFrame(AVPacket *pkt);
  bool copyBuffers(AVFrame *f, uint8_t *rgb, uint8_t *yuv);

  struct Frame {
    AVPacket pkt = {};
    int decoded = false;
    bool failed = false;
  };
  std::vector<Frame> frames_;
  std::unique_ptr<AVFrame, AVFrameDeleter>av_frame_, hw_frame;
  AVFormatContext *input_ctx = nullptr;
  AVCodecContext *decoder_ctx = nullptr;
  int key_frames_count_ = 0;
  bool valid_ = false;
  AVIOContext *avio_ctx_ = nullptr;

  AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
  AVBufferRef *hw_device_ctx = nullptr;
  std::vector<uint8_t> nv12toyuv_buffer;
};
