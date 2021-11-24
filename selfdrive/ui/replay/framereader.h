#pragma once

#include <memory>
#include <string>
#include <vector>

#include "selfdrive/ui/replay/filereader.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

struct AVFrameDeleter {
  void operator()(AVFrame* frame) const { av_frame_free(&frame); }
};

class FrameReader : protected FileReader {
public:
  FrameReader(bool local_cache = false, int chunk_size = -1, int retries = 0);
  ~FrameReader();
  bool load(const std::string &url, bool no_cuda = false, std::atomic<bool> *abort = nullptr);
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
  AVPixelFormat sws_src_format = AV_PIX_FMT_YUV420P;
  SwsContext *rgb_sws_ctx_ = nullptr, *yuv_sws_ctx_ = nullptr;
  std::unique_ptr<AVFrame, AVFrameDeleter>av_frame_, sws_frame, hw_frame;
  AVFormatContext *input_ctx = nullptr;
  AVCodecContext *decoder_ctx = nullptr;
  int key_frames_count_ = 0;
  bool valid_ = false;
  AVIOContext *avio_ctx_ = nullptr;

  AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
  AVBufferRef *hw_device_ctx = nullptr;
};
