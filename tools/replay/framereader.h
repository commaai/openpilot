#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cereal/visionipc/visionbuf.h"
#include "tools/replay/filereader.h"

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
  bool load(const std::string &url, bool no_hw_decoder = false, std::atomic<bool> *abort = nullptr, bool local_cache = false,
            int chunk_size = -1, int retries = 0);
  bool loadFromFile(const std::string &file, bool no_hw_decoder = false, std::atomic<bool> *abort = nullptr);
  bool get(int idx, VisionBuf *buf);
  size_t getFrameCount() const { return packets_info.size(); }

  int width = 0, height = 0;

private:
  bool initHardwareDecoder(AVHWDeviceType hw_device_type);
  bool decode(int idx, VisionBuf *buf);
  AVFrame * decodeFrame(AVPacket *pkt);
  bool copyBuffers(AVFrame *f, VisionBuf *buf);

  std::unique_ptr<AVFrame, AVFrameDeleter>av_frame_, hw_frame;
  AVFormatContext *input_ctx = nullptr;
  AVCodecContext *decoder_ctx = nullptr;

  AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
  AVBufferRef *hw_device_ctx = nullptr;
  int prev_idx = -1;
  struct PacketInfo {
    int flags;
    int64_t pos;
  };
  std::vector<PacketInfo> packets_info;
  inline static std::atomic<bool> has_hw_decoder = true;
};
