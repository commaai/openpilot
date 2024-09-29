#pragma once

#include <string>
#include <vector>

#include "msgq/visionipc/visionbuf.h"
#include "tools/replay/filereader.h"
#include "tools/replay/util.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

class VideoDecoder;

class FrameReader {
public:
  FrameReader();
  ~FrameReader();
  bool load(CameraType type, const std::string &url, bool no_hw_decoder = false, std::atomic<bool> *abort = nullptr, bool local_cache = false,
            int chunk_size = -1, int retries = 0);
  bool loadFromFile(CameraType type, const std::string &file, bool no_hw_decoder = false, std::atomic<bool> *abort = nullptr);
  bool get(int idx, VisionBuf *buf);
  size_t getFrameCount() const { return packets_info.size(); }

  int width = 0, height = 0;

  VideoDecoder *decoder_ = nullptr;
  AVFormatContext *input_ctx = nullptr;
  int prev_idx = -1;
  struct PacketInfo {
    int flags;
    int64_t pos;
  };
  std::vector<PacketInfo> packets_info;
};


class VideoDecoder {
public:
  VideoDecoder();
  ~VideoDecoder();
  bool open(AVCodecParameters *codecpar, bool hw_decoder);
  bool decode(FrameReader *reader, int idx, VisionBuf *buf);
  int width = 0, height = 0;

private:
  bool initHardwareDecoder(AVHWDeviceType hw_device_type);
  AVFrame *decodeFrame(AVPacket *pkt);
  bool copyBuffer(AVFrame *f, VisionBuf *buf);

  AVFrame *av_frame_, *hw_frame_;
  AVCodecContext *decoder_ctx = nullptr;
  AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
  AVBufferRef *hw_device_ctx = nullptr;
};
