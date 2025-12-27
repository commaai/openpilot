#pragma once

#include <string>
#include <vector>

#include "msgq/visionipc/visionbuf.h"
#include "tools/replay/filereader.h"
#include "tools/replay/util.h"

#ifndef __APPLE__
#include "tools/replay/qcom_decoder.h"
#endif

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
  int video_stream_idx_ = -1;
  int prev_idx = -1;
  struct PacketInfo {
    int flags;
    int64_t pos;
  };
  std::vector<PacketInfo> packets_info;
};


class VideoDecoder {
public:
  virtual ~VideoDecoder() = default;
  virtual bool open(AVCodecParameters *codecpar, bool hw_decoder) = 0;
  virtual bool decode(FrameReader *reader, int idx, VisionBuf *buf) = 0;
  int width = 0, height = 0;
};

class FFmpegVideoDecoder : public VideoDecoder {
public:
  FFmpegVideoDecoder();
  ~FFmpegVideoDecoder() override;
  bool open(AVCodecParameters *codecpar, bool hw_decoder) override;
  bool decode(FrameReader *reader, int idx, VisionBuf *buf) override;

private:
  bool initHardwareDecoder(AVHWDeviceType hw_device_type);
  AVFrame *decodeFrame(AVPacket *pkt);
  bool copyBuffer(AVFrame *f, VisionBuf *buf);

  AVFrame *av_frame_, *hw_frame_;
  AVCodecContext *decoder_ctx = nullptr;
  AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
  AVBufferRef *hw_device_ctx = nullptr;
};

#ifndef __APPLE__
class QcomVideoDecoder : public VideoDecoder {
public:
  QcomVideoDecoder() {};
  ~QcomVideoDecoder() override {};
  bool open(AVCodecParameters *codecpar, bool hw_decoder) override;
  bool decode(FrameReader *reader, int idx, VisionBuf *buf) override;

private:
  MsmVidc msm_vidc = MsmVidc();
};
#endif
