#pragma once

#include <string>
#include <vector>

#include "tools/replay/util.h"

#ifdef __EMSCRIPTEN__
#include <cstdint>

// demuxes the video into Annex B access units; decoding happens in the browser (see tools/cabana/web/video.cc)
class FrameReader {
public:
  struct Packet {
    size_t offset;
    size_t size;
    bool key;
  };
  bool load(CameraType type, const std::string &url, bool no_hw_decoder = false, std::atomic<bool> *abort = nullptr, bool local_cache = false);
  size_t getFrameCount() const { return packets.size(); }

  int width = 0, height = 0;
  const char *codec = "";  // WebCodecs codec string
  std::vector<Packet> packets;
  std::vector<uint8_t> data;
};
#else
#include "msgq/visionipc/visionbuf.h"

#ifndef __APPLE__
#include "system/loggerd/encoder/v4l_decoder.h"
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
  bool load(CameraType type, const std::string &url, bool no_hw_decoder = false, std::atomic<bool> *abort = nullptr, bool local_cache = false);
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
class V4LVideoDecoder : public VideoDecoder {
public:
  V4LVideoDecoder() {};
  ~V4LVideoDecoder() override {};
  bool open(AVCodecParameters *codecpar, bool hw_decoder) override;
  bool decode(FrameReader *reader, int idx, VisionBuf *buf) override;

private:
  V4LDecoder v4l_decoder;
};
#endif
#endif  // __EMSCRIPTEN__
