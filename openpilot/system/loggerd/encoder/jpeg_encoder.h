#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openpilot/cereal/messaging/messaging.h"
#include "msgq/visionipc/visionbuf.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

class JpegEncoder {
public:
  JpegEncoder(const std::string &publish_name, int width, int height);
  ~JpegEncoder();
  void pushThumbnail(VisionBuf *buf, const VisionIpcBufExtra &extra);

private:
  void generateThumbnail(const uint8_t *y, const uint8_t *uv, int width, int height, int stride);
  void compressToJpeg(uint8_t *y_plane, uint8_t *u_plane, uint8_t *v_plane);

  int thumbnail_width;
  int thumbnail_height;
  std::string publish_name;
  std::vector<uint8_t> yuv_buffer;
  std::vector<uint8_t> out_buffer;
  std::unique_ptr<PubMaster> pm;

  AVCodecContext *codec_ctx = nullptr;
  AVFrame *frame = nullptr;
  AVPacket *pkt = nullptr;
};
