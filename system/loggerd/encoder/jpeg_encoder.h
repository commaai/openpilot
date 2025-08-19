#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <jpeglib.h>
#include <vector>
#include <memory>
#include "cereal/messaging/messaging.h"
#include "msgq/visionipc/visionbuf.h"

class JpegEncoder {
public:
  JpegEncoder(const std::string &pusblish_name, int width, int height);
  ~JpegEncoder();
  void pushThumbnail(VisionBuf *buf, const VisionIpcBufExtra &extra);

private:
  void generateThumbnail(const uint8_t *y, const uint8_t *uv, int width, int height, int stride);
  void compressToJpeg(uint8_t *y_plane, uint8_t *u_plane, uint8_t *v_plane);

  int thumbnail_width;
  int thumbnail_height;
  std::string publish_name;
  std::vector<uint8_t> yuv_buffer;
  std::unique_ptr<PubMaster> pm;

  // JPEG output buffer
  unsigned char* out_buffer = nullptr;
  unsigned long out_size = 0;
};
