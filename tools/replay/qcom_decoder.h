#pragma once

#include <linux/videodev2.h>

#include "msgq/visionipc/visionbuf.h"
#include "third_party/linux/include/media/msm_vidc.h"

extern "C" {
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
}

#define V4L2_CID_MPEG_MSM_VIDC_BASE 0x00992000
#define V4L2_CID_MPEG_VIDC_VIDEO_DPB_COLOR_FORMAT (V4L2_CID_MPEG_MSM_VIDC_BASE + 44)
#define V4L2_CID_MPEG_VIDC_VIDEO_STREAM_OUTPUT_MODE (V4L2_CID_MPEG_MSM_VIDC_BASE + 22)

#define VIDEO_DEVICE "/dev/video32"

class MsmVidc {
public:
  MsmVidc() = default;
  ~MsmVidc();

  bool init(const char* dev, size_t width, size_t height, uint64_t codec);
  bool decodeFrame(AVPacket* pkt, VisionBuf* output_buf);

private:
  bool initialized = false;
  int fd = 0;
  size_t w = 0, h = 0;
  VisionBuf input_buf;
};