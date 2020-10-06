#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

#include "frame_logger.h"

class FFmpegEncoder : public FrameLogger {
 public:
  FFmpegEncoder(std::string filename, AVCodecID codec_id, int bitrate, int width, int height, int fps);
  virtual ~FFmpegEncoder();

 protected:
  virtual bool ProcessFrame(uint64_t cnt, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                            int in_width, int in_height, const VIPCBufExtra &extra);
  virtual bool Open(const std::string path, int segment);
  virtual void Close();

  AVCodec *codec = NULL;
  AVCodecContext *codec_ctx = NULL;

  AVStream *stream = NULL;
  AVFormatContext *format_ctx = NULL;

  AVFrame *frame = NULL;
};

class RawEncoder : public FFmpegEncoder {
 public:
  RawEncoder(std::string filename, int width, int height, int fps)
      : FFmpegEncoder(filename, AV_CODEC_ID_FFVHUFF, -1, width, height, fps) {}
  virtual bool Open(const std::string path, int segment) {
    if (segment == 0) {  // ignore first segment
      return false;
    }
    return FFmpegEncoder::Open(path, segment);
  }
};

#if !(defined(QCOM) || defined(QCOM2))
#include "camerad/cameras/camera_common.h"
class EncoderState : public FFmpegEncoder {
 public:
  EncoderState(const LogCameraInfo& info, int width, int height, bool is_streaming = false)
      : FFmpegEncoder(info.filename, AV_CODEC_ID_H265, info.bitrate, width, height, info.fps) {}
};
#endif

