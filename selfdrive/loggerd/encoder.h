#pragma once

#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}
#include <OMX_Component.h>

class FFmpegEncoder {
public:
  FFmpegEncoder(AVCodecID codec_id, int width, int height, int fps);
  ~FFmpegEncoder();
  void open(const char *vid_path);
  void remux(OMX_BUFFERHEADERTYPE *out_buf);
  bool encode(AVFrame *frame, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr);
  void writeHeader(const std::vector<uint8_t> &header = {});
  void close();

  AVStream *stream = nullptr;
  AVCodec *codec = nullptr;
  AVCodecContext *codec_ctx = nullptr;
  AVFormatContext *format_ctx = nullptr;
  int fps;
};
