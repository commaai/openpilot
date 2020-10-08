#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <OMX_Component.h>
extern "C" {
#include <libavformat/avformat.h>
}
#include "camerad/cameras/camera_common.h"
#include "common/cqueue.h"
#include "frame_logger.h"

class EncoderState : public FrameLogger {
 public:
  EncoderState(const LogCameraInfo &info, int width, int height, bool streaming = false);
  virtual ~EncoderState();
  bool dirty;

  FILE *of;

  size_t codec_config_len = 0;
  uint8_t *codec_config = nullptr;
  bool wrote_codec_config = false;

  pthread_mutex_t state_lock;
  pthread_cond_t state_cv;
  OMX_STATETYPE state;

  OMX_HANDLETYPE handle;

  int num_in_bufs;
  OMX_BUFFERHEADERTYPE** in_buf_headers;

  int num_out_bufs;
  OMX_BUFFERHEADERTYPE** out_buf_headers;

  Queue free_in;
  Queue done_out;

  AVFormatContext *ofmt_ctx;
  AVCodecContext *codec_ctx;
  AVStream *out_stream;
  bool remuxing;

  void *zmq_ctx = nullptr;
  void *stream_sock_raw = nullptr;

  std::unique_ptr<uint8_t[]> y_ptr2;
  std::unique_ptr<uint8_t[]> u_ptr2;
  std::unique_ptr<uint8_t[]> v_ptr2;

 private:
  bool ProcessFrame(uint64_t cnt, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                    int in_width, int in_height, const VIPCBufExtra &extra);
  bool Open(const std::string path, int segment);
  void Close();
  void Destroy();

  bool downscale;
};
