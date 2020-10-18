#pragma once

#include <OMX_Component.h>
extern "C" {
#include <libavformat/avformat.h>
}
#include <stdint.h>
#include <stdio.h>

#include <mutex>

#include "camerad/cameras/camera_common.h"
#include "common/cqueue.h"
#include "common/visionipc.h"

class EncoderState {
public:
  EncoderState(const LogCameraInfo &camera_info, int width, int height);
  int EncodeFrame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, VIPCBufExtra *extra);
  void Rotate(const char *new_path);
  ~EncoderState();

  std::mutex state_lock;
  std::condition_variable state_cv;
  OMX_STATETYPE state;
  Queue free_in, done_out;

private:
  void Open(const char *path);
  void Close();
  void handle_out_buf(OMX_BUFFERHEADERTYPE *out_buf);
  void wait_for_state(OMX_STATETYPE state);

  LogCameraInfo camera_info;
  int width, height;
  char lock_path[4096];

  bool is_open, dirty, remuxing;
  int counter;

  int fd;
  size_t total_written;

  size_t codec_config_len;
  uint8_t *codec_config;
  bool wrote_codec_config;

  OMX_HANDLETYPE handle;
  int num_in_bufs, num_out_bufs;
  std::unique_ptr<OMX_BUFFERHEADERTYPE *[]> in_buf_headers, out_buf_headers;

  AVFormatContext *ofmt_ctx;
  AVCodecContext *codec_ctx;
  AVStream *out_stream;
};
