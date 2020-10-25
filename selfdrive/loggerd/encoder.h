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

class EncoderState {
public:
  EncoderState(const LogCameraInfo &camera_info, int width, int height);
  int EncodeFrame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, const VIPCBufExtra &extra);
  void Rotate(const char *new_path);
  ~EncoderState();

  std::mutex state_lock;
  std::condition_variable state_cv;
  OMX_STATETYPE state = OMX_StateLoaded;
  Queue free_in, done_out;

private:
  void Open(const char *path);
  void Close();
  void handle_out_buf(OMX_BUFFERHEADERTYPE *out_buf);
  void wait_for_state(OMX_STATETYPE state);

  LogCameraInfo camera_info;
  int in_width, in_height;
  char lock_path[4096];
  bool is_open = false, remuxing = false, dirty = false;
  int counter = 0;

  FILE *of = nullptr;

  std::vector<uint8_t> codec_config;
  bool wrote_codec_config = false;

  OMX_HANDLETYPE handle;
  std::vector<OMX_BUFFERHEADERTYPE *> in_buf_headers, out_buf_headers;

  AVFormatContext *ofmt_ctx;
  AVCodecContext *codec_ctx;
  AVStream *out_stream;
};
