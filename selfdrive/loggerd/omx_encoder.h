#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include <pthread.h>
#include <OMX_Component.h>

extern "C" {
  #include <libavformat/avformat.h>
}

#include "encoder.h"
#include "common/cqueue.h"
#include "visionipc.h"

// OmxEncoder, lossey codec using hardware hevc
class OmxEncoder : public VideoEncoder {
public:
  OmxEncoder(const char* filename, int width, int height, int fps, int bitrate, bool h265, bool downscale);
  ~OmxEncoder();
  int encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                   int in_width, int in_height,
                   int *frame_segment, VisionIpcBufExtra *extra);
  void encoder_open(const char* path, int segment);
  void encoder_close();

  // OMX callbacks
  static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                     OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data);
  static OMX_ERRORTYPE empty_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                         OMX_BUFFERHEADERTYPE *buffer);
  static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                        OMX_BUFFERHEADERTYPE *buffer);

private:
  void wait_for_state(OMX_STATETYPE state);
  static void handle_out_buf(OmxEncoder *e, OMX_BUFFERHEADERTYPE *out_buf);

  pthread_mutex_t lock;
  int width, height, fps;
  char vid_path[1024];
  char lock_path[1024];
  bool is_open = false;
  bool dirty = false;
  int counter = 0;
  int segment = -1;

  const char* filename;
  FILE *of;

  size_t codec_config_len;
  uint8_t *codec_config = NULL;
  bool wrote_codec_config;

  pthread_mutex_t state_lock;
  pthread_cond_t state_cv;
  OMX_STATETYPE state = OMX_StateLoaded;

  OMX_HANDLETYPE handle;

  int num_in_bufs;
  OMX_BUFFERHEADERTYPE** in_buf_headers;

  int num_out_bufs;
  OMX_BUFFERHEADERTYPE** out_buf_headers;

  uint64_t last_t;

  Queue free_in;
  Queue done_out;

  AVFormatContext *ofmt_ctx;
  AVCodecContext *codec_ctx;
  AVStream *out_stream;
  bool remuxing;

  bool downscale;
  uint8_t *y_ptr2, *u_ptr2, *v_ptr2;
};
