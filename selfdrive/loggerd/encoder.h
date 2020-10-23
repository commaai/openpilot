#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include <OMX_Component.h>
#include <libavformat/avformat.h>

#include "common/cqueue.h"
#include "common/visionipc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EncoderState {
  int width, height, fps;
  const char* path;
  char vid_path[1024];
  bool open;
  bool dirty;
  int counter;

  const char* filename;
  FILE *of;

  size_t codec_config_len;
  uint8_t *codec_config;
  bool wrote_codec_config;

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

  bool downscale;
  uint8_t *y_ptr2, *u_ptr2, *v_ptr2;
} EncoderState;

void encoder_init(EncoderState *s, const char* filename, int width, int height, int fps, int bitrate, bool h265, bool downscale);
int encoder_encode_frame(EncoderState *s, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                         int in_width, int in_height, VIPCBufExtra *extra);

void encoder_open(EncoderState *s, const char* path);
void encoder_rotate(EncoderState *s, const char* new_path);
void encoder_close(EncoderState *s);
void encoder_destroy(EncoderState *s);

#ifdef __cplusplus
}
#endif
