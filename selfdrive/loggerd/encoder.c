#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "encoder.h"

#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <OMX_Component.h>
#include <OMX_IndexExt.h>
#include <OMX_VideoExt.h>
#include <OMX_QCOMExtns.h>

#include <libyuv.h>

#include <msm_media_info.h>

#include "common/mutex.h"
#include "common/swaglog.h"

#define PORT_INDEX_IN 0
#define PORT_INDEX_OUT 1

#define OERR(cmd) assert((cmd) == OMX_ErrorNone)

static void wait_for_state(EncoderState *s, OMX_STATETYPE state) {
  pthread_mutex_lock(&s->state_lock);
  while (s->state != state) {
    pthread_cond_wait(&s->state_cv, &s->state_lock);
  }
  pthread_mutex_unlock(&s->state_lock);
}

static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                   OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data) {
  EncoderState *s = app_data;

  switch (event) {
  case OMX_EventCmdComplete:
    assert(data1 == OMX_CommandStateSet);
    LOG("set state event 0x%x", data2);
    pthread_mutex_lock(&s->state_lock);
    s->state = data2;
    pthread_cond_broadcast(&s->state_cv);
    pthread_mutex_unlock(&s->state_lock);
    break;
  case OMX_EventError:
    LOGE("OMX error 0x%08x", data1);
    // assert(false);
    break;
  default:
    LOGE("unhandled event %d", event);
    assert(false);
    break;
  }

  return OMX_ErrorNone;
}

static OMX_ERRORTYPE empty_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                       OMX_BUFFERHEADERTYPE *buffer) {
  EncoderState *s = app_data;
  queue_push(&s->free_in, (void*)buffer);
  return OMX_ErrorNone;
}

static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                      OMX_BUFFERHEADERTYPE *buffer) {
  EncoderState *s = app_data;
  queue_push(&s->done_out, (void*)buffer);
  return OMX_ErrorNone;
}

static OMX_CALLBACKTYPE omx_callbacks = {
  .EventHandler = event_handler,
  .EmptyBufferDone = empty_buffer_done,
  .FillBufferDone = fill_buffer_done,
};

void encoder_init(EncoderState *s, LogCameraInfo *camera_info, int width, int height) {
  int err;

  memset(s, 0, sizeof(*s));
  s->camera_info = *camera_info; 
  s->width = width;
  s->height = height;
  s->state = OMX_StateLoaded;
  if (!camera_info->is_h265) {
    s->remuxing = true;
  }

  queue_init(&s->free_in);
  queue_init(&s->done_out);

  mutex_init_reentrant(&s->lock);
  pthread_mutex_init(&s->state_lock, NULL);
  pthread_cond_init(&s->state_cv, NULL);

  if (camera_info->is_h265) {
    err = OMX_GetHandle(&s->handle, (OMX_STRING)"OMX.qcom.video.encoder.hevc", s, &omx_callbacks);
  } else {
    err = OMX_GetHandle(&s->handle, (OMX_STRING)"OMX.qcom.video.encoder.avc", s, &omx_callbacks);
  }
  if (err != OMX_ErrorNone) {
    LOGE("error getting codec: %x", err);
  }
  assert(err == OMX_ErrorNone);

  // setup input port

  OMX_PARAM_PORTDEFINITIONTYPE in_port = {0};
  in_port.nSize = sizeof(in_port);
  in_port.nPortIndex = (OMX_U32) PORT_INDEX_IN;
  OERR(OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &in_port));

  in_port.format.video.nFrameWidth = s->width;
  in_port.format.video.nFrameHeight = s->height;
  in_port.format.video.nStride = VENUS_Y_STRIDE(COLOR_FMT_NV12, s->width);
  in_port.format.video.nSliceHeight = s->height;
  in_port.nBufferSize = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, s->width, s->height);
  in_port.format.video.xFramerate = (s->camera_info.fps * 65536);
  in_port.format.video.eCompressionFormat = OMX_VIDEO_CodingUnused;
  in_port.format.video.eColorFormat = (OMX_COLOR_FORMATTYPE)QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m;

  OERR(OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &in_port));
  OERR(OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &in_port));
  s->num_in_bufs = in_port.nBufferCountActual;

  // setup output port

  OMX_PARAM_PORTDEFINITIONTYPE out_port = {0};
  out_port.nSize = sizeof(out_port);
  out_port.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  OERR(OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, (OMX_PTR)&out_port));
  out_port.format.video.nFrameWidth = camera_info->downscale ? camera_info->frame_width : s->width;
  out_port.format.video.nFrameHeight = camera_info->downscale ? camera_info->frame_height : s->height;
  out_port.format.video.xFramerate = 0;
  out_port.format.video.nBitrate = camera_info->bitrate;
  if (camera_info->is_h265) {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingHEVC;
  } else {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingAVC;
  }
  out_port.format.video.eColorFormat = OMX_COLOR_FormatUnused;

  OERR(OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &out_port));
  OERR(OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &out_port));
  s->num_out_bufs = out_port.nBufferCountActual;

  OMX_VIDEO_PARAM_BITRATETYPE bitrate_type = {0};
  bitrate_type.nSize = sizeof(bitrate_type);
  bitrate_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  OERR(OMX_GetParameter(s->handle, OMX_IndexParamVideoBitrate, (OMX_PTR) &bitrate_type));

  bitrate_type.eControlRate = OMX_Video_ControlRateVariable;
  bitrate_type.nTargetBitrate = camera_info->bitrate;

  OERR(OMX_SetParameter(s->handle, OMX_IndexParamVideoBitrate, (OMX_PTR) &bitrate_type));

  if (camera_info->is_h265) {
    #ifndef QCOM2
      // setup HEVC
      OMX_VIDEO_PARAM_HEVCTYPE hecv_type = {0};
      hecv_type.nSize = sizeof(hecv_type);
      hecv_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
      OERR(OMX_GetParameter(s->handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc, (OMX_PTR) &hecv_type));

      hecv_type.eProfile = OMX_VIDEO_HEVCProfileMain;
      hecv_type.eLevel = OMX_VIDEO_HEVCHighTierLevel5;

      OERR(OMX_SetParameter(s->handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc, (OMX_PTR) &hecv_type));
    #endif
  } else {
    // setup h264
    OMX_VIDEO_PARAM_AVCTYPE avc = { 0 };
    avc.nSize = sizeof(avc);
    avc.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
    OERR(OMX_GetParameter(s->handle, OMX_IndexParamVideoAvc, &avc));

    avc.nBFrames = 0;
    avc.nPFrames = 15;

    avc.eProfile = OMX_VIDEO_AVCProfileBaseline;
    avc.eLevel = OMX_VIDEO_AVCLevel31;

    avc.nAllowedPictureTypes |= OMX_VIDEO_PictureTypeB;
    avc.eLoopFilterMode = OMX_VIDEO_AVCLoopFilterEnable;

    OERR(OMX_SetParameter(s->handle, OMX_IndexParamVideoAvc, &avc));
  }

  OERR(OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL));

  s->in_buf_headers = calloc(s->num_in_bufs, sizeof(OMX_BUFFERHEADERTYPE*));
  for (int i=0; i<s->num_in_bufs; i++) {
    OERR(OMX_AllocateBuffer(s->handle, &s->in_buf_headers[i], PORT_INDEX_IN, s, in_port.nBufferSize));
  }

  s->out_buf_headers = calloc(s->num_out_bufs, sizeof(OMX_BUFFERHEADERTYPE*));
  for (int i=0; i<s->num_out_bufs; i++) {
    OERR(OMX_AllocateBuffer(s->handle, &s->out_buf_headers[i], PORT_INDEX_OUT, s, out_port.nBufferSize));
  }

  wait_for_state(s, OMX_StateIdle);
  OERR(OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateExecuting, NULL));
  wait_for_state(s, OMX_StateExecuting);

  // give omx all the output buffers
  for (int i = 0; i < s->num_out_bufs; i++) {
    OERR(OMX_FillThisBuffer(s->handle, s->out_buf_headers[i]));
  }

  // fill the input free queue
  for (int i = 0; i < s->num_in_bufs; i++) {
    queue_push(&s->free_in, (void*)s->in_buf_headers[i]);
  }
}

static void handle_out_buf(EncoderState *s, OMX_BUFFERHEADERTYPE *out_buf) {
  int err;
  uint8_t *buf_data = out_buf->pBuffer + out_buf->nOffset;

  if (out_buf->nFlags & OMX_BUFFERFLAG_CODECCONFIG) {
    if (s->codec_config_len < out_buf->nFilledLen) {
      s->codec_config = realloc(s->codec_config, out_buf->nFilledLen);
    }
    s->codec_config_len = out_buf->nFilledLen;
    memcpy(s->codec_config, buf_data, out_buf->nFilledLen);
  }

  if (s->of) {
    fwrite(buf_data, out_buf->nFilledLen, 1, s->of);
  }

  if (s->remuxing) {
    if (!s->wrote_codec_config && s->codec_config_len > 0) {
      if (s->codec_ctx->extradata_size < s->codec_config_len) {
        s->codec_ctx->extradata = realloc(s->codec_ctx->extradata, s->codec_config_len + AV_INPUT_BUFFER_PADDING_SIZE);
      }
      s->codec_ctx->extradata_size = s->codec_config_len;
      memcpy(s->codec_ctx->extradata, s->codec_config, s->codec_config_len);
      memset(s->codec_ctx->extradata + s->codec_ctx->extradata_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

      err = avcodec_parameters_from_context(s->out_stream->codecpar, s->codec_ctx);
      assert(err >= 0);
      err = avformat_write_header(s->ofmt_ctx, NULL);
      assert(err >= 0);

      s->wrote_codec_config = true;
    }

    if (out_buf->nTimeStamp > 0) {
      // input timestamps are in microseconds
      AVRational in_timebase = {1, 1000000};

      AVPacket pkt;
      av_init_packet(&pkt);
      pkt.data = buf_data;
      pkt.size = out_buf->nFilledLen;
      pkt.pts = pkt.dts = av_rescale_q_rnd(out_buf->nTimeStamp, in_timebase, s->ofmt_ctx->streams[0]->time_base, AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
      pkt.duration = av_rescale_q(50*1000, in_timebase, s->ofmt_ctx->streams[0]->time_base);

      if (out_buf->nFlags & OMX_BUFFERFLAG_SYNCFRAME) {
        pkt.flags |= AV_PKT_FLAG_KEY;
      }

      err = av_write_frame(s->ofmt_ctx, &pkt);
      if (err < 0) { LOGW("ts encoder write issue"); }

      av_free_packet(&pkt);
    }
  }

  // give omx back the buffer
  OERR(OMX_FillThisBuffer(s->handle, out_buf));
}

int encoder_encode_frame(EncoderState *s, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, VIPCBufExtra *extra) {
  int err;
  pthread_mutex_lock(&s->lock);

  if (!s->open) {
    pthread_mutex_unlock(&s->lock);
    return -1;
  }

  // this sometimes freezes... put it outside the encoder lock so we can still trigger rotates...
  // THIS IS A REALLY BAD IDEA, but apparently the race has to happen 30 times to trigger this
  //pthread_mutex_unlock(&s->lock);
  OMX_BUFFERHEADERTYPE* in_buf = queue_pop(&s->free_in);
  //pthread_mutex_lock(&s->lock);

  int ret = s->counter;

  uint8_t *in_buf_ptr = in_buf->pBuffer;

  uint8_t *in_y_ptr = in_buf_ptr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, s->width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, s->width);
  uint8_t *in_uv_ptr = in_buf_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, s->height));

  err = I420ToNV12(y_ptr, s->width,
                   u_ptr, s->width/2,
                   v_ptr, s->width/2,
                   in_y_ptr, in_y_stride,
                   in_uv_ptr, in_uv_stride,
                   s->width, s->height);
  assert(err == 0);

  in_buf->nFilledLen = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, s->width, s->height);
  in_buf->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;
  in_buf->nOffset = 0;
  in_buf->nTimeStamp = extra->timestamp_eof/1000LL;  // OMX_TICKS, in microseconds

  OERR(OMX_EmptyThisBuffer(s->handle, in_buf));

  // pump output
  while (true) {
    OMX_BUFFERHEADERTYPE *out_buf = queue_try_pop(&s->done_out);
    if (!out_buf) {
      break;
    }
    handle_out_buf(s, out_buf);
  }

  s->dirty = true;

  s->counter++;

  pthread_mutex_unlock(&s->lock);
  return ret;
}

void encoder_open(EncoderState *s, const char* path) {
  int err;
  char vid_path[4096];
  pthread_mutex_lock(&s->lock);

  snprintf(vid_path, sizeof(vid_path), "%s/%s", path, s->camera_info.filename);
  LOGD("encoder_open %s remuxing:%d", vid_path, s->remuxing);

  if (s->remuxing) {
    avformat_alloc_output_context2(&s->ofmt_ctx, NULL, NULL, vid_path);
    assert(s->ofmt_ctx);

#ifdef QCOM2
    s->ofmt_ctx->oformat->flags = AVFMT_TS_NONSTRICT;
#endif
    s->out_stream = avformat_new_stream(s->ofmt_ctx, NULL);
    assert(s->out_stream);

    // set codec correctly
    av_register_all();

    AVCodec *codec = NULL;
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    assert(codec);

    s->codec_ctx = avcodec_alloc_context3(codec);
    assert(s->codec_ctx);
    s->codec_ctx->width = s->camera_info.downscale ? s->camera_info.frame_width : s->width;
    s->codec_ctx->height = s->camera_info.downscale ? s->camera_info.frame_height : s->height;
    s->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    s->codec_ctx->time_base = (AVRational){ 1, s->camera_info.fps };

    err = avio_open(&s->ofmt_ctx->pb, vid_path, AVIO_FLAG_WRITE);
    assert(err >= 0);

    s->wrote_codec_config = false;
  } else {
    s->of = fopen(vid_path, "wb");
    assert(s->of);
    if (s->codec_config_len > 0) {
      fwrite(s->codec_config, s->codec_config_len, 1, s->of);
    }
  }

  // create camera lock file
  snprintf(s->lock_path, sizeof(s->lock_path), "%s/%s.lock", path, s->camera_info.filename);
  int lock_fd = open(s->lock_path, O_RDWR | O_CREAT, 0777);
  assert(lock_fd >= 0);
  close(lock_fd);

  s->open = true;
  s->counter = 0;

  pthread_mutex_unlock(&s->lock);
}

void encoder_close(EncoderState *s) {
  pthread_mutex_lock(&s->lock);

  if (s->open) {
    if (s->dirty) {
      // drain output only if there could be frames in the encoder

      OMX_BUFFERHEADERTYPE* in_buf = queue_pop(&s->free_in);
      in_buf->nFilledLen = 0;
      in_buf->nOffset = 0;
      in_buf->nFlags = OMX_BUFFERFLAG_EOS;
      in_buf->nTimeStamp = 0;

      OERR(OMX_EmptyThisBuffer(s->handle, in_buf));

      while (true) {
        OMX_BUFFERHEADERTYPE *out_buf = queue_pop(&s->done_out);

        handle_out_buf(s, out_buf);

        if (out_buf->nFlags & OMX_BUFFERFLAG_EOS) {
          break;
        }
      }
      s->dirty = false;
    }

    if (s->remuxing) {
      av_write_trailer(s->ofmt_ctx);
      avcodec_free_context(&s->codec_ctx);
      avio_closep(&s->ofmt_ctx->pb);
      avformat_free_context(s->ofmt_ctx);
    } else {
      fclose(s->of);
    }
    unlink(s->lock_path);
  }
  s->open = false;

  pthread_mutex_unlock(&s->lock);
}

void encoder_rotate(EncoderState *s, const char* new_path) {
  encoder_close(s);
  encoder_open(s, new_path);
}

void encoder_destroy(EncoderState *s) {
  assert(!s->open);

  OERR(OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL));

  wait_for_state(s, OMX_StateIdle);

  OERR(OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateLoaded, NULL));

  for (int i=0; i<s->num_in_bufs; i++) {
    OERR(OMX_FreeBuffer(s->handle, PORT_INDEX_IN, s->in_buf_headers[i]));
  }
  free(s->in_buf_headers);

  for (int i=0; i<s->num_out_bufs; i++) {
    OERR(OMX_FreeBuffer(s->handle, PORT_INDEX_OUT, s->out_buf_headers[i]));
  }
  free(s->out_buf_headers);

  wait_for_state(s, OMX_StateLoaded);

  OERR(OMX_FreeHandle(s->handle));
}
