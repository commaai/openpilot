#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <pthread.h>

#include <OMX_Component.h>
#include <OMX_IndexExt.h>
#include <OMX_VideoExt.h>
#include <OMX_QCOMExtns.h>

#include <libyuv.h>
#include <msm_media_info.h>

#include "common/mutex.h"
#include "common/swaglog.h"

#include "encoder.h"


// ***** OMX callback functions *****

static void wait_for_state(EncoderState *s, OMX_STATETYPE state) {
  pthread_mutex_lock(&s->state_lock);
  while (s->state != state) {
    pthread_cond_wait(&s->state_cv, &s->state_lock);
  }
  pthread_mutex_unlock(&s->state_lock);
}

static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                   OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data) {
  EncoderState *s = (EncoderState*)app_data;

  switch (event) {
  case OMX_EventCmdComplete:
    assert(data1 == OMX_CommandStateSet);
    LOG("set state event 0x%x", data2);
    pthread_mutex_lock(&s->state_lock);
    s->state = (OMX_STATETYPE)data2;
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
  // printf("empty_buffer_done\n");
  EncoderState *s = (EncoderState*)app_data;
  queue_push(&s->free_in, (void*)buffer);
  return OMX_ErrorNone;
}

static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                      OMX_BUFFERHEADERTYPE *buffer) {
  // printf("fill_buffer_done\n");
  EncoderState *s = (EncoderState*)app_data;
  queue_push(&s->done_out, (void*)buffer);
  return OMX_ErrorNone;
}

static OMX_CALLBACKTYPE omx_callbacks = {
  .EventHandler = event_handler,
  .EmptyBufferDone = empty_buffer_done,
  .FillBufferDone = fill_buffer_done,
};

#define PORT_INDEX_IN 0
#define PORT_INDEX_OUT 1

static const char* omx_color_fomat_name(uint32_t format) __attribute__((unused));
static const char* omx_color_fomat_name(uint32_t format) {
  switch (format) {
  case OMX_COLOR_FormatUnused: return "OMX_COLOR_FormatUnused";
  case OMX_COLOR_FormatMonochrome: return "OMX_COLOR_FormatMonochrome";
  case OMX_COLOR_Format8bitRGB332: return "OMX_COLOR_Format8bitRGB332";
  case OMX_COLOR_Format12bitRGB444: return "OMX_COLOR_Format12bitRGB444";
  case OMX_COLOR_Format16bitARGB4444: return "OMX_COLOR_Format16bitARGB4444";
  case OMX_COLOR_Format16bitARGB1555: return "OMX_COLOR_Format16bitARGB1555";
  case OMX_COLOR_Format16bitRGB565: return "OMX_COLOR_Format16bitRGB565";
  case OMX_COLOR_Format16bitBGR565: return "OMX_COLOR_Format16bitBGR565";
  case OMX_COLOR_Format18bitRGB666: return "OMX_COLOR_Format18bitRGB666";
  case OMX_COLOR_Format18bitARGB1665: return "OMX_COLOR_Format18bitARGB1665";
  case OMX_COLOR_Format19bitARGB1666: return "OMX_COLOR_Format19bitARGB1666";
  case OMX_COLOR_Format24bitRGB888: return "OMX_COLOR_Format24bitRGB888";
  case OMX_COLOR_Format24bitBGR888: return "OMX_COLOR_Format24bitBGR888";
  case OMX_COLOR_Format24bitARGB1887: return "OMX_COLOR_Format24bitARGB1887";
  case OMX_COLOR_Format25bitARGB1888: return "OMX_COLOR_Format25bitARGB1888";
  case OMX_COLOR_Format32bitBGRA8888: return "OMX_COLOR_Format32bitBGRA8888";
  case OMX_COLOR_Format32bitARGB8888: return "OMX_COLOR_Format32bitARGB8888";
  case OMX_COLOR_FormatYUV411Planar: return "OMX_COLOR_FormatYUV411Planar";
  case OMX_COLOR_FormatYUV411PackedPlanar: return "OMX_COLOR_FormatYUV411PackedPlanar";
  case OMX_COLOR_FormatYUV420Planar: return "OMX_COLOR_FormatYUV420Planar";
  case OMX_COLOR_FormatYUV420PackedPlanar: return "OMX_COLOR_FormatYUV420PackedPlanar";
  case OMX_COLOR_FormatYUV420SemiPlanar: return "OMX_COLOR_FormatYUV420SemiPlanar";
  case OMX_COLOR_FormatYUV422Planar: return "OMX_COLOR_FormatYUV422Planar";
  case OMX_COLOR_FormatYUV422PackedPlanar: return "OMX_COLOR_FormatYUV422PackedPlanar";
  case OMX_COLOR_FormatYUV422SemiPlanar: return "OMX_COLOR_FormatYUV422SemiPlanar";
  case OMX_COLOR_FormatYCbYCr: return "OMX_COLOR_FormatYCbYCr";
  case OMX_COLOR_FormatYCrYCb: return "OMX_COLOR_FormatYCrYCb";
  case OMX_COLOR_FormatCbYCrY: return "OMX_COLOR_FormatCbYCrY";
  case OMX_COLOR_FormatCrYCbY: return "OMX_COLOR_FormatCrYCbY";
  case OMX_COLOR_FormatYUV444Interleaved: return "OMX_COLOR_FormatYUV444Interleaved";
  case OMX_COLOR_FormatRawBayer8bit: return "OMX_COLOR_FormatRawBayer8bit";
  case OMX_COLOR_FormatRawBayer10bit: return "OMX_COLOR_FormatRawBayer10bit";
  case OMX_COLOR_FormatRawBayer8bitcompressed: return "OMX_COLOR_FormatRawBayer8bitcompressed";
  case OMX_COLOR_FormatL2: return "OMX_COLOR_FormatL2";
  case OMX_COLOR_FormatL4: return "OMX_COLOR_FormatL4";
  case OMX_COLOR_FormatL8: return "OMX_COLOR_FormatL8";
  case OMX_COLOR_FormatL16: return "OMX_COLOR_FormatL16";
  case OMX_COLOR_FormatL24: return "OMX_COLOR_FormatL24";
  case OMX_COLOR_FormatL32: return "OMX_COLOR_FormatL32";
  case OMX_COLOR_FormatYUV420PackedSemiPlanar: return "OMX_COLOR_FormatYUV420PackedSemiPlanar";
  case OMX_COLOR_FormatYUV422PackedSemiPlanar: return "OMX_COLOR_FormatYUV422PackedSemiPlanar";
  case OMX_COLOR_Format18BitBGR666: return "OMX_COLOR_Format18BitBGR666";
  case OMX_COLOR_Format24BitARGB6666: return "OMX_COLOR_Format24BitARGB6666";
  case OMX_COLOR_Format24BitABGR6666: return "OMX_COLOR_Format24BitABGR6666";

  case OMX_COLOR_FormatAndroidOpaque: return "OMX_COLOR_FormatAndroidOpaque";
  case OMX_TI_COLOR_FormatYUV420PackedSemiPlanar: return "OMX_TI_COLOR_FormatYUV420PackedSemiPlanar";
  case OMX_QCOM_COLOR_FormatYVU420SemiPlanar: return "OMX_QCOM_COLOR_FormatYVU420SemiPlanar";
  case OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka: return "OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka";
  case OMX_SEC_COLOR_FormatNV12Tiled: return "OMX_SEC_COLOR_FormatNV12Tiled";
  case OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar32m: return "OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar32m";

  // case QOMX_COLOR_FormatYVU420SemiPlanar: return "QOMX_COLOR_FormatYVU420SemiPlanar";
  case QOMX_COLOR_FormatYVU420PackedSemiPlanar32m4ka: return "QOMX_COLOR_FormatYVU420PackedSemiPlanar32m4ka";
  case QOMX_COLOR_FormatYUV420PackedSemiPlanar16m2ka: return "QOMX_COLOR_FormatYUV420PackedSemiPlanar16m2ka";
  // case QOMX_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka: return "QOMX_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka";
  // case QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m: return "QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m";
  case QOMX_COLOR_FORMATYUV420PackedSemiPlanar32mMultiView: return "QOMX_COLOR_FORMATYUV420PackedSemiPlanar32mMultiView";
  case QOMX_COLOR_FORMATYUV420PackedSemiPlanar32mCompressed: return "QOMX_COLOR_FORMATYUV420PackedSemiPlanar32mCompressed";
  case QOMX_COLOR_Format32bitRGBA8888: return "QOMX_COLOR_Format32bitRGBA8888";
  case QOMX_COLOR_Format32bitRGBA8888Compressed: return "QOMX_COLOR_Format32bitRGBA8888Compressed";

  default:
    return "unkn";
  }
}


// ***** encoder functions *****

void encoder_init(EncoderState *s, const char* filename, int width, int height, int fps, int bitrate, bool h265, bool downscale) {
  int err;

  memset(s, 0, sizeof(*s));
  s->filename = filename;
  s->width = width;
  s->height = height;
  s->fps = fps;
  mutex_init_reentrant(&s->lock);

  if (!h265) {
    s->remuxing = true;
  }

  if (downscale) {
    s->downscale = true;
    s->y_ptr2 = (uint8_t *)malloc(s->width*s->height);
    s->u_ptr2 = (uint8_t *)malloc(s->width*s->height/4);
    s->v_ptr2 = (uint8_t *)malloc(s->width*s->height/4);
  }

  s->segment = -1;

  s->state = OMX_StateLoaded;

  s->codec_config = NULL;

  queue_init(&s->free_in);
  queue_init(&s->done_out);

  pthread_mutex_init(&s->state_lock, NULL);
  pthread_cond_init(&s->state_cv, NULL);

  if (h265) {
    err = OMX_GetHandle(&s->handle, (OMX_STRING)"OMX.qcom.video.encoder.hevc",
                        s, &omx_callbacks);
  } else {
    err = OMX_GetHandle(&s->handle, (OMX_STRING)"OMX.qcom.video.encoder.avc",
                        s, &omx_callbacks);
  }
  if (err != OMX_ErrorNone) {
    LOGE("error getting codec: %x", err);
  }
  assert(err == OMX_ErrorNone);
  // printf("handle: %p\n", s->handle);

  // setup input port

  OMX_PARAM_PORTDEFINITIONTYPE in_port = {0};
  in_port.nSize = sizeof(in_port);
  in_port.nPortIndex = (OMX_U32) PORT_INDEX_IN;
  err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);

  in_port.format.video.nFrameWidth = s->width;
  in_port.format.video.nFrameHeight = s->height;
  in_port.format.video.nStride = VENUS_Y_STRIDE(COLOR_FMT_NV12, s->width);
  in_port.format.video.nSliceHeight = s->height;
  // in_port.nBufferSize = (s->width * s->height * 3) / 2;
  in_port.nBufferSize = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, s->width, s->height);
  in_port.format.video.xFramerate = (s->fps * 65536);
  in_port.format.video.eCompressionFormat = OMX_VIDEO_CodingUnused;
  // in_port.format.video.eColorFormat = OMX_COLOR_FormatYUV420SemiPlanar;
  in_port.format.video.eColorFormat = (OMX_COLOR_FORMATTYPE)QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m;

  err = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);


  err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);
  s->num_in_bufs = in_port.nBufferCountActual;

  // setup output port

  OMX_PARAM_PORTDEFINITIONTYPE out_port = {0};
  out_port.nSize = sizeof(out_port);
  out_port.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR)&out_port);
  assert(err == OMX_ErrorNone);
  out_port.format.video.nFrameWidth = s->width;
  out_port.format.video.nFrameHeight = s->height;
  out_port.format.video.xFramerate = 0;
  out_port.format.video.nBitrate = bitrate;
  if (h265) {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingHEVC;
  } else {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingAVC;
  }
  out_port.format.video.eColorFormat = OMX_COLOR_FormatUnused;

  err = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &out_port);
  assert(err == OMX_ErrorNone);

  err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &out_port);
  assert(err == OMX_ErrorNone);
  s->num_out_bufs = out_port.nBufferCountActual;

  OMX_VIDEO_PARAM_BITRATETYPE bitrate_type = {0};
  bitrate_type.nSize = sizeof(bitrate_type);
  bitrate_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(s->handle, OMX_IndexParamVideoBitrate,
                         (OMX_PTR) &bitrate_type);
  assert(err == OMX_ErrorNone);

  bitrate_type.eControlRate = OMX_Video_ControlRateVariable;
  bitrate_type.nTargetBitrate = bitrate;

  err = OMX_SetParameter(s->handle, OMX_IndexParamVideoBitrate,
                         (OMX_PTR) &bitrate_type);
  assert(err == OMX_ErrorNone);

  if (h265) {
    // setup HEVC
  #ifndef QCOM2
    OMX_VIDEO_PARAM_HEVCTYPE hevc_type = {0};
    OMX_INDEXTYPE index_type = (OMX_INDEXTYPE) OMX_IndexParamVideoHevc;
  #else
    OMX_VIDEO_PARAM_PROFILELEVELTYPE hevc_type = {0};
    OMX_INDEXTYPE index_type = OMX_IndexParamVideoProfileLevelCurrent;
  #endif
    hevc_type.nSize = sizeof(hevc_type);
    hevc_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
    err = OMX_GetParameter(s->handle, index_type,
                           (OMX_PTR) &hevc_type);
    assert(err == OMX_ErrorNone);

    hevc_type.eProfile = OMX_VIDEO_HEVCProfileMain;
    hevc_type.eLevel = OMX_VIDEO_HEVCHighTierLevel5;

    err = OMX_SetParameter(s->handle, index_type,
                           (OMX_PTR) &hevc_type);
    assert(err == OMX_ErrorNone);
  } else {
    // setup h264
    OMX_VIDEO_PARAM_AVCTYPE avc = { 0 };
    avc.nSize = sizeof(avc);
    avc.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
    err = OMX_GetParameter(s->handle, OMX_IndexParamVideoAvc, &avc);
    assert(err == OMX_ErrorNone);

    avc.nBFrames = 0;
    avc.nPFrames = 15;

    avc.eProfile = OMX_VIDEO_AVCProfileBaseline;
    avc.eLevel = OMX_VIDEO_AVCLevel31;

    avc.nAllowedPictureTypes |= OMX_VIDEO_PictureTypeB;
    avc.eLoopFilterMode = OMX_VIDEO_AVCLoopFilterEnable;

    err = OMX_SetParameter(s->handle, OMX_IndexParamVideoAvc, &avc);
    assert(err == OMX_ErrorNone);
  }


  // for (int i = 0; ; i++) {
  //   OMX_VIDEO_PARAM_PORTFORMATTYPE video_port_format = {0};
  //   video_port_format.nSize = sizeof(video_port_format);
  //   video_port_format.nIndex = i;
  //   video_port_format.nPortIndex = PORT_INDEX_IN;
  //   if (OMX_GetParameter(s->handle, OMX_IndexParamVideoPortFormat, &video_port_format) != OMX_ErrorNone)
  //       break;
  //   printf("in %d: compression 0x%x format 0x%x %s\n", i,
  //          video_port_format.eCompressionFormat, video_port_format.eColorFormat,
  //          omx_color_fomat_name(video_port_format.eColorFormat));
  // }

  // for (int i=0; i<32; i++) {
  //   OMX_VIDEO_PARAM_PROFILELEVELTYPE params = {0};
  //   params.nSize = sizeof(params);
  //   params.nPortIndex = PORT_INDEX_OUT;
  //   params.nProfileIndex = i;
  //   if (OMX_GetParameter(s->handle, OMX_IndexParamVideoProfileLevelQuerySupported, &params) != OMX_ErrorNone)
  //       break;
  //   printf("profile %d level 0x%x\n", params.eProfile, params.eLevel);
  // }

  err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
  assert(err == OMX_ErrorNone);

  s->in_buf_headers = (OMX_BUFFERHEADERTYPE **)calloc(s->num_in_bufs, sizeof(OMX_BUFFERHEADERTYPE*));
  for (int i=0; i<s->num_in_bufs; i++) {
    err = OMX_AllocateBuffer(s->handle, &s->in_buf_headers[i], PORT_INDEX_IN, s,
                             in_port.nBufferSize);
    assert(err == OMX_ErrorNone);
  }

  s->out_buf_headers = (OMX_BUFFERHEADERTYPE **)calloc(s->num_out_bufs, sizeof(OMX_BUFFERHEADERTYPE*));
  for (int i=0; i<s->num_out_bufs; i++) {
    err = OMX_AllocateBuffer(s->handle, &s->out_buf_headers[i], PORT_INDEX_OUT, s,
                             out_port.nBufferSize);
    assert(err == OMX_ErrorNone);
  }

  wait_for_state(s, OMX_StateIdle);

  err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateExecuting, NULL);
  assert(err == OMX_ErrorNone);

  wait_for_state(s, OMX_StateExecuting);

  // give omx all the output buffers
  for (int i = 0; i < s->num_out_bufs; i++) {
    // printf("fill %p\n", s->out_buf_headers[i]);
    err = OMX_FillThisBuffer(s->handle, s->out_buf_headers[i]);
    assert(err == OMX_ErrorNone);
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
      s->codec_config = (uint8_t *)realloc(s->codec_config, out_buf->nFilledLen);
    }
    s->codec_config_len = out_buf->nFilledLen;
    memcpy(s->codec_config, buf_data, out_buf->nFilledLen);
#ifdef QCOM2
    out_buf->nTimeStamp = 0;
#endif
  }

  if (s->of) {
    //printf("write %d flags 0x%x\n", out_buf->nFilledLen, out_buf->nFlags);
    fwrite(buf_data, out_buf->nFilledLen, 1, s->of);
  }

  if (s->remuxing) {
    if (!s->wrote_codec_config && s->codec_config_len > 0) {
      if (s->codec_ctx->extradata_size < s->codec_config_len) {
        s->codec_ctx->extradata = (uint8_t *)realloc(s->codec_ctx->extradata, s->codec_config_len + AV_INPUT_BUFFER_PADDING_SIZE);
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

      enum AVRounding rnd = static_cast<enum AVRounding>(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
      pkt.pts = pkt.dts = av_rescale_q_rnd(out_buf->nTimeStamp, in_timebase, s->ofmt_ctx->streams[0]->time_base, rnd);
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
#ifdef QCOM2
  if (out_buf->nFlags & OMX_BUFFERFLAG_EOS) {
    out_buf->nTimeStamp = 0;
  }
#endif
  err = OMX_FillThisBuffer(s->handle, out_buf);
  assert(err == OMX_ErrorNone);
}

int encoder_encode_frame(EncoderState *s,
                         const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                         int in_width, int in_height,
                         int *frame_segment, VisionIpcBufExtra *extra) {
  int err;
  pthread_mutex_lock(&s->lock);

  if (!s->open) {
    pthread_mutex_unlock(&s->lock);
    return -1;
  }

  // this sometimes freezes... put it outside the encoder lock so we can still trigger rotates...
  // THIS IS A REALLY BAD IDEA, but apparently the race has to happen 30 times to trigger this
  //pthread_mutex_unlock(&s->lock);
  OMX_BUFFERHEADERTYPE* in_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&s->free_in);
  //pthread_mutex_lock(&s->lock);

  // if (s->rotating) {
  //   encoder_close(s);
  //   encoder_open(s, s->next_path);
  //   s->segment = s->next_segment;
  //   s->rotating = false;
  // }
  int ret = s->counter;

  uint8_t *in_buf_ptr = in_buf->pBuffer;
  // printf("in_buf ptr %p\n", in_buf_ptr);

  uint8_t *in_y_ptr = in_buf_ptr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, s->width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, s->width);
  // uint8_t *in_uv_ptr = in_buf_ptr + (s->width * s->height);
  uint8_t *in_uv_ptr = in_buf_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, s->height));

  if (s->downscale) {
    I420Scale(y_ptr, in_width,
              u_ptr, in_width/2,
              v_ptr, in_width/2,
              in_width, in_height,
              s->y_ptr2, s->width,
              s->u_ptr2, s->width/2,
              s->v_ptr2, s->width/2,
              s->width, s->height,
              libyuv::kFilterNone);
    y_ptr = s->y_ptr2;
    u_ptr = s->u_ptr2;
    v_ptr = s->v_ptr2;
  }
  err = libyuv::I420ToNV12(y_ptr, s->width,
                   u_ptr, s->width/2,
                   v_ptr, s->width/2,
                   in_y_ptr, in_y_stride,
                   in_uv_ptr, in_uv_stride,
                   s->width, s->height);
  assert(err == 0);

  // in_buf->nFilledLen = (s->width*s->height) + (s->width*s->height/2);
  in_buf->nFilledLen = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, s->width, s->height);
  in_buf->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;
  in_buf->nOffset = 0;
  in_buf->nTimeStamp = extra->timestamp_eof/1000LL;  // OMX_TICKS, in microseconds
  s->last_t = in_buf->nTimeStamp;

  err = OMX_EmptyThisBuffer(s->handle, in_buf);
  assert(err == OMX_ErrorNone);

  // pump output
  while (true) {
    OMX_BUFFERHEADERTYPE *out_buf = (OMX_BUFFERHEADERTYPE *)queue_try_pop(&s->done_out);
    if (!out_buf) {
      break;
    }
    handle_out_buf(s, out_buf);
  }

  s->dirty = true;

  s->counter++;

  if (frame_segment) {
    *frame_segment = s->segment;
  }

  pthread_mutex_unlock(&s->lock);
  return ret;
}

void encoder_open(EncoderState *s, const char* path, int segment) {
  int err;

  pthread_mutex_lock(&s->lock);

  s->segment = segment;
  snprintf(s->vid_path, sizeof(s->vid_path), "%s/%s", path, s->filename);
  LOGD("encoder_open %s remuxing:%d", s->vid_path, s->remuxing);

  if (s->remuxing) {
    avformat_alloc_output_context2(&s->ofmt_ctx, NULL, NULL, s->vid_path);
    assert(s->ofmt_ctx);

    s->out_stream = avformat_new_stream(s->ofmt_ctx, NULL);
    assert(s->out_stream);

    // set codec correctly
    av_register_all();

    AVCodec *codec = NULL;
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    assert(codec);

    s->codec_ctx = avcodec_alloc_context3(codec);
    assert(s->codec_ctx);
    s->codec_ctx->width = s->width;
    s->codec_ctx->height = s->height;
    s->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    s->codec_ctx->time_base = (AVRational){ 1, s->fps };

    err = avio_open(&s->ofmt_ctx->pb, s->vid_path, AVIO_FLAG_WRITE);
    assert(err >= 0);

    s->wrote_codec_config = false;
  } else {
    s->of = fopen(s->vid_path, "wb");
    assert(s->of);
#ifndef QCOM2
    if (s->codec_config_len > 0) {
      fwrite(s->codec_config, s->codec_config_len, 1, s->of);
    }
#endif
  }

  // create camera lock file
  snprintf(s->lock_path, sizeof(s->lock_path), "%s/%s.lock", path, s->filename);
  int lock_fd = open(s->lock_path, O_RDWR | O_CREAT, 0777);
  assert(lock_fd >= 0);
  close(lock_fd);

  s->open = true;
  s->counter = 0;

  pthread_mutex_unlock(&s->lock);
}

void encoder_close(EncoderState *s) {
  int err;

  pthread_mutex_lock(&s->lock);

  if (s->open) {
    if (s->dirty) {
      // drain output only if there could be frames in the encoder

      OMX_BUFFERHEADERTYPE* in_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&s->free_in);
      in_buf->nFilledLen = 0;
      in_buf->nOffset = 0;
      in_buf->nFlags = OMX_BUFFERFLAG_EOS;
      in_buf->nTimeStamp = s->last_t + 1000000LL/s->fps;

      err = OMX_EmptyThisBuffer(s->handle, in_buf);
      assert(err == OMX_ErrorNone);

      while (true) {
        OMX_BUFFERHEADERTYPE *out_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&s->done_out);

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

void encoder_destroy(EncoderState *s) {
  int err;

  assert(!s->open);

  err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
  assert(err == OMX_ErrorNone);

  wait_for_state(s, OMX_StateIdle);

  err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateLoaded, NULL);
  assert(err == OMX_ErrorNone);

  for (int i=0; i<s->num_in_bufs; i++) {
    err = OMX_FreeBuffer(s->handle, PORT_INDEX_IN, s->in_buf_headers[i]);
    assert(err == OMX_ErrorNone);
  }
  free(s->in_buf_headers);

  for (int i=0; i<s->num_out_bufs; i++) {
    err = OMX_FreeBuffer(s->handle, PORT_INDEX_OUT, s->out_buf_headers[i]);
    assert(err == OMX_ErrorNone);
  }
  free(s->out_buf_headers);

  wait_for_state(s, OMX_StateLoaded);

  err = OMX_FreeHandle(s->handle);
  assert(err == OMX_ErrorNone);

  while (queue_try_pop(&s->free_in)); 
  while (queue_try_pop(&s->done_out)); 

  if (s->codec_config) {
    free(s->codec_config);
  }

  if (s->downscale) {
    free(s->y_ptr2);
    free(s->u_ptr2);
    free(s->v_ptr2);
  }
}
