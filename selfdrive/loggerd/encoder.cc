#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>

#include <czmq.h>

#include <pthread.h>

#include <OMX_Component.h>
#include <OMX_IndexExt.h>
#include <OMX_VideoExt.h>
#include <OMX_QCOMExtns.h>

#include <libyuv.h>

#include <msm_media_info.h>

#include "common/mutex.h"
#include "common/swaglog.h"
#include "common/utilpp.h"
#include "encoder.h"

// encoder: lossey codec using hardware hevc
static void wait_for_state(EncoderState *s, OMX_STATETYPE state) {
  pthread_mutex_lock(&s->state_lock);
  while (s->state != state) {
    pthread_cond_wait(&s->state_cv, &s->state_lock);
  }
  pthread_mutex_unlock(&s->state_lock);
}

static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                   OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data) {
  EncoderState *s = (EncoderState *)app_data;

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
  EncoderState *s = (EncoderState *)app_data;

  // printf("empty_buffer_done\n");

  queue_push(&s->free_in, (void*)buffer);

  return OMX_ErrorNone;
}


static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                      OMX_BUFFERHEADERTYPE *buffer) {
  EncoderState *s = (EncoderState *)app_data;

  // printf("fill_buffer_done\n");

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


EncoderState::EncoderState(const LogCameraInfo &info, int width, int height, bool streaming)
    : FrameLogger(info.filename, width, height, info.fps), downscale(info.downscale) {
  int err;
  
  if (!info.is_h265) {
    remuxing = true;
  }

  if (downscale) {
    y_ptr2 = std::make_unique<uint8_t[]>(width*height);
    u_ptr2 = std::make_unique<uint8_t[]>(width*height/4);
    v_ptr2 = std::make_unique<uint8_t[]>(width*height/4);
  }

  state = OMX_StateLoaded;

  queue_init(&free_in);
  queue_init(&done_out);

  pthread_mutex_init(&state_lock, NULL);
  pthread_cond_init(&state_cv, NULL);

  if (info.is_h265) {
    err = OMX_GetHandle(&handle, (OMX_STRING)"OMX.qcom.video.encoder.hevc",
                        this, &omx_callbacks);
  } else {
    err = OMX_GetHandle(&handle, (OMX_STRING)"OMX.qcom.video.encoder.avc",
                        this, &omx_callbacks);
  }
  if (err != OMX_ErrorNone) {
    LOGE("error getting codec: %x", err);
  }
  assert(err == OMX_ErrorNone);
  // printf("handle: %p\n", handle);

  // setup input port

  OMX_PARAM_PORTDEFINITIONTYPE in_port = {0};
  in_port.nSize = sizeof(in_port);
  in_port.nPortIndex = (OMX_U32) PORT_INDEX_IN;
  err = OMX_GetParameter(handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);

  in_port.format.video.nFrameWidth = width;
  in_port.format.video.nFrameHeight = height;
  in_port.format.video.nStride = VENUS_Y_STRIDE(COLOR_FMT_NV12, width);
  in_port.format.video.nSliceHeight = height;
  // in_port.nBufferSize = (width * height * 3) / 2;
  in_port.nBufferSize = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, width, height);
  in_port.format.video.xFramerate = (fps * 65536);
  in_port.format.video.eCompressionFormat = OMX_VIDEO_CodingUnused;
  // in_port.format.video.eColorFormat = OMX_COLOR_FormatYUV420SemiPlanar;
  in_port.format.video.eColorFormat = (OMX_COLOR_FORMATTYPE)QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m;

  err = OMX_SetParameter(handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);


  err = OMX_GetParameter(handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);
  num_in_bufs = in_port.nBufferCountActual;

  // printf("in width: %d, stride: %d\n",
  //   in_port.format.video.nFrameWidth, in_port.format.video.nStride);

  // setup output port

  OMX_PARAM_PORTDEFINITIONTYPE out_port = {0};
  out_port.nSize = sizeof(out_port);
  out_port.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR)&out_port);
  assert(err == OMX_ErrorNone);
  out_port.format.video.nFrameWidth = width;
  out_port.format.video.nFrameHeight = height;
  out_port.format.video.xFramerate = 0;
  out_port.format.video.nBitrate = info.bitrate;
  if (info.is_h265) {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingHEVC;
  } else {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingAVC;
  }
  out_port.format.video.eColorFormat = OMX_COLOR_FormatUnused;

  err = OMX_SetParameter(handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &out_port);
  assert(err == OMX_ErrorNone);

  err = OMX_GetParameter(handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &out_port);
  assert(err == OMX_ErrorNone);
  num_out_bufs = out_port.nBufferCountActual;

  OMX_VIDEO_PARAM_BITRATETYPE bitrate_type = {0};
  bitrate_type.nSize = sizeof(bitrate_type);
  bitrate_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(handle, OMX_IndexParamVideoBitrate,
                         (OMX_PTR) &bitrate_type);
  assert(err == OMX_ErrorNone);

  bitrate_type.eControlRate = OMX_Video_ControlRateVariable;
  bitrate_type.nTargetBitrate = info.bitrate;

  err = OMX_SetParameter(handle, OMX_IndexParamVideoBitrate,
                         (OMX_PTR) &bitrate_type);
  assert(err == OMX_ErrorNone);

  if (info.is_h265) {
    #ifndef QCOM2
      // setup HEVC
      OMX_VIDEO_PARAM_HEVCTYPE hecv_type = {0};
      hecv_type.nSize = sizeof(hecv_type);
      hecv_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
      err = OMX_GetParameter(handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc,
                             (OMX_PTR) &hecv_type);
      assert(err == OMX_ErrorNone);

      hecv_type.eProfile = OMX_VIDEO_HEVCProfileMain;
      hecv_type.eLevel = OMX_VIDEO_HEVCHighTierLevel5;

      err = OMX_SetParameter(handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc,
                             (OMX_PTR) &hecv_type);
      assert(err == OMX_ErrorNone);
    #endif
  } else {
    // setup h264
    OMX_VIDEO_PARAM_AVCTYPE avc = { 0 };
    avc.nSize = sizeof(avc);
    avc.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
    err = OMX_GetParameter(handle, OMX_IndexParamVideoAvc, &avc);
    assert(err == OMX_ErrorNone);

    avc.nBFrames = 0;
    avc.nPFrames = 15;

    avc.eProfile = OMX_VIDEO_AVCProfileBaseline;
    avc.eLevel = OMX_VIDEO_AVCLevel31;

    avc.nAllowedPictureTypes |= OMX_VIDEO_PictureTypeB;
    avc.eLoopFilterMode = OMX_VIDEO_AVCLoopFilterEnable;

    err = OMX_SetParameter(handle, OMX_IndexParamVideoAvc, &avc);
    assert(err == OMX_ErrorNone);
  }

  err = OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
  assert(err == OMX_ErrorNone);

  in_buf_headers = (OMX_BUFFERHEADERTYPE**)calloc(num_in_bufs, sizeof(OMX_BUFFERHEADERTYPE*));
  for (int i=0; i<num_in_bufs; i++) {
    err = OMX_AllocateBuffer(handle, &in_buf_headers[i], PORT_INDEX_IN, this,
                             in_port.nBufferSize);
    assert(err == OMX_ErrorNone);
  }

  out_buf_headers = (OMX_BUFFERHEADERTYPE**)calloc(num_out_bufs, sizeof(OMX_BUFFERHEADERTYPE*));
  for (int i=0; i<num_out_bufs; i++) {
    err = OMX_AllocateBuffer(handle, &out_buf_headers[i], PORT_INDEX_OUT, this,
                             out_port.nBufferSize);
    assert(err == OMX_ErrorNone);
  }

  wait_for_state(this, OMX_StateIdle);

  err = OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateExecuting, NULL);
  assert(err == OMX_ErrorNone);

  wait_for_state(this, OMX_StateExecuting);

  // give omx all the output buffers
  for (int i = 0; i < num_out_bufs; i++) {
    // printf("fill %p\n", out_buf_headers[i]);
    err = OMX_FillThisBuffer(handle, out_buf_headers[i]);
    assert(err == OMX_ErrorNone);
  }

  // fill the input free queue
  for (int i = 0; i < num_in_bufs; i++) {
    queue_push(&free_in, (void*)in_buf_headers[i]);
  }

  if (streaming) {
    zmq_ctx = zmq_ctx_new();
    stream_sock_raw = zmq_socket(zmq_ctx, ZMQ_PUB);
    assert(stream_sock_raw);
    zmq_bind(stream_sock_raw, "tcp://*:9002");
  }
}

EncoderState::~EncoderState() {
  CloseFile();
  Destroy();
}

static void handle_out_buf(EncoderState *s, OMX_BUFFERHEADERTYPE *out_buf) {
  int err;
  uint8_t *buf_data = out_buf->pBuffer + out_buf->nOffset;

  if (out_buf->nFlags & OMX_BUFFERFLAG_CODECCONFIG) {
    if (s->codec_config_len < out_buf->nFilledLen) {
      s->codec_config = (uint8_t*)realloc(s->codec_config, out_buf->nFilledLen);
    }
    s->codec_config_len = out_buf->nFilledLen;
    memcpy(s->codec_config, buf_data, out_buf->nFilledLen);
  }

  if (s->stream_sock_raw) {
    //uint64_t current_time = nanos_since_boot();
    //uint64_t diff = current_time - out_buf->nTimeStamp*1000LL;
    //double msdiff = (double) diff / 1000000.0;
    // printf("encoded latency to tsEof: %f\n", msdiff);
    zmq_send(s->stream_sock_raw, &out_buf->nTimeStamp, sizeof(out_buf->nTimeStamp), ZMQ_SNDMORE);
    zmq_send(s->stream_sock_raw, buf_data, out_buf->nFilledLen, 0);
  }

  if (s->of) {
    //printf("write %d flags 0x%x\n", out_buf->nFilledLen, out_buf->nFlags);
    fwrite(buf_data, out_buf->nFilledLen, 1, s->of);
  }

  if (s->remuxing) {
    if (!s->wrote_codec_config && s->codec_config_len > 0) {
      if (s->codec_ctx->extradata_size < s->codec_config_len) {
        s->codec_ctx->extradata = (uint8_t*)realloc(s->codec_ctx->extradata, s->codec_config_len + AV_INPUT_BUFFER_PADDING_SIZE);
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
      pkt.pts = pkt.dts = av_rescale_q_rnd(out_buf->nTimeStamp, in_timebase, s->ofmt_ctx->streams[0]->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
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
  err = OMX_FillThisBuffer(s->handle, out_buf);
  assert(err == OMX_ErrorNone);
}

bool EncoderState::ProcessFrame(uint64_t cnt, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                                int in_width, int in_height, const VIPCBufExtra &extra) {
  
  OMX_BUFFERHEADERTYPE* in_buf = (OMX_BUFFERHEADERTYPE*)queue_pop(&free_in);
  uint8_t *in_buf_ptr = in_buf->pBuffer;
  // printf("in_buf ptr %p\n", in_buf_ptr);

  uint8_t *in_y_ptr = in_buf_ptr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, width);
  // uint8_t *in_uv_ptr = in_buf_ptr + (width * height);
  uint8_t *in_uv_ptr = in_buf_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, height));

  if (downscale) { // downscale
   libyuv::I420Scale(y_ptr, in_width,
              u_ptr, in_width/2,
              v_ptr, in_width/2,
              in_width, in_height,
              y_ptr2.get(), width,
              u_ptr2.get(), width/2,
              v_ptr2.get(), width/2,
              width, height,
              libyuv::kFilterNone);
    y_ptr = y_ptr2.get();
    u_ptr = u_ptr2.get();
    v_ptr = v_ptr2.get();
  }
  int err = libyuv::I420ToNV12(y_ptr, width,
                   u_ptr, width/2,
                   v_ptr, width/2,
                   in_y_ptr, in_y_stride,
                   in_uv_ptr, in_uv_stride,
                   width, height);
  assert(err == 0);

  // in_buf->nFilledLen = (width*height) + (width*height/2);
  in_buf->nFilledLen = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, width, height);
  in_buf->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;
  in_buf->nOffset = 0;
  in_buf->nTimeStamp = extra.timestamp_eof/1000LL;  // OMX_TICKS, in microseconds

  err = OMX_EmptyThisBuffer(handle, in_buf);
  assert(err == OMX_ErrorNone);

  // pump output
  while (true) {
    OMX_BUFFERHEADERTYPE *out_buf = (OMX_BUFFERHEADERTYPE*)queue_try_pop(&done_out);
    if (!out_buf) {
      break;
    }
    handle_out_buf(this, out_buf);
  }

  dirty = true;

  return true;
}

void EncoderState::Open(const std::string &path) {
  LOGD("encoder_open %s remuxing:%d", path.c_str(), remuxing);

  if (remuxing) {
    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, path.c_str());
    assert(ofmt_ctx);

    out_stream = avformat_new_stream(ofmt_ctx, NULL);
    assert(out_stream);

    // set codec correctly
    av_register_all();

    AVCodec *codec = NULL;
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    assert(codec);

    codec_ctx = avcodec_alloc_context3(codec);
    assert(codec_ctx);
    codec_ctx->width = width;
    codec_ctx->height = height;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->time_base = (AVRational){ 1, fps };

    int err = avio_open(&ofmt_ctx->pb, path.c_str(), AVIO_FLAG_WRITE);
    assert(err >= 0);

    wrote_codec_config = false;
  } else {
    of = fopen(path.c_str(), "wb");
    assert(of);
    if (codec_config_len > 0) {
      fwrite(codec_config, codec_config_len, 1, of);
    }
  }
}

void EncoderState::Close() {
  if (dirty) {
    // drain output only if there could be frames in the encoder

    OMX_BUFFERHEADERTYPE* in_buf = (OMX_BUFFERHEADERTYPE*)queue_pop(&free_in);
    in_buf->nFilledLen = 0;
    in_buf->nOffset = 0;
    in_buf->nFlags = OMX_BUFFERFLAG_EOS;
    in_buf->nTimeStamp = 0;

    int err = OMX_EmptyThisBuffer(handle, in_buf);
    assert(err == OMX_ErrorNone);

    while (true) {
      OMX_BUFFERHEADERTYPE *out_buf = (OMX_BUFFERHEADERTYPE*)queue_pop(&done_out);

      handle_out_buf(this, out_buf);

      if (out_buf->nFlags & OMX_BUFFERFLAG_EOS) {
        break;
      }
    }
    dirty = false;
  }

  if (remuxing) {
    av_write_trailer(ofmt_ctx);
    avcodec_free_context(&codec_ctx);
    avio_closep(&ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
  } else {
    fclose(of);
  }
}

void EncoderState::Destroy() {
  int err = OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
  assert(err == OMX_ErrorNone);

  wait_for_state(this, OMX_StateIdle);

  err = OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateLoaded, NULL);
  assert(err == OMX_ErrorNone);

  for (int i=0; i<num_in_bufs; i++) {
    err = OMX_FreeBuffer(handle, PORT_INDEX_IN, in_buf_headers[i]);
    assert(err == OMX_ErrorNone);
  }
  free(in_buf_headers);

  for (int i=0; i<num_out_bufs; i++) {
    err = OMX_FreeBuffer(handle, PORT_INDEX_OUT, out_buf_headers[i]);
    assert(err == OMX_ErrorNone);
  }
  free(out_buf_headers);

  wait_for_state(this, OMX_StateLoaded);

  err = OMX_FreeHandle(handle);
  assert(err == OMX_ErrorNone);

  if (stream_sock_raw) {
    zmq_close(stream_sock_raw);
    zmq_ctx_term(zmq_ctx);
  }
}

#if 0

// cd one/selfdrive/visiond
// clang
//   -fPIC -pie
//   -std=gnu11 -O2 -g
//   -o encoder
//   -I ~/one/selfdrive
//   -I ~/one/phonelibs/openmax/include
//   -I ~/one/phonelibs/libyuv/include
//   -lOmxVenc -lOmxCore -llog
//   encoder.cc
//   buffering.c
//   -L ~/one/phonelibs/libyuv/lib -l:libyuv.a

int main() {
  int err;

  EncoderState state;
  EncoderState *s = &state;
  memset(s, 0, sizeof(*s));

  int w = 1164;
  int h = 874;

  encoder_init(s, w, h, 20);
  printf("inited\n");

  encoder_open(s, "/sdcard/t1");

  // uint8_t *tmpy = malloc(640*480);
  // uint8_t *tmpu = malloc((640/2)*(480/2));
  // uint8_t *tmpv = malloc((640/2)*(480/2));

  // memset(tmpy, 0, 640*480);
  // // memset(tmpu, 0xff, (640/2)*(480/2));
  // memset(tmpv, 0, (640/2)*(480/2));

// #if 0
  // FILE *infile = fopen("/sdcard/camera_t2.yuv", "rb");
  uint8_t *inbuf = malloc(w*h*3/2);
  memset(inbuf, 0, w*h*3/2);

  for (int i=0; i<20*3+5; i++) {

    // fread(inbuf, w*h*3/2, 1, infile);

    uint8_t *tmpy = inbuf;
    uint8_t *tmpu = inbuf + w*h;
    uint8_t *tmpv = inbuf + w*h + (w/2)*(h/2);

    for (int y=0; y<h/2; y++) {
      for (int x=0; x<w/2; x++) {
        tmpu[y * (w/2) + x] = (i ^ y ^ x);
      }
    }

    encoder_encode_frame(s, 20000*i, tmpy, tmpu, tmpv);
  }
// #endif

  // while(1);

  printf("done\n");

  // encoder_close(s);

  // printf("restart\n");
  // fclose(s->of);
  // s->of = fopen("/sdcard/tmpout2.hevc", "wb");
  // if (s->codec_config) {
  //   fwrite(s->codec_config, s->codec_config_len, 1, s->of);
  // }
  // encoder_open(s, "/sdcard/t1");

  encoder_rotate(s, "/sdcard/t2");

  for (int i=0; i<20*3+5; i++) {

    // fread(inbuf, w*h*3/2, 1, infile);

    uint8_t *tmpy = inbuf;
    uint8_t *tmpu = inbuf + w*h;
    uint8_t *tmpv = inbuf + w*h + (w/2)*(h/2);

    for (int y=0; y<h/2; y++) {
      for (int x=0; x<w/2; x++) {
        tmpu[y * (w/2) + x] = (i ^ y ^ x);
      }
    }

    encoder_encode_frame(s, 20000*i, tmpy, tmpu, tmpv);
  }
  encoder_close(s);

  return 0;
}
#endif
