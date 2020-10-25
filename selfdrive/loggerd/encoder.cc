#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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

#include "common/swaglog.h"

#include "encoder.h"

// encoder: lossey codec using hardware hevc
void EncoderState::wait_for_state(OMX_STATETYPE state) {
  std::unique_lock<std::mutex> lk(state_lock);
  state_cv.wait(lk, [&]{return this->state == state;});
}

static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                   OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data) {
  EncoderState *s = (EncoderState *)app_data;
  if (event == OMX_EventCmdComplete) {
    assert(data1 == OMX_CommandStateSet);
    LOG("set state event 0x%x", data2);

    std::unique_lock<std::mutex> lk(s->state_lock);
    s->state = (OMX_STATETYPE)data2;
    s->state_cv.notify_all();
  } else if (event == OMX_EventError) {
    LOGE("OMX error 0x%08x", data1);
  } else {
    LOGE("OMX unhandled event %d", event);
    assert(false);
  }

  return OMX_ErrorNone;
}

static OMX_ERRORTYPE empty_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_BUFFERHEADERTYPE *buffer) {
  queue_push(&reinterpret_cast<EncoderState *>(app_data)->free_in, (void *)buffer);
  return OMX_ErrorNone;
}

static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_BUFFERHEADERTYPE *buffer) {
  queue_push(&reinterpret_cast<EncoderState *>(app_data)->done_out, (void *)buffer);
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

EncoderState::EncoderState(const LogCameraInfo &camera_info, int width, int height)
    : camera_info(camera_info), in_width(width), in_height(height), state(OMX_StateLoaded), remuxing(!camera_info.is_h265) {
  int err;
  queue_init(&free_in);
  queue_init(&done_out);

  if (camera_info.is_h265) {
    err = OMX_GetHandle(&handle, (OMX_STRING) "OMX.qcom.video.encoder.hevc", this, &omx_callbacks);
  } else {
    err = OMX_GetHandle(&handle, (OMX_STRING) "OMX.qcom.video.encoder.avc", this, &omx_callbacks);
  }
  if (err != OMX_ErrorNone) {
    LOGE("error getting codec: %x", err);
  }
  assert(err == OMX_ErrorNone);

  // setup input port

  OMX_PARAM_PORTDEFINITIONTYPE in_port = {0};
  in_port.nSize = sizeof(in_port);
  in_port.nPortIndex = (OMX_U32)PORT_INDEX_IN;
  err = OMX_GetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&in_port);
  assert(err == OMX_ErrorNone);

  in_port.format.video.nFrameWidth = in_width;
  in_port.format.video.nFrameHeight = in_height;
  in_port.format.video.nStride = VENUS_Y_STRIDE(COLOR_FMT_NV12, in_width);
  in_port.format.video.nSliceHeight = in_height;
  in_port.nBufferSize = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, in_width, in_height);
  in_port.format.video.xFramerate = (camera_info.fps * 65536);
  in_port.format.video.eCompressionFormat = OMX_VIDEO_CodingUnused;
  in_port.format.video.eColorFormat = (OMX_COLOR_FORMATTYPE)QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m;

  err = OMX_SetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&in_port);
  assert(err == OMX_ErrorNone);
  err = OMX_GetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&in_port);
  assert(err == OMX_ErrorNone);
  in_buf_headers.resize(in_port.nBufferCountActual);

  // setup output port

  OMX_PARAM_PORTDEFINITIONTYPE out_port = {0};
  out_port.nSize = sizeof(out_port);
  out_port.nPortIndex = (OMX_U32)PORT_INDEX_OUT;
  err = OMX_GetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&out_port);
  assert(err == OMX_ErrorNone);
  out_port.format.video.nFrameWidth = camera_info.downscale ? camera_info.frame_width : in_width;
  out_port.format.video.nFrameHeight = camera_info.downscale ? camera_info.frame_height : in_height;
  out_port.format.video.xFramerate = 0;
  out_port.format.video.nBitrate = camera_info.bitrate;
  if (camera_info.is_h265) {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingHEVC;
  } else {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingAVC;
  }
  out_port.format.video.eColorFormat = OMX_COLOR_FormatUnused;

  err = OMX_SetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&out_port);
  assert(err == OMX_ErrorNone);

  err = OMX_GetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&out_port);
  assert(err == OMX_ErrorNone);
  out_buf_headers.resize(out_port.nBufferCountActual);

  OMX_VIDEO_PARAM_BITRATETYPE bitrate_type = {0};
  bitrate_type.nSize = sizeof(bitrate_type);
  bitrate_type.nPortIndex = (OMX_U32)PORT_INDEX_OUT;
  err = OMX_GetParameter(handle, OMX_IndexParamVideoBitrate, (OMX_PTR)&bitrate_type);
  assert(err == OMX_ErrorNone);

  bitrate_type.eControlRate = OMX_Video_ControlRateVariable;
  bitrate_type.nTargetBitrate = camera_info.bitrate;

  err = OMX_SetParameter(handle, OMX_IndexParamVideoBitrate, (OMX_PTR)&bitrate_type);
  assert(err == OMX_ErrorNone);

  if (camera_info.is_h265) {
#ifndef QCOM2
    // setup HEVC
    OMX_VIDEO_PARAM_HEVCTYPE hecv_type = {0};
    hecv_type.nSize = sizeof(hecv_type);
    hecv_type.nPortIndex = (OMX_U32)PORT_INDEX_OUT;
    err = OMX_GetParameter(handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc, (OMX_PTR)&hecv_type);
    assert(err == OMX_ErrorNone);

    hecv_type.eProfile = OMX_VIDEO_HEVCProfileMain;
    hecv_type.eLevel = OMX_VIDEO_HEVCHighTierLevel5;

    err = OMX_SetParameter(handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc, (OMX_PTR)&hecv_type);
    assert(err == OMX_ErrorNone);
#endif
  } else {
    // setup h264
    OMX_VIDEO_PARAM_AVCTYPE avc = {0};
    avc.nSize = sizeof(avc);
    avc.nPortIndex = (OMX_U32)PORT_INDEX_OUT;
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

  err = OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
  assert(err == OMX_ErrorNone);

  for (auto &buf : in_buf_headers) {
    err = OMX_AllocateBuffer(handle, &buf, PORT_INDEX_IN, this, in_port.nBufferSize);
    assert(err == OMX_ErrorNone);
  }

  for (auto &buf : out_buf_headers) {
    err = OMX_AllocateBuffer(handle, &buf, PORT_INDEX_OUT, this, out_port.nBufferSize);
    assert(err == OMX_ErrorNone);
  }

  wait_for_state(OMX_StateIdle);
  err = OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateExecuting, NULL);
  assert(err == OMX_ErrorNone);

  wait_for_state(OMX_StateExecuting);

  // give omx all the output buffers
  for (auto &buf : out_buf_headers) {
    err = OMX_FillThisBuffer(handle, buf);
    assert(err == OMX_ErrorNone);
  }

  // fill the input free queue
  for (auto &buf : in_buf_headers) {
    queue_push(&free_in, (void *)buf);
  }
}

void EncoderState::handle_out_buf(OMX_BUFFERHEADERTYPE *out_buf) {
  uint8_t *buf_data = out_buf->pBuffer + out_buf->nOffset;

  if (out_buf->nFlags & OMX_BUFFERFLAG_CODECCONFIG) {
    if (codec_config.size() < out_buf->nFilledLen) {
      codec_config.resize(out_buf->nFilledLen);
    }
    memcpy(codec_config.data(), buf_data, out_buf->nFilledLen);
  }

  if (of) {
    fwrite(buf_data, out_buf->nFilledLen, 1, of);
  }

  if (remuxing) {
    if (!wrote_codec_config && codec_config.size() > 0) {
      // extradata will be freed by avcodec_free_context()
      codec_ctx->extradata = (uint8_t *)calloc(codec_config.size() + AV_INPUT_BUFFER_PADDING_SIZE, sizeof(uint8_t));
      codec_ctx->extradata_size = codec_config.size();
      memcpy(codec_ctx->extradata, codec_config.data(), codec_config.size());

      int err = avcodec_parameters_from_context(out_stream->codecpar, codec_ctx);
      assert(err >= 0);
      err = avformat_write_header(ofmt_ctx, NULL);
      assert(err >= 0);

      wrote_codec_config = true;
    }

    if (out_buf->nTimeStamp > 0) {
      // input timestamps are in microseconds
      AVRational in_timebase = {1, 1000000};

      AVPacket pkt;
      av_init_packet(&pkt);
      pkt.data = buf_data;
      pkt.size = out_buf->nFilledLen;
      pkt.pts = pkt.dts = av_rescale_q_rnd(out_buf->nTimeStamp, in_timebase, ofmt_ctx->streams[0]->time_base,
                                           (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
      pkt.duration = av_rescale_q(50 * 1000, in_timebase, ofmt_ctx->streams[0]->time_base);

      if (out_buf->nFlags & OMX_BUFFERFLAG_SYNCFRAME) {
        pkt.flags |= AV_PKT_FLAG_KEY;
      }

      int err = av_write_frame(ofmt_ctx, &pkt);
      if (err < 0) { LOGW("ts encoder write issue"); }

      av_free_packet(&pkt);
    }
  }
  // give omx back the buffer
  int err = OMX_FillThisBuffer(handle, out_buf);
  assert(err == OMX_ErrorNone);
}

int EncoderState::EncodeFrame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, const VIPCBufExtra &extra) {
  OMX_BUFFERHEADERTYPE *in_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&free_in);
  uint8_t *in_buf_ptr = in_buf->pBuffer;

  uint8_t *in_y_ptr = in_buf_ptr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, in_width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, in_width);
  uint8_t *in_uv_ptr = in_buf_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, in_height));

  int err = libyuv::I420ToNV12(y_ptr, in_width,
                   u_ptr, in_width/2,
                   v_ptr, in_width/2,
                   in_y_ptr, in_y_stride,
                   in_uv_ptr, in_uv_stride,
                   in_width, in_height);
  assert(err == 0);

  in_buf->nFilledLen = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, in_width, in_height);
  in_buf->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;
  in_buf->nOffset = 0;
  in_buf->nTimeStamp = extra.timestamp_eof / 1000LL;  // OMX_TICKS, in microseconds

  err = OMX_EmptyThisBuffer(handle, in_buf);
  assert(err == OMX_ErrorNone);

  // pump output
  while (true) {
    OMX_BUFFERHEADERTYPE *out_buf = (OMX_BUFFERHEADERTYPE *)queue_try_pop(&done_out);
    if (!out_buf) {
      break;
    }
    handle_out_buf(out_buf);
  }

  dirty = true;
  return ++counter;
}

void EncoderState::Open(const char *path) {
  char vid_path[4096];
  snprintf(vid_path, sizeof(vid_path), "%s/%s", path, camera_info.filename);
  LOGD("encoder_open %s remuxing:%d", vid_path, remuxing);

  if (remuxing) {
    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, vid_path);
    assert(ofmt_ctx);

#ifdef QCOM2
    ofmt_ctx->oformat->flags = AVFMT_TS_NONSTRICT;
#endif
    out_stream = avformat_new_stream(ofmt_ctx, NULL);
    assert(out_stream);

    // set codec correctly
    av_register_all();

    AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    assert(codec);

    codec_ctx = avcodec_alloc_context3(codec);
    assert(codec_ctx);
    codec_ctx->width = camera_info.downscale ? camera_info.frame_width : in_width;
    codec_ctx->height = camera_info.downscale ? camera_info.frame_height : in_height;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->time_base = (AVRational){1, camera_info.fps};

    int err = avio_open(&ofmt_ctx->pb, vid_path, AVIO_FLAG_WRITE);
    assert(err >= 0);

    wrote_codec_config = false;
  } else {
    of = fopen(vid_path, "wb");
    assert(of);
    if (codec_config.size() > 0) {
      fwrite(codec_config.data(), codec_config.size(), 1, of);
    }
  }

  // create camera lock file
  snprintf(lock_path, sizeof(lock_path), "%s.lock", vid_path);
  int lock_fd = open(lock_path, O_RDWR | O_CREAT, 0777);
  assert(lock_fd >= 0);
  close(lock_fd);

  is_open = true;
  counter = 0;
}

void EncoderState::Close() {
  if (is_open) {
    if (dirty) {
      // drain output only if there could be frames in the encoder
      OMX_BUFFERHEADERTYPE *in_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&free_in);
      in_buf->nFilledLen = 0;
      in_buf->nOffset = 0;
      in_buf->nFlags = OMX_BUFFERFLAG_EOS;
      in_buf->nTimeStamp = 0;
      int err = OMX_EmptyThisBuffer(handle, in_buf);
      assert(err == OMX_ErrorNone);

      while (true) {
        OMX_BUFFERHEADERTYPE *out_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&done_out);
        handle_out_buf(out_buf);
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
      of = nullptr;
    }
    unlink(lock_path);
  }
  is_open = false;
}

void EncoderState::Rotate(const char *new_path) {
  if (is_open) {
    Close();
  }
  Open(new_path);
}

EncoderState::~EncoderState() {
  if (is_open) {
    Close();
  }

  int err = OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
  assert(err == OMX_ErrorNone);
  wait_for_state(OMX_StateIdle);
  err = OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateLoaded, NULL);
  assert(err == OMX_ErrorNone);

  for (auto &buf : in_buf_headers) {
    err = OMX_FreeBuffer(handle, PORT_INDEX_IN, buf);
    assert(err == OMX_ErrorNone);
  }
  for (auto &buf : out_buf_headers) {
    err = OMX_FreeBuffer(handle, PORT_INDEX_OUT, buf);
    assert(err == OMX_ErrorNone);
  }
  wait_for_state(OMX_StateLoaded);

  err = OMX_FreeHandle(handle);
  assert(err == OMX_ErrorNone);
  LOG("encoder destroy");
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
//   encoder.c
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
