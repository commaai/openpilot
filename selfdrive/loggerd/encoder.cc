#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "encoder.h"

#include <OMX_Component.h>
#include <OMX_IndexExt.h>
#include <OMX_QCOMExtns.h>
#include <OMX_VideoExt.h>
#include <assert.h>
#include <fcntl.h>
#include <libyuv.h>
#include <msm_media_info.h>
#include <stdlib.h>
#include <unistd.h>

#include "common/swaglog.h"

#define PORT_INDEX_IN 0
#define PORT_INDEX_OUT 1

#define OERR(cmd) assert((cmd) == OMX_ErrorNone)

void EncoderState::wait_for_state(OMX_STATETYPE state) {
  std::unique_lock<std::mutex> lk(state_lock);
  state_cv.wait(lk, [&]{return this->state == state;});
}

static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                   OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data) {
  EncoderState *s = (EncoderState *)app_data;

  switch (event) {
    case OMX_EventCmdComplete: {
      assert(data1 == OMX_CommandStateSet);
      LOG("set state event 0x%x", data2);
      std::unique_lock<std::mutex> lk(s->state_lock);
      s->state = (OMX_STATETYPE)data2;
      s->state_cv.notify_all();
      break;
    }
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

static OMX_ERRORTYPE empty_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_BUFFERHEADERTYPE *buffer) {
  EncoderState *s = (EncoderState *)app_data;
  queue_push(&s->free_in, (void *)buffer);
  return OMX_ErrorNone;
}

static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_BUFFERHEADERTYPE *buffer) {
  EncoderState *s = (EncoderState *)app_data;
  queue_push(&s->done_out, (void *)buffer);
  return OMX_ErrorNone;
}

static OMX_CALLBACKTYPE omx_callbacks = {
    .EventHandler = event_handler,
    .EmptyBufferDone = empty_buffer_done,
    .FillBufferDone = fill_buffer_done,
};

EncoderState::EncoderState(const LogCameraInfo &camera_info, int width, int height)
    : camera_info(camera_info), width(width), height(height), state(OMX_StateLoaded), remuxing(!camera_info.is_h265) {
  
  queue_init(&free_in);
  queue_init(&done_out);

  int err;
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
  OERR(OMX_GetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&in_port));

  in_port.format.video.nFrameWidth = width;
  in_port.format.video.nFrameHeight = height;
  in_port.format.video.nStride = VENUS_Y_STRIDE(COLOR_FMT_NV12, width);
  in_port.format.video.nSliceHeight = height;
  in_port.nBufferSize = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, width, height);
  in_port.format.video.xFramerate = (camera_info.fps * 65536);
  in_port.format.video.eCompressionFormat = OMX_VIDEO_CodingUnused;
  in_port.format.video.eColorFormat = (OMX_COLOR_FORMATTYPE)QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m;

  OERR(OMX_SetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&in_port));
  OERR(OMX_GetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&in_port));
  num_in_bufs = in_port.nBufferCountActual;

  // setup output port

  OMX_PARAM_PORTDEFINITIONTYPE out_port = {0};
  out_port.nSize = sizeof(out_port);
  out_port.nPortIndex = (OMX_U32)PORT_INDEX_OUT;
  OERR(OMX_GetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&out_port));
  out_port.format.video.nFrameWidth = camera_info.downscale ? camera_info.frame_width : width;
  out_port.format.video.nFrameHeight = camera_info.downscale ? camera_info.frame_height : height;
  out_port.format.video.xFramerate = 0;
  out_port.format.video.nBitrate = camera_info.bitrate;
  out_port.format.video.eCompressionFormat = camera_info.is_h265 ? OMX_VIDEO_CodingHEVC : OMX_VIDEO_CodingAVC;
  out_port.format.video.eColorFormat = OMX_COLOR_FormatUnused;

  OERR(OMX_SetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&out_port));
  OERR(OMX_GetParameter(handle, OMX_IndexParamPortDefinition, (OMX_PTR)&out_port));
  num_out_bufs = out_port.nBufferCountActual;

  OMX_VIDEO_PARAM_BITRATETYPE bitrate_type = {0};
  bitrate_type.nSize = sizeof(bitrate_type);
  bitrate_type.nPortIndex = (OMX_U32)PORT_INDEX_OUT;
  OERR(OMX_GetParameter(handle, OMX_IndexParamVideoBitrate, (OMX_PTR)&bitrate_type));

  bitrate_type.eControlRate = OMX_Video_ControlRateVariable;
  bitrate_type.nTargetBitrate = camera_info.bitrate;

  OERR(OMX_SetParameter(handle, OMX_IndexParamVideoBitrate, (OMX_PTR)&bitrate_type));

  if (camera_info.is_h265) {
#ifndef QCOM2
    // setup HEVC
    OMX_VIDEO_PARAM_HEVCTYPE hecv_type = {0};
    hecv_type.nSize = sizeof(hecv_type);
    hecv_type.nPortIndex = (OMX_U32)PORT_INDEX_OUT;
    OERR(OMX_GetParameter(handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc, (OMX_PTR)&hecv_type));

    hecv_type.eProfile = OMX_VIDEO_HEVCProfileMain;
    hecv_type.eLevel = OMX_VIDEO_HEVCHighTierLevel5;

    OERR(OMX_SetParameter(handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc, (OMX_PTR)&hecv_type));
#endif
  } else {
    // setup h264
    OMX_VIDEO_PARAM_AVCTYPE avc = {0};
    avc.nSize = sizeof(avc);
    avc.nPortIndex = (OMX_U32)PORT_INDEX_OUT;
    OERR(OMX_GetParameter(handle, OMX_IndexParamVideoAvc, &avc));

    avc.nBFrames = 0;
    avc.nPFrames = 15;

    avc.eProfile = OMX_VIDEO_AVCProfileBaseline;
    avc.eLevel = OMX_VIDEO_AVCLevel31;

    avc.nAllowedPictureTypes |= OMX_VIDEO_PictureTypeB;
    avc.eLoopFilterMode = OMX_VIDEO_AVCLoopFilterEnable;

    OERR(OMX_SetParameter(handle, OMX_IndexParamVideoAvc, &avc));
  }

  OERR(OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateIdle, NULL));

  in_buf_headers = std::make_unique<OMX_BUFFERHEADERTYPE *[]>(num_in_bufs);
  for (int i = 0; i < num_in_bufs; i++) {
    OERR(OMX_AllocateBuffer(handle, &in_buf_headers[i], PORT_INDEX_IN, this, in_port.nBufferSize));
  }

  out_buf_headers = std::make_unique<OMX_BUFFERHEADERTYPE *[]>(num_out_bufs);
  for (int i = 0; i < num_out_bufs; i++) {
    OERR(OMX_AllocateBuffer(handle, &out_buf_headers[i], PORT_INDEX_OUT, this, out_port.nBufferSize));
  }

  wait_for_state(OMX_StateIdle);
  OERR(OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateExecuting, NULL));
  wait_for_state(OMX_StateExecuting);

  // give omx all the output buffers
  for (int i = 0; i < num_out_bufs; i++) {
    OERR(OMX_FillThisBuffer(handle, out_buf_headers[i]));
  }

  // fill the input free queue
  for (int i = 0; i < num_in_bufs; i++) {
    queue_push(&free_in, (void *)in_buf_headers[i]);
  }
}

EncoderState::~EncoderState() {
  if (is_open) {
    Close();
  }

  OERR(OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateIdle, NULL));
  wait_for_state(OMX_StateIdle);

  OERR(OMX_SendCommand(handle, OMX_CommandStateSet, OMX_StateLoaded, NULL));

  for (int i = 0; i < num_in_bufs; i++) {
    OERR(OMX_FreeBuffer(handle, PORT_INDEX_IN, in_buf_headers[i]));
  }
  for (int i = 0; i < num_out_bufs; i++) {
    OERR(OMX_FreeBuffer(handle, PORT_INDEX_OUT, out_buf_headers[i]));
  }
  wait_for_state(OMX_StateLoaded);

  OERR(OMX_FreeHandle(handle));
  LOG("encoder destroy");
}

void EncoderState::handle_out_buf(OMX_BUFFERHEADERTYPE *out_buf) {
  int err;
  uint8_t *buf_data = out_buf->pBuffer + out_buf->nOffset;

  if (out_buf->nFlags & OMX_BUFFERFLAG_CODECCONFIG) {
    if (codec_config.size() < out_buf->nFilledLen) {
      codec_config.resize(out_buf->nFilledLen);
    }
    memcpy(codec_config.data(), buf_data, out_buf->nFilledLen);
  }

  if (remuxing) {
    if (!wrote_codec_config && codec_config.size() > 0) {
      if (codec_ctx->extradata_size < codec_config.size()) {
        if (codec_ctx->extradata) {
          free(codec_ctx->extradata);
        }
        codec_ctx->extradata = (uint8_t *)calloc(codec_config.size() + AV_INPUT_BUFFER_PADDING_SIZE, sizeof(uint8_t));
      }
      codec_ctx->extradata_size = codec_config.size();
      memcpy(codec_ctx->extradata, codec_config.data(), codec_config.size());

      err = avcodec_parameters_from_context(out_stream->codecpar, codec_ctx);
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

      err = av_write_frame(ofmt_ctx, &pkt);
      if (err < 0) {
        LOGW("ts encoder write issue");
      }
      av_free_packet(&pkt);
    }
  } else {
    write(fd, buf_data, out_buf->nFilledLen);
    total_written += out_buf->nFilledLen;
  }

  // give omx back the buffer
  OERR(OMX_FillThisBuffer(handle, out_buf));
}

int EncoderState::EncodeFrame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, VIPCBufExtra *extra) {
  int ret = counter;
  OMX_BUFFERHEADERTYPE *in_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&free_in);
  uint8_t *in_buf_ptr = in_buf->pBuffer;

  uint8_t *in_y_ptr = in_buf_ptr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, width);
  uint8_t *in_uv_ptr = in_buf_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, height));

  int err = libyuv::I420ToNV12(y_ptr, width,
                           u_ptr, width / 2,
                           v_ptr, width / 2,
                           in_y_ptr, in_y_stride,
                           in_uv_ptr, in_uv_stride,
                           width, height);
  assert(err == 0);

  in_buf->nFilledLen = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, width, height);
  in_buf->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;
  in_buf->nOffset = 0;
  in_buf->nTimeStamp = extra->timestamp_eof / 1000LL;  // OMX_TICKS, in microseconds

  OERR(OMX_EmptyThisBuffer(handle, in_buf));

  // pump output
  while (true) {
    OMX_BUFFERHEADERTYPE *out_buf = (OMX_BUFFERHEADERTYPE *)queue_try_pop(&done_out);
    if (!out_buf) {
      break;
    }
    handle_out_buf(out_buf);
  }

  counter++;
  return ret;
}

void EncoderState::Open(const char *path) {
  int err;
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

    AVCodec *codec = NULL;
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    assert(codec);

    codec_ctx = avcodec_alloc_context3(codec);
    assert(codec_ctx);
    codec_ctx->width = camera_info.downscale ? camera_info.frame_width : width;
    codec_ctx->height = camera_info.downscale ? camera_info.frame_height : height;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->time_base = (AVRational){1, camera_info.fps};

    err = avio_open(&ofmt_ctx->pb, vid_path, AVIO_FLAG_WRITE);
    assert(err >= 0);

    wrote_codec_config = false;
  } else {
    fd = open(vid_path, O_CREAT | O_WRONLY);
    assert(fd >= 0);
    fallocate(fd, 0, 0, 50 * 1024 * 1024);
    lseek(fd, 0, SEEK_SET);
    total_written = 0;
    if (codec_config.size() > 0) {
      write(fd, codec_config.data(), codec_config.size());
      total_written += codec_config.size();
    }
  }

  // create camera lock file
  snprintf(lock_path, sizeof(lock_path), "%s/%s.lock", path, camera_info.filename);
  int lock_fd = open(lock_path, O_RDWR | O_CREAT, 0777);
  assert(lock_fd >= 0);
  close(lock_fd);

  is_open = true;
  counter = 0;

}

void EncoderState::Close() {
  if (counter > 0) {
    // drain output only if there could be frames in the encoder

    OMX_BUFFERHEADERTYPE *in_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&free_in);
    in_buf->nFilledLen = 0;
    in_buf->nOffset = 0;
    in_buf->nFlags = OMX_BUFFERFLAG_EOS;
    in_buf->nTimeStamp = 0;
    OERR(OMX_EmptyThisBuffer(handle, in_buf));

    while (true) {
      OMX_BUFFERHEADERTYPE *out_buf = (OMX_BUFFERHEADERTYPE *)queue_pop(&done_out);
      if (out_buf->nFlags & OMX_BUFFERFLAG_EOS) {
        break;
      }
      handle_out_buf(out_buf);
    }
    counter = 0;
  }

  if (remuxing) {
    av_write_trailer(ofmt_ctx);
    avcodec_free_context(&codec_ctx);
    avio_closep(&ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
  } else {
    ftruncate(fd, total_written);
    close(fd);
    fd = -1;
  }
  unlink(lock_path);
  is_open = false;
}

void EncoderState::Rotate(const char *new_path) {
  if (is_open) {
    Close();
  }
  Open(new_path);
}
