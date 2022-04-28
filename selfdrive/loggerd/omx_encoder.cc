#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "selfdrive/loggerd/omx_encoder.h"
#include "cereal/messaging/messaging.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstdio>

#include <OMX_Component.h>
#include <OMX_IndexExt.h>
#include <OMX_QCOMExtns.h>
#include <OMX_VideoExt.h>
#include "libyuv.h"

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/include/msm_media_info.h"

// Check the OMX error code and assert if an error occurred.
#define OMX_CHECK(_expr)              \
  do {                                \
    assert(OMX_ErrorNone == (_expr)); \
  } while (0)

extern ExitHandler do_exit;

// ***** OMX callback functions *****

void OmxEncoder::wait_for_state(OMX_STATETYPE state_) {
  std::unique_lock lk(this->state_lock);
  while (this->state != state_) {
    this->state_cv.wait(lk);
  }
}

static OMX_CALLBACKTYPE omx_callbacks = {
  .EventHandler = OmxEncoder::event_handler,
  .EmptyBufferDone = OmxEncoder::empty_buffer_done,
  .FillBufferDone = OmxEncoder::fill_buffer_done,
};

OMX_ERRORTYPE OmxEncoder::event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                   OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data) {
  OmxEncoder *e = (OmxEncoder*)app_data;
  if (event == OMX_EventCmdComplete) {
    assert(data1 == OMX_CommandStateSet);
    LOG("set state event 0x%x", data2);
    {
      std::unique_lock lk(e->state_lock);
      e->state = (OMX_STATETYPE)data2;
    }
    e->state_cv.notify_all();
  } else if (event == OMX_EventError) {
    LOGE("OMX error 0x%08x", data1);
  } else {
    LOGE("OMX unhandled event %d", event);
    assert(false);
  }

  return OMX_ErrorNone;
}

OMX_ERRORTYPE OmxEncoder::empty_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                                   OMX_BUFFERHEADERTYPE *buffer) {
  OmxEncoder *e = (OmxEncoder*)app_data;
  e->free_in.push(buffer);
  return OMX_ErrorNone;
}

OMX_ERRORTYPE OmxEncoder::fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                                  OMX_BUFFERHEADERTYPE *buffer) {
  OmxEncoder *e = (OmxEncoder*)app_data;
  e->done_out.push(buffer);
  return OMX_ErrorNone;
}

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
OmxEncoder::OmxEncoder(const char* filename, CameraType type, int in_width, int in_height, int fps, int bitrate, bool h265, int out_width, int out_height, bool write)
  : in_width_(in_width), in_height_(in_height), width(out_width), height(out_height) {
  this->filename = filename;
  this->type = type;
  this->write = write;
  this->fps = fps;
  this->remuxing = !h265;

  this->downscale = in_width != out_width || in_height != out_height;
  if (this->downscale) {
    this->y_ptr2 = (uint8_t *)malloc(this->width*this->height);
    this->u_ptr2 = (uint8_t *)malloc(this->width*this->height/4);
    this->v_ptr2 = (uint8_t *)malloc(this->width*this->height/4);
  }

  auto component = (OMX_STRING)(h265 ? "OMX.qcom.video.encoder.hevc" : "OMX.qcom.video.encoder.avc");
  int err = OMX_GetHandle(&this->handle, component, this, &omx_callbacks);
  if (err != OMX_ErrorNone) {
    LOGE("error getting codec: %x", err);
  }
  assert(err == OMX_ErrorNone);
  // printf("handle: %p\n", this->handle);

  // setup input port

  OMX_PARAM_PORTDEFINITIONTYPE in_port = {0};
  in_port.nSize = sizeof(in_port);
  in_port.nPortIndex = (OMX_U32) PORT_INDEX_IN;
  OMX_CHECK(OMX_GetParameter(this->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &in_port));

  in_port.format.video.nFrameWidth = this->width;
  in_port.format.video.nFrameHeight = this->height;
  in_port.format.video.nStride = VENUS_Y_STRIDE(COLOR_FMT_NV12, this->width);
  in_port.format.video.nSliceHeight = this->height;
  // in_port.nBufferSize = (this->width * this->height * 3) / 2;
  in_port.nBufferSize = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, this->width, this->height);
  in_port.format.video.xFramerate = (this->fps * 65536);
  in_port.format.video.eCompressionFormat = OMX_VIDEO_CodingUnused;
  // in_port.format.video.eColorFormat = OMX_COLOR_FormatYUV420SemiPlanar;
  in_port.format.video.eColorFormat = (OMX_COLOR_FORMATTYPE)QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m;

  OMX_CHECK(OMX_SetParameter(this->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &in_port));
  OMX_CHECK(OMX_GetParameter(this->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &in_port));
  this->in_buf_headers.resize(in_port.nBufferCountActual);

  // setup output port

  OMX_PARAM_PORTDEFINITIONTYPE out_port = {0};
  out_port.nSize = sizeof(out_port);
  out_port.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  OMX_CHECK(OMX_GetParameter(this->handle, OMX_IndexParamPortDefinition, (OMX_PTR)&out_port));
  out_port.format.video.nFrameWidth = this->width;
  out_port.format.video.nFrameHeight = this->height;
  out_port.format.video.xFramerate = 0;
  out_port.format.video.nBitrate = bitrate;
  if (h265) {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingHEVC;
  } else {
    out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingAVC;
  }
  out_port.format.video.eColorFormat = OMX_COLOR_FormatUnused;

  OMX_CHECK(OMX_SetParameter(this->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &out_port));

  OMX_CHECK(OMX_GetParameter(this->handle, OMX_IndexParamPortDefinition, (OMX_PTR) &out_port));
  this->out_buf_headers.resize(out_port.nBufferCountActual);

  OMX_VIDEO_PARAM_BITRATETYPE bitrate_type = {0};
  bitrate_type.nSize = sizeof(bitrate_type);
  bitrate_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  OMX_CHECK(OMX_GetParameter(this->handle, OMX_IndexParamVideoBitrate, (OMX_PTR) &bitrate_type));
  bitrate_type.eControlRate = OMX_Video_ControlRateVariable;
  bitrate_type.nTargetBitrate = bitrate;

  OMX_CHECK(OMX_SetParameter(this->handle, OMX_IndexParamVideoBitrate, (OMX_PTR) &bitrate_type));

  if (h265) {
    // setup HEVC
    OMX_VIDEO_PARAM_PROFILELEVELTYPE hevc_type = {0};
    OMX_INDEXTYPE index_type = OMX_IndexParamVideoProfileLevelCurrent;
    hevc_type.nSize = sizeof(hevc_type);
    hevc_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
    OMX_CHECK(OMX_GetParameter(this->handle, index_type, (OMX_PTR) &hevc_type));

    hevc_type.eProfile = OMX_VIDEO_HEVCProfileMain;
    hevc_type.eLevel = OMX_VIDEO_HEVCHighTierLevel5;

    OMX_CHECK(OMX_SetParameter(this->handle, index_type, (OMX_PTR) &hevc_type));
  } else {
    // setup h264
    OMX_VIDEO_PARAM_AVCTYPE avc = { 0 };
    avc.nSize = sizeof(avc);
    avc.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
    OMX_CHECK(OMX_GetParameter(this->handle, OMX_IndexParamVideoAvc, &avc));

    avc.nBFrames = 0;
    avc.nPFrames = 15;

    avc.eProfile = OMX_VIDEO_AVCProfileHigh;
    avc.eLevel = OMX_VIDEO_AVCLevel31;

    avc.nAllowedPictureTypes |= OMX_VIDEO_PictureTypeB;
    avc.eLoopFilterMode = OMX_VIDEO_AVCLoopFilterEnable;

    avc.nRefFrames = 1;
    avc.bUseHadamard = OMX_TRUE;
    avc.bEntropyCodingCABAC = OMX_TRUE;
    avc.bWeightedPPrediction = OMX_TRUE;
    avc.bconstIpred = OMX_TRUE;

    OMX_CHECK(OMX_SetParameter(this->handle, OMX_IndexParamVideoAvc, &avc));
  }


  // for (int i = 0; ; i++) {
  //   OMX_VIDEO_PARAM_PORTFORMATTYPE video_port_format = {0};
  //   video_port_format.nSize = sizeof(video_port_format);
  //   video_port_format.nIndex = i;
  //   video_port_format.nPortIndex = PORT_INDEX_IN;
  //   if (OMX_GetParameter(this->handle, OMX_IndexParamVideoPortFormat, &video_port_format) != OMX_ErrorNone)
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
  //   if (OMX_GetParameter(this->handle, OMX_IndexParamVideoProfileLevelQuerySupported, &params) != OMX_ErrorNone)
  //       break;
  //   printf("profile %d level 0x%x\n", params.eProfile, params.eLevel);
  // }

  OMX_CHECK(OMX_SendCommand(this->handle, OMX_CommandStateSet, OMX_StateIdle, NULL));

  for (auto &buf : this->in_buf_headers) {
    OMX_CHECK(OMX_AllocateBuffer(this->handle, &buf, PORT_INDEX_IN, this,
                             in_port.nBufferSize));
  }

  for (auto &buf : this->out_buf_headers) {
    OMX_CHECK(OMX_AllocateBuffer(this->handle, &buf, PORT_INDEX_OUT, this,
                             out_port.nBufferSize));
  }

  wait_for_state(OMX_StateIdle);

  OMX_CHECK(OMX_SendCommand(this->handle, OMX_CommandStateSet, OMX_StateExecuting, NULL));

  wait_for_state(OMX_StateExecuting);

  // give omx all the output buffers
  for (auto &buf : this->out_buf_headers) {
    // printf("fill %p\n", this->out_buf_headers[i]);
    OMX_CHECK(OMX_FillThisBuffer(this->handle, buf));
  }

  // fill the input free queue
  for (auto &buf : this->in_buf_headers) {
    this->free_in.push(buf);
  }

  LOGE("omx initialized - in: %d - out %d", this->in_buf_headers.size(), this->out_buf_headers.size());

  service_name = this->type == DriverCam ? "driverEncodeData" :
    (this->type == WideRoadCam ? "wideRoadEncodeData" :
    (this->remuxing ? "qRoadEncodeData" : "roadEncodeData"));
  pm.reset(new PubMaster({service_name}));
}

void OmxEncoder::callback_handler(OmxEncoder *e) {
  // OMX documentation specifies to not empty the buffer from the callback function
  // so we use this intermediate handler to copy the buffer for further writing
  // and give it back to OMX. We could also send the data over msgq from here.
  bool exit = false;

  while (!exit) {
    OMX_BUFFERHEADERTYPE *buffer = e->done_out.pop();
    OmxBuffer *new_buffer = (OmxBuffer*)malloc(sizeof(OmxBuffer) + buffer->nFilledLen);
    assert(new_buffer);

    new_buffer->header = *buffer;
    memcpy(new_buffer->data, buffer->pBuffer + buffer->nOffset, buffer->nFilledLen);

    e->to_write.push(new_buffer);

#ifdef QCOM2
    if (buffer->nFlags & OMX_BUFFERFLAG_CODECCONFIG) {
      buffer->nTimeStamp = 0;
    }

    if (buffer->nFlags & OMX_BUFFERFLAG_EOS) {
      buffer->nTimeStamp = 0;
    }
#endif

    if (buffer->nFlags & OMX_BUFFERFLAG_EOS) {
      exit = true;
    }

    // give omx back the buffer
    // TOOD: fails when shutting down
    OMX_CHECK(OMX_FillThisBuffer(e->handle, buffer));
  }
}

void OmxEncoder::write_and_broadcast_handler(OmxEncoder *e){
  bool exit = false;

  e->segment_num++;
  uint32_t idx = 0;
  while (!exit) {
    OmxBuffer *out_buf = e->to_write.pop();

    MessageBuilder msg;
    auto edata = (e->type == DriverCam) ? msg.initEvent(true).initDriverEncodeData() :
      ((e->type == WideRoadCam) ? msg.initEvent(true).initWideRoadEncodeData() :
      (e->remuxing ? msg.initEvent(true).initQRoadEncodeData() : msg.initEvent(true).initRoadEncodeData()));
    edata.setData(kj::heapArray<capnp::byte>(out_buf->data, out_buf->header.nFilledLen));
    edata.setTimestampEof(out_buf->header.nTimeStamp);
    edata.setIdx(idx++);
    edata.setSegmentNum(e->segment_num);
    edata.setFlags(out_buf->header.nFlags);
    e->pm->send(e->service_name, msg);

    OmxEncoder::handle_out_buf(e, out_buf);
    if (out_buf->header.nFlags & OMX_BUFFERFLAG_EOS) {
      exit = true;
    }

    free(out_buf);
  }
}


void OmxEncoder::handle_out_buf(OmxEncoder *e, OmxBuffer *out_buf) {
  if (!(out_buf->header.nFlags & OMX_BUFFERFLAG_EOS) && e->writer) {
    e->writer->write(out_buf->data,
      out_buf->header.nFilledLen,
      out_buf->header.nTimeStamp,
      out_buf->header.nFlags & OMX_BUFFERFLAG_CODECCONFIG,
      out_buf->header.nFlags & OMX_BUFFERFLAG_SYNCFRAME);
  }
}

int OmxEncoder::encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                             int in_width, int in_height, uint64_t ts) {
  assert(in_width == this->in_width_);
  assert(in_height == this->in_height_);
  int err;
  if (!this->is_open) {
    return -1;
  }

  // this sometimes freezes... put it outside the encoder lock so we can still trigger rotates...
  // THIS IS A REALLY BAD IDEA, but apparently the race has to happen 30 times to trigger this
  //pthread_mutex_unlock(&this->lock);
  OMX_BUFFERHEADERTYPE* in_buf = nullptr;
  while (!this->free_in.try_pop(in_buf, 20)) {
    if (do_exit) {
      return -1;
    }
  }

  //pthread_mutex_lock(&this->lock);

  int ret = this->counter;

  uint8_t *in_buf_ptr = in_buf->pBuffer;
  // printf("in_buf ptr %p\n", in_buf_ptr);

  uint8_t *in_y_ptr = in_buf_ptr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, this->width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, this->width);
  // uint8_t *in_uv_ptr = in_buf_ptr + (this->width * this->height);
  uint8_t *in_uv_ptr = in_buf_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, this->height));

  if (this->downscale) {
    I420Scale(y_ptr, in_width,
              u_ptr, in_width/2,
              v_ptr, in_width/2,
              in_width, in_height,
              this->y_ptr2, this->width,
              this->u_ptr2, this->width/2,
              this->v_ptr2, this->width/2,
              this->width, this->height,
              libyuv::kFilterNone);
    y_ptr = this->y_ptr2;
    u_ptr = this->u_ptr2;
    v_ptr = this->v_ptr2;
  }
  err = libyuv::I420ToNV12(y_ptr, this->width,
                   u_ptr, this->width/2,
                   v_ptr, this->width/2,
                   in_y_ptr, in_y_stride,
                   in_uv_ptr, in_uv_stride,
                   this->width, this->height);
  assert(err == 0);

  // in_buf->nFilledLen = (this->width*this->height) + (this->width*this->height/2);
  in_buf->nFilledLen = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, this->width, this->height);
  in_buf->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;
  in_buf->nOffset = 0;
  in_buf->nTimeStamp = ts/1000LL;  // OMX_TICKS, in microseconds
  this->last_t = in_buf->nTimeStamp;

  OMX_CHECK(OMX_EmptyThisBuffer(this->handle, in_buf));

  this->dirty = true;

  this->counter++;

  return ret;
}

void OmxEncoder::encoder_open(const char* path) {
  if (this->write) {
    writer.reset(new VideoWriter(path, this->filename, this->remuxing, this->width, this->height, this->fps, !this->remuxing, false));
  }

  // start writer threads
  callback_handler_thread = std::thread(OmxEncoder::callback_handler, this);
  write_handler_thread = std::thread(OmxEncoder::write_and_broadcast_handler, this);

  this->is_open = true;
  this->counter = 0;
}

void OmxEncoder::encoder_close() {
  if (this->is_open) {
    if (this->dirty) {
      // drain output only if there could be frames in the encoder

      OMX_BUFFERHEADERTYPE* in_buf = this->free_in.pop();
      in_buf->nFilledLen = 0;
      in_buf->nOffset = 0;
      in_buf->nFlags = OMX_BUFFERFLAG_EOS;
      in_buf->nTimeStamp = this->last_t + 1000000LL/this->fps;

      OMX_CHECK(OMX_EmptyThisBuffer(this->handle, in_buf));

      this->dirty = false;
    }

    callback_handler_thread.join();
    write_handler_thread.join();

    writer.reset();
  }
  this->is_open = false;
}

OmxEncoder::~OmxEncoder() {
  assert(!this->is_open);

  OMX_CHECK(OMX_SendCommand(this->handle, OMX_CommandStateSet, OMX_StateIdle, NULL));

  wait_for_state(OMX_StateIdle);

  OMX_CHECK(OMX_SendCommand(this->handle, OMX_CommandStateSet, OMX_StateLoaded, NULL));

  for (auto &buf : this->in_buf_headers) {
    OMX_CHECK(OMX_FreeBuffer(this->handle, PORT_INDEX_IN, buf));
  }

  for (auto &buf : this->out_buf_headers) {
    OMX_CHECK(OMX_FreeBuffer(this->handle, PORT_INDEX_OUT, buf));
  }

  wait_for_state(OMX_StateLoaded);

  OMX_CHECK(OMX_FreeHandle(this->handle));

  OMX_BUFFERHEADERTYPE *buf;
  while (this->free_in.try_pop(buf));
  while (this->done_out.try_pop(buf));

  OmxBuffer *write_buf;
  while (this->to_write.try_pop(write_buf)) {
    free(write_buf);
  };

  if (this->downscale) {
    free(this->y_ptr2);
    free(this->u_ptr2);
    free(this->v_ptr2);
  }
}
