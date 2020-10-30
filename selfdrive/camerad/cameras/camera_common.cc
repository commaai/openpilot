#include <thread>
#include <stdio.h>
#include <signal.h>
#include <assert.h>
#include <unistd.h>

#if defined(QCOM) && !defined(QCOM_REPLAY)
#include "cameras/camera_qcom.h"
#elif QCOM2
#include "cameras/camera_qcom2.h"
#elif WEBCAM
#include "cameras/camera_webcam.h"
#else
#include "cameras/camera_frame_stream.h"
#endif

#include "camera_common.h"
#include <libyuv.h>
#include <jpeglib.h>

#include "clutil.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "imgproc/utils.h"

static cl_program build_debayer_program(cl_device_id device_id, cl_context context, const CameraInfo *ci, const CameraBuf *b) {
  char args[4096];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
           "-DFRAME_WIDTH=%d -DFRAME_HEIGHT=%d -DFRAME_STRIDE=%d "
           "-DRGB_WIDTH=%d -DRGB_HEIGHT=%d -DRGB_STRIDE=%d "
           "-DBAYER_FLIP=%d -DHDR=%d",
           ci->frame_width, ci->frame_height, ci->frame_stride,
           b->rgb_width, b->rgb_height, b->rgb_stride,
           ci->bayer_flip, ci->hdr);
#ifdef QCOM2
  return CLU_LOAD_FROM_FILE(context, device_id, "cameras/real_debayer.cl", args);
#else
  return CLU_LOAD_FROM_FILE(context, device_id, "cameras/debayer.cl", args);
#endif
}

void CameraBuf::init(cl_device_id device_id, cl_context context, CameraState *s, int frame_cnt,
                     const char *name, release_cb relase_callback) {
  const CameraInfo *ci = &s->ci;
  camera_state = s;
  frame_buf_count = frame_cnt;
  frame_size = ci->frame_height * ci->frame_stride;
 
  camera_bufs = std::make_unique<VisionBuf[]>(frame_buf_count);
  camera_bufs_metadata = std::make_unique<FrameMetadata[]>(frame_buf_count);
  for (int i = 0; i < frame_buf_count; i++) {
    camera_bufs[i] = visionbuf_allocate_cl(frame_size, device_id, context);
  }

  rgb_width = ci->frame_width;
  rgb_height = ci->frame_height;
#ifndef QCOM2
  // debayering does a 2x downscale
  if (ci->bayer) {
    rgb_width = ci->frame_width / 2;
    rgb_height = ci->frame_height / 2;
  }
  float db_s = 0.5;
#else
  float db_s = 1.0;
#endif

  if (ci->bayer) {
    yuv_transform = transform_scale_buffer(s->transform, db_s);
  } else {
    yuv_transform = s->transform;
  }

  for (int i = 0; i < UI_BUF_COUNT; i++) {
    VisionImg img = visionimg_alloc_rgb24(device_id, context, rgb_width, rgb_height, &rgb_bufs[i]);
    if (i == 0) {
      rgb_stride = img.stride;
    }
  }
  tbuffer_init(&ui_tb, UI_BUF_COUNT, name);
  tbuffer_init2(&camera_tb, frame_buf_count, "frame", relase_callback, s);

  // yuv back for recording and orbd
  pool_init(&yuv_pool, YUV_COUNT);
  yuv_tb = pool_get_tbuffer(&yuv_pool);

  yuv_width = rgb_width;
  yuv_height = rgb_height;
  yuv_buf_size = rgb_width * rgb_height * 3 / 2;

  for (int i = 0; i < YUV_COUNT; i++) {
    yuv_ion[i] = visionbuf_allocate_cl(yuv_buf_size, device_id, context);
    yuv_bufs[i].y = (uint8_t *)yuv_ion[i].addr;
    yuv_bufs[i].u = yuv_bufs[i].y + (yuv_width * yuv_height);
    yuv_bufs[i].v = yuv_bufs[i].u + (yuv_width / 2 * yuv_height / 2);
  }

  int err;
  if (ci->bayer) {
    cl_program prg_debayer = build_debayer_program(device_id, context, ci, this);
    krnl_debayer = clCreateKernel(prg_debayer, "debayer10", &err);
    assert(err == 0);
    assert(clReleaseProgram(prg_debayer) == 0);
  }

  rgb_to_yuv_init(&rgb_to_yuv_state, context, device_id, yuv_width, yuv_height, rgb_stride);

#ifdef __APPLE__
  q = clCreateCommandQueue(context, device_id, 0, &err);
#else
  const cl_queue_properties props[] = {0};  //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};
  q = clCreateCommandQueueWithProperties(context, device_id, props, &err);
#endif
  assert(err == 0);
}

CameraBuf::~CameraBuf() {
  for (int i = 0; i < frame_buf_count; i++) {
    visionbuf_free(&camera_bufs[i]);
  }
  for (int i = 0; i < UI_BUF_COUNT; i++) {
    visionbuf_free(&rgb_bufs[i]);
  }
  for (int i = 0; i < YUV_COUNT; i++) {
    visionbuf_free(&yuv_ion[i]);
  }
  clReleaseKernel(krnl_debayer);
  clReleaseCommandQueue(q);
}

bool CameraBuf::acquire() {
  const int buf_idx = tbuffer_acquire(&camera_tb);
  if (buf_idx < 0) {
    return false;
  }
  const FrameMetadata &frame_data = camera_bufs_metadata[buf_idx];
  if (frame_data.frame_id == -1) {
    LOGE("no frame data? wtf");
    tbuffer_release(&camera_tb, buf_idx);
    return false;
  }

  cur_frame_data = frame_data;

  cur_rgb_idx = tbuffer_select(&ui_tb);
  cur_rgb_buf = &rgb_bufs[cur_rgb_idx];

  cl_event debayer_event;
  cl_mem camrabuf_cl = camera_bufs[buf_idx].buf_cl;
  if (camera_state->ci.bayer) {
    assert(clSetKernelArg(krnl_debayer, 0, sizeof(cl_mem), &camrabuf_cl) == 0);
    assert(clSetKernelArg(krnl_debayer, 1, sizeof(cl_mem), &cur_rgb_buf->buf_cl) == 0);
#ifdef QCOM2
    assert(clSetKernelArg(krnl_debayer, 2, camera_state->debayer_cl_localMemSize, 0) == 0);
    assert(clEnqueueNDRangeKernel(q, krnl_debayer, 2, NULL,
                                  camera_state->debayer_cl_globalWorkSize, camera_state->debayer_cl_localWorkSize,
                                  0, 0, &debayer_event) == 0);
#else
    float digital_gain = camera_state->digital_gain;
    if ((int)digital_gain == 0) {
      digital_gain = 1.0;
    }
    assert(clSetKernelArg(krnl_debayer, 2, sizeof(float), &digital_gain) == 0);
    const size_t debayer_work_size = rgb_height;  // doesn't divide evenly, is this okay?
    const size_t debayer_local_work_size = 128;
    assert(clEnqueueNDRangeKernel(q, krnl_debayer, 1, NULL,
                                  &debayer_work_size, &debayer_local_work_size, 0, 0, &debayer_event) == 0);
#endif
  } else {
    assert(cur_rgb_buf->len >= frame_size);
    assert(rgb_stride == camera_state->ci.frame_stride);
    assert(clEnqueueCopyBuffer(q, camrabuf_cl, cur_rgb_buf->buf_cl, 0, 0,
                               cur_rgb_buf->len, 0, 0, &debayer_event) == 0);
  }

  clWaitForEvents(1, &debayer_event);
  clReleaseEvent(debayer_event);

  tbuffer_release(&camera_tb, buf_idx);
  visionbuf_sync(cur_rgb_buf, VISIONBUF_SYNC_FROM_DEVICE);

  cur_yuv_idx = pool_select(&yuv_pool);
  yuv_metas[cur_yuv_idx] = frame_data;
  rgb_to_yuv_queue(&rgb_to_yuv_state, q, cur_rgb_buf->buf_cl, yuv_ion[cur_yuv_idx].buf_cl);
  visionbuf_sync(&yuv_ion[cur_yuv_idx], VISIONBUF_SYNC_FROM_DEVICE);

  // keep another reference around till were done processing
  pool_acquire(&yuv_pool, cur_yuv_idx);
  pool_push(&yuv_pool, cur_yuv_idx);

  return true;
}

void CameraBuf::release() {
  tbuffer_dispatch(&ui_tb, cur_rgb_idx);
  pool_release(&yuv_pool, cur_yuv_idx);
}

void CameraBuf::stop() {
  tbuffer_stop(&ui_tb);
  tbuffer_stop(&camera_tb);
  pool_stop(&yuv_pool);
}

// common functions

void fill_frame_data(cereal::FrameData::Builder &framed, const FrameMetadata &frame_data, uint32_t cnt) {
  framed.setFrameId(frame_data.frame_id);
  framed.setEncodeId(cnt);
  framed.setTimestampEof(frame_data.timestamp_eof);
  framed.setFrameLength(frame_data.frame_length);
  framed.setIntegLines(frame_data.integ_lines);
  framed.setGlobalGain(frame_data.global_gain);
  framed.setLensPos(frame_data.lens_pos);
  framed.setLensSag(frame_data.lens_sag);
  framed.setLensErr(frame_data.lens_err);
  framed.setLensTruePos(frame_data.lens_true_pos);
  framed.setGainFrac(frame_data.gain_frac);
}

void create_thumbnail(MultiCameraState *s, CameraState *c, uint8_t *bgr_ptr) {
  const CameraBuf *b = &c->buf;

  uint8_t* thumbnail_buffer = NULL;
  unsigned long thumbnail_len = 0;

  unsigned char *row = (unsigned char *)malloc(b->rgb_width/4*3);

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &thumbnail_buffer, &thumbnail_len);

  cinfo.image_width = b->rgb_width / 4;
  cinfo.image_height = b->rgb_height / 4;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
#ifndef __APPLE__
  jpeg_set_quality(&cinfo, 50, true);
  jpeg_start_compress(&cinfo, true);
#else
  jpeg_set_quality(&cinfo, 50, static_cast<boolean>(true) );
  jpeg_start_compress(&cinfo, static_cast<boolean>(true) );
#endif

  JSAMPROW row_pointer[1];
  for (int ii = 0; ii < b->rgb_height/4; ii+=1) {
    for (int j = 0; j < b->rgb_width*3; j+=12) {
      for (int k = 0; k < 3; k++) {
        uint16_t dat = 0;
        int i = ii * 4;
        dat += bgr_ptr[b->rgb_stride*i + j + k];
        dat += bgr_ptr[b->rgb_stride*i + j+3 + k];
        dat += bgr_ptr[b->rgb_stride*(i+1) + j + k];
        dat += bgr_ptr[b->rgb_stride*(i+1) + j+3 + k];
        dat += bgr_ptr[b->rgb_stride*(i+2) + j + k];
        dat += bgr_ptr[b->rgb_stride*(i+2) + j+3 + k];
        dat += bgr_ptr[b->rgb_stride*(i+3) + j + k];
        dat += bgr_ptr[b->rgb_stride*(i+3) + j+3 + k];
        row[(j/4) + (2-k)] = dat/8;
      }
    }
    row_pointer[0] = row;
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }
  free(row);
  jpeg_finish_compress(&cinfo);

  MessageBuilder msg;
  auto thumbnaild = msg.initEvent().initThumbnail();
  thumbnaild.setFrameId(b->cur_frame_data.frame_id);
  thumbnaild.setTimestampEof(b->cur_frame_data.timestamp_eof);
  thumbnaild.setThumbnail(kj::arrayPtr((const uint8_t*)thumbnail_buffer, thumbnail_len));

  if (s->pm != NULL) {
    s->pm->send("thumbnail", msg);
  }
}

void set_exposure_target(CameraState *c, const uint8_t *pix_ptr, int x_start, int x_end, int x_skip, int y_start, int y_end, int y_skip) {
  const CameraBuf *b = &c->buf;

  uint32_t lum_binning[256] = {0};
  for (int y = y_start; y < y_end; y += y_skip) {
    for (int x = x_start; x < x_end; x += x_skip) {
      uint8_t lum = pix_ptr[(y * b->yuv_width) + x];
      lum_binning[lum]++;
    }
  }

  unsigned int lum_total = (y_end - y_start) * (x_end - x_start) / x_skip / y_skip;
  unsigned int lum_cur = 0;
  int lum_med = 0;
  int lum_med_alt = 0;
  for (lum_med=255; lum_med>=0; lum_med--) {
    lum_cur += lum_binning[lum_med];
#ifdef QCOM2
    bool reach_hlc_perc = false;
    if (c->camera_num == 0) { // wide
      reach_hlc_perc = lum_cur > 2*lum_total / (3*HLC_A);
    } else {
      reach_hlc_perc = lum_cur > lum_total / HLC_A;
    }
    if (reach_hlc_perc && lum_med > HLC_THRESH) {
      lum_med_alt = 86;
    }
#endif
    if (lum_cur >= lum_total / 2) {
      break;
    }
  }
  lum_med = lum_med_alt>lum_med?lum_med_alt:lum_med;
  camera_autoexposure(c, lum_med / 256.0);
}

extern volatile sig_atomic_t do_exit;

void *processing_thread(MultiCameraState *cameras, const char *tname,
                        CameraState *cs, int priority, process_thread_cb callback) {
  set_thread_name(tname);
  int err = set_realtime_priority(priority);
  LOG("%s start! setpriority returns %d", tname, err);

  for (int cnt = 0; !do_exit; cnt++) {
    if (!cs->buf.acquire()) continue;

    callback(cameras, cs, cnt);

    cs->buf.release();
  }
  return NULL;
}

std::thread start_process_thread(MultiCameraState *cameras, const char *tname,
                                 CameraState *cs, int priority, process_thread_cb callback) {
  return std::thread(processing_thread, cameras, tname, cs, priority, callback);
}

void common_camera_process_front(SubMaster *sm, PubMaster *pm, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;

  static int meteringbox_xmin = 0, meteringbox_xmax = 0;
  static int meteringbox_ymin = 0, meteringbox_ymax = 0;
  static const bool rhd_front = Params().read_db_bool("IsRHD");

  sm->update(0);
  if (sm->updated("driverState")) {
    auto state = (*sm)["driverState"].getDriverState();
    float face_prob = state.getFaceProb();
    float face_position[2];
    face_position[0] = state.getFacePosition()[0];
    face_position[1] = state.getFacePosition()[1];

    // set front camera metering target
    if (face_prob > 0.4) {
      int x_offset = rhd_front ? 0:b->rgb_width - 0.5 * b->rgb_height;
      meteringbox_xmin = x_offset + (face_position[0] + 0.5) * (0.5 * b->rgb_height) - 72;
      meteringbox_xmax = x_offset + (face_position[0] + 0.5) * (0.5 * b->rgb_height) + 72;
      meteringbox_ymin = (face_position[1] + 0.5) * (b->rgb_height) - 72;
      meteringbox_ymax = (face_position[1] + 0.5) * (b->rgb_height) + 72;
    } else { // use default setting if no face
      meteringbox_ymin = b->rgb_height * 1 / 3;
      meteringbox_ymax = b->rgb_height * 1;
      meteringbox_xmin = rhd_front ? 0:b->rgb_width * 3 / 5;
      meteringbox_xmax = rhd_front ? b->rgb_width * 2 / 5:b->rgb_width;
    }
  }

  // auto exposure
  if (cnt % 3 == 0) {
    // use driver face crop for AE
    int x_start, x_end, y_start, y_end;
    int skip = 1;

    if (meteringbox_xmax > 0) {
      x_start = std::max(0, meteringbox_xmin);
      x_end = std::min(b->rgb_width - 1, meteringbox_xmax);
      y_start = std::max(0, meteringbox_ymin);
      y_end = std::min(b->rgb_height - 1, meteringbox_ymax);
    } else {
      y_start = b->rgb_height * 1 / 3;
      y_end = b->rgb_height * 1;
      x_start = rhd_front ? 0 : b->rgb_width * 3 / 5;
      x_end = rhd_front ? b->rgb_width * 2 / 5 : b->rgb_width;
    }
#ifdef QCOM2
    x_start = 96;
    x_end = 1832;
    y_start = 242;
    y_end = 1148;
    skip = 4;
#endif
    set_exposure_target(c, (const uint8_t *)b->yuv_bufs[b->cur_yuv_idx].y, x_start, x_end, 2, y_start, y_end, skip);
  }

  MessageBuilder msg;
  auto framed = msg.initEvent().initFrontFrame();
  framed.setFrameType(cereal::FrameData::FrameType::FRONT);
  fill_frame_data(framed, b->cur_frame_data, cnt);
  pm->send("frontFrame", msg);
}
