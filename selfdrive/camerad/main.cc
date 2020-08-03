#include <stdio.h>
#include <signal.h>
#include <cassert>

#if defined(QCOM) && !defined(QCOM_REPLAY)
#include "cameras/camera_qcom.h"
#elif QCOM2
#include "cameras/camera_qcom2.h"
#elif WEBCAM
#include "cameras/camera_webcam.h"
#else
#include "cameras/camera_frame_stream.h"
#endif

#include "common/util.h"
#include "common/swaglog.h"

#include "common/ipc.h"
#include "common/visionipc.h"
#include "common/visionbuf.h"
#include "common/visionimg.h"

#include "messaging.hpp"

#include "transforms/rgb_to_yuv.h"
#include "imgproc/utils.h"

#include "clutil.h"
#include "bufs.h"

#include <libyuv.h>
#include <czmq.h>
#include <jpeglib.h>

#define UI_BUF_COUNT 4
#define YUV_COUNT 40
#define MAX_CLIENTS 5

extern "C" {
volatile sig_atomic_t do_exit = 0;
}

void set_do_exit(int sig) {
  do_exit = 1;
}

struct VisionState;

struct VisionClientState {
  VisionState *s;
  int fd;
  pthread_t thread_handle;
  bool running;
};

struct VisionClientStreamState {
  bool subscribed;
  int bufs_outstanding;
  bool tb;
  TBuffer* tbuffer;
  PoolQueue* queue;
};

struct VisionState {
  int frame_width, frame_height;
  int frame_stride;
  int frame_size;

  int ion_fd;

  // cl state
  cl_device_id device_id;
  cl_context context;

  cl_program prg_debayer_rear;
  cl_program prg_debayer_front;
  cl_kernel krnl_debayer_rear;
  cl_kernel krnl_debayer_front;

  cl_program prg_rgb_laplacian;
  cl_kernel krnl_rgb_laplacian;

  int conv_cl_localMemSize;
  size_t conv_cl_globalWorkSize[2];
  size_t conv_cl_localWorkSize[2];
  size_t pool_cl_globalWorkSize[2];

  // processing
  TBuffer ui_tb;
  TBuffer ui_front_tb;

  mat3 yuv_transform;
  TBuffer *yuv_tb;
  TBuffer *yuv_front_tb;

  // TODO: refactor for both cameras?
  Pool yuv_pool;
  VisionBuf yuv_ion[YUV_COUNT];
  cl_mem yuv_cl[YUV_COUNT];
  YUVBuf yuv_bufs[YUV_COUNT];
  FrameMetadata yuv_metas[YUV_COUNT];
  size_t yuv_buf_size;
  int yuv_width, yuv_height;
  RGBToYUVState rgb_to_yuv_state;

  // for front camera recording
  Pool yuv_front_pool;
  VisionBuf yuv_front_ion[YUV_COUNT];
  cl_mem yuv_front_cl[YUV_COUNT];
  YUVBuf yuv_front_bufs[YUV_COUNT];
  FrameMetadata yuv_front_metas[YUV_COUNT];
  size_t yuv_front_buf_size;
  int yuv_front_width, yuv_front_height;
  RGBToYUVState front_rgb_to_yuv_state;

  size_t rgb_buf_size;
  int rgb_width, rgb_height, rgb_stride;
  VisionBuf rgb_bufs[UI_BUF_COUNT];
  cl_mem rgb_bufs_cl[UI_BUF_COUNT];
  cl_mem rgb_conv_roi_cl, rgb_conv_result_cl, rgb_conv_filter_cl;
  uint16_t lapres[(ROI_X_MAX-ROI_X_MIN+1)*(ROI_Y_MAX-ROI_Y_MIN+1)];

  size_t rgb_front_buf_size;
  int rgb_front_width, rgb_front_height, rgb_front_stride;
  VisionBuf rgb_front_bufs[UI_BUF_COUNT];
  cl_mem rgb_front_bufs_cl[UI_BUF_COUNT];
  bool rhd_front;
  bool rhd_front_checked;
  int front_meteringbox_xmin, front_meteringbox_xmax;
  int front_meteringbox_ymin, front_meteringbox_ymax;


  cl_mem camera_bufs_cl[FRAME_BUF_COUNT];
  VisionBuf camera_bufs[FRAME_BUF_COUNT];
  VisionBuf focus_bufs[FRAME_BUF_COUNT];
  VisionBuf stats_bufs[FRAME_BUF_COUNT];

  cl_mem front_camera_bufs_cl[FRAME_BUF_COUNT];
  VisionBuf front_camera_bufs[FRAME_BUF_COUNT];

  DualCameraState cameras;

  zsock_t *terminate_pub;

  PubMaster *pm;

  pthread_mutex_t clients_lock;
  VisionClientState clients[MAX_CLIENTS];
};

// frontview thread
void* frontview_thread(void *arg) {
  int err;
  VisionState *s = (VisionState*)arg;

  set_thread_name("frontview");
  // we subscribe to this for placement of the AE metering box
  // TODO: the loop is bad, ideally models shouldn't affect sensors
  SubMaster sm({"driverState", "dMonitoringState"});

  cl_command_queue q = clCreateCommandQueue(s->context, s->device_id, 0, &err);
  assert(err == 0);

  for (int cnt = 0; !do_exit; cnt++) {
    int buf_idx = tbuffer_acquire(&s->cameras.front.camera_tb);
    if (buf_idx < 0) {
      break;
    }

    int ui_idx = tbuffer_select(&s->ui_front_tb);
    int rgb_idx = ui_idx;
    FrameMetadata frame_data = s->cameras.front.camera_bufs_metadata[buf_idx];

    //double t1 = millis_since_boot();

    cl_event debayer_event;
    if (s->cameras.front.ci.bayer) {
      err = clSetKernelArg(s->krnl_debayer_front, 0, sizeof(cl_mem), &s->front_camera_bufs_cl[buf_idx]);
      assert(err == 0);
      err = clSetKernelArg(s->krnl_debayer_front, 1, sizeof(cl_mem), &s->rgb_front_bufs_cl[rgb_idx]);
      assert(err == 0);
      float digital_gain = 1.0;
      err = clSetKernelArg(s->krnl_debayer_front, 2, sizeof(float), &digital_gain);
      assert(err == 0);

      const size_t debayer_work_size = s->rgb_front_height; // doesn't divide evenly, is this okay?
      const size_t debayer_local_work_size = 128;
      err = clEnqueueNDRangeKernel(q, s->krnl_debayer_front, 1, NULL,
                                   &debayer_work_size, &debayer_local_work_size, 0, 0, &debayer_event);
      assert(err == 0);
    } else {
      assert(s->rgb_front_buf_size >= s->cameras.front.frame_size);
      assert(s->rgb_front_stride == s->cameras.front.ci.frame_stride);
      err = clEnqueueCopyBuffer(q, s->front_camera_bufs_cl[buf_idx], s->rgb_front_bufs_cl[rgb_idx],
                                0, 0, s->rgb_front_buf_size, 0, 0, &debayer_event);
      assert(err == 0);
    }
    clWaitForEvents(1, &debayer_event);
    clReleaseEvent(debayer_event);
    tbuffer_release(&s->cameras.front.camera_tb, buf_idx);
    visionbuf_sync(&s->rgb_front_bufs[ui_idx], VISIONBUF_SYNC_FROM_DEVICE);

    sm.update(0);
    // no more check after gps check
    if (!s->rhd_front_checked && sm.updated("dMonitoringState")) {
      auto state = sm["dMonitoringState"].getDMonitoringState();
      s->rhd_front = state.getIsRHD();
      s->rhd_front_checked = state.getRhdChecked();
    }

    if (sm.updated("driverState")) {
      auto state = sm["driverState"].getDriverState();
      float face_prob = state.getFaceProb();
      float face_position[2];
      face_position[0] = state.getFacePosition()[0];
      face_position[1] = state.getFacePosition()[1];

      // set front camera metering target
      if (face_prob > 0.4) {
        int x_offset = s->rhd_front ? 0:s->rgb_front_width - 0.5 * s->rgb_front_height;
        s->front_meteringbox_xmin = x_offset + (face_position[0] + 0.5) * (0.5 * s->rgb_front_height) - 72;
        s->front_meteringbox_xmax = x_offset + (face_position[0] + 0.5) * (0.5 * s->rgb_front_height) + 72;
        s->front_meteringbox_ymin = (face_position[1] + 0.5) * (s->rgb_front_height) - 72;
        s->front_meteringbox_ymax = (face_position[1] + 0.5) * (s->rgb_front_height) + 72;
      } else {// use default setting if no face
        s->front_meteringbox_ymin = s->rgb_front_height * 1 / 3;
        s->front_meteringbox_ymax = s->rgb_front_height * 1;
        s->front_meteringbox_xmin = s->rhd_front ? 0:s->rgb_front_width * 3 / 5;
        s->front_meteringbox_xmax = s->rhd_front ? s->rgb_front_width * 2 / 5:s->rgb_front_width;
      }
    }

    // auto exposure
    const uint8_t *bgr_front_ptr = (const uint8_t*)s->rgb_front_bufs[ui_idx].addr;
#ifndef DEBUG_DRIVER_MONITOR
    if (cnt % 3 == 0)
#endif
    {
      // use driver face crop for AE
      int x_start;
      int x_end;
      int y_start;
      int y_end;

      if (s->front_meteringbox_xmax > 0)
      {
        x_start = s->front_meteringbox_xmin<0 ? 0:s->front_meteringbox_xmin;
        x_end = s->front_meteringbox_xmax>=s->rgb_front_width ? s->rgb_front_width-1:s->front_meteringbox_xmax;
        y_start = s->front_meteringbox_ymin<0 ? 0:s->front_meteringbox_ymin;
        y_end = s->front_meteringbox_ymax>=s->rgb_front_height ? s->rgb_front_height-1:s->front_meteringbox_ymax;
      }
      else
      {
        y_start = s->rgb_front_height * 1 / 3;
        y_end = s->rgb_front_height * 1;
        x_start = s->rhd_front ? 0:s->rgb_front_width * 3 / 5;
        x_end = s->rhd_front ? s->rgb_front_width * 2 / 5:s->rgb_front_width;
      }

      uint32_t lum_binning[256] = {0,};
      for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; x += 2) { // every 2nd col
          const uint8_t *pix = &bgr_front_ptr[y * s->rgb_front_stride + x * 3];
          unsigned int lum = (unsigned int)pix[0] + pix[1] + pix[2];
#ifdef DEBUG_DRIVER_MONITOR
          uint8_t *pix_rw = (uint8_t *)pix;

          // set all the autoexposure pixels to pure green (pixel format is bgr)
          pix_rw[0] = pix_rw[2] = 0;
          pix_rw[1] = 0xff;
#endif
          lum_binning[std::min(lum / 3, 255u)]++;
        }
      }
      const unsigned int lum_total = (y_end - y_start) * (x_end - x_start)/2;
      unsigned int lum_cur = 0;
      int lum_med = 0;
      for (lum_med=0; lum_med<256; lum_med++) {
        lum_cur += lum_binning[lum_med];
        if (lum_cur >= lum_total / 2) {
          break;
        }
      }
      camera_autoexposure(&s->cameras.front, lum_med / 256.0);
    }

    // push YUV buffer
    int yuv_idx = pool_select(&s->yuv_front_pool);
    s->yuv_front_metas[yuv_idx] = frame_data;

    rgb_to_yuv_queue(&s->front_rgb_to_yuv_state, q, s->rgb_front_bufs_cl[ui_idx], s->yuv_front_cl[yuv_idx]);
    visionbuf_sync(&s->yuv_front_ion[yuv_idx], VISIONBUF_SYNC_FROM_DEVICE);
    s->yuv_front_metas[yuv_idx] = frame_data;

    // no reference required cause we don't use this in visiond
    //pool_acquire(&s->yuv_front_pool, yuv_idx);
    pool_push(&s->yuv_front_pool, yuv_idx);
    //pool_release(&s->yuv_front_pool, yuv_idx);

    // send frame event
    {
      if (s->pm != NULL) {
        capnp::MallocMessageBuilder msg;
        cereal::Event::Builder event = msg.initRoot<cereal::Event>();
        event.setLogMonoTime(nanos_since_boot());

        auto framed = event.initFrontFrame();
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
        framed.setFrameType(cereal::FrameData::FrameType::FRONT);

        s->pm->send("frontFrame", msg);
      }
    }

    /*FILE *f = fopen("/tmp/test2", "wb");
    printf("%d %d\n", s->rgb_front_height, s->rgb_front_stride);
    fwrite(bgr_front_ptr, 1, s->rgb_front_stride * s->rgb_front_height, f);
    fclose(f);*/

    tbuffer_dispatch(&s->ui_front_tb, ui_idx);

    //double t2 = millis_since_boot();
    //LOGD("front process: %.2fms", t2-t1);
  }

  return NULL;
}
// processing
void* processing_thread(void *arg) {
  int err;
  VisionState *s = (VisionState*)arg;

  set_thread_name("processing");

  err = set_realtime_priority(51);
  LOG("setpriority returns %d", err);

  // init cl stuff
#ifdef __APPLE__
  cl_command_queue q = clCreateCommandQueue(s->context, s->device_id, 0, &err);
#else
  const cl_queue_properties props[] = {0}; //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};
  cl_command_queue q = clCreateCommandQueueWithProperties(s->context, s->device_id, props, &err);
#endif
  assert(err == 0);

  // init the net
  LOG("processing start!");

  for (int cnt = 0; !do_exit; cnt++) {
    int buf_idx = tbuffer_acquire(&s->cameras.rear.camera_tb);
    // int buf_idx = camera_acquire_buffer(s);
    if (buf_idx < 0) {
      break;
    }

    double t1 = millis_since_boot();

    FrameMetadata frame_data = s->cameras.rear.camera_bufs_metadata[buf_idx];
    uint32_t frame_id = frame_data.frame_id;

    if (frame_id == -1) {
      LOGE("no frame data? wtf");
      tbuffer_release(&s->cameras.rear.camera_tb, buf_idx);
      continue;
    }

    int ui_idx = tbuffer_select(&s->ui_tb);
    int rgb_idx = ui_idx;

    cl_event debayer_event;
    if (s->cameras.rear.ci.bayer) {
      err = clSetKernelArg(s->krnl_debayer_rear, 0, sizeof(cl_mem), &s->camera_bufs_cl[buf_idx]);
      assert(err == 0);
      err = clSetKernelArg(s->krnl_debayer_rear, 1, sizeof(cl_mem), &s->rgb_bufs_cl[rgb_idx]);
      assert(err == 0);
      err = clSetKernelArg(s->krnl_debayer_rear, 2, sizeof(float), &s->cameras.rear.digital_gain);
      assert(err == 0);

      const size_t debayer_work_size = s->rgb_height; // doesn't divide evenly, is this okay?
      const size_t debayer_local_work_size = 128;
      err = clEnqueueNDRangeKernel(q, s->krnl_debayer_rear, 1, NULL,
                                   &debayer_work_size, &debayer_local_work_size, 0, 0, &debayer_event);
      assert(err == 0);
    } else {
      assert(s->rgb_buf_size >= s->frame_size);
      assert(s->rgb_stride == s->frame_stride);
      err = clEnqueueCopyBuffer(q, s->camera_bufs_cl[buf_idx], s->rgb_bufs_cl[rgb_idx],
                                0, 0, s->rgb_buf_size, 0, 0, &debayer_event);
      assert(err == 0);
    }

    clWaitForEvents(1, &debayer_event);
    clReleaseEvent(debayer_event);

    tbuffer_release(&s->cameras.rear.camera_tb, buf_idx);

    visionbuf_sync(&s->rgb_bufs[rgb_idx], VISIONBUF_SYNC_FROM_DEVICE);

#if defined(QCOM) && !defined(QCOM_REPLAY)
    /*FILE *dump_rgb_file = fopen("/tmp/process_dump.rgb", "wb");
    fwrite(s->rgb_bufs[rgb_idx].addr, s->rgb_bufs[rgb_idx].len, sizeof(uint8_t), dump_rgb_file);
    fclose(dump_rgb_file);
    printf("ORIGINAL SAVED!!\n");*/

    /*double t10 = millis_since_boot();*/

    // cache rgb roi and write to cl
    uint8_t *rgb_roi_buf = new uint8_t[(s->rgb_width/NUM_SEGMENTS_X)*(s->rgb_height/NUM_SEGMENTS_Y)*3];
    int roi_id = cnt % ((ROI_X_MAX-ROI_X_MIN+1)*(ROI_Y_MAX-ROI_Y_MIN+1)); // rolling roi
    int roi_x_offset = roi_id % (ROI_X_MAX-ROI_X_MIN+1);
    int roi_y_offset = roi_id / (ROI_X_MAX-ROI_X_MIN+1);

    for (int r=0;r<(s->rgb_height/NUM_SEGMENTS_Y);r++) {
      memcpy(rgb_roi_buf + r * (s->rgb_width/NUM_SEGMENTS_X) * 3,
              (uint8_t *) s->rgb_bufs[rgb_idx].addr + \
                (ROI_Y_MIN + roi_y_offset) * s->rgb_height/NUM_SEGMENTS_Y * FULL_STRIDE_X * 3 + \
                (ROI_X_MIN + roi_x_offset) * s->rgb_width/NUM_SEGMENTS_X * 3 + r * FULL_STRIDE_X * 3,
              s->rgb_width/NUM_SEGMENTS_X * 3);
    }

    err = clEnqueueWriteBuffer (q, s->rgb_conv_roi_cl, true, 0,
        s->rgb_width/NUM_SEGMENTS_X * s->rgb_height/NUM_SEGMENTS_Y * 3 * sizeof(uint8_t), rgb_roi_buf, 0, 0, 0);
    assert(err == 0);

    /*double t11 = millis_since_boot();
    printf("cache time: %f ms\n", t11 - t10);
    t10 = millis_since_boot();*/

    err = clSetKernelArg(s->krnl_rgb_laplacian, 0, sizeof(cl_mem), (void *) &s->rgb_conv_roi_cl);
    assert(err == 0);
    err = clSetKernelArg(s->krnl_rgb_laplacian, 1, sizeof(cl_mem), (void *) &s->rgb_conv_result_cl);
    assert(err == 0);
    err = clSetKernelArg(s->krnl_rgb_laplacian, 2, sizeof(cl_mem), (void *) &s->rgb_conv_filter_cl);
    assert(err == 0);
    err = clSetKernelArg(s->krnl_rgb_laplacian, 3, s->conv_cl_localMemSize, 0);
    assert(err == 0);

    cl_event conv_event;
    err = clEnqueueNDRangeKernel(q, s->krnl_rgb_laplacian, 2, NULL,
                                   s->conv_cl_globalWorkSize, s->conv_cl_localWorkSize, 0, 0, &conv_event);
    assert(err == 0);
    clWaitForEvents(1, &conv_event);
    clReleaseEvent(conv_event);

    int16_t *conv_result = new int16_t[(s->rgb_width/NUM_SEGMENTS_X)*(s->rgb_height/NUM_SEGMENTS_Y)];
    err = clEnqueueReadBuffer(q, s->rgb_conv_result_cl, true, 0,
       s->rgb_width/NUM_SEGMENTS_X * s->rgb_height/NUM_SEGMENTS_Y * sizeof(int16_t), conv_result, 0, 0, 0);
    assert(err == 0);

    /*t11 = millis_since_boot();
    printf("conv time: %f ms\n", t11 - t10);
    t10 = millis_since_boot();*/

    get_lapmap_one(conv_result, &s->lapres[roi_id], s->rgb_width/NUM_SEGMENTS_X, s->rgb_height/NUM_SEGMENTS_Y);

    /*t11 = millis_since_boot();
    printf("pool time: %f ms\n", t11 - t10);
    t10 = millis_since_boot();*/

    delete [] rgb_roi_buf;
    delete [] conv_result;

    /*t11 = millis_since_boot();
    printf("process time: %f ms\n ----- \n", t11 - t10);
    t10 = millis_since_boot();*/

    // setup self recover
    if (is_blur(&s->lapres[0]) &&
       (s->cameras.rear.lens_true_pos < (s->cameras.device == DEVICE_LP3? LP3_AF_DAC_DOWN:OP3T_AF_DAC_DOWN)+1 ||
        s->cameras.rear.lens_true_pos > (s->cameras.device == DEVICE_LP3? LP3_AF_DAC_UP:OP3T_AF_DAC_UP)-1) &&
       s->cameras.rear.self_recover < 2) {
      // truly stuck, needs help
      s->cameras.rear.self_recover -= 1;
      if (s->cameras.rear.self_recover < -FOCUS_RECOVER_PATIENCE) {
        LOGW("rear camera bad state detected. attempting recovery from %.1f, recover state is %d",
                                      s->cameras.rear.lens_true_pos, s->cameras.rear.self_recover);
        s->cameras.rear.self_recover = FOCUS_RECOVER_STEPS + ((s->cameras.rear.lens_true_pos < (s->cameras.device == DEVICE_LP3? LP3_AF_DAC_M:OP3T_AF_DAC_M))?1:0); // parity determined by which end is stuck at
      }
    } else if ((s->cameras.rear.lens_true_pos < (s->cameras.device == DEVICE_LP3? LP3_AF_DAC_M - LP3_AF_DAC_3SIG:OP3T_AF_DAC_M - OP3T_AF_DAC_3SIG) ||
               s->cameras.rear.lens_true_pos > (s->cameras.device == DEVICE_LP3? LP3_AF_DAC_M + LP3_AF_DAC_3SIG:OP3T_AF_DAC_M + OP3T_AF_DAC_3SIG)) &&
              s->cameras.rear.self_recover < 2) {
      // in suboptimal position with high prob, but may still recover by itself
      s->cameras.rear.self_recover -= 1;
      if (s->cameras.rear.self_recover < -(FOCUS_RECOVER_PATIENCE*3)) {
        LOGW("rear camera bad state detected. attempting recovery from %.1f, recover state is %d", s->cameras.rear.lens_true_pos, s->cameras.rear.self_recover);
        s->cameras.rear.self_recover = FOCUS_RECOVER_STEPS/2 + ((s->cameras.rear.lens_true_pos < (s->cameras.device == DEVICE_LP3? LP3_AF_DAC_M:OP3T_AF_DAC_M))?1:0);
      }
    } else if (s->cameras.rear.self_recover < 0) {
      s->cameras.rear.self_recover += 1; // reset if fine
    }

#endif

    double t2 = millis_since_boot();

#ifndef QCOM2
    uint8_t *bgr_ptr = (uint8_t*)s->rgb_bufs[rgb_idx].addr;
#endif

    double yt1 = millis_since_boot();

    int yuv_idx = pool_select(&s->yuv_pool);

    s->yuv_metas[yuv_idx] = frame_data;

    uint8_t* yuv_ptr_y = s->yuv_bufs[yuv_idx].y;
    cl_mem yuv_cl = s->yuv_cl[yuv_idx];
    rgb_to_yuv_queue(&s->rgb_to_yuv_state, q, s->rgb_bufs_cl[rgb_idx], yuv_cl);
    visionbuf_sync(&s->yuv_ion[yuv_idx], VISIONBUF_SYNC_FROM_DEVICE);

    double yt2 = millis_since_boot();

    // keep another reference around till were done processing
    pool_acquire(&s->yuv_pool, yuv_idx);
    pool_push(&s->yuv_pool, yuv_idx);

    // send frame event
    {
      if (s->pm != NULL) {
        capnp::MallocMessageBuilder msg;
        cereal::Event::Builder event = msg.initRoot<cereal::Event>();
        event.setLogMonoTime(nanos_since_boot());

        auto framed = event.initFrame();
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

#if defined(QCOM) && !defined(QCOM_REPLAY)
        kj::ArrayPtr<const int16_t> focus_vals(&s->cameras.rear.focus[0], NUM_FOCUS);
        kj::ArrayPtr<const uint8_t> focus_confs(&s->cameras.rear.confidence[0], NUM_FOCUS);
        framed.setFocusVal(focus_vals);
        framed.setFocusConf(focus_confs);
        kj::ArrayPtr<const uint16_t> sharpness_score(&s->lapres[0], (ROI_X_MAX-ROI_X_MIN+1)*(ROI_Y_MAX-ROI_Y_MIN+1));
        framed.setSharpnessScore(sharpness_score);
        framed.setRecoverState(s->cameras.rear.self_recover);
#endif

// TODO: add this back
#if !defined(QCOM) && !defined(QCOM2)
//#ifndef QCOM
        framed.setImage(kj::arrayPtr((const uint8_t*)s->yuv_ion[yuv_idx].addr, s->yuv_buf_size));
#endif

        kj::ArrayPtr<const float> transform_vs(&s->yuv_transform.v[0], 9);
        framed.setTransform(transform_vs);

        s->pm->send("frame", msg);
      }
    }

#ifndef QCOM2
    // TODO: fix on QCOM2, giving scanline error
    // one thumbnail per 5 seconds (instead of %5 == 0 posenet)
    if (cnt % 100 == 3) {
      uint8_t* thumbnail_buffer = NULL;
      unsigned long thumbnail_len = 0;

      unsigned char *row = (unsigned char *)malloc(s->rgb_width/4*3);

      struct jpeg_compress_struct cinfo;
      struct jpeg_error_mgr jerr;

      cinfo.err = jpeg_std_error(&jerr);
      jpeg_create_compress(&cinfo);
      jpeg_mem_dest(&cinfo, &thumbnail_buffer, &thumbnail_len);

      cinfo.image_width = s->rgb_width / 4;
      cinfo.image_height = s->rgb_height / 4;
      cinfo.input_components = 3;
      cinfo.in_color_space = JCS_RGB;

      jpeg_set_defaults(&cinfo);
      jpeg_set_quality(&cinfo, 50, true);
      jpeg_start_compress(&cinfo, true);

      JSAMPROW row_pointer[1];
      for (int i = 0; i < s->rgb_height - 4; i+=4) {
        for (int j = 0; j < s->rgb_width*3; j+=12) {
          for (int k = 0; k < 3; k++) {
            uint16_t dat = 0;
            dat += bgr_ptr[s->rgb_stride*i + j + k];
            dat += bgr_ptr[s->rgb_stride*i + j+3 + k];
            dat += bgr_ptr[s->rgb_stride*(i+1) + j + k];
            dat += bgr_ptr[s->rgb_stride*(i+1) + j+3 + k];
            dat += bgr_ptr[s->rgb_stride*(i+2) + j + k];
            dat += bgr_ptr[s->rgb_stride*(i+2) + j+3 + k];
            dat += bgr_ptr[s->rgb_stride*(i+3) + j + k];
            dat += bgr_ptr[s->rgb_stride*(i+3) + j+3 + k];

            row[(j/4) + (2-k)] = dat/8;
          }
        }
        row_pointer[0] = row;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
      }
      free(row);
      jpeg_finish_compress(&cinfo);

      capnp::MallocMessageBuilder msg;
      cereal::Event::Builder event = msg.initRoot<cereal::Event>();
      event.setLogMonoTime(nanos_since_boot());

      auto thumbnaild = event.initThumbnail();
      thumbnaild.setFrameId(frame_data.frame_id);
      thumbnaild.setTimestampEof(frame_data.timestamp_eof);
      thumbnaild.setThumbnail(kj::arrayPtr((const uint8_t*)thumbnail_buffer, thumbnail_len));

      if (s->pm != NULL) {
        s->pm->send("thumbnail", msg);
      }

      free(thumbnail_buffer);
    }
#endif

    tbuffer_dispatch(&s->ui_tb, ui_idx);

    // auto exposure over big box
    const int exposure_x = 290;
    const int exposure_y = 282 + 40;
    const int exposure_height = 314;
    const int exposure_width = 560;
    if (cnt % 3 == 0) {
      // find median box luminance for AE
      uint32_t lum_binning[256] = {0,};
      for (int y=0; y<exposure_height; y++) {
        for (int x=0; x<exposure_width; x++) {
          uint8_t lum = yuv_ptr_y[((exposure_y+y)*s->yuv_width) + exposure_x + x];
          lum_binning[lum]++;
        }
      }
      const unsigned int lum_total = exposure_height * exposure_width;
      unsigned int lum_cur = 0;
      int lum_med = 0;
      for (lum_med=0; lum_med<256; lum_med++) {
        // shouldn't be any values less than 16 - yuv footroom
        lum_cur += lum_binning[lum_med];
        if (lum_cur >= lum_total / 2) {
          break;
        }
      }
      // double avg = (double)acc / (big_box_width * big_box_height) - 16;
      // printf("avg %d\n", lum_med);

      camera_autoexposure(&s->cameras.rear, lum_med / 256.0);
    }

    pool_release(&s->yuv_pool, yuv_idx);
    double t5 = millis_since_boot();
    LOGD("queued: %.2fms, yuv: %.2f, | processing: %.3fms", (t2-t1), (yt2-yt1), (t5-t1));
  }

  return NULL;
}

// visionserver
void* visionserver_client_thread(void* arg) {
  int err;
  VisionClientState *client = (VisionClientState*)arg;
  VisionState *s = client->s;
  int fd = client->fd;

  set_thread_name("clientthread");

  zsock_t *terminate = zsock_new_sub(">inproc://terminate", "");
  assert(terminate);
  void* terminate_raw = zsock_resolve(terminate);

  VisionClientStreamState streams[VISION_STREAM_MAX] = {{0}};

  LOGW("client start fd %d", fd);

  while (true) {
    zmq_pollitem_t polls[2+VISION_STREAM_MAX] = {{0}};
    polls[0].socket = terminate_raw;
    polls[0].events = ZMQ_POLLIN;
    polls[1].fd = fd;
    polls[1].events = ZMQ_POLLIN;

    int poll_to_stream[2+VISION_STREAM_MAX] = {0};
    int num_polls = 2;
    for (int i=0; i<VISION_STREAM_MAX; i++) {
      if (!streams[i].subscribed) continue;
      polls[num_polls].events = ZMQ_POLLIN;
      if (streams[i].bufs_outstanding >= 2) {
        continue;
      }
      if (streams[i].tb) {
        polls[num_polls].fd = tbuffer_efd(streams[i].tbuffer);
      } else {
        polls[num_polls].fd = poolq_efd(streams[i].queue);
      }
      poll_to_stream[num_polls] = i;
      num_polls++;
    }
    int ret = zmq_poll(polls, num_polls, -1);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }
    if (polls[0].revents) {
      break;
    } else if (polls[1].revents) {
      VisionPacket p;
      err = vipc_recv(fd, &p);
      // printf("recv %d\n", p.type);
      if (err <= 0) {
        break;
      } else if (p.type == VIPC_STREAM_SUBSCRIBE) {
        VisionStreamType stream_type = p.d.stream_sub.type;
        VisionPacket rep = {
          .type = VIPC_STREAM_BUFS,
          .d = { .stream_bufs = { .type = stream_type }, },
        };

        VisionClientStreamState *stream = &streams[stream_type];
        stream->tb = p.d.stream_sub.tbuffer;

        VisionStreamBufs *stream_bufs = &rep.d.stream_bufs;
        if (stream_type == VISION_STREAM_RGB_BACK) {
          stream_bufs->width = s->rgb_width;
          stream_bufs->height = s->rgb_height;
          stream_bufs->stride = s->rgb_stride;
          stream_bufs->buf_len = s->rgb_bufs[0].len;
          rep.num_fds = UI_BUF_COUNT;
          for (int i=0; i<rep.num_fds; i++) {
            rep.fds[i] = s->rgb_bufs[i].fd;
          }
          if (stream->tb) {
            stream->tbuffer = &s->ui_tb;
          } else {
            assert(false);
          }
        } else if (stream_type == VISION_STREAM_RGB_FRONT) {
          stream_bufs->width = s->rgb_front_width;
          stream_bufs->height = s->rgb_front_height;
          stream_bufs->stride = s->rgb_front_stride;
          stream_bufs->buf_len = s->rgb_front_bufs[0].len;
          rep.num_fds = UI_BUF_COUNT;
          for (int i=0; i<rep.num_fds; i++) {
            rep.fds[i] = s->rgb_front_bufs[i].fd;
          }
          if (stream->tb) {
            stream->tbuffer = &s->ui_front_tb;
          } else {
            assert(false);
          }
        } else if (stream_type == VISION_STREAM_YUV) {
          stream_bufs->width = s->yuv_width;
          stream_bufs->height = s->yuv_height;
          stream_bufs->stride = s->yuv_width;
          stream_bufs->buf_len = s->yuv_buf_size;
          rep.num_fds = YUV_COUNT;
          for (int i=0; i<rep.num_fds; i++) {
            rep.fds[i] = s->yuv_ion[i].fd;
          }
          if (stream->tb) {
            stream->tbuffer = s->yuv_tb;
          } else {
            stream->queue = pool_get_queue(&s->yuv_pool);
          }
        } else if (stream_type == VISION_STREAM_YUV_FRONT) {
          stream_bufs->width = s->yuv_front_width;
          stream_bufs->height = s->yuv_front_height;
          stream_bufs->stride = s->yuv_front_width;
          stream_bufs->buf_len = s->yuv_front_buf_size;
          rep.num_fds = YUV_COUNT;
          for (int i=0; i<rep.num_fds; i++) {
            rep.fds[i] = s->yuv_front_ion[i].fd;
          }
          if (stream->tb) {
            stream->tbuffer = s->yuv_front_tb;
          } else {
            stream->queue = pool_get_queue(&s->yuv_front_pool);
          }
        } else {
          assert(false);
        }
        vipc_send(fd, &rep);
        streams[stream_type].subscribed = true;
      } else if (p.type == VIPC_STREAM_RELEASE) {
        // printf("client release f %d  %d\n", p.d.stream_rel.type, p.d.stream_rel.idx);
        int si = p.d.stream_rel.type;
        assert(si < VISION_STREAM_MAX);
        if (streams[si].tb) {
          tbuffer_release(streams[si].tbuffer, p.d.stream_rel.idx);
        } else {
          poolq_release(streams[si].queue, p.d.stream_rel.idx);
        }
        streams[p.d.stream_rel.type].bufs_outstanding--;
      } else {
        assert(false);
      }
    } else {
      int stream_i = VISION_STREAM_MAX;
      for (int i=2; i<num_polls; i++) {
        int si = poll_to_stream[i];
        if (!streams[si].subscribed) continue;
        if (polls[i].revents) {
          stream_i = si;
          break;
        }
      }
      if (stream_i < VISION_STREAM_MAX) {
        streams[stream_i].bufs_outstanding++;
        int idx;
        if (streams[stream_i].tb) {
          idx = tbuffer_acquire(streams[stream_i].tbuffer);
        } else {
          idx = poolq_pop(streams[stream_i].queue);
        }
        if (idx < 0) {
          break;
        }
        VisionPacket rep = {
          .type = VIPC_STREAM_ACQUIRE,
          .d = {.stream_acq = {
            .type = (VisionStreamType)stream_i,
            .idx = idx,
          }},
        };
        if (stream_i == VISION_STREAM_YUV) {
          rep.d.stream_acq.extra.frame_id = s->yuv_metas[idx].frame_id;
          rep.d.stream_acq.extra.timestamp_eof = s->yuv_metas[idx].timestamp_eof;
        } else if (stream_i == VISION_STREAM_YUV_FRONT) {
          rep.d.stream_acq.extra.frame_id = s->yuv_front_metas[idx].frame_id;
          rep.d.stream_acq.extra.timestamp_eof = s->yuv_front_metas[idx].timestamp_eof;
        }
        vipc_send(fd, &rep);
      }
    }
  }

  LOGW("client end fd %d", fd);

  for (int i=0; i<VISION_STREAM_MAX; i++) {
    if (!streams[i].subscribed) continue;
    if (streams[i].tb) {
      tbuffer_release_all(streams[i].tbuffer);
    } else {
      pool_release_queue(streams[i].queue);
    }
  }

  close(fd);
  zsock_destroy(&terminate);

  pthread_mutex_lock(&s->clients_lock);
  client->running = false;
  pthread_mutex_unlock(&s->clients_lock);

  return NULL;
}

void* visionserver_thread(void* arg) {
  int err;
  VisionState *s = (VisionState*)arg;

  set_thread_name("visionserver");

  zsock_t *terminate = zsock_new_sub(">inproc://terminate", "");
  assert(terminate);
  void* terminate_raw = zsock_resolve(terminate);

  int sock = ipc_bind(VIPC_SOCKET_PATH);
  while (!do_exit) {
    zmq_pollitem_t polls[2] = {{0}};
    polls[0].socket = terminate_raw;
    polls[0].events = ZMQ_POLLIN;
    polls[1].fd = sock;
    polls[1].events = ZMQ_POLLIN;

    int ret = zmq_poll(polls, ARRAYSIZE(polls), -1);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }
    if (polls[0].revents) {
      break;
    } else if (!polls[1].revents) {
      continue;
    }

    int fd = accept(sock, NULL, NULL);
    assert(fd >= 0);

    pthread_mutex_lock(&s->clients_lock);

    int client_idx = 0;
    for (; client_idx < MAX_CLIENTS; client_idx++) {
      if (!s->clients[client_idx].running) break;
    }

    if (client_idx >= MAX_CLIENTS) {
      LOG("ignoring visionserver connection, max clients connected");
      close(fd);

      pthread_mutex_unlock(&s->clients_lock);
      continue;
    }

    VisionClientState *client = &s->clients[client_idx];
    client->s = s;
    client->fd = fd;
    client->running = true;

    err = pthread_create(&client->thread_handle, NULL,
                         visionserver_client_thread, client);
    assert(err == 0);

    pthread_mutex_unlock(&s->clients_lock);
  }

  for (int i=0; i<MAX_CLIENTS; i++) {
    pthread_mutex_lock(&s->clients_lock);
    bool running = s->clients[i].running;
    pthread_mutex_unlock(&s->clients_lock);
    if (running) {
      err = pthread_join(s->clients[i].thread_handle, NULL);
      assert(err == 0);
    }
  }

  close(sock);
  zsock_destroy(&terminate);

  return NULL;
}


////////// cl stuff

cl_program build_debayer_program(VisionState *s,
                                 int frame_width, int frame_height, int frame_stride,
                                 int rgb_width, int rgb_height, int rgb_stride,
                                 int bayer_flip, int hdr) {
  assert(rgb_width == frame_width/2);
  assert(rgb_height == frame_height/2);

  #ifdef QCOM2
    int dnew = 1;
  #else
    int dnew = 0;
  #endif

  char args[4096];
  snprintf(args, sizeof(args),
          "-cl-fast-relaxed-math -cl-denorms-are-zero "
          "-DFRAME_WIDTH=%d -DFRAME_HEIGHT=%d -DFRAME_STRIDE=%d "
            "-DRGB_WIDTH=%d -DRGB_HEIGHT=%d -DRGB_STRIDE=%d "
            "-DBAYER_FLIP=%d -DHDR=%d -DNEW=%d",
          frame_width, frame_height, frame_stride,
          rgb_width, rgb_height, rgb_stride,
          bayer_flip, hdr, dnew);
  return CLU_LOAD_FROM_FILE(s->context, s->device_id, "cameras/debayer.cl", args);
}

cl_program build_conv_program(VisionState *s,
                              int image_w, int image_h,
                              int filter_size) {
  char args[4096];
  snprintf(args, sizeof(args),
          "-cl-fast-relaxed-math -cl-denorms-are-zero "
          "-DIMAGE_W=%d -DIMAGE_H=%d -DFLIP_RB=%d "
          "-DFILTER_SIZE=%d -DHALF_FILTER_SIZE=%d -DTWICE_HALF_FILTER_SIZE=%d -DHALF_FILTER_SIZE_IMAGE_W=%d",
          image_w, image_h, 1,
          filter_size, filter_size/2, (filter_size/2)*2, (filter_size/2)*image_w);
  return CLU_LOAD_FROM_FILE(s->context, s->device_id, "imgproc/conv.cl", args);
}

cl_program build_pool_program(VisionState *s,
                              int full_stride_x,
                              int x_pitch, int y_pitch,
                              int roi_x_min, int roi_x_max,
                              int roi_y_min, int roi_y_max) {
  char args[4096];
  snprintf(args, sizeof(args),
          "-cl-fast-relaxed-math -cl-denorms-are-zero "
          "-DFULL_STRIDE_X=%d -DX_PITCH=%d -DY_PITCH=%d "
          "-DROI_X_MIN=%d -DROI_X_MAX=%d -DROI_Y_MIN=%d -DROI_Y_MAX=%d",
          full_stride_x, x_pitch, y_pitch,
          roi_x_min, roi_x_max, roi_y_min, roi_y_max);
  return CLU_LOAD_FROM_FILE(s->context, s->device_id, "imgproc/pool.cl", args);
}

void cl_init(VisionState *s) {
  int err;
  cl_platform_id platform_id = NULL;
  cl_uint num_devices;
  cl_uint num_platforms;

  err = clGetPlatformIDs(1, &platform_id, &num_platforms);
  assert(err == 0);
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                       &s->device_id, &num_devices);
  assert(err == 0);

  cl_print_info(platform_id, s->device_id);
  printf("\n");

  s->context = clCreateContext(NULL, 1, &s->device_id, NULL, NULL, &err);
  assert(err == 0);
}

void cl_free(VisionState *s) {
  int err;

  err = clReleaseContext(s->context);
  assert(err == 0);
}

void init_buffers(VisionState *s) {
  int err;

  // allocate camera buffers

  for (int i=0; i<FRAME_BUF_COUNT; i++) {
    s->camera_bufs[i] = visionbuf_allocate_cl(s->frame_size, s->device_id, s->context,
                                              &s->camera_bufs_cl[i]);
    #ifndef QCOM2
      // TODO: make lengths correct
      s->focus_bufs[i] = visionbuf_allocate(0xb80);
      s->stats_bufs[i] = visionbuf_allocate(0xb80);
    #endif
  }

  for (int i=0; i<FRAME_BUF_COUNT; i++) {
    s->front_camera_bufs[i] = visionbuf_allocate_cl(s->cameras.front.frame_size,
                                                       s->device_id, s->context,
                                                       &s->front_camera_bufs_cl[i]);
  }

  // processing buffers
  if (s->cameras.rear.ci.bayer) {
    s->rgb_width = s->frame_width/2;
    s->rgb_height = s->frame_height/2;
  } else {
    s->rgb_width = s->frame_width;
    s->rgb_height = s->frame_height;
  }

  for (int i=0; i<UI_BUF_COUNT; i++) {
    VisionImg img = visionimg_alloc_rgb24(s->rgb_width, s->rgb_height, &s->rgb_bufs[i]);
    s->rgb_bufs_cl[i] = visionbuf_to_cl(&s->rgb_bufs[i], s->device_id, s->context);
    if (i == 0){
      s->rgb_stride = img.stride;
      s->rgb_buf_size = img.size;
    }
  }
  tbuffer_init(&s->ui_tb, UI_BUF_COUNT, "rgb");

  //assert(s->cameras.front.ci.bayer);
  if (s->cameras.front.ci.bayer) {
    s->rgb_front_width = s->cameras.front.ci.frame_width/2;
    s->rgb_front_height = s->cameras.front.ci.frame_height/2;
  } else {
    s->rgb_front_width = s->cameras.front.ci.frame_width;
    s->rgb_front_height = s->cameras.front.ci.frame_height;
  }


  for (int i=0; i<UI_BUF_COUNT; i++) {
    VisionImg img = visionimg_alloc_rgb24(s->rgb_front_width, s->rgb_front_height, &s->rgb_front_bufs[i]);
    s->rgb_front_bufs_cl[i] = visionbuf_to_cl(&s->rgb_front_bufs[i], s->device_id, s->context);
    if (i == 0){
      s->rgb_front_stride = img.stride;
      s->rgb_front_buf_size = img.size;
    }
  }
  tbuffer_init(&s->ui_front_tb, UI_BUF_COUNT, "frontrgb");

  // yuv back for recording and orbd
  pool_init(&s->yuv_pool, YUV_COUNT);
  s->yuv_tb = pool_get_tbuffer(&s->yuv_pool); //only for visionserver...

  s->yuv_width = s->rgb_width;
  s->yuv_height = s->rgb_height;
  s->yuv_buf_size = s->rgb_width * s->rgb_height * 3 / 2;

  for (int i=0; i<YUV_COUNT; i++) {
    s->yuv_ion[i] = visionbuf_allocate_cl(s->yuv_buf_size, s->device_id, s->context, &s->yuv_cl[i]);
    s->yuv_bufs[i].y = (uint8_t*)s->yuv_ion[i].addr;
    s->yuv_bufs[i].u = s->yuv_bufs[i].y + (s->yuv_width * s->yuv_height);
    s->yuv_bufs[i].v = s->yuv_bufs[i].u + (s->yuv_width/2 * s->yuv_height/2);
  }

  // yuv front for recording
  pool_init(&s->yuv_front_pool, YUV_COUNT);
  s->yuv_front_tb = pool_get_tbuffer(&s->yuv_front_pool);

  s->yuv_front_width = s->rgb_front_width;
  s->yuv_front_height = s->rgb_front_height;
  s->yuv_front_buf_size = s->rgb_front_width * s->rgb_front_height * 3 / 2;

  for (int i=0; i<YUV_COUNT; i++) {
    s->yuv_front_ion[i] = visionbuf_allocate_cl(s->yuv_front_buf_size, s->device_id, s->context, &s->yuv_front_cl[i]);
    s->yuv_front_bufs[i].y = (uint8_t*)s->yuv_front_ion[i].addr;
    s->yuv_front_bufs[i].u = s->yuv_front_bufs[i].y + (s->yuv_front_width * s->yuv_front_height);
    s->yuv_front_bufs[i].v = s->yuv_front_bufs[i].u + (s->yuv_front_width/2 * s->yuv_front_height/2);
  }

  if (s->cameras.rear.ci.bayer) {
    // debayering does a 2x downscale
    s->yuv_transform = transform_scale_buffer(s->cameras.rear.transform, 0.5);
  } else {
    s->yuv_transform = s->cameras.rear.transform;
  }

  if (s->cameras.rear.ci.bayer) {
    s->prg_debayer_rear = build_debayer_program(s, s->cameras.rear.ci.frame_width, s->cameras.rear.ci.frame_height,
                                                   s->cameras.rear.ci.frame_stride,
                                                 s->rgb_width, s->rgb_height, s->rgb_stride,
                                                 s->cameras.rear.ci.bayer_flip, s->cameras.rear.ci.hdr);
    s->krnl_debayer_rear = clCreateKernel(s->prg_debayer_rear, "debayer10", &err);
    assert(err == 0);
  }

  if (s->cameras.front.ci.bayer) {
    s->prg_debayer_front = build_debayer_program(s, s->cameras.front.ci.frame_width, s->cameras.front.ci.frame_height,
                                                    s->cameras.front.ci.frame_stride,
                                                 s->rgb_front_width, s->rgb_front_height, s->rgb_front_stride,
                                                 s->cameras.front.ci.bayer_flip, s->cameras.front.ci.hdr);

    s->krnl_debayer_front = clCreateKernel(s->prg_debayer_front, "debayer10", &err);
    assert(err == 0);
  }

  s->prg_rgb_laplacian = build_conv_program(s, s->rgb_width/NUM_SEGMENTS_X, s->rgb_height/NUM_SEGMENTS_Y,
                                            3);
  s->krnl_rgb_laplacian = clCreateKernel(s->prg_rgb_laplacian, "rgb2gray_conv2d", &err);
  assert(err == 0);
  // TODO: Removed CL_MEM_SVM_FINE_GRAIN_BUFFER, confirm it doesn't matter
  s->rgb_conv_roi_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
      s->rgb_width/NUM_SEGMENTS_X * s->rgb_height/NUM_SEGMENTS_Y * 3 * sizeof(uint8_t), NULL, NULL);
  s->rgb_conv_result_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
      s->rgb_width/NUM_SEGMENTS_X * s->rgb_height/NUM_SEGMENTS_Y * sizeof(int16_t), NULL, NULL);
  s->rgb_conv_filter_cl = clCreateBuffer(s->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      9 * sizeof(int16_t), (void*)&lapl_conv_krnl, NULL);
  s->conv_cl_localMemSize = ( CONV_LOCAL_WORKSIZE + 2 * (3 / 2) ) * ( CONV_LOCAL_WORKSIZE + 2 * (3 / 2) );
  s->conv_cl_localMemSize *= 3 * sizeof(uint8_t);
  s->conv_cl_globalWorkSize[0] = s->rgb_width/NUM_SEGMENTS_X;
  s->conv_cl_globalWorkSize[1] = s->rgb_height/NUM_SEGMENTS_Y;
  s->conv_cl_localWorkSize[0] = CONV_LOCAL_WORKSIZE;
  s->conv_cl_localWorkSize[1] = CONV_LOCAL_WORKSIZE;

  for (int i=0; i<(ROI_X_MAX-ROI_X_MIN+1)*(ROI_Y_MAX-ROI_Y_MIN+1); i++) {s->lapres[i] = 16160;}

  rgb_to_yuv_init(&s->rgb_to_yuv_state, s->context, s->device_id, s->yuv_width, s->yuv_height, s->rgb_stride);
  rgb_to_yuv_init(&s->front_rgb_to_yuv_state, s->context, s->device_id, s->yuv_front_width, s->yuv_front_height, s->rgb_front_stride);
}

void free_buffers(VisionState *s) {
  // free bufs
  for (int i=0; i<FRAME_BUF_COUNT; i++) {
    visionbuf_free(&s->camera_bufs[i]);
    visionbuf_free(&s->focus_bufs[i]);
    visionbuf_free(&s->stats_bufs[i]);
  }

  for (int i=0; i<FRAME_BUF_COUNT; i++) {
   visionbuf_free(&s->front_camera_bufs[i]);
  }

  for (int i=0; i<UI_BUF_COUNT; i++) {
    visionbuf_free(&s->rgb_bufs[i]);
  }

  for (int i=0; i<UI_BUF_COUNT; i++) {
    visionbuf_free(&s->rgb_front_bufs[i]);
  }

  for (int i=0; i<YUV_COUNT; i++) {
    visionbuf_free(&s->yuv_ion[i]);
  }
}

void party(VisionState *s) {
  int err;

  s->terminate_pub = zsock_new_pub("@inproc://terminate");
  assert(s->terminate_pub);

  pthread_t visionserver_thread_handle;
  err = pthread_create(&visionserver_thread_handle, NULL,
                       visionserver_thread, s);
  assert(err == 0);

  pthread_t proc_thread_handle;
  err = pthread_create(&proc_thread_handle, NULL,
                       processing_thread, s);
  assert(err == 0);

#if !defined(QCOM2) && !defined(__APPLE__)
  // TODO: fix front camera on qcom2
  pthread_t frontview_thread_handle;
  err = pthread_create(&frontview_thread_handle, NULL,
                       frontview_thread, s);
  assert(err == 0);
#endif

  // priority for cameras
  err = set_realtime_priority(51);
  LOG("setpriority returns %d", err);

  cameras_run(&s->cameras);

  tbuffer_stop(&s->ui_tb);
  tbuffer_stop(&s->ui_front_tb);
  pool_stop(&s->yuv_pool);
  pool_stop(&s->yuv_front_pool);

  zsock_signal(s->terminate_pub, 0);

#if !defined(QCOM2) && !defined(QCOM_REPLAY) && !defined(__APPLE__)
  LOG("joining frontview_thread");
  err = pthread_join(frontview_thread_handle, NULL);
  assert(err == 0);
#endif

  LOG("joining visionserver_thread");
  err = pthread_join(visionserver_thread_handle, NULL);
  assert(err == 0);

  LOG("joining proc_thread");
  err = pthread_join(proc_thread_handle, NULL);
  assert(err == 0);

  zsock_destroy (&s->terminate_pub);
}

int main(int argc, char *argv[]) {
  set_realtime_priority(51);

  zsys_handler_set(NULL);
  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  VisionState state = {0};
  VisionState *s = &state;

  clu_init();
  cl_init(s);

  cameras_init(&s->cameras);

  s->frame_width = s->cameras.rear.ci.frame_width;
  s->frame_height = s->cameras.rear.ci.frame_height;
  s->frame_stride = s->cameras.rear.ci.frame_stride;
  s->frame_size = s->cameras.rear.frame_size;

  init_buffers(s);

#if (defined(QCOM) && !defined(QCOM_REPLAY)) || defined(QCOM2)
  s->pm = new PubMaster({"frame", "frontFrame", "thumbnail"});
#endif

  cameras_open(&s->cameras, &s->camera_bufs[0], &s->focus_bufs[0], &s->stats_bufs[0], &s->front_camera_bufs[0]);

  party(s);

  if (s->pm != NULL) {
    delete s->pm;
  }

  free_buffers(s);
  cl_free(s);
}
