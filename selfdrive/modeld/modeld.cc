#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>

#include "common/visionbuf.h"
#include "common/visionipc.h"
#include "common/swaglog.h"
#include "common/clutil.h"

#include "models/driving.h"
#include "messaging.hpp"
volatile sig_atomic_t do_exit = 0;

static void set_do_exit(int sig) {
  do_exit = 1;
}

// globals
bool run_model;
mat3 cur_transform;
pthread_mutex_t transform_lock;

void* live_thread(void *arg) {
  set_thread_name("live");

  SubMaster sm({"liveCalibration"});
  /*
     import numpy as np
     from common.transformations.model import medmodel_frame_from_road_frame
     medmodel_frame_from_ground = medmodel_frame_from_road_frame[:, (0, 1, 3)]
     ground_from_medmodel_frame = np.linalg.inv(medmodel_frame_from_ground)
  */
  Eigen::Matrix<float, 3, 3> ground_from_medmodel_frame;
  ground_from_medmodel_frame <<
    0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
    -1.09890110e-03, 0.00000000e+00, 2.81318681e-01,
    -1.84808520e-20, 9.00738606e-04,-4.28751576e-02;

  Eigen::Matrix<float, 3, 3> fcam_intrinsics;
#ifndef QCOM2
  fcam_intrinsics <<
    910.0, 0.0, 582.0,
    0.0, 910.0, 437.0,
    0.0,   0.0,   1.0;
  float db_s = 0.5; // debayering does a 2x downscale
#else
  fcam_intrinsics <<
    2648.0, 0.0, 1928.0/2,
    0.0, 2648.0, 1208.0/2,
    0.0,   0.0,   1.0;
  float db_s = 1.0;
#endif

  mat3 yuv_transform = transform_scale_buffer((mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }}, db_s);

  while (!do_exit) {
    if (sm.update(10) > 0){

      auto extrinsic_matrix = sm["liveCalibration"].getLiveCalibration().getExtrinsicMatrix();
      Eigen::Matrix<float, 3, 4> extrinsic_matrix_eigen;
      for (int i = 0; i < 4*3; i++){
        extrinsic_matrix_eigen(i / 4, i % 4) = extrinsic_matrix[i];
      }

      auto camera_frame_from_road_frame = fcam_intrinsics * extrinsic_matrix_eigen;
      Eigen::Matrix<float, 3, 3> camera_frame_from_ground;
      camera_frame_from_ground.col(0) = camera_frame_from_road_frame.col(0);
      camera_frame_from_ground.col(1) = camera_frame_from_road_frame.col(1);
      camera_frame_from_ground.col(2) = camera_frame_from_road_frame.col(3);

      auto warp_matrix = camera_frame_from_ground * ground_from_medmodel_frame;
      mat3 transform = {};
      for (int i=0; i<3*3; i++) {
        transform.v[i] = warp_matrix(i / 3, i % 3);
      }
      mat3 model_transform = matmul3(yuv_transform, transform);
      pthread_mutex_lock(&transform_lock);
      cur_transform = model_transform;
      run_model = true;
      pthread_mutex_unlock(&transform_lock);
    }
  }
  return NULL;
}

int main(int argc, char **argv) {
  int err;
  set_realtime_priority(51);

#ifdef QCOM
  set_core_affinity(2);
#elif QCOM2
  // CPU usage is much lower when pinned to a single big core
  set_core_affinity(4);
#endif

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  pthread_mutex_init(&transform_lock, NULL);

  // start calibration thread
  pthread_t live_thread_handle;
  err = pthread_create(&live_thread_handle, NULL, live_thread, NULL);
  assert(err == 0);

  // messaging
  PubMaster pm({"modelV2", "model", "cameraOdometry"});
  SubMaster sm({"pathPlan", "frame"});

#if defined(QCOM) || defined(QCOM2)
  cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
#else
  cl_device_type device_type = CL_DEVICE_TYPE_CPU;
#endif

  // cl init
  cl_device_id device_id = cl_get_device_id(device_type);
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  assert(err == 0);

  cl_command_queue q = clCreateCommandQueue(context, device_id, 0, &err);
  assert(err == 0);

  // init the models
  ModelState model;
  model_init(&model, device_id, context, true);
  LOGW("models loaded, modeld starting");

  // loop
  VisionStream stream;
  while (!do_exit) {
    VisionStreamBufs buf_info;
    err = visionstream_init(&stream, VISION_STREAM_YUV, true, &buf_info);
    if (err) {
      LOGW("visionstream connect failed");
      usleep(100000);
      continue;
    }
    LOGW("connected with buffer size: %d", buf_info.buf_len);

    // setup filter to track dropped frames
    const float dt = 1. / MODEL_FREQ;
    const float ts = 5.0;  // 5 s filter time constant
    const float frame_filter_k = (dt / ts) / (1. + dt / ts);
    float frames_dropped = 0;

    // one frame in memory
    VisionBuf yuv_ion = visionbuf_allocate_cl(buf_info.buf_len, device_id, context);

    uint32_t frame_id = 0, last_vipc_frame_id = 0;
    double last = 0;
    int desire = -1;
    while (!do_exit) {
      VIPCBuf *buf;
      VIPCBufExtra extra;
      buf = visionstream_get(&stream, &extra);
      if (buf == NULL) {
        LOGW("visionstream get failed");
        break;
      }

      pthread_mutex_lock(&transform_lock);
      mat3 model_transform = cur_transform;
      const bool run_model_this_iter = run_model;
      pthread_mutex_unlock(&transform_lock);

      if (sm.update(0) > 0){
        // TODO: path planner timeout?
        desire = ((int)sm["pathPlan"].getPathPlan().getDesire());
        frame_id = sm["frame"].getFrame().getFrameId();
      }

      double mt1 = 0, mt2 = 0;
      if (run_model_this_iter) {
        float vec_desire[DESIRE_LEN] = {0};
        if (desire >= 0 && desire < DESIRE_LEN) {
          vec_desire[desire] = 1.0;
        }

        mt1 = millis_since_boot();

        // TODO: don't make copies!
        memcpy(yuv_ion.addr, buf->addr, buf_info.buf_len);

        ModelDataRaw model_buf =
            model_eval_frame(&model, q, yuv_ion.buf_cl, buf_info.width, buf_info.height,
                             model_transform, NULL, vec_desire);
        mt2 = millis_since_boot();

        // tracked dropped frames
        uint32_t vipc_dropped_frames = extra.frame_id - last_vipc_frame_id - 1;
        frames_dropped = (1. - frame_filter_k) * frames_dropped + frame_filter_k * (float)std::min(vipc_dropped_frames, 10U);
        float frame_drop_perc = frames_dropped / MODEL_FREQ;

        model_publish(pm, extra.frame_id, frame_id,  vipc_dropped_frames, frame_drop_perc, model_buf, extra.timestamp_eof);
        model_publish_v2(pm, extra.frame_id, frame_id,  vipc_dropped_frames, frame_drop_perc, model_buf, extra.timestamp_eof);
        posenet_publish(pm, extra.frame_id, frame_id, vipc_dropped_frames, frame_drop_perc, model_buf, extra.timestamp_eof);

        LOGD("model process: %.2fms, from last %.2fms, vipc_frame_id %zu, frame_id, %zu, frame_drop %.3f", mt2-mt1, mt1-last, extra.frame_id, frame_id, frame_drop_perc);
        last = mt1;
        last_vipc_frame_id = extra.frame_id;
      }

    }
    visionbuf_free(&yuv_ion);
    visionstream_destroy(&stream);
  }

  model_free(&model);

  LOG("joining live_thread");
  err = pthread_join(live_thread_handle, NULL);
  assert(err == 0);
  clReleaseCommandQueue(q);
  clReleaseContext(context);

  pthread_mutex_destroy(&transform_lock);
  return 0;
}
