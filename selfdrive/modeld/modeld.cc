#include <stdio.h>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>

#include "common/visionbuf.h"
#include "common/visionipc.h"
#include "common/swaglog.h"

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
  int err;
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

  Eigen::Matrix<float, 3, 3> eon_intrinsics;
  eon_intrinsics <<
    910.0, 0.0, 582.0,
    0.0, 910.0, 437.0,
    0.0,   0.0,   1.0;

  while (!do_exit) {
    if (sm.update(10) > 0){
      
      auto extrinsic_matrix = sm["liveCalibration"].getLiveCalibration().getExtrinsicMatrix();
      Eigen::Matrix<float, 3, 4> extrinsic_matrix_eigen;
      for (int i = 0; i < 4*3; i++){
        extrinsic_matrix_eigen(i / 4, i % 4) = extrinsic_matrix[i];
      }

      auto camera_frame_from_road_frame = eon_intrinsics * extrinsic_matrix_eigen;
      Eigen::Matrix<float, 3, 3> camera_frame_from_ground;
      camera_frame_from_ground.col(0) = camera_frame_from_road_frame.col(0);
      camera_frame_from_ground.col(1) = camera_frame_from_road_frame.col(1);
      camera_frame_from_ground.col(2) = camera_frame_from_road_frame.col(3);

      auto warp_matrix = camera_frame_from_ground * ground_from_medmodel_frame;

      pthread_mutex_lock(&transform_lock);
      for (int i=0; i<3*3; i++) {
        cur_transform.v[i] = warp_matrix(i / 3, i % 3);
      }

      run_model = true;
      pthread_mutex_unlock(&transform_lock);
    }
  }
  return NULL;
}

int main(int argc, char **argv) {
  int err;
  set_realtime_priority(1);

  // start calibration thread
  pthread_t live_thread_handle;
  err = pthread_create(&live_thread_handle, NULL, live_thread, NULL);
  assert(err == 0);

  // messaging
  PubMaster pm({"model", "cameraOdometry"});
  SubMaster sm({"pathPlan"});

  // cl init
  cl_device_id device_id;
  cl_context context;
  cl_command_queue q;
  {
    // TODO: refactor this
    cl_platform_id platform_id[2];
    cl_uint num_devices;
    cl_uint num_platforms;

    err = clGetPlatformIDs(sizeof(platform_id)/sizeof(cl_platform_id), platform_id, &num_platforms);
    assert(err == 0);

    #ifdef QCOM
      int clPlatform = 0;
    #else
      // don't use nvidia on pc, it's broken
      // TODO: write this nicely
      int clPlatform = num_platforms-1;
    #endif

    char cBuffer[1024];
    clGetPlatformInfo(platform_id[clPlatform], CL_PLATFORM_NAME, sizeof(cBuffer), &cBuffer, NULL);
    LOGD("got %d opencl platform(s), using %s", num_platforms, cBuffer);

    err = clGetDeviceIDs(platform_id[clPlatform], CL_DEVICE_TYPE_DEFAULT, 1,
                         &device_id, &num_devices);
    assert(err == 0);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    assert(err == 0);

    q = clCreateCommandQueue(context, device_id, 0, &err);
    assert(err == 0);
  }

  // init the models
  ModelState model;
  model_init(&model, device_id, context, true);
  LOGW("models loaded, modeld starting");

  // debayering does a 2x downscale
  mat3 yuv_transform = transform_scale_buffer((mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }}, 0.5);

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

    // one frame in memory
    cl_mem yuv_cl;
    VisionBuf yuv_ion = visionbuf_allocate_cl(buf_info.buf_len, device_id, context, &yuv_cl);

    double last = 0;
    int desire = -1;
    while (!do_exit) {
      VIPCBuf *buf;
      VIPCBufExtra extra;
      buf = visionstream_get(&stream, &extra);
      if (buf == NULL) {
        LOGW("visionstream get failed");
        visionstream_destroy(&stream);
        break;
      }

      pthread_mutex_lock(&transform_lock);
      mat3 transform = cur_transform;
      const bool run_model_this_iter = run_model;
      pthread_mutex_unlock(&transform_lock);

      if (sm.update(0) > 0){
        // TODO: path planner timeout?
        desire = ((int)sm["pathPlan"].getPathPlan().getDesire()) - 1;
      }

      double mt1 = 0, mt2 = 0;
      if (run_model_this_iter) {
        float vec_desire[DESIRE_LEN] = {0};
        if (desire >= 0 && desire < DESIRE_LEN) {
          vec_desire[desire] = 1.0;
        }

        mat3 model_transform = matmul3(yuv_transform, transform);

        mt1 = millis_since_boot();

        // TODO: don't make copies!
        memcpy(yuv_ion.addr, buf->addr, buf_info.buf_len);

        ModelDataRaw model_buf =
            model_eval_frame(&model, q, yuv_cl, buf_info.width, buf_info.height,
                             model_transform, NULL, vec_desire);
        mt2 = millis_since_boot();

        model_publish(pm, extra.frame_id, model_buf, extra.timestamp_eof);
        posenet_publish(pm, extra.frame_id, model_buf, extra.timestamp_eof);

        LOGD("model process: %.2fms, from last %.2fms", mt2-mt1, mt1-last);
        last = mt1;
      }

    }
    visionbuf_free(&yuv_ion);
  }

  visionstream_destroy(&stream);

  model_free(&model);

  LOG("joining live_thread");
  err = pthread_join(live_thread_handle, NULL);
  assert(err == 0);

  return 0;
}
