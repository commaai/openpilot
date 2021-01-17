#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <thread>
#include "visionbuf.h"
#include "visionipc_client.h"
#include "common/swaglog.h"
#include "common/clutil.h"
#include "common/util.h"

#include "models/driving.h"
#include "messaging.hpp"

ExitHandler do_exit;

class Modeld {
 public:
  struct LiveData {
    int desire;
    uint32_t frame_id;
    mat3 transform;
  };

  Modeld();
  ~Modeld();
  void run();
  void live_thread();

  Modeld::LiveData get_live_data() {
    std::unique_lock lk(mutex);
    return _d;
  }

  std::thread thread;
  cl_context cl_ctx = nullptr;
  ModelState model;
  std::atomic<bool> run_model;
  std::unique_ptr<VisionIpcClient> vipc_client;
  std::mutex mutex;
  // updated by live_thread
  LiveData _d = {};
};

void Modeld::live_thread() {
  set_thread_name("live");
  set_realtime_priority(50);
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

  SubMaster sm({"liveCalibration", "pathPlan", "frame"});
  while (!do_exit) {
    if (!sm.update(100)) continue;

    if (sm.updated("liveCalibration")) {
      auto extrinsic_matrix = sm["liveCalibration"].getLiveCalibration().getExtrinsicMatrix();
      Eigen::Matrix<float, 3, 4> extrinsic_matrix_eigen;
      for (int i = 0; i < 4 * 3; i++) {
        extrinsic_matrix_eigen(i / 4, i % 4) = extrinsic_matrix[i];
      }

      auto camera_frame_from_road_frame = fcam_intrinsics * extrinsic_matrix_eigen;
      Eigen::Matrix<float, 3, 3> camera_frame_from_ground;
      camera_frame_from_ground.col(0) = camera_frame_from_road_frame.col(0);
      camera_frame_from_ground.col(1) = camera_frame_from_road_frame.col(1);
      camera_frame_from_ground.col(2) = camera_frame_from_road_frame.col(3);

      auto warp_matrix = camera_frame_from_ground * ground_from_medmodel_frame;
      mat3 transform = {};
      for (int i = 0; i < 3 * 3; i++) {
        transform.v[i] = warp_matrix(i / 3, i % 3);
      }
      mat3 model_transform = matmul3(yuv_transform, transform);

      run_model = true;
      std::unique_lock lk(mutex);
      _d.transform = model_transform;
    }
    if (sm.updated("pathPlan")) {
      // TODO: path planner timeout?
      std::unique_lock lk(mutex);
      _d.desire = ((int)sm["pathPlan"].getPathPlan().getDesire());
    }
    if (sm.updated("frame")) {
      std::unique_lock lk(mutex);
      _d.frame_id = sm["frame"].getFrame().getFrameId();
    }
  }
}

Modeld::Modeld() {
  thread = std::thread(&Modeld::live_thread, this);

  // cl init
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_ctx = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

  // init the models
  model_init(&model, device_id, cl_ctx);
  LOGW("models loaded, modeld starting");

  vipc_client = std::make_unique<VisionIpcClient>("camerad", VISION_STREAM_YUV_BACK, true, device_id, cl_ctx);
  while (!do_exit && !vipc_client->connect(false)) {
    util::sleep_for(100);
  }

  VisionBuf *b = &vipc_client->buffers[0];
  LOGW("connected with buffer size: %d (%d x %d)", b->len, b->width, b->height);

  // wait liveCalibration
  while (!run_model && !do_exit) util::sleep_for(10);
}

Modeld::~Modeld() {
  model_free(&model);
  thread.join();
  CL_CHECK(clReleaseContext(cl_ctx));
}

void Modeld::run() {
  // setup filter to track dropped frames
  const float dt = 1. / MODEL_FREQ;
  const float ts = 10.0;  // filter time constant (s)
  const float frame_filter_k = (dt / ts) / (1. + dt / ts);
  float frames_dropped = 0;

  uint32_t run_count = 0, last_vipc_frame_id = 0;
  double last = 0;

  PubMaster pm({"modelV2", "model", "cameraOdometry"});

  while (!do_exit) {
    VisionIpcBufExtra extra;
    VisionBuf *buf = vipc_client->recv(&extra);
    if (buf == nullptr) continue;

    Modeld::LiveData data = get_live_data();

    float vec_desire[DESIRE_LEN] = {0};
    if (data.desire >= 0 && data.desire < DESIRE_LEN) {
      vec_desire[data.desire] = 1.0;
    }

    const double mt1 = millis_since_boot();
    ModelDataRaw model_buf = model_eval_frame(&model, buf->buf_cl, buf->width, buf->height,
                                              data.transform, vec_desire);
    const double mt2 = millis_since_boot();
    const float model_execution_time = (mt2 - mt1) / 1000.0;

    // tracked dropped frames
    uint32_t vipc_dropped_frames = extra.frame_id - last_vipc_frame_id - 1;
    frames_dropped = (1. - frame_filter_k) * frames_dropped + frame_filter_k * (float)std::min(vipc_dropped_frames, 10U);
    if (++run_count < 10) frames_dropped = 0;  // let frame drops warm up
    float frame_drop_ratio = frames_dropped / (1 + frames_dropped);

    const float *raw_pred_ptr = send_raw_pred ? &model.output[0] : nullptr;
    model_publish(pm, extra.frame_id, data.frame_id, frame_drop_ratio, model_buf, raw_pred_ptr, extra.timestamp_eof, model_execution_time);
    posenet_publish(pm, extra.frame_id, vipc_dropped_frames, model_buf, extra.timestamp_eof);

    LOGD("model process: %.2fms, from last %.2fms, vipc_frame_id %u, frame_id, %u, frame_drop %.3f", mt2 - mt1, mt1 - last, extra.frame_id, data.frame_id, frame_drop_ratio);
    last = mt1;
    last_vipc_frame_id = extra.frame_id;
  }
}

int main(int argc, char **argv) {
  set_realtime_priority(54);
#ifdef QCOM
  set_core_affinity(2);
#elif QCOM2
  // CPU usage is much lower when pinned to a single big core
  set_core_affinity(4);
#endif
  Modeld modeld;
  modeld.run();
  return 0;
}
