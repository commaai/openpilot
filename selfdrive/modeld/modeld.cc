#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionipc_client.h"
#include "common/clutil.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/modeld/models/driving.h"

ExitHandler do_exit;

mat3 update_calibration(Eigen::Matrix<float, 3, 4> &extrinsics, bool wide_camera, bool bigmodel_frame) {
  /*
     import numpy as np
     from common.transformations.model import medmodel_frame_from_road_frame
     medmodel_frame_from_ground = medmodel_frame_from_road_frame[:, (0, 1, 3)]
     ground_from_medmodel_frame = np.linalg.inv(medmodel_frame_from_ground)
  */
  static const auto ground_from_medmodel_frame = (Eigen::Matrix<float, 3, 3>() <<
     0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
    -1.09890110e-03, 0.00000000e+00, 2.81318681e-01,
    -1.84808520e-20, 9.00738606e-04, -4.28751576e-02).finished();

  static const auto ground_from_sbigmodel_frame = (Eigen::Matrix<float, 3, 3>() <<
     0.00000000e+00,  7.31372216e-19,  1.00000000e+00,
    -2.19780220e-03,  4.11497335e-19,  5.62637363e-01,
    -5.46146580e-20,  1.80147721e-03, -2.73464241e-01).finished();

  const auto cam_intrinsics = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>(wide_camera ? ecam_intrinsic_matrix.v : fcam_intrinsic_matrix.v);
  static const mat3 yuv_transform = get_model_yuv_transform();

  auto ground_from_model_frame = bigmodel_frame ? ground_from_sbigmodel_frame : ground_from_medmodel_frame;
  auto camera_frame_from_road_frame = cam_intrinsics * extrinsics;
  Eigen::Matrix<float, 3, 3> camera_frame_from_ground;
  camera_frame_from_ground.col(0) = camera_frame_from_road_frame.col(0);
  camera_frame_from_ground.col(1) = camera_frame_from_road_frame.col(1);
  camera_frame_from_ground.col(2) = camera_frame_from_road_frame.col(3);

  auto warp_matrix = camera_frame_from_ground * ground_from_model_frame;
  mat3 transform = {};
  for (int i=0; i<3*3; i++) {
    transform.v[i] = warp_matrix(i / 3, i % 3);
  }
  return matmul3(yuv_transform, transform);
}


void run_model(ModelState &model, VisionIpcClient &vipc_client_main, VisionIpcClient &vipc_client_extra, bool main_wide_camera, bool use_extra_client) {
  // messaging
  PubMaster pm({"modelV2", "cameraOdometry"});
  SubMaster sm({"lateralPlan", "roadCameraState", "liveCalibration"});

  // setup filter to track dropped frames
  FirstOrderFilter frame_dropped_filter(0., 10., 1. / MODEL_FREQ);

  uint32_t frame_id = 0, last_vipc_frame_id = 0;
  double last = 0;
  uint32_t run_count = 0;

  mat3 model_transform_main = {};
  mat3 model_transform_extra = {};
  bool live_calib_seen = false;

  VisionBuf *buf_main = nullptr;
  VisionBuf *buf_extra = nullptr;

  VisionIpcBufExtra meta_main = {0};
  VisionIpcBufExtra meta_extra = {0};

  while (!do_exit) {
    // Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while (meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000ULL) {
      buf_main = vipc_client_main.recv(&meta_main);
      if (buf_main == nullptr)  break;
    }

    if (buf_main == nullptr) {
      LOGE("vipc_client_main no frame");
      continue;
    }

    if (use_extra_client) {
      // Keep receiving extra frames until frame id matches main camera
      do {
        buf_extra = vipc_client_extra.recv(&meta_extra);
      } while (buf_extra != nullptr && meta_main.timestamp_sof > meta_extra.timestamp_sof + 25000000ULL);

      if (buf_extra == nullptr) {
        LOGE("vipc_client_extra no frame");
        continue;
      }

      if (std::abs((int64_t)meta_main.timestamp_sof - (int64_t)meta_extra.timestamp_sof) > 10000000ULL) {
        LOGE("frames out of sync! main: %d (%.5f), extra: %d (%.5f)",
          meta_main.frame_id, double(meta_main.timestamp_sof) / 1e9,
          meta_extra.frame_id, double(meta_extra.timestamp_sof) / 1e9);
      }
    } else {
      // Use single camera
      buf_extra = buf_main;
      meta_extra = meta_main;
    }

    // TODO: path planner timeout?
    sm.update(0);
    int desire = ((int)sm["lateralPlan"].getLateralPlan().getDesire());
    frame_id = sm["roadCameraState"].getRoadCameraState().getFrameId();
    if (sm.updated("liveCalibration")) {
      auto extrinsic_matrix = sm["liveCalibration"].getLiveCalibration().getExtrinsicMatrix();
      Eigen::Matrix<float, 3, 4> extrinsic_matrix_eigen;
      for (int i = 0; i < 4*3; i++) {
        extrinsic_matrix_eigen(i / 4, i % 4) = extrinsic_matrix[i];
      }

      model_transform_main = update_calibration(extrinsic_matrix_eigen, main_wide_camera, false);
      model_transform_extra = update_calibration(extrinsic_matrix_eigen, true, true);
      live_calib_seen = true;
    }

    float vec_desire[DESIRE_LEN] = {0};
    if (desire >= 0 && desire < DESIRE_LEN) {
      vec_desire[desire] = 1.0;
    }

    // tracked dropped frames
    uint32_t vipc_dropped_frames = meta_main.frame_id - last_vipc_frame_id - 1;
    float frames_dropped = frame_dropped_filter.update((float)std::min(vipc_dropped_frames, 10U));
    if (run_count < 10) { // let frame drops warm up
      frame_dropped_filter.reset(0);
      frames_dropped = 0.;
    }
    run_count++;

    float frame_drop_ratio = frames_dropped / (1 + frames_dropped);
    bool prepare_only = vipc_dropped_frames > 0;

    if (prepare_only) {
      LOGE("skipping model eval. Dropped %d frames", vipc_dropped_frames);
    }

    double mt1 = millis_since_boot();
    ModelOutput *model_output = model_eval_frame(&model, buf_main, buf_extra, model_transform_main, model_transform_extra, vec_desire, prepare_only);
    double mt2 = millis_since_boot();
    float model_execution_time = (mt2 - mt1) / 1000.0;

    if (model_output != nullptr) {
      model_publish(pm, meta_main.frame_id, meta_extra.frame_id, frame_id, frame_drop_ratio, *model_output, meta_main.timestamp_eof, model_execution_time,
                    kj::ArrayPtr<const float>(model.output.data(), model.output.size()), live_calib_seen);
      posenet_publish(pm, meta_main.frame_id, vipc_dropped_frames, *model_output, meta_main.timestamp_eof, live_calib_seen);
    }

    //printf("model process: %.2fms, from last %.2fms, vipc_frame_id %u, frame_id, %u, frame_drop %.3f\n", mt2 - mt1, mt1 - last, extra.frame_id, frame_id, frame_drop_ratio);
    last = mt1;
    last_vipc_frame_id = meta_main.frame_id;
  }
}

int main(int argc, char **argv) {
  if (!Hardware::PC()) {
    int ret;
    ret = util::set_realtime_priority(54);
    assert(ret == 0);
    util::set_core_affinity({7});
    assert(ret == 0);
  }

  bool main_wide_camera = Params().getBool("WideCameraOnly");
  bool use_extra_client = !main_wide_camera;  // set for single camera mode

  // cl init
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

  // init the models
  ModelState model;
  model_init(&model, device_id, context);
  LOGW("models loaded, modeld starting");

  VisionIpcClient vipc_client_main = VisionIpcClient("camerad", main_wide_camera ? VISION_STREAM_WIDE_ROAD : VISION_STREAM_ROAD, true, device_id, context);
  VisionIpcClient vipc_client_extra = VisionIpcClient("camerad", VISION_STREAM_WIDE_ROAD, false, device_id, context);

  while (!do_exit && !vipc_client_main.connect(false)) {
    util::sleep_for(100);
  }

  while (!do_exit && use_extra_client && !vipc_client_extra.connect(false)) {
    util::sleep_for(100);
  }

  // run the models
  // vipc_client.connected is false only when do_exit is true
  if (!do_exit) {
    const VisionBuf *b = &vipc_client_main.buffers[0];
    LOGW("connected main cam with buffer size: %d (%d x %d)", b->len, b->width, b->height);

    if (use_extra_client) {
      const VisionBuf *wb = &vipc_client_extra.buffers[0];
      LOGW("connected extra cam with buffer size: %d (%d x %d)", wb->len, wb->width, wb->height);
    }

    run_model(model, vipc_client_main, vipc_client_extra, main_wide_camera, use_extra_client);
  }

  model_free(&model);
  CL_CHECK(clReleaseContext(context));
  return 0;
}
