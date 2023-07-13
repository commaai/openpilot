#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "cereal/messaging/messaging.h"
#include "common/transformations/orientation.hpp"

#include "cereal/visionipc/visionipc_client.h"
#include "common/clutil.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "system/hardware/hw.h"
#include "selfdrive/modeld/models/driving.h"
#include "selfdrive/modeld/models/nav.h"


ExitHandler do_exit;

mat3 update_calibration(Eigen::Vector3d device_from_calib_euler, bool wide_camera, bool bigmodel_frame) {
  /*
     import numpy as np
     from common.transformations.model import medmodel_frame_from_calib_frame
     medmodel_frame_from_calib_frame = medmodel_frame_from_calib_frame[:, :3]
     calib_from_smedmodel_frame = np.linalg.inv(medmodel_frame_from_calib_frame)
  */
  static const auto calib_from_medmodel = (Eigen::Matrix<float, 3, 3>() <<
     0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
     1.09890110e-03, 0.00000000e+00, -2.81318681e-01,
    -2.25466395e-20, 1.09890110e-03,-5.23076923e-02).finished();

  static const auto calib_from_sbigmodel = (Eigen::Matrix<float, 3, 3>() <<
     0.00000000e+00,  7.31372216e-19,  1.00000000e+00,
     2.19780220e-03,  4.11497335e-19, -5.62637363e-01,
    -6.66298828e-20,  2.19780220e-03, -3.33626374e-01).finished();

  static const auto view_from_device = (Eigen::Matrix<float, 3, 3>() <<
     0.0,  1.0,  0.0,
     0.0,  0.0,  1.0,
     1.0,  0.0,  0.0).finished();


  const auto cam_intrinsics = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>(wide_camera ? ECAM_INTRINSIC_MATRIX.v : FCAM_INTRINSIC_MATRIX.v);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor>  device_from_calib = euler2rot(device_from_calib_euler).cast <float> ();
  auto calib_from_model = bigmodel_frame ? calib_from_sbigmodel : calib_from_medmodel;
  auto camera_from_calib = cam_intrinsics * view_from_device * device_from_calib;
  auto warp_matrix = camera_from_calib * calib_from_model;

  mat3 transform = {};
  for (int i=0; i<3*3; i++) {
    transform.v[i] = warp_matrix(i / 3, i % 3);
  }
  return transform;
}


void run_model(ModelState &model, VisionIpcClient &vipc_client_main, VisionIpcClient &vipc_client_extra, bool main_wide_camera, bool use_extra_client) {
  // messaging
  PubMaster pm({"modelV2", "cameraOdometry"});
  SubMaster sm({"lateralPlan", "roadCameraState", "liveCalibration", "driverMonitoringState", "navModel"});

  // setup filter to track dropped frames
  FirstOrderFilter frame_dropped_filter(0., 10., 1. / MODEL_FREQ);

  uint32_t frame_id = 0, last_vipc_frame_id = 0;
  double last = 0;
  uint32_t run_count = 0;

  mat3 model_transform_main = {};
  mat3 model_transform_extra = {};
  bool nav_enabled = false;
  bool live_calib_seen = false;
  float driving_style[DRIVING_STYLE_LEN] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
  float nav_features[NAV_FEATURE_LEN] = {0};

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
    bool is_rhd = ((bool)sm["driverMonitoringState"].getDriverMonitoringState().getIsRHD());
    frame_id = sm["roadCameraState"].getRoadCameraState().getFrameId();
    if (sm.updated("liveCalibration")) {
      auto rpy_calib = sm["liveCalibration"].getLiveCalibration().getRpyCalib();
      Eigen::Vector3d device_from_calib_euler;
      for (int i=0; i<3; i++) {
        device_from_calib_euler(i) = rpy_calib[i];
      }
      model_transform_main = update_calibration(device_from_calib_euler, main_wide_camera, false);
      model_transform_extra = update_calibration(device_from_calib_euler, true, true);
      live_calib_seen = true;
    }

    float vec_desire[DESIRE_LEN] = {0};
    if (desire >= 0 && desire < DESIRE_LEN) {
      vec_desire[desire] = 1.0;
    }

    // Enable/disable nav features
    uint64_t timestamp_llk = sm["navModel"].getNavModel().getLocationMonoTime();
    bool nav_valid = sm["navModel"].getValid() && (nanos_since_boot() - timestamp_llk < 1e9);
    if (!nav_enabled && nav_valid) {
      nav_enabled = true;
    } else if (nav_enabled && !nav_valid) {
      memset(nav_features, 0, sizeof(float)*NAV_FEATURE_LEN);
      nav_enabled = false;
    }

    if (nav_enabled && sm.updated("navModel")) {
      auto nav_model_features = sm["navModel"].getNavModel().getFeatures();
      for (int i=0; i<NAV_FEATURE_LEN; i++) {
        nav_features[i] = nav_model_features[i];
      }
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
    ModelOutput *model_output = model_eval_frame(&model, buf_main, buf_extra, model_transform_main, model_transform_extra, vec_desire, is_rhd, driving_style, nav_features, prepare_only);
    double mt2 = millis_since_boot();
    float model_execution_time = (mt2 - mt1) / 1000.0;

    if (model_output != nullptr) {
      model_publish(&model, pm, meta_main.frame_id, meta_extra.frame_id, frame_id, frame_drop_ratio, *model_output, meta_main.timestamp_eof, timestamp_llk, model_execution_time,
                    nav_enabled, live_calib_seen);
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

  // cl init
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

  // init the models
  ModelState model;
  model_init(&model, device_id, context);
  LOGW("models loaded, modeld starting");

  bool main_wide_camera = false;
  bool use_extra_client = true; // set to false to use single camera
  while (!do_exit) {
    auto streams = VisionIpcClient::getAvailableStreams("camerad", false);
    if (!streams.empty()) {
      use_extra_client = streams.count(VISION_STREAM_WIDE_ROAD) > 0 && streams.count(VISION_STREAM_ROAD) > 0;
      main_wide_camera = streams.count(VISION_STREAM_ROAD) == 0;
      break;
    }

    util::sleep_for(100);
  }

  VisionIpcClient vipc_client_main = VisionIpcClient("camerad", main_wide_camera ? VISION_STREAM_WIDE_ROAD : VISION_STREAM_ROAD, true, device_id, context);
  VisionIpcClient vipc_client_extra = VisionIpcClient("camerad", VISION_STREAM_WIDE_ROAD, false, device_id, context);
  LOGW("vision stream set up, main_wide_camera: %d, use_extra_client: %d", main_wide_camera, use_extra_client);

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
