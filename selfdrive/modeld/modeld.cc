#include <cstdio>
#include <cstdlib>
#include <mutex>

#include <eigen3/Eigen/Dense>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/common/clutil.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/modeld/models/driving.h"

ExitHandler do_exit;

mat3 update_calibration(cereal::LiveCalibrationData::Reader live_calib, int frame_width, bool wide_camera) {
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

  static const auto cam_intrinsics = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>(get_camera_intrinsics(frame_width, wide_camera).v);
  static const mat3 yuv_transform = get_model_yuv_transform(frame_width);

  auto extrinsic_matrix = live_calib.getExtrinsicMatrix();
  Eigen::Matrix<float, 3, 4> extrinsic_matrix_eigen;
  for (int i = 0; i < 4*3; i++) {
    extrinsic_matrix_eigen(i / 4, i % 4) = extrinsic_matrix[i];
  }

  auto camera_frame_from_road_frame = cam_intrinsics * extrinsic_matrix_eigen;
  Eigen::Matrix<float, 3, 3> camera_frame_from_ground;
  camera_frame_from_ground.col(0) = camera_frame_from_road_frame.col(0);
  camera_frame_from_ground.col(1) = camera_frame_from_road_frame.col(1);
  camera_frame_from_ground.col(2) = camera_frame_from_road_frame.col(3);

  auto warp_matrix = camera_frame_from_ground * ground_from_medmodel_frame;
  mat3 transform = {};
  for (int i=0; i<3*3; i++) {
    transform.v[i] = warp_matrix(i / 3, i % 3);
  }
  return matmul3(yuv_transform, transform);
}

void run_model(ModelState &model, VisionIpcClient &vipc_client, bool wide_camera) {
  // messaging
  PubMaster pm({"modelV2", "cameraOdometry"});
  SubMaster sm({"lateralPlan", "roadCameraState", "liveCalibration"});

  // setup filter to track dropped frames
  FirstOrderFilter frame_dropped_filter(0., 10., 1. / MODEL_FREQ);

  uint32_t frame_id = 0, last_vipc_frame_id = 0;
  double last = 0;
  uint32_t run_count = 0;

  mat3 model_transform = {};
  bool live_calib_seen = false;

  while (!do_exit) {
    VisionIpcBufExtra extra = {};
    VisionBuf *buf = vipc_client.recv(&extra);
    if (buf == nullptr) continue;

    // TODO: path planner timeout?
    sm.update(0);
    int desire = ((int)sm["lateralPlan"].getLateralPlan().getDesire());
    frame_id = sm["roadCameraState"].getRoadCameraState().getFrameId();
    if (sm.updated("liveCalibration")) {
      model_transform = update_calibration(sm["liveCalibration"].getLiveCalibration(), buf->width, wide_camera);
      live_calib_seen = true;
    }

    float vec_desire[DESIRE_LEN] = {0};
    if (desire >= 0 && desire < DESIRE_LEN) {
      vec_desire[desire] = 1.0;
    }

    double mt1 = millis_since_boot();
    ModelOutput *model_output = model_eval_frame(&model, buf->buf_cl, buf->width, buf->height,
                                              model_transform, vec_desire);
    double mt2 = millis_since_boot();
    float model_execution_time = (mt2 - mt1) / 1000.0;

    // tracked dropped frames
    uint32_t vipc_dropped_frames = extra.frame_id - last_vipc_frame_id - 1;
    float frames_dropped = frame_dropped_filter.update((float)std::min(vipc_dropped_frames, 10U));
    if (run_count < 10) { // let frame drops warm up
      frame_dropped_filter.reset(0);
      frames_dropped = 0.;
    }
    run_count++;

    float frame_drop_ratio = frames_dropped / (1 + frames_dropped);

    model_publish(pm, extra.frame_id, frame_id, frame_drop_ratio, *model_output, extra.timestamp_eof, model_execution_time,
                  kj::ArrayPtr<const float>(model.output.data(), model.output.size()), live_calib_seen);
    posenet_publish(pm, extra.frame_id, vipc_dropped_frames, *model_output, extra.timestamp_eof, live_calib_seen);

    //printf("model process: %.2fms, from last %.2fms, vipc_frame_id %u, frame_id, %u, frame_drop %.3f\n", mt2 - mt1, mt1 - last, extra.frame_id, frame_id, frame_drop_ratio);
    last = mt1;
    last_vipc_frame_id = extra.frame_id;
  }
}

int main(int argc, char **argv) {
  if (!Hardware::PC()) {
    int ret;
    ret = util::set_realtime_priority(54);
    assert(ret == 0);
    util::set_core_affinity({Hardware::EON() ? 2 : 7});
    assert(ret == 0);
  }

  bool wide_camera = Hardware::TICI() ? Params().getBool("EnableWideCamera") : false;

  // cl init
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

  // init the models
  ModelState model;
  model_init(&model, device_id, context);
  LOGW("models loaded, modeld starting");

  VisionIpcClient vipc_client = VisionIpcClient("camerad", wide_camera ? VISION_STREAM_WIDE_ROAD : VISION_STREAM_ROAD, true, device_id, context);
  while (!do_exit && !vipc_client.connect(false)) {
    util::sleep_for(100);
  }

  // run the models
  // vipc_client.connected is false only when do_exit is true
  if (vipc_client.connected) {
    const VisionBuf *b = &vipc_client.buffers[0];
    LOGW("connected with buffer size: %d (%d x %d)", b->len, b->width, b->height);
    run_model(model, vipc_client, wide_camera);
  }

  model_free(&model);
  CL_CHECK(clReleaseContext(context));
  return 0;
}
