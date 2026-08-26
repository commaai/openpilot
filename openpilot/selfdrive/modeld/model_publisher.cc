#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include "cereal/messaging/messaging.h"

namespace {

using half = _Float16;

constexpr size_t IDX_N = 33;
constexpr size_t PLAN_WIDTH = 15;
constexpr size_t LINE_WIDTH = 2;
constexpr size_t LEAD_LEN = 6;
constexpr size_t LEAD_WIDTH = 4;
constexpr size_t DESIRE_LEN = 8;
constexpr size_t DISENGAGE_LEN = 5;

const std::array<float, IDX_N> T_IDXS = []() {
  std::array<float, IDX_N> values = {};
  for (size_t i = 0; i < values.size(); ++i) {
    const double x = static_cast<double>(i) / (IDX_N - 1);
    values[i] = static_cast<float>(10.0 * x * x);
  }
  return values;
}();

const std::array<float, IDX_N> X_IDXS = []() {
  std::array<float, IDX_N> values = {};
  for (size_t i = 0; i < values.size(); ++i) {
    const double x = static_cast<double>(i) / (IDX_N - 1);
    values[i] = static_cast<float>(192.0 * x * x);
  }
  return values;
}();

constexpr std::array<float, LEAD_LEN> LEAD_T_IDXS = {0., 2., 4., 6., 8., 10.};
constexpr std::array<float, 3> LEAD_T_OFFSETS = {0., 2., 4.};
constexpr std::array<float, DISENGAGE_LEN> META_T_IDXS = {2., 4., 6., 8., 10.};
constexpr std::array<float, 5> FCW_THRESHOLDS_5MS2 = {.05, .05, .15, .15, .15};
constexpr std::array<float, 2> FCW_THRESHOLDS_3MS2 = {.7, .7};

struct ModelOutputs {
  const half *plan;
  const half *plan_stds;
  const half *lane_lines;
  const half *lane_lines_stds;
  const half *lane_lines_prob;
  const half *road_edges;
  const half *road_edges_stds;
  const half *lead;
  const half *lead_stds;
  const half *lead_prob;
  const half *desire_state;
  const half *desire_pred;
  const half *meta;
  const half *pose;
  const half *pose_stds;
  const half *wide_from_device_euler;
  const half *wide_from_device_euler_stds;
  const half *road_transform;
  const half *road_transform_stds;
  const void *raw_pred;
  size_t raw_pred_size;
};

struct PublishData {
  uint64_t timestamp_eof;
  uint32_t vipc_frame_id;
  uint32_t vipc_frame_id_extra;
  uint32_t frame_id;
  float frame_drop_perc;
  float model_execution_time;
  float desired_curvature;
  float desired_acceleration;
  uint8_t valid;
  uint8_t camera_odometry_valid;
  uint8_t big;
  uint8_t should_stop;
  uint8_t lane_change_state;
  uint8_t lane_change_direction;
};

template <typename T>
void fill_list(capnp::List<float>::Builder list, const T *values, size_t stride = 1) {
  for (size_t i = 0; i < list.size(); ++i) {
    list.set(i, static_cast<float>(values[i * stride]));
  }
}

void fill_xyzt(cereal::XYZTData::Builder xyzt, const half *values, size_t offset, const half *stds = nullptr) {
  xyzt.setT(kj::arrayPtr(T_IDXS.data(), T_IDXS.size()));
  fill_list(xyzt.initX(IDX_N), values + offset, PLAN_WIDTH);
  fill_list(xyzt.initY(IDX_N), values + offset + 1, PLAN_WIDTH);
  fill_list(xyzt.initZ(IDX_N), values + offset + 2, PLAN_WIDTH);
  if (stds != nullptr) {
    fill_list(xyzt.initXStd(IDX_N), stds + offset, PLAN_WIDTH);
    fill_list(xyzt.initYStd(IDX_N), stds + offset + 1, PLAN_WIDTH);
    fill_list(xyzt.initZStd(IDX_N), stds + offset + 2, PLAN_WIDTH);
  }
}

void fill_line(cereal::XYZTData::Builder line, const half *values) {
  line.initT(0);
  line.setX(kj::arrayPtr(X_IDXS.data(), X_IDXS.size()));
  fill_list(line.initY(IDX_N), values, LINE_WIDTH);
  fill_list(line.initZ(IDX_N), values + 1, LINE_WIDTH);
}

void fill_action(cereal::ModelDataV2::Action::Builder action, const PublishData &data) {
  action.setDesiredCurvature(data.desired_curvature);
  action.setDesiredAcceleration(data.desired_acceleration);
  action.setShouldStop(data.should_stop);
}

struct PublishState {
  std::array<float, DISENGAGE_LEN * DISENGAGE_LEN> disengage_buffer = {};
  std::array<float, 5> prev_brake_5ms2_probs = {};
  std::array<float, 2> prev_brake_3ms2_probs = {};
};

void fill_meta(cereal::ModelDataV2::Builder model, const ModelOutputs &outputs, const PublishData &data, PublishState &state) {
  auto meta = model.initMeta();
  meta.setEngagedProb(static_cast<float>(outputs.meta[0]));
  fill_list(meta.initDesireState(DESIRE_LEN), outputs.desire_state);
  fill_list(meta.initDesirePrediction(4 * DESIRE_LEN), outputs.desire_pred);

  auto disengage = meta.initDisengagePredictions();
  disengage.setT(kj::arrayPtr(META_T_IDXS.data(), META_T_IDXS.size()));
  fill_list(disengage.initGasDisengageProbs(DISENGAGE_LEN), outputs.meta + 1, 6);
  fill_list(disengage.initBrakeDisengageProbs(DISENGAGE_LEN), outputs.meta + 2, 6);
  fill_list(disengage.initSteerOverrideProbs(DISENGAGE_LEN), outputs.meta + 3, 6);
  fill_list(disengage.initBrake3MetersPerSecondSquaredProbs(DISENGAGE_LEN), outputs.meta + 4, 6);
  fill_list(disengage.initBrake4MetersPerSecondSquaredProbs(DISENGAGE_LEN), outputs.meta + 5, 6);
  fill_list(disengage.initBrake5MetersPerSecondSquaredProbs(DISENGAGE_LEN), outputs.meta + 6, 6);
  fill_list(disengage.initGasPressProbs(6), outputs.meta + 31, 4);
  fill_list(disengage.initBrakePressProbs(6), outputs.meta + 32, 4);

  for (size_t i = 1; i < state.prev_brake_5ms2_probs.size(); ++i) {
    state.prev_brake_5ms2_probs[i - 1] = state.prev_brake_5ms2_probs[i];
  }
  state.prev_brake_5ms2_probs.back() = static_cast<float>(outputs.meta[6]);
  for (size_t i = 1; i < state.prev_brake_3ms2_probs.size(); ++i) {
    state.prev_brake_3ms2_probs[i - 1] = state.prev_brake_3ms2_probs[i];
  }
  state.prev_brake_3ms2_probs.back() = static_cast<float>(outputs.meta[4]);

  bool hard_brake = true;
  for (size_t i = 0; i < state.prev_brake_5ms2_probs.size(); ++i) {
    hard_brake &= state.prev_brake_5ms2_probs[i] > FCW_THRESHOLDS_5MS2[i];
  }
  for (size_t i = 0; i < state.prev_brake_3ms2_probs.size(); ++i) {
    hard_brake &= state.prev_brake_3ms2_probs[i] > FCW_THRESHOLDS_3MS2[i];
  }
  meta.setHardBrakePredicted(hard_brake);

  if (data.vipc_frame_id % 40 == 0) {
    for (size_t i = 0; i < state.disengage_buffer.size() - DISENGAGE_LEN; ++i) {
      state.disengage_buffer[i] = state.disengage_buffer[i + DISENGAGE_LEN];
    }
    half previous_any = 0;
    for (size_t i = 0; i < DISENGAGE_LEN; ++i) {
      const half one_minus_brake = static_cast<half>(static_cast<half>(1.) - outputs.meta[2 + i * 6]);
      const half one_minus_gas = static_cast<half>(static_cast<half>(1.) - outputs.meta[1 + i * 6]);
      const half one_minus_steer = static_cast<half>(static_cast<half>(1.) - outputs.meta[3 + i * 6]);
      const half product = static_cast<half>(static_cast<half>(one_minus_brake * one_minus_gas) * one_minus_steer);
      const half any = static_cast<half>(static_cast<half>(1.) - product);
      const half independent = i == 0 ? any : static_cast<half>(static_cast<half>(any - previous_any) /
                                                                static_cast<half>(static_cast<half>(1.) - previous_any));
      state.disengage_buffer[state.disengage_buffer.size() - DISENGAGE_LEN + i] = static_cast<float>(independent);
      previous_any = any;
    }
  }

  double score = 0.;
  for (size_t i = 0; i < DISENGAGE_LEN; ++i) {
    score += state.disengage_buffer[i * DISENGAGE_LEN + DISENGAGE_LEN - 1 - i] / DISENGAGE_LEN;
  }
  model.setConfidence(score < .01165 ? cereal::ModelDataV2::ConfidenceClass::GREEN :
                      score < .06157 ? cereal::ModelDataV2::ConfidenceClass::YELLOW :
                                       cereal::ModelDataV2::ConfidenceClass::RED);
  meta.setLaneChangeState(static_cast<cereal::LaneChangeState>(data.lane_change_state));
  meta.setLaneChangeDirection(static_cast<cereal::LaneChangeDirection>(data.lane_change_direction));
}

struct ModelPublisher {
  ModelPublisher() : pm({"modelV2", "drivingModelData", "cameraOdometry"}) {}

  void publish(const ModelOutputs &outputs, const PublishData &data, const double *path_coefficients) {
    MessageBuilder model_msg;
    MessageBuilder driving_msg;
    MessageBuilder odometry_msg;
    auto model = model_msg.initEvent(data.valid).initModelV2();
    auto driving = driving_msg.initEvent(data.valid).initDrivingModelData();
    auto odometry = odometry_msg.initEvent(data.camera_odometry_valid).initCameraOdometry();

    model.setFrameId(data.vipc_frame_id);
    model.setFrameIdExtra(data.vipc_frame_id_extra);
    model.setFrameAge(data.frame_id > data.vipc_frame_id ? data.frame_id - data.vipc_frame_id : 0);
    model.setFrameDropPerc(data.frame_drop_perc);
    model.setTimestampEof(data.timestamp_eof);
    model.setModelExecutionTime(data.model_execution_time);
    model.setBig(data.big);
    fill_xyzt(model.initPosition(), outputs.plan, 0, outputs.plan_stds);
    fill_xyzt(model.initVelocity(), outputs.plan, 3);
    fill_xyzt(model.initAcceleration(), outputs.plan, 6);
    fill_xyzt(model.initOrientation(), outputs.plan, 9);
    fill_xyzt(model.initOrientationRate(), outputs.plan, 12);
    fill_action(model.initAction(), data);

    auto lane_lines = model.initLaneLines(4);
    for (size_t i = 0; i < lane_lines.size(); ++i) {
      fill_line(lane_lines[i], outputs.lane_lines + i * IDX_N * LINE_WIDTH);
    }
    fill_list(model.initLaneLineStds(4), outputs.lane_lines_stds, IDX_N * LINE_WIDTH);
    fill_list(model.initLaneLineProbs(4), outputs.lane_lines_prob + 1, 2);

    auto road_edges = model.initRoadEdges(2);
    for (size_t i = 0; i < road_edges.size(); ++i) {
      fill_line(road_edges[i], outputs.road_edges + i * IDX_N * LINE_WIDTH);
    }
    fill_list(model.initRoadEdgeStds(2), outputs.road_edges_stds, IDX_N * LINE_WIDTH);

    auto leads = model.initLeadsV3(3);
    for (size_t i = 0; i < leads.size(); ++i) {
      const half *lead = outputs.lead + i * LEAD_LEN * LEAD_WIDTH;
      const half *lead_stds = outputs.lead_stds + i * LEAD_LEN * LEAD_WIDTH;
      leads[i].setT(kj::arrayPtr(LEAD_T_IDXS.data(), LEAD_T_IDXS.size()));
      fill_list(leads[i].initX(LEAD_LEN), lead, LEAD_WIDTH);
      fill_list(leads[i].initY(LEAD_LEN), lead + 1, LEAD_WIDTH);
      fill_list(leads[i].initV(LEAD_LEN), lead + 2, LEAD_WIDTH);
      fill_list(leads[i].initA(LEAD_LEN), lead + 3, LEAD_WIDTH);
      fill_list(leads[i].initXStd(LEAD_LEN), lead_stds, LEAD_WIDTH);
      fill_list(leads[i].initYStd(LEAD_LEN), lead_stds + 1, LEAD_WIDTH);
      fill_list(leads[i].initVStd(LEAD_LEN), lead_stds + 2, LEAD_WIDTH);
      fill_list(leads[i].initAStd(LEAD_LEN), lead_stds + 3, LEAD_WIDTH);
      leads[i].setProb(static_cast<float>(outputs.lead_prob[i]));
      leads[i].setProbTime(LEAD_T_OFFSETS[i]);
    }
    fill_meta(model, outputs, data, state);
    if (outputs.raw_pred != nullptr) {
      model.setRawPredictions(kj::arrayPtr(reinterpret_cast<const capnp::byte *>(outputs.raw_pred), outputs.raw_pred_size));
    }

    driving.setFrameId(data.vipc_frame_id);
    driving.setFrameIdExtra(data.vipc_frame_id_extra);
    driving.setFrameDropPerc(data.frame_drop_perc);
    driving.setModelExecutionTime(data.model_execution_time);
    fill_action(driving.initAction(), data);
    auto lane_meta = driving.initLaneLineMeta();
    lane_meta.setLeftY(static_cast<float>(outputs.lane_lines[IDX_N * LINE_WIDTH]));
    lane_meta.setRightY(static_cast<float>(outputs.lane_lines[2 * IDX_N * LINE_WIDTH]));
    lane_meta.setLeftProb(static_cast<float>(outputs.lane_lines_prob[3]));
    lane_meta.setRightProb(static_cast<float>(outputs.lane_lines_prob[5]));
    auto driving_meta = driving.initMeta();
    driving_meta.setLaneChangeState(static_cast<cereal::LaneChangeState>(data.lane_change_state));
    driving_meta.setLaneChangeDirection(static_cast<cereal::LaneChangeDirection>(data.lane_change_direction));
    auto path = driving.initPath();
    fill_list(path.initXCoefficients(5), path_coefficients, 3);
    fill_list(path.initYCoefficients(5), path_coefficients + 1, 3);
    fill_list(path.initZCoefficients(5), path_coefficients + 2, 3);

    odometry.setFrameId(data.vipc_frame_id);
    odometry.setTimestampEof(data.timestamp_eof);
    fill_list(odometry.initTrans(3), outputs.pose);
    fill_list(odometry.initRot(3), outputs.pose + 3);
    fill_list(odometry.initWideFromDeviceEuler(3), outputs.wide_from_device_euler);
    fill_list(odometry.initRoadTransformTrans(3), outputs.road_transform);
    fill_list(odometry.initTransStd(3), outputs.pose_stds);
    fill_list(odometry.initRotStd(3), outputs.pose_stds + 3);
    fill_list(odometry.initWideFromDeviceEulerStd(3), outputs.wide_from_device_euler_stds);
    fill_list(odometry.initRoadTransformTransStd(3), outputs.road_transform_stds);

    pm.send("modelV2", model_msg);
    pm.send("drivingModelData", driving_msg);
    pm.send("cameraOdometry", odometry_msg);
  }

  PubMaster pm;
  PublishState state;
};

thread_local std::string last_error;

}  // namespace

extern "C" {

ModelPublisher *model_publisher_create() noexcept {
  last_error.clear();
  try {
    return new ModelPublisher;
  } catch (const std::exception &e) {
    last_error = e.what();
  } catch (...) {
    last_error = "unknown C++ exception";
  }
  return nullptr;
}

void model_publisher_destroy(ModelPublisher *publisher) noexcept {
  delete publisher;
}

bool model_publisher_publish(ModelPublisher *publisher, const ModelOutputs *outputs,
                             const PublishData *data, const double *path_coefficients) noexcept {
  last_error.clear();
  try {
    publisher->publish(*outputs, *data, path_coefficients);
    return true;
  } catch (const std::exception &e) {
    last_error = e.what();
  } catch (...) {
    last_error = "unknown C++ exception";
  }
  return false;
}

const char *model_publisher_last_error() noexcept {
  return last_error.c_str();
}

}  // extern "C"
