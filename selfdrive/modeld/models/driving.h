#pragma once

#include <array>
#include <memory>

#include "cereal/messaging/messaging.h"
#include "common/mat.h"
#include "common/modeldata.h"
#include "common/util.h"
#include "selfdrive/modeld/models/nav.h"

#ifdef USE_THNEED
  constexpr bool CPP_USE_THNEED = true;
#else
  constexpr bool CPP_USE_THNEED = false;
#endif

constexpr int FEATURE_LEN = 128;
constexpr int HISTORY_BUFFER_LEN = 99;
constexpr int DESIRE_LEN = 8;
constexpr int DESIRE_PRED_LEN = 4;
constexpr int TRAFFIC_CONVENTION_LEN = 2;
constexpr int DRIVING_STYLE_LEN = 12;
constexpr int MODEL_FREQ = 20;

constexpr int DISENGAGE_LEN = 5;
constexpr int BLINKER_LEN = 6;
constexpr int META_STRIDE = 7;

constexpr int PLAN_MHP_N = 5;
constexpr int LEAD_MHP_N = 2;
constexpr int LEAD_TRAJ_LEN = 6;
constexpr int LEAD_MHP_SELECTION = 3;
// Padding to get output shape as multiple of 4
constexpr int PAD_SIZE = 2;

constexpr float FCW_THRESHOLD_5MS2_HIGH = 0.15;
constexpr float FCW_THRESHOLD_5MS2_LOW = 0.05;
constexpr float FCW_THRESHOLD_3MS2 = 0.7;

struct ModelOutputXYZ {
  float x;
  float y;
  float z;
};
static_assert(sizeof(ModelOutputXYZ) == sizeof(float)*3);

struct ModelOutputYZ {
  float y;
  float z;
};
static_assert(sizeof(ModelOutputYZ) == sizeof(float)*2);

struct ModelOutputPlanElement {
  ModelOutputXYZ position;
  ModelOutputXYZ velocity;
  ModelOutputXYZ acceleration;
  ModelOutputXYZ rotation;
  ModelOutputXYZ rotation_rate;
};
static_assert(sizeof(ModelOutputPlanElement) == sizeof(ModelOutputXYZ)*5);

struct ModelOutputPlanPrediction {
  std::array<ModelOutputPlanElement, TRAJECTORY_SIZE> mean;
  std::array<ModelOutputPlanElement, TRAJECTORY_SIZE> std;
  float prob;
};
static_assert(sizeof(ModelOutputPlanPrediction) == (sizeof(ModelOutputPlanElement)*TRAJECTORY_SIZE*2) + sizeof(float));

struct ModelOutputPlans {
  std::array<ModelOutputPlanPrediction, PLAN_MHP_N> prediction;

  constexpr const ModelOutputPlanPrediction &get_best_prediction() const {
    int max_idx = 0;
    for (int i = 1; i < prediction.size(); i++) {
      if (prediction[i].prob > prediction[max_idx].prob) {
        max_idx = i;
      }
    }
    return prediction[max_idx];
  }
};
static_assert(sizeof(ModelOutputPlans) == sizeof(ModelOutputPlanPrediction)*PLAN_MHP_N);

struct ModelOutputLinesXY {
  std::array<ModelOutputYZ, TRAJECTORY_SIZE> left_far;
  std::array<ModelOutputYZ, TRAJECTORY_SIZE> left_near;
  std::array<ModelOutputYZ, TRAJECTORY_SIZE> right_near;
  std::array<ModelOutputYZ, TRAJECTORY_SIZE> right_far;
};
static_assert(sizeof(ModelOutputLinesXY) == sizeof(ModelOutputYZ)*TRAJECTORY_SIZE*4);

struct ModelOutputLineProbVal {
  float val_deprecated;
  float val;
};
static_assert(sizeof(ModelOutputLineProbVal) == sizeof(float)*2);

struct ModelOutputLinesProb {
  ModelOutputLineProbVal left_far;
  ModelOutputLineProbVal left_near;
  ModelOutputLineProbVal right_near;
  ModelOutputLineProbVal right_far;
};
static_assert(sizeof(ModelOutputLinesProb) == sizeof(ModelOutputLineProbVal)*4);

struct ModelOutputLaneLines {
  ModelOutputLinesXY mean;
  ModelOutputLinesXY std;
  ModelOutputLinesProb prob;
};
static_assert(sizeof(ModelOutputLaneLines) == (sizeof(ModelOutputLinesXY)*2) + sizeof(ModelOutputLinesProb));

struct ModelOutputEdgessXY {
  std::array<ModelOutputYZ, TRAJECTORY_SIZE> left;
  std::array<ModelOutputYZ, TRAJECTORY_SIZE> right;
};
static_assert(sizeof(ModelOutputEdgessXY) == sizeof(ModelOutputYZ)*TRAJECTORY_SIZE*2);

struct ModelOutputRoadEdges {
  ModelOutputEdgessXY mean;
  ModelOutputEdgessXY std;
};
static_assert(sizeof(ModelOutputRoadEdges) == (sizeof(ModelOutputEdgessXY)*2));

struct ModelOutputLeadElement {
  float x;
  float y;
  float velocity;
  float acceleration;
};
static_assert(sizeof(ModelOutputLeadElement) == sizeof(float)*4);

struct ModelOutputLeadPrediction {
  std::array<ModelOutputLeadElement, LEAD_TRAJ_LEN> mean;
  std::array<ModelOutputLeadElement, LEAD_TRAJ_LEN> std;
  std::array<float, LEAD_MHP_SELECTION> prob;
};
static_assert(sizeof(ModelOutputLeadPrediction) == (sizeof(ModelOutputLeadElement)*LEAD_TRAJ_LEN*2) + (sizeof(float)*LEAD_MHP_SELECTION));

struct ModelOutputLeads {
  std::array<ModelOutputLeadPrediction, LEAD_MHP_N> prediction;
  std::array<float, LEAD_MHP_SELECTION> prob;

  constexpr const ModelOutputLeadPrediction &get_best_prediction(int t_idx) const {
    int max_idx = 0;
    for (int i = 1; i < prediction.size(); i++) {
      if (prediction[i].prob[t_idx] > prediction[max_idx].prob[t_idx]) {
        max_idx = i;
      }
    }
    return prediction[max_idx];
  }
};
static_assert(sizeof(ModelOutputLeads) == (sizeof(ModelOutputLeadPrediction)*LEAD_MHP_N) + (sizeof(float)*LEAD_MHP_SELECTION));


struct ModelOutputPose {
  ModelOutputXYZ velocity_mean;
  ModelOutputXYZ rotation_mean;
  ModelOutputXYZ velocity_std;
  ModelOutputXYZ rotation_std;
};
static_assert(sizeof(ModelOutputPose) == sizeof(ModelOutputXYZ)*4);

struct ModelOutputWideFromDeviceEuler {
  ModelOutputXYZ mean;
  ModelOutputXYZ std;
};
static_assert(sizeof(ModelOutputWideFromDeviceEuler) == sizeof(ModelOutputXYZ)*2);

struct ModelOutputTemporalPose {
  ModelOutputXYZ velocity_mean;
  ModelOutputXYZ rotation_mean;
  ModelOutputXYZ velocity_std;
  ModelOutputXYZ rotation_std;
};
static_assert(sizeof(ModelOutputTemporalPose) == sizeof(ModelOutputXYZ)*4);

struct ModelOutputRoadTransform {
  ModelOutputXYZ position_mean;
  ModelOutputXYZ rotation_mean;
  ModelOutputXYZ position_std;
  ModelOutputXYZ rotation_std;
};
static_assert(sizeof(ModelOutputRoadTransform) == sizeof(ModelOutputXYZ)*4);

struct ModelOutputDisengageProb {
  float gas_disengage;
  float brake_disengage;
  float steer_override;
  float brake_3ms2;
  float brake_4ms2;
  float brake_5ms2;
  float gas_pressed;
};
static_assert(sizeof(ModelOutputDisengageProb) == sizeof(float)*7);

struct ModelOutputBlinkerProb {
  float left;
  float right;
};
static_assert(sizeof(ModelOutputBlinkerProb) == sizeof(float)*2);

struct ModelOutputDesireProb {
  union {
    struct {
      float none;
      float turn_left;
      float turn_right;
      float lane_change_left;
      float lane_change_right;
      float keep_left;
      float keep_right;
      float null;
    };
    struct {
      std::array<float, DESIRE_LEN> array;
    };
  };
};
static_assert(sizeof(ModelOutputDesireProb) == sizeof(float)*DESIRE_LEN);

struct ModelOutputMeta {
  ModelOutputDesireProb desire_state_prob;
  float engaged_prob;
  std::array<ModelOutputDisengageProb, DISENGAGE_LEN> disengage_prob;
  std::array<ModelOutputBlinkerProb, BLINKER_LEN> blinker_prob;
  std::array<ModelOutputDesireProb, DESIRE_PRED_LEN> desire_pred_prob;
};
static_assert(sizeof(ModelOutputMeta) == sizeof(ModelOutputDesireProb) + sizeof(float) + (sizeof(ModelOutputDisengageProb)*DISENGAGE_LEN) + (sizeof(ModelOutputBlinkerProb)*BLINKER_LEN) + (sizeof(ModelOutputDesireProb)*DESIRE_PRED_LEN));

struct ModelOutputFeatures {
  std::array<float, FEATURE_LEN> feature;
};
static_assert(sizeof(ModelOutputFeatures) == (sizeof(float)*FEATURE_LEN));

struct ModelOutput {
  const ModelOutputPlans plans;
  const ModelOutputLaneLines lane_lines;
  const ModelOutputRoadEdges road_edges;
  const ModelOutputLeads leads;
  const ModelOutputMeta meta;
  const ModelOutputPose pose;
  const ModelOutputWideFromDeviceEuler wide_from_device_euler;
  const ModelOutputTemporalPose temporal_pose;
  const ModelOutputRoadTransform road_transform;
};

constexpr int OUTPUT_SIZE = sizeof(ModelOutput) / sizeof(float);
constexpr int NET_OUTPUT_SIZE = OUTPUT_SIZE + FEATURE_LEN + PAD_SIZE;

struct PublishState {
  std::array<float, DISENGAGE_LEN * DISENGAGE_LEN> disengage_buffer = {};
  std::array<float, 5> prev_brake_5ms2_probs = {};
  std::array<float, 3> prev_brake_3ms2_probs = {};
};

mat3 update_calibration(float *device_from_calib_euler, bool wide_camera, bool bigmodel_frame);
void fill_model_msg(MessageBuilder &msg, float *net_output_data, PublishState &ps, uint32_t vipc_frame_id, uint32_t vipc_frame_id_extra, uint32_t frame_id, float frame_drop,
                    uint64_t timestamp_eof, uint64_t timestamp_llk, float model_execution_time, const bool nav_enabled, const bool valid);
void fill_pose_msg(MessageBuilder &msg, float *net_outputs, uint32_t vipc_frame_id, uint32_t vipc_dropped_frames, uint64_t timestamp_eof, const bool valid);
