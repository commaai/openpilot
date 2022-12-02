#pragma once

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionipc_client.h"
#include "common/util.h"
#include "common/modeldata.h"
#include "selfdrive/modeld/models/commonmodel.h"
#include "selfdrive/modeld/runners/run.h"

constexpr int NAV_INPUT_SIZE = 256*256;
constexpr int NAV_FEATURE_LEN = 64;
constexpr int NAV_DESIRE_LEN = 32;
constexpr int NAV_PLAN_MHP_N = 5;

struct NavModelOutputXY {
  float x;
  float y;
};
static_assert(sizeof(NavModelOutputXY) == sizeof(float)*2);

struct NavModelOutputPlan {
  std::array<NavModelOutputXY, TRAJECTORY_SIZE> mean;
  std::array<NavModelOutputXY, TRAJECTORY_SIZE> std;
  float prob;
};
static_assert(sizeof(NavModelOutputPlan) == sizeof(NavModelOutputXY)*TRAJECTORY_SIZE*2 + sizeof(float));

struct NavModelOutputPlans {
  std::array<NavModelOutputPlan, NAV_PLAN_MHP_N> predictions;

  constexpr const NavModelOutputPlan &get_best_prediction() const {
    int max_idx = 0;
    for (int i = 1; i < predictions.size(); i++) {
      if (predictions[i].prob > predictions[max_idx].prob) {
        max_idx = i;
      }
    }
    return predictions[max_idx];
  }
};
static_assert(sizeof(NavModelOutputPlans) == sizeof(NavModelOutputPlan)*NAV_PLAN_MHP_N);

struct NavModelOutputDesirePrediction {
  std::array<float, NAV_DESIRE_LEN> values;
};
static_assert(sizeof(NavModelOutputDesirePrediction) == sizeof(float)*NAV_DESIRE_LEN);

struct NavModelOutputFeatures {
  std::array<float, NAV_FEATURE_LEN> values;
};
static_assert(sizeof(NavModelOutputFeatures) == sizeof(float)*NAV_FEATURE_LEN);

struct NavModelResult {
  const NavModelOutputPlans plans;
  const NavModelOutputDesirePrediction desire_pred;
  const NavModelOutputFeatures features;
  float dsp_execution_time;
};
static_assert(sizeof(NavModelResult) == sizeof(NavModelOutputPlans) + sizeof(NavModelOutputDesirePrediction) + sizeof(NavModelOutputFeatures) + sizeof(float));

constexpr int NAV_OUTPUT_SIZE = sizeof(NavModelResult) / sizeof(float);
constexpr int NAV_NET_OUTPUT_SIZE = NAV_OUTPUT_SIZE - 1;

struct NavModelState {
  RunModel *m;
  uint8_t net_input_buf[NAV_INPUT_SIZE];
  float output[NAV_OUTPUT_SIZE];
};

void navmodel_init(NavModelState* s);
NavModelResult* navmodel_eval_frame(NavModelState* s, VisionBuf* buf);
void navmodel_publish(PubMaster &pm, uint32_t frame_id, const NavModelResult &model_res, float execution_time);
void navmodel_free(NavModelState* s);
