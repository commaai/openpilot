#pragma once

#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "common/modeldata.h"
#include "selfdrive/modeld/models/commonmodel.h"
#include "selfdrive/modeld/runners/run.h"

constexpr int INPUT_SIZE = 256*256;
constexpr int FEATURE_LEN = 64;
constexpr int DESIRE_LEN = 32;
constexpr int PLAN_MHP_N = 5;

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
  std::array<NavModelOutputPlan, PLAN_MHP_N> prediction;

  constexpr const NavModelOutputPlan &get_best_prediction() const {
    int max_idx = 0;
    for (int i = 1; i < prediction.size(); i++) {
      if (prediction[i].prob > prediction[max_idx].prob) {
        max_idx = i;
      }
    }
    return prediction[max_idx];
  }
};
static_assert(sizeof(NavModelOutputPlans) == sizeof(NavModelOutputPlan)*PLAN_MHP_N);

struct NavModelOutputDesirePrediction {
  std::array<float, DESIRE_LEN> values;
};
static_assert(sizeof(NavModelOutputDesirePrediction) == sizeof(float)*DESIRE_LEN);

struct NavModelOutputFeatures {
  std::array<float, FEATURE_LEN> values;
};
static_assert(sizeof(NavModelOutputFeatures) == sizeof(float)*FEATURE_LEN);

struct NavModelResult {
  const NavModelOutputPlans plans;
  const NavModelOutputDesirePrediction desire_pred;
  const NavModelOutputFeatures features;
  float dsp_execution_time;
};
static_assert(sizeof(NavModelResult) == sizeof(NavModelOutputPlans) + sizeof(NavModelOutputDesirePrediction) + sizeof(NavModelOutputFeatures) + sizeof(float));

constexpr int OUTPUT_SIZE = sizeof(NavModelResult) / sizeof(float);
constexpr int NET_OUTPUT_SIZE = OUTPUT_SIZE - 1;

struct NavModelState {
  RunModel *m;
  float net_input_buf[INPUT_SIZE];  // TODO: make this uint8_t
  float output[OUTPUT_SIZE];
};

void navmodel_init(NavModelState* s);
NavModelResult* navmodel_eval_frame(NavModelState* s, VisionBuf* buf);
void navmodel_publish(PubMaster &pm, uint32_t frame_id, const NavModelResult &model_res, float execution_time);
void navmodel_free(NavModelState* s);
