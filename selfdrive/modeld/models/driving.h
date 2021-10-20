#pragma once

// gate this here
#define TEMPORAL
#define DESIRE
#define TRAFFIC_CONVENTION

#include <array>
#include <memory>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/modeldata.h"
#include "selfdrive/common/util.h"
#include "selfdrive/modeld/models/commonmodel.h"
#include "selfdrive/modeld/runners/run.h"

constexpr int DESIRE_LEN = 8;
constexpr int TRAFFIC_CONVENTION_LEN = 2;
constexpr int MODEL_FREQ = 20;

constexpr int DESIRE_PRED_SIZE = 32;
constexpr int OTHER_META_SIZE = 48;
constexpr int NUM_META_INTERVALS = 5;
constexpr int META_STRIDE = 7;

constexpr int PLAN_MHP_N = 5;
constexpr int PLAN_MHP_VALS = 15*33;
constexpr int PLAN_MHP_SELECTION = 1;
constexpr int PLAN_MHP_GROUP_SIZE =  (2*PLAN_MHP_VALS + PLAN_MHP_SELECTION);

constexpr int LEAD_MHP_N = 2;
constexpr int LEAD_TRAJ_LEN = 6;
constexpr int LEAD_PRED_DIM = 4;
constexpr int LEAD_MHP_VALS = LEAD_PRED_DIM*LEAD_TRAJ_LEN;
constexpr int LEAD_MHP_SELECTION = 3;
constexpr int LEAD_MHP_GROUP_SIZE = (2*LEAD_MHP_VALS + LEAD_MHP_SELECTION);

constexpr int POSE_SIZE = 12;

constexpr int PLAN_IDX = 0;
constexpr int LL_IDX = PLAN_IDX + PLAN_MHP_N*PLAN_MHP_GROUP_SIZE;
constexpr int LL_PROB_IDX = LL_IDX + 4*2*2*33;
constexpr int RE_IDX = LL_PROB_IDX + 8;
constexpr int LEAD_IDX = RE_IDX + 2*2*2*33;
constexpr int LEAD_PROB_IDX = LEAD_IDX + LEAD_MHP_N*(LEAD_MHP_GROUP_SIZE);
constexpr int DESIRE_STATE_IDX = LEAD_PROB_IDX + 3;
constexpr int META_IDX = DESIRE_STATE_IDX + DESIRE_LEN;
constexpr int POSE_IDX = META_IDX + OTHER_META_SIZE + DESIRE_PRED_SIZE;
constexpr int OUTPUT_SIZE =  POSE_IDX + POSE_SIZE;
#ifdef TEMPORAL
  constexpr int TEMPORAL_SIZE = 512;
#else
  constexpr int TEMPORAL_SIZE = 0;
#endif
constexpr int NET_OUTPUT_SIZE =  OUTPUT_SIZE + TEMPORAL_SIZE;

struct ModelDataRawXYZ {
  float x;
  float y;
  float z;
};
static_assert(sizeof(ModelDataRawXYZ) == sizeof(float)*3);

struct ModelDataRawYZ {
  float y;
  float z;
};
static_assert(sizeof(ModelDataRawYZ) == sizeof(float)*2);

struct ModelDataRawPlanTimeStep {
  ModelDataRawXYZ position;
  ModelDataRawXYZ velocity;
  ModelDataRawXYZ acceleration;
  ModelDataRawXYZ rotation;
  ModelDataRawXYZ rotation_rate;
};
static_assert(sizeof(ModelDataRawPlanTimeStep) == sizeof(ModelDataRawXYZ)*5);

struct ModelDataRawPlanPath {
  std::array<ModelDataRawPlanTimeStep, TRAJECTORY_SIZE> mean;
  std::array<ModelDataRawPlanTimeStep, TRAJECTORY_SIZE> std;
  float prob;
};
static_assert(sizeof(ModelDataRawPlanPath) == (sizeof(ModelDataRawPlanTimeStep)*TRAJECTORY_SIZE*2) + sizeof(float));

struct ModelDataRawPlans {
  std::array<ModelDataRawPlanPath, PLAN_MHP_N> path;

  constexpr const ModelDataRawPlanPath &get_best_plan() const {
    int max_idx = 0;
    for (int i = 1; i < path.size(); i++) {
      if (path[i].prob > path[max_idx].prob) {
        max_idx = i;
      }
    }
    return path[max_idx];
  }
};
static_assert(sizeof(ModelDataRawPlans) == sizeof(ModelDataRawPlanPath)*PLAN_MHP_N);

struct ModelDataRawLinesXY {
  std::array<ModelDataRawYZ, TRAJECTORY_SIZE> left_far;
  std::array<ModelDataRawYZ, TRAJECTORY_SIZE> left_near;
  std::array<ModelDataRawYZ, TRAJECTORY_SIZE> right_near;
  std::array<ModelDataRawYZ, TRAJECTORY_SIZE> right_far;
};
static_assert(sizeof(ModelDataRawLinesXY) == sizeof(ModelDataRawYZ)*TRAJECTORY_SIZE*4);

struct ModelDataRawLineProbVal {
  float val_deprecated;
  float val;
};
static_assert(sizeof(ModelDataRawLineProbVal) == sizeof(float)*2);

struct ModelDataRawLinesProb {
  ModelDataRawLineProbVal left_far;
  ModelDataRawLineProbVal left_near;
  ModelDataRawLineProbVal right_near;
  ModelDataRawLineProbVal right_far;
};
static_assert(sizeof(ModelDataRawLinesProb) == sizeof(ModelDataRawLineProbVal)*4);

struct ModelDataRawLaneLines {
  ModelDataRawLinesXY mean;
  ModelDataRawLinesXY std;
  ModelDataRawLinesProb prob;
};
static_assert(sizeof(ModelDataRawLaneLines) == (sizeof(ModelDataRawLinesXY)*2) + sizeof(ModelDataRawLinesProb));

struct ModelDataRawEdgessXY {
  std::array<ModelDataRawYZ, TRAJECTORY_SIZE> left;
  std::array<ModelDataRawYZ, TRAJECTORY_SIZE> right;
};
static_assert(sizeof(ModelDataRawEdgessXY) == sizeof(ModelDataRawYZ)*TRAJECTORY_SIZE*2);

struct ModelDataRawRoadEdges {
  ModelDataRawEdgessXY mean;
  ModelDataRawEdgessXY std;
};
static_assert(sizeof(ModelDataRawRoadEdges) == (sizeof(ModelDataRawEdgessXY)*2));

struct ModelDataRawPose {
  ModelDataRawXYZ velocity_mean;
  ModelDataRawXYZ rotation_mean;
  ModelDataRawXYZ velocity_std;
  ModelDataRawXYZ rotation_std;
};
static_assert(sizeof(ModelDataRawPose) == sizeof(ModelDataRawXYZ)*4);

struct ModelDataRaw {
  const ModelDataRawPlans *const plan;
  const ModelDataRawLaneLines *const lane_lines;
  const ModelDataRawRoadEdges *const road_edges;
  const float *const lead;
  const float *const lead_prob;
  const float *const desire_state;
  const float *const meta;
  const float *const desire_pred;
  const ModelDataRawPose *const pose;
};

struct ModelState {
  ModelFrame *frame;
  std::vector<float> output;
  std::unique_ptr<RunModel> m;
#ifdef DESIRE
  float prev_desire[DESIRE_LEN] = {};
  float pulse_desire[DESIRE_LEN] = {};
#endif
#ifdef TRAFFIC_CONVENTION
  float traffic_convention[TRAFFIC_CONVENTION_LEN] = {};
#endif
};

void model_init(ModelState* s, cl_device_id device_id, cl_context context);
ModelDataRaw model_eval_frame(ModelState* s, cl_mem yuv_cl, int width, int height,
                           const mat3 &transform, float *desire_in);
void model_free(ModelState* s);
void poly_fit(float *in_pts, float *in_stds, float *out);
void model_publish(PubMaster &pm, uint32_t vipc_frame_id, uint32_t frame_id, float frame_drop,
                   const ModelDataRaw &net_outputs, uint64_t timestamp_eof,
                   float model_execution_time, kj::ArrayPtr<const float> raw_pred);
void posenet_publish(PubMaster &pm, uint32_t vipc_frame_id, uint32_t vipc_dropped_frames,
                     const ModelDataRaw &net_outputs, uint64_t timestamp_eof);
