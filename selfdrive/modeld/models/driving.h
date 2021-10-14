#pragma once

// gate this here
#define TEMPORAL
#define DESIRE
#define TRAFFIC_CONVENTION

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

struct ModelDataRawXYZ {
  float x;
  float y;
  float z;
};

struct ModelDataRawRPY {
  float roll;
  float pitch;
  float yaw;
};

struct ModelDataRawPose {
  ModelDataRawXYZ velocity_mean;
  ModelDataRawRPY rotation_mean;
  ModelDataRawXYZ velocity_std;
  ModelDataRawRPY rotation_std;
};

struct ModelDataRaw {
  float *plan;
  float *lane_lines;
  float *lane_lines_prob;
  float *road_edges;
  float *lead;
  float *lead_prob;
  float *desire_state;
  float *meta;
  float *desire_pred;
  ModelDataRawPose *pose;
};

typedef struct ModelState {
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
} ModelState;

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
