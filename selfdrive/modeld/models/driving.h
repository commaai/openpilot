#pragma once

// gate this here
#define TEMPORAL
#define DESIRE
#define TRAFFIC_CONVENTION


#include "common/mat.h"
#include "common/util.h"
#include "common/modeldata.h"

#include "commonmodel.h"
#include "runners/run.h"

#include <memory>
#include "messaging.hpp"

#define MODEL_NAME "supercombo_dlc"

constexpr int MODEL_WIDTH = 512;
constexpr int MODEL_HEIGHT = 256;
constexpr int MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 / 2;
constexpr int DESIRE_LEN = 8;
constexpr int TRAFFIC_CONVENTION_LEN = 2;

constexpr int PLAN_MHP_N = 5;
constexpr int PLAN_MHP_COLUMNS = 30;
constexpr int PLAN_MHP_VALS = 30*33;
constexpr int PLAN_MHP_SELECTION = 1;
constexpr int PLAN_MHP_GROUP_SIZE =  (2*PLAN_MHP_VALS + PLAN_MHP_SELECTION);

constexpr int LEAD_MHP_N = 5;
constexpr int LEAD_MHP_VALS = 4;
constexpr int LEAD_MHP_SELECTION = 3;
constexpr int LEAD_MHP_GROUP_SIZE = (2*LEAD_MHP_VALS + LEAD_MHP_SELECTION);

constexpr int POSE_SIZE = 12;

constexpr int MODEL_FREQ = 20;

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
    float *pose;
  };


typedef struct ModelState {
  ModelFrame frame;
  std::unique_ptr<float[]> output;
  std::unique_ptr<float[]> input_frames;
  std::unique_ptr<RunModel> m;
#ifdef DESIRE
  std::unique_ptr<float[]> prev_desire;
  std::unique_ptr<float[]> pulse_desire;
#endif
#ifdef TRAFFIC_CONVENTION
  std::unique_ptr<float[]> traffic_convention;
#endif
} ModelState;

void model_init(ModelState* s, cl_device_id device_id,
                cl_context context, int temporal);
ModelDataRaw model_eval_frame(ModelState* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform, void* sock, float *desire_in);
void model_free(ModelState* s);
void poly_fit(float *in_pts, float *in_stds, float *out);

void model_publish(PubMaster &pm, uint32_t vipc_frame_id, uint32_t frame_id,
                   uint32_t vipc_dropped_frames, float frame_drop, const ModelDataRaw &data, const float* raw_pred,
                   uint64_t timestamp_eof, float model_execution_time);
void model_publish_v2(PubMaster &pm, uint32_t vipc_frame_id, uint32_t frame_id,
                   uint32_t vipc_dropped_frames, float frame_drop, const ModelDataRaw &data, const float* raw_pred,
                   uint64_t timestamp_eof, float model_execution_time);
void posenet_publish(PubMaster &pm, uint32_t vipc_frame_id, uint32_t frame_id,
                     uint32_t vipc_dropped_frames, float frame_drop, const ModelDataRaw &data,
                     uint64_t timestamp_eof);
