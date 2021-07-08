#include "selfdrive/modeld/models/driving.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cassert>
#include <cstring>

#include <eigen3/Eigen/Dense>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/timing.h"

constexpr int DESIRE_PRED_SIZE = 32;
constexpr int OTHER_META_SIZE = 32;
constexpr int NUM_META_INTERVALS = 5;
constexpr int META_STRIDE = 6;

constexpr int PLAN_MHP_N = 5;
constexpr int PLAN_MHP_COLUMNS = 15;
constexpr int PLAN_MHP_VALS = 15*33;
constexpr int PLAN_MHP_SELECTION = 1;
constexpr int PLAN_MHP_GROUP_SIZE =  (2*PLAN_MHP_VALS + PLAN_MHP_SELECTION);

constexpr int LEAD_MHP_N = 5;
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

constexpr float FCW_THRESHOLD_5MS2_HIGH = 0.15;
constexpr float FCW_THRESHOLD_5MS2_LOW = 0.05;
constexpr float FCW_THRESHOLD_3MS2 = 0.7;

float prev_brake_5ms2_probs[5] = {0,0,0,0,0};
float prev_brake_3ms2_probs[3] = {0,0,0};

// #define DUMP_YUV

void model_init(ModelState* s, cl_device_id device_id, cl_context context) {
  s->frame = new ModelFrame(device_id, context);

  constexpr int output_size = OUTPUT_SIZE + TEMPORAL_SIZE;
  s->output.resize(output_size);

#if (defined(QCOM) || defined(QCOM2)) && defined(USE_THNEED)
  s->m = std::make_unique<ThneedModel>("../../models/supercombo.thneed", &s->output[0], output_size, USE_GPU_RUNTIME);
#else
  s->m = std::make_unique<DefaultRunModel>("../../models/supercombo.dlc", &s->output[0], output_size, USE_GPU_RUNTIME);
#endif

#ifdef TEMPORAL
  s->m->addRecurrent(&s->output[OUTPUT_SIZE], TEMPORAL_SIZE);
#endif

#ifdef DESIRE
  s->m->addDesire(s->pulse_desire, DESIRE_LEN);
#endif

#ifdef TRAFFIC_CONVENTION
  const int idx = Params().getBool("IsRHD") ? 1 : 0;
  s->traffic_convention[idx] = 1.0;
  s->m->addTrafficConvention(s->traffic_convention, TRAFFIC_CONVENTION_LEN);
#endif
}

ModelDataRaw model_eval_frame(ModelState* s, cl_mem yuv_cl, int width, int height,
                           const mat3 &transform, float *desire_in) {
#ifdef DESIRE
  if (desire_in != NULL) {
    for (int i = 1; i < DESIRE_LEN; i++) {
      // Model decides when action is completed
      // so desire input is just a pulse triggered on rising edge
      if (desire_in[i] - s->prev_desire[i] > .99) {
        s->pulse_desire[i] = desire_in[i];
      } else {
        s->pulse_desire[i] = 0.0;
      }
      s->prev_desire[i] = desire_in[i];
    }
  }
#endif

  //for (int i = 0; i < OUTPUT_SIZE + TEMPORAL_SIZE; i++) { printf("%f ", s->output[i]); } printf("\n");

  auto net_input_buf = s->frame->prepare(yuv_cl, width, height, transform);
  s->m->execute(net_input_buf, s->frame->buf_size);

  // net outputs
  ModelDataRaw net_outputs;
  net_outputs.plan = &s->output[PLAN_IDX];
  net_outputs.lane_lines = &s->output[LL_IDX];
  net_outputs.lane_lines_prob = &s->output[LL_PROB_IDX];
  net_outputs.road_edges = &s->output[RE_IDX];
  net_outputs.lead = &s->output[LEAD_IDX];
  net_outputs.lead_prob = &s->output[LEAD_PROB_IDX];
  net_outputs.meta = &s->output[DESIRE_STATE_IDX];
  net_outputs.pose = &s->output[POSE_IDX];
  return net_outputs;
}

void model_free(ModelState* s) {
  delete s->frame;
}

template <size_t size, float (F)(float)=nullptr> 
class Arr {
public:
  Arr(const float *src, int stride = 1) {
    for (int i = 0, j = 0; i < size; ++i, j += stride) {
      a_[i] = F != nullptr ? F(src[j]) : src[j];
    }
  }
  inline operator kj::ArrayPtr<const float>() { return {a_.data(), a_.size()}; }
  inline float operator[](int id) { return a_[id]; }
  std::array<float, size> a_;
};

static const float *get_best_data(const float *data, int size, int group_size, int offset) {
  int max_idx = 0;
  for (int i = 1; i < size; i++) {
    if (data[(i + 1) * group_size + offset] >
        data[(max_idx + 1) * group_size + offset]) {
      max_idx = i;
    }
  }
  return &data[max_idx * group_size];
}

static const float *get_plan_data(float *plan) {
  return get_best_data(plan, PLAN_MHP_N, PLAN_MHP_GROUP_SIZE, -1);
}

static const float *get_lead_data(const float *lead, int t_offset) {
  return get_best_data(lead, LEAD_MHP_N, LEAD_MHP_GROUP_SIZE, t_offset - LEAD_MHP_SELECTION);
}

void fill_lead_v3(cereal::ModelDataV2::LeadDataV3::Builder lead, const float *lead_data, const float *prob, int t_offset, float prob_t) {
  float t[LEAD_TRAJ_LEN] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
  const float *data = get_lead_data(lead_data, t_offset);
  lead.setProb(sigmoid(prob[t_offset]));
  lead.setProbTime(prob_t);
  lead.setT(t);
  lead.setX(Arr<LEAD_TRAJ_LEN>(&data[0], LEAD_PRED_DIM));
  lead.setY(Arr<LEAD_TRAJ_LEN>(&data[1], LEAD_PRED_DIM));
  lead.setV(Arr<LEAD_TRAJ_LEN>(&data[2], LEAD_PRED_DIM));
  lead.setA(Arr<LEAD_TRAJ_LEN>(&data[3], LEAD_PRED_DIM));
  lead.setXStd(Arr<LEAD_TRAJ_LEN, exp>(&data[LEAD_MHP_VALS+0], LEAD_PRED_DIM));
  lead.setYStd(Arr<LEAD_TRAJ_LEN, exp>(&data[LEAD_MHP_VALS+1], LEAD_PRED_DIM));
  lead.setVStd(Arr<LEAD_TRAJ_LEN, exp>(&data[LEAD_MHP_VALS+2], LEAD_PRED_DIM));
  lead.setAStd(Arr<LEAD_TRAJ_LEN, exp>(&data[LEAD_MHP_VALS+3], LEAD_PRED_DIM));
}

void fill_meta(cereal::ModelDataV2::MetaData::Builder meta, const float *meta_data) {
  float desire_state_softmax[DESIRE_LEN];
  float desire_pred_softmax[4*DESIRE_LEN];
  softmax(&meta_data[0], desire_state_softmax, DESIRE_LEN);
  for (int i=0; i<4; i++) {
    softmax(&meta_data[DESIRE_LEN + OTHER_META_SIZE + i*DESIRE_LEN],
            &desire_pred_softmax[i*DESIRE_LEN], DESIRE_LEN);
  }

  std::memmove(prev_brake_5ms2_probs, &prev_brake_5ms2_probs[1], 4*sizeof(float));
  std::memmove(prev_brake_3ms2_probs, &prev_brake_3ms2_probs[1], 2*sizeof(float));
   auto brake_3ms2_sigmoid = Arr<NUM_META_INTERVALS, sigmoid>(&meta_data[DESIRE_LEN+4], META_STRIDE);
  auto brake_5ms2_sigmoid = Arr<NUM_META_INTERVALS, sigmoid>(&meta_data[DESIRE_LEN+6], META_STRIDE);
  prev_brake_5ms2_probs[4] = brake_5ms2_sigmoid[0];
  prev_brake_3ms2_probs[2] = brake_3ms2_sigmoid[0];

  bool above_fcw_threshold = true;
  for (int i=0; i<5; i++) {
    float threshold = i < 2 ? FCW_THRESHOLD_5MS2_LOW : FCW_THRESHOLD_5MS2_HIGH;
    above_fcw_threshold = above_fcw_threshold && prev_brake_5ms2_probs[i] > threshold;
  }
  for (int i=0; i<3; i++) {
    above_fcw_threshold = above_fcw_threshold && prev_brake_3ms2_probs[i] > FCW_THRESHOLD_3MS2;
  }

  auto disengage = meta.initDisengagePredictions();
  disengage.setT({2,4,6,8,10});
  disengage.setGasDisengageProbs(Arr<NUM_META_INTERVALS, sigmoid>(&meta_data[DESIRE_LEN+1], META_STRIDE));
  disengage.setBrakeDisengageProbs(Arr<NUM_META_INTERVALS, sigmoid>(&meta_data[DESIRE_LEN+2], META_STRIDE));
  disengage.setSteerOverrideProbs(Arr<NUM_META_INTERVALS, sigmoid>(&meta_data[DESIRE_LEN+3], META_STRIDE));
  disengage.setBrake3MetersPerSecondSquaredProbs(brake_3ms2_sigmoid);
  disengage.setBrake4MetersPerSecondSquaredProbs(Arr<NUM_META_INTERVALS, sigmoid>(&meta_data[DESIRE_LEN+5], META_STRIDE));
  disengage.setBrake5MetersPerSecondSquaredProbs(brake_5ms2_sigmoid);

  meta.setEngagedProb(sigmoid(meta_data[DESIRE_LEN]));
  meta.setDesirePrediction(desire_pred_softmax);
  meta.setDesireState(desire_state_softmax);
  meta.setHardBrakePredicted(above_fcw_threshold);
}

void fill_xyzt(cereal::ModelDataV2::XYZTData::Builder xyzt, const float * data,
               int columns, int column_offset, float * plan_t_arr, bool fill_std) {
  typedef Arr<TRAJECTORY_SIZE> arr;
  const float *d = &data[column_offset];
  xyzt.setX(column_offset >= 0 ? arr(&d[0], columns) : arr((float *)X_IDXS));
  xyzt.setY(arr(&d[1], columns));
  xyzt.setZ(arr(&d[2], columns));
  xyzt.setT(column_offset >= 0 ? arr((float *)T_IDXS) : arr(plan_t_arr));
  if (fill_std) {
    float nan_stds[TRAJECTORY_SIZE] = {};
    std::fill_n(nan_stds, std::size(nan_stds), NAN);
    xyzt.setXStd(column_offset >= 0 ? arr(&d[columns*TRAJECTORY_SIZE], columns) : arr(nan_stds));
    xyzt.setYStd(arr(&d[columns*TRAJECTORY_SIZE+1], columns));
    xyzt.setZStd(arr(&d[columns*TRAJECTORY_SIZE+2], columns));
  }
}

void fill_model(cereal::ModelDataV2::Builder &framed, const ModelDataRaw &net_outputs) {
  // plan
  const float *best_plan = get_plan_data(net_outputs.plan);
  float plan_t_arr[TRAJECTORY_SIZE];
  std::fill_n(plan_t_arr, TRAJECTORY_SIZE, NAN);
  plan_t_arr[0] = 0.0;
  for (int xidx=1, tidx=0; xidx<TRAJECTORY_SIZE; xidx++) {
    // increment tidx until we find an element that's further away than the current xidx
    while (tidx < TRAJECTORY_SIZE-1 && best_plan[(tidx+1)*PLAN_MHP_COLUMNS] < X_IDXS[xidx]) {
      tidx++;
    }
    float current_x_val = best_plan[tidx*PLAN_MHP_COLUMNS];
    float next_x_val = best_plan[(tidx+1)*PLAN_MHP_COLUMNS];
    if (next_x_val < X_IDXS[xidx]) {
      // if the plan doesn't extend far enough, set plan_t to the max value (10s), then break
      plan_t_arr[xidx] = T_IDXS[TRAJECTORY_SIZE-1];
      break;
    } else {
      // otherwise, interpolate to find `t` for the current xidx
      float p = (X_IDXS[xidx] - current_x_val) / (next_x_val - current_x_val);
      plan_t_arr[xidx] = p * T_IDXS[tidx+1] + (1 - p) * T_IDXS[tidx];
    }
  }

  fill_xyzt(framed.initPosition(), best_plan, PLAN_MHP_COLUMNS, 0, plan_t_arr, true);
  fill_xyzt(framed.initVelocity(), best_plan, PLAN_MHP_COLUMNS, 3, plan_t_arr, false);
  fill_xyzt(framed.initOrientation(), best_plan, PLAN_MHP_COLUMNS, 9, plan_t_arr, false);
  fill_xyzt(framed.initOrientationRate(), best_plan, PLAN_MHP_COLUMNS, 12, plan_t_arr, false);

  // lane lines
  auto lane_lines = framed.initLaneLines(4);
  for (int i = 0; i < 4; i++) {
    fill_xyzt(lane_lines[i], &net_outputs.lane_lines[i*TRAJECTORY_SIZE*2], 2, -1, plan_t_arr, false);
  }
  framed.setLaneLineProbs(Arr<4, sigmoid>(&net_outputs.lane_lines_prob[1], 2));
  framed.setLaneLineStds(Arr<4, exp>(&net_outputs.lane_lines[2*TRAJECTORY_SIZE*4], 2*TRAJECTORY_SIZE));

  // road edges
  auto road_edges = framed.initRoadEdges(2);
  for (int i = 0; i < 2; i++) {
    fill_xyzt(road_edges[i], &net_outputs.road_edges[i*TRAJECTORY_SIZE*2], 2, -1, plan_t_arr, false);
  }
  framed.setRoadEdgeStds(Arr<2, exp>(&net_outputs.road_edges[4*TRAJECTORY_SIZE], 2*TRAJECTORY_SIZE));

  // meta
  fill_meta(framed.initMeta(), net_outputs.meta);

  // leads
  auto leads = framed.initLeadsV3(LEAD_MHP_SELECTION);
  float t_offsets[LEAD_MHP_SELECTION] = {0.0, 2.0, 4.0};
  for (int t_offset=0; t_offset<LEAD_MHP_SELECTION; t_offset++) {
    fill_lead_v3(leads[t_offset], net_outputs.lead, net_outputs.lead_prob, t_offset, t_offsets[t_offset]);
  }
}

void model_publish(PubMaster &pm, uint32_t vipc_frame_id, uint32_t frame_id, float frame_drop,
                   const ModelDataRaw &net_outputs, uint64_t timestamp_eof,
                   float model_execution_time, kj::ArrayPtr<const float> raw_pred) {
  const uint32_t frame_age = (frame_id > vipc_frame_id) ? (frame_id - vipc_frame_id) : 0;
  MessageBuilder msg;
  auto framed = msg.initEvent().initModelV2();
  framed.setFrameId(vipc_frame_id);
  framed.setFrameAge(frame_age);
  framed.setFrameDropPerc(frame_drop * 100);
  framed.setTimestampEof(timestamp_eof);
  framed.setModelExecutionTime(model_execution_time);
  if (send_raw_pred) {
    framed.setRawPredictions(raw_pred.asBytes());
  }
  fill_model(framed, net_outputs);
  pm.send("modelV2", msg);
}

void posenet_publish(PubMaster &pm, uint32_t vipc_frame_id, uint32_t vipc_dropped_frames,
                     const ModelDataRaw &net_outputs, uint64_t timestamp_eof) {
  MessageBuilder msg;
  auto posenetd = msg.initEvent(vipc_dropped_frames < 1).initCameraOdometry();
  posenetd.setTrans({&net_outputs.pose[0], 3});
  posenetd.setRot({&net_outputs.pose[3], 3});
  posenetd.setTransStd(Arr<3, exp>(&net_outputs.pose[6]));
  posenetd.setRotStd(Arr<3, exp>(&net_outputs.pose[9]));
  posenetd.setTimestampEof(timestamp_eof);
  posenetd.setFrameId(vipc_frame_id);
  pm.send("cameraOdometry", msg);
}
