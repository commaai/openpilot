#include "selfdrive/modeld/models/driving.h"

#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <cstring>

#include <eigen3/Eigen/Dense>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/timing.h"

constexpr int DESIRE_PRED_SIZE = 32;
constexpr int OTHER_META_SIZE = 48;
constexpr int NUM_META_INTERVALS = 5;
constexpr int META_STRIDE = 7;

constexpr int PLAN_MHP_N = 5;
constexpr int PLAN_MHP_COLUMNS = 15;
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

#ifdef USE_THNEED
  s->m = std::make_unique<ThneedModel>("../../models/supercombo.thneed", &s->output[0], output_size, USE_GPU_RUNTIME);
#elif USE_ONNX_MODEL
  s->m = std::make_unique<ONNXModel>("../../models/supercombo.onnx", &s->output[0], output_size, USE_GPU_RUNTIME);
#else
  s->m = std::make_unique<SNPEModel>("../../models/supercombo.dlc", &s->output[0], output_size, USE_GPU_RUNTIME);
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

  // if getInputBuf is not NULL, net_input_buf will be
  auto net_input_buf = s->frame->prepare(yuv_cl, width, height, transform, static_cast<cl_mem*>(s->m->getInputBuf()));
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

static const float *get_best_data(const float *data, int size, int group_size, int weight_idx) {
  int max_idx = 0;
  for (int i = 1; i < size; i++) {
    if (data[i * group_size + weight_idx] >
        data[max_idx * group_size + weight_idx]) {
      max_idx = i;
    }
  }
  return &data[max_idx * group_size];
}

static const float *get_plan_data(float *plan) {
  return get_best_data(plan, PLAN_MHP_N, PLAN_MHP_GROUP_SIZE, PLAN_MHP_GROUP_SIZE - 1);
}

static const float *get_lead_data(const float *lead, int t_offset) {
  return get_best_data(lead, LEAD_MHP_N, LEAD_MHP_GROUP_SIZE, LEAD_MHP_GROUP_SIZE - LEAD_MHP_SELECTION + t_offset);
}


void fill_sigmoid(const float *input, float *output, int len, int stride) {
  for (int i=0; i<len; i++) {
    output[i] = sigmoid(input[i*stride]);
  }
}

void fill_lead_v3(cereal::ModelDataV2::LeadDataV3::Builder lead, const float *lead_data, const float *prob, int t_offset, float prob_t) {
  float t[LEAD_TRAJ_LEN] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
  const float *data = get_lead_data(lead_data, t_offset);
  lead.setProb(sigmoid(prob[t_offset]));
  lead.setProbTime(prob_t);
  float x_arr[LEAD_TRAJ_LEN];
  float y_arr[LEAD_TRAJ_LEN];
  float v_arr[LEAD_TRAJ_LEN];
  float a_arr[LEAD_TRAJ_LEN];
  float x_stds_arr[LEAD_TRAJ_LEN];
  float y_stds_arr[LEAD_TRAJ_LEN];
  float v_stds_arr[LEAD_TRAJ_LEN];
  float a_stds_arr[LEAD_TRAJ_LEN];
  for (int i=0; i<LEAD_TRAJ_LEN; i++) {
    x_arr[i] = data[i*LEAD_PRED_DIM+0];
    y_arr[i] = data[i*LEAD_PRED_DIM+1];
    v_arr[i] = data[i*LEAD_PRED_DIM+2];
    a_arr[i] = data[i*LEAD_PRED_DIM+3];
    x_stds_arr[i] = exp(data[LEAD_MHP_VALS + i*LEAD_PRED_DIM+0]);
    y_stds_arr[i] = exp(data[LEAD_MHP_VALS + i*LEAD_PRED_DIM+1]);
    v_stds_arr[i] = exp(data[LEAD_MHP_VALS + i*LEAD_PRED_DIM+2]);
    a_stds_arr[i] = exp(data[LEAD_MHP_VALS + i*LEAD_PRED_DIM+3]);
  }
  lead.setT(t);
  lead.setX(x_arr);
  lead.setY(y_arr);
  lead.setV(v_arr);
  lead.setA(a_arr);
  lead.setXStd(x_stds_arr);
  lead.setYStd(y_stds_arr);
  lead.setVStd(v_stds_arr);
  lead.setAStd(a_stds_arr);
}

void fill_meta(cereal::ModelDataV2::MetaData::Builder meta, const float *meta_data) {
  float desire_state_softmax[DESIRE_LEN];
  float desire_pred_softmax[4*DESIRE_LEN];
  softmax(&meta_data[0], desire_state_softmax, DESIRE_LEN);
  for (int i=0; i<4; i++) {
    softmax(&meta_data[DESIRE_LEN + OTHER_META_SIZE + i*DESIRE_LEN],
            &desire_pred_softmax[i*DESIRE_LEN], DESIRE_LEN);
  }

  float gas_disengage_sigmoid[NUM_META_INTERVALS];
  float brake_disengage_sigmoid[NUM_META_INTERVALS];
  float steer_override_sigmoid[NUM_META_INTERVALS];
  float brake_3ms2_sigmoid[NUM_META_INTERVALS];
  float brake_4ms2_sigmoid[NUM_META_INTERVALS];
  float brake_5ms2_sigmoid[NUM_META_INTERVALS];

  fill_sigmoid(&meta_data[DESIRE_LEN+1], gas_disengage_sigmoid, NUM_META_INTERVALS, META_STRIDE);
  fill_sigmoid(&meta_data[DESIRE_LEN+2], brake_disengage_sigmoid, NUM_META_INTERVALS, META_STRIDE);
  fill_sigmoid(&meta_data[DESIRE_LEN+3], steer_override_sigmoid, NUM_META_INTERVALS, META_STRIDE);
  fill_sigmoid(&meta_data[DESIRE_LEN+4], brake_3ms2_sigmoid, NUM_META_INTERVALS, META_STRIDE);
  fill_sigmoid(&meta_data[DESIRE_LEN+5], brake_4ms2_sigmoid, NUM_META_INTERVALS, META_STRIDE);
  fill_sigmoid(&meta_data[DESIRE_LEN+6], brake_5ms2_sigmoid, NUM_META_INTERVALS, META_STRIDE);
  //fill_sigmoid(&meta_data[DESIRE_LEN+7], GAS PRESSED, NUM_META_INTERVALS, META_STRIDE);

  std::memmove(prev_brake_5ms2_probs, &prev_brake_5ms2_probs[1], 4*sizeof(float));
  std::memmove(prev_brake_3ms2_probs, &prev_brake_3ms2_probs[1], 2*sizeof(float));
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
  disengage.setGasDisengageProbs(gas_disengage_sigmoid);
  disengage.setBrakeDisengageProbs(brake_disengage_sigmoid);
  disengage.setSteerOverrideProbs(steer_override_sigmoid);
  disengage.setBrake3MetersPerSecondSquaredProbs(brake_3ms2_sigmoid);
  disengage.setBrake4MetersPerSecondSquaredProbs(brake_4ms2_sigmoid);
  disengage.setBrake5MetersPerSecondSquaredProbs(brake_5ms2_sigmoid);

  meta.setEngagedProb(sigmoid(meta_data[DESIRE_LEN]));
  meta.setDesirePrediction(desire_pred_softmax);
  meta.setDesireState(desire_state_softmax);
  meta.setHardBrakePredicted(above_fcw_threshold);
}

void fill_xyzt(cereal::ModelDataV2::XYZTData::Builder xyzt, const float * data,
               int columns, int column_offset, float * plan_t_arr, bool fill_std) {
  float x_arr[TRAJECTORY_SIZE] = {};
  float y_arr[TRAJECTORY_SIZE] = {};
  float z_arr[TRAJECTORY_SIZE] = {};
  float x_std_arr[TRAJECTORY_SIZE];
  float y_std_arr[TRAJECTORY_SIZE];
  float z_std_arr[TRAJECTORY_SIZE];
  float t_arr[TRAJECTORY_SIZE];
  for (int i=0; i<TRAJECTORY_SIZE; i++) {
    // column_offset == -1 means this data is X indexed not T indexed
    if (column_offset >= 0) {
      t_arr[i] = T_IDXS[i];
      x_arr[i] = data[i*columns + 0 + column_offset];
      x_std_arr[i] = data[columns*(TRAJECTORY_SIZE + i) + 0 + column_offset];
    } else {
      t_arr[i] = plan_t_arr[i];
      x_arr[i] = X_IDXS[i];
      x_std_arr[i] = NAN;
    }
    y_arr[i] = data[i*columns + 1 + column_offset];
    y_std_arr[i] = data[columns*(TRAJECTORY_SIZE + i) + 1 + column_offset];
    z_arr[i] = data[i*columns + 2 + column_offset];
    z_std_arr[i] = data[columns*(TRAJECTORY_SIZE + i) + 2 + column_offset];
  }
  xyzt.setX(x_arr);
  xyzt.setY(y_arr);
  xyzt.setZ(z_arr);
  xyzt.setT(t_arr);
  if (fill_std) {
    xyzt.setXStd(x_std_arr);
    xyzt.setYStd(y_std_arr);
    xyzt.setZStd(z_std_arr);
  }
}

void fill_model(cereal::ModelDataV2::Builder &framed, const ModelDataRaw &net_outputs) {
  const float *best_plan = get_plan_data(net_outputs.plan);
  float plan_t_arr[TRAJECTORY_SIZE];
  std::fill_n(plan_t_arr, TRAJECTORY_SIZE, NAN);
  plan_t_arr[0] = 0.0;
  for (int xidx=1, tidx=0; xidx<TRAJECTORY_SIZE; xidx++) {
    // increment tidx until we find an element that's further away than the current xidx
    for (int next_tid = tidx + 1; next_tid < TRAJECTORY_SIZE && best_plan[next_tid*PLAN_MHP_COLUMNS] < X_IDXS[xidx]; next_tid++) {
      tidx++;
    }
    if (tidx == TRAJECTORY_SIZE - 1) {
      // if the plan doesn't extend far enough, set plan_t to the max value (10s), then break
      plan_t_arr[xidx] = T_IDXS[TRAJECTORY_SIZE - 1];
      break;
    }

    // interpolate to find `t` for the current xidx
    float current_x_val = best_plan[tidx * PLAN_MHP_COLUMNS];
    float next_x_val = best_plan[(tidx+1) * PLAN_MHP_COLUMNS];
    float p = (X_IDXS[xidx] - current_x_val) / (next_x_val - current_x_val);
    plan_t_arr[xidx] = p * T_IDXS[tidx+1] + (1 - p) * T_IDXS[tidx];
  }

  fill_xyzt(framed.initPosition(), best_plan, PLAN_MHP_COLUMNS, 0, plan_t_arr, true);
  fill_xyzt(framed.initVelocity(), best_plan, PLAN_MHP_COLUMNS, 3, plan_t_arr, false);
  fill_xyzt(framed.initOrientation(), best_plan, PLAN_MHP_COLUMNS, 9, plan_t_arr, false);
  fill_xyzt(framed.initOrientationRate(), best_plan, PLAN_MHP_COLUMNS, 12, plan_t_arr, false);

  // lane lines
  auto lane_lines = framed.initLaneLines(4);
  float lane_line_probs_arr[4];
  float lane_line_stds_arr[4];
  for (int i = 0; i < 4; i++) {
    fill_xyzt(lane_lines[i], &net_outputs.lane_lines[i*TRAJECTORY_SIZE*2], 2, -1, plan_t_arr, false);
    lane_line_probs_arr[i] = sigmoid(net_outputs.lane_lines_prob[i*2+1]);
    lane_line_stds_arr[i] = exp(net_outputs.lane_lines[2*TRAJECTORY_SIZE*(4 + i)]);
  }
  framed.setLaneLineProbs(lane_line_probs_arr);
  framed.setLaneLineStds(lane_line_stds_arr);

  // road edges
  auto road_edges = framed.initRoadEdges(2);
  float road_edge_stds_arr[2];
  for (int i = 0; i < 2; i++) {
    fill_xyzt(road_edges[i], &net_outputs.road_edges[i*TRAJECTORY_SIZE*2], 2, -1, plan_t_arr, false);
    road_edge_stds_arr[i] = exp(net_outputs.road_edges[2*TRAJECTORY_SIZE*(2 + i)]);
  }
  framed.setRoadEdgeStds(road_edge_stds_arr);

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
  float trans_arr[3];
  float trans_std_arr[3];
  float rot_arr[3];
  float rot_std_arr[3];

  for (int i =0; i < 3; i++) {
    trans_arr[i] = net_outputs.pose[i];
    trans_std_arr[i] = exp(net_outputs.pose[6 + i]);

    rot_arr[i] = net_outputs.pose[3 + i];
    rot_std_arr[i] = exp(net_outputs.pose[9 + i]);
  }

  MessageBuilder msg;
  auto posenetd = msg.initEvent(vipc_dropped_frames < 1).initCameraOdometry();
  posenetd.setTrans(trans_arr);
  posenetd.setRot(rot_arr);
  posenetd.setTransStd(trans_std_arr);
  posenetd.setRotStd(rot_std_arr);

  posenetd.setTimestampEof(timestamp_eof);
  posenetd.setFrameId(vipc_frame_id);

  pm.send("cameraOdometry", msg);
}
