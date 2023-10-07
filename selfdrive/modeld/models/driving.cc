#include "selfdrive/modeld/models/driving.h"

#include <cstring>

#define XYZ_ARRAY(v) {v.x, v.y, v.z}
#define XYZ_ARRAY_STD(v) {exp(v.x), exp(v.y), exp(v.z)}

template <typename Type, size_t size, typename Proj>
struct Array : public std::array<float, size> {
  Array(const std::array<Type, size> &arr, Proj proj) {
    for (int i = 0; i < size; ++i)
      (*this)[i] = std::invoke(proj, arr[i]);
  }
  operator kj::ArrayPtr<const float>() { return kj::ArrayPtr(this->data(), size); }
};

void fill_lead(cereal::ModelDataV2::LeadDataV3::Builder lead, const ModelOutputLeads &leads, int t_idx, float prob_t) {
  std::array<float, LEAD_TRAJ_LEN> lead_t = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
  const auto &best_prediction = leads.get_best_prediction(t_idx);
  lead.setProb(sigmoid(leads.prob[t_idx]));
  lead.setProbTime(prob_t);
  lead.setT(to_kj_array_ptr(lead_t));
  lead.setX(Array(best_prediction.mean, &ModelOutputLeadElement::x));
  lead.setY(Array(best_prediction.mean, &ModelOutputLeadElement::y));
  lead.setV(Array(best_prediction.mean, &ModelOutputLeadElement::velocity));
  lead.setA(Array(best_prediction.mean, &ModelOutputLeadElement::acceleration));
  lead.setXStd(Array(best_prediction.std, [](auto &v){ return exp(v.x); }));
  lead.setYStd(Array(best_prediction.std, [](auto &v){ return exp(v.y); }));
  lead.setVStd(Array(best_prediction.std, [](auto &v){ return exp(v.velocity); }));
  lead.setAStd(Array(best_prediction.std, [](auto &v){ return exp(v.acceleration); }));
}

void fill_meta(cereal::ModelDataV2::MetaData::Builder meta, const ModelOutputMeta &meta_data, PublishState &ps) {
  std::array<float, DESIRE_LEN> desire_state_softmax;
  softmax(meta_data.desire_state_prob.array.data(), desire_state_softmax.data(), DESIRE_LEN);

  std::array<float, DESIRE_PRED_LEN * DESIRE_LEN> desire_pred_softmax;
  for (int i=0; i<DESIRE_PRED_LEN; i++) {
    softmax(meta_data.desire_pred_prob[i].array.data(), desire_pred_softmax.data() + (i * DESIRE_LEN), DESIRE_LEN);
  }

  std::array<float, DISENGAGE_LEN> lat_long_t = {2, 4, 6, 8, 10};
  Array brake_3ms2_sigmoid(meta_data.disengage_prob, [](auto &v) { return sigmoid(v.brake_3ms2); });
  Array brake_5ms2_sigmoid(meta_data.disengage_prob, [](auto &v) { return sigmoid(v.brake_5ms2); });
  std::memmove(ps.prev_brake_5ms2_probs.data(), &ps.prev_brake_5ms2_probs[1], 4*sizeof(float));
  std::memmove(ps.prev_brake_3ms2_probs.data(), &ps.prev_brake_3ms2_probs[1], 2*sizeof(float));
  ps.prev_brake_5ms2_probs[4] = brake_5ms2_sigmoid[0];
  ps.prev_brake_3ms2_probs[2] = brake_3ms2_sigmoid[0];

  bool above_fcw_threshold = true;
  for (int i=0; i<ps.prev_brake_5ms2_probs.size(); i++) {
    float threshold = i < 2 ? FCW_THRESHOLD_5MS2_LOW : FCW_THRESHOLD_5MS2_HIGH;
    above_fcw_threshold = above_fcw_threshold && ps.prev_brake_5ms2_probs[i] > threshold;
  }
  for (int i=0; i<ps.prev_brake_3ms2_probs.size(); i++) {
    above_fcw_threshold = above_fcw_threshold && ps.prev_brake_3ms2_probs[i] > FCW_THRESHOLD_3MS2;
  }

  auto disengage = meta.initDisengagePredictions();
  disengage.setT(to_kj_array_ptr(lat_long_t));
  disengage.setGasDisengageProbs(Array(meta_data.disengage_prob, [](auto &v) { return sigmoid(v.gas_disengage);}));
  disengage.setBrakeDisengageProbs(Array(meta_data.disengage_prob, [](auto &v) { return sigmoid(v.brake_disengage); }));
  disengage.setSteerOverrideProbs(Array(meta_data.disengage_prob, [](auto &v) { return sigmoid(v.steer_override); }));
  disengage.setBrake3MetersPerSecondSquaredProbs(brake_3ms2_sigmoid);
  disengage.setBrake4MetersPerSecondSquaredProbs(Array(meta_data.disengage_prob, [](auto &v) { return sigmoid(v.brake_4ms2); }));
  disengage.setBrake5MetersPerSecondSquaredProbs(brake_5ms2_sigmoid);

  meta.setEngagedProb(sigmoid(meta_data.engaged_prob));
  meta.setDesirePrediction(to_kj_array_ptr(desire_pred_softmax));
  meta.setDesireState(to_kj_array_ptr(desire_state_softmax));
  meta.setHardBrakePredicted(above_fcw_threshold);
}

void fill_confidence(cereal::ModelDataV2::Builder &framed, PublishState &ps) {
  if (framed.getFrameId() % (2*MODEL_FREQ) == 0) {
    // update every 2s to match predictions interval
    auto dbps = framed.getMeta().getDisengagePredictions().getBrakeDisengageProbs();
    auto dgps = framed.getMeta().getDisengagePredictions().getGasDisengageProbs();
    auto dsps = framed.getMeta().getDisengagePredictions().getSteerOverrideProbs();

    float any_dp[DISENGAGE_LEN];
    float dp_ind[DISENGAGE_LEN];

    for (int i = 0; i < DISENGAGE_LEN; i++) {
      any_dp[i] = 1 - ((1-dbps[i])*(1-dgps[i])*(1-dsps[i])); // any disengage prob
    }

    dp_ind[0] = any_dp[0];
    for (int i = 0; i < DISENGAGE_LEN-1; i++) {
      dp_ind[i+1] = (any_dp[i+1] - any_dp[i]) / (1 - any_dp[i]); // independent disengage prob for each 2s slice
    }

    // rolling buf for 2, 4, 6, 8, 10s
    std::memmove(&ps.disengage_buffer[0], &ps.disengage_buffer[DISENGAGE_LEN], sizeof(float) * DISENGAGE_LEN * (DISENGAGE_LEN-1));
    std::memcpy(&ps.disengage_buffer[DISENGAGE_LEN * (DISENGAGE_LEN-1)], &dp_ind[0], sizeof(float) * DISENGAGE_LEN);
  }

  float score = 0;
  for (int i = 0; i < DISENGAGE_LEN; i++) {
    score += ps.disengage_buffer[i*DISENGAGE_LEN+DISENGAGE_LEN-1-i] / DISENGAGE_LEN;
  }

  if (score < RYG_GREEN) {
    framed.setConfidence(cereal::ModelDataV2::ConfidenceClass::GREEN);
  } else if (score < RYG_YELLOW) {
    framed.setConfidence(cereal::ModelDataV2::ConfidenceClass::YELLOW);
  } else {
    framed.setConfidence(cereal::ModelDataV2::ConfidenceClass::RED);
  }
}

template<size_t size>
void fill_xyzt(cereal::XYZTData::Builder xyzt, const std::array<float, size> &t,
               const std::array<float, size> &x, const std::array<float, size> &y, const std::array<float, size> &z) {
  xyzt.setT(to_kj_array_ptr(t));
  xyzt.setX(to_kj_array_ptr(x));
  xyzt.setY(to_kj_array_ptr(y));
  xyzt.setZ(to_kj_array_ptr(z));
}

void fill_plan(cereal::ModelDataV2::Builder &framed, const ModelOutputPlanPrediction &plan) {
  auto position = framed.initPosition();
  fill_xyzt(position, T_IDXS_FLOAT,
            Array(plan.mean, [](auto &v) { return v.position.x; }),
            Array(plan.mean, [](auto &v) { return v.position.y; }),
            Array(plan.mean, [](auto &v) { return v.position.z; }));
  position.setXStd(Array(plan.std, [](auto &v) { return exp(v.position.x); }));
  position.setYStd(Array(plan.std, [](auto &v) { return exp(v.position.y); }));
  position.setZStd(Array(plan.std, [](auto &v) { return exp(v.position.z); }));

  fill_xyzt(framed.initVelocity(), T_IDXS_FLOAT,
            Array(plan.mean, [](auto &v) { return v.velocity.x; }),
            Array(plan.mean, [](auto &v) { return v.velocity.y; }),
            Array(plan.mean, [](auto &v) { return v.velocity.z; }));
  fill_xyzt(framed.initAcceleration(), T_IDXS_FLOAT,
            Array(plan.mean, [](auto &v) { return v.acceleration.x; }),
            Array(plan.mean, [](auto &v) { return v.acceleration.y; }),
            Array(plan.mean, [](auto &v) { return v.acceleration.z; }));
  fill_xyzt(framed.initOrientation(), T_IDXS_FLOAT,
            Array(plan.mean, [](auto &v) { return v.rotation.x; }),
            Array(plan.mean, [](auto &v) { return v.rotation.y; }),
            Array(plan.mean, [](auto &v) { return v.rotation.z; }));
  fill_xyzt(framed.initOrientationRate(), T_IDXS_FLOAT,
            Array(plan.mean, [](auto &v) { return v.rotation_rate.x; }),
            Array(plan.mean, [](auto &v) { return v.rotation_rate.y; }),
            Array(plan.mean, [](auto &v) { return v.rotation_rate.z; }));
}

void fill_lane_lines(cereal::ModelDataV2::Builder &framed, const std::array<float, TRAJECTORY_SIZE> &plan_t,
                     const ModelOutputLaneLines &lanes) {
  auto lane_lines = framed.initLaneLines(4);
  fill_xyzt(lane_lines[0], plan_t, X_IDXS_FLOAT, Array(lanes.mean.left_far, &ModelOutputYZ::y), Array(lanes.mean.left_far, &ModelOutputYZ::z));
  fill_xyzt(lane_lines[1], plan_t, X_IDXS_FLOAT, Array(lanes.mean.left_near, &ModelOutputYZ::y), Array(lanes.mean.left_near, &ModelOutputYZ::z));
  fill_xyzt(lane_lines[2], plan_t, X_IDXS_FLOAT, Array(lanes.mean.right_near, &ModelOutputYZ::y), Array(lanes.mean.right_near, &ModelOutputYZ::z));
  fill_xyzt(lane_lines[3], plan_t, X_IDXS_FLOAT, Array(lanes.mean.right_far, &ModelOutputYZ::y), Array(lanes.mean.right_far, &ModelOutputYZ::z));

  framed.setLaneLineStds({
    exp(lanes.std.left_far[0].y),
    exp(lanes.std.left_near[0].y),
    exp(lanes.std.right_near[0].y),
    exp(lanes.std.right_far[0].y),
  });

  framed.setLaneLineProbs({
    sigmoid(lanes.prob.left_far.val),
    sigmoid(lanes.prob.left_near.val),
    sigmoid(lanes.prob.right_near.val),
    sigmoid(lanes.prob.right_far.val),
  });
}

void fill_road_edges(cereal::ModelDataV2::Builder &framed, const std::array<float, TRAJECTORY_SIZE> &plan_t,
                     const ModelOutputRoadEdges &edges) {
  auto road_edges = framed.initRoadEdges(2);
  fill_xyzt(road_edges[0], plan_t, X_IDXS_FLOAT, Array(edges.mean.left, &ModelOutputYZ::y), Array(edges.mean.left, &ModelOutputYZ::z));
  fill_xyzt(road_edges[1], plan_t, X_IDXS_FLOAT, Array(edges.mean.right, &ModelOutputYZ::y), Array(edges.mean.right, &ModelOutputYZ::z));

  framed.setRoadEdgeStds({
    exp(edges.std.left[0].y),
    exp(edges.std.right[0].y),
  });
}

void fill_model(cereal::ModelDataV2::Builder &framed, const ModelOutput &net_outputs, PublishState &ps) {
  const auto &best_plan = net_outputs.plans.get_best_prediction();
  std::array<float, TRAJECTORY_SIZE> plan_t;
  std::fill_n(plan_t.data(), plan_t.size(), NAN);
  plan_t[0] = 0.0;
  for (int xidx=1, tidx=0; xidx<TRAJECTORY_SIZE; xidx++) {
    // increment tidx until we find an element that's further away than the current xidx
    for (int next_tid = tidx + 1; next_tid < TRAJECTORY_SIZE && best_plan.mean[next_tid].position.x < X_IDXS[xidx]; next_tid++) {
      tidx++;
    }
    if (tidx == TRAJECTORY_SIZE - 1) {
      // if the Plan doesn't extend far enough, set plan_t to the max value (10s), then break
      plan_t[xidx] = T_IDXS[TRAJECTORY_SIZE - 1];
      break;
    }

    // interpolate to find `t` for the current xidx
    float current_x_val = best_plan.mean[tidx].position.x;
    float next_x_val = best_plan.mean[tidx+1].position.x;
    float p = (X_IDXS[xidx] - current_x_val) / (next_x_val - current_x_val);
    plan_t[xidx] = p * T_IDXS[tidx+1] + (1 - p) * T_IDXS[tidx];
  }

  fill_plan(framed, best_plan);
  fill_lane_lines(framed, plan_t, net_outputs.lane_lines);
  fill_road_edges(framed, plan_t, net_outputs.road_edges);

  // meta
  fill_meta(framed.initMeta(), net_outputs.meta, ps);

  // confidence
  fill_confidence(framed, ps);

  // leads
  auto leads = framed.initLeadsV3(LEAD_MHP_SELECTION);
  std::array<float, LEAD_MHP_SELECTION> t_offsets = {0.0, 2.0, 4.0};
  for (int i=0; i<LEAD_MHP_SELECTION; i++) {
    fill_lead(leads[i], net_outputs.leads, i, t_offsets[i]);
  }

  // temporal pose
  auto temporal_pose = framed.initTemporalPose();
  temporal_pose.setTrans(XYZ_ARRAY(net_outputs.temporal_pose.velocity_mean));
  temporal_pose.setRot(XYZ_ARRAY(net_outputs.temporal_pose.rotation_mean));
  temporal_pose.setTransStd(XYZ_ARRAY_STD(net_outputs.temporal_pose.velocity_std));
  temporal_pose.setRotStd(XYZ_ARRAY_STD(net_outputs.temporal_pose.rotation_std));
}

void fill_model_msg(MessageBuilder &msg, float *net_output_data, PublishState &ps, uint32_t vipc_frame_id, uint32_t vipc_frame_id_extra, uint32_t frame_id, float frame_drop,
                    uint64_t timestamp_eof, uint64_t timestamp_llk, float model_execution_time, const bool nav_enabled, const bool valid) {
  const uint32_t frame_age = (frame_id > vipc_frame_id) ? (frame_id - vipc_frame_id) : 0;
  auto framed = msg.initEvent(valid).initModelV2();
  framed.setFrameId(vipc_frame_id);
  framed.setFrameIdExtra(vipc_frame_id_extra);
  framed.setFrameAge(frame_age);
  framed.setFrameDropPerc(frame_drop * 100);
  framed.setTimestampEof(timestamp_eof);
  framed.setLocationMonoTime(timestamp_llk);
  framed.setModelExecutionTime(model_execution_time);
  framed.setNavEnabled(nav_enabled);
  if (send_raw_pred) {
    framed.setRawPredictions(kj::ArrayPtr<const float>(net_output_data, NET_OUTPUT_SIZE).asBytes());
  }
  fill_model(framed, *((ModelOutput*) net_output_data), ps);
}

void fill_pose_msg(MessageBuilder &msg, float *net_output_data, uint32_t vipc_frame_id, uint32_t vipc_dropped_frames, uint64_t timestamp_eof, const bool valid) {
    const ModelOutput &net_outputs = *((ModelOutput*) net_output_data);
    auto posenetd = msg.initEvent(valid && (vipc_dropped_frames < 1)).initCameraOdometry();
    posenetd.setTrans(XYZ_ARRAY(net_outputs.pose.velocity_mean));
    posenetd.setRot(XYZ_ARRAY(net_outputs.pose.rotation_mean));
    posenetd.setWideFromDeviceEuler(XYZ_ARRAY(net_outputs.wide_from_device_euler.mean));
    posenetd.setRoadTransformTrans(XYZ_ARRAY(net_outputs.road_transform.position_mean));
    posenetd.setTransStd(XYZ_ARRAY_STD(net_outputs.pose.velocity_std));
    posenetd.setRotStd(XYZ_ARRAY_STD(net_outputs.pose.rotation_std));
    posenetd.setWideFromDeviceEulerStd(XYZ_ARRAY_STD(net_outputs.wide_from_device_euler.std));
    posenetd.setRoadTransformTransStd(XYZ_ARRAY_STD(net_outputs.road_transform.position_std));

    posenetd.setTimestampEof(timestamp_eof);
    posenetd.setFrameId(vipc_frame_id);
}
