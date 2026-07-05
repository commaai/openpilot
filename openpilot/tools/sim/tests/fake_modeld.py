#!/usr/bin/env python3
"""Deterministic modeld stand-in for the MetaDrive CI bridge test.

The production driving model currently cannot run at 20Hz on free GitHub
Actions CPU runners. This publisher keeps the simulator CI test focused on the
bridge, controls loop, and logging artifacts while preserving the same message
interfaces consumed by the rest of openpilot.
"""

import time

import numpy as np

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.mock.generators import generate_livePose
from openpilot.common.realtime import DT_MDL, Ratekeeper
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.fill_model_msg import fill_driving_model_data, fill_xyzt, fill_xyvat


def _fill_lane_line(line, y_offset: float) -> None:
  x = np.array(ModelConstants.X_IDXS, dtype=np.float32)
  y = np.full_like(x, y_offset)
  z = np.zeros_like(x)
  fill_xyzt(line, [], x, y, z)


def _fill_lead(lead, idx: int) -> None:
  zeros = np.zeros(len(ModelConstants.LEAD_T_IDXS), dtype=np.float32)
  fill_xyvat(lead, ModelConstants.LEAD_T_IDXS, zeros, zeros, zeros, zeros,
             zeros + 10.0, zeros + 10.0, zeros + 10.0, zeros + 10.0)
  lead.prob = 0.0
  lead.probTime = ModelConstants.LEAD_T_OFFSETS[idx]


def _straight_plan(v_ego: float, target_accel: float) -> dict[str, np.ndarray]:
  t = np.array(ModelConstants.T_IDXS, dtype=np.float32)
  x = np.maximum(0.0, v_ego * t + 0.5 * target_accel * t ** 2)
  v = np.maximum(0.0, v_ego + target_accel * t)
  a = np.full_like(t, target_accel)

  plan = np.zeros((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH), dtype=np.float32)
  plan[0, :, Plan.POSITION] = np.column_stack([x, np.zeros_like(t), np.zeros_like(t)])
  plan[0, :, Plan.VELOCITY] = np.column_stack([v, np.zeros_like(t), np.zeros_like(t)])
  plan[0, :, Plan.ACCELERATION] = np.column_stack([a, np.zeros_like(t), np.zeros_like(t)])

  return {
    'plan': plan,
    'plan_stds': np.ones_like(plan) * 0.1,
    'pose': np.zeros((1, ModelConstants.POSE_WIDTH), dtype=np.float32),
    'pose_stds': np.ones((1, ModelConstants.POSE_WIDTH), dtype=np.float32) * 0.01,
    'wide_from_device_euler': np.zeros((1, ModelConstants.WIDE_FROM_DEVICE_WIDTH), dtype=np.float32),
    'wide_from_device_euler_stds': np.ones((1, ModelConstants.WIDE_FROM_DEVICE_WIDTH), dtype=np.float32) * 0.01,
    'road_transform': np.zeros((1, 3), dtype=np.float32),
    'road_transform_stds': np.ones((1, 3), dtype=np.float32) * 0.01,
  }


def fill_fake_model(model_msg, frame_id: int, timestamp_eof: int, v_ego: float) -> None:
  target_accel = 0.8 if v_ego < 15.0 else 0.0
  data = _straight_plan(v_ego, target_accel)

  model_msg.valid = True
  model = model_msg.modelV2
  model.frameId = frame_id
  model.frameIdExtra = frame_id
  model.frameAge = 0
  model.frameDropPerc = 0.0
  model.timestampEof = timestamp_eof
  model.modelExecutionTime = 0.001

  fill_xyzt(model.position, ModelConstants.T_IDXS, *data['plan'][0, :, Plan.POSITION].T,
            *data['plan_stds'][0, :, Plan.POSITION].T)
  fill_xyzt(model.velocity, ModelConstants.T_IDXS, *data['plan'][0, :, Plan.VELOCITY].T)
  fill_xyzt(model.acceleration, ModelConstants.T_IDXS, *data['plan'][0, :, Plan.ACCELERATION].T)
  fill_xyzt(model.orientation, ModelConstants.T_IDXS, *data['plan'][0, :, Plan.T_FROM_CURRENT_EULER].T)
  fill_xyzt(model.orientationRate, ModelConstants.T_IDXS, *data['plan'][0, :, Plan.ORIENTATION_RATE].T)

  model.action.desiredCurvature = 0.0
  model.action.desiredAcceleration = float(target_accel)
  model.action.shouldStop = False

  model.init('laneLines', 4)
  for lane_line, y_offset in zip(model.laneLines, [-5.5, -1.8, 1.8, 5.5], strict=True):
    _fill_lane_line(lane_line, y_offset)
  model.laneLineStds = [0.1, 0.1, 0.1, 0.1]
  model.laneLineProbs = [1.0, 1.0, 1.0, 1.0]

  model.init('roadEdges', 2)
  for road_edge, y_offset in zip(model.roadEdges, [-7.0, 7.0], strict=True):
    _fill_lane_line(road_edge, y_offset)
  model.roadEdgeStds = [0.1, 0.1]

  model.init('leadsV3', 3)
  for i, lead in enumerate(model.leadsV3):
    _fill_lead(lead, i)

  meta = model.meta
  meta.desireState = [0.0] * ModelConstants.DESIRE_LEN
  meta.desirePrediction = [0.0] * (ModelConstants.DESIRE_LEN * ModelConstants.DESIRE_PRED_LEN)
  meta.engagedProb = 1.0
  meta.laneChangeState = log.LaneChangeState.off
  meta.laneChangeDirection = log.LaneChangeDirection.none
  meta.init('disengagePredictions')
  disengage = meta.disengagePredictions
  disengage.t = ModelConstants.META_T_IDXS
  disengage.brakeDisengageProbs = [0.0] * ModelConstants.DISENGAGE_WIDTH
  disengage.gasDisengageProbs = [0.0] * ModelConstants.DISENGAGE_WIDTH
  disengage.steerOverrideProbs = [0.0] * ModelConstants.DISENGAGE_WIDTH
  disengage.brake3MetersPerSecondSquaredProbs = [0.0] * ModelConstants.DISENGAGE_WIDTH
  disengage.brake4MetersPerSecondSquaredProbs = [0.0] * ModelConstants.DISENGAGE_WIDTH
  disengage.brake5MetersPerSecondSquaredProbs = [0.0] * ModelConstants.DISENGAGE_WIDTH
  disengage.gasPressProbs = [0.0] * (len(ModelConstants.META_T_IDXS) + 1)
  disengage.brakePressProbs = [0.0] * (len(ModelConstants.META_T_IDXS) + 1)
  meta.hardBrakePredicted = False
  model.confidence = log.ModelDataV2.ConfidenceClass.green


def fill_fake_camera_odometry(msg, frame_id: int, timestamp_eof: int, v_ego: float) -> None:
  msg.valid = True
  camera_odometry = msg.cameraOdometry
  camera_odometry.frameId = frame_id
  camera_odometry.timestampEof = timestamp_eof
  camera_odometry.trans = [float(v_ego * DT_MDL), 0.0, 0.0]
  camera_odometry.rot = [0.0, 0.0, 0.0]
  camera_odometry.wideFromDeviceEuler = [0.0, 0.0, 0.0]
  camera_odometry.roadTransformTrans = [0.0, 0.0, 0.0]
  camera_odometry.transStd = [0.01, 0.01, 0.01]
  camera_odometry.rotStd = [0.01, 0.01, 0.01]
  camera_odometry.wideFromDeviceEulerStd = [0.01, 0.01, 0.01]
  camera_odometry.roadTransformTransStd = [0.01, 0.01, 0.01]


def main() -> None:
  pm = messaging.PubMaster(['modelV2', 'drivingModelData', 'cameraOdometry', 'livePose'])
  sm = messaging.SubMaster(['carState', 'roadCameraState'])
  rk = Ratekeeper(ModelConstants.MODEL_RUN_FREQ, print_delay_threshold=None)
  fallback_frame_id = 0

  while True:
    sm.update(0)
    fallback_frame_id += 1
    frame_id = sm['roadCameraState'].frameId if sm.seen['roadCameraState'] else fallback_frame_id
    timestamp_eof = sm['roadCameraState'].timestampEof if sm.seen['roadCameraState'] else int(time.monotonic() * 1e9)
    v_ego = max(float(sm['carState'].vEgo), 0.0) if sm.seen['carState'] else 0.0

    model_msg = messaging.new_message('modelV2')
    fill_fake_model(model_msg, frame_id, timestamp_eof, v_ego)

    driving_msg = messaging.new_message('drivingModelData')
    fill_driving_model_data(driving_msg, model_msg)

    odometry_msg = messaging.new_message('cameraOdometry')
    fill_fake_camera_odometry(odometry_msg, frame_id, timestamp_eof, v_ego)

    live_pose_msg = generate_livePose()
    live_pose_msg.valid = True

    pm.send('modelV2', model_msg)
    pm.send('drivingModelData', driving_msg)
    pm.send('cameraOdometry', odometry_msg)
    pm.send('livePose', live_pose_msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
