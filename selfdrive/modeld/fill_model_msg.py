import capnp
import numpy as np
from typing import List, Dict
from openpilot.selfdrive.modeld.models.driving_pyx import PublishState
from openpilot.selfdrive.modeld.constants import T_IDXS, X_IDXS, lEAD_T_IDXS, META_T_IdXS

def fill_xyzt(builder, t, x, y, z, x_std=None, y_std=None, z_std=None):
  builder.t = t
  builder.x = x.tolist()
  builder.y = y.tolist()
  builder.z = z.tolist()
  if x_std is not None: builder.xStd = x_std.tolist()
  if y_std is not None: builder.yStd = y_std.tolist()
  if z_std is not None: builder.zStd = z_std.tolist()

def fill_xyvat(builder, t, x, y, v, a, x_std=None, y_std=None, v_std=None, a_std=None):
  builder.t = t
  builder.x = x.tolist()
  builder.y = y.tolist()
  builder.v = v.tolist()
  builder.a = a.tolist()
  if x_std is not None: builder.xStd = x_std.tolist()
  if y_std is not None: builder.yStd = y_std.tolist()
  if v_std is not None:builder.vStd = v_std.tolist()
  if a_std is not None:builder.aStd = a_std.tolist()

def fill_model_msg(msg: capnp._DynamicStructBuilder, net_output_data: Dict[str, np.ndarray], ps: PublishState,
                   vipc_frame_id: int, vipc_frame_id_extra: int, frame_id: int, frame_drop: float,
                   timestamp_eof: int, timestamp_llk: int, model_execution_time: float,
                   nav_enabled: bool, valid: bool) -> None:
  frame_age = frame_id - vipc_frame_id if frame_id > vipc_frame_id else 0
  msg.valid = valid

  modelV2 = msg.modelV2
  modelV2.frameId = vipc_frame_id
  modelV2.frameIdExtra = vipc_frame_id_extra
  modelV2.frameAge = frame_age
  modelV2.frameDropPerc = frame_drop * 100
  modelV2.timestampEof = timestamp_eof
  modelV2.locationMonoTime = timestamp_llk
  modelV2.modelExecutionTime = model_execution_time
  modelV2.navEnabled = nav_enabled

  # plan
  fill_xyzt(modelV2.position, T_IDXS, net_output_data['plan'][0,:,0], net_output_data['plan'][0,:,1], net_output_data['plan'][0,:,2],
    net_output_data['plan_stds'][0,:,0], net_output_data['plan_stds'][0,:,1], net_output_data['plan_stds'][0,:,2])
  fill_xyzt(modelV2.velocity, T_IDXS, net_output_data['plan'][0,:,3], net_output_data['plan'][0,:,4], net_output_data['plan'][0,:,5])
  fill_xyzt(modelV2.acceleration, T_IDXS, net_output_data['plan'][0,:,6], net_output_data['plan'][0,:,7], net_output_data['plan'][0,:,8])
  fill_xyzt(modelV2.orientation, T_IDXS, net_output_data['plan'][0,:,9], net_output_data['plan'][0,:,10], net_output_data['plan'][0,:,11])
  fill_xyzt(modelV2.orientationRate, T_IDXS, net_output_data['plan'][0,:,12], net_output_data['plan'][0,:,13], net_output_data['plan'][0,:,14])

  # lane lines
  modelV2.init('laneLines', 4)
  for i in range(4):
    fill_xyzt(modelV2.laneLines[i], T_IDXS, np.array(X_IDXS), net_output_data['lane_lines'][0,i,:,0], net_output_data['lane_lines'][0,i,:,1])
  modelV2.laneLineStds = net_output_data['lane_lines_stds'][0,:,0,0].tolist()
  modelV2.laneLineProbs = net_output_data['lane_lines_prob'][0,1::2].tolist()

  # road edges
  modelV2.init('roadEdges', 2)
  for i in range(2):
    fill_xyzt(modelV2.roadEdges[i], T_IDXS, np.array(X_IDXS), net_output_data['road_edges'][0,i,:,0], net_output_data['road_edges'][0,i,:,1])
  modelV2.roadEdgeStds = net_output_data['road_edges_stds'][0,:,0,0].tolist()

  # leads
  modelV2.init('leadsV3', 2)
  for i in range(2):
    fill_xyvat(modelV2.leadsV3[i], lEAD_T_IDXS, net_output_data['lead'][0,i,:,0], net_output_data['lead'][0,i,:,1], net_output_data['lead'][0,i,:,2], net_output_data['lead'][0,i,:,3],
      net_output_data['lead_stds'][0,i,:,0], net_output_data['lead_stds'][0,i,:,1], net_output_data['lead_stds'][0,i,:,2], net_output_data['lead_stds'][0,i,:,3])

  # leads probs
  # TODO

  # confidence
  # TODO
  modelV2.confidence = 0

  # meta
  # TODO
  modelV2.meta.engagedProb = 0.
  modelV2.meta.hardBrakePredicted = False
  modelV2.meta.init('disengagePredictions')
  modelV2.meta.disengagePredictions.t = META_T_IdXS
  modelV2.meta.disengagePredictions.brakeDisengageProbs = np.zeros(5, dtype=np.float32).tolist()
  modelV2.meta.disengagePredictions.gasDisengageProbs = np.zeros(5, dtype=np.float32).tolist()
  modelV2.meta.disengagePredictions.steerOverrideProbs = np.zeros(5, dtype=np.float32).tolist()
  modelV2.meta.disengagePredictions.brake3MetersPerSecondSquaredProbs = np.zeros(5, dtype=np.float32).tolist()
  modelV2.meta.disengagePredictions.brake4MetersPerSecondSquaredProbs = np.zeros(5, dtype=np.float32).tolist()
  modelV2.meta.disengagePredictions.brake5MetersPerSecondSquaredProbs = np.zeros(5, dtype=np.float32).tolist()
  modelV2.meta.desirePrediction = np.zeros(4*8, dtype=np.float32).tolist()

  # temporal pose
  modelV2.temporalPose.trans = net_output_data['sim_pose'][0,:3].tolist()
  modelV2.temporalPose.transStd = net_output_data['sim_pose_stds'][0,:3].tolist()
  modelV2.temporalPose.rot = net_output_data['sim_pose'][0,3:].tolist()
  modelV2.temporalPose.rotStd = net_output_data['sim_pose_stds'][0,3:].tolist()


def fill_pose_msg(msg: capnp._DynamicStructBuilder, net_output_data: Dict[str, np.ndarray],
                  vipc_frame_id: int, vipc_dropped_frames: int, timestamp_eof: int, live_calib_seen: bool) -> None:
  msg.valid = live_calib_seen & (vipc_dropped_frames < 1)
  cameraOdometry = msg.cameraOdometry

  cameraOdometry.frameId = vipc_frame_id
  cameraOdometry.timestampEof = timestamp_eof

  cameraOdometry.trans = net_output_data['pose'][0,:3].tolist()
  cameraOdometry.rot = net_output_data['pose'][0,3:].tolist()
  cameraOdometry.wideFromDeviceEuler = net_output_data['wide_from_device_euler'][0,:].tolist()
  cameraOdometry.roadTransformTrans = net_output_data['road_transform'][0,:3].tolist()
  cameraOdometry.transStd = net_output_data['pose_stds'][0,:3].tolist()
  cameraOdometry.rotStd = net_output_data['pose_stds'][0,3:].tolist()
  cameraOdometry.wideFromDeviceEulerStd = net_output_data['wide_from_device_euler_stds'][0,:].tolist()
  cameraOdometry.roadTransformTransStd = net_output_data['road_transform_stds'][0,:3].tolist()
