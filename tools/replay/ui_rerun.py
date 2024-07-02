#!/usr/bin/env python3
import argparse
import os
import sys
import cereal.messaging as messaging
import numpy as np
from openpilot.common.numpy_fast import clip
import rerun as rr
from msgq.visionipc import VisionIpcClient, VisionStreamType

img = np.zeros((480, 640, 3), dtype='uint8')
ANGLE_SCALE = 5.0

def getMsgs(addr):
  prevCarStateTime = -1
  prevCarControlTime = -1
  prevControlsStateTime = -1
  prevLongitudinalPlanTime = -1

  sm = messaging.SubMaster(['carState', 'longitudinalPlan', 'carControl', 'radarState', 'liveCalibration', 'controlsState',
                          'liveTracks', 'modelV2', 'liveParameters', 'roadCameraState'], addr=addr)
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)

  while True:
    # ***** frame *****
    if not vipc_client.is_connected():
      vipc_client.connect(True)

    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any():
      continue

    sm.update(0)
    carStateTime = sm.logMonoTime['carState']
    if carStateTime != prevCarStateTime:
      rr.set_time_nanos("TIMELINE", carStateTime)
      prevCarStateTime = carStateTime
      a_ego = sm['carState'].aEgo
      rr.log("/a/a_ego", rr.Scalar(a_ego))
      v_ego = sm['carState'].vEgo
      rr.log("/v/v_ego", rr.Scalar(v_ego))
      v_cruise = sm['carState'].cruiseState.speed
      rr.log("/v/v_cruise", rr.Scalar(v_cruise))
      angle_steers = sm['carState'].steeringAngleDeg
      rr.log("/angle/angle_steers", rr.Scalar(angle_steers))
      gas = sm['carState'].gas
      rr.log("acc/gas", rr.Scalar(gas))
      user_brake = sm['carState'].brake
      rr.log("acc/user_brake", rr.Scalar(user_brake))

    carControlTime = sm.logMonoTime['carControl']
    if prevCarControlTime != carControlTime:
      rr.set_time_nanos("TIMELINE", carControlTime)
      prevCarControlTime = carControlTime
      angle_steers_des = sm['carControl'].actuators.steeringAngleDeg
      rr.log("angle/angle_steers_des", rr.Scalar(angle_steers_des))
      steer_torque = sm['carControl'].actuators.steer * ANGLE_SCALE
      rr.log("angle/steer_torque", rr.Scalar(steer_torque))
      computer_gas = clip(sm['carControl'].actuators.accel/4.0, 0.0, 1.0)
      rr.log("acc/computer_gas", rr.Scalar(computer_gas))
      computer_brake = clip(-sm['carControl'].actuators.accel/4.0, 0.0, 1.0)
      rr.log("acc/computer_brake", rr.Scalar(computer_brake))

    controlsStateTime = sm.logMonoTime['controlsState']
    if prevControlsStateTime != controlsStateTime:
      rr.set_time_nanos("TIMELINE", controlsStateTime)
      prevControlsStateTime = controlsStateTime
      w = sm['controlsState'].lateralControlState.which()
      if w == 'lqrStateDEPRECATED':
          angle_steers_k = sm['controlsState'].lateralControlState.lqrStateDEPRECATED.steeringAngleDeg
      elif w == 'indiState':
          angle_steers_k = sm['controlsState'].lateralControlState.indiState.steeringAngleDeg
      else:
          angle_steers_k = np.inf
      rr.log("angle/angle_steers_k", rr.Scalar(angle_steers_k))

    longitudinalPlanTime = sm.logMonoTime['longitudinalPlan']
    if prevLongitudinalPlanTime != longitudinalPlanTime:
      rr.set_time_nanos("TIMELINE", longitudinalPlanTime)
      prevLongitudinalPlanTime = longitudinalPlanTime
      if len(sm['longitudinalPlan'].accels):
        a_target = sm['longitudinalPlan'].accels[0]
        rr.log("/a/a_target", rr.Scalar(a_target))


def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Show replay data in a UI.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("ip_address", nargs="?", default="127.0.0.1",
                      help="The ip address on which to receive zmq messages.")

  parser.add_argument("--frame-address", default=None,
                      help="The frame address (fully qualified ZMQ endpoint for frames) on which to receive zmq messages.")
  return parser


if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])
  rr.init("rerun_test")
  rr.spawn()
  if args.ip_address != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.context = messaging.Context()
  getMsgs(args.ip_address)
