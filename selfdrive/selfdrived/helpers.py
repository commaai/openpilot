import math

from cereal import car, messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.locationd.helpers import Pose
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX, ISO_LATERAL_ACCEL

MIN_EXCESSIVE_ACTUATION_COUNT = int(0.25 / DT_CTRL)
MIN_ENGAGE_BUFFER = int(1 / DT_CTRL)


def check_excessive_actuation(sm: messaging.SubMaster, CS: car.CarState, calibrated_pose: Pose, counter: int) -> tuple[int, bool, float]:
  # CS.aEgo can be noisy to bumps in the road, transitioning from standstill, losing traction, etc.
  # longitudinal
  accel_calibrated = calibrated_pose.acceleration.x
  excessive_long_actuation = sm['carControl'].longActive and (accel_calibrated > ACCEL_MAX * 2 or accel_calibrated < ACCEL_MIN * 2)

  # lateral
  yaw_rate = calibrated_pose.angular_velocity.yaw
  roll = sm['liveParameters'].roll
  roll_compensated_lateral_accel = (CS.vEgo * yaw_rate) - (math.sin(roll) * ACCELERATION_DUE_TO_GRAVITY)

  excessive_lat_actuation = False
  # print('vEgo', CS.vEgo, yaw_rate, roll)
  # print('roll_compensated_lateral_accel', roll_compensated_lateral_accel)
  if sm['carControl'].latActive:
    if not CS.steeringPressed:
      if abs(roll_compensated_lateral_accel) > ISO_LATERAL_ACCEL * 2:
        excessive_lat_actuation = True

  # livePose acceleration can be noisy due to bad mounting or aliased livePose measurements
  livepose_valid = abs(CS.aEgo - accel_calibrated) < 2
  # print('excessive_long_actuation', excessive_long_actuation, 'excessive_lat_actuation', excessive_lat_actuation, 'livepose_valid', livepose_valid)
  counter = counter + 1 if livepose_valid and (excessive_long_actuation or excessive_lat_actuation) else 0

  # if counter > 0:
  #   print('counter', counter, excessive_long_actuation, excessive_lat_actuation, livepose_valid)

  return counter, counter > MIN_EXCESSIVE_ACTUATION_COUNT, roll_compensated_lateral_accel
