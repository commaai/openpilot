import numpy as np
from cereal import car, messaging
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car import structs
from opendbc.car.lateral import get_friction, FRICTION_THRESHOLD
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.locationd.torqued import TorqueEstimator, MIN_BUCKET_POINTS, POINTS_PER_BUCKET, STEER_BUCKET_BOUNDS

np.random.seed(0)

ROLL_BIAS_DEG = 1.0
TOTAL_POINTS = POINTS_PER_BUCKET*len(STEER_BUCKET_BOUNDS)
MAX_STEER = STEER_BUCKET_BOUNDS[-1][-1]
TORQUE_STD = 0.1
IN_OUT_STD = 0.1
V_EGO = 30.0

# how much larger roll compensation is than it should be
roll_compensation_bias = ACCELERATION_DUE_TO_GRAVITY*float(np.sin(np.deg2rad(ROLL_BIAS_DEG)))
gt_torque_tune = structs.CarParams.LateralTorqueTuning(latAccelFactor=2.0, latAccelOffset=-roll_compensation_bias, friction=0.2)

def test_estimated_offset():
  rng = np.random.default_rng(0)
  steer_torques = rng.normal(scale=TORQUE_STD, size=TOTAL_POINTS)
  steer_torques = steer_torques[abs(steer_torques) < MAX_STEER]
  la_errs = rng.normal(scale=0.5, size=steer_torques.size)
  frictions = np.array([get_friction(la_err, 0.0, FRICTION_THRESHOLD, gt_torque_tune) for la_err in la_errs])
  lat_accels = gt_torque_tune.latAccelFactor*steer_torques + gt_torque_tune.latAccelOffset + frictions
  steer_torques += rng.normal(scale=IN_OUT_STD, size=steer_torques.size)
  lat_accels += rng.normal(scale=IN_OUT_STD, size=steer_torques.size)
  import matplotlib.pyplot as plt
  plt.scatter(lat_accels, steer_torques)
  plt.savefig('a.png')
  est = TorqueEstimator(car.CarParams())
  for steer_torque, lat_accel in zip(steer_torques, lat_accels, strict=True):
    est.filtered_points.add_point(steer_torque, lat_accel)
  msg = est.get_msg()
  print(msg)
  # TODO add lataccelfactor and friction check when we have more accurate estimates
  assert abs(msg.liveTorqueParameters.latAccelOffsetRaw - gt_torque_tune.latAccelOffset) < 0.05

def test_straight_road_roll_bias():
  rng = np.random.default_rng(0)
  steer_torques = np.concat([rng.uniform(bound[0], bound[1], int(points))
                             for bound, points in zip(STEER_BUCKET_BOUNDS, 1.5*MIN_BUCKET_POINTS, strict=True)])
  la_errs = rng.normal(size=steer_torques.size)
  frictions = np.array([get_friction(la_err, 0.0, FRICTION_THRESHOLD, gt_torque_tune) for la_err in la_errs])
  lat_accels = gt_torque_tune.latAccelFactor*steer_torques + frictions
  est = TorqueEstimator(car.CarParams())
  for steer_torque, lat_accel in zip(steer_torques, lat_accels, strict=True):
    est.filtered_points.add_point(steer_torque, lat_accel)
  for i in range(2*POINTS_PER_BUCKET):
    t = i*DT_MDL
    sgn = (-1)**(i < POINTS_PER_BUCKET)
    steer_torque = sgn * rng.uniform(0.02, 0.03)
    lat_accel = gt_torque_tune.latAccelFactor * steer_torque
    carControl = messaging.new_message('carControl').carControl
    carControl.latActive = True
    carOutput = messaging.new_message('carOutput').carOutput
    carOutput.actuatorsOutput.torque = -steer_torque
    carState = messaging.new_message('carState').carState
    carState.vEgo = V_EGO
    carState.steeringPressed = False
    livePose = messaging.new_message('livePose').livePose
    livePose.orientationNED.x = float(np.deg2rad(ROLL_BIAS_DEG))
    livePose.angularVelocityDevice.z = lat_accel / V_EGO
    msgs = {'carControl': carControl, 'carOutput': carOutput, 'carState': carState, 'livePose': livePose}
    for which, msg in msgs.items():
      est.handle_log(t, which, msg)
  msg = est.get_msg()
  assert (msg.liveTorqueParameters.latAccelOffsetRaw < -0.05) and np.isfinite(msg.liveTorqueParameters.latAccelOffsetRaw)
