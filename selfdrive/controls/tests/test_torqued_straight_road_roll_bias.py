import numpy as np
from cereal import car, messaging
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car import structs
from opendbc.car.lateral import get_friction, FRICTION_THRESHOLD
from openpilot.selfdrive.locationd.torqued import TorqueEstimator, MIN_BUCKET_POINTS, STEER_BUCKET_BOUNDS, POINTS_PER_BUCKET
from openpilot.common.realtime import DT_MDL
np.random.seed(0)

ROLL_BIAS_DEG = 1.0
V_EGO = 30.0

# how much larger roll compensation is than it should be
roll_compensation_bias = ACCELERATION_DUE_TO_GRAVITY*float(np.sin(np.deg2rad(ROLL_BIAS_DEG)))
gt_torque_tune = structs.CarParams.LateralTorqueTuning(latAccelFactor=2.0, latAccelOffset=0.0, friction=0.2)

def test_straight_road_roll_bias():
  steer_torques = np.concat([np.random.uniform(bound[0], bound[1], int(points))
                             for bound, points in zip(STEER_BUCKET_BOUNDS, 1.5*MIN_BUCKET_POINTS, strict=True)])
  la_errs = np.random.randn(steer_torques.size)
  frictions = np.array([get_friction(la_err, 0.0, FRICTION_THRESHOLD, gt_torque_tune) for la_err in la_errs])
  lat_accels = gt_torque_tune.latAccelFactor*steer_torques + frictions
  est = TorqueEstimator(car.CarParams())
  for steer_torque, lat_accel in zip(steer_torques, lat_accels, strict=True):
    est.filtered_points.add_point(steer_torque, lat_accel)
  for i in range(2*POINTS_PER_BUCKET):
    t = i*DT_MDL
    sgn = (-1)**(i < POINTS_PER_BUCKET)
    steer_torque = sgn * np.random.uniform(0.02, 0.03)
    lat_accel = gt_torque_tune.latAccelFactor * steer_torque
    livePose = messaging.new_message('livePose').livePose
    livePose.orientationNED.x = float(np.deg2rad(ROLL_BIAS_DEG))
    livePose.angularVelocityDevice.z = lat_accel / V_EGO
    est.raw_points["lat_active"].append(True)
    est.raw_points['vego'].append(V_EGO)
    est.raw_points["steer_torque"].append(steer_torque)
    est.raw_points["steer_override"].append(False)
    est.raw_points["carControl_t"].append(t)
    est.raw_points["carOutput_t"].append(t)
    est.raw_points["carState_t"].append(t)
    est.handle_log(t, "livePose", livePose)
  msg = est.get_msg()
  assert (msg.liveTorqueParameters.latAccelOffsetRaw < -0.05) and np.isfinite(msg.liveTorqueParameters.latAccelOffsetRaw)
