from cereal import car
import numpy as np
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car import structs
from opendbc.car.lateral import get_friction, FRICTION_THRESHOLD
from openpilot.selfdrive.locationd.torqued import TorqueEstimator, POINTS_PER_BUCKET, STEER_BUCKET_BOUNDS
np.random.seed(0)

ROLL_BIAS_DEG = 1.0
TOTAL_POINTS = POINTS_PER_BUCKET*len(STEER_BUCKET_BOUNDS)
MAX_STEER = STEER_BUCKET_BOUNDS[-1][-1]

# how much larger roll compensation is than it should be
roll_compensation_bias = ACCELERATION_DUE_TO_GRAVITY*float(np.sin(np.deg2rad(ROLL_BIAS_DEG))) 
gt_torque_tune = structs.CarParams.LateralTorqueTuning(latAccelFactor=2.0, latAccelOffset=-roll_compensation_bias, friction=0.2)

def test_estimated_bias():
  steer_torques = 0.2*np.random.randn(TOTAL_POINTS)
  steer_torques = steer_torques[abs(steer_torques) < MAX_STEER]
  la_errs = np.random.randn(steer_torques.size)
  lat_accels = gt_torque_tune.latAccelFactor*steer_torques + gt_torque_tune.latAccelOffset + gt_torque_tune.friction*np.array([get_friction(la_err, 0.0, FRICTION_THRESHOLD, gt_torque_tune) for la_err in la_errs])
  steer_torques += 0.1*np.random.randn(steer_torques.size)
  lat_accels += 0.1*np.random.randn(steer_torques.size)
  est = TorqueEstimator(car.CarParams())
  for steer_torque, lat_accel in zip(steer_torques, lat_accels):
    est.filtered_points.add_point(steer_torque, lat_accel)
  msg = est.get_msg()
  assert abs(msg.liveTorqueParameters.latAccelFactorRaw - gt_torque_tune.latAccelFactor) < 0.15
  assert abs(msg.liveTorqueParameters.latAccelOffsetRaw - gt_torque_tune.latAccelOffset) < 0.05
