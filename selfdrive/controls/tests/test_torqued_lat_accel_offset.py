import numpy as np
from cereal import car, messaging
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car import structs
from opendbc.car.lateral import get_friction, FRICTION_THRESHOLD
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.locationd.torqued import TorqueEstimator, MIN_BUCKET_POINTS, POINTS_PER_BUCKET, STEER_BUCKET_BOUNDS

np.random.seed(0)

LA_ERR_STD = 1.0
INPUT_NOISE_STD = 0.08
V_EGO = 30.0

WARMUP_BUCKET_POINTS = (1.5*MIN_BUCKET_POINTS).astype(int)
STRAIGHT_ROAD_LA_BOUNDS = (0.02, 0.03)

ROLL_BIAS_DEG = 2.0
ROLL_COMPENSATION_BIAS = ACCELERATION_DUE_TO_GRAVITY*float(np.sin(np.deg2rad(ROLL_BIAS_DEG)))
TORQUE_TUNE = structs.CarParams.LateralTorqueTuning(latAccelFactor=2.0, latAccelOffset=0.0, friction=0.2)
TORQUE_TUNE_BIASED = structs.CarParams.LateralTorqueTuning(latAccelFactor=2.0, latAccelOffset=-ROLL_COMPENSATION_BIAS, friction=0.2)

def generate_inputs(torque_tune, la_err_std, input_noise_std=None):
  rng = np.random.default_rng(0)
  steer_torques = np.concat([rng.uniform(bnd[0], bnd[1], pts) for bnd, pts in zip(STEER_BUCKET_BOUNDS, WARMUP_BUCKET_POINTS, strict=True)])
  la_errs = rng.normal(scale=la_err_std, size=steer_torques.size)
  frictions = np.array([get_friction(la_err, 0.0, FRICTION_THRESHOLD, torque_tune) for la_err in la_errs])
  lat_accels = torque_tune.latAccelFactor*steer_torques + torque_tune.latAccelOffset + frictions
  if input_noise_std is not None:
    steer_torques += rng.normal(scale=input_noise_std, size=steer_torques.size)
    lat_accels += rng.normal(scale=input_noise_std, size=steer_torques.size)
  return steer_torques, lat_accels

def get_warmed_up_estimator(steer_torques, lat_accels):
  est = TorqueEstimator(car.CarParams())
  for steer_torque, lat_accel in zip(steer_torques, lat_accels, strict=True):
    est.filtered_points.add_point(steer_torque, lat_accel)
  return est

def simulate_straight_road_msgs(est):
  carControl = messaging.new_message('carControl').carControl
  carOutput = messaging.new_message('carOutput').carOutput
  carState = messaging.new_message('carState').carState
  livePose = messaging.new_message('livePose').livePose
  carControl.latActive = True
  carState.vEgo = V_EGO
  carState.steeringPressed = False
  ts = DT_MDL*np.arange(2*POINTS_PER_BUCKET)
  steer_torques = np.concat((np.linspace(-0.03, -0.02, POINTS_PER_BUCKET), np.linspace(0.02, 0.03, POINTS_PER_BUCKET)))
  lat_accels = TORQUE_TUNE.latAccelFactor * steer_torques
  for t, steer_torque, lat_accel in zip(ts, steer_torques, lat_accels, strict=True):
    carOutput.actuatorsOutput.torque = float(-steer_torque)
    livePose.orientationNED.x = float(np.deg2rad(ROLL_BIAS_DEG))
    livePose.angularVelocityDevice.z = float(lat_accel / V_EGO)
    for which, msg in (('carControl', carControl), ('carOutput', carOutput), ('carState', carState), ('livePose', livePose)):
      est.handle_log(t, which, msg)

def test_estimated_offset():
  steer_torques, lat_accels = generate_inputs(TORQUE_TUNE_BIASED, la_err_std=LA_ERR_STD, input_noise_std=INPUT_NOISE_STD)
  est = get_warmed_up_estimator(steer_torques, lat_accels)
  msg = est.get_msg()
  # TODO add lataccelfactor and friction check when we have more accurate estimates
  assert abs(msg.liveTorqueParameters.latAccelOffsetRaw - TORQUE_TUNE_BIASED.latAccelOffset) < 0.1

def test_straight_road_roll_bias():
  steer_torques, lat_accels = generate_inputs(TORQUE_TUNE, la_err_std=LA_ERR_STD, input_noise_std=INPUT_NOISE_STD)
  est = get_warmed_up_estimator(steer_torques, lat_accels)
  simulate_straight_road_msgs(est)
  msg = est.get_msg()
  assert (msg.liveTorqueParameters.latAccelOffsetRaw < -0.05) and np.isfinite(msg.liveTorqueParameters.latAccelOffsetRaw)
