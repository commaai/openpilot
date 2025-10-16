import math
import numpy as np
from collections import deque

from cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from opendbc.car.tests.test_lateral_limits import MAX_LAT_JERK_UP
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# add a LOW_SPEED_FACTOR to the proportional gain. Additionally, 
# there is friction in the steering wheel that needs to be overcome
# to move it at all, this is compensated for too.

LOW_SPEED_INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 10, 20, 30]
LOW_SPEED_INTERP_KP = np.array([250, 120, 65, 30, 10, 2.0, 0.3, 0.0])

# filter jerk and measurement rate with cutoff frequency equal jerk up limit
JERK_FILTER_TAU_SECONDS = 1 / (2 * np.pi * MAX_LAT_JERK_UP)
JERK_LOOKAHEAD_SECONDS = 0.19
JERK_GAIN = 0.1
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0

class LatControlTorque(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    k_p = [LOW_SPEED_INTERP_SPEEDS, LOW_SPEED_INTERP_KP + self.torque_params.kp]
    self.pid = PIDController(k_p, self.torque_params.ki, k_f=self.torque_params.kf, rate=1/self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lookahead_frames = int(JERK_LOOKAHEAD_SECONDS / self.dt)
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.] * self.lat_accel_request_buffer_len , maxlen=self.lat_accel_request_buffer_len)
    self.jerk_filter = FirstOrderFilter(0.0, JERK_FILTER_TAU_SECONDS, self.dt)
    self.measurement_rate_filter = FirstOrderFilter(0.0, JERK_FILTER_TAU_SECONDS, self.dt)
    self.previous_measurement = 0.0

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
      roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
      curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      delay_frames = int(np.clip(lat_delay / self.dt, 1, self.lat_accel_request_buffer_len))
      expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
      lookahead_idx = int(np.clip(-delay_frames + self.lookahead_frames, -self.lat_accel_request_buffer_len+1, -2))
      raw_lateral_jerk = (self.lat_accel_request_buffer[lookahead_idx+1] - self.lat_accel_request_buffer[lookahead_idx-1]) / (2 * self.dt)
      desired_lateral_jerk = self.jerk_filter.update(raw_lateral_jerk)
      future_desired_lateral_accel = desired_curvature * CS.vEgo ** 2
      self.lat_accel_request_buffer.append(future_desired_lateral_accel)
      gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation

      measurement = measured_curvature * CS.vEgo ** 2
      measurement_rate = self.measurement_rate_filter.update((measurement - self.previous_measurement) / self.dt)
      self.lat_accel_request_buffer.append(future_desired_lateral_accel)
      self.previous_measurement = measurement

      setpoint = expected_lateral_accel
      error = setpoint - measurement

      # do error correction in lateral acceleration space, convert at end to handle non-linear torque responses correctly
      pid_log.error = float(error)
      ff = gravity_adjusted_future_lateral_accel
      # latAccelOffset corrects roll compensation bias from device roll misalignment relative to car roll
      ff -= self.torque_params.latAccelOffset
      # TODO remove lateral jerk from feed forward - moving it from error means jerk is not scaled by low speed factor
      ff += JERK_GAIN * desired_lateral_jerk
      ff += get_friction(error+JERK_GAIN*desired_lateral_jerk, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_lataccel = self.pid.update(pid_log.error,
                                       -measurement_rate,
                                        feedforward=ff,
                                        speed=CS.vEgo,
                                        freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque)  # TODO: log lat accel?
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
