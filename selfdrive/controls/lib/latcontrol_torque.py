from collections import deque
import math
import numpy as np

from cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from opendbc.car.tests.test_lateral_limits import MAX_LAT_JERK_UP
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# use a LOW_SPEED_FACTOR in the error. Additionally, there is
# friction in the steering wheel that needs to be overcome to
# move it at all, this is compensated for too.

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [15, 13, 10, 5]

class LatControlTorque(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.pid = PIDController(self.torque_params.kp, self.torque_params.ki, k_d=self.torque_params.kd, k_f=self.torque_params.kf, rate=1/self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lataccel_request_buffer_size = int(1 / self.dt)
    self.requested_lateral_accel_buffer = deque([0.] * self.lataccel_request_buffer_size , maxlen=self.lataccel_request_buffer_size)
    self.error_pre = 0.0
    self.measurement_pre = 0.0
    self.gravity_adjusted_lateral_accel_pre = 0.0
    self.error_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * MAX_LAT_JERK_UP), self.dt)
    self.ff_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * MAX_LAT_JERK_UP), self.dt)
    self.jerk_ff_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * MAX_LAT_JERK_UP), self.dt)

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay: float):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
      roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
      curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))

      delay_frames = min(max(int(lat_delay / self.dt), 1), self.lataccel_request_buffer_size - 1)
      plan_future_desired_lateral_accel = desired_curvature * CS.vEgo ** 2
      self.requested_lateral_accel_buffer.append(plan_future_desired_lateral_accel)
      current_expected_lateral_accel = self.requested_lateral_accel_buffer[-(delay_frames + 1)]
      current_expected_curvature = current_expected_lateral_accel / (CS.vEgo ** 2)
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      low_speed_factor = np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)**2
      # pid error calculated as difference between expected and measured lateral acceleration
      setpoint_expected = current_expected_lateral_accel + low_speed_factor * current_expected_curvature
      measurement = actual_lateral_accel + low_speed_factor * actual_curvature
      gravity_adjusted_lateral_accel = plan_future_desired_lateral_accel - roll_compensation
      error = float(setpoint_expected - measurement)
      meas_rate = (measurement - self.measurement_pre) / self.dt
      meas_rate_filtered = self.error_rate_filter.update(meas_rate)
      jerk_ff = (gravity_adjusted_lateral_accel - setpoint_expected) / (delay_frames * self.dt)
      jerk_ff_filtered = self.jerk_ff_filter.update(jerk_ff)
      self.error_pre = error
      self.measurement_pre = measurement
      self.gravity_adjusted_lateral_accel_pre = gravity_adjusted_lateral_accel
      # do error correction in lateral acceleration space, convert at end to handle non-linear torque responses correctly
      pid_log.error = float(error)
      ff = gravity_adjusted_lateral_accel
      # latAccelOffset corrects roll compensation bias from device roll misalignment relative to car roll
      ff -= self.torque_params.latAccelOffset
      ff += self.torque_params.friction * np.tanh(2.5 * jerk_ff_filtered + 0.5*error) # get_friction(jerk_ff_filtered, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_lataccel = self.pid.update(pid_log.error,
                                        -meas_rate_filtered,
                                        feedforward=ff,
                                        speed=CS.vEgo,
                                        freeze_integrator=freeze_integrator,)
                                        #error_expected=error_expected)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque)  # TODO: log lat accel?
      pid_log.actualLateralAccel = float(actual_lateral_accel)
      pid_log.desiredLateralAccel = float(plan_future_desired_lateral_accel)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
