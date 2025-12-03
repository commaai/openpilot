import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL
from opendbc.car.common.pid import PIDController
from opendbc.car.body import bodycan
from opendbc.car.body.values import SPEED_FROM_RPM
from opendbc.car.interfaces import CarControllerBase

MAX_TORQUE = 1000
MAX_TORQUE_RATE = 50
MAX_ANGLE_ERROR = np.radians(7)
ACCEL_INPUT_MAX = 4.0      # expected acceleration command range (m/s^2)
TORQUE_SLEW_ACCEL = 400.0  # torque units per second when ramping up
TORQUE_SLEW_DECEL = 800.0  # torque units per second when ramping down
TORQUE_SLEW_ZERO = 1500.0   # torque units per second when releasing to zero
ACCEL_TO_TORQUE = MAX_TORQUE / ACCEL_INPUT_MAX
MAX_TURN_INTEGRATOR = 0.1  # meters
TURN_GAIN_BP = [0.0, 1.0, 3.0]   # m/s breakpoints for steering scaling
TURN_GAIN_V = [20.0, 15, 12.5]    # gain multipliers (high gain at low speed)
TURN_TORQUE_SLEW_ACCEL = 600.0  # torque units per second when increasing steering differential
TURN_TORQUE_SLEW_DECEL = 1000.0  # torque units per second when unwinding steering differential


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.main])

    self.turn_pid = PIDController(110, k_i=11.5, rate=1 / DT_CTRL)

    self.torque_r_filtered = 0.
    self.torque_l_filtered = 0.
    self._torque_desired_ramped = 0.
    self._torque_diff_ramped = 0.

  @staticmethod
  def deadband_filter(torque, deadband):
    if torque > 0:
      torque += deadband
    else:
      torque -= deadband
    return torque

  def update(self, CC, CS, now_nanos):

    torque_l = 0
    torque_r = 0

    if CC.enabled:
      # Ramp requested torque directly from the higher-level acceleration input
      target_torque = np.clip(CC.actuators.accel * ACCEL_TO_TORQUE, -MAX_TORQUE, MAX_TORQUE)
      current_torque = self._torque_desired_ramped
      delta_req = target_torque - current_torque
      current_mag = abs(current_torque)
      target_mag = abs(target_torque)

      if np.isclose(target_torque, 0.0, atol=1e-3):
        max_delta = TORQUE_SLEW_ZERO * DT_CTRL
      elif np.isclose(current_torque, 0.0, atol=1e-3):
        max_delta = TORQUE_SLEW_ACCEL * DT_CTRL if target_mag > current_mag else TORQUE_SLEW_DECEL * DT_CTRL
      elif np.sign(target_torque) == np.sign(current_torque):
        if target_mag >= current_mag:
          max_delta = TORQUE_SLEW_ACCEL * DT_CTRL
        else:
          max_delta = TORQUE_SLEW_DECEL * DT_CTRL
      else:
        # Crossing zero or reversing direction: drop torque quickly
        max_delta = TORQUE_SLEW_ZERO * DT_CTRL

      delta = np.clip(delta_req, -max_delta, max_delta)
      self._torque_desired_ramped += delta
      torque = self._torque_desired_ramped

      # Differential torque request comes from higher-level steering (positive = turn left)
      avg_speed = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr) / 2.
      turn_gain = np.interp(abs(avg_speed), TURN_GAIN_BP, TURN_GAIN_V)
      speed_diff_desired = - CC.actuators.torque / 2. * turn_gain
      speed_diff_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fr - CS.out.wheelSpeeds.fl)
      turn_error = speed_diff_desired - speed_diff_measured
      freeze_integrator = ((turn_error < 0 and self.turn_pid.error_integral <= -MAX_TURN_INTEGRATOR) or
                           (turn_error > 0 and self.turn_pid.error_integral >= MAX_TURN_INTEGRATOR))
      torque_diff_target = self.turn_pid.update(turn_error, freeze_integrator=freeze_integrator)
      if torque_diff_target == 0.0:
        max_turn_delta = TURN_TORQUE_SLEW_DECEL * DT_CTRL
      elif np.isclose(self._torque_diff_ramped, 0.0, atol=1e-3):
        max_turn_delta = TURN_TORQUE_SLEW_ACCEL * DT_CTRL
      elif (abs(torque_diff_target) >= abs(self._torque_diff_ramped) and
            np.sign(torque_diff_target) == np.sign(self._torque_diff_ramped)):
        max_turn_delta = TURN_TORQUE_SLEW_ACCEL * DT_CTRL
      else:
        max_turn_delta = TURN_TORQUE_SLEW_DECEL * DT_CTRL
      delta_turn = np.clip(torque_diff_target - self._torque_diff_ramped, -max_turn_delta, max_turn_delta)
      self._torque_diff_ramped += delta_turn
      torque_diff = np.clip(self._torque_diff_ramped, -MAX_TORQUE, MAX_TORQUE)

      # Combine base torque with steering differential
      torque_r = torque + torque_diff
      torque_l = torque - torque_diff

      # Torque rate limits
      self.torque_r_filtered = np.clip(self.deadband_filter(torque_r, 10),
                                       self.torque_r_filtered - MAX_TORQUE_RATE,
                                       self.torque_r_filtered + MAX_TORQUE_RATE)
      self.torque_l_filtered = np.clip(self.deadband_filter(torque_l, 10),
                                       self.torque_l_filtered - MAX_TORQUE_RATE,
                                       self.torque_l_filtered + MAX_TORQUE_RATE)
      torque_r = int(np.clip(self.torque_r_filtered, -MAX_TORQUE, MAX_TORQUE))
      torque_l = int(np.clip(self.torque_l_filtered, -MAX_TORQUE, MAX_TORQUE))
    else:
      self._torque_desired_ramped = 0.
      self.torque_r_filtered = 0.
      self.torque_l_filtered = 0.
      self._torque_diff_ramped = 0.
      self.turn_pid.reset()

    can_sends = []
    can_sends.append(bodycan.create_control(self.packer, torque_l, torque_r))

    new_actuators = CC.actuators.as_builder()
    new_actuators.accel = torque_l
    new_actuators.torque = torque_r / MAX_TORQUE
    new_actuators.torqueOutputCan = torque_r

    self.frame += 1
    return new_actuators, can_sends
