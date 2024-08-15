import copy
import numpy as np
from numbers import Number

from opendbc.can.packer import CANPacker
from openpilot.selfdrive.car import DT_CTRL
from openpilot.selfdrive.car.common.numpy_fast import clip, interp
from openpilot.selfdrive.car.body import bodycan
from openpilot.selfdrive.car.body.values import SPEED_FROM_RPM
from openpilot.selfdrive.car.interfaces import CarControllerBase


class PIController:
  def __init__(self, k_p, k_i, pos_limit=1e308, neg_limit=-1e308, rate=100):
    self._k_p = k_p
    self._k_i = k_i
    if isinstance(self._k_p, Number):
      self._k_p = [[0], [self._k_p]]
    if isinstance(self._k_i, Number):
      self._k_i = [[0], [self._k_i]]

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate
    self.speed = 0.0

    self.reset()

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  @property
  def error_integral(self):
    return self.i/self.k_i

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.control = 0

  def update(self, error, speed=0.0, freeze_integrator=False):
    self.speed = speed

    self.p = float(error) * self.k_p

    i = self.i + error * self.k_i * self.i_rate
    control = self.p + i

    # Update when changing i will move the control away from the limits
    # or when i will move towards the sign of the error
    if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or
        (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
       not freeze_integrator:
      self.i = i

    control = self.p + self.i

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control


MAX_TORQUE = 500
MAX_TORQUE_RATE = 50
MAX_ANGLE_ERROR = np.radians(7)
MAX_POS_INTEGRATOR = 0.2   # meters
MAX_TURN_INTEGRATOR = 0.1  # meters


class CarController(CarControllerBase):
  def __init__(self, dbc_name, CP):
    super().__init__(dbc_name, CP)
    self.packer = CANPacker(dbc_name)

    # PIDs
    self.turn_pid = PIController(110, k_i=11.5, rate=1/DT_CTRL)
    self.wheeled_speed_pid = PIController(110, k_i=11.5, rate=1/DT_CTRL)

    self.torque_r_filtered = 0.
    self.torque_l_filtered = 0.

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

    llk_valid = len(CC.orientationNED) > 1 and len(CC.angularVelocity) > 1
    if CC.enabled and llk_valid:
      # Read these from the joystick
      # TODO: this isn't acceleration, okay?
      speed_desired = CC.actuators.accel / 5.
      speed_diff_desired = -CC.actuators.steer / 2.

      speed_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr) / 2.
      speed_error = speed_desired - speed_measured

      torque = self.wheeled_speed_pid.update(speed_error, freeze_integrator=False)

      speed_diff_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)
      turn_error = speed_diff_measured - speed_diff_desired
      freeze_integrator = ((turn_error < 0 and self.turn_pid.error_integral <= -MAX_TURN_INTEGRATOR) or
                           (turn_error > 0 and self.turn_pid.error_integral >= MAX_TURN_INTEGRATOR))
      torque_diff = self.turn_pid.update(turn_error, freeze_integrator=freeze_integrator)

      # Combine 2 PIDs outputs
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

    can_sends = []
    can_sends.append(bodycan.create_control(self.packer, torque_l, torque_r))

    new_actuators = copy.copy(CC.actuators)
    new_actuators.accel = torque_l
    new_actuators.steer = torque_r
    new_actuators.steerOutputCan = torque_r

    self.frame += 1
    return new_actuators, can_sends
