from opendbc.can.packer import CANPacker
from selfdrive.car import apply_toyota_steer_torque_limits
from selfdrive.car.chrysler.chryslercan import create_lkas_hud, create_lkas_command, create_cruise_buttons
from selfdrive.car.chrysler.values import RAM_CARS, CarControllerParams


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.frame = 0
    self.steer_rate_limited = False

    self.hud_count = 0
    self.last_lkas_falling_edge = 0
    self.lkas_active_prev = False

    self.packer = CANPacker(dbc_name)

  def update(self, CC, CS, low_speed_alert):
    can_sends = []

    # EPS faults if LKAS re-enables too quickly
    lkas_active = CC.latActive and not low_speed_alert and (self.frame - self.last_lkas_falling_edge > 200)

    # *** control msgs ***

    das_bus = 2 if self.CP.carFingerprint in RAM_CARS else 0
    if CC.cruiseControl.cancel:
      can_sends.append(create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, cancel=True))
    elif CC.enabled and CS.out.cruiseState.standstill:
      can_sends.append(create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, resume=True))

    # HUD alerts
    if self.frame % 25 == 0:
      if CS.lkas_car_model != -1:
        can_sends.append(create_lkas_hud(self.packer, self.CP, lkas_active, CC.hudControl.visualAlert, self.hud_count, CS.lkas_car_model, CS.auto_high_beam))
        self.hud_count += 1

    # steering
    if self.frame % 2 == 0:
      # steer torque
      new_steer = int(round(CC.actuators.steer * CarControllerParams.STEER_MAX))
      apply_steer = apply_toyota_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorqueEps, CarControllerParams)
      if not lkas_active:
        apply_steer = 0
      self.steer_rate_limited = new_steer != apply_steer
      self.apply_steer_last = apply_steer

      idx = self.frame // 2
      can_sends.append(create_lkas_command(self.packer, self.CP, int(apply_steer), lkas_active, idx))

    self.frame += 1
    if not lkas_active and self.lkas_active_prev:
      self.last_lkas_falling_edge = self.frame
    self.lkas_active_prev = lkas_active

    new_actuators = CC.actuators.copy()
    new_actuators.steer = self.apply_steer_last / CarControllerParams.STEER_MAX

    return new_actuators, can_sends
