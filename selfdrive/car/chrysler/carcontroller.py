from cereal import car
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_toyota_steer_torque_limits
from selfdrive.car.chrysler.chryslercan import create_lkas_hud, create_lkas_command, create_cruise_buttons
from selfdrive.car.chrysler.values import CarControllerParams


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.frame = 0
    self.steer_rate_limited = False

    self.hud_count = 0
    self.prev_lkas_frame = -1

    self.params = CarControllerParams(CP)

    self.packer = CANPacker(dbc_name)

  def update(self, CC, CS, low_speed_alert):
    # TODO: look into whether this is really needed. just need to send LKAS_COMMAND at a lower frequency?
    # this seems needed to avoid steering faults and to force the sync with the EPS counter
    if self.prev_lkas_frame == CS.lkas_counter:
      return car.CarControl.Actuators.new_message(), []

    actuators = CC.actuators

    lkas_active = CC.latActive and not low_speed_alert

    # steer torque
    new_steer = int(round(actuators.steer * self.params.STEER_MAX))
    apply_steer = apply_toyota_steer_torque_limits(new_steer, self.apply_steer_last,
                                                   CS.out.steeringTorqueEps, self.params)
    self.steer_rate_limited = new_steer != apply_steer
    self.apply_steer_last = apply_steer

    can_sends = []

    # *** control msgs ***

    if CC.cruiseControl.cancel:
      can_sends.append(create_cruise_buttons(self.packer, CS.button_counter + 1, self.params.BUTTONS_BUS, cancel=True))

    if self.frame % 25 == 0:
      if CS.lkas_car_model != -1:
        can_sends.append(create_lkas_hud(self.packer, lkas_active, CC.hudControl.visualAlert, self.hud_count, CS, self.CP.carFingerprint))
        self.hud_count += 1

    can_sends.append(create_lkas_command(self.packer, self.CP, int(apply_steer), lkas_active, self.frame))

    self.frame += 1
    self.prev_lkas_frame = CS.lkas_counter

    new_actuators = actuators.copy()
    new_actuators.steer = apply_steer / self.params.STEER_MAX

    return new_actuators, can_sends
