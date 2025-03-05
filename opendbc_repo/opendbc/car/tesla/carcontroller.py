import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import Bus, apply_std_steer_angle_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.tesla.teslacan import TeslaCAN
from opendbc.car.tesla.values import CarControllerParams


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.apply_angle_last = 0
    self.packer = CANPacker(dbc_names[Bus.party])
    self.tesla_can = TeslaCAN(self.packer)

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    can_sends = []

    # Disengage and allow for user override on high torque inputs
    # TODO: move this to a generic disengageRequested carState field and set CC.cruiseControl.cancel based on it
    hands_on_fault = CS.hands_on_level >= 3
    cruise_cancel = CC.cruiseControl.cancel or hands_on_fault
    lat_active = CC.latActive and not hands_on_fault

    if self.frame % 2 == 0:
      # Angular rate limit based on speed
      self.apply_angle_last = apply_std_steer_angle_limits(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgo,
                                                           CS.out.steeringAngleDeg, CC.latActive, CarControllerParams.ANGLE_LIMITS)

      can_sends.append(self.tesla_can.create_steering_control(self.apply_angle_last, lat_active, (self.frame // 2) % 16))

    if self.frame % 10 == 0:
      can_sends.append(self.tesla_can.create_steering_allowed((self.frame // 10) % 16))

    # Longitudinal control
    if self.CP.openpilotLongitudinalControl:
      if self.frame % 4 == 0:
        state = 13 if cruise_cancel else 4  # 4=ACC_ON, 13=ACC_CANCEL_GENERIC_SILENT
        accel = float(np.clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
        cntr = (self.frame // 4) % 8
        can_sends.append(self.tesla_can.create_longitudinal_command(state, accel, cntr, CC.longActive))

    else:
      # Increment counter so cancel is prioritized even without openpilot longitudinal
      if cruise_cancel:
        cntr = (CS.das_control["DAS_controlCounter"] + 1) % 8
        can_sends.append(self.tesla_can.create_longitudinal_command(13, 0, cntr, False))

    # TODO: HUD control
    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last

    self.frame += 1
    return new_actuators, can_sends
