from opendbc.can.packer import CANPacker
from opendbc.car import Bus
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.psa.psacan import create_lka_steering
from opendbc.car.psa.values import CarControllerParams


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.main])
    self.apply_angle_last = 0
    self.status = 2

  def update(self, CC, CS, now_nanos):
    can_sends = []
    actuators = CC.actuators

    # lateral control
    if self.frame % 5 == 0:
      apply_angle = apply_std_steer_angle_limits(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgoRaw,
                                                 CS.out.steeringAngleDeg, CC.latActive, CarControllerParams.ANGLE_LIMITS)

      # EPS disengages on steering override, activation sequence 2->3->4 to re-engage
      # STATUS  -  0: UNAVAILABLE, 1: UNSELECTED, 2: READY, 3: AUTHORIZED, 4: ACTIVE
      if not CC.latActive:
        self.status = 2
      elif not CS.eps_active and not CS.out.steeringPressed:
        self.status = 2 if self.status == 4 else self.status + 1
      else:
        self.status = 4

      can_sends.append(create_lka_steering(self.packer, CC.latActive, apply_angle, self.status))

      self.apply_angle_last = apply_angle

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    self.frame += 1
    return new_actuators, can_sends
