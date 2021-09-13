from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.ram.ramcan import create_lkas_command, create_wheel_buttons
from selfdrive.car.ram.values import CAR, CarControllerParams
from opendbc.can.packer import CANPacker


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0
    self.braking = False
    #    self.alert_active = False
    #    self.hud_count = 0
    self.car_fingerprint = CP.carFingerprint
    self.steer_rate_limited = False
    self.prev_frame = -1
    self.steer_command_bit = False

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, CS, frame, actuators, pcm_cancel_cmd):  # , leftLaneVisible,
    #   rightLaneVisible, autoHighBeamBit):  # TODO make HUD better and fix auto high beams before re-enabling this
    P = CarControllerParams

    steer_ready = CS.out.vEgo > CS.CP.minSteerSpeed

    if steer_ready:
      self.steer_command_bit = True
    if not steer_ready:
      self.steer_command_bit = False

    bad_to_bone = enabled and steer_ready

    if bad_to_bone:
      new_steer = int(round(actuators.steer * P.STEER_MAX))
    else:
      new_steer = 0

    apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
    self.steer_rate_limited = new_steer != apply_steer
    self.apply_steer_last = apply_steer

    # *** control msgs ***

    can_sends = []

    # *** control msgs ***

    #    if pcm_cancel_cmd:  # TODO: ENABLE ONCE STEERING WORKS
    #      new_msg = create_wheel_buttons(self.packer, self.frame, cancel=True)
    #      can_sends.append(new_msg)
    counter = (frame / P.STEER_STEP) % 16

    if frame % 2 == 0:
      can_sends.append(create_lkas_command(self.packer, int(apply_steer), counter, self.steer_command_bit))

    #    if frame % 5 == 0:
    #      can_sends.append(create_lkas_hud(self.packer, enabled, leftLaneVisible, rightLaneVisible, autoHighBeamBit))

    return can_sends
