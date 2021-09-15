from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.stellantis.stellantiscan import create_lkas_command, create_lkas_hud, create_wheel_buttons
from selfdrive.car.stellantis.values import CarControllerParams as P
from opendbc.can.packer import CANPacker


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0
    self.steer_rate_limited = False
    self.steer_command_bit = False

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, CS, frame, actuators, hudControl, pcm_cancel_cmd):
    can_sends = []

    # **** Steering Controls ************************************************ #

    if frame % P.STEER_STEP == 0:

      steer_ready = CS.out.vEgo > CS.CP.minSteerSpeed
      if steer_ready:
        self.steer_command_bit = True
        new_steer = int(round(actuators.steer * P.STEER_MAX)) if enabled else 0
      else:
        self.steer_command_bit = False
        new_steer = 0

      apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
      self.steer_rate_limited = new_steer != apply_steer
      self.apply_steer_last = apply_steer

      counter = (frame / P.STEER_STEP) % 16
      can_sends.append(create_lkas_command(self.packer, int(apply_steer), counter, self.steer_command_bit))

    # **** HUD Controls ***************************************************** #

    if frame % P.HUD_STEP == 0:
      can_sends.append(create_lkas_hud(self.packer, enabled, hudControl.leftLaneVisible, hudControl.rightLaneVisible,
                                       CS.stock_lkas_hud_values))

    # **** ACC Button Controls ********************************************** #

    if pcm_cancel_cmd:
      new_msg = create_wheel_buttons(self.packer, frame, cancel=True)
      can_sends.append(new_msg)

    return can_sends
