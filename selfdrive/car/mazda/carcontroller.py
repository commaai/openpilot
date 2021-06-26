from selfdrive.car.mazda import mazdacan
from selfdrive.car.mazda.values import CarControllerParams, Buttons
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_std_steer_torque_limits
from common.dp_common import common_controller_ctrl

class CarController():
  def __init__(self, dbc_name, CP, VM):
    # dp
    self.last_blinker_on = False
    self.blinker_end_frame = 0.

    self.apply_steer_last = 0
    self.packer = CANPacker(dbc_name)
    self.steer_rate_limited = False

  def update(self, enabled, CS, frame, actuators, dragonconf):
    """ Controls thread """

    can_sends = []

    ### STEER ###

    if enabled:
      # calculate steer and also set limits due to driver torque
      new_steer = int(round(actuators.steer * CarControllerParams.STEER_MAX))
      apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last,
                                                  CS.out.steeringTorque, CarControllerParams)
      self.steer_rate_limited = new_steer != apply_steer

      if CS.out.standstill and frame % 5 == 0:
        # Mazda Stop and Go requires a RES button (or gas) press if the car stops more than 3 seconds
        # Send Resume button at 20hz if we're engaged at standstill to support full stop and go!
        # TODO: improve the resume trigger logic by looking at actual radar data
        can_sends.append(mazdacan.create_button_cmd(self.packer, CS.CP.carFingerprint, Buttons.RESUME))
    else:
      apply_steer = 0
      self.steer_rate_limited = False
      if CS.out.cruiseState.enabled and frame % 20 == 0:
        # Cancel Stock ACC if it's enabled while OP is disengaged
        # Send at a rate of 5hz until we sync with stock ACC state
        can_sends.append(mazdacan.create_button_cmd(self.packer, CS.CP.carFingerprint, Buttons.CANCEL))

    # dp
    blinker_on = CS.out.leftBlinker or CS.out.rightBlinker
    if not enabled:
      self.blinker_end_frame = 0
    if self.last_blinker_on and not blinker_on:
      self.blinker_end_frame = frame + dragonconf.dpSignalOffDelay
    apply_steer = common_controller_ctrl(enabled,
                                         dragonconf,
                                         blinker_on or frame < self.blinker_end_frame,
                                         apply_steer, CS.out.vEgo)
    self.last_blinker_on = blinker_on

    self.apply_steer_last = apply_steer

    can_sends.append(mazdacan.create_steering_control(self.packer, CS.CP.carFingerprint,
                                                      frame, apply_steer, CS.cam_lkas))
    return can_sends
