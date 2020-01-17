from selfdrive.car import apply_toyota_steer_torque_limits
from selfdrive.car.chrysler.chryslercan import create_lkas_hud, create_lkas_command, \
                                               create_wheel_buttons
from selfdrive.car.chrysler.values import ECU, CAR, SteerLimitParams
from opendbc.can.packer import CANPacker

class CarController():
  def __init__(self, dbc_name, car_fingerprint, enable_camera):
    self.braking = False
    # redundant safety check with the board
    self.controls_allowed = True
    self.apply_steer_last = 0
    self.ccframe = 0
    self.prev_frame = -1
    self.hud_count = 0
    self.car_fingerprint = car_fingerprint
    self.alert_active = False
    self.gone_fast_yet = False
    self.steer_rate_limited = False

    self.fake_ecus = set()
    if enable_camera:
      self.fake_ecus.add(ECU.CAM)

    self.packer = CANPacker(dbc_name)


  def update(self, enabled, CS, actuators, pcm_cancel_cmd, hud_alert):
    # this seems needed to avoid steering faults and to force the sync with the EPS counter
    frame = CS.lkas_counter
    if self.prev_frame == frame:
      return []

    # *** compute control surfaces ***
    # steer torque
    new_steer = actuators.steer * SteerLimitParams.STEER_MAX
    apply_steer = apply_toyota_steer_torque_limits(new_steer, self.apply_steer_last,
                                                   CS.steer_torque_motor, SteerLimitParams)
    self.steer_rate_limited = new_steer != apply_steer

    moving_fast = CS.v_ego > CS.CP.minSteerSpeed  # for status message
    if CS.v_ego > (CS.CP.minSteerSpeed - 0.5):  # for command high bit
      self.gone_fast_yet = True
    elif self.car_fingerprint in (CAR.PACIFICA_2019_HYBRID, CAR.JEEP_CHEROKEE_2019):
      if CS.v_ego < (CS.CP.minSteerSpeed - 3.0):
        self.gone_fast_yet = False  # < 14.5m/s stock turns off this bit, but fine down to 13.5
    lkas_active = moving_fast and enabled

    if not lkas_active:
      apply_steer = 0

    self.apply_steer_last = apply_steer

    can_sends = []

    #*** control msgs ***

    if pcm_cancel_cmd:
      # TODO: would be better to start from frame_2b3
      new_msg = create_wheel_buttons(self.ccframe)
      can_sends.append(new_msg)

    # LKAS_HEARTBIT is forwarded by Panda so no need to send it here.
    # frame is 100Hz (0.01s period)
    if (self.ccframe % 25 == 0):  # 0.25s period
      if (CS.lkas_car_model != -1):
        new_msg = create_lkas_hud(
            self.packer, CS.gear_shifter, lkas_active, hud_alert,
            self.hud_count, CS.lkas_car_model)
        can_sends.append(new_msg)
        self.hud_count += 1

    new_msg = create_lkas_command(self.packer, int(apply_steer), self.gone_fast_yet, frame)
    can_sends.append(new_msg)

    self.ccframe += 1
    self.prev_frame = frame

    return can_sends
