from cereal import car
from common.numpy_fast import interp
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car import apply_toyota_steer_torque_limits
from selfdrive.car.chrysler.chryslercan import create_lkas_hud, create_lkas_command, \
                                               create_wheel_buttons, create_lkas_heartbit, \
                                               create_chimes
from selfdrive.car.modules.ALCA_module import ALCAController
from selfdrive.car.chrysler.values import ECU, CAR
from selfdrive.can.packer import CANPacker

AudibleAlert = car.CarControl.HUDControl.AudibleAlert
LOUD_ALERTS = [AudibleAlert.chimeWarning1, AudibleAlert.chimeWarning2, AudibleAlert.chimeWarningRepeat]

class SteerLimitParams:
  STEER_MAX = 261         # 262 faults
  STEER_DELTA_UP = 3      # 3 is stock. 100 is fine. 200 is too much it seems
  STEER_DELTA_DOWN = 3    # no faults on the way down it seems
  STEER_ERROR_MAX = 80


class CarController(object):
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
    self.send_chime = False
    self.gone_fast_yet = False
    
    self.ALCA = ALCAController(self,True,False)  # Enabled  True and SteerByAngle only False
    
    self.fake_ecus = set()
    if enable_camera:
      self.fake_ecus.add(ECU.CAM)

    self.packer = CANPacker(dbc_name)


  def update(self, sendcan, enabled, CS, frame, actuators,
             pcm_cancel_cmd, hud_alert, audible_alert):
    #update custom UI buttons and alerts
    CS.UE.update_custom_ui()
    if (frame % 1000 == 0):
      CS.cstm_btns.send_button_info()
      CS.UE.uiSetCarEvent(CS.cstm_btns.car_folder,CS.cstm_btns.car_name)

    # Get the angle from ALCA.
    alca_enabled = False
    alca_steer = 0.
    alca_angle = 0.
    turn_signal_needed = 0
    # Update ALCA status and custom button every 0.1 sec.
    if self.ALCA.pid == None:
      self.ALCA.set_pid(CS)
    if (frame % 10 == 0):
      self.ALCA.update_status(CS.cstm_btns.get_button_status("alca") > 0)
    # steer torque
    alca_angle, alca_steer, alca_enabled, turn_signal_needed = self.ALCA.update(enabled, CS, frame, actuators)
    
    # this seems needed to avoid steering faults and to force the sync with the EPS counter
    frame = CS.lkas_counter
    if self.prev_frame == frame:
      return

    # *** compute control surfaces ***
    # steer torque
    apply_steer = alca_steer * SteerLimitParams.STEER_MAX
    apply_steer = apply_toyota_steer_torque_limits(apply_steer, self.apply_steer_last,
                                                   CS.steer_torque_motor, SteerLimitParams)

    moving_fast = CS.v_ego > CS.CP.minSteerSpeed  # for status message
    if CS.v_ego > (CS.CP.minSteerSpeed - 0.5):  # for command high bit
      self.gone_fast_yet = True
    elif self.car_fingerprint in (CAR.PACIFICA_2019_HYBRID, CAR.JEEP_CHEROKEE_2019):
      if CS.v_ego < (CS.CP.minSteerSpeed - 3.0):
        self.gone_fast_yet = False  # < 14.5m/s stock turns off this bit, but fine down to 13.5
    lkas_active = moving_fast and enabled

    if not lkas_active:
      apply_steer = 0
      
    if not CS.lane_departure_toggle_on:
        apply_steer = 0
    self.apply_steer_last = apply_steer
    
        
    if audible_alert in LOUD_ALERTS:
      self.send_chime = True

    can_sends = []

    #*** control msgs ***

    if self.send_chime:
      new_msg = create_chimes(AudibleAlert)
      can_sends.append(new_msg)
      if audible_alert not in LOUD_ALERTS:
        self.send_chime = False

    if pcm_cancel_cmd:
      # TODO: would be better to start from frame_2b3
      new_msg = create_wheel_buttons(self.ccframe)
      can_sends.append(new_msg)

    # frame is 100Hz (0.01s period)
    if (self.ccframe % 10 == 0):  # 0.1s period
      new_msg = create_lkas_heartbit(self.packer, CS.lkas_status_ok)
      can_sends.append(new_msg)

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
    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
