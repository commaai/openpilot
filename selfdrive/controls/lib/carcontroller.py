from collections import namedtuple

import common.numpy_fast as np
import selfdrive.controls.lib.hondacan as hondacan
from common.realtime import sec_since_boot
from selfdrive.config import CruiseButtons
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.alert_database import process_hud_alert
from selfdrive.controls.lib.drive_helpers import actuator_hystereses, rate_limit

HUDData = namedtuple("HUDData",
                     ["pcm_accel", "v_cruise", "X2", "car", "X4", "X5",
                      "lanes", "beep", "X8", "chime", "acc_alert"])

class CarController(object):
  def __init__(self):
    self.controls_allowed = False
    self.mismatch_start, self.pcm_mismatch_start = 0, 0
    self.braking = False
    self.brake_steady = 0.
    self.final_brake_last = 0.

  def update(self, sendcan, enabled, CS, frame, final_gas, final_brake, final_steer, \
             pcm_speed, pcm_override, pcm_cancel_cmd, pcm_accel, \
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert, \
             snd_beep, snd_chime):
    """ Controls thread """

    # *** apply brake hysteresis ***
    final_brake, self.braking, self.brake_steady = actuator_hystereses(final_brake, self.braking, self.brake_steady, CS.v_ego, CS.civic)

    # *** no output if not enabled ***
    if not enabled:
      final_gas = 0.
      final_brake = 0.
      final_steer = 0.
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      if CS.pcm_acc_status:
        pcm_cancel_cmd = True

    # *** rate limit after the enable check ***
    final_brake = rate_limit(final_brake, self.final_brake_last, -2., 1./100)
    self.final_brake_last = final_brake

    # vehicle hud display, wait for one update from 10Hz 0x304 msg
    #TODO: use enum!!
    if hud_show_lanes:
      hud_lanes = 0x04
    else:
      hud_lanes = 0x00

    # TODO: factor this out better
    if enabled:
      if hud_show_car:
        hud_car = 0xe0
      else:
        hud_car = 0xd0
    else:
      hud_car = 0xc0

    #print chime, alert_id, hud_alert
    fcw_display, steer_required, acc_alert = process_hud_alert(hud_alert)

    hud = HUDData(pcm_accel, hud_v_cruise, 0x41, hud_car,
                  0xc1, 0x41, hud_lanes + steer_required,
                  snd_beep, 0x48, (snd_chime << 5) + fcw_display, acc_alert)

    if not all(isinstance(x, int) and 0 <= x < 256 for x in hud):
      print "INVALID HUD", hud
      hud = HUDData(0xc6, 255, 64, 0xc0, 209, 0x41, 0x40, 0, 0x48, 0, 0)

    # **** process the car messages ****

    user_brake_ctrl = CS.user_brake/0.015625  # FIXME: factor needed to convert to old scale

    # *** compute control surfaces ***
    tt = sec_since_boot()
    GAS_MAX = 1004
    BRAKE_MAX = 1024/4
    #STEER_MAX = 0xF00 if not CS.torque_mod else 0xF00/4    # ilx has 8x steering torque limit, used as a 2x
    STEER_MAX = 0xF00  # ilx has 8x steering torque limit, used as a 2x
    GAS_OFFSET = 328

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_gas = int(np.clip(final_gas*GAS_MAX, 0, GAS_MAX-1))
    apply_brake = int(np.clip(final_brake*BRAKE_MAX, 0, BRAKE_MAX-1))
    apply_steer = int(np.clip(-final_steer*STEER_MAX, -STEER_MAX, STEER_MAX))

    # no gas if you are hitting the brake or the user is
    if apply_gas > 0 and (apply_brake != 0 or user_brake_ctrl > 10):
      print "CANCELLING GAS", apply_brake, user_brake_ctrl
      apply_gas = 0

    # no computer brake if the user is hitting the gas
    # if the computer is trying to brake, it can't be hitting the gas
    # TODO: car_gas can override brakes without canceling... this is bad 
    if CS.car_gas > 0 and apply_brake != 0:
      apply_brake = 0

    if (CS.prev_cruise_buttons == CruiseButtons.DECEL_SET or CS.prev_cruise_buttons == CruiseButtons.RES_ACCEL) and \
        CS.cruise_buttons == 0 and not self.controls_allowed:
      print "CONTROLS ARE LIVE"
      self.controls_allowed = True

    # to avoid race conditions, check if control has been disabled for at least 0.2s
    # keep resetting start timer if mismatch isn't true
    if not (self.controls_allowed and not enabled):
      self.mismatch_start = tt

    # to avoid race conditions, check if control is disabled but pcm control is on for at least 0.2s
    if not (not self.controls_allowed and CS.pcm_acc_status):
      self.pcm_mismatch_start = tt

    # something is very wrong, since pcm control is active but controls should not be allowed; TODO: send pcm fault cmd?
    if (tt - self.pcm_mismatch_start) > 0.2:
      pcm_cancel_cmd = True

    # TODO: clean up gear condition, ideally only D (and P for debug) shall be valid gears
    if (CS.cruise_buttons == CruiseButtons.CANCEL or CS.brake_pressed or
        CS.user_gas_pressed or (tt - self.mismatch_start) > 0.2 or
        not CS.main_on or not CS.gear_shifter_valid or
        (CS.pedal_gas > 0 and CS.brake_only)) and self.controls_allowed:
      self.controls_allowed = False

    # 5 is a permanent fault, no torque request will be fullfilled
    if CS.steer_error:
      print "STEER ERROR"
      self.controls_allowed = False

    # any other cp.vl[0x18F]['STEER_STATUS'] is common and can happen during user override. sending 0 torque to avoid EPS sending error 5
    elif CS.steer_not_allowed:
      print "STEER ALERT, TORQUE INHIBITED"
      apply_steer = 0

    if CS.brake_error:
      print "BRAKE ERROR"
      self.controls_allowed = False

    if not CS.can_valid and self.controls_allowed:   # 200 ms
      print "CAN INVALID"
      self.controls_allowed = False

    if not self.controls_allowed:
      apply_steer = 0
      apply_gas = 0
      apply_brake = 0
      pcm_speed = 0          # make sure you send 0 target speed to pcm
      #pcm_cancel_cmd = 1    # prevent pcm control from turning on. FIXME: we can't just do this

    # Send CAN commands.
    can_sends = []

    # Send steering command.
    idx = frame % 4
    can_sends.append(hondacan.create_steering_control(apply_steer, idx))

    # Send gas and brake commands.
    if (frame % 2) == 0:
      idx = (frame / 2) % 4
      can_sends.append(
        hondacan.create_brake_command(apply_brake, pcm_override,
                                      pcm_cancel_cmd, hud.chime, idx))

      if not CS.brake_only:
        # send exactly zero if apply_gas is zero. Interceptor will send the max between read value and apply_gas. 
        # This prevents unexpected pedal range rescaling
        gas_amount = (apply_gas + GAS_OFFSET) * (apply_gas > 0)
        can_sends.append(hondacan.create_gas_command(gas_amount, idx))

    # Send dashboard UI commands.
    if (frame % 10) == 0:
      idx = (frame/10) % 4
      can_sends.extend(hondacan.create_ui_commands(pcm_speed, hud, CS.civic, idx))

    # radar at 20Hz, but these msgs need to be sent at 50Hz on ilx (seems like an Acura bug)
    if CS.civic:
      radar_send_step = 5
    else:
      radar_send_step = 2

    if (frame % radar_send_step) == 0:
      idx = (frame/radar_send_step) % 4
      can_sends.extend(hondacan.create_radar_commands(CS.v_ego, CS.civic, idx))

    sendcan.send(can_list_to_can_capnp(can_sends).to_bytes())
