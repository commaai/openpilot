from collections import namedtuple

import common.numpy_fast as np
from common.realtime import sec_since_boot
from selfdrive.config import CruiseButtons
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit
from common.numpy_fast import clip, interp

def actuator_hystereses(final_brake, braking, brake_steady, v_ego, civic):
  # hyst params... TODO: move these to VehicleParams
  brake_hyst_on = 0.055 if civic else 0.1    # to activate brakes exceed this value
  brake_hyst_off = 0.005                     # to deactivate brakes below this value
  brake_hyst_gap = 0.01                      # don't change brake command for small ocilalitons within this value

  #*** histeresys logic to avoid brake blinking. go above 0.1 to trigger
  if (final_brake < brake_hyst_on and not braking) or final_brake < brake_hyst_off:
    final_brake = 0.
  braking = final_brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  if final_brake == 0.:
    brake_steady = 0.
  elif final_brake > brake_steady + brake_hyst_gap:
    brake_steady = final_brake - brake_hyst_gap
  elif final_brake < brake_steady - brake_hyst_gap:
    brake_steady = final_brake + brake_hyst_gap
  final_brake = brake_steady

  if not civic:
    brake_on_offset_v  = [.25, .15]   # min brake command on brake activation. below this no decel is perceived
    brake_on_offset_bp = [15., 30.]     # offset changes VS speed to not have too abrupt decels at high speeds
    # offset the brake command for threshold in the brake system. no brake torque perceived below it
    brake_on_offset = interp(v_ego, brake_on_offset_bp, brake_on_offset_v)
    brake_offset = brake_on_offset - brake_hyst_on
    if final_brake > 0.0:
      final_brake += brake_offset

  return final_brake, braking, brake_steady

class AH:
  #[alert_idx, value]
  # See dbc files for info on values"
  NONE           = [0, 0]
  FCW            = [1, 0x8]
  STEER          = [2, 1]
  BRAKE_PRESSED  = [3, 10]
  GEAR_NOT_D     = [4, 6]
  SEATBELT       = [5, 5]
  SPEED_TOO_HIGH = [6, 8]


def process_hud_alert(hud_alert):
  # initialize to no alert
  fcw_display = 0
  steer_required = 0
  acc_alert = 0
  if hud_alert == AH.NONE:          # no alert
    pass
  elif hud_alert == AH.FCW:         # FCW
    fcw_display = hud_alert[1]
  elif hud_alert == AH.STEER:       # STEER
    steer_required = hud_alert[1]
  else:                             # any other ACC alert
    acc_alert = hud_alert[1]

  return fcw_display, steer_required, acc_alert

import selfdrive.car.honda.hondacan as hondacan

HUDData = namedtuple("HUDData",
                     ["pcm_accel", "v_cruise", "X2", "car", "X4", "X5",
                      "lanes", "beep", "X8", "chime", "acc_alert"])

class CarController(object):
  def __init__(self):
    self.braking = False
    self.brake_steady = 0.
    self.final_brake_last = 0.

    # redundant safety check with the board
    self.controls_allowed = False

  def update(self, sendcan, enabled, CS, frame, final_gas, final_brake, final_steer, \
             pcm_speed, pcm_override, pcm_cancel_cmd, pcm_accel, \
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert, \
             snd_beep, snd_chime):
    """ Controls thread """

    # TODO: Make the accord work.
    if CS.accord:
      return

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

    hud = HUDData(int(pcm_accel), int(hud_v_cruise), 0x01, hud_car,
                  0xc1, 0x41, hud_lanes + steer_required,
                  int(snd_beep), 0x48, (snd_chime << 5) + fcw_display, acc_alert)

    if not all(isinstance(x, int) and 0 <= x < 256 for x in hud):
      print "INVALID HUD", hud
      hud = HUDData(0xc6, 255, 64, 0xc0, 209, 0x41, 0x40, 0, 0x48, 0, 0)

    # **** process the car messages ****

    # *** compute control surfaces ***
    tt = sec_since_boot()
    GAS_MAX = 1004
    BRAKE_MAX = 1024/4
    if CS.crv:
      STEER_MAX = 0x380  # CR-V only uses 12-bits and requires a lower value
    else:
      STEER_MAX = 0xF00
    GAS_OFFSET = 328

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_gas = int(clip(final_gas*GAS_MAX, 0, GAS_MAX-1))
    apply_brake = int(clip(final_brake*BRAKE_MAX, 0, BRAKE_MAX-1))
    # crvtodo: tweak steering to match precision of 0xE4 code.
    apply_steer = int(clip(-final_steer*STEER_MAX, -STEER_MAX, STEER_MAX))

    # no gas if you are hitting the brake or the user is
    if apply_gas > 0 and (apply_brake != 0 or CS.brake_pressed):
      print "CANCELLING GAS", apply_brake
      apply_gas = 0

    # no computer brake if the gas is being pressed
    if CS.car_gas > 0 and apply_brake != 0:
      print "CANCELLING BRAKE"
      apply_brake = 0

    # any other cp.vl[0x18F]['STEER_STATUS'] is common and can happen during user override. sending 0 torque to avoid EPS sending error 5
    if CS.steer_not_allowed:
      print "STEER ALERT, TORQUE INHIBITED"
      apply_steer = 0

    # *** entry into controls state ***
    if (CS.prev_cruise_buttons == CruiseButtons.DECEL_SET or CS.prev_cruise_buttons == CruiseButtons.RES_ACCEL) and \
        CS.cruise_buttons == 0 and not self.controls_allowed:
      print "CONTROLS ARE LIVE"
      self.controls_allowed = True

    # *** exit from controls state on cancel, gas, or brake ***
    if (CS.cruise_buttons == CruiseButtons.CANCEL or CS.brake_pressed or
        CS.user_gas_pressed or (CS.pedal_gas > 0 and CS.brake_only)) and self.controls_allowed:
      print "CONTROLS ARE DEAD"
      self.controls_allowed = False

    # *** controls fail on steer error, brake error, or invalid can ***
    if CS.steer_error:
      print "STEER ERROR"
      self.controls_allowed = False

    # crvtodo, fix brake error, might be issue with dbc.
    if CS.brake_error and not CS.crv:
      print "BRAKE ERROR"
      self.controls_allowed = False

    if not CS.can_valid and self.controls_allowed:   # 200 ms
      print "CAN INVALID"
      self.controls_allowed = False

    # Send CAN commands.
    can_sends = []

    # Send steering command.
    if CS.accord:
      idx = frame % 2
      can_sends.append(hondacan.create_accord_steering_control(apply_steer, idx))
    else:
      idx = frame % 4
      can_sends.extend(hondacan.create_steering_control(apply_steer, CS.crv, idx))

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
      can_sends.extend(hondacan.create_ui_commands(pcm_speed, hud, CS.civic, CS.accord, CS.crv, idx))

    # radar at 20Hz, but these msgs need to be sent at 50Hz on ilx (seems like an Acura bug)
    if CS.civic or CS.accord or CS.crv:
      radar_send_step = 5
    else:
      radar_send_step = 2

    if (frame % radar_send_step) == 0:
      idx = (frame/radar_send_step) % 4
      can_sends.extend(hondacan.create_radar_commands(CS.v_ego, CS.civic, CS.accord, CS.crv, idx))

    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
