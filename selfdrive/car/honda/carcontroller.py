from collections import namedtuple
import os
import common.numpy_fast as np
from common.numpy_fast import clip, interp
from common.realtime import sec_since_boot

from selfdrive.config import CruiseButtons
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit

from . import hondacan
from .values import AH


def actuator_hystereses(brake, braking, brake_steady, v_ego, civic):
  # hyst params... TODO: move these to VehicleParams
  brake_hyst_on = 0.02     # to activate brakes exceed this value
  brake_hyst_off = 0.005                     # to deactivate brakes below this value
  brake_hyst_gap = 0.01                      # don't change brake command for small ocilalitons within this value

  #*** histeresys logic to avoid brake blinking. go above 0.1 to trigger
  if (brake < brake_hyst_on and not braking) or brake < brake_hyst_off:
    brake = 0.
  braking = brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  if brake == 0.:
    brake_steady = 0.
  elif brake > brake_steady + brake_hyst_gap:
    brake_steady = brake - brake_hyst_gap
  elif brake < brake_steady - brake_hyst_gap:
    brake_steady = brake + brake_hyst_gap
  brake = brake_steady

  if not civic and brake > 0.0:
    brake += 0.15

  return brake, braking, brake_steady


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


HUDData = namedtuple("HUDData",
                     ["pcm_accel", "v_cruise", "X2", "car", "X4", "X5",
                      "lanes", "beep", "X8", "chime", "acc_alert"])


class CarController(object):
  def __init__(self, enable_camera=True):
    self.braking = False
    self.brake_steady = 0.
    self.brake_last = 0.
    self.enable_camera = enable_camera

  def update(self, sendcan, enabled, CS, frame, actuators, \
             pcm_speed, pcm_override, pcm_cancel_cmd, pcm_accel, \
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert, \
             snd_beep, snd_chime):
    """ Controls thread """

    # TODO: Make the accord work.
    if CS.accord or not self.enable_camera:
      return

    # *** apply brake hysteresis ***
    brake, self.braking, self.brake_steady = actuator_hystereses(actuators.brake, self.braking, self.brake_steady, CS.v_ego, CS.civic)

    # *** no output if not enabled ***
    if not enabled and CS.pcm_acc_status:
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      pcm_cancel_cmd = True

    # *** rate limit after the enable check ***
    self.brake_last = rate_limit(brake, self.brake_last, -2., 1./100)

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

    hud = HUDData(int(pcm_accel), int(round(hud_v_cruise)), 0x01, hud_car,
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
    if CS.civic:
      is_fw_modified = os.getenv("DONGLE_ID") in ['b0f5a01cf604185cxxx']
      STEER_MAX = 0x1FFF if is_fw_modified else 0x1000
    elif CS.crv:
      STEER_MAX = 0x300  # CR-V only uses 12-bits and requires a lower value
    else:
      STEER_MAX = 0xF00
    GAS_OFFSET = 328

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_gas = int(clip(actuators.gas * GAS_MAX, 0, GAS_MAX - 1))
    apply_brake = int(clip(self.brake_last * BRAKE_MAX, 0, BRAKE_MAX - 1))
    apply_steer = int(clip(-actuators.steer * STEER_MAX, -STEER_MAX, STEER_MAX))

    # any other cp.vl[0x18F]['STEER_STATUS'] is common and can happen during user override. sending 0 torque to avoid EPS sending error 5
    if CS.steer_not_allowed:
      apply_steer = 0

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
