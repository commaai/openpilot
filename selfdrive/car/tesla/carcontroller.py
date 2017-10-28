from collections import namedtuple

import common.numpy_fast as np
from common.numpy_fast import clip, interp
from common.realtime import sec_since_boot

from selfdrive.config import CruiseButtons
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit

from . import teslacan

def actuator_hystereses(final_brake, braking, brake_steady, v_ego):
  # hyst params... TODO: move these to VehicleParams
  brake_hyst_on = 0.1                        # to activate brakes exceed this value
  brake_hyst_off = 0.005                     # to deactivate brakes below this value
  brake_hyst_gap = 0.01                      # don't change brake command for small ocilalitons within this value

  final_brake = 0.
  braking = False
  brake_steady = 0.
  
  
  #*** histeresys logic to avoid brake blinking. go above 0.1 to trigger
  #if (final_brake < brake_hyst_on and not braking) or final_brake < brake_hyst_off:
  #  final_brake = 0.
  #braking = final_brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  #if final_brake == 0.:
  #  brake_steady = 0.
  #elif final_brake > brake_steady + brake_hyst_gap:
  #  brake_steady = final_brake - brake_hyst_gap
  #elif final_brake < brake_steady - brake_hyst_gap:
  #  brake_steady = final_brake + brake_hyst_gap
  #final_brake = brake_steady

  #if not civic:
  #  brake_on_offset_v  = [.25, .15]   # min brake command on brake activation. below this no decel is perceived
  #  brake_on_offset_bp = [15., 30.]     # offset changes VS speed to not have too abrupt decels at high speeds
    # offset the brake command for threshold in the brake system. no brake torque perceived below it
  #  brake_on_offset = interp(v_ego, brake_on_offset_bp, brake_on_offset_v)
  #  brake_offset = brake_on_offset - brake_hyst_on
  #  if final_brake > 0.0:
  #    final_brake += brake_offset

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

#HUDData = namedtuple("HUDData",
#                     ["pcm_accel", "v_cruise", "X2", "car", "X4", "X5",
#                      "lanes", "beep", "X8", "chime", "acc_alert"])

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

    # *** apply brake hysteresis ***
    #final_brake, self.braking, self.brake_steady = actuator_hystereses(final_brake, self.braking, self.brake_steady, CS.v_ego)

    # *** no output if not enabled ***
    if not enabled:
      final_gas = 0.
      final_brake = 0.
      final_steer = 0.
    
    #Override for now
    final_gas = 0.
    final_brake = 0.
    
    # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
    #if CS.pcm_acc_status:
    #  pcm_cancel_cmd = True

    # *** rate limit after the enable check ***
    #final_brake = rate_limit(final_brake, self.final_brake_last, -2., 1./100)
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

    # **** process the car messages ****

    # *** compute control surfaces ***
    tt = sec_since_boot()
    GAS_MAX = 1004
    BRAKE_MAX = 1024/4
    SIGNAL_STEER_MAX = 16384
    GAS_OFFSET = 328
    
    if CS.v_ego < 16.7: #60.12 km/h divided by 3.6 = 16.7 meter per sec
      USER_STEER_MAX = 180 # 180 degrees
    elif CS.v_ego < 28: # 100.8 km/h 
      USER_STEER_MAX = 900 # 90 degrees
    else:
      USER_STEER_MAX = 300 # 30 degrees 
	
    apply_gas = 0 #int(clip(final_gas*GAS_MAX, 0, GAS_MAX-1))
    apply_brake = 0 #int(clip(final_brake*BRAKE_MAX, 0, BRAKE_MAX-1))

    #final_steer is between -1 and 1
    if not enabled:
      final_steer = 0
    else:
      # steer torque is converted back to CAN reference (positive when steering right)
	  final_steer = final_steer * -1
    apply_steer = int(clip((final_steer * 100) + SIGNAL_STEER_MAX - (CS.angle_steers * 10), SIGNAL_STEER_MAX - USER_STEER_MAX, SIGNAL_STEER_MAX + USER_STEER_MAX))

    # no gas if you are hitting the brake or the user is
    if apply_gas > 0 and (apply_brake != 0 or CS.brake_pressed):
      print "CANCELLING ACCELERATOR", apply_brake
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
    #if (CS.prev_cruise_buttons == 2) and CS.cruise_buttons == 0 and not self.controls_allowed:
    if CS.cruise_buttons == 2 and not self.controls_allowed:
      print "CONTROLS ARE LIVE"
      self.controls_allowed = True

    # *** exit from controls state on cancel, gas, or brake ***
    if (CS.cruise_buttons == 1 or CS.brake_pressed ) and self.controls_allowed:
      print "CONTROLS ARE DEAD (Cruise CANCEL or BRAKE pressed"
      self.controls_allowed = False

    # *** controls fail on steer error, brake error, or invalid can ***
    if CS.steer_error:
      print "STEER ERROR"
      self.controls_allowed = False

    if CS.brake_error:
      print "BRAKE ERROR"
      self.controls_allowed = False

    #if not CS.can_valid and self.controls_allowed:   # 200 ms
    #  print "CAN INVALID"
    #  self.controls_allowed = False

    # Send CAN commands.
    can_sends = []

    # Send steering command.
    if (frame % 5) == 0:
      idx = (frame / 5) % 16
      enable_steer_control = (not CS.right_blinker_on and not CS.left_blinker_on and self.controls_allowed and enabled)
      can_sends.append(teslacan.create_steering_control(apply_steer, idx, enable_steer_control))
      #if idx == 0: print "carcontroller.py: CS.angle_steers: %.2f CS.steer_override: %.2f final_steer: %.2f apply_steer: %.2f" % (CS.angle_steers, CS.steer_override, final_steer, apply_steer)
      #print "carcontroller.py: apply_steer = " + str(apply_steer) + ", idx = " + str(idx)
      sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())

