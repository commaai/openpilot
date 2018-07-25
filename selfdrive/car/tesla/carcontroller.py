import os
from collections import namedtuple
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit
from common.numpy_fast import clip
from selfdrive.car.tesla import teslacan
from selfdrive.car.tesla.values import AH, CruiseButtons, CAR
from selfdrive.can.packer import CANPacker
from selfdrive.config import Conversions as CV


def actuator_hystereses(brake, braking, brake_steady, v_ego, car_fingerprint):
  # hyst params... TODO: move these to VehicleParams
  brake_hyst_on = 0.02     # to activate brakes exceed this value
  brake_hyst_off = 0.005   # to deactivate brakes below this value
  brake_hyst_gap = 0.01    # don't change brake command for small ocilalitons within this value

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
                     ["pcm_accel", "v_cruise", "mini_car", "car", "X4",
                      "lanes", "beep", "chime", "fcw", "acc_alert", "steer_required"])


class CarController(object):
  def __init__(self, dbc_name, enable_camera=True):
    self.braking = False
    self.brake_steady = 0.
    self.brake_last = 0.
    self.accelerating = False
    self.accel_steady = 0.
    self.accel_last = 0.
    self.enable_camera = enable_camera
    self.packer = CANPacker(dbc_name)
    self.epas_disabled = True

  def update(self, sendcan, enabled, CS, frame, actuators, \
             pcm_speed, pcm_override, pcm_cancel_cmd, pcm_accel, \
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert, \
             snd_beep, snd_chime):

    """ Controls thread """

    ## Todo add code to detect Tesla DAS (camera) and go into listen and record mode only (for AP1 / AP2 cars)
    if not self.enable_camera:
      return

    # *** apply brake hysteresis ***
    brake, self.braking, self.brake_steady = actuator_hystereses(actuators.brake, self.braking, self.brake_steady, CS.v_ego, CS.CP.carFingerprint)
    accel, self.accelerating, self.accel_steady =  actuator_hystereses(actuators.gas, self.accelerating, self.accel_steady, CS.v_ego, CS.CP.carFingerprint)
    # *** no output if not enabled ***
    if not enabled and CS.pcm_acc_status:
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      pcm_cancel_cmd = True

    # *** rate limit after the enable check ***
    self.brake_last = rate_limit(brake, self.brake_last, -2., 1./100)

    # vehicle hud display, wait for one update from 10Hz 0x304 msg
    if hud_show_lanes:
      hud_lanes = 1
    else:
      hud_lanes = 0

    # TODO: factor this out better
    if enabled:
      if hud_show_car:
        hud_car = 2
      else:
        hud_car = 1
    else:
      hud_car = 0
    
    # For lateral control-only, send chimes as a beep since we don't send 0x1fa
    #if CS.CP.radarOffCan:

    #print chime, alert_id, hud_alert
    fcw_display, steer_required, acc_alert = process_hud_alert(hud_alert)

    hud = HUDData(int(pcm_accel), int(round(hud_v_cruise)), 1, hud_car,
                  0xc1, hud_lanes, int(snd_beep), snd_chime, fcw_display, acc_alert, steer_required)

    if not all(isinstance(x, int) and 0 <= x < 256 for x in hud):
      print "INVALID HUD", hud
      hud = HUDData(0xc6, 255, 64, 0xc0, 209, 0x40, 0, 0, 0, 0)

    # **** process the car messages ****

    # *** compute control surfaces ***
    STEER_MAX = 0x4000 #16384

    # Angle Max. slope versus car speed
    # Graphical view: https://slack-files.com/T02Q83UUV-FBQFZR5PW-7962eb2adb
    # and https://slack-files.com/T02Q83UUV-FBQ6SPPRP-b110efb723
    # Model 1: USER_STEER_MAX = (-62.0 * CS.v_ego) + 2314.6
    # Model 2: USER_STEER_MAX  = 2.43 * CS.v_ego * CS.v_ego - 193.52 * CS.v_ego + 4000
    # Model 3: USER_STEER_MAX  = 1.485 * CS.v_ego * CS.v_ego - 154.51 * CS.v_ego + 4000
    USER_STEER_MAX  = 1.485 * CS.v_ego * CS.v_ego - 154.51 * CS.v_ego + 4000
    
    # Basic highway lane change logic
    changing_lanes = CS.right_blinker_on or CS.left_blinker_on
    
    enable_steer_control = (enabled and not changing_lanes)
        
    # Angle
    apply_steer = int(clip((-actuators.steerAngle * 10) + STEER_MAX, STEER_MAX - USER_STEER_MAX, STEER_MAX + USER_STEER_MAX)) # steer angle is converted back to CAN reference (positive when steering right)

    # Send CAN commands.
    can_sends = []
    send_step = 5

    if (frame % send_step) == 0:
      idx = (frame/send_step) % 16 
      can_sends.append(teslacan.create_steering_control(enable_steer_control, apply_steer, idx))
      can_sends.append(teslacan.create_epb_enable_signal(idx))
      
      if idx == 0:
        print "(Brakes: %s) (Gas: %s) (v_ego: %s) (v_cruise_pcm: %s) (v_cruise_actual: %s) (pcm_override: %s) (pcm_accel: %s)" % (
        str(brake),
        str(actuators.gas),
        str(CS.v_ego),
        str(CS.v_cruise_pcm),
        str(CS.v_cruise_actual),
        str(pcm_override),
        str(pcm_accel))
      
      # Adaptive cruise control
      if enable_steer_control and CS.pcm_acc_status == 2:
        cruise_msg = None
        # Reduce cruise speed if necessary.
        if accel > 0.5:
          # Send cruise stalk dn_2nd.
          cruise_msg = teslacan.create_cruise_adjust_msg(8, CS.steering_wheel_stalk)
        # Reduce speed more slightly if necessary.
        elif accel > 0.3:
          # Send cruise stalk dn_1st.
          cruise_msg = teslacan.create_cruise_adjust_msg(32, CS.steering_wheel_stalk)
        # Increase cruise speed if possible.
        elif (CS.v_ego > 18 * CV.MPH_TO_MS  # cruise only works >18mph.
              # only add cruise speed if real speed is near cruise speed.
              and CS.v_ego * CV.MS_TO_KPH >= CS.v_cruise_actual - 1
              # Check that the current cruise speed is below the allowed max.
              and CS.v_cruise_actual <= CS.v_cruise_pcm - 1):
          if actuators.gas > 0.6:
            # Send cruise stalk up_2nd
            cruise_msg = teslacan.create_cruise_adjust_msg(4, CS.steering_wheel_stalk)
          elif actuators.gas > 0.3:
            # Send cruise stalk up_1st
            cruise_msg = teslacan.create_cruise_adjust_msg(16, CS.steering_wheel_stalk)
        if cruise_msg:
          can_sends.append(cruise_msg)

      sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())