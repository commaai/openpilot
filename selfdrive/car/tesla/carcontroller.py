import os
from collections import namedtuple
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit
from common.numpy_fast import clip
from selfdrive.car.tesla import teslacan
from selfdrive.car.tesla.values import AH, CruiseButtons, CAR
from selfdrive.can.packer import CANPacker


def actuator_hystereses(brake, braking, brake_steady, v_ego, car_fingerprint):
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
    self.human_steered_frame = 0
    self.enable_camera = enable_camera
    self.packer = CANPacker(dbc_name)

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

    # Prototype Angle Max. slope of 180 degree at 8.3 m/s (30 km/h) and 25 degree at 33.3 m/s (120 KM/h)
    # = -62x + 2314.6
    #   Angle max = -62*V(m/s) + 2314.6
    # Gives for example: 
    #                    180 degree at  30 km/h
    #                    145 degree at  50 km/h
    #                     94 degree at  80 km/h
    #                     59 degree at 100 km/h
    #                     42 degree at 110 km/h
    #                     25 degree at 120 km/h
    USER_STEER_MAX = (-62.0 * CS.v_ego) + 2314.6

    # Prevent steering while stopped
    MIN_STEERING_VEHICLE_VELOCITY = 0.05 # m/s
    vehicle_moving = (CS.v_ego >= MIN_STEERING_VEHICLE_VELOCITY)
    
    enable_steer_control = (enabled and not CS.steer_not_allowed) # See below for human override logic
    
    # Torque
    #steer_correction = actuators.steer if enable_steer_control else 0
    #apply_steer = int(clip((-steer_correction * 100) + STEER_MAX - (CS.angle_steers * 10), STEER_MAX - USER_STEER_MAX, STEER_MAX + USER_STEER_MAX)) # steer torque is converted back to CAN reference (positive when steering right)

    # Angle
    steer_correction = actuators.steerAngle if enable_steer_control  else CS.angle_steers
    apply_steer = int(clip((-actuators.steerAngle * 10) + STEER_MAX, STEER_MAX - (USER_STEER_MAX*2), STEER_MAX + (USER_STEER_MAX*2))) # steer torque is converted back to CAN reference (positive when steering right)

    # If we were previously disengaged via CS.steer_override>0 then we want [x] in a row where our planned steering is within 3 degrees of where we are steering
    if (CS.steer_override>0): 
      self.human_steered_frame = frame
      enable_steer_control = False
    else:
      if (frame - self.human_steered_frame < 50): # Need more human testing of handoff timing
        # Find steering difference between visiond model and human (no need to do every frame if we run out of CPU):
        steer_current=STEER_MAX-(CS.angle_steers*10)  # Formula to convert current steering angle to match apply_steer calculated number
        angle = abs(apply_steer-steer_current)

        # If OP steering > 5 degrees different from human than count that as human still steering..
        # Tesla rack doesn't report accurate enough, i.e. lane switch we show no human steering when they
        # still are crossing road at an angle clearly they don't want OP to take over
        if (angle > 50):
          self.human_steered_frame = frame
          enable_steer_control = False

    # Send CAN commands.
    can_sends = []
    send_step = 5

    if (frame % send_step) == 0:
      idx = (frame/send_step) % 16 
      can_sends.append(teslacan.create_steering_control(enable_steer_control, apply_steer, idx))
      can_sends.append(teslacan.create_epb_enable_signal(idx))

      sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
