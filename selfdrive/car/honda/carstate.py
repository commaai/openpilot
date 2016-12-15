import numpy as np

import selfdrive.messaging as messaging
from selfdrive.boardd.boardd import can_capnp_to_can_list
from selfdrive.config import VehicleParams
from common.realtime import sec_since_boot

from selfdrive.car.fingerprints import fingerprints
from selfdrive.car.honda.can_parser import CANParser

def get_can_parser(civic, brake_only):
  # this function generates lists for signal, messages and initial values
  if civic:
    dbc_f = 'honda_civic_touring_2016_can.dbc'
    signals = [
      ("XMISSION_SPEED", 0x158, 0),
      ("WHEEL_SPEED_FL", 0x1d0, 0),
      ("WHEEL_SPEED_FR", 0x1d0, 0),
      ("WHEEL_SPEED_RL", 0x1d0, 0),
      ("STEER_ANGLE", 0x14a, 0),
      ("STEER_TORQUE_SENSOR", 0x18f, 0),
      ("GEAR", 0x191, 0),
      ("WHEELS_MOVING", 0x1b0, 1),
      ("DOOR_OPEN_FL", 0x405, 1),
      ("DOOR_OPEN_FR", 0x405, 1),
      ("DOOR_OPEN_RL", 0x405, 1),
      ("DOOR_OPEN_RR", 0x405, 1),
      ("CRUISE_SPEED_PCM", 0x324, 0),
      ("SEATBELT_DRIVER_LAMP", 0x305, 1),
      ("SEATBELT_DRIVER_LATCHED", 0x305, 0),
      ("BRAKE_PRESSED", 0x17c, 0),
      ("CAR_GAS", 0x130, 0),
      ("CRUISE_BUTTONS", 0x296, 0),
      ("ESP_DISABLED", 0x1a4, 1),
      ("HUD_LEAD", 0x30c, 0),
      ("USER_BRAKE", 0x1a4, 0),
      ("STEER_STATUS", 0x18f, 5),
      ("WHEEL_SPEED_RR", 0x1d0, 0),
      ("BRAKE_ERROR_1", 0x1b0, 1),
      ("BRAKE_ERROR_2", 0x1b0, 1),
      ("GEAR_SHIFTER", 0x191, 0),
      ("MAIN_ON", 0x326, 0),
      ("ACC_STATUS", 0x17c, 0),
      ("PEDAL_GAS", 0x17c, 0),
      ("CRUISE_SETTING", 0x296, 0),
      ("LEFT_BLINKER", 0x326, 0),
      ("RIGHT_BLINKER", 0x326, 0),
      ("COUNTER", 0x324, 0),
      ("ENGINE_RPM", 0x17C, 0)
    ]
    checks = [
      (0x14a, 100),
      (0x158, 100),
      (0x17c, 100),
      (0x191, 100),
      (0x1a4, 50),
      (0x326, 10),
      (0x1b0, 50),
      (0x1d0, 50),
      (0x305, 10),
      (0x324, 10),
      (0x405, 3),
    ]

  else:
    dbc_f = 'acura_ilx_2016_can.dbc'
    signals = [
      ("XMISSION_SPEED", 0x158, 0),
      ("WHEEL_SPEED_FL", 0x1d0, 0),
      ("WHEEL_SPEED_FR", 0x1d0, 0),
      ("WHEEL_SPEED_RL", 0x1d0, 0),
      ("STEER_ANGLE", 0x156, 0),
      ("STEER_TORQUE_SENSOR", 0x18f, 0),
      ("GEAR", 0x1a3, 0),
      ("WHEELS_MOVING", 0x1b0, 1),
      ("DOOR_OPEN_FL", 0x405, 1),
      ("DOOR_OPEN_FR", 0x405, 1),
      ("DOOR_OPEN_RL", 0x405, 1),
      ("DOOR_OPEN_RR", 0x405, 1),
      ("CRUISE_SPEED_PCM", 0x324, 0),
      ("SEATBELT_DRIVER_LAMP", 0x305, 1),
      ("SEATBELT_DRIVER_LATCHED", 0x305, 0),
      ("BRAKE_PRESSED", 0x17c, 0),
      ("CAR_GAS", 0x130, 0),
      ("CRUISE_BUTTONS", 0x1a6, 0),
      ("ESP_DISABLED", 0x1a4, 1),
      ("HUD_LEAD", 0x30c, 0),
      ("USER_BRAKE", 0x1a4, 0),
      ("STEER_STATUS", 0x18f, 5),
      ("WHEEL_SPEED_RR", 0x1d0, 0),
      ("BRAKE_ERROR_1", 0x1b0, 1),
      ("BRAKE_ERROR_2", 0x1b0, 1),
      ("GEAR_SHIFTER", 0x1a3, 0),
      ("MAIN_ON", 0x1a6, 0),
      ("ACC_STATUS", 0x17c, 0),
      ("PEDAL_GAS", 0x17c, 0),
      ("CRUISE_SETTING", 0x1a6, 0),
      ("LEFT_BLINKER", 0x294, 0),
      ("RIGHT_BLINKER", 0x294, 0),
      ("COUNTER", 0x324, 0),
      ("ENGINE_RPM", 0x17C, 0)
    ]
    checks = [
      (0x156, 100),
      (0x158, 100),
      (0x17c, 100),
      (0x1a3, 50),
      (0x1a4, 50),
      (0x1a6, 50),
      (0x1b0, 50),
      (0x1d0, 50),
      (0x305, 10),
      (0x324, 10),
      (0x405, 3),
    ]

  # add gas interceptor reading if we are using it
  if not brake_only:
    signals.append(("INTERCEPTOR_GAS", 0x201, 0))
    checks.append((0x201, 50))

  return CANParser(dbc_f, signals, checks)

def fingerprint(logcan):
  print "waiting for fingerprint..."
  brake_only = True

  finger = {}
  st = None
  while 1:
    possible_cars = []
    for a in messaging.drain_sock(logcan, wait_for_one=True):
      if st is None:
        st = sec_since_boot()
      for adr, _, msg, idx in can_capnp_to_can_list(a.can):
        # pedal
        if adr == 0x201 and idx == 0:
          brake_only = False
        if idx == 0:
          finger[adr] = len(msg)

    # check for a single match
    for f in fingerprints:
      is_possible = True
      for adr in finger:
        # confirm all messages we have seen match
        if adr not in fingerprints[f] or fingerprints[f][adr] != finger[adr]:
          #print "mismatch", f, adr
          is_possible = False
          break
      if is_possible:
        possible_cars.append(f)

    # if we only have one car choice and it's been 100ms since we got our first message, exit
    if len(possible_cars) == 1 and st is not None and (sec_since_boot()-st) > 0.1:
      break
    elif len(possible_cars) == 0:
      print finger
      raise Exception("car doesn't match any fingerprints")

  print "fingerprinted", possible_cars[0]
  return brake_only, possible_cars[0]

class CarState(object):
  def __init__(self, logcan):
    self.torque_mod = False
    self.brake_only, self.car_type = fingerprint(logcan)

    # assuming if you have a pedal interceptor you also have a torque mod
    if not self.brake_only:
      self.torque_mod = True

    if self.car_type == "HONDA CIVIC 2016 TOURING":
      self.civic = True
    elif self.car_type == "ACURA ILX 2016 ACURAWATCH PLUS":
      self.civic = False
    else:
      raise ValueError("unsupported car %s" % self.car_type)

    # initialize can parser
    self.cp = get_can_parser(self.civic, self.brake_only)

    self.user_gas, self.user_gas_pressed = 0., 0

    self.cruise_buttons = 0
    self.cruise_setting = 0
    self.blinker_on = 0

    self.left_blinker_on = 0
    self.right_blinker_on = 0

    # TODO: actually make this work
    self.a_ego = 0.

    # speed in UI is shown as few % higher
    self.ui_speed_fudge = 1.01 if self.civic else 1.025

    # load vehicle params
    self.VP = VehicleParams(self.civic, self.brake_only, self.torque_mod)

  def update(self, can_pub_main):
    cp = self.cp
    cp.update_can(can_pub_main)

    # copy can_valid
    self.can_valid = cp.can_valid

    # car params
    v_weight_v  = [0., 1. ]  # don't trust smooth speed at low values to avoid premature zero snapping
    v_weight_bp = [1., 6.]   # smooth blending, below ~0.6m/s the smooth speed snaps to zero

    # update prevs, update must run once per loop
    self.prev_cruise_buttons = self.cruise_buttons
    self.prev_cruise_setting = self.cruise_setting
    self.prev_blinker_on = self.blinker_on

    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.rpm = cp.vl[0x17C]['ENGINE_RPM']

    # ******************* parse out can *******************
    self.door_all_closed = not any([cp.vl[0x405]['DOOR_OPEN_FL'], cp.vl[0x405]['DOOR_OPEN_FR'],
                               cp.vl[0x405]['DOOR_OPEN_RL'], cp.vl[0x405]['DOOR_OPEN_RR']])
    self.seatbelt = not cp.vl[0x305]['SEATBELT_DRIVER_LAMP'] and cp.vl[0x305]['SEATBELT_DRIVER_LATCHED']
    # error 2 = temporary
    # error 4 = temporary, hit a bump
    # error 5 (permanent)
    # error 6 = temporary
    # error 7 (permanent)
    #self.steer_error = cp.vl[0x18F]['STEER_STATUS'] in [5,7]
    # whitelist instead of blacklist, safer at the expense of disengages
    self.steer_error = cp.vl[0x18F]['STEER_STATUS'] not in [0,2,4,6]
    self.steer_not_allowed = cp.vl[0x18F]['STEER_STATUS'] != 0
    if cp.vl[0x18F]['STEER_STATUS'] != 0:
      print cp.vl[0x18F]['STEER_STATUS']
    self.brake_error = cp.vl[0x1B0]['BRAKE_ERROR_1'] or cp.vl[0x1B0]['BRAKE_ERROR_2']
    self.esp_disabled = cp.vl[0x1A4]['ESP_DISABLED']
    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel = (
      cp.vl[0x1D0]['WHEEL_SPEED_FL'] + cp.vl[0x1D0]['WHEEL_SPEED_FR'] +
      cp.vl[0x1D0]['WHEEL_SPEED_RL'] + cp.vl[0x1D0]['WHEEL_SPEED_RR']) / 4.
    # blend in transmission speed at low speed, since it has more low speed accuracy
    self.v_weight = np.interp(self.v_wheel, v_weight_bp, v_weight_v)
    self.v_ego = (1. - self.v_weight) * cp.vl[0x158]['XMISSION_SPEED'] + self.v_weight * self.v_wheel
    if not self.brake_only:
      self.user_gas = cp.vl[0x201]['INTERCEPTOR_GAS']
      self.user_gas_pressed = self.user_gas > 0 # this works because interceptor read < 0 when pedal position is 0. Once calibrated, this will change
      #print user_gas, user_gas_pressed
    if self.civic:
      self.gear_shifter = cp.vl[0x191]['GEAR_SHIFTER']
      self.angle_steers = cp.vl[0x14A]['STEER_ANGLE']
      self.gear = 0  # TODO: civic has CVT... needs rev engineering
      self.cruise_setting = cp.vl[0x296]['CRUISE_SETTING']
      self.cruise_buttons = cp.vl[0x296]['CRUISE_BUTTONS']
      self.main_on = cp.vl[0x326]['MAIN_ON']
      self.gear_shifter_valid = self.gear_shifter in [1,8]  # TODO: 1/P allowed for debug
      self.blinker_on = cp.vl[0x326]['LEFT_BLINKER'] or cp.vl[0x326]['RIGHT_BLINKER']
      self.left_blinker_on = cp.vl[0x326]['LEFT_BLINKER']
      self.right_blinker_on = cp.vl[0x326]['RIGHT_BLINKER']
    else:
      self.gear_shifter = cp.vl[0x1A3]['GEAR_SHIFTER']
      self.angle_steers = cp.vl[0x156]['STEER_ANGLE']
      self.gear = cp.vl[0x1A3]['GEAR']
      self.cruise_setting = cp.vl[0x1A6]['CRUISE_SETTING']
      self.cruise_buttons = cp.vl[0x1A6]['CRUISE_BUTTONS']
      self.main_on = cp.vl[0x1A6]['MAIN_ON']
      self.gear_shifter_valid = self.gear_shifter in [1,4]  # TODO: 1/P allowed for debug
      self.blinker_on = cp.vl[0x294]['LEFT_BLINKER'] or cp.vl[0x294]['RIGHT_BLINKER']
      self.left_blinker_on = cp.vl[0x294]['LEFT_BLINKER']
      self.right_blinker_on = cp.vl[0x294]['RIGHT_BLINKER']
    self.car_gas = cp.vl[0x130]['CAR_GAS']
    self.brake_pressed = cp.vl[0x17C]['BRAKE_PRESSED']
    self.user_brake = cp.vl[0x1A4]['USER_BRAKE']
    self.standstill = not cp.vl[0x1B0]['WHEELS_MOVING']
    self.steer_override = abs(cp.vl[0x18F]['STEER_TORQUE_SENSOR']) > 1200
    self.v_cruise_pcm = cp.vl[0x324]['CRUISE_SPEED_PCM']
    self.pcm_acc_status = cp.vl[0x17C]['ACC_STATUS']
    self.pedal_gas = cp.vl[0x17C]['PEDAL_GAS']
    self.hud_lead = cp.vl[0x30C]['HUD_LEAD']
    self.counter_pcm = cp.vl[0x324]['COUNTER']

