import os
import time
from cereal import car
from common.numpy_fast import interp
import selfdrive.messaging as messaging
from selfdrive.can.parser import CANParser
from selfdrive.config import Conversions as CV
import numpy as np

def parse_gear_shifter(can_gear_shifter, is_acura):

  if can_gear_shifter == 0x1:
    return "park"
  elif can_gear_shifter == 0x2:
    return "reverse"

  if is_acura:
    if can_gear_shifter == 0x3:
      return "neutral"
    elif can_gear_shifter == 0x4:
      return "drive"
    elif can_gear_shifter == 0xa:
      return "sport"

  else:
    if can_gear_shifter == 0x4:
      return "neutral"
    elif can_gear_shifter == 0x8:
      return "drive"
    elif can_gear_shifter == 0x10:
      return "sport"
    elif can_gear_shifter == 0x20:
      return "low"

  return "unknown"

_K0 = -0.3
_K1 = -0.01879
_K2 = 0.01013

def calc_cruise_offset(offset, speed):
  # euristic formula so that speed is controlled to ~ 0.3m/s below pid_speed
  # constraints to solve for _K0, _K1, _K2 are:
  # - speed = 0m/s, out = -0.3
  # - speed = 34m/s, offset = 20, out = -0.25
  # - speed = 34m/s, offset = -2.5, out = -1.8
  return min(_K0 + _K1 * speed + _K2 * speed * offset, 0.)

def get_can_signals(CP):
# this function generates lists for signal, messages and initial values
  if CP.carFingerprint == "HONDA CIVIC 2016 TOURING":
    dbc_f = 'honda_civic_touring_2016_can.dbc'
    signals = [
      # sig_name, sig_address, default
      ("XMISSION_SPEED", 0x158, 0),
      ("WHEEL_SPEED_FL", 0x1d0, 0),
      ("WHEEL_SPEED_FR", 0x1d0, 0),
      ("WHEEL_SPEED_RL", 0x1d0, 0),
      ("WHEEL_SPEED_RR", 0x1d0, 0),
      ("STEER_ANGLE", 0x14a, 0),
      ("STEER_ANGLE_RATE", 0x14a, 0),
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
      ("BRAKE_SWITCH", 0x17c, 0),
      ("CAR_GAS", 0x130, 0),
      ("CRUISE_BUTTONS", 0x296, 0),
      ("ESP_DISABLED", 0x1a4, 1),
      ("HUD_LEAD", 0x30c, 0),
      ("USER_BRAKE", 0x1a4, 0),
      ("STEER_STATUS", 0x18f, 5),
      ("BRAKE_ERROR_1", 0x1b0, 1),
      ("BRAKE_ERROR_2", 0x1b0, 1),
      ("GEAR_SHIFTER", 0x191, 0),
      ("MAIN_ON", 0x326, 0),
      ("ACC_STATUS", 0x17c, 0),
      ("PEDAL_GAS", 0x17c, 0),
      ("CRUISE_SETTING", 0x296, 0),
      ("LEFT_BLINKER", 0x326, 0),
      ("RIGHT_BLINKER", 0x326, 0),
      ("CRUISE_SPEED_OFFSET", 0x37c, 0),
      ("EPB_STATE", 0x1c2, 0),
      ("BRAKE_HOLD_ACTIVE", 0x1A4, 0),
    ]
    checks = [
      # address, frequency
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
      (0x37c, 10),
      (0x405, 3),
    ]

  elif CP.carFingerprint == "ACURA ILX 2016 ACURAWATCH PLUS":
    dbc_f = 'acura_ilx_2016_can.dbc'
    signals = [
      ("XMISSION_SPEED", 0x158, 0),
      ("WHEEL_SPEED_FL", 0x1d0, 0),
      ("WHEEL_SPEED_FR", 0x1d0, 0),
      ("WHEEL_SPEED_RL", 0x1d0, 0),
      ("WHEEL_SPEED_RR", 0x1d0, 0),
      ("STEER_ANGLE", 0x156, 0),
      ("STEER_ANGLE_RATE", 0x156, 0),
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
      ("BRAKE_SWITCH", 0x17c, 0),
      ("CAR_GAS", 0x130, 0),
      ("CRUISE_BUTTONS", 0x1a6, 0),
      ("ESP_DISABLED", 0x1a4, 1),
      ("HUD_LEAD", 0x30c, 0),
      ("USER_BRAKE", 0x1a4, 0),
      ("STEER_STATUS", 0x18f, 5),
      ("BRAKE_ERROR_1", 0x1b0, 1),
      ("BRAKE_ERROR_2", 0x1b0, 1),
      ("GEAR_SHIFTER", 0x1a3, 0),
      ("MAIN_ON", 0x1a6, 0),
      ("ACC_STATUS", 0x17c, 0),
      ("PEDAL_GAS", 0x17c, 0),
      ("CRUISE_SETTING", 0x1a6, 0),
      ("LEFT_BLINKER", 0x294, 0),
      ("RIGHT_BLINKER", 0x294, 0),
      ("CRUISE_SPEED_OFFSET", 0x37c, 0)
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
      (0x37c, 10),
      (0x405, 3),
    ]
  elif CP.carFingerprint == "HONDA ACCORD 2016 TOURING":
    dbc_f = 'honda_accord_touring_2016_can.dbc'
    signals = [
      ("XMISSION_SPEED", 0x158, 0),
      ("WHEEL_SPEED_FL", 0x1d0, 0),
      ("WHEEL_SPEED_FR", 0x1d0, 0),
      ("WHEEL_SPEED_RL", 0x1d0, 0),
      ("WHEEL_SPEED_RR", 0x1d0, 0),
      ("STEER_ANGLE", 0x156, 0),
      ("STEER_ANGLE_RATE", 0x156, 0),
      #("STEER_TORQUE_SENSOR", 0x18f, 0),
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
      ("BRAKE_SWITCH", 0x17c, 0),
      #("CAR_GAS", 0x130, 0),
      ("PEDAL_GAS", 0x17C, 0),
      ("CRUISE_BUTTONS", 0x1a6, 0),
      ("ESP_DISABLED", 0x1a4, 1),
      ("HUD_LEAD", 0x30c, 0),
      ("USER_BRAKE", 0x1a4, 0),
      #("STEER_STATUS", 0x18f, 5),
      ("BRAKE_ERROR_1", 0x1b0, 1),
      ("BRAKE_ERROR_2", 0x1b0, 1),
      ("GEAR_SHIFTER", 0x191, 0),
      ("MAIN_ON", 0x1a6, 0),
      ("ACC_STATUS", 0x17c, 0),
      ("PEDAL_GAS", 0x17c, 0),
      ("CRUISE_SETTING", 0x1a6, 0),
      ("LEFT_BLINKER", 0x294, 0),
      ("RIGHT_BLINKER", 0x294, 0),
    ]
    checks = [
      (0x156, 100),
      (0x158, 100),
      (0x17c, 100),
      (0x191, 100),
      (0x1a4, 50),
      (0x1a6, 50),
      (0x1b0, 50),
      (0x1d0, 50),
      (0x305, 10),
      (0x324, 10),
      (0x405, 3),
    ]
  elif CP.carFingerprint == "HONDA CR-V 2016 TOURING":
    dbc_f = 'honda_crv_touring_2016_can.dbc'
    signals = [
      ("XMISSION_SPEED", 0x158, 0),
      ("WHEEL_SPEED_FL", 0x1d0, 0),
      ("WHEEL_SPEED_FR", 0x1d0, 0),
      ("WHEEL_SPEED_RL", 0x1d0, 0),
      ("WHEEL_SPEED_RR", 0x1d0, 0),
      ("STEER_ANGLE", 0x156, 0),
      ("STEER_ANGLE_RATE", 0x156, 0),
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
      ("BRAKE_SWITCH", 0x17c, 0),
      #("CAR_GAS", 0x130, 0),
      ("CRUISE_BUTTONS", 0x1a6, 0),
      ("ESP_DISABLED", 0x1a4, 1),
      ("HUD_LEAD", 0x30c, 0),
      ("USER_BRAKE", 0x1a4, 0),
      ("STEER_STATUS", 0x18f, 5),
      ("BRAKE_ERROR_1", 0x1b0, 1),
      ("BRAKE_ERROR_2", 0x1b0, 1),
      ("GEAR_SHIFTER", 0x191, 0),
      ("MAIN_ON", 0x1a6, 0),
      ("ACC_STATUS", 0x17c, 0),
      ("PEDAL_GAS", 0x17c, 0),
      ("CRUISE_SETTING", 0x1a6, 0),
      ("LEFT_BLINKER", 0x294, 0),
      ("RIGHT_BLINKER", 0x294, 0),
    ]
    checks = [
      (0x156, 100),
      (0x158, 100),
      (0x17c, 100),
      (0x191, 100),
      (0x1a4, 50),
      (0x1a6, 50),
      (0x1b0, 50),
      (0x1d0, 50),
      (0x305, 10),
      (0x324, 10),
      (0x405, 3),
    ]
  # add gas interceptor reading if we are using it
  if CP.enableGas:
    signals.append(("INTERCEPTOR_GAS", 0x201, 0))
    checks.append((0x201, 50))

  return dbc_f, signals, checks

def get_can_parser(CP):
  dbc_f, signals, checks = get_can_signals(CP)
  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 0)

class CarState(object):
  def __init__(self, CP):
    self.acura = False
    self.civic = False
    self.accord = False
    self.crv = False
    if CP.carFingerprint == "HONDA CIVIC 2016 TOURING":
      self.civic = True
    elif CP.carFingerprint == "ACURA ILX 2016 ACURAWATCH PLUS":
      self.acura = True
    elif CP.carFingerprint == "HONDA ACCORD 2016 TOURING":
      self.accord = True
    elif CP.carFingerprint == "HONDA CR-V 2016 TOURING":
      self.crv = True
    else:
      raise ValueError("unsupported car %s" % CP.carFingerprint)

    self.brake_only = CP.enableCruise
    self.CP = CP

    self.user_gas, self.user_gas_pressed = 0., 0
    self.brake_switch_prev = 0
    self.brake_switch_ts = 0

    self.cruise_buttons = 0
    self.cruise_setting = 0
    self.blinker_on = 0

    self.left_blinker_on = 0
    self.right_blinker_on = 0

    # vEgo kalman filter
    dt = 0.01
    self.v_ego_x = np.matrix([[0.0], [0.0]])
    self.v_ego_A = np.matrix([[1.0, dt], [0.0, 1.0]])
    self.v_ego_C = np.matrix([1.0, 0.0])
    self.v_ego_Q = np.matrix([[10.0, 0.0], [0.0, 100.0]])
    self.v_ego_R = 1e3
    self.v_ego = 0.0
    # import control
    # (x, l, K) = control.dare(np.transpose(A), np.transpose(C), Q, R)
    # self.v_ego_K = np.transpose(K)
    self.v_ego_K = np.matrix([[0.12287673], [0.29666309]])

  def update(self, cp):

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
    if self.accord:
      self.steer_error = False
      self.steer_not_allowed = False
    else:
      self.steer_error = cp.vl[0x18F]['STEER_STATUS'] not in [0,2,4,6]
      self.steer_not_allowed = cp.vl[0x18F]['STEER_STATUS'] != 0
    self.brake_error = cp.vl[0x1B0]['BRAKE_ERROR_1'] or cp.vl[0x1B0]['BRAKE_ERROR_2']
    self.esp_disabled = cp.vl[0x1A4]['ESP_DISABLED']
    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = cp.vl[0x1D0]['WHEEL_SPEED_FL']
    self.v_wheel_fr = cp.vl[0x1D0]['WHEEL_SPEED_FR']
    self.v_wheel_rl = cp.vl[0x1D0]['WHEEL_SPEED_RL']
    self.v_wheel_rr = cp.vl[0x1D0]['WHEEL_SPEED_RR']
    self.v_wheel = (self.v_wheel_fl + self.v_wheel_fr + self.v_wheel_rl + self.v_wheel_rr) / 4.

    # blend in transmission speed at low speed, since it has more low speed accuracy
    self.v_weight = interp(self.v_wheel, v_weight_bp, v_weight_v)
    speed = (1. - self.v_weight) * cp.vl[0x158]['XMISSION_SPEED'] + self.v_weight * self.v_wheel

    if abs(speed - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_x = np.matrix([[speed], [0.0]])
    self.v_ego_x = np.dot((self.v_ego_A - np.dot(self.v_ego_K, self.v_ego_C)), self.v_ego_x) + np.dot(self.v_ego_K, speed)

    self.v_ego_raw = speed
    self.v_ego = float(self.v_ego_x[0])
    self.a_ego = float(self.v_ego_x[1])

    if self.CP.enableGas:
      # this is a hack
      self.user_gas = cp.vl[0x201]['INTERCEPTOR_GAS']
      self.user_gas_pressed = self.user_gas > 0 # this works because interceptor read < 0 when pedal position is 0. Once calibrated, this will change
      #print self.user_gas, self.user_gas_pressed
    if self.civic:
      can_gear_shifter = cp.vl[0x191]['GEAR_SHIFTER']
      self.angle_steers = cp.vl[0x14A]['STEER_ANGLE']
      self.angle_steers_rate = cp.vl[0x14A]['STEER_ANGLE_RATE']
      self.gear = 0  # TODO: civic has CVT... needs rev engineering
      self.cruise_setting = cp.vl[0x296]['CRUISE_SETTING']
      self.cruise_buttons = cp.vl[0x296]['CRUISE_BUTTONS']
      self.main_on = cp.vl[0x326]['MAIN_ON']
      self.blinker_on = cp.vl[0x326]['LEFT_BLINKER'] or cp.vl[0x326]['RIGHT_BLINKER']
      self.left_blinker_on = cp.vl[0x326]['LEFT_BLINKER']
      self.right_blinker_on = cp.vl[0x326]['RIGHT_BLINKER']
      self.cruise_speed_offset = calc_cruise_offset(cp.vl[0x37c]['CRUISE_SPEED_OFFSET'], self.v_ego)
      self.park_brake = cp.vl[0x1c2]['EPB_STATE'] != 0
      self.brake_hold = cp.vl[0x1A4]['BRAKE_HOLD_ACTIVE']
    elif self.accord:
      can_gear_shifter = cp.vl[0x191]['GEAR_SHIFTER']
      self.angle_steers = cp.vl[0x156]['STEER_ANGLE']
      self.angle_steers_rate = cp.vl[0x156]['STEER_ANGLE_RATE']
      self.gear = 0  # TODO: accord has CVT... needs rev engineering
      self.cruise_setting = cp.vl[0x1A6]['CRUISE_SETTING']
      self.cruise_buttons = cp.vl[0x1A6]['CRUISE_BUTTONS']
      self.main_on = cp.vl[0x1A6]['MAIN_ON']
      self.blinker_on = cp.vl[0x294]['LEFT_BLINKER'] or cp.vl[0x294]['RIGHT_BLINKER']
      self.left_blinker_on = cp.vl[0x294]['LEFT_BLINKER']
      self.right_blinker_on = cp.vl[0x294]['RIGHT_BLINKER']
      self.cruise_speed_offset = -0.3
      self.park_brake = 0  # TODO
      self.brake_hold = 0  # TODO
    elif self.crv:
      can_gear_shifter = cp.vl[0x191]['GEAR_SHIFTER']
      self.angle_steers = cp.vl[0x156]['STEER_ANGLE']
      self.angle_steers_rate = cp.vl[0x156]['STEER_ANGLE_RATE']
      self.gear = cp.vl[0x191]['GEAR']
      self.cruise_setting = cp.vl[0x1A6]['CRUISE_SETTING']
      self.cruise_buttons = cp.vl[0x1A6]['CRUISE_BUTTONS']
      self.main_on = cp.vl[0x1A6]['MAIN_ON']
      self.blinker_on = cp.vl[0x294]['LEFT_BLINKER'] or cp.vl[0x294]['RIGHT_BLINKER']
      self.left_blinker_on = cp.vl[0x294]['LEFT_BLINKER']
      self.right_blinker_on = cp.vl[0x294]['RIGHT_BLINKER']
      self.cruise_speed_offset = -0.3
      self.park_brake = 0  # TODO
      self.brake_hold = 0  # TODO
    elif self.acura:
      can_gear_shifter = cp.vl[0x1A3]['GEAR_SHIFTER']
      self.angle_steers = cp.vl[0x156]['STEER_ANGLE']
      self.angle_steers_rate = cp.vl[0x156]['STEER_ANGLE_RATE']
      self.gear = cp.vl[0x1A3]['GEAR']
      self.cruise_setting = cp.vl[0x1A6]['CRUISE_SETTING']
      self.cruise_buttons = cp.vl[0x1A6]['CRUISE_BUTTONS']
      self.main_on = cp.vl[0x1A6]['MAIN_ON']
      self.blinker_on = cp.vl[0x294]['LEFT_BLINKER'] or cp.vl[0x294]['RIGHT_BLINKER']
      self.left_blinker_on = cp.vl[0x294]['LEFT_BLINKER']
      self.right_blinker_on = cp.vl[0x294]['RIGHT_BLINKER']
      self.cruise_speed_offset = calc_cruise_offset(cp.vl[0x37c]['CRUISE_SPEED_OFFSET'], self.v_ego)
      self.park_brake = 0  # TODO
      self.brake_hold = 0

    self.gear_shifter = parse_gear_shifter(can_gear_shifter, self.acura)

    if self.accord:
      # on the accord, this doesn't seem to include cruise control
      self.car_gas = cp.vl[0x17C]['PEDAL_GAS']
      self.steer_override = False
    elif self.crv:
      # like accord, crv doesn't include cruise control
      self.car_gas = cp.vl[0x17C]['PEDAL_GAS']
      self.steer_override = abs(cp.vl[0x18F]['STEER_TORQUE_SENSOR']) > 1200
    else:
      self.car_gas = cp.vl[0x130]['CAR_GAS']
      self.steer_override = abs(cp.vl[0x18F]['STEER_TORQUE_SENSOR']) > 1200
    self.steer_torque_driver = cp.vl[0x18F]['STEER_TORQUE_SENSOR']

    # brake switch has shown some single time step noise, so only considered when
    # switch is on for at least 2 consecutive CAN samples
    self.brake_switch = cp.vl[0x17C]['BRAKE_SWITCH']
    self.brake_pressed = cp.vl[0x17C]['BRAKE_PRESSED'] or \
                         (self.brake_switch and self.brake_switch_prev and \
                         cp.ts[0x17C]['BRAKE_SWITCH'] != self.brake_switch_ts)
    self.brake_switch_prev = self.brake_switch
    self.brake_switch_ts = cp.ts[0x17C]['BRAKE_SWITCH']

    self.user_brake = cp.vl[0x1A4]['USER_BRAKE']
    self.standstill = not cp.vl[0x1B0]['WHEELS_MOVING']
    self.v_cruise_pcm = cp.vl[0x324]['CRUISE_SPEED_PCM']
    self.pcm_acc_status = cp.vl[0x17C]['ACC_STATUS']
    self.pedal_gas = cp.vl[0x17C]['PEDAL_GAS']
    self.hud_lead = cp.vl[0x30C]['HUD_LEAD']


# carstate standalone tester
if __name__ == '__main__':
  import zmq
  import time
  from selfdrive.services import service_list
  context = zmq.Context()

  class CarParams(object):
    def __init__(self):
      self.carFingerprint = "HONDA CIVIC 2016 TOURING"
      self.enableGas = 0
      self.enableCruise = 0
  CP = CarParams()
  CS = CarState(CP)

  while 1:
    CS.update()
    time.sleep(0.01)
