import os
import selfdrive.messaging as messaging
from selfdrive.car.toyota.values import CAR
from selfdrive.can.parser import CANParser
from selfdrive.config import Conversions as CV
import numpy as np

def parse_gear_shifter(can_gear, car_fingerprint):

  if car_fingerprint == CAR.PRIUS:
    if can_gear == 0x0:
      return "park"
    elif can_gear == 0x1:
      return "reverse"
    elif can_gear == 0x2:
      return "neutral"
    elif can_gear == 0x3:
      return "drive"
    elif can_gear == 0x4:
      return "brake"
  elif car_fingerprint in [CAR.RAV4, CAR.RAV4H]:
    if can_gear == 0x20:
      return "park"
    elif can_gear == 0x10:
      return "reverse"
    elif can_gear == 0x8:
      return "neutral"
    elif can_gear == 0x0:
      return "drive"
    elif can_gear == 0x1:
      return "sport"

  return "unknown"


def get_can_parser(CP):
  # this function generates lists for signal, messages and initial values
  if CP.carFingerprint == CAR.PRIUS:
    dbc_f = 'toyota_prius_2017_pt.dbc'
    signals = [
      ("GEAR", 295, 0),
      ("BRAKE_PRESSED", 550, 0),
      ("GAS_PEDAL", 581, 0),
    ]
    checks = [
      (550, 40),
      (581, 33)
    ]
  elif CP.carFingerprint == CAR.RAV4H:
    dbc_f = 'toyota_rav4_hybrid_2017_pt.dbc'
    signals = [
      ("GEAR", 956, 0),
      ("BRAKE_PRESSED", 550, 0),
      ("GAS_PEDAL", 581, 0),
    ]
    checks = [
      (550, 40),
      (581, 33)
    ]
  elif CP.carFingerprint == CAR.RAV4:
    dbc_f = 'toyota_rav4_2017_pt.dbc'
    signals = [
      ("GEAR", 956, 0x20),
      ("BRAKE_PRESSED", 548, 0),
      ("GAS_PEDAL", 705, 0),
    ]
    checks = [
      (548, 40),
      (705, 33)
    ]

  # TODO: DOORS, GAS_PEDAL, BRAKE_PRESSED for RAV4
  signals += [
    # sig_name, sig_address, default
    ("WHEEL_SPEED_FL", 170, 0),
    ("WHEEL_SPEED_FR", 170, 0),
    ("WHEEL_SPEED_RL", 170, 0),
    ("WHEEL_SPEED_RR", 170, 0),
    ("DOOR_OPEN_FL", 1568, 1),
    ("DOOR_OPEN_FR", 1568, 1),
    ("DOOR_OPEN_RL", 1568, 1),
    ("DOOR_OPEN_RR", 1568, 1),
    ("SEATBELT_DRIVER_UNLATCHED", 1568, 1),
    ("TC_DISABLED", 951, 1),
    ("STEER_ANGLE", 37, 0),
    ("STEER_FRACTION", 37, 0),
    ("STEER_RATE", 37, 0),
    ("GAS_RELEASED", 466, 0),
    ("CRUISE_STATE", 466, 0),
    ("MAIN_ON", 467, 0),
    ("SET_SPEED", 467, 0),
    ("LOW_SPEED_LOCKOUT", 467, 0),
    ("STEER_TORQUE_DRIVER", 608, 0),
    ("STEER_TORQUE_EPS", 608, 0),
    ("TURN_SIGNALS", 1556, 3),   # 3 is no blinkers
    ("LKA_STATE", 610, 0),
    ("BRAKE_LIGHTS_ACC", 951, 0),
  ]

  checks += [
    (170, 80),
    (37, 80),
    (466, 33),
    (467, 33),
    (608, 50),
    (610, 25),
  ]

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 0)


class CarState(object):
  def __init__(self, CP):

    self.CP = CP
    self.left_blinker_on = 0
    self.right_blinker_on = 0

    # initialize can parser
    self.car_fingerprint = CP.carFingerprint

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

    if self.car_fingerprint == CAR.PRIUS:
      can_gear = cp.vl[295]['GEAR']
      self.brake_pressed = cp.vl[550]['BRAKE_PRESSED']
      self.pedal_gas = cp.vl[581]['GAS_PEDAL']
    elif self.car_fingerprint == CAR.RAV4H:
      can_gear = cp.vl[956]['GEAR']
      self.brake_pressed = cp.vl[550]['BRAKE_PRESSED']
      self.pedal_gas = cp.vl[581]['GAS_PEDAL']
    elif self.car_fingerprint == CAR.RAV4:
      can_gear = cp.vl[956]['GEAR']
      self.brake_pressed = cp.vl[548]['BRAKE_PRESSED']
      self.pedal_gas = cp.vl[705]['GAS_PEDAL']

    # update prevs, update must run once per loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    # ******************* parse out can *******************
    self.door_all_closed = not any([cp.vl[1568]['DOOR_OPEN_FL'], cp.vl[1568]['DOOR_OPEN_FR'],
                                    cp.vl[1568]['DOOR_OPEN_RL'], cp.vl[1568]['DOOR_OPEN_RR']])
    self.seatbelt = not cp.vl[1568]['SEATBELT_DRIVER_UNLATCHED']
    # whitelist instead of blacklist, safer at the expense of disengages
    self.steer_error = False
    self.brake_error = 0
    self.esp_disabled = cp.vl[951]['TC_DISABLED']
    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = cp.vl[170]['WHEEL_SPEED_FL']
    self.v_wheel_fr = cp.vl[170]['WHEEL_SPEED_FR']
    self.v_wheel_rl = cp.vl[170]['WHEEL_SPEED_RL']
    self.v_wheel_rr = cp.vl[170]['WHEEL_SPEED_RR']
    self.v_wheel = (
      cp.vl[170]['WHEEL_SPEED_FL'] + cp.vl[170]['WHEEL_SPEED_FR'] +
      cp.vl[170]['WHEEL_SPEED_RL'] + cp.vl[170]['WHEEL_SPEED_RR']) / 4. * CV.KPH_TO_MS

    # Kalman filter
    if abs(self.v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_x = np.matrix([[self.v_wheel], [0.0]])
    self.v_ego_x = np.dot((self.v_ego_A - np.dot(self.v_ego_K, self.v_ego_C)), self.v_ego_x) + np.dot(self.v_ego_K, self.v_wheel)
    self.v_ego_raw = self.v_wheel
    self.v_ego = float(self.v_ego_x[0])
    self.a_ego = float(self.v_ego_x[1])
    self.standstill = not self.v_wheel > 0.001

    self.angle_steers = cp.vl[37]['STEER_ANGLE'] + cp.vl[37]['STEER_FRACTION']
    self.angle_steers_rate = cp.vl[37]['STEER_RATE']
    self.gear_shifter = parse_gear_shifter(can_gear, self.car_fingerprint)
    self.main_on = cp.vl[467]['MAIN_ON']
    self.left_blinker_on = cp.vl[1556]['TURN_SIGNALS'] == 1
    self.right_blinker_on = cp.vl[1556]['TURN_SIGNALS'] == 2

    # we could use the override bit from dbc, but it's triggered at too high torque values
    self.steer_override = abs(cp.vl[608]['STEER_TORQUE_DRIVER']) > 100
    self.steer_error = cp.vl[610]['LKA_STATE'] == 50
    self.steer_torque_driver = cp.vl[608]['STEER_TORQUE_DRIVER']
    self.steer_torque_motor = cp.vl[608]['STEER_TORQUE_EPS']

    self.user_brake = 0
    self.v_cruise_pcm = cp.vl[467]['SET_SPEED']
    self.pcm_acc_status = cp.vl[466]['CRUISE_STATE']
    self.car_gas = self.pedal_gas
    self.gas_pressed = not cp.vl[466]['GAS_RELEASED']
    self.low_speed_lockout = cp.vl[467]['LOW_SPEED_LOCKOUT'] == 2
    self.brake_lights = bool(cp.vl[951]['BRAKE_LIGHTS_ACC'] or self.brake_pressed)
