import os
from selfdrive.can.parser import CANParser
from selfdrive.config import Conversions as CV
from common.kalman.simple_kalman import KF1D
import numpy as np


def parse_gear_shifter(can_gear):
  if can_gear == 0x1:
    return "park"
  elif can_gear == 0x2:
    return "reverse"
  elif can_gear == 0x3:
    return "neutral"
  elif can_gear == 0x4:
    return "drive"
  return "unknown"


def get_can_parser(CP):

  signals = [
    # sig_name, sig_address, default
    ("PRNDL", "GEAR", 0),
    ("DOOR_OPEN_FL", "DOORS", 0),
    ("DOOR_OPEN_FR", "DOORS", 0),
    ("DOOR_OPEN_RL", "DOORS", 0),
    ("DOOR_OPEN_RR", "DOORS", 0),
    ("BRAKE_PRESSED_2", "BRAKE_2", 0),
    ("ACCEL_PEDAL", "ACCEL_PEDAL_MSG", 0),
    ("SPEED_LEFT", "SPEED_1", 0),
    ("SPEED_RIGHT", "SPEED_1", 0),
    ("STEER_ANGLE", "STEERING", 0),
    ("STEERING_RATE", "STEERING", 0),
    ("TURN_SIGNALS", "STEERING_LEVERS", 0),
    ("ACC_STATUS_2", "ACC_2", 0),
    ("HIGH_BEAM_FLASH", "STEERING_LEVERS", 0),
  ]

  checks = [
    # TODO what are checks used for?
    # ("BRAKE_MODULE", 40),
    # ("GAS_PEDAL", 33),
    # ("WHEEL_SPEEDS", 80),
    # ("STEER_ANGLE_SENSOR", 80),
    # ("PCM_CRUISE", 33),
    # ("PCM_CRUISE_2", 33),
    # ("STEER_TORQUE_SENSOR", 50),
    # ("EPS_STATUS", 25),
  ]

  dbc_f = 'chrysler_pacifica_2017_hybrid.dbc'

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
    # Q = np.matrix([[10.0, 0.0], [0.0, 100.0]])
    # R = 1e3
    self.v_ego_kf = KF1D(x0=np.matrix([[0.0], [0.0]]),
                         A=np.matrix([[1.0, dt], [0.0, 1.0]]),
                         C=np.matrix([1.0, 0.0]),
                         K=np.matrix([[0.12287673], [0.29666309]]))
    self.v_ego = 0.0

  def update(self, cp):
    # copy can_valid
    self.can_valid = cp.can_valid

    # update prevs, update must run once per loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.door_all_closed = not any([cp.vl["DOORS"]['DOOR_OPEN_FL'],
                                    cp.vl["DOORS"]['DOOR_OPEN_FR'],
                                    cp.vl["DOORS"]['DOOR_OPEN_RL'],
                                    cp.vl["DOORS"]['DOOR_OPEN_RR']])
    self.seatbelt = True  # TODO

    self.brake_pressed = cp.vl["BRAKE_2"]['BRAKE_PRESSED_2'] == 5 # human-only
    self.pedal_gas = cp.vl["ACCEL_PEDAL_MSG"]['ACCEL_PEDAL']
    self.car_gas = self.pedal_gas
    # self.esp_disabled = cp.vl["ESP_CONTROL"]['TC_DISABLED']

    self.v_wheel = (cp.vl['SPEED_1']['SPEED_LEFT'] + cp.vl['SPEED_1']['SPEED_RIGHT']) / 2.

    # Kalman filter
    if abs(self.v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_x = np.matrix([[self.v_wheel], [0.0]])

    self.v_ego_raw = self.v_wheel
    v_ego_x = self.v_ego_kf.update(self.v_wheel)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    self.standstill = not self.v_wheel > 0.001

    self.angle_steers = cp.vl["STEERING"]['STEER_ANGLE']  # TODO verify units op wants.
    self.angle_steers_rate = cp.vl["STEERING"]['STEERING_RATE']  # TODO verify units op wants.
    self.gear_shifter = parse_gear_shifter(cp.vl['GEAR']['PRNDL'])
    self.main_on = cp.vl["ACC_2"]['ACC_STATUS_2'] == 7  # ACC is green.
    self.left_blinker_on = cp.vl["STEERING_LEVERS"]['TURN_SIGNALS'] == 1
    self.right_blinker_on = cp.vl["STEERING_LEVERS"]['TURN_SIGNALS'] == 2

    # TODO continue here with these values and see which ones we need.
    # # we could use the override bit from dbc, but it's triggered at too high torque values
    # self.steer_override = abs(cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_DRIVER']) > 100
    # # 2 is standby, 10 is active. TODO: check that everything else is really a faulty state
    # self.steer_state = cp.vl["EPS_STATUS"]['LKA_STATE']
    # self.steer_error = cp.vl["EPS_STATUS"]['LKA_STATE'] not in [1, 5]
    # self.ipas_active = cp.vl['EPS_STATUS']['IPAS_STATE'] == 3
    # self.brake_error = 0
    # self.steer_torque_driver = cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_DRIVER']
    # self.steer_torque_motor = cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_EPS']

    # self.user_brake = 0
    # self.v_cruise_pcm = cp.vl["PCM_CRUISE_2"]['SET_SPEED']
    # self.pcm_acc_status = cp.vl["PCM_CRUISE"]['CRUISE_STATE']
    # self.gas_pressed = not cp.vl["PCM_CRUISE"]['GAS_RELEASED']
    # self.low_speed_lockout = cp.vl["PCM_CRUISE_2"]['LOW_SPEED_LOCKOUT'] == 2
    # self.brake_lights = bool(cp.vl["ESP_CONTROL"]['BRAKE_LIGHTS_ACC'] or self.brake_pressed)

    self.generic_toggle = bool(cp.vl["STEERING_LEVERS"]['HIGH_BEAM_FLASH'])
