import logging
from selfdrive.can.parser import CANParser
from selfdrive.config import Conversions as CV
from selfdrive.car.chrysler.values import DBC
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
    ("WHEEL_SPEED_FL", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_RR", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_RL", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_FR", "WHEEL_SPEEDS", 0),
    ("STEER_ANGLE", "STEERING", 0),
    ("STEERING_RATE", "STEERING", 0),
    ("TURN_SIGNALS", "STEERING_LEVERS", 0),
    ("ACC_STATUS_2", "ACC_2", 0),
    ("HIGH_BEAM_FLASH", "STEERING_LEVERS", 0),
    ("ACC_SPEED_CONFIG_KPH", "DASHBOARD", 0),
    ("INCREMENTING_220", "LKAS_INDICATOR_1", -1),
  ]

  # It's considered invalid if it is not received for 10x the expected period (1/f).
  checks = [
    # sig_address, frequency
    # ("GEAR", 50),
    ("BRAKE_2", 50),
    ("LKAS_INDICATOR_1", 100),
    ("SPEED_1", 100),
    ("WHEEL_SPEEDS", 50),
    ("STEERING", 100),
    ("ACC_2", 50),
  ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)


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
    logging.basicConfig(level=logging.DEBUG, filename="/tmp/chrylog-cs", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('CarState init')


  def update(self, cp):
    # copy can_valid
    self.can_valid = cp.can_valid

    # update prevs, update must run once per loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.frame_220 = int(cp.vl["LKAS_INDICATOR_1"]['INCREMENTING_220'])
    logging.info('frame_220 %d' % self.frame_220)

    self.door_all_closed = not any([cp.vl["DOORS"]['DOOR_OPEN_FL'],
                                    cp.vl["DOORS"]['DOOR_OPEN_FR'],
                                    cp.vl["DOORS"]['DOOR_OPEN_RL'],
                                    cp.vl["DOORS"]['DOOR_OPEN_RR']])
    self.seatbelt = True  # TODO

    self.brake_pressed = cp.vl["BRAKE_2"]['BRAKE_PRESSED_2'] == 5 # human-only
    self.pedal_gas = 0  # TODO Disabled until we can find the message on Pacifica 2018
    # self.pedal_gas = cp.vl["ACCEL_PEDAL_MSG"]['ACCEL_PEDAL']
    self.car_gas = self.pedal_gas
    self.esp_disabled = False # cp.vl["ESP_CONTROL"]['TC_DISABLED']  # TODO

    self.v_wheel_fl = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_FL']
    self.v_wheel_rr = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RR']
    self.v_wheel_rl = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RL']
    self.v_wheel_fr = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_FR']
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
    self.steer_override = False  # abs(cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_DRIVER']) > 100  # TODO
    # # 2 is standby, 10 is active. TODO: check that everything else is really a faulty state
    # self.steer_state = cp.vl["EPS_STATUS"]['LKA_STATE']
    self.steer_error = cp.vl["LKAS_INDICATOR_1"]['LKAS_IS_GREEN'] == 0  # 0 if wheel will not actuate
    # self.ipas_active = cp.vl['EPS_STATUS']['IPAS_STATE'] == 3
    # self.brake_error = 0
    self.steer_torque_driver = 0 # cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_DRIVER'] # TODO
    # self.steer_torque_motor = cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_EPS']

    self.user_brake = 0  # TODO perhaps just use brake_pressed
    self.v_cruise_pcm = cp.vl["DASHBOARD"]['ACC_SPEED_CONFIG_KPH']
    self.pcm_acc_status = self.main_on # cp.vl["PCM_CRUISE"]['CRUISE_STATE']
    # self.gas_pressed = not cp.vl["PCM_CRUISE"]['GAS_RELEASED']
    self.brake_lights = self.brake_pressed # bool(cp.vl["ESP_CONTROL"]['BRAKE_LIGHTS_ACC'] or self.brake_pressed)  # TODO

    self.generic_toggle = bool(cp.vl["STEERING_LEVERS"]['HIGH_BEAM_FLASH'])
