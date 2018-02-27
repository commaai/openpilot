import os
from common.numpy_fast import interp
from selfdrive.can.parser import CANParser
from selfdrive.config import Conversions as CV
from common.kalman.simple_kalman import KF1D
from common.fingerprints import HONDA as CAR
import numpy as np


def parse_gear_shifter(can_gear_shifter, car_fingerprint):
  # TODO: Use values from DBC to parse this field
  if can_gear_shifter == 0x1:
    return "park"
  elif can_gear_shifter == 0x2:
    return "reverse"

  if car_fingerprint in (CAR.ACURA_ILX, CAR.ODYSSEY):
    if can_gear_shifter == 0x3:
      return "neutral"
    elif can_gear_shifter == 0x4:
      return "drive"
    elif can_gear_shifter == 0xa:
      return "sport"

  elif car_fingerprint in (CAR.CIVIC, CAR.CRV, CAR.ACURA_RDX):
    if can_gear_shifter == 0x4:
      return "neutral"
    elif can_gear_shifter == 0x8:
      return "drive"
    elif can_gear_shifter == 0x10:
      return "sport"
    elif can_gear_shifter == 0x20:
      return "low"
    
  elif car_fingerprint in (CAR.PILOT):
     if can_gear_shifter == 0x8:
       return "reverse"
     elif can_gear_shifter == 0x4:
       return "park"
     elif can_gear_shifter == 0x20:
       return "drive"
     elif can_gear_shifter == 0x2:
        return "sport"

  return "unknown"


def calc_cruise_offset(offset, speed):
  # euristic formula so that speed is controlled to ~ 0.3m/s below pid_speed
  # constraints to solve for _K0, _K1, _K2 are:
  # - speed = 0m/s, out = -0.3
  # - speed = 34m/s, offset = 20, out = -0.25
  # - speed = 34m/s, offset = -2.5, out = -1.8
  _K0 = -0.3
  _K1 = -0.01879
  _K2 = 0.01013
  return min(_K0 + _K1 * speed + _K2 * speed * offset, 0.)


def get_can_signals(CP):
# this function generates lists for signal, messages and initial values
  signals = [
      ("XMISSION_SPEED", "ENGINE_DATA", 0),
      ("WHEEL_SPEED_FL", "WHEEL_SPEEDS", 0),
      ("WHEEL_SPEED_FR", "WHEEL_SPEEDS", 0),
      ("WHEEL_SPEED_RL", "WHEEL_SPEEDS", 0),
      ("WHEEL_SPEED_RR", "WHEEL_SPEEDS", 0),
      ("STEER_ANGLE", "STEERING_SENSORS", 0),
      ("STEER_ANGLE_RATE", "STEERING_SENSORS", 0),
      ("STEER_TORQUE_SENSOR", "STEER_STATUS", 0),
      ("DOOR_OPEN_FL", "DOORS_STATUS", 1),
      ("DOOR_OPEN_FR", "DOORS_STATUS", 1),
      ("DOOR_OPEN_RL", "DOORS_STATUS", 1),
      ("DOOR_OPEN_RR", "DOORS_STATUS", 1),
      ("LEFT_BLINKER", "SCM_FEEDBACK", 0),
      ("RIGHT_BLINKER", "SCM_FEEDBACK", 0),
      ("CRUISE_SPEED_OFFSET", "CRUISE_PARAMS", 0),
      ("GEAR", "GEARBOX", 0),
      ("WHEELS_MOVING", "STANDSTILL", 1),
      ("BRAKE_ERROR_1", "STANDSTILL", 1),
      ("BRAKE_ERROR_2", "STANDSTILL", 1),
      ("CRUISE_SPEED_PCM", "CRUISE", 0),
      ("SEATBELT_DRIVER_LAMP", "SEATBELT_STATUS", 1),
      ("SEATBELT_DRIVER_LATCHED", "SEATBELT_STATUS", 0),
      ("BRAKE_PRESSED", "POWERTRAIN_DATA", 0),
      ("BRAKE_SWITCH", "POWERTRAIN_DATA", 0),
      ("CRUISE_BUTTONS", "SCM_BUTTONS", 0),
      ("ESP_DISABLED", "VSA_STATUS", 1),
      ("HUD_LEAD", "ACC_HUD", 0),
      ("USER_BRAKE", "VSA_STATUS", 0),
      ("STEER_STATUS", "STEER_STATUS", 5),
      ("GEAR_SHIFTER", "GEARBOX", 0),
      ("PEDAL_GAS", "POWERTRAIN_DATA", 0),
      ("CRUISE_SETTING", "SCM_BUTTONS", 0),
      ("ACC_STATUS", "POWERTRAIN_DATA", 0),
  ]

  checks = [
      ("ENGINE_DATA", 100),
      ("WHEEL_SPEEDS", 50),
      ("STEERING_SENSORS", 100),
      ("DOORS_STATUS", 3),
      ("SCM_FEEDBACK", 10),
      ("CRUISE_PARAMS", 10),
      ("GEARBOX", 100),
      ("STANDSTILL", 50),
      ("SEATBELT_STATUS", 10),
      ("CRUISE", 10),
      ("POWERTRAIN_DATA", 100),
      ("VSA_STATUS", 50),
      ("SCM_BUTTONS", 25),
  ]

  if CP.carFingerprint == CAR.CIVIC:
    dbc_f = 'honda_civic_touring_2016_can_generated.dbc'
    signals += [("CAR_GAS", "GAS_PEDAL_2", 0),
                ("MAIN_ON", "SCM_FEEDBACK", 0),
                ("EPB_STATE", "EPB_STATUS", 0),
                ("BRAKE_HOLD_ACTIVE", "VSA_STATUS", 0)]
  elif CP.carFingerprint == CAR.ACURA_ILX:
    dbc_f = 'acura_ilx_2016_can_generated.dbc'
    signals += [("CAR_GAS", "GAS_PEDAL_2", 0),
                ("MAIN_ON", "SCM_BUTTONS", 0)]
  elif CP.carFingerprint == CAR.CRV:
    dbc_f = 'honda_crv_touring_2016_can_generated.dbc'
    signals += [("MAIN_ON", "SCM_BUTTONS", 0)]
  elif CP.carFingerprint == CAR.ACURA_RDX:
    dbc_f = 'acura_rdx_2018_can_generated.dbc'
    signals += [("MAIN_ON", "SCM_BUTTONS", 0)]
  elif CP.carFingerprint == CAR.ODYSSEY:
    dbc_f = 'honda_odyssey_exl_2018_generated.dbc'
    signals += [("MAIN_ON", "SCM_FEEDBACK", 0),
                ("EPB_STATE", "EPB_STATUS", 0),
                ("BRAKE_HOLD_ACTIVE", "VSA_STATUS", 0)]
    checks += [("EPB_STATUS", 50)]
  elif CP.carFingerprint == CAR.PILOT:
    dbc_f = 'honda_pilot_touring_2017_can_generated.dbc'
    signals += [("MAIN_ON", "SCM_BUTTONS", 0),
                ("CAR_GAS", "GAS_PEDAL_2", 0)]

  # add gas interceptor reading if we are using it
  if CP.enableGas:
    signals.append(("INTERCEPTOR_GAS", "GAS_SENSOR", 0))
    checks.append(("GAS_SENSOR", 50))

  return dbc_f, signals, checks


def get_can_parser(CP):
  dbc_f, signals, checks = get_can_signals(CP)
  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 0)


class CarState(object):
  def __init__(self, CP):
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

    # car params
    v_weight_v = [0., 1.]  # don't trust smooth speed at low values to avoid premature zero snapping
    v_weight_bp = [1., 6.]   # smooth blending, below ~0.6m/s the smooth speed snaps to zero

    # update prevs, update must run once per loop
    self.prev_cruise_buttons = self.cruise_buttons
    self.prev_cruise_setting = self.cruise_setting
    self.prev_blinker_on = self.blinker_on

    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    # ******************* parse out can *******************
    self.door_all_closed = not any([cp.vl["DOORS_STATUS"]['DOOR_OPEN_FL'], cp.vl["DOORS_STATUS"]['DOOR_OPEN_FR'],
                                    cp.vl["DOORS_STATUS"]['DOOR_OPEN_RL'], cp.vl["DOORS_STATUS"]['DOOR_OPEN_RR']])
    self.seatbelt = not cp.vl["SEATBELT_STATUS"]['SEATBELT_DRIVER_LAMP'] and cp.vl["SEATBELT_STATUS"]['SEATBELT_DRIVER_LATCHED']

    # 2 = temporary 3= TBD 4 = temporary, hit a bump 5 (permanent) 6 = temporary 7 (permanent)
    # TODO: Use values from DBC to parse this field
    self.steer_error = cp.vl["STEER_STATUS"]['STEER_STATUS'] not in [0, 2, 3, 4, 6]
    self.steer_not_allowed = cp.vl["STEER_STATUS"]['STEER_STATUS'] != 0
    self.brake_error = cp.vl["STANDSTILL"]['BRAKE_ERROR_1'] or cp.vl["STANDSTILL"]['BRAKE_ERROR_2']
    self.esp_disabled = cp.vl["VSA_STATUS"]['ESP_DISABLED']

    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_FL'] * CV.KPH_TO_MS
    self.v_wheel_fr = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_FR'] * CV.KPH_TO_MS
    self.v_wheel_rl = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_RL'] * CV.KPH_TO_MS
    self.v_wheel_rr = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_RR'] * CV.KPH_TO_MS
    self.v_wheel = (self.v_wheel_fl + self.v_wheel_fr + self.v_wheel_rl + self.v_wheel_rr) / 4.

    # blend in transmission speed at low speed, since it has more low speed accuracy
    self.v_weight = interp(self.v_wheel, v_weight_bp, v_weight_v)
    speed = (1. - self.v_weight) * cp.vl["ENGINE_DATA"]['XMISSION_SPEED'] + self.v_weight * self.v_wheel

    if abs(speed - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_x = np.matrix([[speed], [0.0]])

    self.v_ego_raw = speed
    v_ego_x = self.v_ego_kf.update(speed)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])

    # this is a hack for the interceptor. This is now only used in the simulation
    # TODO: Replace tests by toyota so this can go away
    if self.CP.enableGas:
      self.user_gas = cp.vl["GAS_SENSOR"]['INTERCEPTOR_GAS']
      self.user_gas_pressed = self.user_gas > 0 # this works because interceptor read < 0 when pedal position is 0. Once calibrated, this will change

    can_gear_shifter = cp.vl["GEARBOX"]['GEAR_SHIFTER']
    self.gear = 0 if self.CP.carFingerprint == CAR.CIVIC else cp.vl["GEARBOX"]['GEAR']
    self.angle_steers = cp.vl["STEERING_SENSORS"]['STEER_ANGLE']
    self.angle_steers_rate = cp.vl["STEERING_SENSORS"]['STEER_ANGLE_RATE']

    self.cruise_setting = cp.vl["SCM_BUTTONS"]['CRUISE_SETTING']
    self.cruise_buttons = cp.vl["SCM_BUTTONS"]['CRUISE_BUTTONS']

    self.blinker_on = cp.vl["SCM_FEEDBACK"]['LEFT_BLINKER'] or cp.vl["SCM_FEEDBACK"]['RIGHT_BLINKER']
    self.left_blinker_on = cp.vl["SCM_FEEDBACK"]['LEFT_BLINKER']
    self.right_blinker_on = cp.vl["SCM_FEEDBACK"]['RIGHT_BLINKER']

    if self.CP.carFingerprint in (CAR.CIVIC, CAR.ODYSSEY):
      self.park_brake = cp.vl["EPB_STATUS"]['EPB_STATE'] != 0
      self.brake_hold = cp.vl["VSA_STATUS"]['BRAKE_HOLD_ACTIVE']
      self.main_on = cp.vl["SCM_FEEDBACK"]['MAIN_ON']
    else:
      self.park_brake = 0  # TODO
      self.brake_hold = 0  # TODO

      self.main_on = cp.vl["SCM_BUTTONS"]['MAIN_ON']

    self.cruise_speed_offset = calc_cruise_offset(cp.vl["CRUISE_PARAMS"]['CRUISE_SPEED_OFFSET'], self.v_ego)
    self.gear_shifter = parse_gear_shifter(can_gear_shifter, self.CP.carFingerprint)

    self.pedal_gas = cp.vl["POWERTRAIN_DATA"]['PEDAL_GAS']
    # crv doesn't include cruise control
    if self.CP.carFingerprint in (CAR.CRV, CAR.ODYSSEY, CAR.ACURA_RDX):
      self.car_gas = self.pedal_gas
    else:
      self.car_gas = cp.vl["GAS_PEDAL_2"]['CAR_GAS']

    #rdx has different steer override threshold
    if self.CP.carFingerprint in (CAR.ACURA_RDX):
      self.steer_override = abs(cp.vl["STEER_STATUS"]['STEER_TORQUE_SENSOR']) > 400
    else:
      self.steer_override = abs(cp.vl["STEER_STATUS"]['STEER_TORQUE_SENSOR']) > 1200
    self.steer_torque_driver = cp.vl["STEER_STATUS"]['STEER_TORQUE_SENSOR']

    # brake switch has shown some single time step noise, so only considered when
    # switch is on for at least 2 consecutive CAN samples
    self.brake_switch = cp.vl["POWERTRAIN_DATA"]['BRAKE_SWITCH']
    self.brake_pressed = cp.vl["POWERTRAIN_DATA"]['BRAKE_PRESSED'] or \
                         (self.brake_switch and self.brake_switch_prev and \
                         cp.ts["POWERTRAIN_DATA"]['BRAKE_SWITCH'] != self.brake_switch_ts)
    self.brake_switch_prev = self.brake_switch
    self.brake_switch_ts = cp.ts["POWERTRAIN_DATA"]['BRAKE_SWITCH']

    self.user_brake = cp.vl["VSA_STATUS"]['USER_BRAKE']
    self.standstill = not cp.vl["STANDSTILL"]['WHEELS_MOVING']
    self.v_cruise_pcm = cp.vl["CRUISE"]['CRUISE_SPEED_PCM']
    self.pcm_acc_status = cp.vl["POWERTRAIN_DATA"]['ACC_STATUS']
    self.hud_lead = cp.vl["ACC_HUD"]['HUD_LEAD']


# carstate standalone tester
if __name__ == '__main__':
  import zmq
  import time
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
