from cereal import car
from collections import defaultdict
from common.numpy_fast import interp
from common.kalman.simple_kalman import KF1D
from opendbc.can.can_define import CANDefine
from opendbc.can.parser import CANParser
from selfdrive.config import Conversions as CV
from selfdrive.car.honda.values import CAR, DBC, STEER_THRESHOLD, SPEED_FACTOR, HONDA_BOSCH

GearShifter = car.CarState.GearShifter

def parse_gear_shifter(gear):
  return {'P': GearShifter.park, 'R': GearShifter.reverse, 'N': GearShifter.neutral,
            'D': GearShifter.drive, 'S': GearShifter.sport, 'L': GearShifter.low}.get(gear, GearShifter.unknown)


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
      ("MOTOR_TORQUE", "STEER_MOTOR_TORQUE", 0),
      ("STEER_TORQUE_SENSOR", "STEER_STATUS", 0),
      ("LEFT_BLINKER", "SCM_FEEDBACK", 0),
      ("RIGHT_BLINKER", "SCM_FEEDBACK", 0),
      ("GEAR", "GEARBOX", 0),
      ("SEATBELT_DRIVER_LAMP", "SEATBELT_STATUS", 1),
      ("SEATBELT_DRIVER_LATCHED", "SEATBELT_STATUS", 0),
      ("BRAKE_PRESSED", "POWERTRAIN_DATA", 0),
      ("BRAKE_SWITCH", "POWERTRAIN_DATA", 0),
      ("CRUISE_BUTTONS", "SCM_BUTTONS", 0),
      ("ESP_DISABLED", "VSA_STATUS", 1),
      ("USER_BRAKE", "VSA_STATUS", 0),
      ("BRAKE_HOLD_ACTIVE", "VSA_STATUS", 0),
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
      ("SEATBELT_STATUS", 10),
      ("CRUISE", 10),
      ("POWERTRAIN_DATA", 100),
      ("VSA_STATUS", 50),
  ]

  if CP.carFingerprint == CAR.ODYSSEY_CHN:
    checks += [
      ("SCM_FEEDBACK", 25),
      ("SCM_BUTTONS", 50),
    ]
  else:
    checks += [
      ("SCM_FEEDBACK", 10),
      ("SCM_BUTTONS", 25),
    ]

  if CP.carFingerprint == CAR.CRV_HYBRID:
    checks += [
      ("GEARBOX", 50),
    ]
  else:
    checks += [
      ("GEARBOX", 100),
    ]

  if CP.radarOffCan:
    # Civic is only bosch to use the same brake message as other hondas.
    if CP.carFingerprint not in (CAR.ACCORDH, CAR.CIVIC_BOSCH, CAR.CRV_HYBRID):
      signals += [("BRAKE_PRESSED", "BRAKE_MODULE", 0)]
      checks += [("BRAKE_MODULE", 50)]
    signals += [("CAR_GAS", "GAS_PEDAL_2", 0),
                ("MAIN_ON", "SCM_FEEDBACK", 0),
                ("CRUISE_CONTROL_LABEL", "ACC_HUD", 0),
                ("EPB_STATE", "EPB_STATUS", 0),
                ("CRUISE_SPEED", "ACC_HUD", 0)]
    checks += [("GAS_PEDAL_2", 100)]
  else:
    # Nidec signals.
    signals += [("BRAKE_ERROR_1", "STANDSTILL", 1),
                ("BRAKE_ERROR_2", "STANDSTILL", 1),
                ("CRUISE_SPEED_PCM", "CRUISE", 0),
                ("CRUISE_SPEED_OFFSET", "CRUISE_PARAMS", 0)]
    checks += [("STANDSTILL", 50)]

    if CP.carFingerprint == CAR.ODYSSEY_CHN:
      checks += [("CRUISE_PARAMS", 10)]
    else:
      checks += [("CRUISE_PARAMS", 50)]

  if CP.carFingerprint in (CAR.ACCORD, CAR.ACCORD_15, CAR.ACCORDH, CAR.CIVIC_BOSCH, CAR.CRV_HYBRID):
    signals += [("DRIVERS_DOOR_OPEN", "SCM_FEEDBACK", 1)]
  elif CP.carFingerprint == CAR.ODYSSEY_CHN:
    signals += [("DRIVERS_DOOR_OPEN", "SCM_BUTTONS", 1)]
  else:
    signals += [("DOOR_OPEN_FL", "DOORS_STATUS", 1),
                ("DOOR_OPEN_FR", "DOORS_STATUS", 1),
                ("DOOR_OPEN_RL", "DOORS_STATUS", 1),
                ("DOOR_OPEN_RR", "DOORS_STATUS", 1),
                ("WHEELS_MOVING", "STANDSTILL", 1)]
    checks += [("DOORS_STATUS", 3)]

  if CP.carFingerprint == CAR.CIVIC:
    signals += [("CAR_GAS", "GAS_PEDAL_2", 0),
                ("MAIN_ON", "SCM_FEEDBACK", 0),
                ("IMPERIAL_UNIT", "HUD_SETTING", 0),
                ("EPB_STATE", "EPB_STATUS", 0)]
  elif CP.carFingerprint == CAR.ACURA_ILX:
    signals += [("CAR_GAS", "GAS_PEDAL_2", 0),
                ("MAIN_ON", "SCM_BUTTONS", 0)]
  elif CP.carFingerprint in (CAR.CRV, CAR.ACURA_RDX, CAR.PILOT_2019, CAR.RIDGELINE):
    signals += [("MAIN_ON", "SCM_BUTTONS", 0)]
  elif CP.carFingerprint == CAR.FIT:
    signals += [("CAR_GAS", "GAS_PEDAL_2", 0),
                ("MAIN_ON", "SCM_BUTTONS", 0),
                ("BRAKE_HOLD_ACTIVE", "VSA_STATUS", 0)]
  elif CP.carFingerprint == CAR.ODYSSEY:
    signals += [("MAIN_ON", "SCM_FEEDBACK", 0),
                ("EPB_STATE", "EPB_STATUS", 0)]
    checks += [("EPB_STATUS", 50)]
  elif CP.carFingerprint == CAR.PILOT:
    signals += [("MAIN_ON", "SCM_BUTTONS", 0),
                ("CAR_GAS", "GAS_PEDAL_2", 0)]
  elif CP.carFingerprint == CAR.ODYSSEY_CHN:
    signals += [("MAIN_ON", "SCM_BUTTONS", 0),
                ("EPB_STATE", "EPB_STATUS", 0)]
    checks += [("EPB_STATUS", 50)]

  # add gas interceptor reading if we are using it
  if CP.enableGasInterceptor:
    signals.append(("INTERCEPTOR_GAS", "GAS_SENSOR", 0))
    signals.append(("INTERCEPTOR_GAS2", "GAS_SENSOR", 0))
    checks.append(("GAS_SENSOR", 50))

  return signals, checks


def get_can_parser(CP):
  signals, checks = get_can_signals(CP)
  bus_pt = 1 if CP.isPandaBlack and CP.carFingerprint in HONDA_BOSCH else 0
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, bus_pt)


def get_cam_can_parser(CP):
  signals = []

  if CP.carFingerprint in HONDA_BOSCH:
    signals += [("ACCEL_COMMAND", "ACC_CONTROL", 0),
                ("AEB_STATUS", "ACC_CONTROL", 0)]
  else:
    signals += [("COMPUTER_BRAKE", "BRAKE_COMMAND", 0),
                ("AEB_REQ_1", "BRAKE_COMMAND", 0),
                ("FCW", "BRAKE_COMMAND", 0),
                ("CHIME", "BRAKE_COMMAND", 0),
                ("FCM_OFF", "ACC_HUD", 0),
                ("FCM_OFF_2", "ACC_HUD", 0),
                ("FCM_PROBLEM", "ACC_HUD", 0),
                ("ICONS", "ACC_HUD", 0)]


  # all hondas except CRV, RDX and 2019 Odyssey@China use 0xe4 for steering
  checks = [(0xe4, 100)]
  if CP.carFingerprint in [CAR.CRV, CAR.ACURA_RDX, CAR.ODYSSEY_CHN]:
    checks = [(0x194, 100)]

  bus_cam = 1 if CP.carFingerprint in HONDA_BOSCH  and not CP.isPandaBlack else 2
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, bus_cam)

class CarState():
  def __init__(self, CP):
    self.CP = CP
    self.can_define = CANDefine(DBC[CP.carFingerprint]['pt'])
    self.shifter_values = self.can_define.dv["GEARBOX"]["GEAR_SHIFTER"]
    self.steer_status_values = defaultdict(lambda: "UNKNOWN", self.can_define.dv["STEER_STATUS"]["STEER_STATUS"])

    self.user_gas, self.user_gas_pressed = 0., 0
    self.brake_switch_prev = 0
    self.brake_switch_ts = 0

    self.cruise_buttons = 0
    self.cruise_setting = 0
    self.v_cruise_pcm_prev = 0
    self.blinker_on = 0

    self.left_blinker_on = 0
    self.right_blinker_on = 0

    self.cruise_mode = 0
    self.stopped = 0

    # vEgo kalman filter
    dt = 0.01
    # Q = np.matrix([[10.0, 0.0], [0.0, 100.0]])
    # R = 1e3
    self.v_ego_kf = KF1D(x0=[[0.0], [0.0]],
                         A=[[1.0, dt], [0.0, 1.0]],
                         C=[1.0, 0.0],
                         K=[[0.12287673], [0.29666309]])
    self.v_ego = 0.0

  def update(self, cp, cp_cam):

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
    if self.CP.carFingerprint in (CAR.ACCORD, CAR.ACCORD_15, CAR.ACCORDH, CAR.CIVIC_BOSCH, CAR.CRV_HYBRID): # TODO: find wheels moving bit in dbc
      self.standstill = cp.vl["ENGINE_DATA"]['XMISSION_SPEED'] < 0.1
      self.door_all_closed = not cp.vl["SCM_FEEDBACK"]['DRIVERS_DOOR_OPEN']
    elif self.CP.carFingerprint == CAR.ODYSSEY_CHN:
      self.standstill = cp.vl["ENGINE_DATA"]['XMISSION_SPEED'] < 0.1
      self.door_all_closed = not cp.vl["SCM_BUTTONS"]['DRIVERS_DOOR_OPEN']
    else:
      self.standstill = not cp.vl["STANDSTILL"]['WHEELS_MOVING']
      self.door_all_closed = not any([cp.vl["DOORS_STATUS"]['DOOR_OPEN_FL'], cp.vl["DOORS_STATUS"]['DOOR_OPEN_FR'],
                                      cp.vl["DOORS_STATUS"]['DOOR_OPEN_RL'], cp.vl["DOORS_STATUS"]['DOOR_OPEN_RR']])
    self.seatbelt = not cp.vl["SEATBELT_STATUS"]['SEATBELT_DRIVER_LAMP'] and cp.vl["SEATBELT_STATUS"]['SEATBELT_DRIVER_LATCHED']

    steer_status = self.steer_status_values[cp.vl["STEER_STATUS"]['STEER_STATUS']]
    self.steer_error = steer_status not in ['NORMAL', 'NO_TORQUE_ALERT_1', 'NO_TORQUE_ALERT_2', 'LOW_SPEED_LOCKOUT', 'TMP_FAULT']
    # NO_TORQUE_ALERT_2 can be caused by bump OR steering nudge from driver
    self.steer_not_allowed = steer_status not in ['NORMAL', 'NO_TORQUE_ALERT_2']
    # LOW_SPEED_LOCKOUT is not worth a warning
    self.steer_warning = steer_status not in ['NORMAL', 'LOW_SPEED_LOCKOUT', 'NO_TORQUE_ALERT_2']

    if self.CP.radarOffCan:
      self.brake_error = 0
    else:
      self.brake_error = cp.vl["STANDSTILL"]['BRAKE_ERROR_1'] or cp.vl["STANDSTILL"]['BRAKE_ERROR_2']
    self.esp_disabled = cp.vl["VSA_STATUS"]['ESP_DISABLED']

    # calc best v_ego estimate, by averaging two opposite corners
    speed_factor = SPEED_FACTOR[self.CP.carFingerprint]
    self.v_wheel_fl = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_FL'] * CV.KPH_TO_MS * speed_factor
    self.v_wheel_fr = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_FR'] * CV.KPH_TO_MS * speed_factor
    self.v_wheel_rl = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_RL'] * CV.KPH_TO_MS * speed_factor
    self.v_wheel_rr = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_RR'] * CV.KPH_TO_MS * speed_factor
    v_wheel = (self.v_wheel_fl + self.v_wheel_fr + self.v_wheel_rl + self.v_wheel_rr)/4.

    # blend in transmission speed at low speed, since it has more low speed accuracy
    self.v_weight = interp(v_wheel, v_weight_bp, v_weight_v)
    speed = (1. - self.v_weight) * cp.vl["ENGINE_DATA"]['XMISSION_SPEED'] * CV.KPH_TO_MS * speed_factor + \
      self.v_weight * v_wheel

    if abs(speed - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_kf.x = [[speed], [0.0]]

    self.v_ego_raw = speed
    v_ego_x = self.v_ego_kf.update(speed)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])

    # this is a hack for the interceptor. This is now only used in the simulation
    # TODO: Replace tests by toyota so this can go away
    if self.CP.enableGasInterceptor:
      self.user_gas = (cp.vl["GAS_SENSOR"]['INTERCEPTOR_GAS'] + cp.vl["GAS_SENSOR"]['INTERCEPTOR_GAS2']) / 2.
      self.user_gas_pressed = self.user_gas > 0 # this works because interceptor read < 0 when pedal position is 0. Once calibrated, this will change

    self.gear = 0 if self.CP.carFingerprint == CAR.CIVIC else cp.vl["GEARBOX"]['GEAR']
    self.angle_steers = cp.vl["STEERING_SENSORS"]['STEER_ANGLE']
    self.angle_steers_rate = cp.vl["STEERING_SENSORS"]['STEER_ANGLE_RATE']

    self.cruise_setting = cp.vl["SCM_BUTTONS"]['CRUISE_SETTING']
    self.cruise_buttons = cp.vl["SCM_BUTTONS"]['CRUISE_BUTTONS']

    self.blinker_on = cp.vl["SCM_FEEDBACK"]['LEFT_BLINKER'] or cp.vl["SCM_FEEDBACK"]['RIGHT_BLINKER']
    self.left_blinker_on = cp.vl["SCM_FEEDBACK"]['LEFT_BLINKER']
    self.right_blinker_on = cp.vl["SCM_FEEDBACK"]['RIGHT_BLINKER']
    self.brake_hold = cp.vl["VSA_STATUS"]['BRAKE_HOLD_ACTIVE']

    if self.CP.carFingerprint in (CAR.CIVIC, CAR.ODYSSEY, CAR.CRV_5G, CAR.ACCORD, CAR.ACCORD_15, CAR.ACCORDH, CAR.CIVIC_BOSCH, CAR.CRV_HYBRID):
      self.park_brake = cp.vl["EPB_STATUS"]['EPB_STATE'] != 0
      self.main_on = cp.vl["SCM_FEEDBACK"]['MAIN_ON']
    elif self.CP.carFingerprint == CAR.ODYSSEY_CHN:
      self.park_brake = cp.vl["EPB_STATUS"]['EPB_STATE'] != 0
      self.main_on = cp.vl["SCM_BUTTONS"]['MAIN_ON']
    else:
      self.park_brake = 0  # TODO
      self.main_on = cp.vl["SCM_BUTTONS"]['MAIN_ON']

    can_gear_shifter = int(cp.vl["GEARBOX"]['GEAR_SHIFTER'])
    self.gear_shifter = parse_gear_shifter(self.shifter_values.get(can_gear_shifter, None))

    self.pedal_gas = cp.vl["POWERTRAIN_DATA"]['PEDAL_GAS']
    # crv doesn't include cruise control
    if self.CP.carFingerprint in (CAR.CRV, CAR.ODYSSEY, CAR.ACURA_RDX, CAR.RIDGELINE, CAR.PILOT_2019, CAR.ODYSSEY_CHN):
      self.car_gas = self.pedal_gas
    else:
      self.car_gas = cp.vl["GAS_PEDAL_2"]['CAR_GAS']

    self.steer_torque_driver = cp.vl["STEER_STATUS"]['STEER_TORQUE_SENSOR']
    self.steer_torque_motor = cp.vl["STEER_MOTOR_TORQUE"]['MOTOR_TORQUE']
    self.steer_override = abs(self.steer_torque_driver) > STEER_THRESHOLD[self.CP.carFingerprint]

    self.brake_switch = cp.vl["POWERTRAIN_DATA"]['BRAKE_SWITCH']

    if self.CP.radarOffCan:
      self.cruise_mode = cp.vl["ACC_HUD"]['CRUISE_CONTROL_LABEL']
      self.stopped = cp.vl["ACC_HUD"]['CRUISE_SPEED'] == 252.
      self.cruise_speed_offset = calc_cruise_offset(0, self.v_ego)
      if self.CP.carFingerprint in (CAR.CIVIC_BOSCH, CAR.ACCORDH, CAR.CRV_HYBRID):
        self.brake_switch = cp.vl["POWERTRAIN_DATA"]['BRAKE_SWITCH']
        self.brake_pressed = cp.vl["POWERTRAIN_DATA"]['BRAKE_PRESSED'] or \
                          (self.brake_switch and self.brake_switch_prev and \
                          cp.ts["POWERTRAIN_DATA"]['BRAKE_SWITCH'] != self.brake_switch_ts)
        self.brake_switch_prev = self.brake_switch
        self.brake_switch_ts = cp.ts["POWERTRAIN_DATA"]['BRAKE_SWITCH']
      else:
        self.brake_pressed = cp.vl["BRAKE_MODULE"]['BRAKE_PRESSED']
      # On set, cruise set speed pulses between 254~255 and the set speed prev is set to avoid this.
      self.v_cruise_pcm = self.v_cruise_pcm_prev if cp.vl["ACC_HUD"]['CRUISE_SPEED'] > 160.0 else cp.vl["ACC_HUD"]['CRUISE_SPEED']
      self.v_cruise_pcm_prev = self.v_cruise_pcm
    else:
      self.brake_switch = cp.vl["POWERTRAIN_DATA"]['BRAKE_SWITCH']
      self.cruise_speed_offset = calc_cruise_offset(cp.vl["CRUISE_PARAMS"]['CRUISE_SPEED_OFFSET'], self.v_ego)
      self.v_cruise_pcm = cp.vl["CRUISE"]['CRUISE_SPEED_PCM']
      # brake switch has shown some single time step noise, so only considered when
      # switch is on for at least 2 consecutive CAN samples
      self.brake_pressed = cp.vl["POWERTRAIN_DATA"]['BRAKE_PRESSED'] or \
                         (self.brake_switch and self.brake_switch_prev and \
                         cp.ts["POWERTRAIN_DATA"]['BRAKE_SWITCH'] != self.brake_switch_ts)
      self.brake_switch_prev = self.brake_switch
      self.brake_switch_ts = cp.ts["POWERTRAIN_DATA"]['BRAKE_SWITCH']

    self.user_brake = cp.vl["VSA_STATUS"]['USER_BRAKE']
    self.pcm_acc_status = cp.vl["POWERTRAIN_DATA"]['ACC_STATUS']

    # Gets rid of Pedal Grinding noise when brake is pressed at slow speeds for some models
    if self.CP.carFingerprint in (CAR.PILOT, CAR.PILOT_2019, CAR.RIDGELINE):
      if self.user_brake > 0.05:
        self.brake_pressed = 1

    # TODO: discover the CAN msg that has the imperial unit bit for all other cars
    self.is_metric = not cp.vl["HUD_SETTING"]['IMPERIAL_UNIT'] if self.CP.carFingerprint in (CAR.CIVIC) else False

    if self.CP.carFingerprint in HONDA_BOSCH:
      self.stock_aeb = bool(cp_cam.vl["ACC_CONTROL"]["AEB_STATUS"] and cp_cam.vl["ACC_CONTROL"]["ACCEL_COMMAND"] < -1e-5)
    else:
      self.stock_aeb = bool(cp_cam.vl["BRAKE_COMMAND"]["AEB_REQ_1"] and cp_cam.vl["BRAKE_COMMAND"]["COMPUTER_BRAKE"] > 1e-5)

    if self.CP.carFingerprint in HONDA_BOSCH:
      self.stock_hud = False
      self.stock_fcw = False
    else:
      self.stock_fcw = bool(cp_cam.vl["BRAKE_COMMAND"]["FCW"] != 0)
      self.stock_hud = cp_cam.vl["ACC_HUD"]
      self.stock_brake = cp_cam.vl["BRAKE_COMMAND"]
