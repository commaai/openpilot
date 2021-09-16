from cereal import car
from collections import defaultdict
from common.numpy_fast import interp
from opendbc.can.can_define import CANDefine
from opendbc.can.parser import CANParser
from selfdrive.config import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.honda.values import CAR, DBC, STEER_THRESHOLD, SPEED_FACTOR, HONDA_BOSCH, HONDA_NIDEC_ALT_SCM_MESSAGES, HONDA_BOSCH_ALT_BRAKE_SIGNAL, HONDA_NO_WHEELS_MOVING_SIGNAL

TransmissionType = car.CarParams.TransmissionType


def get_can_signals(CP, gearbox_msg, main_on_sig_msg, door_sig_msg, door_signals):
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
    ("GEAR", gearbox_msg, 0),
    ("SEATBELT_DRIVER_LAMP", "SEATBELT_STATUS", 1),
    ("SEATBELT_DRIVER_LATCHED", "SEATBELT_STATUS", 0),
    ("BRAKE_PRESSED", "POWERTRAIN_DATA", 0),
    ("BRAKE_SWITCH", "POWERTRAIN_DATA", 0),
    ("CRUISE_BUTTONS", "SCM_BUTTONS", 0),
    ("ESP_DISABLED", "VSA_STATUS", 1),
    ("USER_BRAKE", "VSA_STATUS", 0),
    ("BRAKE_HOLD_ACTIVE", "VSA_STATUS", 0),
    ("STEER_STATUS", "STEER_STATUS", 5),
    ("GEAR_SHIFTER", gearbox_msg, 0),
    ("PEDAL_GAS", "POWERTRAIN_DATA", 0),
    ("CRUISE_SETTING", "SCM_BUTTONS", 0),
    ("ACC_STATUS", "POWERTRAIN_DATA", 0),
    ("MAIN_ON", main_on_sig_msg, 0),
  ]

  checks = [
    ("ENGINE_DATA", 100),
    ("WHEEL_SPEEDS", 50),
    ("STEERING_SENSORS", 100),
    ("SEATBELT_STATUS", 10),
    ("CRUISE", 10),
    ("POWERTRAIN_DATA", 100),
    ("VSA_STATUS", 50),
    ("STEER_STATUS", 100),
    ("STEER_MOTOR_TORQUE", 0), # TODO: not on every car
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

  if CP.carFingerprint in (CAR.CRV_HYBRID, CAR.CIVIC_BOSCH_DIESEL, CAR.ACURA_RDX_3G, CAR.HONDA_E):
    checks += [
      (gearbox_msg, 50),
    ]
  else:
    checks += [
      (gearbox_msg, 100),
    ]

  if CP.carFingerprint in HONDA_BOSCH_ALT_BRAKE_SIGNAL:
    signals += [("BRAKE_PRESSED", "BRAKE_MODULE", 0)]
    checks += [("BRAKE_MODULE", 50)]

  if CP.carFingerprint in HONDA_BOSCH:
    signals += [
      ("EPB_STATE", "EPB_STATUS", 0),
      ("IMPERIAL_UNIT", "CAR_SPEED", 1),
    ]
    checks += [
      ("EPB_STATUS", 50),
      ("CAR_SPEED", 10),
    ]

    if not CP.openpilotLongitudinalControl:
      signals += [
        ("CRUISE_CONTROL_LABEL", "ACC_HUD", 0),
        ("CRUISE_SPEED", "ACC_HUD", 0),
        ("ACCEL_COMMAND", "ACC_CONTROL", 0),
        ("AEB_STATUS", "ACC_CONTROL", 0),
      ]
      checks += [
        ("ACC_HUD", 10),
        ("ACC_CONTROL", 50),
      ]
  else:  # Nidec signals
    signals += [("CRUISE_SPEED_PCM", "CRUISE", 0),
                ("CRUISE_SPEED_OFFSET", "CRUISE_PARAMS", 0)]

    if CP.carFingerprint == CAR.ODYSSEY_CHN:
      checks += [("CRUISE_PARAMS", 10)]
    else:
      checks += [("CRUISE_PARAMS", 50)]

  signals.extend((signal, door_sig_msg, 1) for signal in door_signals)

  if CP.carFingerprint not in HONDA_NO_WHEELS_MOVING_SIGNAL:
    signals += [("WHEELS_MOVING", "STANDSTILL", 1)]
    if CP.carFingerprint != CAR.HRV:
      checks += [
        ("DOORS_STATUS", 3),
        ("STANDSTILL", 50),
      ]

  if CP.carFingerprint == CAR.CIVIC:
    signals += [("IMPERIAL_UNIT", "HUD_SETTING", 0),
                ("EPB_STATE", "EPB_STATUS", 0)]
    checks += [
      ("HUD_SETTING", 50),
      ("EPB_STATUS", 50),
    ]
  elif CP.carFingerprint in (CAR.ODYSSEY, CAR.ODYSSEY_CHN):
    signals += [("EPB_STATE", "EPB_STATUS", 0)]
    checks += [("EPB_STATUS", 50)]

  # add gas interceptor reading if we are using it
  if CP.enableGasInterceptor:
    signals.append(("INTERCEPTOR_GAS", "GAS_SENSOR", 0))
    signals.append(("INTERCEPTOR_GAS2", "GAS_SENSOR", 0))
    checks.append(("GAS_SENSOR", 50))

  if CP.openpilotLongitudinalControl:
    signals += [
      ("BRAKE_ERROR_1", "STANDSTILL", 1),
      ("BRAKE_ERROR_2", "STANDSTILL", 1)
    ]
    checks += [("STANDSTILL", 50)]

  return signals, checks


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])

    self.gearbox_msg = "GEARBOX"
    if CP.carFingerprint == CAR.ACCORD and CP.transmissionType == TransmissionType.cvt:
      self.gearbox_msg = "GEARBOX_15T"

    self.main_on_sig_msg = "SCM_FEEDBACK"
    if CP.carFingerprint in HONDA_NIDEC_ALT_SCM_MESSAGES:
      self.main_on_sig_msg = "SCM_BUTTONS"

    self.door_sig_msg = "DOORS_STATUS"
    self.door_signals = ("DOOR_OPEN_FL", "DOOR_OPEN_FR", "DOOR_OPEN_RL", "DOOR_OPEN_RR")
    if CP.carFingerprint in (CAR.ACCORD, CAR.ACCORD_2021, CAR.ACCORDH, CAR.ACCORDH_2021, CAR.ACURA_RDX_3G,
                             CAR.CIVIC_BOSCH, CAR.CIVIC_BOSCH_DIESEL, CAR.CRV_HYBRID, CAR.HONDA_E, CAR.INSIGHT):
      self.door_sig_msg = "SCM_FEEDBACK"
      self.door_signals = ("DRIVERS_DOOR_OPEN", )
    elif CP.carFingerprint in (CAR.FREED, CAR.HRV, CAR.ODYSSEY_CHN):
      self.door_sig_msg = "SCM_BUTTONS"
      self.door_signals = ("DRIVERS_DOOR_OPEN", )

    self.shifter_values = can_define.dv[self.gearbox_msg]["GEAR_SHIFTER"]
    self.steer_status_values = defaultdict(lambda: "UNKNOWN", can_define.dv["STEER_STATUS"]["STEER_STATUS"])

    self.brake_switch_prev = 0
    self.brake_switch_prev_ts = 0
    self.cruise_setting = 0
    self.v_cruise_pcm_prev = 0

  def update(self, cp, cp_cam, cp_body):
    ret = car.CarState.new_message()

    # car params
    v_weight_v = [0., 1.]  # don't trust smooth speed at low values to avoid premature zero snapping
    v_weight_bp = [1., 6.]   # smooth blending, below ~0.6m/s the smooth speed snaps to zero

    # update prevs, update must run once per loop
    self.prev_cruise_buttons = self.cruise_buttons
    self.prev_cruise_setting = self.cruise_setting

    # ******************* parse out can *******************
    if self.CP.carFingerprint in HONDA_NO_WHEELS_MOVING_SIGNAL:
      # TODO: find wheels moving bit in dbc
      ret.standstill = cp.vl["ENGINE_DATA"]["XMISSION_SPEED"] < 0.1
    else:
      ret.standstill = not cp.vl["STANDSTILL"]["WHEELS_MOVING"]

    ret.doorOpen = any(cp.vl[self.door_sig_msg][signal] for signal in self.door_signals)
    ret.seatbeltUnlatched = bool(cp.vl["SEATBELT_STATUS"]["SEATBELT_DRIVER_LAMP"] or not cp.vl["SEATBELT_STATUS"]["SEATBELT_DRIVER_LATCHED"])

    steer_status = self.steer_status_values[cp.vl["STEER_STATUS"]["STEER_STATUS"]]
    ret.steerError = steer_status not in ["NORMAL", "NO_TORQUE_ALERT_1", "NO_TORQUE_ALERT_2", "LOW_SPEED_LOCKOUT", "TMP_FAULT"]
    # NO_TORQUE_ALERT_2 can be caused by bump OR steering nudge from driver
    self.steer_not_allowed = steer_status not in ["NORMAL", "NO_TORQUE_ALERT_2"]
    # LOW_SPEED_LOCKOUT is not worth a warning
    ret.steerWarning = steer_status not in ["NORMAL", "LOW_SPEED_LOCKOUT", "NO_TORQUE_ALERT_2"]

    if not self.CP.openpilotLongitudinalControl:
      self.brake_error = 0
    else:
      self.brake_error = cp.vl["STANDSTILL"]["BRAKE_ERROR_1"] or cp.vl["STANDSTILL"]["BRAKE_ERROR_2"]
    ret.espDisabled = cp.vl["VSA_STATUS"]["ESP_DISABLED"] != 0

    speed_factor = SPEED_FACTOR.get(self.CP.carFingerprint, 1.)
    ret.wheelSpeeds.fl = cp.vl["WHEEL_SPEEDS"]["WHEEL_SPEED_FL"] * CV.KPH_TO_MS * speed_factor
    ret.wheelSpeeds.fr = cp.vl["WHEEL_SPEEDS"]["WHEEL_SPEED_FR"] * CV.KPH_TO_MS * speed_factor
    ret.wheelSpeeds.rl = cp.vl["WHEEL_SPEEDS"]["WHEEL_SPEED_RL"] * CV.KPH_TO_MS * speed_factor
    ret.wheelSpeeds.rr = cp.vl["WHEEL_SPEEDS"]["WHEEL_SPEED_RR"] * CV.KPH_TO_MS * speed_factor
    v_wheel = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr)/4.

    # blend in transmission speed at low speed, since it has more low speed accuracy
    v_weight = interp(v_wheel, v_weight_bp, v_weight_v)
    ret.vEgoRaw = (1. - v_weight) * cp.vl["ENGINE_DATA"]["XMISSION_SPEED"] * CV.KPH_TO_MS * speed_factor + v_weight * v_wheel
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    ret.steeringAngleDeg = cp.vl["STEERING_SENSORS"]["STEER_ANGLE"]
    ret.steeringRateDeg = cp.vl["STEERING_SENSORS"]["STEER_ANGLE_RATE"]

    self.cruise_setting = cp.vl["SCM_BUTTONS"]["CRUISE_SETTING"]
    self.cruise_buttons = cp.vl["SCM_BUTTONS"]["CRUISE_BUTTONS"]

    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_stalk(
      250, cp.vl["SCM_FEEDBACK"]["LEFT_BLINKER"], cp.vl["SCM_FEEDBACK"]["RIGHT_BLINKER"])
    ret.brakeHoldActive = cp.vl["VSA_STATUS"]["BRAKE_HOLD_ACTIVE"] == 1

    if self.CP.carFingerprint in (CAR.CIVIC, CAR.ODYSSEY, CAR.ODYSSEY_CHN, CAR.CRV_5G, CAR.ACCORD, CAR.ACCORD_2021, CAR.ACCORDH, CAR.ACCORDH_2021, CAR.CIVIC_BOSCH,
                                  CAR.CIVIC_BOSCH_DIESEL, CAR.CRV_HYBRID, CAR.INSIGHT, CAR.ACURA_RDX_3G, CAR.HONDA_E):
      self.park_brake = cp.vl["EPB_STATUS"]["EPB_STATE"] != 0
    else:
      self.park_brake = 0  # TODO

    gear = int(cp.vl[self.gearbox_msg]["GEAR_SHIFTER"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(gear, None))

    ret.gas = cp.vl["POWERTRAIN_DATA"]["PEDAL_GAS"]

    # this is a hack for the interceptor. This is now only used in the simulation
    # TODO: Replace tests by toyota so this can go away
    if self.CP.enableGasInterceptor:
      user_gas = (cp.vl["GAS_SENSOR"]["INTERCEPTOR_GAS"] + cp.vl["GAS_SENSOR"]["INTERCEPTOR_GAS2"]) / 2.
      ret.gasPressed = user_gas > 1e-5  # this works because interceptor reads < 0 when pedal position is 0. Once calibrated, this will change
    else:
      ret.gasPressed = ret.gas > 1e-5

    ret.steeringTorque = cp.vl["STEER_STATUS"]["STEER_TORQUE_SENSOR"]
    ret.steeringTorqueEps = cp.vl["STEER_MOTOR_TORQUE"]["MOTOR_TORQUE"]
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD.get(self.CP.carFingerprint, 1200)

    if self.CP.carFingerprint in HONDA_BOSCH:
      if not self.CP.openpilotLongitudinalControl:
        ret.cruiseState.nonAdaptive = cp.vl["ACC_HUD"]["CRUISE_CONTROL_LABEL"] != 0
        ret.cruiseState.standstill = cp.vl["ACC_HUD"]["CRUISE_SPEED"] == 252.

        # On set, cruise set speed pulses between 254~255 and the set speed prev is set to avoid this.
        ret.cruiseState.speed = self.v_cruise_pcm_prev if cp.vl["ACC_HUD"]["CRUISE_SPEED"] > 160.0 else cp.vl["ACC_HUD"]["CRUISE_SPEED"] * CV.KPH_TO_MS
        self.v_cruise_pcm_prev = ret.cruiseState.speed
    else:
      ret.cruiseState.speed = cp.vl["CRUISE"]["CRUISE_SPEED_PCM"] * CV.KPH_TO_MS

    self.brake_switch = cp.vl["POWERTRAIN_DATA"]["BRAKE_SWITCH"] != 0
    if self.CP.carFingerprint in HONDA_BOSCH_ALT_BRAKE_SIGNAL:
      ret.brakePressed = cp.vl["BRAKE_MODULE"]["BRAKE_PRESSED"] != 0
    else:
      # brake switch has shown some single time step noise, so only considered when
      # switch is on for at least 2 consecutive CAN samples
      # panda safety only checks BRAKE_PRESSED signal
      ret.brakePressed = bool(cp.vl["POWERTRAIN_DATA"]["BRAKE_PRESSED"] or
                              (self.brake_switch and self.brake_switch_prev and cp.ts["POWERTRAIN_DATA"]["BRAKE_SWITCH"] != self.brake_switch_prev_ts))

      self.brake_switch_prev = self.brake_switch
      self.brake_switch_prev_ts = cp.ts["POWERTRAIN_DATA"]["BRAKE_SWITCH"]

    ret.brake = cp.vl["VSA_STATUS"]["USER_BRAKE"]
    ret.cruiseState.enabled = cp.vl["POWERTRAIN_DATA"]["ACC_STATUS"] != 0
    ret.cruiseState.available = bool(cp.vl[self.main_on_sig_msg]["MAIN_ON"])

    # Gets rid of Pedal Grinding noise when brake is pressed at slow speeds for some models
    if self.CP.carFingerprint in (CAR.PILOT, CAR.PILOT_2019, CAR.PASSPORT, CAR.RIDGELINE):
      if ret.brake > 0.05:
        ret.brakePressed = True

    # TODO: discover the CAN msg that has the imperial unit bit for all other cars
    if self.CP.carFingerprint in (CAR.CIVIC, ):
      self.is_metric = not cp.vl["HUD_SETTING"]["IMPERIAL_UNIT"]
    elif self.CP.carFingerprint in HONDA_BOSCH:
      self.is_metric = not cp.vl["CAR_SPEED"]["IMPERIAL_UNIT"]
    else:
      self.is_metric = False

    if self.CP.carFingerprint in HONDA_BOSCH:
      ret.stockAeb = (not self.CP.openpilotLongitudinalControl) and bool(cp.vl["ACC_CONTROL"]["AEB_STATUS"] and cp.vl["ACC_CONTROL"]["ACCEL_COMMAND"] < -1e-5)
    else:
      ret.stockAeb = bool(cp_cam.vl["BRAKE_COMMAND"]["AEB_REQ_1"] and cp_cam.vl["BRAKE_COMMAND"]["COMPUTER_BRAKE"] > 1e-5)

    if self.CP.carFingerprint in HONDA_BOSCH:
      self.stock_hud = False
      ret.stockFcw = False
    else:
      ret.stockFcw = cp_cam.vl["BRAKE_COMMAND"]["FCW"] != 0
      self.stock_hud = cp_cam.vl["ACC_HUD"]
      self.stock_brake = cp_cam.vl["BRAKE_COMMAND"]

    if self.CP.enableBsm and self.CP.carFingerprint in (CAR.CRV_5G, ):
      # BSM messages are on B-CAN, requires a panda forwarding B-CAN messages to CAN 0
      # more info here: https://github.com/commaai/openpilot/pull/1867
      ret.leftBlindspot = cp_body.vl["BSM_STATUS_LEFT"]["BSM_ALERT"] == 1
      ret.rightBlindspot = cp_body.vl["BSM_STATUS_RIGHT"]["BSM_ALERT"] == 1

    return ret

  def get_can_parser(self, CP):
    signals, checks = get_can_signals(CP, self.gearbox_msg, self.main_on_sig_msg,
                                      self.door_sig_msg, self.door_signals)
    bus_pt = 1 if CP.carFingerprint in HONDA_BOSCH else 0
    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, bus_pt)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = []
    checks = [
      ("STEERING_CONTROL", 100),
    ]

    if CP.carFingerprint not in HONDA_BOSCH:
      signals += [("COMPUTER_BRAKE", "BRAKE_COMMAND", 0),
                  ("AEB_REQ_1", "BRAKE_COMMAND", 0),
                  ("FCW", "BRAKE_COMMAND", 0),
                  ("CHIME", "BRAKE_COMMAND", 0),
                  ("FCM_OFF", "ACC_HUD", 0),
                  ("FCM_OFF_2", "ACC_HUD", 0),
                  ("FCM_PROBLEM", "ACC_HUD", 0),
                  ("ICONS", "ACC_HUD", 0)]
      checks += [
        ("ACC_HUD", 10),
        ("BRAKE_COMMAND", 50),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)

  @staticmethod
  def get_body_can_parser(CP):
    if CP.enableBsm and CP.carFingerprint == CAR.CRV_5G:
      signals = [("BSM_ALERT", "BSM_STATUS_RIGHT", 0),
                 ("BSM_ALERT", "BSM_STATUS_LEFT", 0)]

      checks = [
        ("BSM_STATUS_LEFT", 3),
        ("BSM_STATUS_RIGHT", 3),
      ]
      bus_body = 0 # B-CAN is forwarded to ACC-CAN radar side (CAN 0 on fake ethernet port)
      return CANParser(DBC[CP.carFingerprint]["body"], signals, checks, bus_body)
    return None
