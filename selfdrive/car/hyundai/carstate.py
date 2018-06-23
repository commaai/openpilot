from selfdrive.car.hyundai.values import CAR, DBC
from selfdrive.can.parser import CANParser
from selfdrive.config import Conversions as CV
from common.kalman.simple_kalman import KF1D
import numpy as np


def parse_gear_shifter(can_gear, car_fingerprint):
  # TODO: Use values from DBC to parse this field
  if car_fingerprint == CAR.ELANTRA:
    if can_gear_shifter == 0x0:
      return "park"
    elif can_gear_shifter == 0x5:
      return "drive"
    elif can_gear_shifter == 0x6:
      return "neutral"
    elif can_gear_shifter == 0x7:
      return "reverse"

  return "unknown"


def get_can_parser(CP):

  signals = [
    # sig_name, sig_address, default
    #("XMISSION_SPEED", 0x158, 0),
    ("WHL_SPD_FL", "WHL_SPD11", 0),
    ("WHL_SPD_FR", "WHL_SPD11", 0),
    ("WHL_SPD_RL", "WHL_SPD11", 0),
    ("WHL_SPD_RR", "WHL_SPD11", 0),
    ("SAS_Angle", "SAS11", 0),
    ("SAS_Speed", "SAS11", 0),
    ("CR_Lkas_StrToqReq", "LKAS11", 0),
    ("GEAR_TYPE", "TCU11", 0),
    #("WHEELS_MOVING", 0x1b0, 1),
    # ("C_DRVDoorStatus", 0x521, 1), # not on elantra
    # ("C_ASTDoorStatus", 0x521, 1), # not on elantra
    # ("C_RLDoorStatus", 0x521, 1), # not on elantra
    # ("C_RRDoorStatus", 0x521, 1), # not on elantra
    #("CRUISE_SPEED_PCM", 0x324, 0), # not on elantra
    ("CF_Gway_DrvSeatBeltInd", "CGW4", 1),
    ("CF_Gway_DrvSeatBeltSw", "CGW1", 0),
    ("BRAKE_ACT", "EMS12", 0),
    #("BRAKE_SWITCH", 0x17c, 0),
    #("CAR_GAS", 0x130, 0),
    ("CF_Clu_CruiseSwState", "CLU11", 0),
    #("ESP_DISABLED", 0x1a4, 1),
    #("HUD_LEAD", 0x30c, 0),
    ("CYL_PRES", "ESP12", 0),
    #("STEER_STATUS", 0x18f, 5),
    #("BRAKE_ERROR_1", 0x1b0, 1),
    #("BRAKE_ERROR_2", 0x1b0, 1),
    # ("CF_Lvr_GearInf", 0x354, 0), # not on elantra
    ("CF_Clu_CruiseSwMain", "CLU11", 0),
    ("ACCEnable", "TCS13", 0),
    ("PV_AV_CAN", "EMS12", 0),
    #("CRUISE_SETTING", 0x296, 0),
    ("CF_Gway_TurnSigLh", "CGW1", 0),
    ("CF_Gway_TurnSigRh", "CGW1", 0),
    #("CRUISE_SPEED_OFFSET", 0x37c, 0),
    # ("EPB_SWITCH", 0x490, 0),
    ("C_parkingBrakeSW", "GW_IPM_PE_1", 0),
  ]

  checks = [
    ("BRAKE_MODULE", 40), ## TODO: DO THESE
    ("GAS_PEDAL", 33),
    ("WHEEL_SPEEDS", 80),
    ("STEER_ANGLE_SENSOR", 80),
    ("PCM_CRUISE", 33),
    ("PCM_CRUISE_2", 33),
    ("STEER_TORQUE_SENSOR", 50),
    ("EPS_STATUS", 25),
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

  def update(self, cp):
    # copy can_valid
    self.can_valid = cp.can_valid

    # update prevs, update must run once per loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.door_all_closed = not any ([cp.vl["CGW1"]['CF_Gway_DrvDrSw'], cp.vl["CGW1"]['CF_Gway_AstDrSw'],
      cp.vl["CGW2"]['CF_Gway_RlDrSw'], cp.vl["CGW2"]['CF_Gway_RrDrSw']])
    self.seatbelt = ["CGW1"]['CF_Gway_DrvSeatBeltSw']

    can_gear = cp.vl["GEAR_PACKET"]['GEAR']
    self.brake_pressed = cp.vl["TCS13"]['DriverBraking']
    self.pedal_gas = cp.vl["GAS_PEDAL"]['GAS_PEDAL'] ## TODO: find this that is idle when acc accels
    self.car_gas = self.pedal_gas
    self.esp_disabled = cp.vl["TCS15"]['ESC_Off_Step']

    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = cp.vl["WHL_SPD11"]['WHL_SPD_FL'] * CV.KPH_TO_MS
    self.v_wheel_fr = cp.vl["WHL_SPD11"]['WHL_SPD_FR'] * CV.KPH_TO_MS
    self.v_wheel_rl = cp.vl["WHL_SPD11"]['WHL_SPD_RL'] * CV.KPH_TO_MS
    self.v_wheel_rr = cp.vl["WHL_SPD11"]['WHL_SPD_RR'] * CV.KPH_TO_MS
    self.v_wheel = (self.v_wheel_fl + self.v_wheel_fr + self.v_wheel_rl + self.v_wheel_rr) / 4.

    # Kalman filter
    if abs(self.v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_x = np.matrix([[self.v_wheel], [0.0]])

    self.v_ego_raw = self.v_wheel
    v_ego_x = self.v_ego_kf.update(self.v_wheel)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    self.standstill = not self.v_wheel > 0.001

    self.angle_steers = cp.vl["SAS11"]['SAS_Angle']
    self.angle_steers_rate = cp.vl["SAS11"]['SAS_Speed']
    self.gear_shifter = cp.vl["TCU11"]['GEAR_TYPE']
    self.main_on = cp.vl["CLU11"]['CF_Clu_CruiseSwMain']
    self.left_blinker_on = cp.vl["CGW1"]['CF_Gway_TurnSigLh']
    self.right_blinker_on = cp.vl["CGW1"]['CF_Gway_TurnSigRh']

    # we could use the override bit from dbc, but it's triggered at too high torque values
    self.steer_override = abs(cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_DRIVER']) > 100 ## TODO: FIND THIS
    # 2 is standby, 10 is active. TODO: check that everything else is really a faulty state
    self.steer_state = cp.vl["MDPS12"]['CF_Mdps_ToiActive'] #0 NOT ACTIVE, 1 ACTIVE
    self.steer_error = not cp.vl["MDPS12"]['CF_Mdps_FailStat'] or cp.vl["MDPS12"]['CF_Mdps_ToiUnavail'] ## TODO: VERIFY THIS
    # self.ipas_active = cp.vl['EPS_STATUS']['IPAS_STATE'] == 3
    self.brake_error = 0
    self.steer_torque_driver = cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_DRIVER'] ## TODO: FIND THIS
    self.steer_torque_motor = cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_EPS']

    self.user_brake = 0
    self.v_cruise_pcm = cp.vl["SCC11"]['VSetDis'] ## TODO: find the unit
    self.pcm_acc_status = cp.vl["PCM_CRUISE"]['CRUISE_STATE']
    self.gas_pressed = not cp.vl["PCM_CRUISE"]['GAS_RELEASED']
    self.low_speed_lockout = cp.vl["PCM_CRUISE_2"]['LOW_SPEED_LOCKOUT'] == 2
    self.brake_lights = bool(cp.vl["ESP_CONTROL"]['BRAKE_LIGHTS_ACC'] or self.brake_pressed)
