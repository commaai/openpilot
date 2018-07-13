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
  elif car_fingerprint == CAR.SORENTO:
    if can_gear == 0:
      return "park"
    elif can_gear == 14:
      return "reverse"
    else:
      return "drive"
  return "unknown"


def get_can_parser(CP):

  if CP.carFingerprint == CAR.ELANTRA:
      signals = [
        # sig_name, sig_address, default
        ("WHL_SPD_FL", "WHL_SPD11", 0),
        ("WHL_SPD_FR", "WHL_SPD11", 0),
        ("WHL_SPD_RL", "WHL_SPD11", 0),
        ("WHL_SPD_RR", "WHL_SPD11", 0),
        ("YAW_RATE", "ESP12", 0),
        ("CR_Lkas_StrToqReq", "LKAS11", 0),
        ("GEAR_TYPE", "TCU11", 0),
        ("CF_Gway_DrvSeatBeltInd", "CGW4", 1),
        ("CF_Gway_DrvSeatBeltSw", "CGW1", 0),
        ("BRAKE_ACT", "EMS12", 0),
        ("CF_Clu_CruiseSwState", "CLU11", 0),
        ("CYL_PRES", "ESP12", 0),
        ("CF_Clu_CruiseSwMain", "CLU11", 0),
        ("ACCEnable", "TCS13", 0),
        ("PV_AV_CAN", "EMS12", 0),
        ("CF_Gway_TurnSigLh", "CGW1", 0),
        ("CF_Gway_TurnSigRh", "CGW1", 0),
        ("C_parkingBrakeSW", "GW_IPM_PE_1", 0),
      ]

      checks = [
        ("LKAS11", 100),
        ("MDPS12", 50),
        ("TCS15", 10),
        ("TCS13", 50),
        ("CLU11", 50),
        ("ESP12", 100), 
        ("EMS12", 100),
        ("CGW1", 10),
        ("CGW4", 5),
        ("WHL_SPD11", 50),
      ]
  elif CP.carFingerprint == CAR.SORENTO or CP.carFingerprint == CAR.STINGER:
      signals = [
        # sig_name, sig_address, default
        ("WHL_SPD_FL", "WHL_SPD11", 0),
        ("WHL_SPD_FR", "WHL_SPD11", 0),
        ("WHL_SPD_RL", "WHL_SPD11", 0),
        ("WHL_SPD_RR", "WHL_SPD11", 0),

        ("YAW_RATE", "ESP12", 0),
        
        ("CR_Lkas_StrToqReq", "LKAS11", 0),

        ("CF_Gway_DrvSeatBeltInd", "CGW4", 1),

        ("CF_Gway_DrvSeatBeltSw", "CGW1", 0),
        ("CF_Gway_TurnSigLh", "CGW1", 0),
        ("CF_Gway_TurnSigRh", "CGW1", 0),

        ("BRAKE_ACT", "EMS12", 0),
        ("PV_AV_CAN", "EMS12", 0),
        ("TPS", "EMS12", 0),

        ("CYL_PRES", "ESP12", 0),

        ("CF_Clu_CruiseSwState", "CLU11", 0),
        ("CF_Clu_CruiseSwMain", "CLU11", 0),

        ("ACCEnable", "TCS13", 0),
        ("ACC_REQ", "TCS13", 0),
        ("DriverBraking", "TCS13", 0),
        ("DriverOverride", "TCS13", 0),

        ("ESC_Off_Step", "TCS15", 0),

        ("Gear", "AT01", 0),        #Transmission Gear (0 = N or P, 1-8 = Fwd, 14 = Rev)

        ("CR_Mdps_StrColTq", "MDPS12", 0),
        ("CF_Mdps_ToiActive", "MDPS12", 0),
        ("CF_Mdps_FailStat", "MDPS12", 0),
        ("CR_Mdps_OutTq", "MDPS12", 0),

      ]
      checks = [
        # address, frequency
        ## TODO - DO THIS PROPERLY
        ("LKAS11", 100),
        ("MDPS12", 50),
        ("TCS15", 10),
        ("TCS13", 50),
        ("CLU11", 50),
        ("ESP12", 100), 
        ("EMS12", 100),
        ("CGW1", 10),
        ("CGW4", 5),
        ("WHL_SPD11", 50),
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

    self.door_all_closed = True #not any ([cp.vl["CGW1"]['CF_Gway_DrvDrSw'], cp.vl["CGW1"]['CF_Gway_AstDrSw'], cp.vl["CGW2"]['CF_Gway_RlDrSw'], cp.vl["CGW2"]['CF_Gway_RrDrSw']])
    self.seatbelt = cp.vl["CGW1"]['CF_Gway_DrvSeatBeltSw']

    
    self.brake_pressed = cp.vl["TCS13"]['DriverBraking']
    self.esp_disabled = cp.vl["TCS15"]['ESC_Off_Step']

    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = cp.vl["WHL_SPD11"]['WHL_SPD_FL'] * CV.KPH_TO_MS
    self.v_wheel_fr = cp.vl["WHL_SPD11"]['WHL_SPD_FR'] * CV.KPH_TO_MS
    self.v_wheel_rl = cp.vl["WHL_SPD11"]['WHL_SPD_RL'] * CV.KPH_TO_MS
    self.v_wheel_rr = cp.vl["WHL_SPD11"]['WHL_SPD_RR'] * CV.KPH_TO_MS
    self.v_wheel = (self.v_wheel_fl + self.v_wheel_fr + self.v_wheel_rl + self.v_wheel_rr) / 4.
    if self.car_fingerprint == CAR.SORENTO:
      self.v_wheel = self.v_wheel * 1.02 # There is a 2 percent error on Sorento GT due to slightly larger wheel diameter compared to poverty packs.  Dash assumes about 4% which is excessive

    # Kalman filter
    if abs(self.v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_x = np.matrix([[self.v_wheel], [0.0]])

    self.v_ego_raw = self.v_wheel
    v_ego_x = self.v_ego_kf.update(self.v_wheel)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    self.standstill = not self.v_wheel > 0.001

    self.angle_steers = cp.vl["ESP12"]['YAW_RATE']
    self.main_on = cp.vl["CLU11"]['CF_Clu_CruiseSwMain']
    self.left_blinker_on = cp.vl["CGW1"]['CF_Gway_TurnSigLh']
    self.right_blinker_on = cp.vl["CGW1"]['CF_Gway_TurnSigRh']

    # we could use the override bit from dbc, but it's triggered at too high torque values
    self.steer_override = abs(cp.vl["MDPS12"]['CR_Mdps_StrColTq']) > 100  ## TODO: FIND THIS
    # 2 is standby, 10 is active. TODO: check that everything else is really a faulty state
    self.steer_state = cp.vl["MDPS12"]['CF_Mdps_ToiActive'] #0 NOT ACTIVE, 1 ACTIVE
    self.steer_error = not cp.vl["MDPS12"]['CF_Mdps_FailStat'] or cp.vl["MDPS12"]['CF_Mdps_ToiUnavail'] ## TODO: VERIFY THIS
    self.brake_error = 0
    self.steer_torque_driver = cp.vl["MDPS12"]['CR_Mdps_StrColTq'] ## TODO: FIND THIS
    self.steer_torque_motor = cp.vl["MDPS12"]['CR_Mdps_OutTq']

    self.user_brake = 0
    self.brake_lights = bool(self.brake_pressed)

    if self.car_fingerprint == CAR.ELANTRA:
        can_gear = cp.vl["GEAR_PACKET"]['GEAR']
        self.pedal_gas = cp.vl["GAS_PEDAL"]['GAS_PEDAL'] ## TODO: find this that is idle when acc accels
        self.car_gas = self.pedal_gas
        self.gear_shifter = cp.vl["TCU11"]['GEAR_TYPE']
        self.v_cruise_pcm = cp.vl["SCC11"]['VSetDis'] ## TODO: find the unit
        self.pcm_acc_status = cp.vl["PCM_CRUISE"]['CRUISE_STATE']
        self.gas_pressed = not cp.vl["PCM_CRUISE"]['GAS_RELEASED']
        self.low_speed_lockout = cp.vl["PCM_CRUISE_2"]['LOW_SPEED_LOCKOUT'] == 2
    elif self.car_fingerprint == CAR.SORENTO or self.car_fingerprint == CAR.STINGER:
        can_gear = cp.vl["AT01"]['Gear']
        self.brake_pressed = cp.vl["TCS13"]['DriverBraking']
        if (cp.vl["TCS13"]["DriverOverride"] == 0 and cp.vl["TCS13"]['ACC_REQ'] == 1):
          self.pedal_gas = 0
        else: 
          self.pedal_gas = cp.vl["EMS12"]['TPS']
        self.car_gas = cp.vl["EMS12"]['TPS']
        self.gear_shifter = parse_gear_shifter(can_gear, self.car_fingerprint)

