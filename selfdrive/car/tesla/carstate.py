from common.numpy_fast import interp
from common.kalman.simple_kalman import KF1D
from selfdrive.can.parser import CANParser
from selfdrive.config import Conversions as CV
from selfdrive.car.tesla.values import CAR, DBC
import numpy as np
from ctypes import create_string_buffer

def parse_gear_shifter(can_gear_shifter, car_fingerprint):

  # TODO: Use VAL from DBC to parse this field
  if car_fingerprint == CAR.MODELS:
    # VAL_ 280 DI_gear 4 "D" 0 "INVALID" 3 "N" 1 "P" 2 "R" 7 "SNA" ;
    if can_gear_shifter == 0x1:
      return "park"
    elif can_gear_shifter == 0x2:
      return "reverse"
    elif can_gear_shifter == 0x3:
      return "neutral"
    elif can_gear_shifter == 0x4:
      return "drive"

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
      ("MCU_gpsVehicleHeading", "MCU_gpsVehicleSpeed", 0),
      ("MCU_gpsAccuracy", "MCU_locationStatus", 0),
      ("MCU_latitude", "MCU_locationStatus", 0),
      ("MCU_longitude", "MCU_locationStatus", 0),
      ("DI_vehicleSpeed", "DI_torque2", 0),
      ("StW_AnglHP", "STW_ANGLHP_STAT", 0),
      ("EPAS_torsionBarTorque", "EPAS_sysStatus", 0), # Used in interface.py
      ("EPAS_eacStatus", "EPAS_sysStatus", 0),
      ("EPAS_handsOnLevel", "EPAS_sysStatus", 0),
      ("EPAS_steeringFault", "EPAS_sysStatus", 0),
      ("DI_gear", "DI_torque2", 3),
      ("DOOR_STATE_FL", "GTW_carState", 1),
      ("DOOR_STATE_FR", "GTW_carState", 1),
      ("DOOR_STATE_RL", "GTW_carState", 1),
      ("DOOR_STATE_RR", "GTW_carState", 1),
      ("YEAR", "GTW_carState", 1), # Added for debug
      ("MONTH", "GTW_carState", 1), # Added for debug
      ("DAY", "GTW_carState", 1), # Added for debug
      ("MINUTE", "GTW_carState", 1), # Added for debug
      ("HOUR", "GTW_carState", 1), # Added for debug
      ("SECOND", "GTW_carState", 1), # Added for debug
      # Added for debug ---below
      ("DI_torque2Counter", "DI_torque2", 0),
      ("DI_brakePedalState", "DI_torque2", 0),
      ("DI_epbParkRequest", "DI_torque2", 0),
      ("DI_epbInterfaceReady", "DI_torque2", 0),
      ("DI_torqueEstimate", "DI_torque2", 0),
      # Added for debug --above
      ("DI_stateCounter", "DI_state", 0),
      ("GTW_driverPresent", "GTW_status", 0),
      ("DI_brakePedal", "DI_torque2", 0),
      ("SpdCtrlLvr_Stat", "STW_ACTN_RQ", 0),
      ("DI_cruiseSet", "DI_state", 0),
      ("DI_cruiseState", "DI_state", 0),
      ("VSL_Enbl_Rq", "STW_ACTN_RQ", 0), # Used in interface.py
      ("TSL_RND_Posn_StW","SBW_RQ_SCCM" , 0),
      ("TSL_P_Psd_StW","SBW_RQ_SCCM" , 0),
      ("TurnIndLvr_Stat", "STW_ACTN_RQ", 0),
      ("DI_motorRPM", "DI_torque1", 0),
      ("DI_speedUnits", "DI_state", 0),
      
  ]

  checks = [
      ("STW_ANGLHP_STAT",  200), #JCT Actual message freq is 40 Hz (0.025 sec)
      ("STW_ACTN_RQ",  17), #JCT Actual message freq is 3.5 Hz (0.285 sec)
      ("SBW_RQ_SCCM", 175), #JCT Actual message freq is 35 Hz (0.0286 sec)
      ("DI_torque1", 59), #JCT Actual message freq is 11.8 Hz (0.084 sec)
      ("DI_torque2", 18), #JCT Actual message freq is 3.7 Hz (0.275 sec)
      ("MCU_gpsVehicleSpeed", 2), #JCT Actual message freq is 0.487 Hz (2.05 sec)
      ("GTW_carState", 16), #JCT Actual message freq is 3.3 Hz (0.3 sec)
      ("GTW_status", 2), #JCT Actual message freq is 0.5 Hz (2 sec)
      ("DI_state", 5), #JCT Actual message freq is 1 Hz (1 sec)
      ("EPAS_sysStatus", 0), #JCT Actual message freq is 1.3 Hz (0.76 sec)
      ("MCU_locationStatus", 5), #JCT Actual message freq is 1.3 Hz (0.76 sec)
  ]


  return signals, checks
  
def get_epas_can_signals(CP):
# this function generates lists for signal, messages and initial values
  signals = [
      ("EPAS_torsionBarTorque", "EPAS_sysStatus", 0), # Used in interface.py
      ("EPAS_eacStatus", "EPAS_sysStatus", 0),
      ("EPAS_eacErrorCode", "EPAS_sysStatus", 0),
      ("EPAS_handsOnLevel", "EPAS_sysStatus", 0),
      ("EPAS_steeringFault", "EPAS_sysStatus", 0),
  ]

  checks = [
      ("EPAS_sysStatus", 5), #JCT Actual message freq is 1.3 Hz (0.76 sec)
  ]


  return signals, checks
  
def get_can_parser(CP):
  signals, checks = get_can_signals(CP)
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

def get_epas_parser(CP):
  signals, checks = get_epas_can_signals(CP)
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)


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
    self.steer_warning = 0
    # TODO - Replace with a default value:
    self.last_cruise_message = create_string_buffer(8) 
    
    self.stopped = 0
    self.frame_humanSteered = 0    # Last frame human steered

    self.v_cruise_pcm = 20 * CV.MPH_TO_KPH
    self.v_cruise_car = 20 * CV.MPH_TO_KPH
    # vEgo kalman filter
    dt = 0.01
    # Q = np.matrix([[10.0, 0.0], [0.0, 100.0]])
    # R = 1e3
    self.v_ego_kf = KF1D(x0=np.matrix([[0.0], [0.0]]),
                         A=np.matrix([[1.0, dt], [0.0, 1.0]]),
                         C=np.matrix([1.0, 0.0]),
                         K=np.matrix([[0.12287673], [0.29666309]]))
    self.v_ego = 0.0

  def update(self, cp, epas_cp):

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
    self.door_all_closed = not any([cp.vl["GTW_carState"]['DOOR_STATE_FL'], cp.vl["GTW_carState"]['DOOR_STATE_FR'],
                               cp.vl["GTW_carState"]['DOOR_STATE_RL'], cp.vl["GTW_carState"]['DOOR_STATE_RR']])  #JCT
    self.seatbelt = cp.vl["GTW_status"]['GTW_driverPresent']

    # 2 = temporary 3= TBD 4 = temporary, hit a bump 5 (permanent) 6 = temporary 7 (permanent)
    # TODO: Use values from DBC to parse this field
    self.steer_error = epas_cp.vl["EPAS_sysStatus"]['EPAS_steeringFault'] == 1
    self.steer_not_allowed = epas_cp.vl["EPAS_sysStatus"]['EPAS_eacStatus'] not in [2,1] # 2 "EAC_ACTIVE" 1 "EAC_AVAILABLE" 3 "EAC_FAULT" 0 "EAC_INHIBITED"
    self.brake_error = 0 #NOT WORKINGcp.vl[309]['ESP_brakeLamp'] #JCT
    # JCT More ESP errors available, these needs to be added once car steers on its own to disable / alert driver
    self.esp_disabled = 0 #NEED TO CORRECT DBC cp.vl[309]['ESP_espOffLamp'] or cp.vl[309]['ESP_tcOffLamp'] or cp.vl[309]['ESP_tcLampFlash'] or cp.vl[309]['ESP_espFaultLamp'] #JCT

    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = 0 #JCT
    self.v_wheel_fr = 0 #JCT
    self.v_wheel_rl = 0 #JCT
    self.v_wheel_rr = 0 #JCT
    self.v_wheel = 0 #JCT
    self.v_weight = 0 #JCT
    speed = (cp.vl["DI_torque2"]['DI_vehicleSpeed'])*CV.MPH_TO_KPH/3.6 #JCT MPH_TO_MS. Tesla is in MPH, v_ego is expected in M/S
    speed = speed * 1.01 # To match car's displayed speed
    self.v_ego_x = np.matrix([[speed], [0.0]])
    self.v_ego_raw = speed
    v_ego_x = self.v_ego_kf.update(speed)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    
    # this is a hack for the interceptor. This is now only used in the simulation
    # TODO: Replace tests by toyota so this can go away
    self.user_gas = 0 #for now
    self.user_gas_pressed = self.user_gas > 0 # this works because interceptor read < 0 when pedal position is 0. Once calibrated, this will change

    can_gear_shifter = cp.vl["DI_torque2"]['DI_gear']
    self.gear = 0 # JCT
    self.angle_steers = -(cp.vl["STW_ANGLHP_STAT"]['StW_AnglHP']) #JCT polarity reversed from Honda/Acura
    self.angle_steers_rate = 0 #JCT

    self.cruise_setting = cp.vl["STW_ACTN_RQ"]['SpdCtrlLvr_Stat']
    self.cruise_buttons = cp.vl["STW_ACTN_RQ"]['SpdCtrlLvr_Stat']
    self.last_cruise_message = cp.vl["STW_ACTN_RQ"]
    self.blinker_on = (cp.vl["STW_ACTN_RQ"]['TurnIndLvr_Stat'] == 1) or (cp.vl["STW_ACTN_RQ"]['TurnIndLvr_Stat'] == 2)
    self.left_blinker_on = cp.vl["STW_ACTN_RQ"]['TurnIndLvr_Stat'] == 1
    self.right_blinker_on = cp.vl["STW_ACTN_RQ"]['TurnIndLvr_Stat'] == 2

    #if self.CP.carFingerprint in (CAR.CIVIC, CAR.ODYSSEY):
    #  self.park_brake = cp.vl["EPB_STATUS"]['EPB_STATE'] != 0
    #  self.brake_hold = cp.vl["VSA_STATUS"]['BRAKE_HOLD_ACTIVE']
    #  self.main_on = cp.vl["SCM_FEEDBACK"]['MAIN_ON']
    #else:
    self.park_brake = 0  # TODO
    self.brake_hold = 0  # TODO

    self.main_on = 1 #cp.vl["SCM_BUTTONS"]['MAIN_ON']

    self.gear_shifter = parse_gear_shifter(can_gear_shifter, self.CP.carFingerprint)

    self.pedal_gas = 0 #cp.vl["DI_torque1"]['DI_pedalPos']
    self.car_gas = self.pedal_gas

    self.steer_override = abs(epas_cp.vl["EPAS_sysStatus"]['EPAS_handsOnLevel']) > 0
    self.steer_torque_driver = 0 #JCT

    # brake switch has shown some single time step noise, so only considered when
    # switch is on for at least 2 consecutive CAN samples
    self.brake_switch = epas_cp.vl["EPAS_sysStatus"]['EPAS_eacErrorCode'] == 3 and epas_cp.vl["EPAS_sysStatus"]['EPAS_eacStatus'] == 0 #cp.vl["DI_torque2"]['DI_brakePedal']
    self.brake_pressed = epas_cp.vl["EPAS_sysStatus"]['EPAS_eacErrorCode'] == 3 and epas_cp.vl["EPAS_sysStatus"]['EPAS_eacStatus'] == 0  #cp.vl["DI_torque2"]['DI_brakePedal']

    self.user_brake = cp.vl["DI_torque2"]['DI_brakePedal']
    self.standstill = cp.vl["DI_torque2"]['DI_vehicleSpeed'] == 0
    if cp.vl["DI_state"]['DI_speedUnits'] == 0:
      self.v_cruise_car = (cp.vl["DI_state"]['DI_cruiseSet'])*CV.MPH_TO_KPH # Reported in MPH, expected in KPH??
    else:
      self.v_cruise_car = cp.vl["DI_state"]['DI_cruiseSet']
    
    self.pcm_acc_status = cp.vl["DI_state"]['DI_cruiseState']
    self.hud_lead = 0 #JCT
    self.cruise_speed_offset = calc_cruise_offset(self.v_cruise_pcm, self.v_ego)


# carstate standalone tester
if __name__ == '__main__':
  import zmq
  context = zmq.Context()

  class CarParams(object):
    def __init__(self):
      self.carFingerprint = "TESLA MODEL S"
      self.enableCruise = 0
  CP = CarParams()
  CS = CarState(CP)

  # while 1:
  #   CS.update()
  #   time.sleep(0.01)
