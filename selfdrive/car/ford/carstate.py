from cereal import car
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.config import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.ford.values import DBC

 #WHEEL_RADIUS = 0.33

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]['pt'])
    self.shifter_values = can_define.dv["TransGearData"]['GearLvrPos_D_Actl']
    
  def update(self, cp):
    ret = car.CarState.new_message()
    ret.wheelSpeeds.rr = cp.vl["WheelSpeed"]['WhlRr_W_Meas'] * CV.MPH_TO_MS
    ret.wheelSpeeds.rl = cp.vl["WheelSpeed"]['WhlRl_W_Meas'] * CV.MPH_TO_MS
    ret.wheelSpeeds.fr = cp.vl["WheelSpeed"]['WhlFr_W_Meas'] * CV.MPH_TO_MS
    ret.wheelSpeeds.fl = cp.vl["WheelSpeed"]['WhlFl_W_Meas'] * CV.MPH_TO_MS
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = not ret.vEgoRaw > 0.001
    ret.steeringAngle = cp.vl["Steering_Wheel_Data_CG1"]['SteWhlRelInit_An_Sns']
    ret.steeringPressed = not cp.vl["Lane_Keep_Assist_Status"]['LaHandsOff_B_Actl']
    ret.steerError = cp.vl["Lane_Keep_Assist_Status"]['LaActDeny_B_Actl'] == 1
    ret.cruiseState.speed = cp.vl["Cruise_Status"]['Set_Speed'] * CV.MPH_TO_MS
    ret.cruiseState.enabled = not (cp.vl["Cruise_Status"]['Cruise_State'] in [0, 3])
    ret.cruiseState.available = cp.vl["Cruise_Status"]['Cruise_State'] != 0
    ret.gas = cp.vl["EngineData_14"]['ApedPosScal_Pc_Actl'] / 100.
    ret.gasPressed = ret.gas > 1e-6
    ret.brakePressed = bool(cp.vl["Cruise_Status"]['Brake_Drv_Appl'])
    ret.brakeLights = bool(cp.vl["BCM_to_HS_Body"]['Brake_Lights'])
    ret.genericToggle = bool(cp.vl["Steering_Buttons"]['Dist_Incr'])
    self.latLimit = cp.vl["Lane_Keep_Assist_Status"]['LatCtlLim_D_Stat']
    self.lkas_state = cp.vl["Lane_Keep_Assist_Status"]['LaActAvail_D_Actl']
    self.left_blinker_on = bool(cp.vl["Steering_Buttons"]['Left_Turn_Light'])
    ret.leftBlinker = self.left_blinker_on > 0
    self.right_blinker_on = bool(cp.vl["Steering_Buttons"]['Right_Turn_Light'])    
    ret.rightBlinker = self.right_blinker_on > 0
    ret.doorOpen = any([cp.vl["Doors"]['Door_FL_Open'],cp.vl["Doors"]['Door_FR_Open'],
                        cp.vl["Doors"]['Door_RL_Open'], cp.vl["Doors"]['Door_RR_Open']]) 
    ret.steeringTorque = cp.vl["EPAS_INFO"]['DrvSte_Tq_Actl']

    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(cp.vl["TransGearData"]['GearLvrPos_D_Actl'], None))
    ret.seatbeltUnlatched = cp.vl["RCMStatusMessage2_FD1"]['FirstRowBuckleDriver'] == 2
    print ("Lateral_Limit:", self.latLimit, "lkas_state:", self.lkas_state, "steer_override:", ret.steeringPressed)
    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
    # sig_name, sig_address, default
    ("WhlRr_W_Meas", "WheelSpeed", 0.),
    ("WhlRl_W_Meas", "WheelSpeed", 0.),
    ("WhlFr_W_Meas", "WheelSpeed", 0.),
    ("WhlFl_W_Meas", "WheelSpeed", 0.),
    ("SteWhlRelInit_An_Sns", "Steering_Wheel_Data_CG1", 0.),
    ("Cruise_State", "Cruise_Status", 0.),
    ("Set_Speed", "Cruise_Status", 0.),
    ("LaActAvail_D_Actl", "Lane_Keep_Assist_Status", 0),
    ("LaHandsOff_B_Actl", "Lane_Keep_Assist_Status", 0),
    ("LaActDeny_B_Actl", "Lane_Keep_Assist_Status", 0),
    ("ApedPosScal_Pc_Actl", "EngineData_14", 0.),
    ("Dist_Incr", "Steering_Buttons", 0.),
    ("Lane_Keep_Toggle", "Steering_Buttons", 0.),
    #("Dist_Decr", "Steering_Buttons", 0.),
    #("Cancel", "Steering_Buttons", 0.),
    #("Resume", "Steering_Buttons", 0.),
    ("Brake_Drv_Appl", "Cruise_Status", 0.),
    ("Brake_Lights", "BCM_to_HS_Body", 0.),
    ("Left_Turn_Light", "Steering_Buttons", 0.),
    ("Right_Turn_Light", "Steering_Buttons", 0.),
    ("Door_FL_Open", "Doors", 0.),
    ("Door_FR_Open", "Doors", 0.),
    ("Door_RL_Open", "Doors", 0.),
    ("Door_RR_Open", "Doors", 0.),
    ("DrvSte_Tq_Actl", "EPAS_INFO", 0.),
    ("GearLvrPos_D_Actl", "TransGearData", 0.),
    ("FirstRowBuckleDriver", "RCMStatusMessage2_FD1", 0.),
    ("LatCtlLim_D_Stat", "Lane_Keep_Assist_Status", 0.),
  ]
    checks = []
    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)
