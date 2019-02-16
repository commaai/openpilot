import os
import subprocess
import sys
import numpy as np
from cereal import car
from common.kalman.simple_kalman import KF1D
from selfdrive.config import Conversions as CV
from selfdrive.can.parser import CANParser
from selfdrive.car.modules.UIBT_module import UIButtons,UIButton
from selfdrive.car.modules.UIEV_module import UIEvents
from selfdrive.car.gm.values import DBC, CAR, parse_gear_shifter, \
                                    CruiseButtons, is_eps_status_ok, \
                                    STEER_THRESHOLD

def get_powertrain_can_parser(CP, canbus):
  # this function generates lists for signal, messages and initial values
  signals = [
    # sig_name, sig_address, default
    ("BrakePedalPosition", "EBCMBrakePedalPosition", 0),
    ("FrontLeftDoor", "BCMDoorBeltStatus", 0),
    ("FrontRightDoor", "BCMDoorBeltStatus", 0),
    ("RearLeftDoor", "BCMDoorBeltStatus", 0),
    ("RearRightDoor", "BCMDoorBeltStatus", 0),
    ("LeftSeatBelt", "BCMDoorBeltStatus", 0),
    ("RightSeatBelt", "BCMDoorBeltStatus", 0),
    ("TurnSignals", "BCMTurnSignals", 0),
    ("AcceleratorPedal", "AcceleratorPedal", 0),
    ("ACCButtons", "ASCMSteeringButton", CruiseButtons.UNPRESS),
    ("LKAButton", "ASCMSteeringButton", 0),
    ("SteeringWheelAngle", "PSCMSteeringAngle", 0),
    ("FLWheelSpd", "EBCMWheelSpdFront", 0),
    ("FRWheelSpd", "EBCMWheelSpdFront", 0),
    ("RLWheelSpd", "EBCMWheelSpdRear", 0),
    ("RRWheelSpd", "EBCMWheelSpdRear", 0),
    ("PRNDL", "ECMPRDNL", 0),
    ("LKADriverAppldTrq", "PSCMStatus", 0),
    ("LKATorqueDeliveredStatus", "PSCMStatus", 0),
    ("DistanceButton", "ASCMSteeringButton", 0),
  ]

  if CP.carFingerprint == CAR.VOLT:
    signals += [
      ("RegenPaddle", "EBCMRegenPaddle", 0),
    ]
  if CP.carFingerprint in (CAR.VOLT, CAR.MALIBU, CAR.HOLDEN_ASTRA, CAR.ACADIA, CAR.CADILLAC_ATS):
    signals += [
      ("TractionControlOn", "ESPStatus", 0),
      ("EPBClosed", "EPBStatus", 0),
      ("CruiseMainOn", "ECMEngineStatus", 0),
      ("CruiseState", "AcceleratorPedal2", 0),
    ]
  if CP.carFingerprint == CAR.CADILLAC_CT6:
    signals += [
      ("ACCCmdActive", "ASCMActiveCruiseControlStatus", 0)
    ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, [], canbus.powertrain)

class CarState(object):
  def __init__(self, CP, canbus):
    self.CP = CP
    # initialize can parser
    self.alcaLabels = ["MadMax","Normal","Wifey"]
    self.alcaMode = 2
    self.visionLabels = ["normal","wiggly"]
    self.visionMode = 0
    self.car_fingerprint = CP.carFingerprint
    self.cruise_buttons = CruiseButtons.UNPRESS
    self.prev_distance_button = 0
    self.distance_button = 0
    self.left_blinker_on = False
    self.prev_left_blinker_on = False
    self.right_blinker_on = False
    self.prev_right_blinker_on = False
    self.follow_level = 3
    self.prev_lka_button = 0
    self.lka_button = 0
    self.lkMode = True
    
    # ALCA PARAMS
    self.blind_spot_on = bool(0)
    # max REAL delta angle for correction vs actuator
    self.CL_MAX_ANGLE_DELTA_BP = [10., 32., 44.]
    self.CL_MAX_ANGLE_DELTA = [2.0, 1., 0.5]
    # adjustment factor for merging steer angle to actuator; should be over 4; the higher the smoother
    self.CL_ADJUST_FACTOR_BP = [10., 44.]
    self.CL_ADJUST_FACTOR = [16. , 8.]
    # reenrey angle when to let go
    self.CL_REENTRY_ANGLE_BP = [10., 44.]
    self.CL_REENTRY_ANGLE = [5. , 5.]
    # a jump in angle above the CL_LANE_DETECT_FACTOR means we crossed the line
    self.CL_LANE_DETECT_BP = [10., 44.]
    self.CL_LANE_DETECT_FACTOR = [1.5, 2.5]
    self.CL_LANE_PASS_BP = [10., 20., 44.]
    self.CL_LANE_PASS_TIME = [40.,10., 4.] 
    # change lane delta angles and other params
    self.CL_MAXD_BP = [10., 32., 44.]
    self.CL_MAXD_A = [.358, 0.084, 0.040] #delta angle based on speed; needs fine tune, based on Tesla steer ratio of 16.75
    self.CL_MIN_V = 8.9 # do not turn if speed less than x m/2; 20 mph = 8.9 m/s
    # do not turn if actuator wants more than x deg for going straight; this should be interp based on speed
    self.CL_MAX_A_BP = [10., 44.]
    self.CL_MAX_A = [10., 10.] 
    # define limits for angle change every 0.1 s
    # we need to force correction above 10 deg but less than 20
    # anything more means we are going to steep or not enough in a turn
    self.CL_MAX_ACTUATOR_DELTA = 2.
    self.CL_MIN_ACTUATOR_DELTA = 0. 
    self.CL_CORRECTION_FACTOR = [1.,1.1,1.2]
    self.CL_CORRECTION_FACTOR_BP = [10., 32., 44.]
    #duration after we cross the line until we release is a factor of speed
    self.CL_TIMEA_BP = [10., 32., 44.]
    self.CL_TIMEA_T = [0.7 ,0.30, 0.30]
    #duration to wait (in seconds) with blinkers on before starting to turn
    self.CL_WAIT_BEFORE_START = 1
    #END OF ALCA PARAMS
    
    self.CP = CP
    
    #BB UIEvents
    self.UE = UIEvents(self)
    
    #BB variable for custom buttons
    self.cstm_btns = UIButtons(self,"Gm","gm")
    
    #BB pid holder for ALCA
    self.pid = None
    
    #BB custom message counter
    self.custom_alert_counter = -1 #set to 100 for 1 second display; carcontroller will take down to zero
    
    #BB visiond last type
    self.last_visiond = self.cstm_btns.btns[0].btn_label2
    
    # vEgo kalman filter
    dt = 0.01
    self.v_ego_kf = KF1D(x0=np.matrix([[0.], [0.]]),
                         A=np.matrix([[1., dt], [0., 1.]]),
                         C=np.matrix([1., 0.]),
                         K=np.matrix([[0.12287673], [0.29666309]]))
    self.v_ego = 0.
    #BB init ui buttons
  def init_ui_buttons(self):
    btns = []
    btns.append(UIButton("sound", "SND", 0, "", 0))
    btns.append(UIButton("alca", "ALC", 0, self.alcaLabels[self.alcaMode], 1))
    btns.append(UIButton("stop","",1,"SNG",2))
    btns.append(UIButton("vision","VIS",0,self.visionLabels[self.visionMode],3))
    btns.append(UIButton("gas","GAS",1,"",4))
    btns.append(UIButton("lka","LKA",1,"",5))
    return btns
  #BB update ui buttons
  def update_ui_buttons(self,id,btn_status):
    if self.cstm_btns.btns[id].btn_status > 0:
      if (id == 1) and (btn_status == 0) and self.cstm_btns.btns[id].btn_name=="alca":
          if self.cstm_btns.btns[id].btn_label2 == self.alcaLabels[self.alcaMode]:
            self.alcaMode = (self.alcaMode + 1 ) % 3
          else:
            self.alcaMode = 0
          self.cstm_btns.btns[id].btn_label2 = self.alcaLabels[self.alcaMode]
          self.cstm_btns.hasChanges = True
      if (id == 3) and (btn_status == 0) and self.cstm_btns.btns[id].btn_name=="vision":
          if self.cstm_btns.btns[id].btn_label2 == self.visionLabels[self.visionMode]:
            self.visionMode = (self.visionMode + 1 ) % 2
          else:
            self.visionMode = 0
          self.cstm_btns.btns[id].btn_label2 = self.visionLabels[self.visionMode]
          self.cstm_btns.hasChanges = True
          self.last_visiond = self.cstm_btns.btns[id].btn_label2
          args = ["/data/openpilot/selfdrive/car/modules/ch_visiond.sh", self.cstm_btns.btns[id].btn_label2]
          subprocess.Popen(args, shell = False, stdin=None, stdout=None, stderr=None, env = dict(os.environ), close_fds=True)
      else:
        self.cstm_btns.btns[id].btn_status = btn_status * self.cstm_btns.btns[id].btn_status
    else:
        self.cstm_btns.btns[id].btn_status = btn_status

  def update(self, pt_cp):

    self.can_valid = pt_cp.can_valid
    self.prev_cruise_buttons = self.cruise_buttons
    self.cruise_buttons = pt_cp.vl["ASCMSteeringButton"]['ACCButtons']
    self.prev_distance_button = self.distance_button
    self.distance_button = pt_cp.vl["ASCMSteeringButton"]["DistanceButton"]
    self.v_wheel_fl = pt_cp.vl["EBCMWheelSpdFront"]['FLWheelSpd'] * CV.KPH_TO_MS
    self.v_wheel_fr = pt_cp.vl["EBCMWheelSpdFront"]['FRWheelSpd'] * CV.KPH_TO_MS
    self.v_wheel_rl = pt_cp.vl["EBCMWheelSpdRear"]['RLWheelSpd'] * CV.KPH_TO_MS
    self.v_wheel_rr = pt_cp.vl["EBCMWheelSpdRear"]['RRWheelSpd'] * CV.KPH_TO_MS
    speed_estimate = float(np.mean([self.v_wheel_fl, self.v_wheel_fr, self.v_wheel_rl, self.v_wheel_rr]))

    self.v_ego_raw = speed_estimate
    v_ego_x = self.v_ego_kf.update(speed_estimate)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])

    self.prev_lka_button = self.lka_button
    self.lka_button = pt_cp.vl["ASCMSteeringButton"]['LKAButton']
    
    self.standstill = self.v_ego_raw < 0.01

    self.angle_steers = pt_cp.vl["PSCMSteeringAngle"]['SteeringWheelAngle']
    self.gear_shifter = parse_gear_shifter(pt_cp.vl["ECMPRDNL"]['PRNDL'])
    self.user_brake = pt_cp.vl["EBCMBrakePedalPosition"]['BrakePedalPosition']

    self.pedal_gas = pt_cp.vl["AcceleratorPedal"]['AcceleratorPedal']
    self.user_gas_pressed = self.pedal_gas > 0

    self.steer_torque_driver = pt_cp.vl["PSCMStatus"]['LKADriverAppldTrq']
    self.steer_override = abs(self.steer_torque_driver) > STEER_THRESHOLD

    # 0 - inactive, 1 - active, 2 - temporary limited, 3 - failed
    self.lkas_status = pt_cp.vl["PSCMStatus"]['LKATorqueDeliveredStatus']
    self.steer_not_allowed = not is_eps_status_ok(self.lkas_status, self.car_fingerprint)

    # 1 - open, 0 - closed
    self.door_all_closed = (pt_cp.vl["BCMDoorBeltStatus"]['FrontLeftDoor'] == 0 and
      pt_cp.vl["BCMDoorBeltStatus"]['FrontRightDoor'] == 0 and
      pt_cp.vl["BCMDoorBeltStatus"]['RearLeftDoor'] == 0 and
      pt_cp.vl["BCMDoorBeltStatus"]['RearRightDoor'] == 0)

    # 1 - latched
    self.seatbelt = pt_cp.vl["BCMDoorBeltStatus"]['LeftSeatBelt'] == 1

    self.steer_error = False

    self.brake_error = False
    self.can_valid = True

    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on
    self.left_blinker_on = pt_cp.vl["BCMTurnSignals"]['TurnSignals'] == 1
    self.right_blinker_on = pt_cp.vl["BCMTurnSignals"]['TurnSignals'] == 2

    if self.car_fingerprint in (CAR.VOLT, CAR.MALIBU, CAR.HOLDEN_ASTRA, CAR.ACADIA, CAR.CADILLAC_ATS):
      self.park_brake = pt_cp.vl["EPBStatus"]['EPBClosed']
      self.main_on = pt_cp.vl["ECMEngineStatus"]['CruiseMainOn']
      self.acc_active = False
      self.esp_disabled = pt_cp.vl["ESPStatus"]['TractionControlOn'] != 1
      self.pcm_acc_status = pt_cp.vl["AcceleratorPedal2"]['CruiseState']
      if self.car_fingerprint == CAR.VOLT:
        self.regen_pressed = bool(pt_cp.vl["EBCMRegenPaddle"]['RegenPaddle'])
      else:
        self.regen_pressed = False
    if self.car_fingerprint == CAR.CADILLAC_CT6:
      self.park_brake = False
      self.main_on = False
      self.acc_active = pt_cp.vl["ASCMActiveCruiseControlStatus"]['ACCCmdActive']
      self.esp_disabled = False
      self.regen_pressed = False
      self.pcm_acc_status = int(self.acc_active)

    # Brake pedal's potentiometer returns near-zero reading
    # even when pedal is not pressed.
    if self.user_brake < 10:
      self.user_brake = 0

    # Regen braking is braking
    self.brake_pressed = self.user_brake > 10 or self.regen_pressed

    self.gear_shifter_valid = self.gear_shifter == car.CarState.GearShifter.drive
    
  def get_follow_level(self):
    return self.follow_level
  
