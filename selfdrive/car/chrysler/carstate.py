from selfdrive.can.parser import CANParser
from selfdrive.car.chrysler.values import DBC, STEER_THRESHOLD
from common.kalman.simple_kalman import KF1D
from selfdrive.car.modules.UIBT_module import UIButtons,UIButton
from selfdrive.car.modules.UIEV_module import UIEvents
import numpy as np
import selfdrive.kegman_conf as kegman


def parse_gear_shifter(can_gear):
  if can_gear == 0x1:
    return "park"
  elif can_gear == 0x2:
    return "reverse"
  elif can_gear == 0x3:
    return "neutral"
  elif can_gear == 0x4:
    return "drive"
  elif can_gear == 0x5:
    return "low"
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
    ("TORQUE_DRIVER", "EPS_STATUS", 0),
    ("TORQUE_MOTOR", "EPS_STATUS", 0),
    ("LKAS_STATE", "EPS_STATUS", 1),
    ("COUNTER", "EPS_STATUS", -1),
    ("TRACTION_OFF", "TRACTION_BUTTON", 0),
    ("SEATBELT_DRIVER_UNLATCHED", "SEATBELT_STATUS", 0),
    ("COUNTER", "WHEEL_BUTTONS", -1),  # incrementing counter for 23b
  ]

  # It's considered invalid if it is not received for 10x the expected period (1/f).
  checks = [
    # sig_address, frequency
    ("BRAKE_2", 50),
    ("EPS_STATUS", 100),
    ("SPEED_1", 100),
    ("WHEEL_SPEEDS", 50),
    ("STEERING", 100),
    ("ACC_2", 50),
  ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

def get_camera_parser(CP):
  signals = [
    # sig_name, sig_address, default
    # TODO read in all the other values
    ("COUNTER", "LKAS_COMMAND", -1),
    ("CAR_MODEL", "LKAS_HUD", -1),
    ("LKAS_STATUS_OK", "LKAS_HEARTBIT", -1)
  ]
  checks = []
  
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)

class CarState(object):
  def __init__(self, CP):
    self.gasMode = int(kegman.conf['lastGasMode'])
    self.gasLabels = ["dynamic","sport","eco"]
    self.alcaLabels = ["MadMax","Normal","Wifey","off"]
    self.alcaMode = int(kegman.conf['lastALCAMode'])
    steerRatio = CP.steerRatio
    self.prev_distance_button = 0
    self.distance_button = 0
    self.prev_lka_button = 0
    self.lka_button = 0
    self.lkMode = True
    self.lane_departure_toggle_on = True
    # ALCA PARAMS
    self.blind_spot_on = bool(0)
    # max REAL delta angle for correction vs actuator
    self.CL_MAX_ANGLE_DELTA_BP = [10., 32., 55.]
    self.CL_MAX_ANGLE_DELTA = [0.77 * 16.2 / steerRatio, 0.86 * 16.2 / steerRatio, 0.535 * 16.2 / steerRatio]
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
    self.CL_TIMEA_T = [0.2 ,0.2, 0.2]
    #duration to wait (in seconds) with blinkers on before starting to turn
    self.CL_WAIT_BEFORE_START = 1
    #END OF ALCA PARAMS
    
    self.CP = CP
    
    #BB UIEvents
    self.UE = UIEvents(self)
    
   #BB variable for custom buttons
    self.cstm_btns = UIButtons(self,"Chrysler","chrysler")

    #BB pid holder for ALCA
    self.pid = None

    #BB custom message counter
    self.custom_alert_counter = -1 #set to 100 for 1 second display; carcontroller will take down to zero

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
    
   #BB init ui buttons
  def init_ui_buttons(self):
    btns = []
    btns.append(UIButton("sound", "SND", 0, "", 0))
    btns.append(UIButton("alca", "ALC", 0, self.alcaLabels[self.alcaMode], 1))
    btns.append(UIButton("mad","MAD",0,"",2))
    btns.append(UIButton("","",0,"",3))
    btns.append(UIButton("gas","GAS",1,self.gasLabels[self.gasMode],4))
    btns.append(UIButton("lka","LKA",1,"",5))
    return btns
  #BB update ui buttons
  def update_ui_buttons(self,id,btn_status):
    if self.cstm_btns.btns[id].btn_status > 0:
      if (id == 1) and (btn_status == 0) and self.cstm_btns.btns[id].btn_name=="alca":
          if self.cstm_btns.btns[id].btn_label2 == self.alcaLabels[self.alcaMode]:
            self.alcaMode = (self.alcaMode + 1 ) % 4
            kegman.save({'lastALCAMode': int(self.alcaMode)})  # write last distance bar setting to file
          else:
            self.alcaMode = 0
            kegman.save({'lastALCAMode': int(self.alcaMode)})  # write last distance bar setting to file
          self.cstm_btns.btns[id].btn_label2 = self.alcaLabels[self.alcaMode]
          self.cstm_btns.hasChanges = True
          if self.alcaMode == 3:
            self.cstm_btns.set_button_status("alca", 0)
      elif (id == 4) and (btn_status == 0) and self.cstm_btns.btns[id].btn_name=="gas":
          if self.cstm_btns.btns[id].btn_label2 == self.gasLabels[self.gasMode]:
            self.gasMode = (self.gasMode + 1 ) % 3
            kegman.save({'lastGasMode': int(self.gasMode)})  # write last GasMode setting to file
          else:
            self.gasMode = 0
            kegman.save({'lastGasMode': int(self.gasMode)})  # write last GasMode setting to file
          self.cstm_btns.btns[id].btn_label2 = self.gasLabels[self.gasMode]
          self.cstm_btns.hasChanges = True
      else:
        self.cstm_btns.btns[id].btn_status = btn_status * self.cstm_btns.btns[id].btn_status
    else:
        self.cstm_btns.btns[id].btn_status = btn_status
        if (id == 1) and self.cstm_btns.btns[id].btn_name=="alca":
          self.alcaMode = (self.alcaMode + 1 ) % 4
          kegman.save({'lastALCAMode': int(self.alcaMode)})  # write last distance bar setting to file
          self.cstm_btns.btns[id].btn_label2 = self.alcaLabels[self.alcaMode]
          self.cstm_btns.hasChanges = True
    
  def update(self, cp, cp_cam):
    # copy can_valid
    self.can_valid = cp.can_valid

    # update prevs, update must run once per loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.frame_23b = int(cp.vl["WHEEL_BUTTONS"]['COUNTER'])
    self.frame = int(cp.vl["EPS_STATUS"]['COUNTER'])

    self.door_all_closed = not any([cp.vl["DOORS"]['DOOR_OPEN_FL'],
                                    cp.vl["DOORS"]['DOOR_OPEN_FR'],
                                    cp.vl["DOORS"]['DOOR_OPEN_RL'],
                                    cp.vl["DOORS"]['DOOR_OPEN_RR']])
    self.seatbelt = (cp.vl["SEATBELT_STATUS"]['SEATBELT_DRIVER_UNLATCHED'] == 0)

    self.brake_pressed = cp.vl["BRAKE_2"]['BRAKE_PRESSED_2'] == 5 # human-only
    self.pedal_gas = cp.vl["ACCEL_PEDAL_MSG"]['ACCEL_PEDAL']
    self.car_gas = self.pedal_gas
    self.esp_disabled = (cp.vl["TRACTION_BUTTON"]['TRACTION_OFF'] == 1)

    self.v_wheel_fl = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_FL']
    self.v_wheel_rr = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RR']
    self.v_wheel_rl = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RL']
    self.v_wheel_fr = cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_FR']
    v_wheel = (cp.vl['SPEED_1']['SPEED_LEFT'] + cp.vl['SPEED_1']['SPEED_RIGHT']) / 2.

    # Kalman filter
    if abs(v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_kf.x = np.matrix([[v_wheel], [0.0]])

    self.v_ego_raw = v_wheel
    v_ego_x = self.v_ego_kf.update(v_wheel)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    self.standstill = not v_wheel > 0.001

    self.angle_steers = cp.vl["STEERING"]['STEER_ANGLE']
    self.angle_steers_rate = cp.vl["STEERING"]['STEERING_RATE']
    self.gear_shifter = parse_gear_shifter(cp.vl['GEAR']['PRNDL'])
    self.main_on = cp.vl["ACC_2"]['ACC_STATUS_2'] == 7  # ACC is green.
    self.left_blinker_on = cp.vl["STEERING_LEVERS"]['TURN_SIGNALS'] == 1
    self.right_blinker_on = cp.vl["STEERING_LEVERS"]['TURN_SIGNALS'] == 2

    self.steer_torque_driver = cp.vl["EPS_STATUS"]["TORQUE_DRIVER"]
    self.steer_torque_motor = cp.vl["EPS_STATUS"]["TORQUE_MOTOR"]
    self.steer_override = abs(self.steer_torque_driver) > STEER_THRESHOLD
    steer_state = cp.vl["EPS_STATUS"]["LKAS_STATE"]
    self.steer_error = steer_state == 4 or (steer_state == 0 and self.v_ego > self.CP.minSteerSpeed)

    self.user_brake = 0
    self.brake_lights = self.brake_pressed
    self.v_cruise_pcm = cp.vl["DASHBOARD"]['ACC_SPEED_CONFIG_KPH']
    self.pcm_acc_status = self.main_on

    self.generic_toggle = bool(cp.vl["STEERING_LEVERS"]['HIGH_BEAM_FLASH'])
    
    self.lkas_counter = cp_cam.vl["LKAS_COMMAND"]['COUNTER']
    self.lkas_car_model = cp_cam.vl["LKAS_HUD"]['CAR_MODEL']
    self.lkas_status_ok = cp_cam.vl["LKAS_HEARTBIT"]['LKAS_STATUS_OK']
    
    if self.cstm_btns.get_button_status("lka") == 0:
      self.lane_departure_toggle_on = False
    else:
      if self.alcaMode == 3 and (self.left_blinker_on or self.right_blinker_on):
        self.lane_departure_toggle_on = False
      else:
        self.lane_departure_toggle_on = True
