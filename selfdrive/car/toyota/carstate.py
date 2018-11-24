import zmq
from math import acos,sin,cos,radians
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from common.kalman.simple_kalman import KF1D
import numpy as np
from common.numpy_fast import interp
from selfdrive.can.parser import CANParser, CANDefine
from selfdrive.config import Conversions as CV
from selfdrive.car.toyota.values import CAR, DBC, STEER_THRESHOLD
from common.kalman.simple_kalman import KF1D
from selfdrive.car.modules.UIBT_module import UIButtons,UIButton
from selfdrive.car.modules.UIEV_module import UIEvents

def gps_distance(gpsLat, gpsLon, gpsAlt, gpsAcc):
  A = np.array([(6371010+gpsAlt)*sin(radians(gpsLat-90))*cos(radians(gpsLon)),(6371010+gpsAlt)*sin(radians(gpsLat-90))*sin(radians(gpsLon)),(6371010+gpsAlt)*cos(radians(gpsLat-90))])
  #x y z alt altacc includeradius approachradius speedlimit
  B = np.array([[-4190726.5,-723704.4,4744593.25,575.5,15.000001,22.000001,100.000001,25.000001],[-4182729.45,-730269.75,4750656.55,587.7,15.000001,22.000001,100.000001,25.000001],[-4182656.23190573,-728404.35947068,4750970.77416399,560.4,5.000001,25.000001,100.000001,25.000001],[-4182340.93735163,-728347.57108258,4751257.30835904,560.6,15.000001,23.000001,100.000001,25.000001],[1997559.7577528,4923507.37728524,3516097.79232965,341.001,150.000001,22.000001,100.000001,35.000001],[1999313.05346417,4923689.65187786,3514838.56252996,341.001,150.000001,22.000001,100.000001,35.000001],[1999253.66753953,  4922766.06552066,  3516116.35225406, 309.701,15.000001,22.000001,100.000001,35.000001]])
  dist = 999.000001
  #lat=48.128939
  #lon=9.797879048
  #lonlatacc=1.00001
  #alt=575.5
  includeradius = 22.000001
  approachradius = 100.0001
  speedlimit = 25.000001
  minindex = np.argmin((np.sum((B[:,[0,1,2]] - A)**2,axis=1))**0.5, axis=0)
  altacc = B[minindex,4]
  includeradius = B[minindex,5]
  approachradius = B[minindex,6]
  speedlimit = float(B[minindex,7])
  
  if abs(gpsAlt -B[minindex,3]) < altacc:
    if gpsAcc<1.00001:
      #dist = 6371010*acos(sin(radians(gpsLat))*sin(radians(lat))+cos(radians(gpsLat))*cos(radians(lat))*cos(radians(gpsLon-lon)))
      dist = (np.sum((B[minindex,[0,1,2]] - A)**2))**0.5
  #else:
    #print "Altitude inacurate"
  return dist, includeradius, approachradius, speedlimit

def parse_gear_shifter(gear, vals):

  val_to_capnp = {'P': 'park', 'R': 'reverse', 'N': 'neutral',
                  'D': 'drive', 'B': 'brake'}
  try:
    return val_to_capnp[vals[gear]]
  except KeyError:
    return "unknown"


def get_can_parser(CP):

  signals = [
    # sig_name, sig_address, default
    ("GEAR", "GEAR_PACKET", 0),
    ("BRAKE_PRESSED", "BRAKE_MODULE", 0),
    ("GAS_PEDAL", "GAS_PEDAL", 0),
    ("WHEEL_SPEED_FL", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_FR", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_RL", "WHEEL_SPEEDS", 0),
    ("WHEEL_SPEED_RR", "WHEEL_SPEEDS", 0),
    ("DOOR_OPEN_FL", "SEATS_DOORS", 1),
    ("DOOR_OPEN_FR", "SEATS_DOORS", 1),
    ("DOOR_OPEN_RL", "SEATS_DOORS", 1),
    ("DOOR_OPEN_RR", "SEATS_DOORS", 1),
    ("SEATBELT_DRIVER_UNLATCHED", "SEATS_DOORS", 1),
    ("TC_DISABLED", "ESP_CONTROL", 1),
    ("STEER_ANGLE", "STEER_ANGLE_SENSOR", 0),
    ("STEER_FRACTION", "STEER_ANGLE_SENSOR", 0),
    ("STEER_RATE", "STEER_ANGLE_SENSOR", 0),
    ("GAS_RELEASED", "PCM_CRUISE", 0),
    ("CRUISE_STATE", "PCM_CRUISE", 0),
    ("MAIN_ON", "PCM_CRUISE_2", 0),
    ("SET_SPEED", "PCM_CRUISE_2", 0),
    ("LOW_SPEED_LOCKOUT", "PCM_CRUISE_2", 0),
    ("STEER_TORQUE_DRIVER", "STEER_TORQUE_SENSOR", 0),
    ("STEER_TORQUE_EPS", "STEER_TORQUE_SENSOR", 0),
    ("TURN_SIGNALS", "STEERING_LEVERS", 3),   # 3 is no blinkers
    ("LKA_STATE", "EPS_STATUS", 0),
    ("IPAS_STATE", "EPS_STATUS", 1),
    ("BRAKE_LIGHTS_ACC", "ESP_CONTROL", 0),
    ("AUTO_HIGH_BEAM", "LIGHT_STALK", 0),
    ("BLINDSPOT","DEBUG", 0),
    ("BLINDSPOTSIDE","DEBUG",65),
    ("BLINDSPOTD1","DEBUG", 0),
    ("BLINDSPOTD2","DEBUG", 0),
    ("ACC_DISTANCE", "JOEL_ID", 2),
    ("LANE_WARNING", "JOEL_ID", 1),
    ("ACC_SLOW", "JOEL_ID", 0),
    ("DISTANCE_LINES", "PCM_CRUISE_SM", 0),
  ]

  checks = [
    ("BRAKE_MODULE", 40),
    ("GAS_PEDAL", 33),
    ("WHEEL_SPEEDS", 80),
    ("STEER_ANGLE_SENSOR", 80),
    ("PCM_CRUISE", 33),
    ("PCM_CRUISE_2", 33),
    ("STEER_TORQUE_SENSOR", 50),
    ("EPS_STATUS", 25),
  ]

  if CP.carFingerprint == CAR.PRIUS:
    signals += [("STATE", "AUTOPARK_STATUS", 0)]

  # add gas interceptor reading if we are using it
  if CP.enableGasInterceptor:
    signals.append(("INTERCEPTOR_GAS", "GAS_SENSOR", 0))
    checks.append(("GAS_SENSOR", 50))
    
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)


class CarState(object):
  def __init__(self, CP):
    self.Angle = [0, 5, 10, 15,20,25,30,35,60,100,180,270,500]
    self.Angle_Speed = [255,160,100,80,70,60,55,50,40,33,27,17,12]
    #labels for ALCA modes
    self.alcaLabels = ["MadMax","Normal","Wifey"]
    self.alcaMode = 0
    #if (CP.carFingerprint == CAR.MODELS):
    # ALCA PARAMS
    # max REAL delta angle for correction vs actuator
    self.CL_MAX_ANGLE_DELTA_BP = [10., 15., 32., 44.]#[10., 44.]
    self.CL_MAX_ANGLE_DELTA = [2.0, 1.75, 0.96, 0.4]
     # adjustment factor for merging steer angle to actuator; should be over 4; the higher the smoother
    self.CL_ADJUST_FACTOR_BP = [10., 44.]
    self.CL_ADJUST_FACTOR = [16. , 8.]
     # reenrey angle when to let go
    self.CL_REENTRY_ANGLE_BP = [10., 44.]
    self.CL_REENTRY_ANGLE = [5. , 5.]
     # a jump in angle above the CL_LANE_DETECT_FACTOR means we crossed the line
    self.CL_LANE_DETECT_BP = [10., 44.]
    self.CL_LANE_DETECT_FACTOR = [1.3, 1.3]
    self.CL_LANE_PASS_BP = [10., 20., 44.]
    self.CL_LANE_PASS_TIME = [40.,10., 3.] 
     # change lane delta angles and other params
    self.CL_MAXD_BP = [10., 32., 44.]
    self.CL_MAXD_A = [.358, 0.084, 0.042] #delta angle based on speed; needs fine tune, based on Tesla steer ratio of 16.75
    self.CL_MIN_V = 8.9 # do not turn if speed less than x m/2; 20 mph = 8.9 m/s
     # do not turn if actuator wants more than x deg for going straight; this should be interp based on speed
    self.CL_MAX_A_BP = [10., 44.]
    self.CL_MAX_A = [10., 10.] 
     # define limits for angle change every 0.1 s
    # we need to force correction above 10 deg but less than 20
    # anything more means we are going to steep or not enough in a turn
    self.CL_MAX_ACTUATOR_DELTA = 2.
    self.CL_MIN_ACTUATOR_DELTA = 0. 
    self.CL_CORRECTION_FACTOR = [1.3,1.2,1.2]
    self.CL_CORRECTION_FACTOR_BP = [10., 32., 44.]
     #duration after we cross the line until we release is a factor of speed
    self.CL_TIMEA_BP = [10., 32., 44.]
    self.CL_TIMEA_T = [0.7 ,0.30, 0.20]
    #duration to wait (in seconds) with blinkers on before starting to turn
    self.CL_WAIT_BEFORE_START = 1
    #END OF ALCA PARAMS
    
    context = zmq.Context()
    #gps_ext_sock = messaging.sub_sock(context, service_list['gpsLocationExternal'].port, poller)
    self.gps_location = messaging.sub_sock(context, service_list['gpsLocationExternal'].port)
    self.CP = CP
    self.can_define = CANDefine(DBC[CP.carFingerprint]['pt'])
    self.shifter_values = self.can_define.dv["GEAR_PACKET"]['GEAR']
    self.left_blinker_on = 0
    self.right_blinker_on = 0
    self.distance = 999
    self.inaccuracy = 1.01
    self.speedlimit = 25
    self.approachradius = 100
    self.includeradius = 22
    self.blind_spot_on = bool(0)
    self.distance_toggle_prev = 2
    self.read_distance_lines_prev = 3
    self.lane_departure_toggle_on_prev = True
    self.acc_slow_on_prev = False
    #BB UIEvents
    self.UE = UIEvents(self)

    #BB variable for custom buttons
    self.cstm_btns = UIButtons(self,"Toyota","toyota")

    #BB pid holder for ALCA
    self.pid = None

    #BB custom message counter
    self.custom_alert_counter = 100 #set to 100 for 1 second display; carcontroller will take down to zero
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
    btns.append(UIButton("alca", "ALC", 1, self.alcaLabels[self.alcaMode], 1))
    btns.append(UIButton("slow", "SLO", 1, "", 2))
    btns.append(UIButton("lka", "LKA", 1, "", 3))
    btns.append(UIButton("tr", "TR", 0, "", 4))
    btns.append(UIButton("gas", "GAS", 0, "", 5))
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
      else:
        self.cstm_btns.btns[id].btn_status = btn_status * self.cstm_btns.btns[id].btn_status
    else:
        self.cstm_btns.btns[id].btn_status = btn_status

  def update(self, cp):
    # copy can_valid
    self.can_valid = cp.can_valid

    msg = messaging.recv_one_or_none(self.gps_location)
    if msg is not None:
      gps_pkt = msg.gpsLocationExternal
      self.inaccuracy = gps_pkt.accuracy
      self.prev_distance = self.distance
      self.distance, self.includeradius, self.approachradius, self.speedlimit = gps_distance(gps_pkt.latitude,gps_pkt.longitude,gps_pkt.altitude,gps_pkt.accuracy)
      #self.distance = 6371010*acos(sin(radians(gps_pkt.latitude))*sin(radians(48.12893908))+cos(radians(gps_pkt.latitude))*cos(radians(48.12893908))*cos(radians(gps_pkt.longitude-9.797879048)))
    # update prevs, update must run once per loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.door_all_closed = not any([cp.vl["SEATS_DOORS"]['DOOR_OPEN_FL'], cp.vl["SEATS_DOORS"]['DOOR_OPEN_FR'],
                                    cp.vl["SEATS_DOORS"]['DOOR_OPEN_RL'], cp.vl["SEATS_DOORS"]['DOOR_OPEN_RR']])
    self.seatbelt = not cp.vl["SEATS_DOORS"]['SEATBELT_DRIVER_UNLATCHED']

    self.brake_pressed = cp.vl["BRAKE_MODULE"]['BRAKE_PRESSED']
    if self.CP.enableGasInterceptor:
      self.pedal_gas = cp.vl["GAS_SENSOR"]['INTERCEPTOR_GAS']
    else:
      self.pedal_gas = cp.vl["GAS_PEDAL"]['GAS_PEDAL']
    self.car_gas = self.pedal_gas
    self.esp_disabled = cp.vl["ESP_CONTROL"]['TC_DISABLED']

    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_FL'] * CV.KPH_TO_MS
    self.v_wheel_fr = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_FR'] * CV.KPH_TO_MS
    self.v_wheel_rl = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_RL'] * CV.KPH_TO_MS
    self.v_wheel_rr = cp.vl["WHEEL_SPEEDS"]['WHEEL_SPEED_RR'] * CV.KPH_TO_MS
    self.v_wheel = float(np.mean([self.v_wheel_fl, self.v_wheel_fr, self.v_wheel_rl, self.v_wheel_rr]))

    # Kalman filter
    if abs(self.v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_x = np.matrix([[self.v_wheel], [0.0]])

    self.v_ego_raw = self.v_wheel
    v_ego_x = self.v_ego_kf.update(self.v_wheel)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    #self.standstill = not self.v_wheel > 0.001
    self.standstill = False

    self.angle_steers = cp.vl["STEER_ANGLE_SENSOR"]['STEER_ANGLE'] + cp.vl["STEER_ANGLE_SENSOR"]['STEER_FRACTION']
    self.angle_steers_rate = cp.vl["STEER_ANGLE_SENSOR"]['STEER_RATE']
    can_gear = int(cp.vl["GEAR_PACKET"]['GEAR'])
    self.gear_shifter = parse_gear_shifter(can_gear, self.shifter_values)
    self.main_on = cp.vl["PCM_CRUISE_2"]['MAIN_ON']
    self.left_blinker_on = cp.vl["STEERING_LEVERS"]['TURN_SIGNALS'] == 1
    self.right_blinker_on = cp.vl["STEERING_LEVERS"]['TURN_SIGNALS'] == 2
    self.blind_spot_side = cp.vl["DEBUG"]['BLINDSPOTSIDE']

    if (cp.vl["DEBUG"]['BLINDSPOTD1'] > 10) or (cp.vl["DEBUG"]['BLINDSPOTD1'] > 10):
      self.blind_spot_on = bool(1)
    else:
      self.blind_spot_on = bool(0)
    
    # we could use the override bit from dbc, but it's triggered at too high torque values
    self.steer_override = abs(cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_DRIVER']) > 100

    # 2 is standby, 10 is active. TODO: check that everything else is really a faulty state
    self.steer_state = cp.vl["EPS_STATUS"]['LKA_STATE']
    self.steer_error = cp.vl["EPS_STATUS"]['LKA_STATE'] not in [1, 5, 9, 17, 25]
    self.ipas_active = cp.vl['EPS_STATUS']['IPAS_STATE'] == 3
    self.brake_error = 0
    self.steer_torque_driver = cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_DRIVER']
    self.steer_torque_motor = cp.vl["STEER_TORQUE_SENSOR"]['STEER_TORQUE_EPS']
    if bool(cp.vl["JOEL_ID"]['LANE_WARNING']) <> self.lane_departure_toggle_on_prev:
      self.lane_departure_toggle_on = bool(cp.vl["JOEL_ID"]['LANE_WARNING'])
      if self.lane_departure_toggle_on:
        self.cstm_btns.set_button_status("lka", 1)
      else:
        self.cstm_btns.set_button_status("lka", 0)
      self.lane_departure_toggle_on_prev = self.lane_departure_toggle_on
    else:
      if self.cstm_btns.get_button_status("lka") == 0:
        self.lane_departure_toggle_on = False
      else:
        self.lane_departure_toggle_on = True
      
    self.distance_toggle = cp.vl["JOEL_ID"]['ACC_DISTANCE']
    self.read_distance_lines = cp.vl["PCM_CRUISE_SM"]['DISTANCE_LINES']
    if self.distance_toggle <> self.distance_toggle_prev:
      if self.read_distance_lines == self.distance_toggle:
        self.distance_toggle_prev = self.distance_toggle
      else:
        self.cstm_btns.set_button_status("tr", 1)
    if self.read_distance_lines <> self.read_distance_lines_prev:
      if self.read_distance_lines == 1:
        self.UE.custom_alert_message(2,"Following distance set to 0.9s",200,3)
      if self.read_distance_lines == 2:
        self.UE.custom_alert_message(2,"Following distance set to 1.8s",200,3)
      if self.read_distance_lines == 3:
        self.UE.custom_alert_message(2,"Following distance set to 2.7s",200,3)
      self.read_distance_lines_prev = self.read_distance_lines
    
    if bool(cp.vl["JOEL_ID"]['ACC_SLOW']) <> self.acc_slow_on_prev:
      self.acc_slow_on = bool(cp.vl["JOEL_ID"]['ACC_SLOW'])
      if self.acc_slow_on:
        self.cstm_btns.set_button_status("slow", 1)
      else:
        self.cstm_btns.set_button_status("slow", 0)
      self.acc_slow_on_prev = self.acc_slow_on
    else:
      if self.cstm_btns.get_button_status("slow") == 0:
        self.acc_slow_on = False
      else:
        self.acc_slow_on = True

    # we could use the override bit from dbc, but it's triggered at too high torque values
    self.steer_override = abs(self.steer_torque_driver) > STEER_THRESHOLD

    self.user_brake = 0
    if self.acc_slow_on:
      self.v_cruise_pcm = max(7, cp.vl["PCM_CRUISE_2"]['SET_SPEED'] - 34.0)
    else:
      self.v_cruise_pcm = cp.vl["PCM_CRUISE_2"]['SET_SPEED']
    self.v_cruise_pcm = int(min(self.v_cruise_pcm, interp(abs(self.angle_steers), self.Angle, self.Angle_Speed)))
    #print "distane"
    #print self.distance
    if self.distance < self.approachradius + self.includeradius:
      print "speed"
      print self.prev_distance - self.distance
      #if speed is 5% higher than the speedlimit
      if self.prev_distance - self.distance > self.speedlimit*0.00263889:
       if self.v_cruise_pcm > self.speedlimit:
         self.v_cruise_pcm = self.speedlimit
    if self.distance < self.includeradius:
      print "inside"
      if self.v_cruise_pcm > self.speedlimit:
        self.v_cruise_pcm =  self.speedlimit
    
    self.pcm_acc_status = cp.vl["PCM_CRUISE"]['CRUISE_STATE']
    self.gas_pressed = not cp.vl["PCM_CRUISE"]['GAS_RELEASED']
    self.low_speed_lockout = cp.vl["PCM_CRUISE_2"]['LOW_SPEED_LOCKOUT'] == 2
    self.brake_lights = bool(cp.vl["ESP_CONTROL"]['BRAKE_LIGHTS_ACC'] or self.brake_pressed)
    if self.CP.carFingerprint == CAR.PRIUS:
      self.generic_toggle = cp.vl["AUTOPARK_STATUS"]['STATE'] != 0
    else:
      self.generic_toggle = bool(cp.vl["LIGHT_STALK"]['AUTO_HIGH_BEAM'])
    
