import os
import time

import common.numpy_fast as np
from common.realtime import sec_since_boot

import selfdrive.messaging as messaging

from common.realtime import sec_since_boot

from selfdrive.car.tesla.can_parser import CANParser
from selfdrive.can.parser import CANParser as CANParserC

from selfdrive.services import service_list

import zmq

NEW_CAN = os.getenv("OLD_CAN") is None
NEW_CAN = False

def get_can_parser(CP):
  # this function generates lists for signal, messages and initial values
  if CP.carFingerprint == "TESLA CLASSIC MODEL S":
    dbc_f = 'tesla_can.dbc'
    signals = [
      #("XMISSION_SPEED", 0x158, 0),
      #("WHEEL_SPEED_FL", 0x1d0, 0),
      #("WHEEL_SPEED_FR", 0x1d0, 0),
      #("WHEEL_SPEED_RL", 0x1d0, 0),
      #("WHEEL_SPEED_RR", 0x1d0, 0), #robertc (from JCT) -- might add later
      #("STEER_ANGLE", 0x156, 0),
      #("STEER_TORQUE_SENSOR", 0x18f, 0),
      #("MCU_gpsHDOP", 760, 0),
      ("MCU_gpsVehicleHeading", 760, 0),
      ("MCU_gpsVehicleSpeed", 760, 0),
      #("MCU_mppSpeedLimit", 760, 0),
      #("MCU_speedLimitUnits", 760, 0),
      #("MCU_userSpeedOffset", 760, 0),
      #("MCU_userSpeedOffsetUnits", 760, 0),
      #("MCU_clusterBrightnessLevel", 904, 0),
      #("LgtSens_Tunnel", 643, 0),
      #("LgtSens_Night", 643, 0),
      #("LgtSens_Twlgt", 643, 0),
      ("MCU_gpsAccuracy", 984, 0),
      #("StW_Angl", 3, 0),
      #("MC_STW_ANGL_STAT", 3, 0),
      #("SECOND", 792,0),
      #("MINUTE", 792,0),
      ("MCU_latitude", 984, 0),
      ("MCU_longitude", 984, 0),
      ("DI_vehicleSpeed", 280, 0),
      ("StW_AnglHP", 14, 0),
      ("EPAS_torsionBarTorque", 880, 0), # Used in interface.py
      ("EPAS_eacStatus", 880, 0),
      ("EPAS_handsOnLevel", 880, 0),
      ("DI_gear", 280, 3),
      ("DOOR_STATE_FL", 792, 1),
      ("DOOR_STATE_FR", 792, 1),
      ("DOOR_STATE_RL", 792, 1),
      ("DOOR_STATE_RR", 792, 1),
      ("DI_stateCounter", 872, 0),
      ("GTW_driverPresent", 840, 0),
      ("DI_brakePedal", 280, 0),
      ("SpdCtrlLvr_Stat", 69, 0),
      #("EPAS_eacErrorCode", 880, 5),
      #("ESP_brakeLamp", 309, 0),
      #("ESP_espOffLamp", 309, 0),
      #("ESP_tcOffLamp", 309, 0),
      #("ESP_tcLampFlash", 309, 0),
      #("ESP_espFaultLamp", 309, 0),
      ("DI_cruiseSet", 872, 0),
      ("DI_cruiseState", 872, 0),
      #("DI_pedalPos", 264, 0),
      ("VSL_Enbl_Rq", 69, 0), # Used in interface.py
      ("TSL_RND_Posn_StW",109 , 0),
      ("TSL_P_Psd_StW",109 , 0),
      ("TurnIndLvr_Stat", 69, 0),
      #("GTW_statusCounter", 840, 0),
      ("DI_motorRPM", 264, 0)
    ]
    # robertc: all frequencies must be integers with new C CAN parser (2.5 => 2)
    checks = [
      (3,   175), #JCT Actual message freq is 35 Hz (0.0286 sec)
      (14,  200), #JCT Actual message freq is 40 Hz (0.025 sec)
      (69,  17), #JCT Actual message freq is 3.5 Hz (0.285 sec)
      (109, 175), #JCT Actual message freq is 35 Hz (0.0286 sec)
      (264, 59), #JCT Actual message freq is 11.8 Hz (0.084 sec)
      (280, 18), #JCT Actual message freq is 3.7 Hz (0.275 sec)
      (309, 2), #JCT Actual message freq is 0.487 Hz (2.05 sec)
      (760, 2), #JCT Actual message freq is 0.487 Hz (2.05 sec)
      (792, 16), #JCT Actual message freq is 3.3 Hz (0.3 sec)
      (840, 2), #JCT Actual message freq is 0.5 Hz (2 sec)
      (872, 5), #JCT Actual message freq is 1 Hz (1 sec)
      (643, 5), #JCT Actual message freq is 1.3 Hz (0.76 sec)
      (880, 5), #JCT Actual message freq is 1.3 Hz (0.76 sec)
      (904, 5), #JCT Actual message freq is 1.3 Hz (0.76 sec)
      (984, 5), #JCT Actual message freq is 1.3 Hz (0.76 sec)
    ]

  # add gas interceptor reading if we are using it
  # robertc: What is this exactly?
  #if CP.enableGas:
  #  signals.append(("INTERCEPTOR_GAS", 0x201, 0))
  #  checks.append((0x201, 50))

  if NEW_CAN:
    return CANParserC(os.path.splitext(dbc_f)[0], signals, checks, 0)
  else:
    return CANParser(dbc_f, signals, checks)

class CarState(object):
  def __init__(self, CP, logcan):
    # robertc: got rid of all the Honda/Acura-specific stuff
    self.brake_only = CP.enableCruise
    self.CP = CP

    # initialize can parser
    self.cp = get_can_parser(CP)

    self.user_gas, self.user_gas_pressed = 0., 0

    self.cruise_buttons = 0
    self.cruise_setting = 0
    self.blinker_on = 0
    
    self.log_interval = sec_since_boot()

    #self.prev_user_brightness = 50
    #self.user_brightness = 50
    #self.display_brightness = 50
    #self.day_or_night = 0

    self.left_blinker_on = 0
    self.right_blinker_on = 0

    context = zmq.Context()
    self.gps_sock = messaging.pub_sock(context, service_list['gpsLocationExternal'].port)

    # TODO: actually make this work
    self.a_ego = 0.

  def update(self, can_pub_main=None):
    cp = self.cp

    if NEW_CAN:
      cp.update(int(sec_since_boot() * 1e9), False)
    else:
      cp.update_can(can_pub_main)
    
    # copy can_valid
    self.can_valid = cp.can_valid

    # car params
    v_weight_v  = [0., 1. ]  # don't trust smooth speed at low values to avoid premature zero snapping
    v_weight_bp = [1., 6.]   # smooth blending, below ~0.6m/s the smooth speed snaps to zero

    # update prevs, update must run once per loop
    self.prev_cruise_buttons = self.cruise_buttons
    self.prev_cruise_setting = self.cruise_setting
    self.prev_blinker_on = self.blinker_on

    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.rpm = cp.vl[264]['DI_motorRPM']

    # ******************* parse out can *******************
    self.door_all_closed = not any([cp.vl[792]['DOOR_STATE_FL'], cp.vl[792]['DOOR_STATE_FR'],
                               cp.vl[792]['DOOR_STATE_RL'], cp.vl[792]['DOOR_STATE_RR']])  #JCT
    self.seatbelt = cp.vl[840]['GTW_driverPresent']
    self.steer_error = cp.vl[880]['EPAS_eacStatus'] not in [2,1]
    self.steer_not_allowed = 0 #cp.vl[880]['EPAS_eacErrorCode'] != 0
    self.brake_error = 0 #NOT WORKINGcp.vl[309]['ESP_brakeLamp'] #JCT
    # JCT More ESP errors available, these needs to be added once car steers on its own to disable / alert driver
    # JCT More ESP errors available, these needs to be added once car steers on its own to disable / alert driver
    self.esp_disabled = 0 #NEED TO CORRECT DBC cp.vl[309]['ESP_espOffLamp'] or cp.vl[309]['ESP_tcOffLamp'] or cp.vl[309]['ESP_tcLampFlash'] or cp.vl[309]['ESP_espFaultLamp'] #JCT
    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel = 0 #JCT
    self.v_weight = 0 #JCT
    self.v_ego = (cp.vl[280]['DI_vehicleSpeed'])*1.609/3.6 #JCT MPH_TO_MS. Tesla is in MPH, v_ego is expected in M/S
    self.user_gas = 0 #for now
    self.user_gas_pressed = 0 #for now
    self.angle_steers = -(cp.vl[14]['StW_AnglHP']) #JCT polarity reversed from Honda/Acura
    self.gear = 0 
    #JCT removed for now, will be used for cruise ON/OFF aka yellow led: self.cruise_setting = cp.vl[69]['VSL_Enbl_Rq']    
    self.cruise_buttons = cp.vl[69]['SpdCtrlLvr_Stat']
    self.main_on = 1 #0 #JCT Need to find the Cruise ON/OFF switch that illuminate the yellow led on the cruise stalk
    self.gear_shifter = cp.vl[280]['DI_gear']
    self.gear_shifter_valid = (cp.vl[109]['TSL_RND_Posn_StW'] == 0) and (cp.vl[109]['TSL_P_Psd_StW'] ==0) and (cp.vl[280]['DI_gear'] ==4) #JCT PND stalk idle and Park button not pressed and in DRIVE
    self.blinker_on = 0 #cp.vl[69]['TurnIndLvr_Stat'] in [1,2]
    self.left_blinker_on = cp.vl[69]['TurnIndLvr_Stat'] == 1
    self.right_blinker_on = cp.vl[69]['TurnIndLvr_Stat'] == 2
    self.car_gas = 0 #cp.vl[264]['DI_pedalPos']
    self.steer_override = abs(cp.vl[880]['EPAS_handsOnLevel']) > 0 #JCT need to be tested and adjusted. 8=38.7% of range as george uses. Range -20.4 to 20.4 Nm
    self.brake_pressed = cp.vl[280]['DI_brakePedal']
    self.user_brake = cp.vl[280]['DI_brakePedal']
    self.standstill = cp.vl[280]['DI_vehicleSpeed'] == 0
    self.v_cruise_pcm = cp.vl[872]['DI_cruiseSet']
    self.pcm_acc_status = cp.vl[872]['DI_cruiseState']
    self.pedal_gas = 0 #cp.vl[264]['DI_pedalPos']
    self.hud_lead = 0 #JCT TBD
    self.counter_pcm = cp.vl[872]['DI_stateCounter']
    #self.display_brightness = cp.vl[904]['MCU_clusterBrightnessLevel']
    #self.day_or_night = cp.vl[643]['LgtSens_Night']
    #self.tunnel = cp.vl[643]['LgtSens_Tunnel']
    #self.LgtSens_Twlgt = cp.vl[643]['LgtSens_Twlgt']


    ## JCT Action every 1 sec
    if (sec_since_boot() - self.log_interval) > 1:
      
      # GPS Addon
      dat = messaging.new_message()
      dat.init('gpsLocation')
      dat.gpsLocation.source = 2 # getting gps from car (on CAN bus)
      dat.gpsLocation.flags = 0x1D # see line 89 of https://source.android.com/devices/halref/gps_8h_source.html#l00531
      dat.gpsLocation.latitude = cp.vl[984]['MCU_latitude'] # in degrees
      dat.gpsLocation.longitude = cp.vl[984]['MCU_longitude'] # in degrees
      dat.gpsLocation.altitude = 0 # Flags += 2 once available (will give flags=0x1F) in meters above the WGS 84 reference ellipsoid
      dat.gpsLocation.speed = (cp.vl[760]['MCU_gpsVehicleSpeed'])*1./3.6 # in meters per second
      dat.gpsLocation.bearing = cp.vl[760]['MCU_gpsVehicleHeading'] # in degrees
      dat.gpsLocation.accuracy = cp.vl[984]['MCU_gpsAccuracy'] # in meters
      dat.gpsLocation.timestamp = int(time.time()) # Milliseconds since January 1, 1970. (GPS time N/A for now)
      self.gps_sock.send(dat.to_bytes())
      
      #if not self.day_or_night and not self.tunnel:
      #    self.user_brightness = self.display_brightness if self.display_brightness > 50 else 50 # During day anything bellow 50% is too dim
      #else:
      #    self.user_brightness = self.display_brightness # Full brightness range (0-100%) if night or tunnel
      #
      #if self.user_brightness != self.prev_user_brightness:
      #  f2 = open('/sdcard/brightness', 'w')
      #  s = "%3.1f" % (self.user_brightness)
      #  f2.write(s)
      #  f2.close()
      
      #self.prev_user_brightness = self.user_brightness 
      
      #if self.gear_shifter == 4:
      #  gear = 'D'
      #elif self.gear_shifter == 1:
      #  gear = 'P'
      #elif self.gear_shifter == 2:
      #  gear = 'R'
      #else:
      #  gear = '?'
      #f = open('/sdcard/candiac.log', 'a')
      #s = "SPD=%3.1f RPM=%5d ANGLE=%5.1f GEAR=%s DOORS=%2d Standstill=%d Gas=%3.1f Brake=%1d GearValid=%1d D=%3.1f%% Night=%d Tunnel=%d Twlgt=%d\n" % (self.v_ego*3.6, self.rpm, self.angle_steers, gear, self.door_all_closed, self.standstill, self.car_gas, self.user_brake, self.gear_shifter_valid, self.display_brightness, self.day_or_night, self.tunnel, self.LgtSens_Twlgt)
      #f.write(s)
      #f.close()
      
      self.log_interval = sec_since_boot()
    
