from cereal import car
from common.conversions import Conversions as CV
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.chrysler.values import DBC, STEER_THRESHOLD, CAR
class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["Transmission_Status"]["Gear_State"]
    self.lkasdisabled = 0
    self.lkasbuttonprev = 0
  
  def update(self, cp, cp_cam):

    ret = car.CarState.new_message() 

    # lock info 
    ret.doorOpen = any([cp.vl["BCM_1"]["Driver_Door_Ajar"],
                        cp.vl["BCM_1"]["Passenger_Door_Ajar"],
                        cp.vl["BCM_1"]["Left_Rear_Door_Ajar"],
                        cp.vl["BCM_1"]["Right_Rear_Door_Ajar"]])
    ret.seatbeltUnlatched = cp.vl["ORC_1"]['Driver_Seatbelt_Status'] == 1 #1 is unbuckled

    # brake pedal
    ret.brakePressed = cp.vl["ESP_1"]['Brake_Pedal_State'] ==1  # Physical brake pedal switch
    ret.brake = 0

    # gas pedal
    ret.gas = cp.vl["ECM_5"]["Accelerator_Position"]
    ret.gasPressed = ret.gas > 1e-5 
  
  # car speed
    ret.vEgoRaw = cp.vl["ESP_8"]["Vehicle_Speed"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = not ret.vEgoRaw > 0.001
    ret.wheelSpeeds = self.get_wheel_speeds(
    cp.vl["ESP_6"]["Wheel_RPM_Front_Left"],
    cp.vl["ESP_6"]["Wheel_RPM_Rear_Right"],
    cp.vl["ESP_6"]["Wheel_RPM_Rear_Left"],
    cp.vl["ESP_6"]["Wheel_RPM_Front_Right"],
    unit=1,
    )
    #ret.aEgo = cp.vl["ESP_4"]["Acceleration"] #m/s2
    #ret.yawRate = cp.vl["ESP_4"]["Yaw_Rate"] #deg/s

    # button presses
    ret.leftBlinker = (cp.vl["Steering_Column_Commands"]["Turn_Signal_Status"] == 1)
    ret.rightBlinker = (cp.vl["Steering_Column_Commands"]["Turn_Signal_Status"] == 2)
    ret.genericToggle = bool(cp.vl["Steering_Column_Commands"]["High_Beam_Lever_Status"])

  # steering wheel  
    ret.steeringAngleDeg = cp.vl["Steering_Column_Angle_Status"]["Steering_Wheel_Angle"]
    ret.steeringRateDeg = cp.vl["Steering_Column_Angle_Status"]["Steering_Rate"]
    ret.steeringTorque = cp.vl["EPS_2"]["Steering_Column_Torque"]
    ret.steeringTorqueEps = cp.vl["EPS_2"]["EPS_Motor_Torque"]
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD
    ret.espDisabled = (cp.vl["Center_Stack_1"]["Traction_Button"] == 1) #button is pressed. This doesn't mean ESP is diabled.
    self.frame = int(cp.vl["EPS_2"]["COUNTER"])
    # steer_state = cp.vl["EPS_2"]["LKAS_STATE"]
    # ret.steerFaultPermanent = steer_state == 4 or (steer_state == 0 and ret.vEgo > self.CP.minSteerSpeed)

    # gear
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(cp.vl["Transmission_Status"]["Gear_State"], None))

  # cruise state  
    self.lkas_counter = cp_cam.vl["DAS_3"]["COUNTER"]
    self.lanelines = cp_cam.vl["DAS_6"]["LKAS_LANE_LINES"]
    self.iconcolor = cp_cam.vl["DAS_6"]["LKAS_ICON_COLOR"]
    self.lkas_car_model = cp_cam.vl["DAS_6"]["CAR_MODEL"] 
    self.lkasalerts = cp_cam.vl["DAS_6"]["LKAS_ALERTS"]
    self.accaccel = cp.vl["Cruise_Control_Buttons"]["ACC_Accel"]
    self.accdecel = cp.vl["Cruise_Control_Buttons"]["ACC_Decel"]
    self.acccancel = cp.vl["Cruise_Control_Buttons"]["ACC_Cancel"]
    self.accdistancedec = cp.vl["Cruise_Control_Buttons"]["ACC_Distance_Dec"]
    self.accdistanceinc = cp.vl["Cruise_Control_Buttons"]["ACC_Distance_Inc"]
    self.accresume = cp.vl["Cruise_Control_Buttons"]["ACC_Resume"]
    self.cruiseonoff = cp.vl["Cruise_Control_Buttons"]["Cruise_OnOff"]
    self.acconoff = cp.vl["Cruise_Control_Buttons"]["ACC_OnOff"]
    self.button_counter = cp.vl["Cruise_Control_Buttons"]["COUNTER"]
    self.cruise = cp.vl["Cruise_Control_Buttons"]


    if self.CP.carFingerprint in (CAR.PACIFICA_2017_HYBRID, CAR.PACIFICA_2018_HYBRID, CAR.PACIFICA_2019_HYBRID, CAR.PACIFICA_2018, CAR.PACIFICA_2020, CAR.JEEP_CHEROKEE_2019, CAR.JEEP_CHEROKEE):
      self.lkasbutton = (cp.vl["Center_Stack_1"]["LKAS_Button"] == 1)
      #if self.lkasbutton ==1 and self.lkasdisabled== 0 and self.lkasbuttonprev == 0:
      #  self.lkasdisabled = 1
      #elif self.lkasbutton ==1 and self.lkasdisabled== 1 and self.lkasbuttonprev == 0:
      #  self.lkasdisabled = 0
      self.lkasbuttonprev = self.lkasbutton
      ret.cruiseState.enabled = cp.vl["DAS_3"]["ACC_Engaged"] == 1 # and self.lkasdisabled == 0 # ACC is green.
      ret.cruiseState.standstill = cp.vl["DAS_3"]["ACC_StandStill"] == 1
      ret.cruiseState.speed = cp.vl["DAS_4"]["ACC_Set_Speed"] * CV.KPH_TO_MS
      # ACC_Activation_Status is a three bit msg, 0 is off, 1 and 2 are Non-ACC mode, 3 and 4 are ACC mode
      ret.cruiseState.available = cp.vl["DAS_4"]['ACC_Activation_Status'] in [3, 4]  #3 ACCOn and 4 ACCSet
      ret.cruiseState.nonAdaptive = cp.vl["DAS_4"]["ACC_Activation_Status"] in (1, 2) #1 NormalCCOn and 2 NormalCCSet
      #ret.cruiseState.speedOffset = ret.cruiseState.speed - ret.vEgo
      self.dashboard = cp.vl["DAS_4"]  
      self.steer_state = cp.vl["EPS_2"]["Torque_Overlay_Status"]
      ret.steerFaultPermanent  = self.steer_state == 4 or (self.steer_state == 0 and ret.vEgo > self.CP.minSteerSpeed)
      
    if self.CP.carFingerprint in (CAR.RAM_1500, CAR.RAM_2500):
      self.lkasbutton = (cp.vl["Center_Stack_2"]["LKAS_Button"] == 1) or (cp.vl["Center_Stack_1"]["LKAS_Button"] == 1)
      if self.lkasbutton ==1 and self.lkasdisabled== 0 and self.lkasbuttonprev == 0:
        self.lkasdisabled = 1
      elif self.lkasbutton ==1 and self.lkasdisabled== 1 and self.lkasbuttonprev == 0:
        self.lkasdisabled = 0
      self.lkasbuttonprev = self.lkasbutton
      ret.cruiseState.enabled = cp_cam.vl["DAS_3"]["ACC_Engaged"] == 1  and self.lkasdisabled == 0 # ACC is green.
      ret.cruiseState.standstill = cp_cam.vl["DAS_3"]["ACC_StandStill"] == 1
      ret.cruiseState.speed = cp_cam.vl["DAS_4"]["ACC_Set_Speed"] * CV.KPH_TO_MS
      # ACC_Activation_Status is a three bit msg, 0 is off, 1 and 2 are Non-ACC mode, 3 and 4 are ACC mode
      ret.cruiseState.available = cp_cam.vl["DAS_4"]['ACC_Activation_Status'] in [3, 4]  #3 ACCOn and 4 ACCSet
      ret.cruiseState.nonAdaptive = cp_cam.vl["DAS_4"]["ACC_Activation_Status"] in [1, 2] #1 NormalCCOn and 2 NormalCCSet
      self.autoHighBeamBit = cp_cam.vl["DAS_6"]['Auto_High_Beam'] #Auto High Beam isn't Located in this message on chrysler or jeep currently located in 729 message
      #ret.cruiseState.speedOffset = ret.cruiseState.speed - ret.vEgo
      self.dashboard = cp_cam.vl["DAS_4"]
      ret.steerFaultTemporary  = cp.vl["EPS_3"]["DASM_FAULT"]==1


  # blindspot sensors
    if self.CP.enableBsm:
      ret.leftBlindspot = cp.vl["BSM_1"]["Blind_Spot_Monitor_Left"] == 1
      ret.rightBlindspot = cp.vl["BSM_1"]["Blind_Spot_Monitor_Right"] == 1    

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
      ("Gear_State", "Transmission_Status"), #Gear Position
      ("Vehicle_Speed", "ESP_8"),#Vehicle Speed
      ("Acceleration", "ESP_4"),#Acceleration Rate
      ("Yaw_Rate", "ESP_4"),#Yaw Rate
      ("Wheel_RPM_Front_Left", "ESP_6"),#FL Wheel Speed
      ("Wheel_RPM_Front_Right", "ESP_6"),#FR Wheel Speed
      ("Wheel_RPM_Rear_Left", "ESP_6"),#RL Wheel Speed
      ("Wheel_RPM_Rear_Right", "ESP_6"),#RR Wheel Speed
      ("Accelerator_Position", "ECM_5"), #Accelerator Position
      ("Brake_Pedal_State", "ESP_1"),#Brake Pedal Pressed
      ("Steering_Wheel_Angle", "Steering_Column_Angle_Status"),#Steering Angle
      ("Steering_Rate", "Steering_Column_Angle_Status"),#Steering rate
      ("Steering_Column_Torque", "EPS_2"),#EPS Driver applied torque
      ("EPS_Motor_Torque", "EPS_2"),#EPS Motor Torque output
      ("Torque_Overlay_Status", "EPS_2"),
      ("Traction_Button", "Center_Stack_1"),#Traction Control Button
      ("LKAS_Button", "Center_Stack_1"),#LKAS Button
      ("Turn_Signal_Status", "Steering_Column_Commands"),#Blinker 
      ("High_Beam_Lever_Status", "Steering_Column_Commands"),#High Beam Lever
      ("ACC_Accel", "Cruise_Control_Buttons"),#ACC Accel Button
      ("ACC_Decel", "Cruise_Control_Buttons"),#ACC Decel Button
      ("ACC_Cancel", "Cruise_Control_Buttons"),#ACC Cancel Button
      ("ACC_Distance_Dec", "Cruise_Control_Buttons"),#ACC Distance Decrement Button
      ("ACC_Distance_Inc", "Cruise_Control_Buttons"),#ACC Distance Increment Button
      ("ACC_Resume", "Cruise_Control_Buttons"),#ACC Resume Button
      ("Cruise_OnOff", "Cruise_Control_Buttons"),#Cruise On Off Button
      ("ACC_OnOff", "Cruise_Control_Buttons"),#ACC On Off Button
      ("COUNTER", "Cruise_Control_Buttons"),#ACC Counter Button
      ("ACC_Distance_Inc", "Cruise_Control_Buttons"),#ACC Distance Increase Button
      ("Driver_Door_Ajar", "BCM_1"),#driver Door
      ("Passenger_Door_Ajar", "BCM_1"),#Passenger Door
      ("Left_Rear_Door_Ajar", "BCM_1"),#Driver Rear Door
      ("Right_Rear_Door_Ajar", "BCM_1"),#Passenger Rear Door
      ("Driver_Seatbelt_Status", "ORC_1"), #Driver Sear Belt
      ("COUNTER", "EPS_2"),#EPS Counter  
      ("DASM_FAULT", "EPS_3"),#EPS Fault  
    ]

    checks = [
      # sig_address, frequency
      ("Transmission_Status", 50),
      ("ESP_1", 50),
      ("ESP_4", 50),
      ("ESP_6", 50),
      ("ESP_8", 50),
      ("ECM_5", 50),
      ("Steering_Column_Angle_Status", 100),
      ("EPS_2", 100),
      ("EPS_3", 100),
      ("Center_Stack_1", 1),
      ("Steering_Column_Commands", 10),
      ("Cruise_Control_Buttons", 50),
      ("BCM_1", 1),
      ("ORC_1", 1),
    ]

    if CP.enableBsm:
      signals += [
        ("Blind_Spot_Monitor_Left", "BSM_1"),
        ("Blind_Spot_Monitor_Right", "BSM_1"),
      ]
      checks += [("BSM_1", 2)]

    if CP.carFingerprint in (CAR.PACIFICA_2017_HYBRID, CAR.PACIFICA_2018_HYBRID, CAR.PACIFICA_2019_HYBRID, CAR.PACIFICA_2018, CAR.PACIFICA_2020, CAR.JEEP_CHEROKEE_2019, CAR.JEEP_CHEROKEE):
      signals += [
        ("ACC_Engaged", "DAS_3"),#ACC Engaged
        ("ACC_StandStill", "DAS_3"),#ACC Engaged
        ("ACC_Set_Speed", "DAS_4"),
        ("ACC_Activation_Status", "DAS_4"),
      ]
      checks += [
        ("DAS_3", 50),
        ("DAS_4", 50),
        ]

    if CP.carFingerprint in (CAR.RAM_1500, CAR.RAM_2500):
      signals += [
        ("LKAS_Button", "Center_Stack_2"),#LKAS Button
      ]

      checks += [
        ("Center_Stack_2", 1),
        ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 0)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
      ("LKAS_LANE_LINES", "DAS_6"),
      ("LKAS_ICON_COLOR", "DAS_6"),
      ("LKAS_Disabled", "DAS_6"),
      ("CAR_MODEL", "DAS_6"),
      ("LKAS_ALERTS", "DAS_6"),
    ]
    checks = [
      ("DAS_6", 15),
    ]

    if CP.carFingerprint in (CAR.RAM_1500, CAR.RAM_2500):
      signals += [
        ("ACC_Engaged", "DAS_3"),#ACC Engaged
        ("ACC_StandStill", "DAS_3"),#ACC Engaged
        ("COUNTER", "DAS_3"),
        ("ACC_Set_Speed", "DAS_4"),
        ("ACC_Activation_Status", "DAS_4"),
        ("Auto_High_Beam", "DAS_6"),
      ]
      checks += [
        ("DAS_3", 50),
        ("DAS_4", 50),
        ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)
