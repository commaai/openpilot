from selfdrive.car import dbc_dict

class CarControllerParams:
  HCA_STEP = 2                   # HCA_01 message frequency 50Hz
  LDW_STEP = 10                  # LDW_02 message frequency 10Hz
  GRA_ACC_STEP = 3               # GRA_ACC_01 message frequency 33Hz

  GRA_VBP_STEP = 100             # Send ACC virtual button presses once a second
  GRA_VBP_COUNT = 16             # Send VBP messages for ~0.5s (GRA_ACC_STEP * 16)

  # Observed documented MQB limits: 3.00 Nm max, rate of change 5.00 Nm/sec.
  # Limiting rate-of-change based on real-world testing and Comma's safety
  # requirements for minimum time to lane departure.
  STEER_MAX = 300                # Max heading control assist torque 3.00 Nm
  STEER_DELTA_UP = 4             # Max HCA reached in 1.50s (STEER_MAX / (50Hz * 1.50))
  STEER_DELTA_DOWN = 10          # Min HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))
  STEER_DRIVER_ALLOWANCE = 80
  STEER_DRIVER_MULTIPLIER = 3    # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1        # from dbc

class CANBUS:
  pt = 0
  cam = 2

BUTTON_STATES = {
  "accelCruise": False,
  "decelCruise": False,
  "cancel": False,
  "setCruise": False,
  "resumeCruise": False,
  "gapAdjustCruise": False
}

MQB_LDW_MESSAGES = {
  "none": 0,                            # Nothing to display
  "laneAssistUnavailChime": 1,          # "Lane Assist currently not available." with chime
  "laneAssistUnavailNoSensorChime": 3,  # "Lane Assist not available. No sensor view." with chime
  "laneAssistTakeOverUrgent": 4,        # "Lane Assist: Please Take Over Steering" with urgent beep
  "emergencyAssistUrgent": 6,           # "Emergency Assist: Please Take Over Steering" with urgent beep
  "laneAssistTakeOverChime": 7,         # "Lane Assist: Please Take Over Steering" with chime
  "laneAssistTakeOverSilent": 8,        # "Lane Assist: Please Take Over Steering" silent
  "emergencyAssistChangingLanes": 9,    # "Emergency Assist: Changing lanes..." with urgent beep
  "laneAssistDeactivated": 10,          # "Lane Assist deactivated." silent with persistent icon afterward
}

class CAR:
  VW_ATLAS_MK1 = "Volkswagen Atlas 1st Gen"
  VW_GOLF_R_MK7 = "Volkswagen Golf R 7th Gen"
  VW_TIGUAN_MK2 = "Volkswagen Tiguan 2nd Gen"

MQB_CARS = {
  CAR.VW_ATLAS_MK1,
  CAR.VW_GOLF_R_MK7,
  CAR.VW_TIGUAN_MK2
}

# Volkswagen port using FP 2.0 exclusively
FINGERPRINTS = {}

FW_VERSIONS = {
  CAR.VW_ATLAS_MK1: {
    # Mk1 2018-2020 and Mk1.5 facelift 2021
    (Ecu.eps, 0x712, None): [b'\x0571B60924A1'],  # 2019 Atlas
  },
  CAR.VW_GOLF_R_MK7: {
    # Mk7 2013-2017 and Mk7.5 facelift 2018-2020
    (Ecu.eps, 0x712, None): [b'\x0571A0JA15A1'],  # 2018 Golf R
  },
  CAR.VW_TIGUAN_MK2: {
    # Mk2 2018-2020
    (Ecu.eps, 0x712, None): [b'\x0521A60804A1'],  # 2020 Tiguan SEL Premium R-Line
  },
}

DBC = {
  CAR.VW_ATLAS_MK1: dbc_dict('vw_mqb_2010', None),
  CAR.VW_GOLF_R_MK7: dbc_dict('vw_mqb_2010', None),
  CAR.VW_TIGUAN_MK2: dbc_dict('vw_mqb_2010', None),
}
