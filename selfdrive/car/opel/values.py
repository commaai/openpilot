# Opel Corsa F (PSA CMP Platform) support for FlowPilot
# Based on PSA/Stellantis CAN protocol
from collections import defaultdict
from typing import Dict

from cereal import car
from selfdrive.car import dbc_dict

Ecu = car.CarParams.Ecu
NetworkLocation = car.CarParams.NetworkLocation
TransmissionType = car.CarParams.TransmissionType
GearShifter = car.CarState.GearShifter


class CarControllerParams:
  LKAS_STEP = 2                   # LKAS steering message frequency 50Hz
  LDW_STEP = 10                   # LDW_02 message frequency 10Hz
  ACC_STEP = 3                    # ACC control message frequency 33Hz

  ACC_VBP_STEP = 100              # Send ACC virtual button presses once a second
  ACC_VBP_COUNT = 16              # Send VBP messages for ~0.5s

  # PSA EPS torque limits - Corsa F specific
  STEER_MAX = 300                 # Max steering assist torque 3.00 Nm
  STEER_DELTA_UP = 4              # Max torque ramp up rate
  STEER_DELTA_DOWN = 10           # Max torque ramp down rate
  STEER_ERROR_MAX = 80            # Max steering err
  STEER_DRIVER_ALLOWANCE = 80
  STEER_DRIVER_MULTIPLIER = 3     # Weight driver torque heavily
  STEER_DRIVER_FACTOR = 1         # From DBC


class CANBUS:
  pt = 0       # Powertrain CAN bus
  cam = 2      # Camera CAN bus


class DBC_FILES:
  # PSA CMP platform DBC file
  # NOTE: This DBC file needs to be added to the opendbc repository
  # For now using a placeholder name - replace with actual DBC filename
  psa_cmp = "psa_cmp_2019"


# DBC dictionary mapping car models to DBC files
DBC = defaultdict(lambda: dbc_dict(DBC_FILES.psa_cmp, None))  # type: Dict[str, Dict[str, str]]


BUTTON_STATES = {
  "accelCruise": False,
  "decelCruise": False,
  "cancel": False,
  "setCruise": False,
  "resumeCruise": False,
  "gapAdjustCruise": False
}

# PSA LDW/LKAS HUD messages
PSA_LDW_MESSAGES = {
  "none": 0,                             # Nothing to display
  "laneAssistUnavail": 1,                # Lane Assist not available
  "laneAssistTakeOver": 2,               # Lane Assist: Please Take Over
  "laneAssistActive": 3,                 # Lane Assist active
  "laneAssistDeactivated": 4,            # Lane Assist deactivated
  "emergencyAssist": 5,                  # Emergency Assist active
}


class CAR:
  CORSA_F = "OPEL CORSA 6TH GEN 2020+"         # Chassis F/CMP, Mk6 Opel/Vauxhall Corsa (2020+)
  PEUGEOT_208 = "PEUGEOT 208 2ND GEN"    # Chassis CMP, Peugeot 208 (2019+)
  PEUGEOT_2008 = "PEUGEOT 2008 2ND GEN"  # Chassis CMP, Peugeot 2008 (2019+)


# CAN bus fingerprints for vehicle identification
# These are the same for PSA CMP platform vehicles
FINGERPRINTS = {
  CAR.CORSA_F: [{
    0x0B6: 8, 0x0DA: 8, 0x0E2: 8, 0x0F6: 8, 0x128: 8, 0x12C: 8, 0x131: 8,
    0x161: 8, 0x1A8: 8, 0x1E9: 8, 0x217: 8, 0x221: 8, 0x23A: 8, 0x260: 8,
    0x2B0: 8, 0x36C: 8,
  }],
  CAR.PEUGEOT_208: [{
    0x0B6: 8, 0x0DA: 8, 0x0E2: 8, 0x0F6: 8, 0x128: 8, 0x12C: 8, 0x131: 8,
    0x161: 8, 0x1A8: 8, 0x1E9: 8, 0x217: 8, 0x221: 8, 0x23A: 8, 0x260: 8,
    0x2B0: 8, 0x36C: 8,
  }],
  CAR.PEUGEOT_2008: [{
    0x0B6: 8, 0x0DA: 8, 0x0E2: 8, 0x0F6: 8, 0x128: 8, 0x12C: 8, 0x131: 8,
    0x161: 8, 0x1A8: 8, 0x1E9: 8, 0x217: 8, 0x221: 8, 0x23A: 8, 0x260: 8,
    0x2B0: 8, 0x36C: 8,
  }],
}


# Firmware versions for FW-based vehicle identification
FW_VERSIONS = {
  CAR.CORSA_F: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x879836431280\xf1\x890001',
      b'\xf1\x879836431380\xf1\x890002',
    ],
    (Ecu.transmission, 0x7e8, None): [
      b'\xf1\x879836431480\xf1\x890001',
    ],
    (Ecu.eps, 0x712, None): [
      b'\xf1\x879836431580\xf1\x890001',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'\xf1\x879836431680\xf1\x890001',
    ],
  },
  CAR.PEUGEOT_208: {
    (Ecu.eps, 0x712, None): [
      b'\xf1\x879812345678\xf1\x890001', # Placeholder
    ],
  },
}
