from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Union

from cereal import car
from selfdrive.car import AngleRateLimit, dbc_dict
from selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarInfo, CarParts, Column, \
                                           Device
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu


class CarControllerParams:
  STEER_STEP = 5        # LateralMotionControl, 20Hz
  LKA_STEP = 3          # Lane_Assist_Data1, 33Hz
  ACC_CONTROL_STEP = 2  # ACCDATA, 50Hz
  LKAS_UI_STEP = 100    # IPMA_Data, 1Hz
  ACC_UI_STEP = 20      # ACCDATA_3, 5Hz
  BUTTONS_STEP = 5      # Steering_Data_FD1, 10Hz, but send twice as fast

  CURVATURE_MAX = 0.02  # Max curvature for steering command, m^-1
  STEER_DRIVER_ALLOWANCE = 1.0  # Driver intervention threshold, Nm

  # Curvature rate limits
  # The curvature signal is limited to 0.003 to 0.009 m^-1/sec by the EPS depending on speed and direction
  # Limit to ~2 m/s^3 up, ~3 m/s^3 down at 75 mph
  # Worst case, the low speed limits will allow 4.3 m/s^3 up, 4.9 m/s^3 down at 75 mph
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.0002, 0.0001])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.000225, 0.00015])
  CURVATURE_ERROR = 0.002  # ~6 degrees at 10 m/s, ~10 degrees at 35 m/s

  ACCEL_MAX = 2.0               # m/s^2 max acceleration
  ACCEL_MIN = -3.5              # m/s^2 max deceleration
  MIN_GAS = -0.5
  INACTIVE_GAS = -5.0

  def __init__(self, CP):
    pass


class CAR:
  BRONCO_SPORT_MK1 = "FORD BRONCO SPORT 1ST GEN"
  ESCAPE_MK4 = "FORD ESCAPE 4TH GEN"
  EXPLORER_MK6 = "FORD EXPLORER 6TH GEN"
  F_150_MK14 = "FORD F-150 14TH GEN"
  FOCUS_MK4 = "FORD FOCUS 4TH GEN"
  MAVERICK_MK1 = "FORD MAVERICK 1ST GEN"


CANFD_CARS = {CAR.F_150_MK14}


class RADAR:
  DELPHI_ESR = 'ford_fusion_2018_adas'
  DELPHI_MRR = 'FORD_CADS'


DBC: Dict[str, Dict[str, str]] = defaultdict(lambda: dbc_dict("ford_lincoln_base_pt", RADAR.DELPHI_MRR))

DBC[CAR.F_150_MK14] = dbc_dict("ford_lincoln_base_pt", None)


class Footnote(Enum):
  FOCUS = CarFootnote(
    "Refers only to the Focus Mk4 (C519) available in Europe/China/Taiwan/Australasia, not the Focus Mk3 (C346) in " +
    "North and South America/Southeast Asia.",
    Column.MODEL,
  )


@dataclass
class FordCarInfo(CarInfo):
  package: str = "Co-Pilot360 Assist+"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.ford_q3]))

  def init_make(self, CP: car.CarParams):
    harness = CarHarness.ford_q3
    if CP.carFingerprint in CANFD_CARS:
      harness = CarHarness.ford_q4

    if CP.carFingerprint in (CAR.BRONCO_SPORT_MK1, CAR.F_150_MK14, CAR.MAVERICK_MK1):
      self.car_parts = CarParts([Device.three_angled_mount, harness])
    else:
      self.car_parts = CarParts.common([harness])


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {
  CAR.BRONCO_SPORT_MK1: FordCarInfo("Ford Bronco Sport 2021-22"),
  CAR.ESCAPE_MK4: [
    FordCarInfo("Ford Escape 2020-22"),
    FordCarInfo("Ford Kuga 2020-22", "Adaptive Cruise Control with Lane Centering"),
  ],
  CAR.EXPLORER_MK6: [
    FordCarInfo("Ford Explorer 2020-22"),
    FordCarInfo("Lincoln Aviator 2020-21", "Co-Pilot360 Plus"),
  ],
  CAR.F_150_MK14: FordCarInfo("Ford F-150 2023", "Co-Pilot360 Active 2.0"),
  CAR.FOCUS_MK4: FordCarInfo("Ford Focus 2018", "Adaptive Cruise Control with Lane Centering", footnotes=[Footnote.FOCUS]),
  CAR.MAVERICK_MK1: FordCarInfo("Ford Maverick 2022-23", "Co-Pilot360 Assist"),
}

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
      auxiliary=True,
    ),
  ],
  extra_ecus=[
    (Ecu.debug, 0x703, None),             # All Wheel Drive (AWD)
    # 0x706                                 Image Processing Module A (IPMA)
    (Ecu.debug, 0x714, None),             # Heated Steering Wheel Module (HSWM)
    (Ecu.gateway, 0x716, None),           # Gateway Module A (GWM)
    (Ecu.combinationMeter, 0x720, None),  # Instrument Panel Cluster (IPC)
    (Ecu.debug, 0x721, None),             # Vehicle Dynamics Control Module (VDM)
    (Ecu.debug, 0x724, None),             # Steering Column Control Module (SCCM)
    (Ecu.debug, 0x725, None),             # Wireless Accessory Charging Module (WACM)
    (Ecu.body, 0x726, None),              # Body Control Module (BCM)
    (Ecu.debug, 0x727, None),             # Audio Control Module (ACM)
    # 0x730                                 Power Steering Control Module (PSCM)
    (Ecu.debug, 0x731, None),             # Remote Function Actuator (RFA)
    (Ecu.shiftByWire, 0x732, None),       # Gear Shift Module (GSM)
    (Ecu.hvac, 0x733, None),              # HVAC
    (Ecu.debug, 0x734, None),             # Headlamp Control Module (HCM)
    (Ecu.parkingAdas, 0x736, None),       # Parking Aid Module (PAM)
    (Ecu.srs, 0x737, None),               # Restraints Control Module (RCM)
    (Ecu.debug, 0x740, None),             # Driver Door Module (DDM)
    (Ecu.debug, 0x741, None),             # Passenger Door Module (PDM)
    (Ecu.debug, 0x744, None),             # Driver Seat Module (DSM)
    (Ecu.debug, 0x746, None),             # Direct Current/Direct Current Converter Control Module (DCDC)
    (Ecu.debug, 0x751, None),             # Radio Transceiver Module (RTM)
    (Ecu.telematics, 0x754, None),        # Telematic Control Unit Module (TCU)
    # 0x760                                 Anti-lock Brake System (ABS)
    # 0x764                                 Cruise Control Module (CCM)
    (Ecu.debug, 0x765, None),             # Occupant Classification System (OCS)
    (Ecu.debug, 0x775, None),             # Rear Gate Trunk Module (RGTM)
    (Ecu.debug, 0x776, None),             # Front Seat Climate Control Module (SCME)
    (Ecu.debug, 0x783, None),             # Audio Sigital Signal Processing Module (DSP)
    (Ecu.debug, 0x791, None),             # Trailer Module (TRM)
    (Ecu.debug, 0x7A0, None),             # Front Controls Interface Module B (FCIMB)
    (Ecu.debug, 0x7B1, None),             # Image Processing Module B (IPMB)
    (Ecu.cornerRadar, 0x7C4, None),       # Side Obstacle Detection Control Module LH (SODL)
    (Ecu.cornerRadar, 0x7C6, None),       # Side Obstacle Detection Control Module RH (SODR)
    (Ecu.debug, 0x7D0, None),             # Accessory Protocol Interface Module (APIM)
    (Ecu.debug, 0x7D5, None),             # Digital Audio Control Module C (DACMC)
    # 0x7E0                                 Powertrain Control Module (PCM)
  ],
)

FW_VERSIONS = {
  CAR.BRONCO_SPORT_MK1: {
    (Ecu.eps, 0x730, None): [
      b'LX6C-14D003-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6C-14D003-AK\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x760, None): [
      b'LX6C-2D053-RD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6C-2D053-RE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'M1PT-14F397-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'M1PA-14C204-GF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'N1PA-14C204-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'N1PA-14C204-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.ESCAPE_MK4: {
    (Ecu.eps, 0x730, None): [
      b'LX6C-14D003-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6C-14D003-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6C-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x760, None): [
      b'LX6C-2D053-NS\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6C-2D053-NT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6C-2D053-NY\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6C-2D053-SA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'LJ6T-14F397-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LJ6T-14F397-AE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'LX6A-14C204-BJV\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6A-14C204-BJX\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6A-14C204-CNG\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LX6A-14C204-ESG\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'MX6A-14C204-BEF\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'MX6A-14C204-BEJ\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'MX6A-14C204-CAB\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'NX6A-14C204-BLE\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.EXPLORER_MK6: {
    (Ecu.eps, 0x730, None): [
      b'L1MC-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-14D003-AK\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'M1MC-14D003-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'M1MC-14D003-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x760, None): [
      b'L1MC-2D053-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-2D053-BA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-2D053-BB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-2D053-BF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-2D053-KB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'LB5T-14F397-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LB5T-14F397-AE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LB5T-14F397-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LC5T-14F397-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'LB5A-14C204-ATJ\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LB5A-14C204-AZL\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LB5A-14C204-BUJ\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LB5A-14C204-EAC\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'MB5A-14C204-MD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'MB5A-14C204-RC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'NB5A-14C204-AZD\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'NB5A-14C204-HB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.F_150_MK14: {
    (Ecu.eps, 0x730, None): [
      b'ML3V-14D003-BC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x760, None): [
      b'PL34-2D053-CA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'ML3T-14D049-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'PJ6T-14H102-ABJ\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'PL3A-14C204-BRB\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.FOCUS_MK4: {
    (Ecu.eps, 0x730, None): [
      b'JX6C-14D003-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x760, None): [
      b'JX61-2D053-CJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'JX7T-14D049-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'JX7T-14F397-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'JX6A-14C204-BPL\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.MAVERICK_MK1: {
    (Ecu.eps, 0x730, None): [
      b'NZ6C-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x760, None): [
      b'NZ6C-2D053-AG\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'PZ6C-2D053-ED\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'NZ6T-14D049-AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'NZ6T-14F397-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'NZ6A-14C204-AAA\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'NZ6A-14C204-PA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'NZ6A-14C204-ZA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'PZ6A-14C204-JC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
}
