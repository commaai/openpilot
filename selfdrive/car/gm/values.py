from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Dict, List, Union

from cereal import car
from openpilot.selfdrive.car import dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarInfo, CarParts, Column
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu


class CarControllerParams:
  STEER_MAX = 300  # GM limit is 3Nm. Used by carcontroller to generate LKA output
  STEER_STEP = 3  # Active control frames per command (~33hz)
  INACTIVE_STEER_STEP = 10  # Inactive control frames per command (10hz)
  STEER_DELTA_UP = 10  # Delta rates require review due to observed EPS weakness
  STEER_DELTA_DOWN = 15
  STEER_DRIVER_ALLOWANCE = 65
  STEER_DRIVER_MULTIPLIER = 4
  STEER_DRIVER_FACTOR = 100
  NEAR_STOP_BRAKE_PHASE = 0.5  # m/s

  # Heartbeat for dash "Service Adaptive Cruise" and "Service Front Camera"
  ADAS_KEEPALIVE_STEP = 100
  CAMERA_KEEPALIVE_STEP = 100

  # Allow small margin below -3.5 m/s^2 from ISO 15622:2018 since we
  # perform the closed loop control, and might need some
  # to apply some more braking if we're on a downhill slope.
  # Our controller should still keep the 2 second average above
  # -3.5 m/s^2 as per planner limits
  ACCEL_MAX = 2.  # m/s^2
  ACCEL_MIN = -4.  # m/s^2

  def __init__(self, CP):
    # Gas/brake lookups
    self.ZERO_GAS = 2048  # Coasting
    self.MAX_BRAKE = 400  # ~ -4.0 m/s^2 with regen

    if CP.carFingerprint in CAMERA_ACC_CAR:
      self.MAX_GAS = 3400
      self.MAX_ACC_REGEN = 1514
      self.INACTIVE_REGEN = 1554
      # Camera ACC vehicles have no regen while enabled.
      # Camera transitions to MAX_ACC_REGEN from ZERO_GAS and uses friction brakes instantly
      max_regen_acceleration = 0.

    else:
      self.MAX_GAS = 3072  # Safety limit, not ACC max. Stock ACC >4096 from standstill.
      self.MAX_ACC_REGEN = 1404  # Max ACC regen is slightly less than max paddle regen
      self.INACTIVE_REGEN = 1404
      # ICE has much less engine braking force compared to regen in EVs,
      # lower threshold removes some braking deadzone
      max_regen_acceleration = -1. if CP.carFingerprint in EV_CAR else -0.1

    self.GAS_LOOKUP_BP = [max_regen_acceleration, 0., self.ACCEL_MAX]
    self.GAS_LOOKUP_V = [self.MAX_ACC_REGEN, self.ZERO_GAS, self.MAX_GAS]

    self.BRAKE_LOOKUP_BP = [self.ACCEL_MIN, max_regen_acceleration]
    self.BRAKE_LOOKUP_V = [self.MAX_BRAKE, 0.]


class CAR(StrEnum):
  HOLDEN_ASTRA = "HOLDEN ASTRA RS-V BK 2017"
  VOLT = "CHEVROLET VOLT PREMIER 2017"
  CADILLAC_ATS = "CADILLAC ATS Premium Performance 2018"
  MALIBU = "CHEVROLET MALIBU PREMIER 2017"
  ACADIA = "GMC ACADIA DENALI 2018"
  BUICK_LACROSSE = "BUICK LACROSSE 2017"
  BUICK_REGAL = "BUICK REGAL ESSENCE 2018"
  ESCALADE = "CADILLAC ESCALADE 2017"
  ESCALADE_ESV = "CADILLAC ESCALADE ESV 2016"
  ESCALADE_ESV_2019 = "CADILLAC ESCALADE ESV 2019"
  BOLT_EUV = "CHEVROLET BOLT EUV 2022"
  SILVERADO = "CHEVROLET SILVERADO 1500 2020"
  EQUINOX = "CHEVROLET EQUINOX 2019"
  TRAILBLAZER = "CHEVROLET TRAILBLAZER 2021"


class Footnote(Enum):
  OBD_II = CarFootnote(
    'Requires a <a href="https://github.com/commaai/openpilot/wiki/GM#hardware" target="_blank">community built ASCM harness</a>. ' +
    '<b><i>NOTE: disconnecting the ASCM disables Automatic Emergency Braking (AEB).</i></b>',
    Column.MODEL)


@dataclass
class GMCarInfo(CarInfo):
  package: str = "Adaptive Cruise Control (ACC)"

  def init_make(self, CP: car.CarParams):
    if CP.networkLocation == car.CarParams.NetworkLocation.fwdCamera:
      self.car_parts = CarParts.common([CarHarness.gm])
    else:
      self.car_parts = CarParts.common([CarHarness.obd_ii])
      self.footnotes.append(Footnote.OBD_II)


CAR_INFO: Dict[str, Union[GMCarInfo, List[GMCarInfo]]] = {
  CAR.HOLDEN_ASTRA: GMCarInfo("Holden Astra 2017"),
  CAR.VOLT: GMCarInfo("Chevrolet Volt 2017-18", min_enable_speed=0, video_link="https://youtu.be/QeMCN_4TFfQ"),
  CAR.CADILLAC_ATS: GMCarInfo("Cadillac ATS Premium Performance 2018"),
  CAR.MALIBU: GMCarInfo("Chevrolet Malibu Premier 2017"),
  CAR.ACADIA: GMCarInfo("GMC Acadia 2018", video_link="https://www.youtube.com/watch?v=0ZN6DdsBUZo"),
  CAR.BUICK_LACROSSE: GMCarInfo("Buick LaCrosse 2017-19", "Driver Confidence Package 2"),
  CAR.BUICK_REGAL: GMCarInfo("Buick Regal Essence 2018"),
  CAR.ESCALADE: GMCarInfo("Cadillac Escalade 2017", "Driver Assist Package"),
  CAR.ESCALADE_ESV: GMCarInfo("Cadillac Escalade ESV 2016", "Adaptive Cruise Control (ACC) & LKAS"),
  CAR.ESCALADE_ESV_2019: GMCarInfo("Cadillac Escalade ESV 2019", "Adaptive Cruise Control (ACC) & LKAS"),
  CAR.BOLT_EUV: [
    GMCarInfo("Chevrolet Bolt EUV 2022-23", "Premier or Premier Redline Trim without Super Cruise Package", video_link="https://youtu.be/xvwzGMUA210"),
    GMCarInfo("Chevrolet Bolt EV 2022-23", "2LT Trim with Adaptive Cruise Control Package"),
  ],
  CAR.SILVERADO: [
    GMCarInfo("Chevrolet Silverado 1500 2020-21", "Safety Package II"),
    GMCarInfo("GMC Sierra 1500 2020-21", "Driver Alert Package II", video_link="https://youtu.be/5HbNoBLzRwE"),
  ],
  CAR.EQUINOX: GMCarInfo("Chevrolet Equinox 2019-22"),
  CAR.TRAILBLAZER: GMCarInfo("Chevrolet Trailblazer 2021-22"),
}


class CruiseButtons:
  INIT = 0
  UNPRESS = 1
  RES_ACCEL = 2
  DECEL_SET = 3
  MAIN = 5
  CANCEL = 6

class AccState:
  OFF = 0
  ACTIVE = 1
  FAULTED = 3
  STANDSTILL = 4

class CanBus:
  POWERTRAIN = 0
  OBSTACLE = 1
  CAMERA = 2
  CHASSIS = 2
  LOOPBACK = 128
  DROPPED = 192


# In a Data Module, an identifier is a string used to recognize an object,
# either by itself or together with the identifiers of parent objects.
# Each returns a 4 byte hex representation of the decimal part number. `b"\x02\x8c\xf0'"` -> 42790951
GM_SOFTWARE_MODULE_1_REQUEST = b'\x1a\xc1'
GM_SOFTWARE_MODULE_2_REQUEST = b'\x1a\xc2'
GM_SOFTWARE_MODULE_3_REQUEST = b'\x1a\xc3'
# This DID is for identifying the part number that reflects the mix of hardware,
# software, and calibrations in the ECU when it first arrives at the vehicle assembly plant.
# If there's an Alpha Code, it's associated with this part number and stored in the DID $DB.
GM_END_MODEL_PART_NUMBER_REQUEST = b'\x1a\xcb'
GM_BASE_MODEL_PART_NUMBER_REQUEST = b'\x1a\xcc'
GM_FW_RESPONSE = b'\x5a'

GM_FW_REQUESTS = [
  GM_SOFTWARE_MODULE_1_REQUEST,
  GM_SOFTWARE_MODULE_2_REQUEST,
  GM_SOFTWARE_MODULE_3_REQUEST,
  GM_END_MODEL_PART_NUMBER_REQUEST,
  GM_BASE_MODEL_PART_NUMBER_REQUEST,
]

GM_RX_OFFSET = 0x400

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[request for req in GM_FW_REQUESTS for request in [
    Request(
      [StdQueries.SHORT_TESTER_PRESENT_REQUEST, req],
      [StdQueries.SHORT_TESTER_PRESENT_RESPONSE, GM_FW_RESPONSE + bytes([req[-1]])],
      rx_offset=GM_RX_OFFSET,
      bus=0,
      logging=True,
    ),
  ]],
  extra_ecus=[(Ecu.fwdCamera, 0x24b, None)],
)

DBC: Dict[str, Dict[str, str]] = defaultdict(lambda: dbc_dict('gm_global_a_powertrain_generated', 'gm_global_a_object', chassis_dbc='gm_global_a_chassis'))

EV_CAR = {CAR.VOLT, CAR.BOLT_EUV}

# We're integrated at the camera with VOACC on these cars (instead of ASCM w/ OBD-II harness)
CAMERA_ACC_CAR = {CAR.BOLT_EUV, CAR.SILVERADO, CAR.EQUINOX, CAR.TRAILBLAZER}

STEER_THRESHOLD = 1.0
