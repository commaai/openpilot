from dataclasses import dataclass, field
from enum import Enum, IntFlag

from opendbc.car import Bus, PlatformConfig, DbcDict, Platforms, CarSpecs
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import CarDocs, CarFootnote, CarHarness, CarParts, Column
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = CarParams.Ecu


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
    self.MAX_BRAKE = 400  # ~ -4.0 m/s^2 with regen

    if CP.carFingerprint in (CAMERA_ACC_CAR | SDGM_CAR):
      self.MAX_GAS = 1346.0
      self.MAX_ACC_REGEN = -540.0
      self.INACTIVE_REGEN = -500.0
      # Camera ACC vehicles have no regen while enabled.
      # Camera transitions to MAX_ACC_REGEN from zero gas and uses friction brakes instantly
      max_regen_acceleration = 0.

    else:
      self.MAX_GAS = 1018.0  # Safety limit, not ACC max. Stock ACC >2042 from standstill.
      self.MAX_ACC_REGEN = -650.0  # Max ACC regen is slightly less than max paddle regen
      self.INACTIVE_REGEN = -650.0
      # ICE has much less engine braking force compared to regen in EVs,
      # lower threshold removes some braking deadzone
      max_regen_acceleration = -1. if CP.carFingerprint in EV_CAR else -0.1

    self.GAS_LOOKUP_BP = [max_regen_acceleration, 0., self.ACCEL_MAX]
    self.GAS_LOOKUP_V = [self.MAX_ACC_REGEN, 0., self.MAX_GAS]

    self.BRAKE_LOOKUP_BP = [self.ACCEL_MIN, max_regen_acceleration]
    self.BRAKE_LOOKUP_V = [self.MAX_BRAKE, 0.]


class GMSafetyFlags(IntFlag):
  HW_CAM = 1
  HW_CAM_LONG = 2
  EV = 4


class Footnote(Enum):
  SETUP = CarFootnote(
    "See more setup details for <a href=\"https://github.com/commaai/openpilot/wiki/gm\" target=\"_blank\">GM</a>.",
    Column.MAKE, setup_note=True)


@dataclass
class GMCarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC)"

  def init_make(self, CP: CarParams):
    if CP.networkLocation == CarParams.NetworkLocation.fwdCamera:
      if CP.carFingerprint in SDGM_CAR:
        self.car_parts = CarParts.common([CarHarness.gmsdgm])
      else:
        self.car_parts = CarParts.common([CarHarness.gm])
    else:
      self.footnotes.insert(0, Footnote.SETUP)
      self.car_parts = CarParts.common([CarHarness.obd_ii])


@dataclass(frozen=True, kw_only=True)
class GMCarSpecs(CarSpecs):
  tireStiffnessFactor: float = 0.444  # not optimized yet


@dataclass
class GMPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'gm_global_a_powertrain_generated',
    Bus.radar: 'gm_global_a_object',
    Bus.chassis: 'gm_global_a_chassis',
  })


@dataclass
class GMASCMPlatformConfig(GMPlatformConfig):
  def init(self):
    # ASCM is supported, but due to a janky install and hardware configuration, we are not showing in the car docs
    self.car_docs = []


@dataclass
class GMSDGMPlatformConfig(GMPlatformConfig):
  def init(self):
    # Don't show in docs until the harness is sold. See https://github.com/commaai/openpilot/issues/32471
    self.car_docs = []


class CAR(Platforms):
  HOLDEN_ASTRA = GMASCMPlatformConfig(
    [GMCarDocs("Holden Astra 2017")],
    GMCarSpecs(mass=1363, wheelbase=2.662, steerRatio=15.7, centerToFrontRatio=0.4),
  )
  CHEVROLET_VOLT = GMASCMPlatformConfig(
    [GMCarDocs("Chevrolet Volt 2017-18", min_enable_speed=0, video="https://youtu.be/QeMCN_4TFfQ")],
    GMCarSpecs(mass=1607, wheelbase=2.69, steerRatio=17.7, centerToFrontRatio=0.45, tireStiffnessFactor=0.469),
  )
  CADILLAC_ATS = GMASCMPlatformConfig(
    [GMCarDocs("Cadillac ATS Premium Performance 2018")],
    GMCarSpecs(mass=1601, wheelbase=2.78, steerRatio=15.3),
  )
  CHEVROLET_MALIBU = GMASCMPlatformConfig(
    [GMCarDocs("Chevrolet Malibu Premier 2017")],
    GMCarSpecs(mass=1496, wheelbase=2.83, steerRatio=15.8, centerToFrontRatio=0.4),
  )
  GMC_ACADIA = GMASCMPlatformConfig(
    [GMCarDocs("GMC Acadia 2018", video="https://www.youtube.com/watch?v=0ZN6DdsBUZo")],
    GMCarSpecs(mass=1975, wheelbase=2.86, steerRatio=14.4, centerToFrontRatio=0.4),
  )
  BUICK_LACROSSE = GMASCMPlatformConfig(
    [GMCarDocs("Buick LaCrosse 2017-19", "Driver Confidence Package 2")],
    GMCarSpecs(mass=1712, wheelbase=2.91, steerRatio=15.8, centerToFrontRatio=0.4),
  )
  BUICK_REGAL = GMASCMPlatformConfig(
    [GMCarDocs("Buick Regal Essence 2018")],
    GMCarSpecs(mass=1714, wheelbase=2.83, steerRatio=14.4, centerToFrontRatio=0.4),
  )
  CADILLAC_ESCALADE = GMASCMPlatformConfig(
    [GMCarDocs("Cadillac Escalade 2017", "Driver Assist Package")],
    GMCarSpecs(mass=2564, wheelbase=2.95, steerRatio=17.3),
  )
  CADILLAC_ESCALADE_ESV = GMASCMPlatformConfig(
    [GMCarDocs("Cadillac Escalade ESV 2016", "Adaptive Cruise Control (ACC) & LKAS")],
    GMCarSpecs(mass=2739, wheelbase=3.302, steerRatio=17.3, tireStiffnessFactor=1.0),
  )
  CADILLAC_ESCALADE_ESV_2019 = GMASCMPlatformConfig(
    [GMCarDocs("Cadillac Escalade ESV 2019", "Adaptive Cruise Control (ACC) & LKAS")],
    CADILLAC_ESCALADE_ESV.specs,
  )
  CHEVROLET_BOLT_EUV = GMPlatformConfig(
    [
      GMCarDocs("Chevrolet Bolt EUV 2022-23", "Premier or Premier Redline Trim without Super Cruise Package", video="https://youtu.be/xvwzGMUA210"),
      GMCarDocs("Chevrolet Bolt EV 2022-23", "2LT Trim with Adaptive Cruise Control Package"),
    ],
    GMCarSpecs(mass=1669, wheelbase=2.63779, steerRatio=16.8, centerToFrontRatio=0.4, tireStiffnessFactor=1.0),
  )
  CHEVROLET_SILVERADO = GMPlatformConfig(
    [
      GMCarDocs("Chevrolet Silverado 1500 2020-21", "Safety Package II"),
      GMCarDocs("GMC Sierra 1500 2020-21", "Driver Alert Package II", video="https://youtu.be/5HbNoBLzRwE"),
    ],
    GMCarSpecs(mass=2450, wheelbase=3.75, steerRatio=16.3, tireStiffnessFactor=1.0),
  )
  CHEVROLET_EQUINOX = GMPlatformConfig(
    [GMCarDocs("Chevrolet Equinox 2019-22")],
    GMCarSpecs(mass=1588, wheelbase=2.72, steerRatio=14.4, centerToFrontRatio=0.4),
  )
  CHEVROLET_TRAILBLAZER = GMPlatformConfig(
    [GMCarDocs("Chevrolet Trailblazer 2021-22")],
    GMCarSpecs(mass=1345, wheelbase=2.64, steerRatio=16.8, centerToFrontRatio=0.4, tireStiffnessFactor=1.0),
  )
  CADILLAC_XT4 = GMSDGMPlatformConfig(
    [GMCarDocs("Cadillac XT4 2023", "Driver Assist Package")],
    GMCarSpecs(mass=1660, wheelbase=2.78, steerRatio=14.4, centerToFrontRatio=0.4),
  )
  CHEVROLET_VOLT_2019 = GMSDGMPlatformConfig(
    [GMCarDocs("Chevrolet Volt 2019", "Adaptive Cruise Control (ACC) & LKAS")],
    GMCarSpecs(mass=1607, wheelbase=2.69, steerRatio=15.7, centerToFrontRatio=0.45),
  )
  CHEVROLET_TRAVERSE = GMSDGMPlatformConfig(
    [GMCarDocs("Chevrolet Traverse 2022-23", "RS, Premier, or High Country Trim")],
    GMCarSpecs(mass=1955, wheelbase=3.07, steerRatio=17.9, centerToFrontRatio=0.4),
  )
  GMC_YUKON = GMPlatformConfig(
    [GMCarDocs("GMC Yukon 2019-20", "Adaptive Cruise Control (ACC) & LKAS")],
    GMCarSpecs(mass=2490, wheelbase=2.94, steerRatio=17.3, centerToFrontRatio=0.5, tireStiffnessFactor=1.0),
  )


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
GM_BOOT_SOFTWARE_PART_NUMER_REQUEST = b'\x1a\xc0'  # likely does not contain anything useful
GM_SOFTWARE_MODULE_1_REQUEST = b'\x1a\xc1'
GM_SOFTWARE_MODULE_2_REQUEST = b'\x1a\xc2'
GM_SOFTWARE_MODULE_3_REQUEST = b'\x1a\xc3'

# Part number of XML data file that is used to configure ECU
GM_XML_DATA_FILE_PART_NUMBER = b'\x1a\x9c'
GM_XML_CONFIG_COMPAT_ID = b'\x1a\x9b'  # used to know if XML file is compatible with the ECU software/hardware

# This DID is for identifying the part number that reflects the mix of hardware,
# software, and calibrations in the ECU when it first arrives at the vehicle assembly plant.
# If there's an Alpha Code, it's associated with this part number and stored in the DID $DB.
GM_END_MODEL_PART_NUMBER_REQUEST = b'\x1a\xcb'
GM_END_MODEL_PART_NUMBER_ALPHA_CODE_REQUEST = b'\x1a\xdb'
GM_BASE_MODEL_PART_NUMBER_REQUEST = b'\x1a\xcc'
GM_BASE_MODEL_PART_NUMBER_ALPHA_CODE_REQUEST = b'\x1a\xdc'
GM_FW_RESPONSE = b'\x5a'

GM_FW_REQUESTS = [
  GM_BOOT_SOFTWARE_PART_NUMER_REQUEST,
  GM_SOFTWARE_MODULE_1_REQUEST,
  GM_SOFTWARE_MODULE_2_REQUEST,
  GM_SOFTWARE_MODULE_3_REQUEST,
  GM_XML_DATA_FILE_PART_NUMBER,
  GM_XML_CONFIG_COMPAT_ID,
  GM_END_MODEL_PART_NUMBER_REQUEST,
  GM_END_MODEL_PART_NUMBER_ALPHA_CODE_REQUEST,
  GM_BASE_MODEL_PART_NUMBER_REQUEST,
  GM_BASE_MODEL_PART_NUMBER_ALPHA_CODE_REQUEST,
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

# TODO: detect most of these sets live
EV_CAR = {CAR.CHEVROLET_VOLT, CAR.CHEVROLET_VOLT_2019, CAR.CHEVROLET_BOLT_EUV}

# We're integrated at the camera with VOACC on these cars (instead of ASCM w/ OBD-II harness)
CAMERA_ACC_CAR = {CAR.CHEVROLET_BOLT_EUV, CAR.CHEVROLET_SILVERADO, CAR.CHEVROLET_EQUINOX, CAR.CHEVROLET_TRAILBLAZER, CAR.GMC_YUKON}

# Alt ASCMActiveCruiseControlStatus
ALT_ACCS = {CAR.GMC_YUKON}

# We're integrated at the Safety Data Gateway Module on these cars
SDGM_CAR = {CAR.CADILLAC_XT4, CAR.CHEVROLET_VOLT_2019, CAR.CHEVROLET_TRAVERSE}

STEER_THRESHOLD = 1.0

DBC = CAR.create_dbc_map()
