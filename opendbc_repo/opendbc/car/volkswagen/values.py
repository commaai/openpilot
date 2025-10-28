from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum, IntFlag, StrEnum

from opendbc.car import Bus, CanBusBase, CarSpecs, DbcDict, PlatformConfig, Platforms, structs, uds
from opendbc.can import CANDefine
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Column, \
                                                     Device
from opendbc.car.fw_query_definitions import EcuAddrSubAddr, FwQueryConfig, Request, p16
from opendbc.car.vin import Vin

Ecu = structs.CarParams.Ecu
NetworkLocation = structs.CarParams.NetworkLocation
TransmissionType = structs.CarParams.TransmissionType
GearShifter = structs.CarState.GearShifter
Button = namedtuple('Button', ['event_type', 'can_addr', 'can_msg', 'values'])


class CanBus(CanBusBase):
  def __init__(self, CP=None, fingerprint=None) -> None:
    super().__init__(CP, fingerprint)

    self._ext = self.offset
    if CP is not None:
      self._ext = self.offset + 2 if CP.networkLocation == NetworkLocation.gateway else self.offset

  @property
  def pt(self) -> int:
    # ADAS / Extended CAN, gateway side of the relay
    return self.offset

  @property
  def aux(self) -> int:
    # NetworkLocation.fwdCamera: radar-camera object fusion CAN
    # NetworkLocation.gateway: powertrain CAN
    return self.offset + 1

  @property
  def cam(self) -> int:
    # ADAS / Extended CAN, camera side of the relay
    return self.offset + 2

  @property
  def ext(self) -> int:
    # ADAS / Extended CAN, side of the relay with the ACC radar
    return self._ext


class CarControllerParams:
  STEER_STEP = 2                           # HCA_01/HCA_1 message frequency 50Hz
  ACC_CONTROL_STEP = 2                     # ACC_06/ACC_07/ACC_System frequency 50Hz
  AEB_CONTROL_STEP = 2                     # ACC_10 frequency 50Hz
  AEB_HUD_STEP = 20                        # ACC_15 frequency 5Hz

  # Documented lateral limits: 3.00 Nm max, rate of change 5.00 Nm/sec.
  # MQB vs PQ maximums are shared, but rate-of-change limited differently
  # based on safety requirements driven by lateral accel testing.

  STEER_MAX = 300                          # Max heading control assist torque 3.00 Nm
  STEER_DRIVER_MULTIPLIER = 3              # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1                  # from dbc

  STEER_TIME_MAX = 360                     # Max time that EPS allows uninterrupted HCA steering control
  STEER_TIME_ALERT = STEER_TIME_MAX - 10   # If mitigation fails, time to soft disengage before EPS timer expires
  STEER_TIME_STUCK_TORQUE = 1.9            # EPS limits same torque to 6 seconds, reset timer 3x within that period

  DEFAULT_MIN_STEER_SPEED = 0.4            # m/s, newer EPS racks fault below this speed, don't show a low speed alert

  ACCEL_MAX = 2.0                          # 2.0 m/s max acceleration
  ACCEL_MIN = -3.5                         # 3.5 m/s max deceleration

  def __init__(self, CP):
    can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])

    if CP.flags & VolkswagenFlags.PQ:
      self.LDW_STEP = 5                   # LDW_1 message frequency 20Hz
      self.ACC_HUD_STEP = 4               # ACC_GRA_Anzeige frequency 25Hz
      self.STEER_DRIVER_ALLOWANCE = 80    # Driver intervention threshold 0.8 Nm
      self.STEER_DELTA_UP = 6             # Max HCA reached in 1.00s (STEER_MAX / (50Hz * 1.00))
      self.STEER_DELTA_DOWN = 10          # Min HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))

      if CP.transmissionType == TransmissionType.automatic:
        self.shifter_values = can_define.dv["Getriebe_1"]["Waehlhebelposition__Getriebe_1_"]
      self.hca_status_values = can_define.dv["Lenkhilfe_2"]["LH2_Sta_HCA"]

      self.BUTTONS = [
        Button(structs.CarState.ButtonEvent.Type.setCruise, "GRA_Neu", "GRA_Neu_Setzen", [1]),
        Button(structs.CarState.ButtonEvent.Type.resumeCruise, "GRA_Neu", "GRA_Recall", [1]),
        Button(structs.CarState.ButtonEvent.Type.accelCruise, "GRA_Neu", "GRA_Up_kurz", [1]),
        Button(structs.CarState.ButtonEvent.Type.decelCruise, "GRA_Neu", "GRA_Down_kurz", [1]),
        Button(structs.CarState.ButtonEvent.Type.cancel, "GRA_Neu", "GRA_Abbrechen", [1]),
        Button(structs.CarState.ButtonEvent.Type.gapAdjustCruise, "GRA_Neu", "GRA_Zeitluecke", [1]),
      ]

      self.LDW_MESSAGES = {
        "none": 0,  # Nothing to display
        "laneAssistUnavail": 1,  # "Lane Assist currently not available."
        "laneAssistUnavailSysError": 2,  # "Lane Assist system error"
        "laneAssistUnavailNoSensorView": 3,  # "Lane Assist not available. No sensor view."
        "laneAssistTakeOver": 4,  # "Lane Assist: Please Take Over Steering"
        "laneAssistDeactivTrailer": 5,  # "Lane Assist: no function with trailer"
      }

    else:
      self.LDW_STEP = 10                  # LDW_02 message frequency 10Hz
      self.ACC_HUD_STEP = 6               # ACC_02 message frequency 16Hz
      self.STEER_DRIVER_ALLOWANCE = 80    # Driver intervention threshold 0.8 Nm
      self.STEER_DELTA_UP = 4             # Max HCA reached in 1.50s (STEER_MAX / (50Hz * 1.50))
      self.STEER_DELTA_DOWN = 10          # Min HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))

      if CP.transmissionType == TransmissionType.automatic:
        self.shifter_values = can_define.dv["Gateway_73"]["GE_Fahrstufe"]
      elif CP.transmissionType == TransmissionType.direct:
        self.shifter_values = can_define.dv["Motor_EV_01"]["MO_Waehlpos"]
      self.hca_status_values = can_define.dv["LH_EPS_03"]["EPS_HCA_Status"]

      self.BUTTONS = [
        Button(structs.CarState.ButtonEvent.Type.setCruise, "GRA_ACC_01", "GRA_Tip_Setzen", [1]),
        Button(structs.CarState.ButtonEvent.Type.resumeCruise, "GRA_ACC_01", "GRA_Tip_Wiederaufnahme", [1]),
        Button(structs.CarState.ButtonEvent.Type.accelCruise, "GRA_ACC_01", "GRA_Tip_Hoch", [1]),
        Button(structs.CarState.ButtonEvent.Type.decelCruise, "GRA_ACC_01", "GRA_Tip_Runter", [1]),
        Button(structs.CarState.ButtonEvent.Type.cancel, "GRA_ACC_01", "GRA_Abbrechen", [1]),
        Button(structs.CarState.ButtonEvent.Type.gapAdjustCruise, "GRA_ACC_01", "GRA_Verstellung_Zeitluecke", [1]),
      ]

      self.LDW_MESSAGES = {
        "none": 0,                            # Nothing to display
        "laneAssistUnavailChime": 1,          # "Lane Assist currently not available." with chime
        "laneAssistUnavailNoSensorChime": 3,  # "Lane Assist not available. No sensor view." with chime
        "laneAssistTakeOverUrgent": 4,        # "Lane Assist: Please Take Over Steering" with urgent beep
        "emergencyAssistUrgent": 6,           # "Emergency Assist: Please Take Over Steering" with urgent beep
        "laneAssistTakeOverChime": 7,         # "Lane Assist: Please Take Over Steering" with chime
        "laneAssistTakeOver": 8,              # "Lane Assist: Please Take Over Steering" silent
        "emergencyAssistChangingLanes": 9,    # "Emergency Assist: Changing lanes..." with urgent beep
        "laneAssistDeactivated": 10,          # "Lane Assist deactivated." silent with persistent icon afterward
      }


class WMI(StrEnum):
  VOLKSWAGEN_USA_SUV = "1V2"
  VOLKSWAGEN_USA_CAR = "1VW"
  VOLKSWAGEN_MEXICO_SUV = "3VV"
  VOLKSWAGEN_MEXICO_CAR = "3VW"
  VOLKSWAGEN_ARGENTINA = "8AW"
  VOLKSWAGEN_BRASIL = "9BW"
  SAIC_VOLKSWAGEN = "LSV"
  SKODA = "TMB"
  SEAT = "VSS"
  AUDI_EUROPE_MPV = "WA1"
  AUDI_GERMANY_CAR = "WAU"
  MAN = "WMA"
  AUDI_SPORT = "WUA"
  VOLKSWAGEN_COMMERCIAL = "WV1"
  VOLKSWAGEN_COMMERCIAL_BUS_VAN = "WV2"
  VOLKSWAGEN_EUROPE_SUV = "WVG"
  VOLKSWAGEN_EUROPE_CAR = "WVW"
  VOLKSWAGEN_GROUP_RUS = "XW8"


class VolkswagenSafetyFlags(IntFlag):
  LONG_CONTROL = 1


class VolkswagenFlags(IntFlag):
  # Detected flags
  STOCK_HCA_PRESENT = 1
  KOMBI_PRESENT = 4

  # Static flags
  PQ = 2


@dataclass
class VolkswagenMQBPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'vw_mqb'})
  # Volkswagen uses the VIN WMI and chassis code to match in the absence of the comma power
  # on camera-integrated cars, as we lose too many ECUs to reliably identify the vehicle
  chassis_codes: set[str] = field(default_factory=set)
  wmis: set[WMI] = field(default_factory=set)


@dataclass
class VolkswagenPQPlatformConfig(VolkswagenMQBPlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'vw_pq'})

  def init(self):
    self.flags |= VolkswagenFlags.PQ


@dataclass(frozen=True, kw_only=True)
class VolkswagenCarSpecs(CarSpecs):
  centerToFrontRatio: float = 0.45
  steerRatio: float = 15.6
  minSteerSpeed: float = CarControllerParams.DEFAULT_MIN_STEER_SPEED


class Footnote(Enum):
  KAMIQ = CarFootnote(
    "Not including the China market Kamiq, which is based on the (currently) unsupported PQ34 platform.",
    Column.MODEL)
  PASSAT = CarFootnote(
    "Refers only to the MQB-based European B8 Passat, not the NMS Passat in the USA/China/Mideast markets.",
    Column.MODEL)
  SKODA_HEATED_WINDSHIELD = CarFootnote(
    "Some Škoda vehicles are equipped with heated windshields, which are known " +
    "to block GPS signal needed for some comma 3X functionality.",
    Column.MODEL)
  VW_EXP_LONG = CarFootnote(
    "Only available for vehicles using a gateway (J533) harness. At this time, vehicles using a camera harness " +
    "are limited to using stock ACC.",
    Column.LONGITUDINAL, docs_only=True)
  VW_MQB_A0 = CarFootnote(
    "Model-years 2022 and beyond may have a combined CAN gateway and BCM, which is supported by openpilot " +
    "in software, but doesn't yet have a harness available from the comma store.",
    Column.HARDWARE)


@dataclass
class VWCarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.vw_j533]))

  def init_make(self, CP: structs.CarParams):
    self.footnotes.append(Footnote.VW_EXP_LONG)
    if "SKODA" in CP.carFingerprint:
      self.footnotes.append(Footnote.SKODA_HEATED_WINDSHIELD)

    if CP.carFingerprint in (CAR.VOLKSWAGEN_CRAFTER_MK2, CAR.VOLKSWAGEN_TRANSPORTER_T61):
      self.car_parts = CarParts([Device.threex_angled_mount, CarHarness.vw_j533])

    if abs(CP.minSteerSpeed - CarControllerParams.DEFAULT_MIN_STEER_SPEED) < 1e-3:
      self.min_steer_speed = 0


# Check the 7th and 8th characters of the VIN before adding a new CAR. If the
# chassis code is already listed below, don't add a new CAR, just add to the
# FW_VERSIONS for that existing CAR.

class CAR(Platforms):
  config: VolkswagenMQBPlatformConfig | VolkswagenPQPlatformConfig

  VOLKSWAGEN_ARTEON_MK1 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Arteon 2018-23", video="https://youtu.be/FAomFKPFlDA"),
      VWCarDocs("Volkswagen Arteon R 2020-23", video="https://youtu.be/FAomFKPFlDA"),
      VWCarDocs("Volkswagen Arteon eHybrid 2020-23", video="https://youtu.be/FAomFKPFlDA"),
      VWCarDocs("Volkswagen Arteon Shooting Brake 2020-23", video="https://youtu.be/FAomFKPFlDA"),
      VWCarDocs("Volkswagen CC 2018-22", video="https://youtu.be/FAomFKPFlDA"),
    ],
    VolkswagenCarSpecs(mass=1733, wheelbase=2.84),
    chassis_codes={"AN", "3H"},
    wmis={WMI.VOLKSWAGEN_EUROPE_CAR},
  )
  VOLKSWAGEN_ATLAS_MK1 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Atlas 2018-23"),
      VWCarDocs("Volkswagen Atlas Cross Sport 2020-22"),
      VWCarDocs("Volkswagen Teramont 2018-22"),
      VWCarDocs("Volkswagen Teramont Cross Sport 2021-22"),
      VWCarDocs("Volkswagen Teramont X 2021-22"),
    ],
    VolkswagenCarSpecs(mass=2011, wheelbase=2.98),
    chassis_codes={"CA"},
    wmis={WMI.VOLKSWAGEN_USA_SUV, WMI.VOLKSWAGEN_EUROPE_SUV},
  )
  VOLKSWAGEN_CADDY_MK3 = VolkswagenPQPlatformConfig(
    [
      VWCarDocs("Volkswagen Caddy 2019"),
      VWCarDocs("Volkswagen Caddy Maxi 2019"),
    ],
    VolkswagenCarSpecs(mass=1613, wheelbase=2.6, minSteerSpeed=21 * CV.KPH_TO_MS),
    chassis_codes={"2K"},
    wmis={WMI.VOLKSWAGEN_COMMERCIAL_BUS_VAN},
  )
  VOLKSWAGEN_CRAFTER_MK2 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Crafter 2017-24", video="https://youtu.be/4100gLeabmo"),
      VWCarDocs("Volkswagen e-Crafter 2018-24", video="https://youtu.be/4100gLeabmo"),
      VWCarDocs("Volkswagen Grand California 2019-24", video="https://youtu.be/4100gLeabmo"),
      VWCarDocs("MAN TGE 2017-24", video="https://youtu.be/4100gLeabmo"),
      VWCarDocs("MAN eTGE 2020-24", video="https://youtu.be/4100gLeabmo"),
    ],
    VolkswagenCarSpecs(mass=2100, wheelbase=3.64, minSteerSpeed=50 * CV.KPH_TO_MS),
    chassis_codes={"SY", "SZ", "UY", "UZ"},
    wmis={WMI.VOLKSWAGEN_COMMERCIAL, WMI.MAN},
  )
  VOLKSWAGEN_GOLF_MK7 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen e-Golf 2014-20"),
      VWCarDocs("Volkswagen Golf 2015-20", auto_resume=False),
      VWCarDocs("Volkswagen Golf Alltrack 2015-19", auto_resume=False),
      VWCarDocs("Volkswagen Golf GTD 2015-20"),
      VWCarDocs("Volkswagen Golf GTE 2015-20"),
      VWCarDocs("Volkswagen Golf GTI 2015-21", auto_resume=False),
      VWCarDocs("Volkswagen Golf R 2015-19"),
      VWCarDocs("Volkswagen Golf SportsVan 2015-20"),
    ],
    VolkswagenCarSpecs(mass=1397, wheelbase=2.62),
    chassis_codes={"5G", "AU", "BA", "BE"},
    wmis={WMI.VOLKSWAGEN_MEXICO_CAR, WMI.VOLKSWAGEN_EUROPE_CAR},
  )
  VOLKSWAGEN_JETTA_MK6 = VolkswagenPQPlatformConfig(
    [VWCarDocs("Volkswagen Jetta 2015-18")],
    VolkswagenCarSpecs(mass=1518, wheelbase=2.65, minSteerSpeed=50 * CV.KPH_TO_MS, minEnableSpeed=20 * CV.KPH_TO_MS),
    chassis_codes={"5K", "AJ"},
    wmis={WMI.VOLKSWAGEN_MEXICO_CAR},
  )
  VOLKSWAGEN_JETTA_MK7 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Jetta 2018-23"),
      VWCarDocs("Volkswagen Jetta GLI 2021-23"),
    ],
    VolkswagenCarSpecs(mass=1328, wheelbase=2.71),
    chassis_codes={"BU"},
    wmis={WMI.VOLKSWAGEN_MEXICO_CAR, WMI.VOLKSWAGEN_EUROPE_CAR},
  )
  VOLKSWAGEN_PASSAT_MK8 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Passat 2015-22", footnotes=[Footnote.PASSAT]),
      VWCarDocs("Volkswagen Passat Alltrack 2015-22"),
      VWCarDocs("Volkswagen Passat GTE 2015-22"),
    ],
    VolkswagenCarSpecs(mass=1551, wheelbase=2.79),
    chassis_codes={"3C", "3G"},
    wmis={WMI.VOLKSWAGEN_EUROPE_CAR},
  )
  VOLKSWAGEN_PASSAT_NMS = VolkswagenPQPlatformConfig(
    [VWCarDocs("Volkswagen Passat NMS 2017-22")],
    VolkswagenCarSpecs(mass=1503, wheelbase=2.80, minSteerSpeed=50 * CV.KPH_TO_MS, minEnableSpeed=20 * CV.KPH_TO_MS),
    chassis_codes={"A3"},
    wmis={WMI.VOLKSWAGEN_USA_CAR},
  )
  VOLKSWAGEN_POLO_MK6 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Polo 2018-23", footnotes=[Footnote.VW_MQB_A0]),
      VWCarDocs("Volkswagen Polo GTI 2018-23", footnotes=[Footnote.VW_MQB_A0]),
    ],
    VolkswagenCarSpecs(mass=1230, wheelbase=2.55),
    chassis_codes={"AW"},
    wmis={WMI.VOLKSWAGEN_EUROPE_CAR},
  )
  VOLKSWAGEN_SHARAN_MK2 = VolkswagenPQPlatformConfig(
    [
      VWCarDocs("Volkswagen Sharan 2018-22"),
      VWCarDocs("SEAT Alhambra 2018-20"),
    ],
    VolkswagenCarSpecs(mass=1639, wheelbase=2.92, minSteerSpeed=50 * CV.KPH_TO_MS),
    chassis_codes={"7N"},
    wmis={WMI.VOLKSWAGEN_EUROPE_CAR},
  )
  VOLKSWAGEN_TAOS_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Volkswagen Taos 2022-24")],
    VolkswagenCarSpecs(mass=1498, wheelbase=2.69),
    chassis_codes={"B2"},
    wmis={WMI.VOLKSWAGEN_MEXICO_SUV, WMI.VOLKSWAGEN_ARGENTINA},
  )
  VOLKSWAGEN_TCROSS_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Volkswagen T-Cross 2021", footnotes=[Footnote.VW_MQB_A0])],
    VolkswagenCarSpecs(mass=1150, wheelbase=2.60),
    chassis_codes={"C1"},
    wmis={WMI.VOLKSWAGEN_EUROPE_SUV},
  )
  VOLKSWAGEN_TIGUAN_MK2 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Tiguan 2018-24"),
      VWCarDocs("Volkswagen Tiguan eHybrid 2021-23"),
    ],
    VolkswagenCarSpecs(mass=1715, wheelbase=2.74),
    chassis_codes={"5N", "AD", "AX", "BW"},
    wmis={WMI.VOLKSWAGEN_EUROPE_SUV, WMI.VOLKSWAGEN_MEXICO_SUV},
  )
  VOLKSWAGEN_TOURAN_MK2 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Volkswagen Touran 2016-23")],
    VolkswagenCarSpecs(mass=1516, wheelbase=2.79),
    chassis_codes={"1T"},
    wmis={WMI.VOLKSWAGEN_EUROPE_SUV},
  )
  VOLKSWAGEN_TRANSPORTER_T61 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Caravelle 2020"),
      VWCarDocs("Volkswagen California 2021-23"),
    ],
    VolkswagenCarSpecs(mass=1926, wheelbase=3.00, minSteerSpeed=14.0),
    chassis_codes={"7H", "7L"},
    wmis={WMI.VOLKSWAGEN_COMMERCIAL_BUS_VAN},
  )
  VOLKSWAGEN_TROC_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Volkswagen T-Roc 2018-23")],
    VolkswagenCarSpecs(mass=1413, wheelbase=2.63),
    chassis_codes={"A1"},
    wmis={WMI.VOLKSWAGEN_EUROPE_SUV},
  )
  AUDI_A3_MK3 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Audi A3 2014-19"),
      VWCarDocs("Audi A3 Sportback e-tron 2017-18"),
      VWCarDocs("Audi RS3 2018"),
      VWCarDocs("Audi S3 2015-17"),
    ],
    VolkswagenCarSpecs(mass=1335, wheelbase=2.61),
    chassis_codes={"8V", "FF"},
    wmis={WMI.AUDI_GERMANY_CAR, WMI.AUDI_SPORT},
  )
  AUDI_Q2_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Audi Q2 2018")],
    VolkswagenCarSpecs(mass=1205, wheelbase=2.61),
    chassis_codes={"GA"},
    wmis={WMI.AUDI_GERMANY_CAR},
  )
  AUDI_Q3_MK2 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Audi Q3 2019-24")],
    VolkswagenCarSpecs(mass=1623, wheelbase=2.68),
    chassis_codes={"8U", "F3", "FS"},
    wmis={WMI.AUDI_EUROPE_MPV, WMI.AUDI_GERMANY_CAR},
  )
  SEAT_ATECA_MK1 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("CUPRA Ateca 2018-23"),
      VWCarDocs("SEAT Ateca 2016-23"),
      VWCarDocs("SEAT Leon 2014-20"),
    ],
    VolkswagenCarSpecs(mass=1300, wheelbase=2.64),
    chassis_codes={"5F"},
    wmis={WMI.SEAT},
  )
  SKODA_FABIA_MK4 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Škoda Fabia 2022-23", footnotes=[Footnote.VW_MQB_A0])],
    VolkswagenCarSpecs(mass=1266, wheelbase=2.56),
    chassis_codes={"PJ"},
    wmis={WMI.SKODA},
  )
  SKODA_KAMIQ_MK1 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Škoda Kamiq 2021-23", footnotes=[Footnote.VW_MQB_A0, Footnote.KAMIQ]),
      VWCarDocs("Škoda Scala 2020-23", footnotes=[Footnote.VW_MQB_A0]),
    ],
    VolkswagenCarSpecs(mass=1230, wheelbase=2.66),
    chassis_codes={"NW"},
    wmis={WMI.SKODA},
  )
  SKODA_KAROQ_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Škoda Karoq 2019-23")],
    VolkswagenCarSpecs(mass=1278, wheelbase=2.66),
    chassis_codes={"NU"},
    wmis={WMI.SKODA},
  )
  SKODA_KODIAQ_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Škoda Kodiaq 2017-23")],
    VolkswagenCarSpecs(mass=1569, wheelbase=2.79),
    chassis_codes={"NS"},
    wmis={WMI.SKODA, WMI.VOLKSWAGEN_GROUP_RUS},
  )
  SKODA_OCTAVIA_MK3 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Škoda Octavia 2015-19"),
      VWCarDocs("Škoda Octavia RS 2016"),
      VWCarDocs("Škoda Octavia Scout 2017-19"),
    ],
    VolkswagenCarSpecs(mass=1388, wheelbase=2.68),
    chassis_codes={"NE"},
    wmis={WMI.SKODA},
  )
  SKODA_SUPERB_MK3 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Škoda Superb 2015-22")],
    VolkswagenCarSpecs(mass=1505, wheelbase=2.84),
    chassis_codes={"3V", "NP"},
    wmis={WMI.SKODA},
  )


def match_fw_to_car_fuzzy(live_fw_versions, vin, offline_fw_versions) -> set[str]:
  candidates = set()

  # Compile all FW versions for each ECU
  all_ecu_versions: dict[EcuAddrSubAddr, set[str]] = defaultdict(set)
  for ecus in offline_fw_versions.values():
    for ecu, versions in ecus.items():
      all_ecu_versions[ecu] |= set(versions)

  # Check the WMI and chassis code to determine the platform
  # https://www.clubvw.org.au/vwreference/vwvin
  vin_obj = Vin(vin)
  chassis_code = vin_obj.vds[3:5]

  for platform in CAR:
    valid_ecus = set()
    for ecu in offline_fw_versions[platform]:
      addr = ecu[1:]
      if ecu[0] not in CHECK_FUZZY_ECUS:
        continue

      # Sanity check that live FW is in the superset of all FW, Volkswagen ECU part numbers are commonly shared
      found_versions = live_fw_versions.get(addr, [])
      expected_versions = all_ecu_versions[ecu]
      if not any(found_version in expected_versions for found_version in found_versions):
        break

      valid_ecus.add(ecu[0])

    if valid_ecus != CHECK_FUZZY_ECUS:
      continue

    if vin_obj.wmi in platform.config.wmis and chassis_code in platform.config.chassis_codes:
      candidates.add(platform)

  return {str(c) for c in candidates}


# These ECUs are required to match to gain a VIN match
# TODO: do we want to check camera when we add its FW?
CHECK_FUZZY_ECUS = {Ecu.fwdRadar}

# All supported cars should return FW from the engine, srs, eps, and fwdRadar. Cars
# with a manual trans won't return transmission firmware, but all other cars will.
#
# The 0xF187 SW part number query should return in the form of N[NX][NX] NNN NNN [X[X]],
# where N=number, X=letter, and the trailing two letters are optional. Performance
# tuners sometimes tamper with that field (e.g. 8V0 9C0 BB0 1 from COBB/EQT). Tampered
# ECU SW part numbers are invalid for vehicle ID and compatibility checks. Try to have
# them repaired by the tuner before including them in openpilot.

VOLKSWAGEN_VERSION_REQUEST_MULTI = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_VERSION_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)
VOLKSWAGEN_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])

VOLKSWAGEN_RX_OFFSET = 0x6a

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[request for bus, obd_multiplexing in [(1, True), (1, False), (0, False)] for request in [
    Request(
      [VOLKSWAGEN_VERSION_REQUEST_MULTI],
      [VOLKSWAGEN_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.srs, Ecu.eps, Ecu.fwdRadar, Ecu.fwdCamera],
      rx_offset=VOLKSWAGEN_RX_OFFSET,
      bus=bus,
      obd_multiplexing=obd_multiplexing,
    ),
    Request(
      [VOLKSWAGEN_VERSION_REQUEST_MULTI],
      [VOLKSWAGEN_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.transmission],
      bus=bus,
      obd_multiplexing=obd_multiplexing,
    ),
  ]],
  non_essential_ecus={Ecu.eps: list(CAR)},
  extra_ecus=[(Ecu.fwdCamera, 0x74f, None)],
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)

DBC = CAR.create_dbc_map()
