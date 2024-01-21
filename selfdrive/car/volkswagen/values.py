from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum, IntFlag, StrEnum
from typing import Dict, List, Union

from cereal import car
from panda.python import uds
from opendbc.can.can_define import CANDefine
from openpilot.selfdrive.car import dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarInfo, CarParts, Column, \
                                           Device
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = car.CarParams.Ecu
NetworkLocation = car.CarParams.NetworkLocation
TransmissionType = car.CarParams.TransmissionType
GearShifter = car.CarState.GearShifter
Button = namedtuple('Button', ['event_type', 'can_addr', 'can_msg', 'values'])


class CarControllerParams:
  STEER_STEP = 2                           # HCA_01/HCA_1 message frequency 50Hz
  ACC_CONTROL_STEP = 2                     # ACC_06/ACC_07/ACC_System frequency 50Hz

  # Documented lateral limits: 3.00 Nm max, rate of change 5.00 Nm/sec.
  # MQB vs PQ maximums are shared, but rate-of-change limited differently
  # based on safety requirements driven by lateral accel testing.

  STEER_MAX = 300                          # Max heading control assist torque 3.00 Nm
  STEER_DRIVER_MULTIPLIER = 3              # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1                  # from dbc

  STEER_TIME_MAX = 360                     # Max time that EPS allows uninterrupted HCA steering control
  STEER_TIME_ALERT = STEER_TIME_MAX - 10   # If mitigation fails, time to soft disengage before EPS timer expires
  STEER_TIME_STUCK_TORQUE = 1.9            # EPS limits same torque to 6 seconds, reset timer 3x within that period

  ACCEL_MAX = 2.0                          # 2.0 m/s max acceleration
  ACCEL_MIN = -3.5                         # 3.5 m/s max deceleration

  def __init__(self, CP):
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])

    if CP.carFingerprint in PQ_CARS:
      self.LDW_STEP = 5                   # LDW_1 message frequency 20Hz
      self.ACC_HUD_STEP = 4               # ACC_GRA_Anzeige frequency 25Hz
      self.STEER_DRIVER_ALLOWANCE = 80    # Driver intervention threshold 0.8 Nm
      self.STEER_DELTA_UP = 6             # Max HCA reached in 1.00s (STEER_MAX / (50Hz * 1.00))
      self.STEER_DELTA_DOWN = 10          # Min HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))

      if CP.transmissionType == TransmissionType.automatic:
        self.shifter_values = can_define.dv["Getriebe_1"]["Waehlhebelposition__Getriebe_1_"]
      self.hca_status_values = can_define.dv["Lenkhilfe_2"]["LH2_Sta_HCA"]

      self.BUTTONS = [
        Button(car.CarState.ButtonEvent.Type.setCruise, "GRA_Neu", "GRA_Neu_Setzen", [1]),
        Button(car.CarState.ButtonEvent.Type.resumeCruise, "GRA_Neu", "GRA_Recall", [1]),
        Button(car.CarState.ButtonEvent.Type.accelCruise, "GRA_Neu", "GRA_Up_kurz", [1]),
        Button(car.CarState.ButtonEvent.Type.decelCruise, "GRA_Neu", "GRA_Down_kurz", [1]),
        Button(car.CarState.ButtonEvent.Type.cancel, "GRA_Neu", "GRA_Abbrechen", [1]),
        Button(car.CarState.ButtonEvent.Type.gapAdjustCruise, "GRA_Neu", "GRA_Zeitluecke", [1]),
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
        self.shifter_values = can_define.dv["Getriebe_11"]["GE_Fahrstufe"]
      elif CP.transmissionType == TransmissionType.direct:
        self.shifter_values = can_define.dv["EV_Gearshift"]["GearPosition"]
      self.hca_status_values = can_define.dv["LH_EPS_03"]["EPS_HCA_Status"]

      self.BUTTONS = [
        Button(car.CarState.ButtonEvent.Type.setCruise, "GRA_ACC_01", "GRA_Tip_Setzen", [1]),
        Button(car.CarState.ButtonEvent.Type.resumeCruise, "GRA_ACC_01", "GRA_Tip_Wiederaufnahme", [1]),
        Button(car.CarState.ButtonEvent.Type.accelCruise, "GRA_ACC_01", "GRA_Tip_Hoch", [1]),
        Button(car.CarState.ButtonEvent.Type.decelCruise, "GRA_ACC_01", "GRA_Tip_Runter", [1]),
        Button(car.CarState.ButtonEvent.Type.cancel, "GRA_ACC_01", "GRA_Abbrechen", [1]),
        Button(car.CarState.ButtonEvent.Type.gapAdjustCruise, "GRA_ACC_01", "GRA_Verstellung_Zeitluecke", [1]),
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


class CANBUS:
  pt = 0
  cam = 2


class VolkswagenFlags(IntFlag):
  STOCK_HCA_PRESENT = 1


# Check the 7th and 8th characters of the VIN before adding a new CAR. If the
# chassis code is already listed below, don't add a new CAR, just add to the
# FW_VERSIONS for that existing CAR.
# Exception: SEAT Leon and SEAT Ateca share a chassis code

class CAR(StrEnum):
  ARTEON_MK1 = "VOLKSWAGEN ARTEON 1ST GEN"          # Chassis AN, Mk1 VW Arteon and variants
  ATLAS_MK1 = "VOLKSWAGEN ATLAS 1ST GEN"            # Chassis CA, Mk1 VW Atlas and Atlas Cross Sport
  CRAFTER_MK2 = "VOLKSWAGEN CRAFTER 2ND GEN"        # Chassis SY/SZ, Mk2 VW Crafter, VW Grand California, MAN TGE
  GOLF_MK7 = "VOLKSWAGEN GOLF 7TH GEN"              # Chassis 5G/AU/BA/BE, Mk7 VW Golf and variants
  JETTA_MK7 = "VOLKSWAGEN JETTA 7TH GEN"            # Chassis BU, Mk7 VW Jetta
  PASSAT_MK8 = "VOLKSWAGEN PASSAT 8TH GEN"          # Chassis 3G, Mk8 VW Passat and variants
  PASSAT_NMS = "VOLKSWAGEN PASSAT NMS"              # Chassis A3, North America/China/Mideast NMS Passat, incl. facelift
  POLO_MK6 = "VOLKSWAGEN POLO 6TH GEN"              # Chassis AW, Mk6 VW Polo
  SHARAN_MK2 = "VOLKSWAGEN SHARAN 2ND GEN"          # Chassis 7N, Mk2 Volkswagen Sharan and SEAT Alhambra
  TAOS_MK1 = "VOLKSWAGEN TAOS 1ST GEN"              # Chassis B2, Mk1 VW Taos and Tharu
  TCROSS_MK1 = "VOLKSWAGEN T-CROSS 1ST GEN"         # Chassis C1, Mk1 VW T-Cross SWB and LWB variants
  TIGUAN_MK2 = "VOLKSWAGEN TIGUAN 2ND GEN"          # Chassis AD/BW, Mk2 VW Tiguan and variants
  TOURAN_MK2 = "VOLKSWAGEN TOURAN 2ND GEN"          # Chassis 1T, Mk2 VW Touran and variants
  TRANSPORTER_T61 = "VOLKSWAGEN TRANSPORTER T6.1"   # Chassis 7H/7L, T6-facelift Transporter/Multivan/Caravelle/California
  TROC_MK1 = "VOLKSWAGEN T-ROC 1ST GEN"             # Chassis A1, Mk1 VW T-Roc and variants
  AUDI_A3_MK3 = "AUDI A3 3RD GEN"                   # Chassis 8V/FF, Mk3 Audi A3 and variants
  AUDI_Q2_MK1 = "AUDI Q2 1ST GEN"                   # Chassis GA, Mk1 Audi Q2 (RoW) and Q2L (China only)
  AUDI_Q3_MK2 = "AUDI Q3 2ND GEN"                   # Chassis 8U/F3/FS, Mk2 Audi Q3 and variants
  SEAT_ATECA_MK1 = "SEAT ATECA 1ST GEN"             # Chassis 5F, Mk1 SEAT Ateca and CUPRA Ateca
  SEAT_LEON_MK3 = "SEAT LEON 3RD GEN"               # Chassis 5F, Mk3 SEAT Leon and variants
  SKODA_FABIA_MK4 = "SKODA FABIA 4TH GEN"           # Chassis PJ, Mk4 Skoda Fabia
  SKODA_KAMIQ_MK1 = "SKODA KAMIQ 1ST GEN"           # Chassis NW, Mk1 Skoda Kamiq
  SKODA_KAROQ_MK1 = "SKODA KAROQ 1ST GEN"           # Chassis NU, Mk1 Skoda Karoq
  SKODA_KODIAQ_MK1 = "SKODA KODIAQ 1ST GEN"         # Chassis NS, Mk1 Skoda Kodiaq
  SKODA_SCALA_MK1 = "SKODA SCALA 1ST GEN"           # Chassis NW, Mk1 Skoda Scala and Skoda Kamiq
  SKODA_SUPERB_MK3 = "SKODA SUPERB 3RD GEN"         # Chassis 3V/NP, Mk3 Skoda Superb and variants
  SKODA_OCTAVIA_MK3 = "SKODA OCTAVIA 3RD GEN"       # Chassis NE, Mk3 Skoda Octavia and variants


PQ_CARS = {CAR.PASSAT_NMS, CAR.SHARAN_MK2}


DBC: Dict[str, Dict[str, str]] = defaultdict(lambda: dbc_dict("vw_mqb_2010", None))
for car_type in PQ_CARS:
  DBC[car_type] = dbc_dict("vw_golf_mk4", None)


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
    Column.LONGITUDINAL)
  VW_MQB_A0 = CarFootnote(
    "Model-years 2022 and beyond may have a combined CAN gateway and BCM, which is supported by openpilot " +
    "in software, but doesn't yet have a harness available from the comma store.",
    Column.HARDWARE)


@dataclass
class VWCarInfo(CarInfo):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.j533]))

  def init_make(self, CP: car.CarParams):
    self.footnotes.append(Footnote.VW_EXP_LONG)
    if "SKODA" in CP.carFingerprint:
      self.footnotes.append(Footnote.SKODA_HEATED_WINDSHIELD)

    if CP.carFingerprint in (CAR.CRAFTER_MK2, CAR.TRANSPORTER_T61):
      self.car_parts = CarParts([Device.threex_angled_mount, CarHarness.j533])


CAR_INFO: Dict[str, Union[VWCarInfo, List[VWCarInfo]]] = {
  CAR.ARTEON_MK1: [
    VWCarInfo("Volkswagen Arteon 2018-23", video_link="https://youtu.be/FAomFKPFlDA"),
    VWCarInfo("Volkswagen Arteon R 2020-23", video_link="https://youtu.be/FAomFKPFlDA"),
    VWCarInfo("Volkswagen Arteon eHybrid 2020-23", video_link="https://youtu.be/FAomFKPFlDA"),
    VWCarInfo("Volkswagen CC 2018-22", video_link="https://youtu.be/FAomFKPFlDA"),
  ],
  CAR.ATLAS_MK1: [
    VWCarInfo("Volkswagen Atlas 2018-23"),
    VWCarInfo("Volkswagen Atlas Cross Sport 2020-22"),
    VWCarInfo("Volkswagen Teramont 2018-22"),
    VWCarInfo("Volkswagen Teramont Cross Sport 2021-22"),
    VWCarInfo("Volkswagen Teramont X 2021-22"),
  ],
  CAR.CRAFTER_MK2: [
    VWCarInfo("Volkswagen Crafter 2017-23", video_link="https://youtu.be/4100gLeabmo"),
    VWCarInfo("Volkswagen e-Crafter 2018-23", video_link="https://youtu.be/4100gLeabmo"),
    VWCarInfo("Volkswagen Grand California 2019-23", video_link="https://youtu.be/4100gLeabmo"),
    VWCarInfo("MAN TGE 2017-23", video_link="https://youtu.be/4100gLeabmo"),
    VWCarInfo("MAN eTGE 2020-23", video_link="https://youtu.be/4100gLeabmo"),
  ],
  CAR.GOLF_MK7: [
    VWCarInfo("Volkswagen e-Golf 2014-20"),
    VWCarInfo("Volkswagen Golf 2015-20", auto_resume=False),
    VWCarInfo("Volkswagen Golf Alltrack 2015-19", auto_resume=False),
    VWCarInfo("Volkswagen Golf GTD 2015-20"),
    VWCarInfo("Volkswagen Golf GTE 2015-20"),
    VWCarInfo("Volkswagen Golf GTI 2015-21", auto_resume=False),
    VWCarInfo("Volkswagen Golf R 2015-19"),
    VWCarInfo("Volkswagen Golf SportsVan 2015-20"),
  ],
  CAR.JETTA_MK7: [
    VWCarInfo("Volkswagen Jetta 2018-22"),
    VWCarInfo("Volkswagen Jetta GLI 2021-22"),
  ],
  CAR.PASSAT_MK8: [
    VWCarInfo("Volkswagen Passat 2015-22", footnotes=[Footnote.PASSAT]),
    VWCarInfo("Volkswagen Passat Alltrack 2015-22"),
    VWCarInfo("Volkswagen Passat GTE 2015-22"),
  ],
  CAR.PASSAT_NMS: VWCarInfo("Volkswagen Passat NMS 2017-22"),
  CAR.POLO_MK6: [
    VWCarInfo("Volkswagen Polo 2018-23", footnotes=[Footnote.VW_MQB_A0]),
    VWCarInfo("Volkswagen Polo GTI 2018-23", footnotes=[Footnote.VW_MQB_A0]),
  ],
  CAR.SHARAN_MK2: [
    VWCarInfo("Volkswagen Sharan 2018-22"),
    VWCarInfo("SEAT Alhambra 2018-20"),
  ],
  CAR.TAOS_MK1: VWCarInfo("Volkswagen Taos 2022-23"),
  CAR.TCROSS_MK1: VWCarInfo("Volkswagen T-Cross 2021", footnotes=[Footnote.VW_MQB_A0]),
  CAR.TIGUAN_MK2: [
    VWCarInfo("Volkswagen Tiguan 2018-24"),
    VWCarInfo("Volkswagen Tiguan eHybrid 2021-23"),
  ],
  CAR.TOURAN_MK2: VWCarInfo("Volkswagen Touran 2016-23"),
  CAR.TRANSPORTER_T61: [
    VWCarInfo("Volkswagen Caravelle 2020"),
    VWCarInfo("Volkswagen California 2021-23"),
  ],
  CAR.TROC_MK1: VWCarInfo("Volkswagen T-Roc 2018-22", footnotes=[Footnote.VW_MQB_A0]),
  CAR.AUDI_A3_MK3: [
    VWCarInfo("Audi A3 2014-19"),
    VWCarInfo("Audi A3 Sportback e-tron 2017-18"),
    VWCarInfo("Audi RS3 2018"),
    VWCarInfo("Audi S3 2015-17"),
  ],
  CAR.AUDI_Q2_MK1: VWCarInfo("Audi Q2 2018"),
  CAR.AUDI_Q3_MK2: VWCarInfo("Audi Q3 2019-23"),
  CAR.SEAT_ATECA_MK1: VWCarInfo("SEAT Ateca 2018"),
  CAR.SEAT_LEON_MK3: VWCarInfo("SEAT Leon 2014-20"),
  CAR.SKODA_FABIA_MK4: VWCarInfo("Škoda Fabia 2022-23", footnotes=[Footnote.VW_MQB_A0]),
  CAR.SKODA_KAMIQ_MK1: VWCarInfo("Škoda Kamiq 2021-23", footnotes=[Footnote.VW_MQB_A0, Footnote.KAMIQ]),
  CAR.SKODA_KAROQ_MK1: VWCarInfo("Škoda Karoq 2019-23"),
  CAR.SKODA_KODIAQ_MK1: VWCarInfo("Škoda Kodiaq 2017-23"),
  CAR.SKODA_SCALA_MK1: VWCarInfo("Škoda Scala 2020-23", footnotes=[Footnote.VW_MQB_A0]),
  CAR.SKODA_SUPERB_MK3: VWCarInfo("Škoda Superb 2015-22"),
  CAR.SKODA_OCTAVIA_MK3: [
    VWCarInfo("Škoda Octavia 2015-19"),
    VWCarInfo("Škoda Octavia RS 2016"),
  ],
}


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
  requests=[
    Request(
      [VOLKSWAGEN_VERSION_REQUEST_MULTI],
      [VOLKSWAGEN_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.srs, Ecu.eps, Ecu.fwdRadar],
      rx_offset=VOLKSWAGEN_RX_OFFSET,
    ),
    Request(
      [VOLKSWAGEN_VERSION_REQUEST_MULTI],
      [VOLKSWAGEN_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.transmission],
    ),
  ],
)
