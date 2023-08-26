from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Dict, List, Union

from cereal import car
from panda.python import uds
from openpilot.selfdrive.car import dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarInfo, CarParts, Tool, Column
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries, p16

Ecu = car.CarParams.Ecu


class CarControllerParams:
  def __init__(self, CP):
    self.STEER_STEP = 2                # how often we update the steer cmd
    self.STEER_DELTA_UP = 50           # torque increase per refresh, 0.8s to max
    self.STEER_DELTA_DOWN = 70         # torque decrease per refresh
    self.STEER_DRIVER_ALLOWANCE = 60   # allowed driver torque before start limiting
    self.STEER_DRIVER_MULTIPLIER = 50  # weight driver torque heavily
    self.STEER_DRIVER_FACTOR = 1       # from dbc

    if CP.carFingerprint in GLOBAL_GEN2:
      self.STEER_MAX = 1000
      self.STEER_DELTA_UP = 40
      self.STEER_DELTA_DOWN = 40
    elif CP.carFingerprint == CAR.IMPREZA_2020:
      self.STEER_MAX = 1439
    else:
      self.STEER_MAX = 2047

  THROTTLE_MIN = 808
  THROTTLE_MAX = 3400

  THROTTLE_INACTIVE     = 1818   # corresponds to zero acceleration
  THROTTLE_ENGINE_BRAKE = 808    # while braking, eyesight sets throttle to this, probably for engine braking

  BRAKE_MIN = 0
  BRAKE_MAX = 600                # about -3.5m/s2 from testing

  RPM_MIN = 0
  RPM_MAX = 2400

  RPM_INACTIVE = 600             # a good base rpm for zero acceleration

  THROTTLE_LOOKUP_BP = [0, 1]
  THROTTLE_LOOKUP_V = [THROTTLE_INACTIVE, THROTTLE_MAX]

  RPM_LOOKUP_BP = [0, 1]
  RPM_LOOKUP_V = [RPM_INACTIVE, RPM_MAX]

  BRAKE_LOOKUP_BP = [-1, 0]
  BRAKE_LOOKUP_V = [BRAKE_MAX, BRAKE_MIN]


class SubaruFlags(IntFlag):
  SEND_INFOTAINMENT = 1


class CanBus:
  main = 0
  alt = 1
  camera = 2


class CAR:
  # Global platform
  ASCENT = "SUBARU ASCENT LIMITED 2019"
  ASCENT_2023 = "SUBARU ASCENT 2023"
  IMPREZA = "SUBARU IMPREZA LIMITED 2019"
  IMPREZA_2020 = "SUBARU IMPREZA SPORT 2020"
  FORESTER = "SUBARU FORESTER 2019"
  OUTBACK = "SUBARU OUTBACK 6TH GEN"
  CROSSTREK_HYBRID = "SUBARU CROSSTREK HYBRID 2020"
  FORESTER_HYBRID = "SUBARU FORESTER HYBRID 2020"
  LEGACY = "SUBARU LEGACY 7TH GEN"
  FORESTER_2022 = "SUBARU FORESTER 2022"
  OUTBACK_2023 = "SUBARU OUTBACK 7TH GEN"

  # Pre-global
  FORESTER_PREGLOBAL = "SUBARU FORESTER 2017 - 2018"
  LEGACY_PREGLOBAL = "SUBARU LEGACY 2015 - 2018"
  OUTBACK_PREGLOBAL = "SUBARU OUTBACK 2015 - 2017"
  OUTBACK_PREGLOBAL_2018 = "SUBARU OUTBACK 2018 - 2019"


class Footnote(Enum):
  GLOBAL = CarFootnote(
    "In the non-US market, openpilot requires the car to come equipped with EyeSight with Lane Keep Assistance.",
    Column.PACKAGE)


@dataclass
class SubaruCarInfo(CarInfo):
  package: str = "EyeSight Driver Assistance"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.subaru_a]))
  footnotes: List[Enum] = field(default_factory=lambda: [Footnote.GLOBAL])

  def init_make(self, CP: car.CarParams):
    self.car_parts.parts.extend([Tool.socket_8mm_deep, Tool.pry_tool])

CAR_INFO: Dict[str, Union[SubaruCarInfo, List[SubaruCarInfo]]] = {
  CAR.ASCENT: SubaruCarInfo("Subaru Ascent 2019-21", "All"),
  CAR.OUTBACK: SubaruCarInfo("Subaru Outback 2020-22", "All", car_parts=CarParts.common([CarHarness.subaru_b])),
  CAR.LEGACY: SubaruCarInfo("Subaru Legacy 2020-22", "All", car_parts=CarParts.common([CarHarness.subaru_b])),
  CAR.IMPREZA: [
    SubaruCarInfo("Subaru Impreza 2017-19"),
    SubaruCarInfo("Subaru Crosstrek 2018-19", video_link="https://youtu.be/Agww7oE1k-s?t=26"),
    SubaruCarInfo("Subaru XV 2018-19", video_link="https://youtu.be/Agww7oE1k-s?t=26"),
  ],
  CAR.IMPREZA_2020: [
    SubaruCarInfo("Subaru Impreza 2020-22"),
    SubaruCarInfo("Subaru Crosstrek 2020-23"),
    SubaruCarInfo("Subaru XV 2020-21"),
  ],
  # TODO: is there an XV and Impreza too?
  CAR.CROSSTREK_HYBRID: SubaruCarInfo("Subaru Crosstrek Hybrid 2020"),
  CAR.FORESTER_HYBRID: SubaruCarInfo("Subaru Forester Hybrid 2020"),
  CAR.FORESTER: SubaruCarInfo("Subaru Forester 2019-21", "All"),
  CAR.FORESTER_PREGLOBAL: SubaruCarInfo("Subaru Forester 2017-18"),
  CAR.LEGACY_PREGLOBAL: SubaruCarInfo("Subaru Legacy 2015-18"),
  CAR.OUTBACK_PREGLOBAL: SubaruCarInfo("Subaru Outback 2015-17"),
  CAR.OUTBACK_PREGLOBAL_2018: SubaruCarInfo("Subaru Outback 2018-19"),
  CAR.FORESTER_2022: SubaruCarInfo("Subaru Forester 2022", "All", car_parts=CarParts.common([CarHarness.subaru_c])),
  CAR.OUTBACK_2023: SubaruCarInfo("Subaru Outback 2023", "All", car_parts=CarParts.common([CarHarness.subaru_d])),
  CAR.ASCENT_2023: SubaruCarInfo("Subaru Ascent 2023", "All", car_parts=CarParts.common([CarHarness.subaru_d])),
}

SUBARU_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)
SUBARU_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, SUBARU_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, SUBARU_VERSION_RESPONSE],
    ),
    # Some Eyesight modules fail on TESTER_PRESENT_REQUEST
    # TODO: check if this resolves the fingerprinting issue for the 2023 Ascent and other new Subaru cars
    Request(
      [SUBARU_VERSION_REQUEST],
      [SUBARU_VERSION_RESPONSE],
    ),
  ],
)

FW_VERSIONS = {
  CAR.ASCENT: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa5 \x19\x02\x00',
      b'\xa5 !\002\000',
      b'\xf1\x82\xa5 \x19\x02\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x85\xc0\xd0\x00',
      b'\005\xc0\xd0\000',
      b'\x95\xc0\xd0\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00d\xb9\x1f@ \x10',
      b'\000\000e~\037@ \'',
      b'\x00\x00e@\x1f@ $',
      b'\x00\x00d\xb9\x00\x00\x00\x00',
      b'\x00\x00e@\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xbb,\xa0t\a',
      b'\xf1\x82\xbb,\xa0t\x87',
      b'\xf1\x82\xbb,\xa0t\a',
      b'\xf1\x82\xd9,\xa0@\a',
      b'\xf1\x82\xd1,\xa0q\x07',
      b'\xd1,\xa0q\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\x00\xfe\xf7\x00\x00',
      b'\001\xfe\xf9\000\000',
      b'\x01\xfe\xf7\x00\x00',
      b'\x01\xfe\xfa\x00\x00',
    ],
  },
  CAR.ASCENT_2023: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa5 #\x03\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'%\xc0\xd0\x11',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x05!\x08\x1dK\x05!\x08\x01/',
    ],
    (Ecu.engine, 0x7a2, None): [
      b'\xe5,\xa0P\x07',
    ],
    (Ecu.transmission, 0x7a3, None): [
      b'\x04\xfe\xf3\x00\x00',
    ],
  },
  CAR.LEGACY: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa1\\  x04\x01',
      b'\xa1  \x03\x03',
      b'\xa1  \x02\x01',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x9b\xc0\x11\x00',
      b'\x9b\xc0\x11\x02',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00e\x80\x00\x1f@ \x19\x00',
      b'\x00\x00e\x9a\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xde\"a0\x07',
      b'\xe2"aq\x07',
      b'\xde,\xa0@\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xa5\xf6\x05@\x00',
      b'\xa7\xf6\x04@\x00',
      b'\xa5\xfe\xc7@\x00',
    ],
  },
  CAR.IMPREZA: {
    (Ecu.abs, 0x7b0, None): [
      b'\x7a\x94\x3f\x90\x00',
      b'\xa2 \x185\x00',
      b'\xa2 \x193\x00',
      b'\xa2 \x194\x00',
      b'z\x94.\x90\x00',
      b'z\x94\b\x90\x01',
      b'\xa2 \x19`\x00',
      b'z\x94\f\x90\001',
      b'z\x9c\x19\x80\x01',
      b'z\x94\x08\x90\x00',
      b'z\x84\x19\x90\x00',
      b'\xf1\x00\xb2\x06\x04',
      b'z\x94\x0c\x90\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x7a\xc0\x0c\x00',
      b'z\xc0\x08\x00',
      b'\x8a\xc0\x00\x00',
      b'z\xc0\x04\x00',
      b'z\xc0\x00\x00',
      b'\x8a\xc0\x10\x00',
      b'z\xc0\n\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00\x64\xb5\x1f\x40\x20\x0e',
      b'\x00\x00d\xdc\x1f@ \x0e',
      b'\x00\x00e\x1c\x1f@ \x14',
      b'\x00\x00d)\x1f@ \a',
      b'\x00\x00e+\x1f@ \x14',
      b'\000\000e+\000\000\000\000',
      b'\000\000dd\037@ \016',
      b'\000\000e\002\037@ \024',
      b'\x00\x00d)\x00\x00\x00\x00',
      b'\x00\x00c\xf4\x00\x00\x00\x00',
      b'\x00\x00d\xdc\x00\x00\x00\x00',
      b'\x00\x00dd\x00\x00\x00\x00',
      b'\x00\x00c\xf4\x1f@ \x07',
      b'\x00\x00e\x1c\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xaa\x61\x66\x73\x07',
      b'\xbeacr\a',
      b'\xc5!`r\a',
      b'\xaa!ds\a',
      b'\xaa!`u\a',
      b'\xaa!dq\a',
      b'\xaa!dt\a',
      b'\xc5!ar\a',
      b'\xbe!as\a',
      b'\xc5!as\x07',
      b'\xc5!ds\a',
      b'\xc5!`s\a',
      b'\xaa!au\a',
      b'\xbe!at\a',
      b'\xaa\x00Bu\x07',
      b'\xc5!dr\x07',
      b'\xaa!aw\x07',
      b'\xaa!av\x07',
      b'\xaa\x01bt\x07',
      b'\xc5!ap\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xe3\xe5\x46\x31\x00',
      b'\xe4\xe5\x061\x00',
      b'\xe5\xf5\x04\x00\x00',
      b'\xe3\xf5G\x00\x00',
      b'\xe3\xf5\a\x00\x00',
      b'\xe3\xf5C\x00\x00',
      b'\xe5\xf5B\x00\x00',
      b'\xe5\xf5$\000\000',
      b'\xe4\xf5\a\000\000',
      b'\xe3\xf5F\000\000',
      b'\xe4\xf5\002\000\000',
      b'\xe3\xd0\x081\x00',
      b'\xe3\xf5\x06\x00\x00',
      b'\xe3\xd5\x161\x00',
    ],
  },
  CAR.IMPREZA_2020: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa2 \0314\000',
      b'\xa2 \0313\000',
      b'\xa2 !i\000',
      b'\xa2 !`\000',
      b'\xf1\x00\xb2\x06\x04',
      b'\xa2  `\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x9a\xc0\000\000',
      b'\n\xc0\004\000',
      b'\x9a\xc0\x04\x00',
      b'\n\xc0\x04\x01',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\000\000eb\037@ \"',
      b'\x00\x00e\x8f\x1f@ )',
      b'\x00\x00eq\x1f@ "',
      b'\x00\x00eq\x00\x00\x00\x00',
      b'\x00\x00e\x8f\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xca!ap\a',
      b'\xca!`p\a',
      b'\xca!`0\a',
      b'\xcc\"f0\a',
      b'\xcc!fp\a',
      b'\xca!f@\x07',
      b'\xca!fp\x07',
      b'\xf3"f@\x07',
      b'\xe6!fp\x07',
      b'\xf3"fp\x07',
      b'\xe6"f0\x07',
      b'\xe6"fp\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xe6\xf5\004\000\000',
      b'\xe6\xf5$\000\000',
      b'\xe7\xf6B0\000',
      b'\xe7\xf5D0\000',
      b'\xf1\x00\xd7\x10@',
      b'\xe6\xf5D0\x00',
      b'\xe9\xf6F0\x00',
      b'\xe9\xf5B0\x00',
      b'\xe9\xf6B0\x00',
    ],
  },
  CAR.CROSSTREK_HYBRID: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa2 \x19e\x01',
      b'\xa2 !e\x01',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x9a\xc2\x01\x00',
      b'\n\xc2\x01\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00el\x1f@ #',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xd7!`@\x07',
      b'\xd7!`p\a',
      b'\xf4!`0\x07',
    ],
  },
  CAR.FORESTER: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa3 \x18\x14\x00',
      b'\xa3  \024\000',
      b'\xa3 \031\024\000',
      b'\xa3  \x14\x01',
      b'\xf1\x00\xbb\r\x05',
      b'\xa3 \x18&\x00',
      b'\xa3 \x19&\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x8d\xc0\x04\x00',
      b'\x8d\xc0\x00\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00e!\x1f@ \x11',
      b'\x00\x00e\x97\x1f@ 0',
      b'\000\000e`\037@  ',
      b'\xf1\x00\xac\x02\x00',
      b'\x00\x00e!\x00\x00\x00\x00',
      b'\x00\x00e\x97\x00\x00\x00\x00',
      b'\x00\x00e^\x1f@ !',
      b'\x00\x00e^\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xb6"`A\x07',
      b'\xcf"`0\x07',
      b'\xcb\"`@\a',
      b'\xcb\"`p\a',
      b'\xf1\x00\xa2\x10\n',
      b'\xcf"`p\x07',
      b'\xb6\xa2`A\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\032\xf6B0\000',
      b'\x1a\xf6F`\x00',
      b'\032\xf6b`\000',
      b'\x1a\xf6B`\x00',
      b'\x1a\xf6b0\x00',
      b'\x1a\xe6B1\x00',
      b'\x1a\xe6F1\x00',
    ],
  },
  CAR.FORESTER_HYBRID: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa3 \x19T\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x8d\xc2\x00\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00eY\x1f@ !',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xd2\xa1`r\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
       b'\x1b\xa7@a\x00',
    ],
  },
  CAR.FORESTER_PREGLOBAL: {
    (Ecu.abs, 0x7b0, None): [
      b'\x7d\x97\x14\x40',
      b'\xf1\x00\xbb\x0c\x04',
    ],
    (Ecu.eps, 0x746, None): [
      b'}\xc0\x10\x00',
      b'm\xc0\x10\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00\x64\x35\x1f\x40\x20\x09',
      b'\x00\x00c\xe9\x1f@ \x03',
      b'\x00\x00d\xd3\x1f@ \t'
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xba"@p\a',
      b'\xa7)\xa0q\a',
      b'\xf1\x82\xa7)\xa0q\a',
      b'\xba"@@\a',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xdc\xf2\x60\x60\x00',
      b'\xdc\xf2@`\x00',
      b'\xda\xfd\xe0\x80\x00',
      b'\xdc\xf2`\x81\000',
      b'\xdc\xf2`\x80\x00',
      b'\x1a\xf6F`\x00',
    ],
  },
  CAR.LEGACY_PREGLOBAL: {
    (Ecu.abs, 0x7b0, None): [
      b'k\x97D\x00',
      b'[\xba\xc4\x03',
      b'{\x97D\x00',
      b'[\x97D\000',
    ],
    (Ecu.eps, 0x746, None): [
      b'[\xb0\x00\x01',
      b'K\xb0\x00\x01',
      b'k\xb0\x00\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00c\xb7\x1f@\x10\x16',
      b'\x00\x00c\x94\x1f@\x10\x08',
      b'\x00\x00c\xec\x1f@ \x04',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xab*@r\a',
      b'\xa0+@p\x07',
      b'\xb4"@0\x07',
      b'\xa0"@q\a',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xbe\xf2\x00p\x00',
      b'\xbf\xfb\xc0\x80\x00',
      b'\xbd\xf2\x00`\x00',
      b'\xbf\xf2\000\x80\000',
    ],
  },
  CAR.OUTBACK_PREGLOBAL: {
    (Ecu.abs, 0x7b0, None): [
      b'{\x9a\xac\x00',
      b'k\x97\xac\x00',
      b'\x5b\xf7\xbc\x03',
      b'[\xf7\xac\x03',
      b'{\x97\xac\x00',
      b'k\x9a\xac\000',
      b'[\xba\xac\x03',
      b'[\xf7\xac\000',
    ],
    (Ecu.eps, 0x746, None): [
      b'k\xb0\x00\x00',
      b'[\xb0\x00\x00',
      b'\x4b\xb0\x00\x02',
      b'K\xb0\x00\x00',
      b'{\xb0\x00\x01',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00c\xec\x1f@ \x04',
      b'\x00\x00c\xd1\x1f@\x10\x17',
      b'\xf1\x00\xf0\xe0\x0e',
      b'\x00\x00c\x94\x00\x00\x00\x00',
      b'\x00\x00c\x94\x1f@\x10\b',
      b'\x00\x00c\xb7\x1f@\x10\x16',
      b'\000\000c\x90\037@\020\016',
      b'\x00\x00c\xec\x37@\x04',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xb4+@p\a',
      b'\xab\"@@\a',
      b'\xa0\x62\x41\x71\x07',
      b'\xa0*@q\a',
      b'\xab*@@\a',
      b'\xb4"@0\a',
      b'\xb4"@p\a',
      b'\xab"@s\a',
      b'\xab+@@\a',
      b'\xb4"@r\a',
      b'\xa0+@@\x07',
      b'\xa0\"@\x80\a',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xbd\xfb\xe0\x80\x00',
      b'\xbe\xf2@\x80\x00',
      b'\xbf\xe2\x40\x80\x00',
      b'\xbf\xf2@\x80\x00',
      b'\xbe\xf2@p\x00',
      b'\xbd\xf2@`\x00',
      b'\xbd\xf2@\x81\000',
      b'\xbe\xfb\xe0p\000',
      b'\xbf\xfb\xe0b\x00',
    ],
  },
  CAR.OUTBACK_PREGLOBAL_2018: {
    (Ecu.abs, 0x7b0, None): [
      b'\x8b\x97\xac\x00',
      b'\x8b\x9a\xac\x00',
      b'\x9b\x97\xac\x00',
      b'\x8b\x97\xbc\x00',
      b'\x8b\x99\xac\x00',
      b'\x9b\x9a\xac\000',
      b'\x9b\x97\xbe\x10',
    ],
    (Ecu.eps, 0x746, None): [
      b'{\xb0\x00\x00',
      b'{\xb0\x00\x01',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00df\x1f@ \n',
      b'\x00\x00d\xfe\x1f@ \x15',
      b'\x00\x00d\x95\x00\x00\x00\x00',
      b'\x00\x00d\x95\x1f@ \x0f',
      b'\x00\x00d\xfe\x00\x00\x00\x00',
      b'\x00\x00e\x19\x1f@ \x15',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xb5"@p\a',
      b'\xb5+@@\a',
      b'\xb5"@P\a',
      b'\xc4"@0\a',
      b'\xb5b@1\x07',
      b'\xb5q\xe0@\a',
      b'\xc4+@0\a',
      b'\xc4b@p\a',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xbc\xf2@\x81\x00',
      b'\xbc\xfb\xe0\x80\x00',
      b'\xbc\xf2@\x80\x00',
      b'\xbb\xf2@`\x00',
      b'\xbc\xe2@\x80\x00',
      b'\xbc\xfb\xe0`\x00',
      b'\xbc\xaf\xe0`\x00',
      b'\xbb\xfb\xe0`\000',
    ],
  },
  CAR.OUTBACK: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa1  \x06\x01',
      b'\xa1  \a\x00',
      b'\xa1  \b\001',
      b'\xa1  \x06\x00',
      b'\xa1 "\t\x01',
      b'\xa1  \x08\x02',
      b'\xa1 \x06\x02',
      b'\xa1  \x07\x02',
      b'\xa1  \x08\x00',
      b'\xa1 "\t\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x9b\xc0\x10\x00',
      b'\x9b\xc0\x20\x00',
      b'\x1b\xc0\x10\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00eJ\x00\x1f@ \x19\x00',
      b'\000\000e\x80\000\037@ \031\000',
      b'\x00\x00e\x9a\x00\x00\x00\x00\x00\x00',
      b'\x00\x00e\x9a\x00\x1f@ 1\x00',
      b'\x00\x00eJ\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xbc,\xa0q\x07',
      b'\xbc\"`@\a',
      b'\xde"`0\a',
      b'\xf1\x82\xbc,\xa0q\a',
      b'\xf1\x82\xe3,\xa0@\x07',
      b'\xe2"`0\x07',
      b'\xe2"`p\x07',
      b'\xf1\x82\xe2,\xa0@\x07',
      b'\xbc"`q\x07',
      b'\xe3,\xa0@\x07',
      b'\xbc,\xa0u\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xa5\xfe\xf7@\x00',
      b'\xa5\xf6D@\x00',
      b'\xa5\xfe\xf6@\x00',
      b'\xa7\x8e\xf40\x00',
      b'\xf1\x82\xa7\xf6D@\x00',
      b'\xa7\xf6D@\x00',
      b'\xa7\xfe\xf4@\x00',
      b'\xa5\xfe\xf8@\x00',
    ],
  },
  CAR.FORESTER_2022: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa3 !x\x00',
      b'\xa3 !v\x00',
      b'\xa3 "v\x00',
      b'\xa3 "x\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'-\xc0%0',
      b'-\xc0\x040',
      b'=\xc0%\x02',
      b'=\xc04\x02',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x04!\x01\x1eD\x07!\x00\x04,'
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xd5"a0\x07',
      b'\xd5"`0\x07',
      b'\xf1"aq\x07',
      b'\xf1"`q\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\x1d\x86B0\x00',
      b'\x1d\xf6B0\x00',
      b'\x1e\x86B0\x00',
      b'\x1e\xf6D0\x00',
    ],
  },
  CAR.OUTBACK_2023: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa1 #\x17\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'+\xc0\x12\x11\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\t!\x08\x046\x05!\x08\x01/',
    ],
    (Ecu.engine, 0x7a2, None): [
      b'\xed,\xa2q\x07',
    ],
    (Ecu.transmission, 0x7a3, None): [
      b'\xa8\x8e\xf41\x00',
    ]
  }
}

DBC = {
  CAR.ASCENT: dbc_dict('subaru_global_2017_generated', None),
  CAR.ASCENT_2023: dbc_dict('subaru_global_2017_generated', None),
  CAR.IMPREZA: dbc_dict('subaru_global_2017_generated', None),
  CAR.IMPREZA_2020: dbc_dict('subaru_global_2017_generated', None),
  CAR.FORESTER: dbc_dict('subaru_global_2017_generated', None),
  CAR.FORESTER_2022: dbc_dict('subaru_global_2017_generated', None),
  CAR.OUTBACK: dbc_dict('subaru_global_2017_generated', None),
  CAR.FORESTER_HYBRID: dbc_dict('subaru_global_2020_hybrid_generated', None),
  CAR.CROSSTREK_HYBRID: dbc_dict('subaru_global_2020_hybrid_generated', None),
  CAR.OUTBACK_2023: dbc_dict('subaru_global_2017_generated', None),
  CAR.LEGACY: dbc_dict('subaru_global_2017_generated', None),
  CAR.FORESTER_PREGLOBAL: dbc_dict('subaru_forester_2017_generated', None),
  CAR.LEGACY_PREGLOBAL: dbc_dict('subaru_outback_2015_generated', None),
  CAR.OUTBACK_PREGLOBAL: dbc_dict('subaru_outback_2015_generated', None),
  CAR.OUTBACK_PREGLOBAL_2018: dbc_dict('subaru_outback_2019_generated', None),
}

LKAS_ANGLE = {CAR.FORESTER_2022, CAR.OUTBACK_2023, CAR.ASCENT_2023}
GLOBAL_GEN2 = {CAR.OUTBACK, CAR.LEGACY, CAR.OUTBACK_2023, CAR.ASCENT_2023}
PREGLOBAL_CARS = {CAR.FORESTER_PREGLOBAL, CAR.LEGACY_PREGLOBAL, CAR.OUTBACK_PREGLOBAL, CAR.OUTBACK_PREGLOBAL_2018}
HYBRID_CARS = {CAR.CROSSTREK_HYBRID, CAR.FORESTER_HYBRID}
