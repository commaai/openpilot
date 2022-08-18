from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo, Harness
Ecu = car.CarParams.Ecu


class CAR:
  # Chrysler
  PACIFICA_2017_HYBRID = "CHRYSLER PACIFICA HYBRID 2017"
  PACIFICA_2018_HYBRID = "CHRYSLER PACIFICA HYBRID 2018"
  PACIFICA_2019_HYBRID = "CHRYSLER PACIFICA HYBRID 2019"
  PACIFICA_2018 = "CHRYSLER PACIFICA 2018"
  PACIFICA_2020 = "CHRYSLER PACIFICA 2020"

  # Jeep
  JEEP_CHEROKEE = "JEEP GRAND CHEROKEE V6 2018"   # includes 2017 Trailhawk
  JEEP_CHEROKEE_2019 = "JEEP GRAND CHEROKEE 2019" # includes 2020 Trailhawk

  # Ram
  RAM_1500 = "RAM 1500 5TH GEN"
  RAM_HD = "RAM HD 5TH GEN"


class CarControllerParams:
  def __init__(self, CP):
    self.STEER_ERROR_MAX = 80
    if CP.carFingerprint in RAM_HD:
      self.STEER_DELTA_UP = 14
      self.STEER_DELTA_DOWN = 14
      self.STEER_MAX = 361  # higher than this faults the EPS
    elif CP.carFingerprint in RAM_DT:
      self.STEER_DELTA_UP = 6
      self.STEER_DELTA_DOWN = 6
      self.STEER_MAX = 261  # EPS allows more, up to 350?
    else:
      self.STEER_DELTA_UP = 3
      self.STEER_DELTA_DOWN = 3
      self.STEER_MAX = 261  # higher than this faults the EPS

STEER_THRESHOLD = 120

RAM_DT = {CAR.RAM_1500, }
RAM_HD = {CAR.RAM_HD, }
RAM_CARS = RAM_DT | RAM_HD

@dataclass
class ChryslerCarInfo(CarInfo):
  package: str = "Adaptive Cruise Control"
  harness: Enum = Harness.fca

CAR_INFO: Dict[str, Optional[Union[ChryslerCarInfo, List[ChryslerCarInfo]]]] = {
  CAR.PACIFICA_2017_HYBRID: ChryslerCarInfo("Chrysler Pacifica Hybrid 2017-18"),
  CAR.PACIFICA_2018_HYBRID: None,  # same platforms
  CAR.PACIFICA_2019_HYBRID: ChryslerCarInfo("Chrysler Pacifica Hybrid 2019-22"),
  CAR.PACIFICA_2018: ChryslerCarInfo("Chrysler Pacifica 2017-18"),
  CAR.PACIFICA_2020: [
    ChryslerCarInfo("Chrysler Pacifica 2019-20"),
    ChryslerCarInfo("Chrysler Pacifica 2021", package="All"),
  ],
  CAR.JEEP_CHEROKEE: ChryslerCarInfo("Jeep Grand Cherokee 2016-18", video_link="https://www.youtube.com/watch?v=eLR9o2JkuRk"),
  CAR.JEEP_CHEROKEE_2019: ChryslerCarInfo("Jeep Grand Cherokee 2019-21", video_link="https://www.youtube.com/watch?v=jBe4lWnRSu4"),
  CAR.RAM_1500: ChryslerCarInfo("Ram 1500 2019-22", harness=Harness.ram),
  CAR.RAM_HD: [
    ChryslerCarInfo("Ram 2500 2020-22", harness=Harness.ram),
    ChryslerCarInfo("Ram 3500 2020-22", harness=Harness.ram),
  ],
}

FW_VERSIONS = {
  CAR.PACIFICA_2017_HYBRID: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68239262AH',
    ],
    (Ecu.srs, 0x744, None): [
      b'68238840AH',
    ],
    (Ecu.fwdCamera, 0x764, None): [
      b'68223694AG',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'68226356AI',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68288309AC',
    ],
  },

  CAR.PACIFICA_2018_HYBRID: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68358439AE',
    ],
    (Ecu.srs, 0x744, None): [
      b'68405939AA',
      b'68358990AC',
    ],
    (Ecu.fwdCamera, 0x764, None): [
      b'04672731AB',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672758AA',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68288309AD',
    ],
    (Ecu.gateway, 0x18dacbf1, None): [
      b'68398623AA',
    ],
  },

  CAR.PACIFICA_2019_HYBRID: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68434960AE',
      b'68529064AB',
      b'68434956AD',
      b'68434956AC',
    ],
    (Ecu.srs, 0x744, None): [
      b'68526665AB',
      b'68405567AC',
      b'68453076AD',
      b'68480710AC',
    ],
    (Ecu.fwdCamera, 0x764, None): [
      b'68551091AA',
      b'04672776AA',
      b'68493395AA',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'68417813AF',
      b'68540436AB',
      b'68540436AA',
      b'68540436AC',
      b'04672758AB',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68460392AA',
      b'68525339AA',
    ],
    (Ecu.gateway, 0x18dacbf1, None): [
      b'68488429AB',
      b'68525871AB',
      b'68447193AB',
    ],
  },

  CAR.PACIFICA_2018: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68360252AC',
      b'68227902AH',
      b'68227902AG',
    ],
    (Ecu.srs, 0x744, None): [
      b'68211617AG',
      b'68405937AA',
      b'68358974AC',
    ],
    (Ecu.esp, 0x747, None): [
      b'68352227AA',
      b'68330876AB',
      b'68330876AA',
    ],
    (Ecu.fwdCamera, 0x764, None): [
      b'04672731AB',
      b'68223694AG',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672758AA',
      b'68226356AI',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68378884AA',
      b'68288891AE',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'68367471AC',
      b'68277372AD',
      b'68380571AB',
    ],
    (Ecu.gateway, 0x18dacbf1, None): [
      b'68398623AA',
    ],
  },

  CAR.PACIFICA_2020: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68405327AC',
      b'68436233AC',
    ],
    (Ecu.srs, 0x744, None): [
      b'68405565AC',
      b'68453074AC',
      b'68405565AB',
    ],
    (Ecu.esp, 0x747, None): [
      b'68453575AF',
      b'68433480AB',
      b'68397394AA',
    ],
    (Ecu.fwdCamera, 0x764, None): [
      b'68493395AA',
      b'04672776AA',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672758AA',
      b'68417813AF',
      b'04672758AB',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68416742AA',
      b'68460393AA',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'68443154AB',
      b'68414271AC',
      b'68443158AB',
      b'68443158AC',
      b'68414271AD',
    ],
    (Ecu.gateway, 0x18dacbf1, None): [
      b'68420261AA',
      b'68447193AB',
    ],
  },

  CAR.JEEP_CHEROKEE: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68331511AC',
      b'68331512AC',
      b'68331687AC',
      b'68331574AC',
    ],
    (Ecu.srs, 0x744, None): [
      b'68355363AB',
    ],
    (Ecu.esp, 0x747, None): [
      b'68336276AB',
    ],
    (Ecu.fwdCamera, 0x764, None): [
      b'04672631AC',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672627AC',
      b'04672627AB',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68321644AC',
      b'68321646AC',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'68361911AH',
      b'68361916AE',
      b'68361916AD',
      b'68361911AE',
    ],
    (Ecu.gateway, 0x18dacbf1, None): [
      b'68398627AA',
    ],
  },

  CAR.JEEP_CHEROKEE_2019: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68454144AD',
      b'68424747AB',
    ],
    (Ecu.srs, 0x744, None): [
      b'68355363AB',
    ],
    (Ecu.esp, 0x747, None): [
      b'68408639AD',
      b'68408639AC',
    ],
    (Ecu.fwdCamera, 0x764, None): [
      b'04672774AA',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672788AA',
      b'68456722AC',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68453433AA',
      b'68417279AA',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'68419672AC',
      b'68495807AA',
    ],
    (Ecu.gateway, 0x18dacbf1, None): [
      b'68445288AB',
      b'68419206AA',
    ],
  },

  CAR.RAM_1500: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68294051AI',
      b'68453513AD',
      b'68527375AD',
      b'68453503AD',
      b'68453471AC',
      b'68294063AH',
      b'68294063AG',
      b'68434858AC',
      b'68434860AC',
      b'68453473AD',
      b'68453503AC',
      b'68453505AD',
      b'68453514AD',
      b'68434846AC',
      b'68294063AI',
    ],
    (Ecu.srs, 0x744, None): [
      b'68473844AB',
      b'68490898AA',
      b'68428609AB',
      b'68441329AB',
      b'68500728AA',
    ],
    (Ecu.esp, 0x747, None): [
      b'68438456AF',
      b'68432418AB',
      b'68535469AB',
      b'68432418AD',
      b'68438454AD',
      b'68436004AE',
      b'68438454AC',
      b'68436004AD',
      b'68535470AC',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'68454268AB',
      b'04672892AB',
      b'68320950AL',
      b'68475160AF',
      b'68320950AN',
      b'68475160AE',
      b'68475160AG',
      b'68320950AM',
      b'68320950AJ',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68469901AA',
      b'68273275AG',
      b'68273275AH',
      b'68440789AC',
      b'68552788AA',
      b'68466110AA',
      b'68466110AB',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'68448163AJ',
      b'68500630AD',
      b'68539650AD',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'68360066AH',
      b'68466083AB',
      b'68360086AM',
      b'68360085AL',
      b'68360081AM',
      b'68484470AC',
      b'68540431AB',
      b'68502994AD',
      b'68384328AD',
      b'68360080AM',
      b'68445533AB',
      b'68484467AC',
      b'68360078AL',
      b'05035706AE',
    ],
    (Ecu.gateway, 0x18dacbf1, None): [
      b'68445283AB',
      b'68533631AB',
      b'68500483AB',
      b'68402660AB',
    ],
  },

  CAR.RAM_HD: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68361606AH',
      b'68492693AD',
    ],
    (Ecu.srs, 0x744, None): [
      b'68399794AC',
      b'68428503AA',
      b'68428505AA',
    ],
    (Ecu.esp, 0x747, None): [
      b'68334977AH',
      b'68504022AB',
      b'68530686AB',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672895AB',
      b'56029827AG',
      b'68484694AE',
    ],
    (Ecu.eps, 0x761, None): [
      b'68421036AC',
      b'68507906AB',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'52421132AF',
      b'M2370131MB',
      b'M2421132MB',
    ],
    (Ecu.gateway, 0x18DACBF1, None): [
      b'68488419AB',
      b'68535476AB',
    ],
  },
}

DBC = {
  CAR.PACIFICA_2017_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2018: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2020: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2018_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2019_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.JEEP_CHEROKEE: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.JEEP_CHEROKEE_2019: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.RAM_1500: dbc_dict('chrysler_ram_dt_generated', None),
  CAR.RAM_HD: dbc_dict('chrysler_ram_hd_generated', None),
}
