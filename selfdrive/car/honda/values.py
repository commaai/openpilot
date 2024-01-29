from dataclasses import dataclass
from enum import Enum, IntFlag, StrEnum
from typing import Dict, List, Optional, Union

from cereal import car
from openpilot.common.conversions import Conversions as CV
from panda.python import uds
from openpilot.selfdrive.car import dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarInfo, CarParts, Column
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries, p16

Ecu = car.CarParams.Ecu
VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarControllerParams:
  # Allow small margin below -3.5 m/s^2 from ISO 15622:2018 since we
  # perform the closed loop control, and might need some
  # to apply some more braking if we're on a downhill slope.
  # Our controller should still keep the 2 second average above
  # -3.5 m/s^2 as per planner limits
  NIDEC_ACCEL_MIN = -4.0  # m/s^2
  NIDEC_ACCEL_MAX = 1.6  # m/s^2, lower than 2.0 m/s^2 for tuning reasons

  NIDEC_ACCEL_LOOKUP_BP = [-1., 0., .6]
  NIDEC_ACCEL_LOOKUP_V = [-4.8, 0., 2.0]

  NIDEC_MAX_ACCEL_V = [0.5, 2.4, 1.4, 0.6]
  NIDEC_MAX_ACCEL_BP = [0.0, 4.0, 10., 20.]

  NIDEC_GAS_MAX = 198  # 0xc6
  NIDEC_BRAKE_MAX = 1024 // 4

  BOSCH_ACCEL_MIN = -3.5  # m/s^2
  BOSCH_ACCEL_MAX = 2.0  # m/s^2

  BOSCH_GAS_LOOKUP_BP = [-0.2, 2.0]  # 2m/s^2
  BOSCH_GAS_LOOKUP_V = [0, 1600]

  def __init__(self, CP):
    self.STEER_MAX = CP.lateralParams.torqueBP[-1]
    # mirror of list (assuming first item is zero) for interp of signed request values
    assert(CP.lateralParams.torqueBP[0] == 0)
    assert(CP.lateralParams.torqueBP[0] == 0)
    self.STEER_LOOKUP_BP = [v * -1 for v in CP.lateralParams.torqueBP][1:][::-1] + list(CP.lateralParams.torqueBP)
    self.STEER_LOOKUP_V = [v * -1 for v in CP.lateralParams.torqueV][1:][::-1] + list(CP.lateralParams.torqueV)


class HondaFlags(IntFlag):
  # Bosch models with alternate set of LKAS_HUD messages
  BOSCH_EXT_HUD = 1


# Car button codes
class CruiseButtons:
  RES_ACCEL = 4
  DECEL_SET = 3
  CANCEL = 2
  MAIN = 1


# See dbc files for info on values
VISUAL_HUD = {
  VisualAlert.none: 0,
  VisualAlert.fcw: 1,
  VisualAlert.steerRequired: 1,
  VisualAlert.ldw: 1,
  VisualAlert.brakePressed: 10,
  VisualAlert.wrongGear: 6,
  VisualAlert.seatbeltUnbuckled: 5,
  VisualAlert.speedTooHigh: 8
}


class CAR(StrEnum):
  ACCORD = "HONDA ACCORD 2018"
  ACCORDH = "HONDA ACCORD HYBRID 2018"
  CIVIC = "HONDA CIVIC 2016"
  CIVIC_BOSCH = "HONDA CIVIC (BOSCH) 2019"
  CIVIC_BOSCH_DIESEL = "HONDA CIVIC SEDAN 1.6 DIESEL 2019"
  CIVIC_2022 = "HONDA CIVIC 2022"
  ACURA_ILX = "ACURA ILX 2016"
  CRV = "HONDA CR-V 2016"
  CRV_5G = "HONDA CR-V 2017"
  CRV_EU = "HONDA CR-V EU 2016"
  CRV_HYBRID = "HONDA CR-V HYBRID 2019"
  FIT = "HONDA FIT 2018"
  FREED = "HONDA FREED 2020"
  HRV = "HONDA HRV 2019"
  HRV_3G = "HONDA HR-V 2023"
  ODYSSEY = "HONDA ODYSSEY 2018"
  ODYSSEY_CHN = "HONDA ODYSSEY CHN 2019"
  ACURA_RDX = "ACURA RDX 2018"
  ACURA_RDX_3G = "ACURA RDX 2020"
  PILOT = "HONDA PILOT 2017"
  RIDGELINE = "HONDA RIDGELINE 2017"
  INSIGHT = "HONDA INSIGHT 2019"
  HONDA_E = "HONDA E 2020"


class Footnote(Enum):
  CIVIC_DIESEL = CarFootnote(
    "2019 Honda Civic 1.6L Diesel Sedan does not have ALC below 12mph.",
    Column.FSR_STEERING)


@dataclass
class HondaCarInfo(CarInfo):
  package: str = "Honda Sensing"

  def init_make(self, CP: car.CarParams):
    if CP.carFingerprint in HONDA_BOSCH:
      self.car_parts = CarParts.common([CarHarness.bosch_b]) if CP.carFingerprint in HONDA_BOSCH_RADARLESS else CarParts.common([CarHarness.bosch_a])
    else:
      self.car_parts = CarParts.common([CarHarness.nidec])


CAR_INFO: Dict[str, Optional[Union[HondaCarInfo, List[HondaCarInfo]]]] = {
  CAR.ACCORD: [
    HondaCarInfo("Honda Accord 2018-22", "All", video_link="https://www.youtube.com/watch?v=mrUwlj3Mi58", min_steer_speed=3. * CV.MPH_TO_MS),
    HondaCarInfo("Honda Inspire 2018", "All", min_steer_speed=3. * CV.MPH_TO_MS),
  ],
  CAR.ACCORDH: HondaCarInfo("Honda Accord Hybrid 2018-22", "All", min_steer_speed=3. * CV.MPH_TO_MS),
  CAR.CIVIC: HondaCarInfo("Honda Civic 2016-18", min_steer_speed=12. * CV.MPH_TO_MS, video_link="https://youtu.be/-IkImTe1NYE"),
  CAR.CIVIC_BOSCH: [
    HondaCarInfo("Honda Civic 2019-21", "All", video_link="https://www.youtube.com/watch?v=4Iz1Mz5LGF8",
                 footnotes=[Footnote.CIVIC_DIESEL], min_steer_speed=2. * CV.MPH_TO_MS),
    HondaCarInfo("Honda Civic Hatchback 2017-21", min_steer_speed=12. * CV.MPH_TO_MS),
  ],
  CAR.CIVIC_BOSCH_DIESEL: None,  # same platform
  CAR.CIVIC_2022: [
    HondaCarInfo("Honda Civic 2022-23", "All", video_link="https://youtu.be/ytiOT5lcp6Q"),
    HondaCarInfo("Honda Civic Hatchback 2022-23", "All", video_link="https://youtu.be/ytiOT5lcp6Q"),
  ],
  CAR.ACURA_ILX: HondaCarInfo("Acura ILX 2016-19", "AcuraWatch Plus", min_steer_speed=25. * CV.MPH_TO_MS),
  CAR.CRV: HondaCarInfo("Honda CR-V 2015-16", "Touring Trim", min_steer_speed=12. * CV.MPH_TO_MS),
  CAR.CRV_5G: HondaCarInfo("Honda CR-V 2017-22", min_steer_speed=12. * CV.MPH_TO_MS),
  CAR.CRV_EU: None,  # HondaCarInfo("Honda CR-V EU", "Touring"),  # Euro version of CRV Touring
  CAR.CRV_HYBRID: HondaCarInfo("Honda CR-V Hybrid 2017-20", min_steer_speed=12. * CV.MPH_TO_MS),
  CAR.FIT: HondaCarInfo("Honda Fit 2018-20", min_steer_speed=12. * CV.MPH_TO_MS),
  CAR.FREED: HondaCarInfo("Honda Freed 2020", min_steer_speed=12. * CV.MPH_TO_MS),
  CAR.HRV: HondaCarInfo("Honda HR-V 2019-22", min_steer_speed=12. * CV.MPH_TO_MS),
  CAR.HRV_3G: HondaCarInfo("Honda HR-V 2023", "All"),
  CAR.ODYSSEY: HondaCarInfo("Honda Odyssey 2018-20"),
  CAR.ODYSSEY_CHN: None,  # Chinese version of Odyssey
  CAR.ACURA_RDX: HondaCarInfo("Acura RDX 2016-18", "AcuraWatch Plus", min_steer_speed=12. * CV.MPH_TO_MS),
  CAR.ACURA_RDX_3G: HondaCarInfo("Acura RDX 2019-22", "All", min_steer_speed=3. * CV.MPH_TO_MS),
  CAR.PILOT: [
    HondaCarInfo("Honda Pilot 2016-22", min_steer_speed=12. * CV.MPH_TO_MS),
    HondaCarInfo("Honda Passport 2019-23", "All", min_steer_speed=12. * CV.MPH_TO_MS),
  ],
  CAR.RIDGELINE: HondaCarInfo("Honda Ridgeline 2017-23", min_steer_speed=12. * CV.MPH_TO_MS),
  CAR.INSIGHT: HondaCarInfo("Honda Insight 2019-22", "All", min_steer_speed=3. * CV.MPH_TO_MS),
  CAR.HONDA_E: HondaCarInfo("Honda e 2020", "All", min_steer_speed=3. * CV.MPH_TO_MS),
}

HONDA_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xF112)
HONDA_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(0xF112)

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # Currently used to fingerprint
    Request(
      [StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.UDS_VERSION_RESPONSE],
      bus=1,
    ),

    # Data collection requests:
    # Attempt to get the radarless Civic 2022+ camera FW
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
      logging=True
    ),
    # Log extra identifiers for current ECUs
    Request(
      [HONDA_VERSION_REQUEST],
      [HONDA_VERSION_RESPONSE],
      bus=1,
      logging=True,
    ),
    # Nidec PT bus
    Request(
      [StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
      logging=True,
    ),
    # Bosch PT bus
    Request(
      [StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.UDS_VERSION_RESPONSE],
      bus=1,
      logging=True,
      obd_multiplexing=False,
    ),
  ],
  extra_ecus=[
    # The only other ECU on PT bus accessible by camera on radarless Civic
    (Ecu.unknown, 0x18DAB3F1, None),
  ],
)


DBC = {
  CAR.ACCORD: dbc_dict('honda_accord_2018_can_generated', None),
  CAR.ACCORDH: dbc_dict('honda_accord_2018_can_generated', None),
  CAR.ACURA_ILX: dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.ACURA_RDX: dbc_dict('acura_rdx_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.ACURA_RDX_3G: dbc_dict('acura_rdx_2020_can_generated', None),
  CAR.CIVIC: dbc_dict('honda_civic_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CIVIC_BOSCH: dbc_dict('honda_civic_hatchback_ex_2017_can_generated', None),
  CAR.CIVIC_BOSCH_DIESEL: dbc_dict('honda_accord_2018_can_generated', None),
  CAR.CRV: dbc_dict('honda_crv_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CRV_5G: dbc_dict('honda_crv_ex_2017_can_generated', None, body_dbc='honda_crv_ex_2017_body_generated'),
  CAR.CRV_EU: dbc_dict('honda_crv_executive_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CRV_HYBRID: dbc_dict('honda_accord_2018_can_generated', None),
  CAR.FIT: dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.FREED: dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.HRV: dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.HRV_3G: dbc_dict('honda_civic_ex_2022_can_generated', None),
  CAR.ODYSSEY: dbc_dict('honda_odyssey_exl_2018_generated', 'acura_ilx_2016_nidec'),
  CAR.ODYSSEY_CHN: dbc_dict('honda_odyssey_extreme_edition_2018_china_can_generated', 'acura_ilx_2016_nidec'),
  CAR.PILOT: dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.RIDGELINE: dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.INSIGHT: dbc_dict('honda_insight_ex_2019_can_generated', None),
  CAR.HONDA_E: dbc_dict('acura_rdx_2020_can_generated', None),
  CAR.CIVIC_2022: dbc_dict('honda_civic_ex_2022_can_generated', None),
}

STEER_THRESHOLD = {
  # default is 1200, overrides go here
  CAR.ACURA_RDX: 400,
  CAR.CRV_EU: 400,
}

HONDA_NIDEC_ALT_PCM_ACCEL = {CAR.ODYSSEY}
HONDA_NIDEC_ALT_SCM_MESSAGES = {CAR.ACURA_ILX, CAR.ACURA_RDX, CAR.CRV, CAR.CRV_EU, CAR.FIT, CAR.FREED, CAR.HRV, CAR.ODYSSEY_CHN,
                                CAR.PILOT, CAR.RIDGELINE}
HONDA_BOSCH = {CAR.ACCORD, CAR.ACCORDH, CAR.CIVIC_BOSCH, CAR.CIVIC_BOSCH_DIESEL, CAR.CRV_5G,
               CAR.CRV_HYBRID, CAR.INSIGHT, CAR.ACURA_RDX_3G, CAR.HONDA_E, CAR.CIVIC_2022, CAR.HRV_3G}
HONDA_BOSCH_ALT_BRAKE_SIGNAL = {CAR.ACCORD, CAR.CRV_5G, CAR.ACURA_RDX_3G, CAR.HRV_3G}
HONDA_BOSCH_RADARLESS = {CAR.CIVIC_2022, CAR.HRV_3G}
