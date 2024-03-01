from dataclasses import dataclass
from enum import Enum, IntFlag

from cereal import car
from openpilot.common.conversions import Conversions as CV
from panda.python import uds
from openpilot.selfdrive.car import CarSpecs, PlatformConfig, Platforms, dbc_dict
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
  # Detected flags
  # Bosch models with alternate set of LKAS_HUD messages
  BOSCH_EXT_HUD = 1
  BOSCH_ALT_BRAKE = 2

  # Static flags
  BOSCH = 4
  BOSCH_RADARLESS = 8

  NIDEC = 16
  NIDEC_ALT_PCM_ACCEL = 32
  NIDEC_ALT_SCM_MESSAGES = 64

  AUTORESUME_SNG = 128
  ELECTRIC_PARKING_BRAKE = 256

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


@dataclass
class HondaCarInfo(CarInfo):
  package: str = "Honda Sensing"

  def init_make(self, CP: car.CarParams):
    if CP.flags & HondaFlags.BOSCH:
      self.car_parts = CarParts.common([CarHarness.bosch_b]) if CP.flags & HondaFlags.BOSCH_RADARLESS else CarParts.common([CarHarness.bosch_a])
    else:
      self.car_parts = CarParts.common([CarHarness.nidec])


class Footnote(Enum):
  CIVIC_DIESEL = CarFootnote(
    "2019 Honda Civic 1.6L Diesel Sedan does not have ALC below 12mph.",
    Column.FSR_STEERING)


class HondaPlatformConfig(PlatformConfig):
  def init(self):
    if self.flags & HondaFlags.BOSCH:
      self.flags |= HondaFlags.AUTORESUME_SNG
      self.flags |= HondaFlags.ELECTRIC_PARKING_BRAKE


class CAR(Platforms):
  # Bosch Cars
  ACCORD = HondaPlatformConfig(
    "HONDA ACCORD 2018",
    [
      HondaCarInfo("Honda Accord 2018-22", "All", video_link="https://www.youtube.com/watch?v=mrUwlj3Mi58", min_steer_speed=3. * CV.MPH_TO_MS),
      HondaCarInfo("Honda Inspire 2018", "All", min_steer_speed=3. * CV.MPH_TO_MS),
      HondaCarInfo("Honda Accord Hybrid 2018-22", "All", min_steer_speed=3. * CV.MPH_TO_MS),
    ],
    dbc_dict('honda_accord_2018_can_generated', None),
    specs=CarSpecs(mass=3279 * CV.LB_TO_KG, wheelbase=2.83, steerRatio=16.33, centerToFrontRatio=0.39),  # steerRatio: 11.82 is spec end-to-end
    flags=HondaFlags.BOSCH,
  )
  CIVIC_BOSCH = HondaPlatformConfig(
    "HONDA CIVIC (BOSCH) 2019",
    [
      HondaCarInfo("Honda Civic 2019-21", "All", video_link="https://www.youtube.com/watch?v=4Iz1Mz5LGF8",
                   footnotes=[Footnote.CIVIC_DIESEL], min_steer_speed=2. * CV.MPH_TO_MS),
      HondaCarInfo("Honda Civic Hatchback 2017-21", min_steer_speed=12. * CV.MPH_TO_MS),
    ],
    dbc_dict('honda_civic_hatchback_ex_2017_can_generated', None),
    specs=CarSpecs(mass=1326, wheelbase=2.7, steerRatio=15.38, centerToFrontRatio=0.4),  # steerRatio: 10.93 is end-to-end spec
    flags=HondaFlags.BOSCH
  )
  CIVIC_BOSCH_DIESEL = HondaPlatformConfig(
    "HONDA CIVIC SEDAN 1.6 DIESEL 2019",
    None, # don't show in docs
    dbc_dict('honda_accord_2018_can_generated', None),
    specs=CIVIC_BOSCH.specs,
    flags=HondaFlags.BOSCH
  )
  CIVIC_2022 = HondaPlatformConfig(
    "HONDA CIVIC 2022",
    [
      HondaCarInfo("Honda Civic 2022-23", "All", video_link="https://youtu.be/ytiOT5lcp6Q"),
      HondaCarInfo("Honda Civic Hatchback 2022-23", "All", video_link="https://youtu.be/ytiOT5lcp6Q"),
    ],
    dbc_dict('honda_civic_ex_2022_can_generated', None),
    specs=CIVIC_BOSCH.specs,
    flags=HondaFlags.BOSCH | HondaFlags.BOSCH_RADARLESS,
  )
  CRV_5G = HondaPlatformConfig(
    "HONDA CR-V 2017",
    HondaCarInfo("Honda CR-V 2017-22", min_steer_speed=12. * CV.MPH_TO_MS),
    dbc_dict('honda_crv_ex_2017_can_generated', None, body_dbc='honda_crv_ex_2017_body_generated'),
    specs=CarSpecs(mass=3410 * CV.LB_TO_KG, wheelbase=2.66, steerRatio=16.0, centerToFrontRatio=0.41),  # steerRatio: 12.3 is spec end-to-end
    flags=HondaFlags.BOSCH,
  )
  CRV_HYBRID = HondaPlatformConfig(
    "HONDA CR-V HYBRID 2019",
    HondaCarInfo("Honda CR-V Hybrid 2017-20", min_steer_speed=12. * CV.MPH_TO_MS),
    dbc_dict('honda_accord_2018_can_generated', None),
    specs=CarSpecs(mass=1667, wheelbase=2.66, steerRatio=16, centerToFrontRatio=0.41),  # mass: mean of 4 models in kg, steerRatio: 12.3 is spec end-to-end
    flags=HondaFlags.BOSCH
  )
  HRV_3G = HondaPlatformConfig(
    "HONDA HR-V 2023",
    HondaCarInfo("Honda HR-V 2023", "All"),
    dbc_dict('honda_civic_ex_2022_can_generated', None),
    specs=CarSpecs(mass=3125 * CV.LB_TO_KG, wheelbase=2.61, steerRatio=15.2, centerToFrontRatio=0.41),
    flags=HondaFlags.BOSCH | HondaFlags.BOSCH_RADARLESS
  )
  ACURA_RDX_3G = HondaPlatformConfig(
    "ACURA RDX 2020",
    HondaCarInfo("Acura RDX 2019-22", "All", min_steer_speed=3. * CV.MPH_TO_MS),
    dbc_dict('acura_rdx_2020_can_generated', None),
    specs=CarSpecs(mass=4068 * CV.LB_TO_KG, wheelbase=2.75, steerRatio=11.95, centerToFrontRatio=0.41),  # as spec
    flags=HondaFlags.BOSCH
  )
  INSIGHT = HondaPlatformConfig(
    "HONDA INSIGHT 2019",
    HondaCarInfo("Honda Insight 2019-22", "All", min_steer_speed=3. * CV.MPH_TO_MS),
    dbc_dict('honda_insight_ex_2019_can_generated', None),
    specs=CarSpecs(mass=2987 * CV.LB_TO_KG, wheelbase=2.7, steerRatio=15.0, centerToFrontRatio=0.39),  # as spec
    flags=HondaFlags.BOSCH
  )
  HONDA_E = HondaPlatformConfig(
    "HONDA E 2020",
    HondaCarInfo("Honda e 2020", "All", min_steer_speed=3. * CV.MPH_TO_MS),
    dbc_dict('acura_rdx_2020_can_generated', None),
    specs=CarSpecs(mass=3338.8 * CV.LB_TO_KG, wheelbase=2.5, centerToFrontRatio=0.5, steerRatio=16.71),
    flags=HondaFlags.BOSCH
  )

  # Nidec Cars
  ACURA_ILX = HondaPlatformConfig(
    "ACURA ILX 2016",
    HondaCarInfo("Acura ILX 2016-19", "AcuraWatch Plus", min_steer_speed=25. * CV.MPH_TO_MS),
    dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=3095 * CV.LB_TO_KG, wheelbase=2.67, steerRatio=18.61, centerToFrontRatio=0.37),  # 15.3 is spec end-to-end
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  CRV = HondaPlatformConfig(
    "HONDA CR-V 2016",
    HondaCarInfo("Honda CR-V 2015-16", "Touring Trim", min_steer_speed=12. * CV.MPH_TO_MS),
    dbc_dict('honda_crv_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=3572 * CV.LB_TO_KG, wheelbase=2.62, steerRatio=16.89, centerToFrontRatio=0.41),  # as spec
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  CRV_EU = HondaPlatformConfig(
    "HONDA CR-V EU 2016",
    None, # Euro version of CRV Touring, don't show in docs
    dbc_dict('honda_crv_executive_2016_can_generated', 'acura_ilx_2016_nidec'),
    specs=CRV.specs,
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  FIT = HondaPlatformConfig(
    "HONDA FIT 2018",
    HondaCarInfo("Honda Fit 2018-20", min_steer_speed=12. * CV.MPH_TO_MS),
    dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=2644 * CV.LB_TO_KG, wheelbase=2.53, steerRatio=13.06, centerToFrontRatio=0.39),
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  FREED = HondaPlatformConfig(
    "HONDA FREED 2020",
    HondaCarInfo("Honda Freed 2020", min_steer_speed=12. * CV.MPH_TO_MS),
    dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=3086. * CV.LB_TO_KG, wheelbase=2.74, steerRatio=13.06, centerToFrontRatio=0.39),  # mostly copied from FIT
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HRV = HondaPlatformConfig(
    "HONDA HRV 2019",
    HondaCarInfo("Honda HR-V 2019-22", min_steer_speed=12. * CV.MPH_TO_MS),
    dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
    specs=HRV_3G.specs,
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  ODYSSEY = HondaPlatformConfig(
    "HONDA ODYSSEY 2018",
    HondaCarInfo("Honda Odyssey 2018-20"),
    dbc_dict('honda_odyssey_exl_2018_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=1900, wheelbase=3.0, steerRatio=14.35, centerToFrontRatio=0.41),
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_PCM_ACCEL
  )
  ODYSSEY_CHN = HondaPlatformConfig(
    "HONDA ODYSSEY CHN 2019",
    None, # Chinese version of Odyssey, don't show in docs
    dbc_dict('honda_odyssey_extreme_edition_2018_china_can_generated', 'acura_ilx_2016_nidec'),
    specs=ODYSSEY.specs,
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES
  )
  ACURA_RDX = HondaPlatformConfig(
    "ACURA RDX 2018",
    HondaCarInfo("Acura RDX 2016-18", "AcuraWatch Plus", min_steer_speed=12. * CV.MPH_TO_MS),
    dbc_dict('acura_rdx_2018_can_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=3925 * CV.LB_TO_KG, wheelbase=2.68, steerRatio=15.0, centerToFrontRatio=0.38),  # as spec
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  PILOT = HondaPlatformConfig(
    "HONDA PILOT 2017",
    [
      HondaCarInfo("Honda Pilot 2016-22", min_steer_speed=12. * CV.MPH_TO_MS),
      HondaCarInfo("Honda Passport 2019-23", "All", min_steer_speed=12. * CV.MPH_TO_MS),
    ],
    dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=4278 * CV.LB_TO_KG, wheelbase=2.86, centerToFrontRatio=0.428, steerRatio=16.0),  # as spec
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  RIDGELINE = HondaPlatformConfig(
    "HONDA RIDGELINE 2017",
    HondaCarInfo("Honda Ridgeline 2017-24", min_steer_speed=12. * CV.MPH_TO_MS),
    dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=4515 * CV.LB_TO_KG, wheelbase=3.18, centerToFrontRatio=0.41, steerRatio=15.59),  # as spec
    flags=HondaFlags.NIDEC | HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  CIVIC = HondaPlatformConfig(
    "HONDA CIVIC 2016",
    HondaCarInfo("Honda Civic 2016-18", min_steer_speed=12. * CV.MPH_TO_MS, video_link="https://youtu.be/-IkImTe1NYE"),
    dbc_dict('honda_civic_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
    specs=CarSpecs(mass=1326, wheelbase=2.70, centerToFrontRatio=0.4, steerRatio=15.38),  # 10.93 is end-to-end spec
    flags=HondaFlags.NIDEC | HondaFlags.AUTORESUME_SNG | HondaFlags.ELECTRIC_PARKING_BRAKE,
  )


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
    ),
    # Bosch PT bus
    Request(
      [StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.UDS_VERSION_RESPONSE],
      bus=1,
      obd_multiplexing=False,
    ),
  ],
  # We lose these ECUs without the comma power on these cars.
  # Note that we still attempt to match with them when they are present
  non_essential_ecus={
    Ecu.programmedFuelInjection: [CAR.ACCORD, CAR.CIVIC, CAR.CIVIC_BOSCH, CAR.CRV_5G],
    Ecu.transmission: [CAR.ACCORD, CAR.CIVIC, CAR.CIVIC_BOSCH, CAR.CRV_5G],
    Ecu.srs: [CAR.ACCORD],
    Ecu.eps: [CAR.ACCORD],
    Ecu.vsa: [CAR.ACCORD, CAR.CIVIC, CAR.CIVIC_BOSCH, CAR.CRV_5G],
    Ecu.combinationMeter: [CAR.ACCORD, CAR.CIVIC, CAR.CIVIC_BOSCH, CAR.CRV_5G],
    Ecu.gateway: [CAR.ACCORD, CAR.CIVIC, CAR.CIVIC_BOSCH, CAR.CRV_5G],
    Ecu.electricBrakeBooster: [CAR.ACCORD, CAR.CIVIC_BOSCH, CAR.CRV_5G],
    Ecu.shiftByWire: [CAR.ACCORD],  # existence correlates with transmission type for ICE
    Ecu.hud: [CAR.ACCORD],  # existence correlates with trim level
  },
  extra_ecus=[
    # The only other ECU on PT bus accessible by camera on radarless Civic
    (Ecu.unknown, 0x18DAB3F1, None),
  ],
)


STEER_THRESHOLD = {
  # default is 1200, overrides go here
  CAR.ACURA_RDX: 400,
  CAR.CRV_EU: 400,
}

HONDA_NIDEC_ALT_PCM_ACCEL = CAR.with_flags(HondaFlags.NIDEC_ALT_PCM_ACCEL)
HONDA_NIDEC_ALT_SCM_MESSAGES = CAR.with_flags(HondaFlags.NIDEC_ALT_SCM_MESSAGES)
HONDA_BOSCH = CAR.with_flags(HondaFlags.BOSCH)
HONDA_BOSCH_RADARLESS = CAR.with_flags(HondaFlags.BOSCH_RADARLESS)

CAR_INFO = CAR.create_carinfo_map()
DBC = CAR.create_dbc_map()
