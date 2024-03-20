from dataclasses import dataclass
from enum import Enum, IntFlag

from cereal import car
from openpilot.common.conversions import Conversions as CV
from panda.python import uds
from openpilot.selfdrive.car import CarSpecs, PlatformConfig, Platforms, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Column
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


# Car button codes
class CruiseButtons:
  RES_ACCEL = 4
  DECEL_SET = 3
  CANCEL = 2
  MAIN = 1


class CruiseSettings:
  DISTANCE = 3
  LKAS = 1


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
class HondaCarDocs(CarDocs):
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


class HondaBoschPlatformConfig(PlatformConfig):
  def init(self):
    self.flags |= HondaFlags.BOSCH


class HondaNidecPlatformConfig(PlatformConfig):
  def init(self):
    self.flags |= HondaFlags.NIDEC


class CAR(Platforms):
  # Bosch Cars
  HONDA_ACCORD = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Honda Accord 2018-22", "All", video_link="https://www.youtube.com/watch?v=mrUwlj3Mi58", min_steer_speed=3. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Inspire 2018", "All", min_steer_speed=3. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Accord Hybrid 2018-22", "All", min_steer_speed=3. * CV.MPH_TO_MS),
    ],
    # steerRatio: 11.82 is spec end-to-end
    CarSpecs(mass=3279 * CV.LB_TO_KG, wheelbase=2.83, steerRatio=16.33, centerToFrontRatio=0.39, tireStiffnessFactor=0.8467),
    dbc_dict('honda_accord_2018_can_generated', None),
  )
  HONDA_CIVIC_BOSCH = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Honda Civic 2019-21", "All", video_link="https://www.youtube.com/watch?v=4Iz1Mz5LGF8",
                   footnotes=[Footnote.CIVIC_DIESEL], min_steer_speed=2. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Civic Hatchback 2017-21", min_steer_speed=12. * CV.MPH_TO_MS),
    ],
    CarSpecs(mass=1326, wheelbase=2.7, steerRatio=15.38, centerToFrontRatio=0.4),  # steerRatio: 10.93 is end-to-end spec
    dbc_dict('honda_civic_hatchback_ex_2017_can_generated', None),
  )
  HONDA_CIVIC_BOSCH_DIESEL = HondaBoschPlatformConfig(
    [],  # don't show in docs
    HONDA_CIVIC_BOSCH.specs,
    dbc_dict('honda_accord_2018_can_generated', None),
  )
  HONDA_CIVIC_2022 = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Honda Civic 2022-23", "All", video_link="https://youtu.be/ytiOT5lcp6Q"),
      HondaCarDocs("Honda Civic Hatchback 2022-23", "All", video_link="https://youtu.be/ytiOT5lcp6Q"),
    ],
    HONDA_CIVIC_BOSCH.specs,
    dbc_dict('honda_civic_ex_2022_can_generated', None),
    flags=HondaFlags.BOSCH_RADARLESS,
  )
  HONDA_CRV_5G = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda CR-V 2017-22", min_steer_speed=12. * CV.MPH_TO_MS)],
    # steerRatio: 12.3 is spec end-to-end
    CarSpecs(mass=3410 * CV.LB_TO_KG, wheelbase=2.66, steerRatio=16.0, centerToFrontRatio=0.41, tireStiffnessFactor=0.677),
    dbc_dict('honda_crv_ex_2017_can_generated', None, body_dbc='honda_crv_ex_2017_body_generated'),
    flags=HondaFlags.BOSCH_ALT_BRAKE,
  )
  HONDA_CRV_HYBRID = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda CR-V Hybrid 2017-20", min_steer_speed=12. * CV.MPH_TO_MS)],
    # mass: mean of 4 models in kg, steerRatio: 12.3 is spec end-to-end
    CarSpecs(mass=1667, wheelbase=2.66, steerRatio=16, centerToFrontRatio=0.41, tireStiffnessFactor=0.677),
    dbc_dict('honda_accord_2018_can_generated', None),
  )
  HONDA_HRV_3G = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda HR-V 2023", "All")],
    CarSpecs(mass=3125 * CV.LB_TO_KG, wheelbase=2.61, steerRatio=15.2, centerToFrontRatio=0.41, tireStiffnessFactor=0.5),
    dbc_dict('honda_civic_ex_2022_can_generated', None),
    flags=HondaFlags.BOSCH_RADARLESS | HondaFlags.BOSCH_ALT_BRAKE,
  )
  ACURA_RDX_3G = HondaBoschPlatformConfig(
    [HondaCarDocs("Acura RDX 2019-22", "All", min_steer_speed=3. * CV.MPH_TO_MS)],
    CarSpecs(mass=4068 * CV.LB_TO_KG, wheelbase=2.75, steerRatio=11.95, centerToFrontRatio=0.41, tireStiffnessFactor=0.677),  # as spec
    dbc_dict('acura_rdx_2020_can_generated', None),
    flags=HondaFlags.BOSCH_ALT_BRAKE,
  )
  HONDA_INSIGHT = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda Insight 2019-22", "All", min_steer_speed=3. * CV.MPH_TO_MS)],
    CarSpecs(mass=2987 * CV.LB_TO_KG, wheelbase=2.7, steerRatio=15.0, centerToFrontRatio=0.39, tireStiffnessFactor=0.82),  # as spec
    dbc_dict('honda_insight_ex_2019_can_generated', None),
  )
  HONDA_E = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda e 2020", "All", min_steer_speed=3. * CV.MPH_TO_MS)],
    CarSpecs(mass=3338.8 * CV.LB_TO_KG, wheelbase=2.5, centerToFrontRatio=0.5, steerRatio=16.71, tireStiffnessFactor=0.82),
    dbc_dict('acura_rdx_2020_can_generated', None),
  )

  # Nidec Cars
  ACURA_ILX = HondaNidecPlatformConfig(
    [HondaCarDocs("Acura ILX 2016-19", "AcuraWatch Plus", min_steer_speed=25. * CV.MPH_TO_MS)],
    CarSpecs(mass=3095 * CV.LB_TO_KG, wheelbase=2.67, steerRatio=18.61, centerToFrontRatio=0.37, tireStiffnessFactor=0.72),  # 15.3 is spec end-to-end
    dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_CRV = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda CR-V 2015-16", "Touring Trim", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=3572 * CV.LB_TO_KG, wheelbase=2.62, steerRatio=16.89, centerToFrontRatio=0.41, tireStiffnessFactor=0.444),  # as spec
    dbc_dict('honda_crv_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_CRV_EU = HondaNidecPlatformConfig(
    [],  # Euro version of CRV Touring, don't show in docs
    HONDA_CRV.specs,
    dbc_dict('honda_crv_executive_2016_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_FIT = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Fit 2018-20", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=2644 * CV.LB_TO_KG, wheelbase=2.53, steerRatio=13.06, centerToFrontRatio=0.39, tireStiffnessFactor=0.75),
    dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_FREED = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Freed 2020", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=3086. * CV.LB_TO_KG, wheelbase=2.74, steerRatio=13.06, centerToFrontRatio=0.39, tireStiffnessFactor=0.75),  # mostly copied from FIT
    dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_HRV = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda HR-V 2019-22", min_steer_speed=12. * CV.MPH_TO_MS)],
    HONDA_HRV_3G.specs,
    dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_ODYSSEY = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Odyssey 2018-20")],
    CarSpecs(mass=1900, wheelbase=3.0, steerRatio=14.35, centerToFrontRatio=0.41, tireStiffnessFactor=0.82),
    dbc_dict('honda_odyssey_exl_2018_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_PCM_ACCEL,
  )
  HONDA_ODYSSEY_CHN = HondaNidecPlatformConfig(
    [],  # Chinese version of Odyssey, don't show in docs
    HONDA_ODYSSEY.specs,
    dbc_dict('honda_odyssey_extreme_edition_2018_china_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  ACURA_RDX = HondaNidecPlatformConfig(
    [HondaCarDocs("Acura RDX 2016-18", "AcuraWatch Plus", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=3925 * CV.LB_TO_KG, wheelbase=2.68, steerRatio=15.0, centerToFrontRatio=0.38, tireStiffnessFactor=0.444),  # as spec
    dbc_dict('acura_rdx_2018_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_PILOT = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Honda Pilot 2016-22", min_steer_speed=12. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Passport 2019-23", "All", min_steer_speed=12. * CV.MPH_TO_MS),
    ],
    CarSpecs(mass=4278 * CV.LB_TO_KG, wheelbase=2.86, centerToFrontRatio=0.428, steerRatio=16.0, tireStiffnessFactor=0.444),  # as spec
    dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_RIDGELINE = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Ridgeline 2017-24", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=4515 * CV.LB_TO_KG, wheelbase=3.18, centerToFrontRatio=0.41, steerRatio=15.59, tireStiffnessFactor=0.444),  # as spec
    dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_CIVIC = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Civic 2016-18", min_steer_speed=12. * CV.MPH_TO_MS, video_link="https://youtu.be/-IkImTe1NYE")],
    CarSpecs(mass=1326, wheelbase=2.70, centerToFrontRatio=0.4, steerRatio=15.38),  # 10.93 is end-to-end spec
    dbc_dict('honda_civic_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
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

  # RIDGELINE
  # ready for PR!
  # ('combinationMeter', 'eps', 'fwdRadar', 'srs', 'vsa'): routes: 17109, dongles: {'c5e0d17adaaefd82', '1b45e552040c796b', 'bad4ac6254c650b7', '7f34938a5e86a378', 'f53658bd0c1ccf07', 'ac97f88de8b23b6b', '35bc9ea7179f80dd', 'b3d7eee4b125d7b6', 'aaa8dcc35bcea08f', 'a050c5221639b008', '4d09b970840d3f08', 'd084ea2309575a71', 'ea5b4d9ab5ecd245', '24187316e568f2ec', '701b7d276e45a7bf', 'f55d4cc06d920e27', 'bf0fac55013ab937', '565d371991091752', '3aa992e52b91bf8f', 'b3c88ee6c378a66e', '732c29f8108b54a9', '441dd16054acf306', '7f498d0d7b378766', '9bdc08e34137d504', '93c3794271c0a6c1', 'adbbce5339634eae', '715416e4439e555e', '1551280de648df83', 'f3f2fc37c69f2311', '600ed71b1d966e6f', 'b6d146e61d3a64b2', '9b29873dd03e9357', 'a203050ebd930a6f', 'd5dd26679a932af9', 'da3ab625d901eb2b', '1a074a6025c49094', 'd319fb90bf4c979d', '27ffb245d4fc2f39', '123bb10637490c9b', 'a30ea383d0e05350', '3cc94a71a1c61666', 'a8a2f1ff07c9ab11', '6e2f19b5263c9d65', 'e51512df42d845d0', '49eb8c6a69caa049', 'e52c055a77cac84c', '19fe0c3f46444eeb', 'bbd41ebd2179ae49', '1ce6e69a54819ac9', 'e36f7195f1bb11c7', '1697535ec227491d', '97bbe58ea225ad1d', '1ce42f7a69e4b11c', '7952acf99389e037', 'a9648fa1431cf7a8', '0ddc5f974851a022', '07f62799f4e6944b', '6b8635e52c3e4e73', '4a080fe908e25eb2', 'ff6a03d8e5bab6bc', 'aaa226a738d37659', '263d65b0fd18f5c4', '69410e74ad24dda3', '0231272ec6feef74', 'e4bcdeea7c6f1fd3', '5d9a427622c55b6c', '1eb0e8547f8ff382', '6acf79c245a32eb3', '805f226ab2a8b1d7', '93f7c48437b4cc0f', '758cfd09da38d125', 'bf72e541df745f48', '9e334ca61b5089ee', 'be8ff657d8e5179f', '3c1a2c5bfa0fc55d', '9824e24dd1574abc', 'cbcc412104157ab0', 'd169250d735e1346', '306b1963ae64982a', 'a7252f38a01fc213'}
  # (): routes: 21, dongles: {'1eb0e8547f8ff382', '1a074a6025c49094'}
  # ('combinationMeter', 'eps', 'fwdRadar', 'srs'): routes: 1, dongles: {'b3d7eee4b125d7b6'}

  # INSIGHT
  # ready for PR!
  # ('eps', 'fwdCamera', 'fwdRadar', 'srs'): routes: 4114, dongles: {'018c9e8786f70889', 'fdf1eae234f98180', 'c1a906e7bdbac9fe', 'a411418e62a4d6f7', '21726e67ee95b707', '6ca0bbb0b026b5a5', 'baeb84fd6ad76707', '9dae484044b3cfe8', 'df113ca24ae4dbb5', '76abc19439073a17', '8958942b212e8cd9', '64d21f4115f0e66a', 'bfde60a60008c36d', '907c91003e36d85d', 'f28026c92de2742e', 'b93205748d65caff', '29c9d54d036ce5ac', 'b41ccf51a762d116', 'bad28a64a886f4c5', 'acdb227003c7aee1', 'ba9a88a070809873', '61ced3dc823c6830', '63a8c388394d258f', '9c63237c6c04ec9a', 'f7d92a003486dc37', 'b77d071405389d2a', '13c9a6610b7d0c4f', 'eb5e7e06aa56a47f', 'f9b412667f9b0669', '663bc1c495789296', '46ed1f05072e029a'}
  # (): routes: 146, dongles: {'29c9d54d036ce5ac'}
  # ('fwdRadar',): routes: 5, dongles: {'eb5e7e06aa56a47f'}
  # ('combinationMeter', 'eps', 'fwdCamera', 'fwdRadar', 'gateway', 'shiftByWire', 'srs', 'vsa'): routes: 1, dongles: {'baeb84fd6ad76707'}

  non_essential_ecus={
    Ecu.programmedFuelInjection: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G],
    Ecu.transmission: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G],
    Ecu.srs: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_E],
    Ecu.eps: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_E],
    Ecu.vsa: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E],
    Ecu.combinationMeter: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_FIT,
                           CAR.HONDA_HRV, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E, CAR.HONDA_ODYSSEY_CHN],
    Ecu.gateway: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_FIT, CAR.HONDA_FREED,
                  CAR.HONDA_HRV, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E, CAR.HONDA_ODYSSEY_CHN],
    Ecu.electricBrakeBooster: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G],
    # existence correlates with transmission type for Accord ICE
    Ecu.shiftByWire: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E],
    # existence correlates with trim level
    Ecu.hud: [CAR.HONDA_ACCORD],
  },
  extra_ecus=[
    # The only other ECU on PT bus accessible by camera on radarless Civic
    (Ecu.unknown, 0x18DAB3F1, None),
  ],
)

STEER_THRESHOLD = {
  # default is 1200, overrides go here
  CAR.ACURA_RDX: 400,
  CAR.HONDA_CRV_EU: 400,
}

HONDA_NIDEC_ALT_PCM_ACCEL = CAR.with_flags(HondaFlags.NIDEC_ALT_PCM_ACCEL)
HONDA_NIDEC_ALT_SCM_MESSAGES = CAR.with_flags(HondaFlags.NIDEC_ALT_SCM_MESSAGES)
HONDA_BOSCH = CAR.with_flags(HondaFlags.BOSCH)
HONDA_BOSCH_RADARLESS = CAR.with_flags(HondaFlags.BOSCH_RADARLESS)


DBC = CAR.create_dbc_map()
