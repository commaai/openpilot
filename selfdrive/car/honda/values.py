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

  # ('combinationMeter', 'eps', 'fwdRadar', 'programmedFuelInjection', 'shiftByWire', 'srs', 'transmission', 'vsa'): routes: 17745, dongles: {'1a5d045d2c531a6d', '22dbdf2ca2bce921', '2f46ad8f2394b80f', '812b2084c6bdfe4c', '254980a9b782d242', 'a7f52f253cc2f654', '5e7110874b7927ba', '214a25adc54636fc', '177b1675a5e94d48', 'cd4f95472b0c45e1', 'd53a9253d19264e5', '679c6710f7fd12e3', '0ece3917a1e2f698', 'da7d4c7c00a25e08', '55f24af017e5fea9', '382eda70e7ac1346', '3848aeae9b19e2c9', '3ac08dbb3c723ffb', 'bd80a1c23664b3b7', 'fd927469d32709fa', '4901b57aa4393ee1', '8046d2596a5ecb1f', '087f2f3dd24b9fb9', 'aadbd1a661fb4da3', '9dae484044b3cfe8', 'cee1f89fd5ca2190', '6868a58803419504', '30bdd852020fbd54', 'f4e5206d7968b785', '18bbbb7606cddb6a', '429b106d0b800755', '37af41cdc47b6c52', '10b2687dae6797a8', 'af1fff76900d65ae', '388d69b9e05375c0', '9632eaf6d58168db', 'f5d77d171afb18c1', 'f7494f2397b654f8', '8466d3d83e91a51c', '8a1e4b961c93de0f', '33414660dc3cf77c', 'a52ea98510c41307', 'fd1b73324838a949', '2c2136b519bd120f', '06bfc881f92877b1', '21e8d07ec53def2a', '859cfb7352fa2786', 'a694c4195025549f', '2205ba1364b32b3e', 'a9fa4a5b741f256c', '9bac7575fac62395', 'b168dde5f920d11a', '059c8352ab54bf9b', 'fff9af2c280cebe7', '4ff67da1be2f4f43', 'df999d922fb413af', '5dccf40c6d8d548e', 'f1da8a586607e54b', '74f888a4d6b5e618', 'b7a6058b2d90e035', 'd4908ed13766afa0', 'f4b656c4aaf63360', 'e46188fc5225777f'}
  # ('combinationMeter', 'eps', 'fwdRadar', 'srs', 'vsa'): routes: 3756, dongles: {'13449b2f2ad4f094', '9cb432f9376cfa89', '6a50c87482bea213', 'e07e1900bfdf7bde', '83ccc4c48741b67a', 'd9c43a383a53494b', '7e6a6ab0bf5d8832', '48d49ee5acf4d6df', 'ad1b9150af890d9e', '59f712e1791a7331', '688d5abe0b0ec76c', '31e47fc2fc02c165', '7dea1868be52b68f', '35b4e5307cce2504', '796b61c8ede8bc4e', '695826b02fd8fe16', '8a9bfee29c5a277a', '1c8910b52e443f18', '13c9a6610b7d0c4f', '6795ded7f9db1720', '11bfdda7cffeb882'}

  non_essential_ecus={
    Ecu.programmedFuelInjection: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G],
    Ecu.transmission: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G],
    Ecu.srs: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_E],
    Ecu.eps: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_E],
    Ecu.vsa: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID,
              CAR.HONDA_E, CAR.HONDA_INSIGHT],
    Ecu.combinationMeter: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_FIT,
                           CAR.HONDA_HRV, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E, CAR.HONDA_INSIGHT, CAR.HONDA_ODYSSEY_CHN],
    Ecu.gateway: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_FIT, CAR.HONDA_FREED,
                  CAR.HONDA_HRV, CAR.HONDA_RIDGELINE, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E, CAR.HONDA_INSIGHT, CAR.HONDA_ODYSSEY_CHN],
    Ecu.electricBrakeBooster: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G],
    # existence correlates with transmission type for Accord ICE
    Ecu.shiftByWire: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E, CAR.HONDA_INSIGHT],
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
