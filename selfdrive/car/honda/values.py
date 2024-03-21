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

  # ACURA_ILX
  # ('fwdRadar', 'srs'): routes: 146, dongles: {'23893c6e32ab0d30', '2c5ec6ab9c348a77'}
  # ---
  # ('combinationMeter', 'fwdRadar', 'srs'): routes: 36, dongles: {'692d65abf7666482', '9d091e299fc54701'}
  # ---
  # ('combinationMeter', 'srs'): routes: 1, dongles: {'9d091e299fc54701'}
  # ---

  # ODYSSEY
  # ('combinationMeter', 'eps', 'fwdRadar', 'programmedFuelInjection', 'shiftByWire', 'srs', 'transmission', 'vsa'): routes: 22018, dongles: {'dd8e9caf07f5b828', '6da2ceb7bf6a818c', '2e90c996d0ba97a2', '349a10da75da84b3', '5a28fb7518234651', '20994bc5331f98c0', 'c28699378e81deb4', 'd1a0dab0337a49bb', '772b833c28f493b5', '83670f5527f21a71', '2c81142b45056e26', '0c23034a3a1e7bf4', '49a8f288f2c70bd9', '17b629a57688957e', '105a691de81f5d0a', '0ffff9faf2699ebc', '180ead4c4c126f9d', 'b14eb1e87e5c7b04', '1b83b62ce1e15374', '23d8d880f7936549', '4b830dcb5a64ac09', 'a3636d74d9353700', '564e1c4005971631', 'e1e997c8166094ce', '44af831cccaf24f6', '9b7e78f7511c0d95', '0e4bac1d711fd551', '2069b4092db64099', '7074dfeeffbb7f3b', 'f70097378779394a', '6993681710231e24', '550b1bafb9c53324', 'e462034ac9e0c71d', '89ae461c07c586a3', '59ca2e25c9f03aae', '3eb25233bf8777b2', '20999527e9966d57', '499ab6011a6f0115', '568f8a1ce998fe8c', '0494ab1688baf0d1', 'c7fa6fb6f4d42407', 'b700d19967c0136b', 'b95acf45b72e104d', 'c549a7d0ddceb251', '7d1b9fb1892e3415', 'ff001cab8ff97dd1', '75d1c547e2d4725b', '4480efadd4e5e9b0', '67cc68f0290328c2', 'f30eee173ffa893d', '761e1503448b32d5', '8eb3d1f10274ac44', 'a064118e09822a5e', '967aaf3f479fd2fb', '633911d60cf45223', '1af3445804c80894', '5de06f504727ef4e', '0c21c391abd382af', 'b288b21f47092c68', '394f8443c72ba946', '10702b68bb2db4f7', '3a02fd61e8d654b9', '7e870a7ec3860511', '326e5318a0ce5357', '131c0a3222186596', 'fa0bfc645891d426', '10d6a69109431519', '9954025f3a2aec2d', '24f032bbbc29f607', 'b83fc39362bdde40', '20608a5877a0b57d', '8906faebfc695953', '7a7eba6f9ce36254', '2e531d8181892032', '76b6bcf9925e9293', 'cf748e7031d411c8', 'e67fa35fcc6db9d5', 'b15734b667e40249', '94ad75c81a0d530d', '6d3b5e3ce824a6be', 'c57d8a1adaecd30e', 'e1aa4eae17f7b626', '98e0ab15472d3642', 'aa7176d108cb2f97', '4b5b4f473fae352b'}
  # ---
  # (): routes: 6, dongles: {'0ffff9faf2699ebc', '20608a5877a0b57d'}
  # ---
  # ('programmedFuelInjection',): routes: 2, dongles: {'75d1c547e2d4725b'}
  # ---
  # ('combinationMeter', 'eps', 'programmedFuelInjection', 'shiftByWire', 'srs', 'transmission', 'vsa'): routes: 2, dongles: {'20999527e9966d57'}
  # ---
  # ('programmedFuelInjection', 'srs'): routes: 1, dongles: {'633911d60cf45223'}
  # ---
  # ('programmedFuelInjection', 'srs', 'vsa'): routes: 1, dongles: {'633911d60cf45223'}
  # ---

  # ODYSSEY (filtered with a minimum of 5 routes per ECU group)
  # ('combinationMeter', 'eps', 'fwdRadar', 'programmedFuelInjection', 'shiftByWire', 'srs', 'transmission', 'vsa'): routes: 21613, dongles: {'dd8e9caf07f5b828', '6da2ceb7bf6a818c', '2e90c996d0ba97a2', '349a10da75da84b3', '5a28fb7518234651', '20994bc5331f98c0', 'c28699378e81deb4', 'd1a0dab0337a49bb', '772b833c28f493b5', '2c81142b45056e26', '0c23034a3a1e7bf4', '49a8f288f2c70bd9', '17b629a57688957e', '105a691de81f5d0a', '0ffff9faf2699ebc', '180ead4c4c126f9d', 'b14eb1e87e5c7b04', '1b83b62ce1e15374', '23d8d880f7936549', '4b830dcb5a64ac09', 'a3636d74d9353700', '564e1c4005971631', '44af831cccaf24f6', '9b7e78f7511c0d95', '0e4bac1d711fd551', '2069b4092db64099', '7074dfeeffbb7f3b', 'f70097378779394a', '6993681710231e24', '550b1bafb9c53324', 'e462034ac9e0c71d', '89ae461c07c586a3', '59ca2e25c9f03aae', '3eb25233bf8777b2', '20999527e9966d57', '499ab6011a6f0115', '0494ab1688baf0d1', 'c7fa6fb6f4d42407', 'b700d19967c0136b', 'b95acf45b72e104d', 'c549a7d0ddceb251', '7d1b9fb1892e3415', 'ff001cab8ff97dd1', '75d1c547e2d4725b', '4480efadd4e5e9b0', '67cc68f0290328c2', 'f30eee173ffa893d', '761e1503448b32d5', '8eb3d1f10274ac44', 'a064118e09822a5e', '967aaf3f479fd2fb', '633911d60cf45223', '1af3445804c80894', '5de06f504727ef4e', '0c21c391abd382af', 'b288b21f47092c68', '394f8443c72ba946', '3a02fd61e8d654b9', '7e870a7ec3860511', '131c0a3222186596', 'fa0bfc645891d426', '10d6a69109431519', '9954025f3a2aec2d', '24f032bbbc29f607', 'b83fc39362bdde40', '20608a5877a0b57d', '7a7eba6f9ce36254', '2e531d8181892032', '76b6bcf9925e9293', 'cf748e7031d411c8', 'e67fa35fcc6db9d5', 'b15734b667e40249', '94ad75c81a0d530d', '6d3b5e3ce824a6be', 'c57d8a1adaecd30e', 'e1aa4eae17f7b626', '98e0ab15472d3642', 'aa7176d108cb2f97', '4b5b4f473fae352b'}


  non_essential_ecus={
    Ecu.programmedFuelInjection: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G, CAR.HONDA_PILOT],
    Ecu.transmission: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G, CAR.HONDA_PILOT],
    Ecu.srs: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_E],
    Ecu.eps: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_E],
    Ecu.vsa: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID,
              CAR.HONDA_E, CAR.HONDA_INSIGHT],
    Ecu.combinationMeter: [CAR.ACURA_ILX, CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_FIT,
                           CAR.HONDA_HRV, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E, CAR.HONDA_INSIGHT, CAR.HONDA_ODYSSEY_CHN],
    Ecu.gateway: [CAR.ACURA_ILX, CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_FIT, CAR.HONDA_FREED,
                  CAR.HONDA_HRV, CAR.HONDA_RIDGELINE, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E, CAR.HONDA_INSIGHT, CAR.HONDA_ODYSSEY,
                  CAR.HONDA_ODYSSEY_CHN, CAR.HONDA_PILOT],
    Ecu.electricBrakeBooster: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G],
    # existence correlates with transmission type for Accord ICE
    Ecu.shiftByWire: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E, CAR.HONDA_INSIGHT, CAR.HONDA_PILOT],
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
