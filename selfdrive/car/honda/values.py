from cereal import car
from selfdrive.car import dbc_dict

Ecu = car.CarParams.Ecu
VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarControllerParams():
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

class CAR:
  ACCORD = "HONDA ACCORD 2018"
  ACCORDH = "HONDA ACCORD HYBRID 2018"
  CIVIC = "HONDA CIVIC 2016"
  CIVIC_BOSCH = "HONDA CIVIC (BOSCH) 2019"
  CIVIC_BOSCH_DIESEL = "HONDA CIVIC SEDAN 1.6 DIESEL 2019"
  ACURA_ILX = "ACURA ILX 2016"
  CRV = "HONDA CR-V 2016"
  CRV_5G = "HONDA CR-V 2017"
  CRV_EU = "HONDA CR-V EU 2016"
  CRV_HYBRID = "HONDA CR-V HYBRID 2019"
  FIT = "HONDA FIT 2018"
  FREED = "HONDA FREED 2020"
  HRV = "HONDA HRV 2019"
  ODYSSEY = "HONDA ODYSSEY 2018"
  ODYSSEY_CHN = "HONDA ODYSSEY CHN 2019"
  ACURA_RDX = "ACURA RDX 2018"
  ACURA_RDX_3G = "ACURA RDX 2020"
  PILOT = "HONDA PILOT 2017"
  PILOT_2019 = "HONDA PILOT 2019"
  PASSPORT = "HONDA PASSPORT 2021"
  RIDGELINE = "HONDA RIDGELINE 2017"
  INSIGHT = "HONDA INSIGHT 2019"
  HONDA_E = "HONDA E 2020"

FW_VERSIONS = {
  CAR.ACCORD: {
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-6A0-9520\x00\x00',
      b'37805-6A0-9620\x00\x00',
      b'37805-6A0-9720\x00\x00',
      b'37805-6A0-A540\x00\x00',
      b'37805-6A0-A550\x00\x00',
      b'37805-6A0-A640\x00\x00',
      b'37805-6A0-A650\x00\x00',
      b'37805-6A0-A740\x00\x00',
      b'37805-6A0-A750\x00\x00',
      b'37805-6A0-A840\x00\x00',
      b'37805-6A0-A850\x00\x00',
      b'37805-6A0-AG30\x00\x00',
      b'37805-6A0-C540\x00\x00',
      b'37805-6A1-H650\x00\x00',
      b'37805-6B2-A550\x00\x00',
      b'37805-6B2-A560\x00\x00',
      b'37805-6B2-A650\x00\x00',
      b'37805-6B2-A660\x00\x00',
      b'37805-6B2-A720\x00\x00',
      b'37805-6B2-A810\x00\x00',
      b'37805-6B2-A820\x00\x00',
      b'37805-6B2-A920\x00\x00',
      b'37805-6B2-M520\x00\x00',
      b'37805-6B2-Y810\x00\x00',
      b'37805-6M4-B730\x00\x00',
    ],
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-TVC-A910\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-6A7-A220\x00\x00',
      b'28101-6A7-A230\x00\x00',
      b'28101-6A7-A320\x00\x00',
      b'28101-6A7-A330\x00\x00',
      b'28101-6A7-A410\x00\x00',
      b'28101-6A7-A510\x00\x00',
      b'28101-6A7-A610\x00\x00',
      b'28101-6A9-H140\x00\x00',
      b'28101-6A9-H420\x00\x00',
      b'28102-6B8-A560\x00\x00',
      b'28102-6B8-A570\x00\x00',
      b'28102-6B8-A700\x00\x00',
      b'28102-6B8-A800\x00\x00',
      b'28102-6B8-C560\x00\x00',
      b'28102-6B8-C570\x00\x00',
      b'28102-6B8-M520\x00\x00',
      b'28102-6B8-R700\x00\x00',
    ],
    (Ecu.electricBrakeBooster, 0x18da2bf1, None): [
      b'46114-TVA-A060\x00\x00',
      b'46114-TVA-A080\x00\x00',
      b'46114-TVA-A120\x00\x00',
      b'46114-TVA-A320\x00\x00',
      b'46114-TVA-A050\x00\x00',
      b'46114-TVE-H550\x00\x00',
      b'46114-TVE-H560\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TVA-B040\x00\x00',
      b'57114-TVA-B050\x00\x00',
      b'57114-TVA-B060\x00\x00',
      b'57114-TVA-B530\x00\x00',
      b'57114-TVA-C040\x00\x00',
      b'57114-TVA-C050\x00\x00',
      b'57114-TVA-C060\x00\x00',
      b'57114-TVA-C530\x00\x00',
      b'57114-TVE-H250\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TBX-H120\x00\x00',
      b'39990-TVA-A140\x00\x00',
      b'39990-TVA-A150\x00\x00',
      b'39990-TVA-A160\x00\x00',
      b'39990-TVA-A340\x00\x00',
      b'39990-TVA-X030\x00\x00',
      b'39990-TVA-X040\x00\x00',
      b'39990-TVA,A150\x00\x00',
      b'39990-TVE-H130\x00\x00',
    ],
    (Ecu.unknown, 0x18da3af1, None): [
      b'39390-TVA-A020\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TBX-H230\x00\x00',
      b'77959-TVA-A460\x00\x00',
      b'77959-TVA-F330\x00\x00',
      b'77959-TVA-H230\x00\x00',
      b'77959-TVA-L420\x00\x00',
      b'77959-TVA-X330\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TBX-H310\x00\x00',
      b'78109-TVA-A010\x00\x00',
      b'78109-TVA-A020\x00\x00',
      b'78109-TVA-A030\x00\x00',
      b'78109-TVA-A110\x00\x00',
      b'78109-TVA-A120\x00\x00',
      b'78109-TVA-A210\x00\x00',
      b'78109-TVA-A220\x00\x00',
      b'78109-TVA-A310\x00\x00',
      b'78109-TVA-C010\x00\x00',
      b'78109-TVA-L010\x00\x00',
      b'78109-TVA-L210\x00\x00',
      b'78109-TVC-A010\x00\x00',
      b'78109-TVC-A020\x00\x00',
      b'78109-TVC-A030\x00\x00',
      b'78109-TVC-A110\x00\x00',
      b'78109-TVC-A130\x00\x00',
      b'78109-TVC-A210\x00\x00',
      b'78109-TVC-A220\x00\x00',
      b'78109-TVC-C010\x00\x00',
      b'78109-TVC-C110\x00\x00',
      b'78109-TVC-L010\x00\x00',
      b'78109-TVC-L210\x00\x00',
      b'78109-TVC-M510\x00\x00',
      b'78109-TVC-YF10\x00\x00',
      b'78109-TVE-H610\x00\x00',
      b'78109-TWA-A210\x00\x00',
    ],
    (Ecu.hud, 0x18da61f1, None): [
      b'78209-TVA-A010\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36802-TBX-H140\x00\x00',
      b'36802-TVA-A150\x00\x00',
      b'36802-TVA-A160\x00\x00',
      b'36802-TVA-A170\x00\x00',
      b'36802-TVA-A330\x00\x00',
      b'36802-TVC-A330\x00\x00',
      b'36802-TVE-H070\x00\x00',
      b'36802-TWA-A070\x00\x00',
      b'36802-TWA-A080\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab5f1, None): [
      b'36161-TBX-H130\x00\x00',
      b'36161-TVA-A060\x00\x00',
      b'36161-TVA-A330\x00\x00',
      b'36161-TVC-A330\x00\x00',
      b'36161-TVE-H050\x00\x00',
      b'36161-TWA-A070\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TVA-A010\x00\x00',
      b'38897-TVA-A020\x00\x00',
      b'38897-TVA-A230\x00\x00',
      b'38897-TVA-A240\x00\x00',
    ],
  },
  CAR.ACCORDH: {
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TWA-A120\x00\x00',
      b'38897-TWD-J020\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TWA-A040\x00\x00',
      b'57114-TWA-A050\x00\x00',
      b'57114-TWA-A530\x00\x00',
      b'57114-TWA-B520\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TWA-A440\x00\x00',
      b'77959-TWA-L420\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TWA-A010\x00\x00',
      b'78109-TWA-A020\x00\x00',
      b'78109-TWA-A030\x00\x00',
      b'78109-TWA-A110\x00\x00',
      b'78109-TWA-A120\x00\x00',
      b'78109-TWA-A210\x00\x00',
      b'78109-TWA-A220\x00\x00',
      b'78109-TWA-A230\x00\x00',
      b'78109-TWA-L010\x00\x00',
      b'78109-TWA-L210\x00\x00',
    ],
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-TWA-A910\x00\x00',
    ],
    (Ecu.hud, 0x18da61f1, None): [
      b'78209-TVA-A010\x00\x00',
      b'78209-TVA-A110\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab5f1, None): [
      b'36161-TWA-A070\x00\x00',
      b'36161-TWA-A330\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36802-TWA-A080\x00\x00',
      b'36802-TWA-A070\x00\x00',
      b'36802-TWA-A330\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TVA-A160\x00\x00',
      b'39990-TVA-A150\x00\x00',
      b'39990-TVA-A340\x00\x00',
    ],
  },
  CAR.CIVIC: {
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-5AA-A640\x00\x00',
      b'37805-5AA-A650\x00\x00',
      b'37805-5AA-A670\x00\x00',
      b'37805-5AA-A680\x00\x00',
      b'37805-5AA-A810\x00\x00',
      b'37805-5AA-C640\x00\x00',
      b'37805-5AA-C680\x00\x00',
      b'37805-5AA-C820\x00\x00',
      b'37805-5AA-L650\x00\x00',
      b'37805-5AA-L660\x00\x00',
      b'37805-5AA-L680\x00\x00',
      b'37805-5AA-L690\x00\x00',
      b'37805-5AA-L810\000\000',
      b'37805-5AG-Q710\x00\x00',
      b'37805-5AJ-A610\x00\x00',
      b'37805-5AJ-A620\x00\x00',
      b'37805-5AJ-L610\x00\x00',
      b'37805-5BA-A310\x00\x00',
      b'37805-5BA-A510\x00\x00',
      b'37805-5BA-A740\x00\x00',
      b'37805-5BA-A760\x00\x00',
      b'37805-5BA-A930\x00\x00',
      b'37805-5BA-A960\x00\x00',
      b'37805-5BA-C860\x00\x00',
      b'37805-5BA-L410\x00\x00',
      b'37805-5BA-L760\x00\x00',
      b'37805-5BA-L930\x00\x00',
      b'37805-5BA-L940\x00\x00',
      b'37805-5BA-L960\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-5CG-A040\x00\x00',
      b'28101-5CG-A050\x00\x00',
      b'28101-5CG-A070\x00\x00',
      b'28101-5CG-A080\x00\x00',
      b'28101-5CG-A320\x00\x00',
      b'28101-5CG-A810\x00\x00',
      b'28101-5CG-A820\x00\x00',
      b'28101-5DJ-A040\x00\x00',
      b'28101-5DJ-A060\x00\x00',
      b'28101-5DJ-A510\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TBA-A540\x00\x00',
      b'57114-TBA-A550\x00\x00',
      b'57114-TBA-A560\x00\x00',
      b'57114-TBA-A570\x00\x00',
      b'57114-TEA-Q220\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TBA,A030\x00\x00', # modified firmware
      b'39990-TBA-A030\x00\x00',
      b'39990-TBG-A030\x00\x00',
      b'39990-TEA-T020\x00\x00',
      b'39990-TEG-A010\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TBA-A030\x00\x00',
      b'77959-TBA-A040\x00\x00',
      b'77959-TBG-A030\x00\x00',
      b'77959-TEA-Q820\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TBA-A510\x00\x00',
      b'78109-TBA-A520\x00\x00',
      b'78109-TBA-A530\x00\x00',
      b'78109-TBA-C520\x00\x00',
      b'78109-TBC-A310\x00\x00',
      b'78109-TBC-A320\x00\x00',
      b'78109-TBC-A510\x00\x00',
      b'78109-TBC-A520\x00\x00',
      b'78109-TBC-A530\x00\x00',
      b'78109-TBC-C510\x00\x00',
      b'78109-TBC-C520\x00\x00',
      b'78109-TBC-C530\x00\x00',
      b'78109-TBH-A510\x00\x00',
      b'78109-TBH-A530\x00\x00',
      b'78109-TED-Q510\x00\x00',
      b'78109-TEG-A310\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab0f1, None): [
      b'36161-TBA-A020\x00\x00',
      b'36161-TBA-A030\x00\x00',
      b'36161-TBA-A040\x00\x00',
      b'36161-TBC-A020\x00\x00',
      b'36161-TBC-A030\x00\x00',
      b'36161-TED-Q320\x00\x00',
      b'36161-TEG-A010\x00\x00',
      b'36161-TEG-A020\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TBA-A010\x00\x00',
      b'38897-TBA-A020\x00\x00',
    ],
  },
  CAR.CIVIC_BOSCH: {
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-5AA-A940\x00\x00',
      b'37805-5AA-A950\x00\x00',
      b'37805-5AA-L940\x00\x00',
      b'37805-5AA-L950\x00\x00',
      b'37805-5AG-Z910\x00\x00',
      b'37805-5AJ-A750\x00\x00',
      b'37805-5AJ-L750\x00\x00',
      b'37805-5AK-T530\x00\x00',
      b'37805-5AN-A750\x00\x00',
      b'37805-5AN-A830\x00\x00',
      b'37805-5AN-A840\x00\x00',
      b'37805-5AN-A930\x00\x00',
      b'37805-5AN-A940\x00\x00',
      b'37805-5AN-A950\x00\x00',
      b'37805-5AN-AG20\x00\x00',
      b'37805-5AN-AH20\x00\x00',
      b'37805-5AN-AJ30\x00\x00',
      b'37805-5AN-AK20\x00\x00',
      b'37805-5AN-AR20\x00\x00',
      b'37805-5AN-CH20\x00\x00',
      b'37805-5AN-E630\x00\x00',
      b'37805-5AN-E720\x00\x00',
      b'37805-5AN-E820\x00\x00',
      b'37805-5AN-J820\x00\x00',
      b'37805-5AN-L840\x00\x00',
      b'37805-5AN-L930\x00\x00',
      b'37805-5AN-L940\x00\x00',
      b'37805-5AN-LF20\x00\x00',
      b'37805-5AN-LH20\x00\x00',
      b'37805-5AN-LJ20\x00\x00',
      b'37805-5AN-LR20\x00\x00',
      b'37805-5AN-LS20\x00\x00',
      b'37805-5AW-G720\x00\x00',
      b'37805-5AZ-E850\x00\x00',
      b'37805-5AZ-G540\x00\x00',
      b'37805-5AZ-G740\x00\x00',
      b'37805-5AZ-G840\x00\x00',
      b'37805-5BB-A530\x00\x00',
      b'37805-5BB-A540\x00\x00',
      b'37805-5BB-A630\x00\x00',
      b'37805-5BB-A640\x00\x00',
      b'37805-5BB-C540\x00\x00',
      b'37805-5BB-C630\x00\x00',
      b'37805-5BB-C640\x00\x00',
      b'37805-5BB-L540\x00\x00',
      b'37805-5BB-L630\x00\x00',
      b'37805-5BB-L640\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-5CG-A920\x00\x00',
      b'28101-5CG-AB10\x00\x00',
      b'28101-5CG-C110\x00\x00',
      b'28101-5CG-C220\x00\x00',
      b'28101-5CG-C320\x00\x00',
      b'28101-5CG-G020\x00\x00',
      b'28101-5CG-L020\x00\x00',
      b'28101-5CK-A130\x00\x00',
      b'28101-5CK-A140\x00\x00',
      b'28101-5CK-A150\x00\x00',
      b'28101-5CK-C130\x00\x00',
      b'28101-5CK-C140\x00\x00',
      b'28101-5CK-C150\x00\x00',
      b'28101-5CK-G210\x00\x00',
      b'28101-5CK-J710\x00\x00',
      b'28101-5CK-Q610\x00\x00',
      b'28101-5DJ-A610\x00\x00',
      b'28101-5DJ-A710\x00\x00',
      b'28101-5DV-E330\x00\x00',
      b'28101-5DV-E610\x00\x00',
      b'28101-5DV-E820\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TBG-A330\x00\x00',
      b'57114-TBG-A340\x00\x00',
      b'57114-TBG-A350\x00\x00',
      b'57114-TGG-A340\x00\x00',
      b'57114-TGG-C320\x00\x00',
      b'57114-TGG-G320\x00\x00',
      b'57114-TGG-L320\x00\x00',
      b'57114-TGG-L330\x00\x00',
      b'57114-TGK-T320\x00\x00',
      b'57114-TGL-G330\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TBA-C020\x00\x00',
      b'39990-TBA-C120\x00\x00',
      b'39990-TEA-T820\x00\x00',
      b'39990-TEZ-T020\x00\x00',
      b'39990-TGG-A020\x00\x00',
      b'39990-TGG-A120\x00\x00',
      b'39990-TGG-J510\x00\x00',
      b'39990-TGL-E130\x00\x00',
      b'39990-TGN-E120\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TBA-A060\x00\x00',
      b'77959-TBG-A050\x00\x00',
      b'77959-TEA-G020\x00\x00',
      b'77959-TGG-A020\x00\x00',
      b'77959-TGG-A030\x00\x00',
      b'77959-TGG-E010\x00\x00',
      b'77959-TGG-G010\x00\x00',
      b'77959-TGG-G110\x00\x00',
      b'77959-TGG-J320\x00\x00',
      b'77959-TGG-Z820\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TBA-A110\x00\x00',
      b'78109-TBA-A910\x00\x00',
      b'78109-TBA-C340\x00\x00',
      b'78109-TBA-C910\x00\x00',
      b'78109-TBC-A740\x00\x00',
      b'78109-TBG-A110\x00\x00',
      b'78109-TEG-A720\x00\x00',
      b'78109-TFJ-G020\x00\x00',
      b'78109-TGG-9020\x00\x00',
      b'78109-TGG-A210\x00\x00',
      b'78109-TGG-A220\x00\x00',
      b'78109-TGG-A310\x00\x00',
      b'78109-TGG-A320\x00\x00',
      b'78109-TGG-A330\x00\x00',
      b'78109-TGG-A610\x00\x00',
      b'78109-TGG-A620\x00\x00',
      b'78109-TGG-A810\x00\x00',
      b'78109-TGG-A820\x00\x00',
      b'78109-TGG-C220\x00\x00',
      b'78109-TGG-E110\x00\x00',
      b'78109-TGG-G030\x00\x00',
      b'78109-TGG-G230\x00\x00',
      b'78109-TGG-G410\x00\x00',
      b'78109-TGK-Z410\x00\x00',
      b'78109-TGL-G120\x00\x00',
      b'78109-TGL-G130\x00\x00',
      b'78109-TGL-G210\x00\x00',
      b'78109-TGL-G230\x00\x00',
      b'78109-TGL-GM10\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36802-TBA-A150\x00\x00',
      b'36802-TBA-A160\x00\x00',
      b'36802-TFJ-G060\x00\x00',
      b'36802-TGG-A050\x00\x00',
      b'36802-TGG-A060\x00\x00',
      b'36802-TGG-A130\x00\x00',
      b'36802-TGG-G040\x00\x00',
      b'36802-TGG-G130\x00\x00',
      b'36802-TGK-Q120\x00\x00',
      b'36802-TGL-G040\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab5f1, None): [
      b'36161-TBA-A130\x00\x00',
      b'36161-TBA-A140\x00\x00',
      b'36161-TFJ-G070\x00\x00',
      b'36161-TGG-A060\x00\x00',
      b'36161-TGG-A080\x00\x00',
      b'36161-TGG-A120\x00\x00',
      b'36161-TGG-G050\x00\x00',
      b'36161-TGG-G130\x00\x00',
      b'36161-TGG-G140\x00\x00',
      b'36161-TGK-Q120\x00\x00',
      b'36161-TGL-G050\x00\x00',
      b'36161-TGL-G070\x00\x00',
      b'36161-TGG-G070\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TBA-A110\x00\x00',
      b'38897-TBA-A020\x00\x00',
    ],
    (Ecu.electricBrakeBooster, 0x18da2bf1, None): [
      b'39494-TGL-G030\x00\x00',
    ],
  },
  CAR.CIVIC_BOSCH_DIESEL: {
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-59N-G630\x00\x00',
      b'37805-59N-G830\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-59Y-G220\x00\x00',
      b'28101-59Y-G620\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TGN-E320\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TFK-G020\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TFK-G210\x00\x00',
      b'77959-TGN-G220\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TFK-G020\x00\x00',
      b'78109-TGN-G120\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36802-TFK-G130\x00\x00',
      b'36802-TGN-G130\x00\x00',
    ],
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-TGN-E010\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab5f1, None): [
      b'36161-TFK-G130\x00\x00',
      b'36161-TGN-G130\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TBA-A020\x00\x00',
    ],
  },
  CAR.CRV: {
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-T1W-A230\x00\x00',
      b'57114-T1W-A240\x00\x00',
      b'57114-TFF-A940\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-T0A-A230\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-T1W-A210\x00\x00',
      b'78109-T1W-C210\x00\x00',
      b'78109-T1X-A210\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36161-T1W-A830\x00\x00',
      b'36161-T1W-C830\x00\x00',
      b'36161-T1X-A830\x00\x00',
    ],
  },
  CAR.CRV_5G: {
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-5PA-AH20\x00\x00',
      b'37805-5PA-3060\x00\x00',
      b'37805-5PA-3080\x00\x00',
      b'37805-5PA-3180\x00\x00',
      b'37805-5PA-4050\x00\x00',
      b'37805-5PA-4150\x00\x00',
      b'37805-5PA-6520\x00\x00',
      b'37805-5PA-6530\x00\x00',
      b'37805-5PA-6630\x00\x00',
      b'37805-5PA-6640\x00\x00',
      b'37805-5PA-7630\x00\x00',
      b'37805-5PA-9630\x00\x00',
      b'37805-5PA-9640\x00\x00',
      b'37805-5PA-9730\x00\x00',
      b'37805-5PA-9830\x00\x00',
      b'37805-5PA-9840\x00\x00',
      b'37805-5PA-A650\x00\x00',
      b'37805-5PA-A670\x00\x00',
      b'37805-5PA-A680\x00\x00',
      b'37805-5PA-A850\x00\x00',
      b'37805-5PA-A870\x00\x00',
      b'37805-5PA-A880\x00\x00',
      b'37805-5PA-A890\x00\x00',
      b'37805-5PA-AB10\x00\x00',
      b'37805-5PA-AD10\x00\x00',
      b'37805-5PA-AF20\x00\x00',
      b'37805-5PA-C680\x00\x00',
      b'37805-5PD-Q630\x00\x00',
      b'37805-5PF-F730\x00\x00',
      b'37805-5PF-M630\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-5RG-A020\x00\x00',
      b'28101-5RG-A030\x00\x00',
      b'28101-5RG-A040\x00\x00',
      b'28101-5RG-A120\x00\x00',
      b'28101-5RG-A220\x00\x00',
      b'28101-5RH-A020\x00\x00',
      b'28101-5RH-A030\x00\x00',
      b'28101-5RH-A040\x00\x00',
      b'28101-5RH-A120\x00\x00',
      b'28101-5RH-A220\x00\x00',
      b'28101-5RL-Q010\x00\x00',
      b'28101-5RM-F010\x00\x00',
      b'28101-5RM-K010\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TLA-A040\x00\x00',
      b'57114-TLA-A050\x00\x00',
      b'57114-TLA-A060\x00\x00',
      b'57114-TLB-A830\x00\x00',
      b'57114-TMC-Z040\x00\x00',
      b'57114-TMC-Z050\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TLA-A040\x00\x00',
      b'39990-TLA-A110\x00\x00',
      b'39990-TLA-A220\x00\x00',
      b'39990-TLA,A040\x00\x00', # modified firmware
      b'39990-TME-T030\x00\x00',
      b'39990-TME-T120\x00\x00',
      b'39990-TMT-T010\x00\x00',
    ],
    (Ecu.electricBrakeBooster, 0x18da2bf1, None): [
      b'46114-TLA-A040\x00\x00',
      b'46114-TLA-A050\x00\x00',
      b'46114-TLA-A930\x00\x00',
      b'46114-TMC-U020\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TLA-A110\x00\x00',
      b'78109-TLA-A120\x00\x00',
      b'78109-TLA-A210\x00\x00',
      b'78109-TLA-A220\x00\x00',
      b'78109-TLA-C110\x00\x00',
      b'78109-TLA-C210\x00\x00',
      b'78109-TLA-C310\x00\x00',
      b'78109-TLB-A020\x00\x00',
      b'78109-TLB-A110\x00\x00',
      b'78109-TLB-A120\x00\x00',
      b'78109-TLB-A210\x00\x00',
      b'78109-TLB-A220\x00\x00',
      b'78109-TMC-Q210\x00\x00',
      b'78109-TMM-F210\x00\x00',
      b'78109-TMM-M110\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TLA-A010\x00\x00',
      b'38897-TLA-A110\x00\x00',
      b'38897-TNY-G010\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36802-TLA-A040\x00\x00',
      b'36802-TLA-A050\x00\x00',
      b'36802-TLA-A060\x00\x00',
      b'36802-TMC-Q040\x00\x00',
      b'36802-TMC-Q070\x00\x00',
      b'36802-TNY-A030\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab5f1, None): [
      b'36161-TLA-A060\x00\x00',
      b'36161-TLA-A070\x00\x00',
      b'36161-TLA-A080\x00\x00',
      b'36161-TMC-Q020\x00\x00',
      b'36161-TMC-Q030\x00\x00',
      b'36161-TMC-Q040\x00\x00',
      b'36161-TNY-A020\x00\x00',
      b'36161-TNY-A030\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TLA-A240\x00\x00',
      b'77959-TLA-A250\x00\x00',
      b'77959-TLA-A320\x00\x00',
      b'77959-TLA-A410\x00\x00',
      b'77959-TLA-A420\x00\x00',
      b'77959-TLA-Q040\x00\x00',
      b'77959-TLA-Z040\x00\x00',
      b'77959-TMM-F040\x00\x00',
    ],
  },
  CAR.CRV_EU: {
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-R5Z-G740\x00\x00',
      b'37805-R5Z-G780\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [b'57114-T1V-G920\x00\x00'],
    (Ecu.fwdRadar, 0x18dab0f1, None): [b'36161-T1V-G520\x00\x00'],
    (Ecu.shiftByWire, 0x18da0bf1, None): [b'54008-T1V-G010\x00\x00'],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-5LH-E120\x00\x00',
      b'28103-5LH-E100\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-T1V-G020\x00\x00',
      b'78109-T1B-3050\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [b'77959-T1G-G940\x00\x00'],
  },
  CAR.CRV_HYBRID: {
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TPA-G020\x00\x00',
      b'57114-TPG-A020\x00\x00',
      b'57114-TMB-H030\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TPA-G030\x00\x00',
      b'39990-TPG-A020\x00\x00',
      b'39990-TMA-H020\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TMA-H110\x00\x00',
      b'38897-TPG-A110\x00\x00',
      b'38897-TPG-A210\x00\x00',
    ],
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-TMB-H510\x00\x00',
      b'54008-TMB-H610\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab5f1, None): [
      b'36161-TMB-H040\x00\x00',
      b'36161-TPA-E050\x00\x00',
      b'36161-TPG-A030\x00\x00',
      b'36161-TPG-A040\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TMB-H220\x00\x00',
      b'78109-TPA-G520\x00\x00',
      b'78109-TPG-A110\x00\x00',
      b'78109-TPG-A210\x00\x00',
    ],
    (Ecu.hud, 0x18da61f1, None): [
      b'78209-TLA-X010\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36802-TPA-E040\x00\x00',
      b'36802-TPG-A020\x00\x00',
      b'36802-TMB-H040\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TLA-C320\x00\x00',
      b'77959-TLA-C410\x00\x00',
      b'77959-TLA-C420\x00\x00',
      b'77959-TLA-G220\x00\x00',
      b'77959-TLA-H240\x00\x00',
    ],
  },
  CAR.FIT: {
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-T5R-L020\x00\x00',
      b'57114-T5R-L220\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-T5R-C020\x00\x00',
      b'39990-T5R-C030\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-T5A-J010\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-T5A-A210\x00\x00',
      b'78109-T5A-A410\x00\x00',
      b'78109-T5A-A420\x00\x00',
      b'78109-T5A-A910\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36161-T5R-A040\x00\x00',
      b'36161-T5R-A240\x00\x00',
      b'36161-T5R-A520\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-T5R-A230\x00\x00',
    ],
  },
  CAR.FREED: {
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TDK-J010\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TDK-J050\x00\x00',
      b'39990-TDK-N020\x00\x00',
    ],
    # TODO: vsa is "essential" for fpv2 but doesn't appear on some models
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TDK-J120\x00\x00',
      b'57114-TDK-J330\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TDK-J310\x00\x00',
      b'78109-TDK-J320\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36161-TDK-J070\x00\x00',
      b'36161-TDK-J080\x00\x00',
      b'36161-TDK-J530\x00\x00',
    ],
  },
  CAR.ODYSSEY: {
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-THR-A010\x00\x00',
      b'38897-THR-A020\x00\x00',
    ],
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-5MR-4080\x00\x00',
      b'37805-5MR-A240\x00\x00',
      b'37805-5MR-A250\x00\x00',
      b'37805-5MR-A310\x00\x00',
      b'37805-5MR-A740\x00\x00',
      b'37805-5MR-A750\x00\x00',
      b'37805-5MR-A840\x00\x00',
      b'37805-5MR-C620\x00\x00',
      b'37805-5MR-D530\x00\x00',
      b'37805-5MR-K730\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-THR-A020\x00\x00',
      b'39990-THR-A030\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-THR-A010\x00\x00',
      b'77959-THR-A110\x00\x00',
      b'77959-THR-X010\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab0f1, None): [
      b'36161-THR-A020\x00\x00',
      b'36161-THR-A030\x00\x00',
      b'36161-THR-A110\x00\x00',
      b'36161-THR-A720\x00\x00',
      b'36161-THR-A730\x00\x00',
      b'36161-THR-A810\x00\x00',
      b'36161-THR-A910\x00\x00',
      b'36161-THR-C010\x00\x00',
      b'36161-THR-D110\x00\x00',
      b'36161-THR-K020\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-5NZ-A110\x00\x00',
      b'28101-5NZ-A310\x00\x00',
      b'28101-5NZ-C310\x00\x00',
      b'28102-5MX-A001\x00\x00',
      b'28102-5MX-A600\x00\x00',
      b'28102-5MX-A610\x00\x00',
      b'28102-5MX-A710\x00\x00',
      b'28102-5MX-A900\x00\x00',
      b'28102-5MX-A910\x00\x00',
      b'28102-5MX-C001\x00\x00',
      b'28102-5MX-D001\x00\x00',
      b'28102-5MX-D710\x00\x00',
      b'28102-5MX-K610\x00\x00',
      b'28103-5NZ-A100\x00\x00',
      b'28103-5NZ-A300\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-THR-A040\x00\x00',
      b'57114-THR-A110\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-THR-A220\x00\x00',
      b'78109-THR-A230\x00\x00',
      b'78109-THR-A420\x00\x00',
      b'78109-THR-A430\x00\x00',
      b'78109-THR-A720\x00\x00',
      b'78109-THR-A820\x00\x00',
      b'78109-THR-A830\x00\x00',
      b'78109-THR-AB20\x00\x00',
      b'78109-THR-AB30\x00\x00',
      b'78109-THR-AB40\x00\x00',
      b'78109-THR-AC20\x00\x00',
      b'78109-THR-AC30\x00\x00',
      b'78109-THR-AC40\x00\x00',
      b'78109-THR-AC50\x00\x00',
      b'78109-THR-AD30\x00\x00',
      b'78109-THR-AE20\x00\x00',
      b'78109-THR-AE30\x00\x00',
      b'78109-THR-AE40\x00\x00',
      b'78109-THR-AK10\x00\x00',
      b'78109-THR-AL10\x00\x00',
      b'78109-THR-AN10\x00\x00',
      b'78109-THR-C220\x00\x00',
      b'78109-THR-C330\x00\x00',
      b'78109-THR-CE20\x00\x00',
      b'78109-THR-DA20\x00\x00',
      b'78109-THR-DA30\x00\x00',
      b'78109-THR-DA40\x00\x00',
      b'78109-THR-K120\x00\x00',
    ],
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-THR-A020\x00\x00',
    ],
  },
  CAR.PILOT: {
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-TG7-A520\x00\x00',
      b'54008-TG7-A530\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-5EY-A050\x00\x00',
      b'28101-5EY-A100\x00\x00',
      b'28101-5EZ-A050\x00\x00',
      b'28101-5EZ-A060\x00\x00',
      b'28101-5EZ-A100\x00\x00',
      b'28101-5EZ-A210\x00\x00',
    ],
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-RLV-4060\x00\x00',
      b'37805-RLV-4070\x00\x00',
      b'37805-RLV-A830\x00\x00',
      b'37805-RLV-A840\x00\x00',
      b'37805-RLV-C430\x00\x00',
      b'37805-RLV-C510\x00\x00',
      b'37805-RLV-C520\x00\x00',
      b'37805-RLV-C530\x00\x00',
      b'37805-RLV-C910\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TG7-A030\x00\x00',
      b'39990-TG7-A040\x00\x00',
      b'39990-TG7-A060\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab0f1, None): [
      b'36161-TG7-A520\x00\x00',
      b'36161-TG7-A720\x00\x00',
      b'36161-TG7-A820\x00\x00',
      b'36161-TG7-C520\x00\x00',
      b'36161-TG7-D520\x00\x00',
      b'36161-TG8-A520\x00\x00',
      b'36161-TG8-A720\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TG7-A110\x00\x00',
      b'77959-TG7-A020\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TG7-A040\x00\x00',
      b'78109-TG7-A050\x00\x00',
      b'78109-TG7-A420\x00\x00',
      b'78109-TG7-A520\x00\x00',
      b'78109-TG7-A720\x00\x00',
      b'78109-TG7-D020\x00\x00',
      b'78109-TG8-A420\x00\x00',
      b'78109-TG8-A520\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TG7-A130\x00\x00',
      b'57114-TG7-A140\x00\x00',
      b'57114-TG7-A230\x00\x00',
      b'57114-TG7-A240\x00\x00',
      b'57114-TG8-A140\x00\x00',
      b'57114-TG8-A240\x00\x00',
    ],

  },
  CAR.PILOT_2019: {
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TG7-A060\x00\x00',
      b'39990-TG7-A070\x00\x00',
      b'39990-TGS-A230\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TG7-A030\x00\x00',
      b'38897-TG7-A040\x00\x00',
      b'38897-TG7-A110\x00\x00',
      b'38897-TG7-A210\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab0f1, None): [
      b'36161-TG7-A310\x00\x00',
      b'36161-TG7-A630\x00\x00',
      b'36161-TG7-A930\x00\x00',
      b'36161-TG7-D630\x00\x00',
      b'36161-TG7-Y630\x00\x00',
      b'36161-TG8-A630\x00\x00',
      b'36161-TG8-A830\x00\x00',
      b'36161-TGS-A130\x00\x00',
      b'36161-TGT-A030\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TG7-A210\x00\x00',
      b'77959-TG7-Y210\x00\x00',
      b'77959-TGS-A010\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TG7-AJ10\x00\x00',
      b'78109-TG7-AJ20\x00\x00',
      b'78109-TG7-AK10\x00\x00',
      b'78109-TG7-AK20\x00\x00',
      b'78109-TG7-AM20\x00\x00',
      b'78109-TG7-AP10\x00\x00',
      b'78109-TG7-AP20\x00\x00',
      b'78109-TG7-AS20\x00\x00',
      b'78109-TG7-AU20\x00\x00',
      b'78109-TG7-DJ10\x00\x00',
      b'78109-TG7-YK20\x00\x00',
      b'78109-TG8-AJ10\x00\x00',
      b'78109-TG8-AJ20\x00\x00',
      b'78109-TG8-AK20\x00\x00',
      b'78109-TGS-AK20\x00\x00',
      b'78109-TGS-AP20\x00\x00',
      b'78109-TGT-AJ20\x00\x00',
      b'78109-TG7-AT20\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TG7-A630\x00\x00',
      b'57114-TG7-A730\x00\x00',
      b'57114-TG8-A630\x00\x00',
      b'57114-TG8-A730\x00\x00',
      b'57114-TGS-A530\x00\x00',
      b'57114-TGT-A530\x00\x00',
    ],
  },
  CAR.PASSPORT: {
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-RLV-B220\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TGS-A230\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36161-TGS-A030\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TG7-A040\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TGS-A010\x00\x00',
    ],
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-TG7-A530\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28101-5EZ-A600\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TGS-AT20\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TGS-A530\x00\x00',
    ],
  },
  CAR.ACURA_RDX: {
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TX5-A220\x00\x00',
      b'57114-TX4-A220\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab0f1, None): [
      b'36161-TX5-A030\x00\x00',
      b'36161-TX4-A030\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TX4-C010\x00\x00',
      b'77959-TX4-B010\x00\x00',
      b'77959-TX4-C020\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TX5-A310\x00\x00',
      b'78109-TX4-A210\x00\x00',
      b'78109-TX4-A310\x00\x00',
    ],
  },
  CAR.ACURA_RDX_3G: {
    (Ecu.programmedFuelInjection, 0x18da10f1, None): [
      b'37805-5YF-A130\x00\x00',
      b'37805-5YF-A230\x00\x00',
      b'37805-5YF-A320\x00\x00',
      b'37805-5YF-A330\x00\x00',
      b'37805-5YF-A420\x00\x00',
      b'37805-5YF-A430\x00\x00',
      b'37805-5YF-A750\x00\x00',
      b'37805-5YF-A850\x00\x00',
      b'37805-5YF-A870\x00\x00',
      b'37805-5YF-C210\x00\x00',
      b'37805-5YF-C220\x00\x00',
      b'37805-5YF-C410\000\000',
      b'37805-5YF-C420\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TJB-A030\x00\x00',
      b'57114-TJB-A040\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36802-TJB-A040\x00\x00',
      b'36802-TJB-A050\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab5f1, None): [
      b'36161-TJB-A040\x00\x00',
    ],
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-TJB-A520\x00\x00',
    ],
    (Ecu.transmission, 0x18da1ef1, None): [
      b'28102-5YK-A610\x00\x00',
      b'28102-5YK-A620\x00\x00',
      b'28102-5YK-A630\x00\x00',
      b'28102-5YK-A700\x00\x00',
      b'28102-5YK-A711\x00\x00',
      b'28102-5YL-A620\x00\x00',
      b'28102-5YL-A700\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TJB-A140\x00\x00',
      b'78109-TJB-A240\x00\x00',
      b'78109-TJB-A420\x00\x00',
      b'78109-TJB-AB10\x00\x00',
      b'78109-TJB-AD10\x00\x00',
      b'78109-TJB-AF10\x00\x00',
      b'78109-TJB-AS10\000\000',
      b'78109-TJB-AU10\x00\x00',
      b'78109-TJB-AW10\x00\x00',
      b'78109-TJC-A420\x00\x00',
      b'78109-TJC-AA10\x00\x00',
      b'78109-TJC-AD10\x00\x00',
      b'78109-TJC-AF10\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TJB-A040\x00\x00',
      b'77959-TJB-A210\x00\x00',
    ],
    (Ecu.electricBrakeBooster, 0x18da2bf1, None): [
      b'46114-TJB-A040\x00\x00',
      b'46114-TJB-A050\x00\x00',
      b'46114-TJB-A060\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TJB-A040\x00\x00',
      b'38897-TJB-A110\x00\x00',
      b'38897-TJB-A120\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TJB-A030\x00\x00',
      b'39990-TJB-A040\x00\x00',
      b'39990-TJB-A130\x00\x00'
    ],
  },
  CAR.RIDGELINE: {
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-T6Z-A020\x00\x00',
      b'39990-T6Z-A030\x00\x00',
      b'39990-T6Z-A050\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab0f1, None): [
      b'36161-T6Z-A020\x00\x00',
      b'36161-T6Z-A310\x00\x00',
      b'36161-T6Z-A420\x00\x00',
      b'36161-T6Z-A520\x00\x00',
      b'36161-T6Z-A620\x00\x00',
      b'36161-TJZ-A120\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-T6Z-A010\x00\x00',
      b'38897-T6Z-A110\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-T6Z-A420\x00\x00',
      b'78109-T6Z-A510\x00\x00',
      b'78109-T6Z-A710\x00\x00',
      b'78109-T6Z-A810\x00\x00',
      b'78109-T6Z-A910\x00\x00',
      b'78109-T6Z-AA10\x00\x00',
      b'78109-T6Z-C620\x00\x00',
      b'78109-TJZ-A510\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-T6Z-A020\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-T6Z-A120\x00\x00',
      b'57114-T6Z-A130\x00\x00',
      b'57114-T6Z-A520\x00\x00',
      b'57114-TJZ-A520\x00\x00',
    ],
  },
  CAR.INSIGHT: {
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-TXM-A040\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36802-TXM-A070\x00\x00',
    ],
    (Ecu.fwdCamera, 0x18dab5f1, None): [
      b'36161-TXM-A050\x00\x00',
      b'36161-TXM-A060\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TXM-A230\x00\x00',
    ],
    (Ecu.vsa, 0x18da28f1, None): [
      b'57114-TXM-A030\x00\x00',
      b'57114-TXM-A040\x00\x00',
    ],
    (Ecu.shiftByWire, 0x18da0bf1, None): [
      b'54008-TWA-A910\x00\x00',
    ],
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TXM-A020\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-TXM-A010\x00\x00',
      b'78109-TXM-A020\x00\x00',
      b'78109-TXM-A110\x00\x00',
      b'78109-TXM-C010\x00\x00',
      b'78109-TXM-A030\x00\x00',
    ],
  },
  CAR.HRV: {
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-T7A-A010\x00\x00',
      b'38897-T7A-A110\x00\x00',
    ],
    (Ecu.eps, 0x18da30f1, None): [
      b'39990-THX-A020\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36161-T7A-A140\x00\x00',
      b'36161-T7A-A240\x00\x00',
      b'36161-T7A-C440\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-T7A-A230\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-THX-A110\x00\x00',
      b'78109-THX-A210\x00\x00',
      b'78109-THX-A220\x00\x00',
      b'78109-THX-C220\x00\x00',
    ],
  },
  CAR.ACURA_ILX: {
    (Ecu.gateway, 0x18daeff1, None): [
      b'38897-TX6-A010\x00\x00',
    ],
    (Ecu.fwdRadar, 0x18dab0f1, None): [
      b'36161-TV9-A140\x00\x00',
      b'36161-TX6-A030\x00\x00',
    ],
    (Ecu.srs, 0x18da53f1, None): [
      b'77959-TX6-A230\x00\x00',
      b'77959-TX6-C210\x00\x00',
    ],
    (Ecu.combinationMeter, 0x18da60f1, None): [
      b'78109-T3R-A120\x00\x00',
      b'78109-T3R-A410\x00\x00',
      b'78109-TV9-A510\x00\x00',
    ],
  },
  CAR.HONDA_E:{
    (Ecu.eps, 0x18DA30F1, None):[
      b'39990-TYF-N030\x00\x00'
    ],
    (Ecu.gateway, 0x18DAEFF1, None):[
      b'38897-TYF-E140\x00\x00'
    ],
    (Ecu.shiftByWire, 0x18DA0BF1, None):[
      b'54008-TYF-E010\x00\x00'
    ],
    (Ecu.srs, 0x18DA53F1, None):[
      b'77959-TYF-G430\x00\x00'
    ],
    (Ecu.combinationMeter, 0x18DA60F1, None):[
      b'78108-TYF-G610\x00\x00'
    ],
    (Ecu.fwdRadar, 0x18DAB0F1, None):[
      b'36802-TYF-E030\x00\x00'
    ],
    (Ecu.fwdCamera, 0x18DAB5F1, None):[
      b'36161-TYF-E020\x00\x00'
    ],
    (Ecu.vsa, 0x18DA28F1, None):[
      b'57114-TYF-E030\x00\x00'
    ],
  },
}

DBC = {
  CAR.ACCORD: dbc_dict('honda_accord_2018_can_generated', None),
  CAR.ACCORDH: dbc_dict('honda_accord_2018_can_generated', None),
  CAR.ACURA_ILX: dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.ACURA_RDX: dbc_dict('acura_rdx_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.ACURA_RDX_3G: dbc_dict('acura_rdx_2020_can_generated', None),
  CAR.CIVIC: dbc_dict('honda_civic_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CIVIC_BOSCH: dbc_dict('honda_civic_hatchback_ex_2017_can_generated', None),
  CAR.CIVIC_BOSCH_DIESEL: dbc_dict('honda_civic_sedan_16_diesel_2019_can_generated', None),
  CAR.CRV: dbc_dict('honda_crv_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CRV_5G: dbc_dict('honda_crv_ex_2017_can_generated', None, body_dbc='honda_crv_ex_2017_body_generated'),
  CAR.CRV_EU: dbc_dict('honda_crv_executive_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CRV_HYBRID: dbc_dict('honda_crv_hybrid_2019_can_generated', None),
  CAR.FIT: dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.FREED: dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.HRV: dbc_dict('honda_fit_ex_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.ODYSSEY: dbc_dict('honda_odyssey_exl_2018_generated', 'acura_ilx_2016_nidec'),
  CAR.ODYSSEY_CHN: dbc_dict('honda_odyssey_extreme_edition_2018_china_can_generated', 'acura_ilx_2016_nidec'),
  CAR.PILOT: dbc_dict('honda_pilot_touring_2017_can_generated', 'acura_ilx_2016_nidec'),
  CAR.PILOT_2019: dbc_dict('honda_pilot_touring_2017_can_generated', 'acura_ilx_2016_nidec'),
  CAR.PASSPORT: dbc_dict('honda_pilot_touring_2017_can_generated', 'acura_ilx_2016_nidec'),
  CAR.RIDGELINE: dbc_dict('honda_ridgeline_black_edition_2017_can_generated', 'acura_ilx_2016_nidec'),
  CAR.INSIGHT: dbc_dict('honda_insight_ex_2019_can_generated', None),
  CAR.HONDA_E: dbc_dict('acura_rdx_2020_can_generated', None),
}

STEER_THRESHOLD = {
  # default is 1200, overrides go here
  CAR.ACURA_RDX: 400,
  CAR.CRV_EU: 400,
}

# TODO: is this real?
SPEED_FACTOR = {
  # default is 1, overrides go here
  CAR.CRV: 1.025,
  CAR.CRV_5G: 1.025,
  CAR.CRV_EU: 1.025,
  CAR.CRV_HYBRID: 1.025,
  CAR.HRV: 1.025,
}

HONDA_NIDEC_ALT_PCM_ACCEL = set([CAR.ODYSSEY])
HONDA_NIDEC_ALT_SCM_MESSAGES = set([CAR.ACURA_ILX, CAR.ACURA_RDX, CAR.CRV, CAR.CRV_EU, CAR.FIT, CAR.FREED, CAR.HRV, CAR.ODYSSEY_CHN,
                                    CAR.PILOT, CAR.PILOT_2019, CAR.PASSPORT, CAR.RIDGELINE])
HONDA_BOSCH = set([CAR.ACCORD, CAR.ACCORDH, CAR.CIVIC_BOSCH, CAR.CIVIC_BOSCH_DIESEL, CAR.CRV_5G, CAR.CRV_HYBRID, CAR.INSIGHT, CAR.ACURA_RDX_3G, CAR.HONDA_E])
HONDA_BOSCH_ALT_BRAKE_SIGNAL = set([CAR.ACCORD, CAR.CRV_5G, CAR.ACURA_RDX_3G])
