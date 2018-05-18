from common.fingerprints import CHRYSLER as CAR

class ECU:
  CAM = 0 # LKAS camera


# addr: (ecu, cars, bus, 1/freq*100, vl)
STATIC_MSGS = [(0x2d9, ECU.CAM, (CAR.PACIFICA), 0,   10, '\x00\x00\x00\x08\x20'),
               # TODO verify the 10 here is for every 0.1 seconds
               # 0x2a6 and 0x292 are not static, so they're not included here.
              ]


def check_ecu_msgs(fingerprint, candidate, ecu):
  # return True if fingerprint contains messages normally sent by a given ecu
  ecu_msgs = [x[0] for x in STATIC_MSGS if (x[1] == ecu and
                                            candidate in x[2] and
                                            x[3] == 0)]

  return any(msg for msg in fingerprint if msg in ecu_msgs)
