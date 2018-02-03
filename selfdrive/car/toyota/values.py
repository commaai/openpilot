from common.fingerprints import TOYOTA as CAR

class ECU:
  CAM = 0 # camera
  DSU = 1 # driving support unit
  APGS = 2 # advanced parking guidance system


# addr: (ecu, cars, bus, 1/freq*100, vl)
STATIC_MSGS = [(0x141, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   2, '\x00\x00\x00\x46'),
               (0x128, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   3, '\xf4\x01\x90\x83\x00\x37'),

               (0x292, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0,   3, '\x00\x00\x00\x00\x00\x00\x00\x9e'),
               (0x283, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0,   3, '\x00\x00\x00\x00\x00\x00\x8c'),
               (0x2E6, ECU.DSU, (CAR.PRIUS, CAR.RAV4H), 0,   3, '\xff\xf8\x00\x08\x7f\xe0\x00\x4e'),
               (0x2E7, ECU.DSU, (CAR.PRIUS, CAR.RAV4H), 0,   3, '\xa8\x9c\x31\x9c\x00\x00\x00\x02'),

               (0x240, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               (0x241, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               (0x244, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               (0x245, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               (0x248, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x00\x00\x00\x00\x00\x01'),
               (0x344, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0,   5, '\x00\x00\x01\x00\x00\x00\x00\x50'),

               (0x160, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   7, '\x00\x00\x08\x12\x01\x31\x9c\x51'),
               (0x161, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1,   7, '\x00\x1e\x00\x00\x00\x80\x07'),

               (0x32E, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0,  20, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               (0x33E, ECU.DSU, (CAR.PRIUS, CAR.RAV4H), 0,  20, '\x0f\xff\x26\x40\x00\x1f\x00'),
               (0x365, ECU.DSU, (CAR.PRIUS, CAR.RAV4H), 0,  20, '\x00\x00\x00\x80\x03\x00\x08'),
               (0x365, ECU.DSU, (CAR.RAV4, CAR.COROLLA), 0,  20, '\x00\x00\x00\x80\xfc\x00\x08'),
               (0x366, ECU.DSU, (CAR.PRIUS, CAR.RAV4H), 0,  20, '\x00\x00\x4d\x82\x40\x02\x00'),
               (0x366, ECU.DSU, (CAR.RAV4, CAR.COROLLA), 0,  20, '\x00\x72\x07\xff\x09\xfe\x00'),

               (0x367, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0,  40, '\x06\x00'),

               (0x414, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x17\x00'),
               (0x489, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
               (0x48a, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
               (0x48b, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x66\x06\x08\x0a\x02\x00\x00\x00'),
               (0x4d3, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x1C\x00\x00\x01\x00\x00\x00\x00'),
               (0x130, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 1, 100, '\x00\x00\x00\x00\x00\x00\x38'),
               (0x466, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4), 1, 100, '\x20\x20\xAD'),
               (0x466, ECU.CAM, (CAR.COROLLA), 1, 100, '\x24\x20\xB1'),
               (0x396, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\xBD\x00\x00\x00\x60\x0F\x02\x00'),
               (0x43A, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x84\x00\x00\x00\x00\x00\x00\x00'),
               (0x43B, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               (0x497, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               (0x4CC, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x0D\x00\x00\x00\x00\x00\x00\x00'),
               (0x4CB, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.RAV4, CAR.COROLLA), 0, 100, '\x0c\x00\x00\x00\x00\x00\x00\x00'),
               (0x470, ECU.DSU, (CAR.PRIUS, CAR.RAV4H), 1, 100, '\x00\x00\x02\x7a'),
              ]


def check_ecu_msgs(fingerprint, candidate, ecu):
  # return True if fingerprint contains messages normally sent by a given ecu
  ecu_msgs = [x[0] for x in STATIC_MSGS if (x[1] == ecu and
                                            candidate in x[2] and
                                            x[3] == 0)]

  return any(msg for msg in fingerprint if msg in ecu_msgs)
