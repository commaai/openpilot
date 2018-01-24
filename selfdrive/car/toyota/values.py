#
# Adding a car:
#   Add CAR (name used most everywhere)
#   Add STATIC_MSGS the car uses (refactoring soon)
#   Add CAN_GEAR_DICT (may be able to copy an existing one, like Prius Prime does)
#   Add CAR_DETAILS (may be able to base on another, like Prius Prime)
#   --- Don't forget to add the Fingerprint
#

class CAR: 
  PRIUS = "TOYOTA PRIUS 2017"
  RAV4H = "TOYOTA RAV4 2017 HYBRID"
  RAV4 = "TOYOTA RAV4 2017"
  PRIUSP = "TOYOTA PRIUS PRIME 2017"

class ECU: 
  CAM = 0 # camera 
  DSU = 1 # driving support unit 
  APGS = 2 # advanced parking guidance system 


# addr, [ecu, bus, 1/freq*100, vl]
#   TODO: Refactor the list of cars to come from the car, and say car has signals 0x141, 0x128, etc. then iterate signals and add to the list for that signal
#         I think that's easier for a human to read / understand

STATIC_MSGS = {0x141: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   2, '\x00\x00\x00\x46'),
               0x128: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   3, '\xf4\x01\x90\x83\x00\x37'),

               0x292: (ECU.APGS, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0,   3, '\x00\x00\x00\x00\x00\x00\x00\x9e'),
               0x283: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0,   3, '\x00\x00\x00\x00\x00\x00\x8c'),
               0x2E6: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H,), 0,   3, '\xff\xf8\x00\x08\x7f\xe0\x00\x4e'),
               0x2E7: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H,), 0,   3, '\xa8\x9c\x31\x9c\x00\x00\x00\x02'),

               0x240: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               0x241: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               0x244: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               0x245: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
               0x248: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   5, '\x00\x00\x00\x00\x00\x00\x01'),
               0x344: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0,   5, '\x00\x00\x01\x00\x00\x00\x00\x50'),

               0x160: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   7, '\x00\x00\x08\x12\x01\x31\x9c\x51'),
               0x161: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1,   7, '\x00\x1e\x00\x00\x00\x80\x07'),

               0x32E: (ECU.APGS, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0,  20, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               0x33E: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H,), 0,  20, '\x0f\xff\x26\x40\x00\x1f\x00'),
               0x365: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H,), 0,  20, '\x00\x00\x00\x80\x03\x00\x08'),
               0x365: (ECU.DSU, (CAR.RAV4,), 0,  20, '\x00\x00\x00\x80\xfc\x00\x08'),
               0x366: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H,), 0,  20, '\x00\x00\x4d\x82\x40\x02\x00'),
               0x366: (ECU.DSU, (CAR.RAV4,), 0,  20, '\x00\x72\x07\xff\x09\xfe\x00'),

               0x367: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0,  40, '\x06\x00'),

               0x414: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x17\x00'),
               0x489: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
               0x48a: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
               0x48b: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x66\x06\x08\x0a\x02\x00\x00\x00'),
               0x4d3: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x1C\x00\x00\x01\x00\x00\x00\x00'),
               0x130: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1, 100, '\x00\x00\x00\x00\x00\x00\x38'),
               0x466: (ECU.CAM, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 1, 100, '\x20\x20\xAD'),
               0x396: (ECU.APGS, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\xBD\x00\x00\x00\x60\x0F\x02\x00'),
               0x43A: (ECU.APGS, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x84\x00\x00\x00\x00\x00\x00\x00'),
               0x43B: (ECU.APGS, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               0x497: (ECU.APGS, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
               0x4CC: (ECU.APGS, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x0D\x00\x00\x00\x00\x00\x00\x00'),
               0x4CB: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H, CAR.RAV4), 0, 100, '\x0c\x00\x00\x00\x00\x00\x00\x00'),
               0x470: (ECU.DSU, (CAR.PRIUS, CAR.PRIUSP, CAR.RAV4H,), 1, 100, '\x00\x00\x02\x7a'),
              }

# CAN_GEAR Dictionary - Get this from DBC soon:
CAN_GEAR_DICT = {}

CAN_GEAR_DICT[CAR.PRIUS] = {
  0x0: "park",
  0x1: "reverse",
  0x2: "neutral",
  0x3: "drive",
  0x4: "brake"
  }

# Prius Prime seems same as Prius:
CAN_GEAR_DICT[CAR.PRIUSP] = CAN_GEAR_DICT[CAR.PRIUS]

CAN_GEAR_DICT[CAR.RAV4] = {
  0x20: "park",
  0x10: "reverse",
  0x8: "neutral",
  0x0: "drive",
  0x1: "sport"
  }

# For this RAV4H - Hybrid - seems same as RAV4:
CAN_GEAR_DICT[CAR.RAV4H] = CAN_GEAR_DICT[CAR.RAV4]

# CAR_DETAILS - Get this from DBC soon: 
# Maybe iterate all DBC files and load?
# Ideally even add Fingerprint from comment in DBC?

CAR_DETAILS = {}

CAR_DETAILS[CAR.PRIUS] = {
    "dbc_f": 'toyota_prius_2017_pt.dbc',
    "signals": [
      ("GEAR", 295, 0),
      ("BRAKE_PRESSED", 550, 0),
      ("GAS_PEDAL", 581, 0),
    ],
    "checks": [
      (550, 40),
      (581, 33)
    ]
  }

CAR_DETAILS[CAR.RAV4H] = {
    "dbc_f": 'toyota_rav4_hybrid_2017_pt.dbc',
    "signals": [
      ("GEAR", 956, 0),
      ("BRAKE_PRESSED", 550, 0),
      ("GAS_PEDAL", 581, 0)
    ],
    "checks": [
      (550, 40),
      (581, 33)
    ]
  }

CAR_DETAILS[CAR.RAV4] = {
    "dbc_f": 'toyota_rav4_2017_pt.dbc',
    "signals" : [
      ("GEAR", 956, 0x20),
      ("BRAKE_PRESSED", 548, 0),
      ("GAS_PEDAL", 705, 0)
    ],
    "checks": [
      (548, 40),
      (705, 33)
    ]
  }

CAR_DETAILS[CAR.PRIUSP] = CAR_DETAILS[CAR.PRIUS]
CAR_DETAILS[CAR.PRIUSP]["dbc_f"] = 'toyota_prius_2017_pt.dbc'  # TODO: This had worked, but we should have a new dbc file


# This really is just spinning the CAR_DETAILS into a different lookup format
CAN_PEDALS_DICT = {}

for key in CAR_DETAILS:
  CAN_PEDAL = {}
  for signal in CAR_DETAILS[key]["signals"]:
    if signal[0] in ['GEAR', 'BRAKE_PRESSED']:
      CAN_PEDAL["{}_ID".format(signal[0])] = signal[1]
      # This is redundant? Keeping for now for backwards compat, check later if needed
      CAN_PEDAL["{}_STRING".format(signal[0])] = signal[0]
    elif signal[0] == 'GAS_PEDAL':
      CAN_PEDAL["PEDAL_GAS_ID"] = signal[1]
      # This is redundant? Keeping for now for backwards compat, check later if needed
      CAN_PEDAL["PEDAL_GAS_STRING"] = signal[0]

  CAN_PEDALS_DICT[key] = CAN_PEDAL



def check_ecu_msgs(fingerprint, candidate, ecu):
  # return True if fingerprint contains messages normally sent by a given ecu
  ecu_msgs = [x for x in STATIC_MSGS if (ecu == STATIC_MSGS[x][0] and 
                                         candidate in STATIC_MSGS[x][1] and 
                                         STATIC_MSGS[x][2] == 0)]

  return any(msg for msg in fingerprint if msg in ecu_msgs)
