
_FINGERPRINTS = {
  "ACURA ILX 2016 ACURAWATCH PLUS": {
    1024L: 5, 513L: 5, 1027L: 5, 1029L: 8, 929L: 4, 1057L: 5, 777L: 8, 1034L: 5, 1036L: 8, 398L: 3, 399L: 7, 145L: 8, 660L: 8, 985L: 3, 923L: 2, 542L: 7, 773L: 7, 800L: 8, 432L: 7, 419L: 8, 420L: 8, 1030L: 5, 422L: 8, 808L: 8, 428L: 8, 304L: 8, 819L: 7, 821L: 5, 57L: 3, 316L: 8, 545L: 4, 464L: 8, 1108L: 8, 597L: 8, 342L: 6, 983L: 8, 344L: 8, 804L: 8, 1039L: 8, 476L: 4, 892L: 8, 490L: 8, 1064L: 7, 882L: 2, 884L: 7, 887L: 8, 888L: 8, 380L: 8, 1365L: 5,
    # sent messages
    0xe4: 5, 0x1fa: 8, 0x200: 3, 0x30c: 8, 0x33d: 5,
  },
  "HONDA CIVIC 2016 TOURING": {
    1024L: 5, 513L: 5, 1027L: 5, 1029L: 8, 777L: 8, 1036L: 8, 1039L: 8, 1424L: 5, 401L: 8, 148L: 8, 662L: 4, 985L: 3, 795L: 8, 773L: 7, 800L: 8, 545L: 6, 420L: 8, 806L: 8, 808L: 8, 1322L: 5, 427L: 3, 428L: 8, 304L: 8, 432L: 7, 57L: 3, 450L: 8, 929L: 8, 330L: 8, 1302L: 8, 464L: 8, 1361L: 5, 1108L: 8, 597L: 8, 470L: 2, 344L: 8, 804L: 8, 399L: 7, 476L: 7, 1633L: 8, 487L: 4, 892L: 8, 490L: 8, 493L: 5, 884L: 8, 891L: 8, 380L: 8, 1365L: 5,
    # sent messages
    0xe4: 5, 0x1fa: 8, 0x200: 3, 0x30c: 8, 0x33d: 5, 0x35e: 8, 0x39f: 8,
  },
  "HONDA ACCORD 2016 TOURING": {
    1024L: 5, 929L: 8, 1027L: 5, 773L: 7, 1601L: 8, 777L: 8, 1036L: 8, 398L: 3, 1039L: 8, 401L: 8, 145L: 8, 1424L: 5, 660L: 8, 661L: 4, 918L: 7, 985L: 3, 923L: 2, 542L: 7, 927L: 8, 800L: 8, 545L: 4, 420L: 8, 422L: 8, 808L: 8, 426L: 8, 1029L: 8, 432L: 7, 57L: 3, 316L: 8, 829L: 5, 1600L: 5, 1089L: 8, 1057L: 5, 780L: 8, 1088L: 8, 464L: 8, 1108L: 8, 597L: 8, 342L: 6, 983L: 8, 344L: 8, 804L: 8, 476L: 4, 1296L: 3, 891L: 8, 1125L: 8, 487L: 4, 892L: 8, 490L: 8, 871L: 8, 1064L: 7, 882L: 2, 884L: 8, 506L: 8, 507L: 1, 380L: 8, 1365L: 5
  },
  "HONDA CR-V 2016 TOURING": {
    57L: 3, 145L: 8, 316L: 8, 340L: 8, 342L: 6, 344L: 8, 380L: 8, 398L: 3, 399L: 6, 401L: 8, 420L: 8, 422L: 8, 426L: 8, 432L: 7, 464L: 8, 474L: 5, 476L: 4, 487L: 4, 490L: 8, 493L: 3, 507L: 1, 542L: 7, 545L: 4, 597L: 8, 660L: 8, 661L: 4, 773L: 7, 777L: 8, 800L: 8, 804L: 8, 808L: 8, 882L: 2, 884L: 7, 888L: 8, 891L: 8, 892L: 8, 923L: 2, 929L: 8, 983L: 8, 985L: 3, 1024L: 5, 1027L: 5, 1029L: 8, 1033L: 5, 1036L: 8, 1039L: 8, 1057L: 5, 1064L: 7, 1108L: 8, 1125L: 8, 1296L: 8, 1365L: 5, 1424L: 5, 1600L: 5, 1601L: 8,
    # sent messages
    0x194: 4, 0x1fa: 8, 0x30c: 8, 0x33d: 5,
  }
}

def eliminate_incompatible_cars(msg, candidate_cars):
  """Removes cars that could not have sent msg.

     Inputs:
      msg: A cereal/log CanData message from the car.
      candidate_cars: A list of cars to consider.

     Returns:
      A list containing the subset of candidate_cars that could have sent msg.
  """
  compatible_cars = []
  for car_name in candidate_cars:
    adr = msg.address
    if msg.src != 0 or (adr in _FINGERPRINTS[car_name] and
                        _FINGERPRINTS[car_name][adr] == len(msg.dat)):
      compatible_cars.append(car_name)
    else:
      pass
      #isin = adr in _FINGERPRINTS[car_name]
      #print "eliminate", car_name, hex(adr), isin, len(msg.dat), msg.dat.encode("hex")
  return compatible_cars

def all_known_cars():
  """Returns a list of all known car strings."""
  return _FINGERPRINTS.keys()

# **** for use live only ****
def fingerprint(logcan):
  import selfdrive.messaging as messaging
  from cereal import car
  from common.realtime import sec_since_boot
  import os
  if os.getenv("SIMULATOR") is not None or logcan is None:
    # send message
    ret = car.CarParams.new_message()

    ret.carName = "simulator"
    ret.radarName = "nidec"
    ret.carFingerprint = "THE LOW QUALITY SIMULATOR"

    ret.enableSteer = True
    ret.enableBrake = True
    ret.enableGas = True
    ret.enableCruise = False

    ret.wheelBase = 2.67
    ret.steerRatio = 15.3
    ret.slipFactor = 0.0014

    ret.steerKp, ret.steerKi = 12.0, 1.0
    return ret

  print "waiting for fingerprint..."
  brake_only = True

  candidate_cars = all_known_cars()
  finger = {}
  st = None
  while 1:
    for a in messaging.drain_sock(logcan, wait_for_one=True):
      if st is None:
        st = sec_since_boot()
      for can in a.can:
        # pedal
        if can.address == 0x201 and can.src == 0:
          brake_only = False
        if can.src == 0:
          finger[can.address] = len(can.dat)
        candidate_cars = eliminate_incompatible_cars(can, candidate_cars)

    # if we only have one car choice and it's been 100ms since we got our first message, exit
    if len(candidate_cars) == 1 and st is not None and (sec_since_boot()-st) > 0.1:
      break
    elif len(candidate_cars) == 0:
      print map(hex, finger.keys())
      raise Exception("car doesn't match any fingerprints")

  print "fingerprinted", candidate_cars[0]

  # send message
  ret = car.CarParams.new_message()

  ret.carName = "honda"
  ret.radarName = "nidec"
  ret.carFingerprint = candidate_cars[0]

  ret.enableSteer = True
  ret.enableBrake = True
  ret.enableGas = not brake_only
  ret.enableCruise = brake_only
  #ret.enableCruise = False

  ret.wheelBase = 2.67
  ret.steerRatio = 15.3
  ret.slipFactor = 0.0014

  if candidate_cars[0] == "HONDA CIVIC 2016 TOURING":
    ret.steerKp, ret.steerKi = 12.0, 1.0
  elif candidate_cars[0] == "ACURA ILX 2016 ACURAWATCH PLUS":
    if not brake_only:
      # assuming if we have an interceptor we also have a torque mod
      ret.steerKp, ret.steerKi = 6.0, 0.5
    else:
      ret.steerKp, ret.steerKi = 12.0, 1.0
  elif candidate_cars[0] == "HONDA ACCORD 2016 TOURING":
    ret.steerKp, ret.steerKi = 12.0, 1.0
  elif candidate_cars[0] == "HONDA CR-V 2016 TOURING":
    ret.steerKp, ret.steerKi = 6.0, 0.5
  else:
    raise ValueError("unsupported car %s" % candidate_cars[0])

  return ret
