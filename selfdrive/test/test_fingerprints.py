#!/usr/bin/env python3
import os
import sys
from common.basedir import BASEDIR

# messages reserved for CAN based ignition (see can_ignition_hook function in panda/board/drivers/can)
# (addr, len)
CAN_IGNITION_MSGS = {
  'gm': [(0x1F1, 8), (0x160, 5)],
  #'tesla' : [(0x348, 8)],
}

def _get_fingerprints():
  # read all the folders in selfdrive/car and return a dict where:
  # - keys are all the car names that which we have a fingerprint dict for
  # - values are dicts of fingeprints for each trim
  fingerprints = {}
  for car_folder in [x[0] for x in os.walk(BASEDIR + '/selfdrive/car')]:
    car_name = car_folder.split('/')[-1]
    try:
      fingerprints[car_name] = __import__(f'selfdrive.car.{car_name}.values', fromlist=['FINGERPRINTS']).FINGERPRINTS
    except (ImportError, IOError, AttributeError):
      pass

  return fingerprints


def check_fingerprint_consistency(f1, f2):
  # return false if it finds a fingerprint fully included in another
  # max message worth checking is 1800, as above that they usually come too infrequently and not
  # usable for fingerprinting

  max_msg = 1800

  is_f1_in_f2 = True
  for k in f1:
    if (k not in f2 or f1[k] != f2[k]) and k < max_msg:
       is_f1_in_f2 = False

  is_f2_in_f1 = True
  for k in f2:
    if (k not in f1 or f2[k] != f1[k]) and k < max_msg:
       is_f2_in_f1 = False

  return not is_f1_in_f2 and not is_f2_in_f1


def check_can_ignition_conflicts(fingerprints, brands):
  # loops through all the fingerprints and exits if CAN ignition dedicated messages
  # are found in unexpected fingerprints

  for brand_can, msgs_can in CAN_IGNITION_MSGS.items():
    for i, f in enumerate(fingerprints):
      for msg_can in msgs_can:
        if brand_can != brands[i] and msg_can[0] in f and msg_can[1] == f[msg_can[0]]:
          print("CAN ignition dedicated msg %d with len %d found in %s fingerprints!" % (msg_can[0], msg_can[1], brands[i]))
          print("TEST FAILED")
          sys.exit(1)



if __name__ == "__main__":
  fingerprints = _get_fingerprints()

  fingerprints_flat = []
  car_names = []
  brand_names = []
  for brand in fingerprints:
    for car in fingerprints[brand]:
      fingerprints_flat += fingerprints[brand][car]
      for i in range(len(fingerprints[brand][car])):
        car_names.append(car)
        brand_names.append(brand)

  # first check if CAN ignition specific messages are unexpectedly included in other fingerprints
  check_can_ignition_conflicts(fingerprints_flat, brand_names)

  valid = True
  for idx1, f1 in enumerate(fingerprints_flat):
    for idx2, f2 in enumerate(fingerprints_flat):
      if idx1 < idx2 and not check_fingerprint_consistency(f1, f2):
        valid = False
        print(f"Those two fingerprints are inconsistent {car_names[idx1]} {car_names[idx2]}")
        print("")
        print(', '.join("%d: %d" % v for v in sorted(f1.items())))
        print("")
        print(', '.join("%d: %d" % v for v in sorted(f2.items())))
        print("")

  print(f"Found {len(fingerprints_flat)} individual fingerprints")
  if not valid or len(fingerprints_flat) == 0:
    print("TEST FAILED")
    sys.exit(1)
  else:
    print("TEST SUCESSFUL")
