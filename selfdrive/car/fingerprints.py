import os
from common.basedir import BASEDIR


def get_attr_from_cars(attr, result=dict, combine_brands=True):
  # read all the folders in selfdrive/car and return a dict where:
  # - keys are all the car models
  # - values are attr values from all car folders
  result = result()

  for car_folder in [x[0] for x in os.walk(BASEDIR + '/selfdrive/car')]:
    try:
      car_name = car_folder.split('/')[-1]
      values = __import__(f'selfdrive.car.{car_name}.values', fromlist=[attr])
      if hasattr(values, attr):
        attr_values = getattr(values, attr)
      else:
        continue

      if isinstance(attr_values, dict):
        for f, v in attr_values.items():
          if combine_brands:
            result[f] = v
          else:
            if car_name not in result:
              result[car_name] = {}
            result[car_name][f] = v
      elif isinstance(attr_values, list):
        result += attr_values

    except (ImportError, IOError):
      pass

  return result


FW_VERSIONS = get_attr_from_cars('FW_VERSIONS')
_FINGERPRINTS = get_attr_from_cars('FINGERPRINTS')

_DEBUG_ADDRESS = {1880: 8}   # reserved for debug purposes

def is_valid_for_fingerprint(msg, car_fingerprint):
  adr = msg.address
  # ignore addresses that are more than 11 bits
  return (adr in car_fingerprint and car_fingerprint[adr] == len(msg.dat)) or adr >= 0x800


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
    car_fingerprints = _FINGERPRINTS[car_name]

    for fingerprint in car_fingerprints:
      fingerprint.update(_DEBUG_ADDRESS)  # add alien debug address

      if is_valid_for_fingerprint(msg, fingerprint):
        compatible_cars.append(car_name)
        break

  return compatible_cars


def all_known_cars():
  """Returns a list of all known car strings."""
  return list({*FW_VERSIONS.keys(), *_FINGERPRINTS.keys()})


def all_legacy_fingerprint_cars():
  """Returns a list of all known car strings, FPv1 only."""
  return list(_FINGERPRINTS.keys())
