import os
from cereal import car
from common.params import Params
from common.vin import get_vin, VIN_UNKNOWN
from common.basedir import BASEDIR
from common.fingerprints import eliminate_incompatible_cars, all_known_cars
from selfdrive.swaglog import cloudlog
import selfdrive.messaging as messaging


def get_startup_alert(car_recognized, controller_available):
  alert = 'startup'
  if not car_recognized:
    alert = 'startupNoCar'
  elif car_recognized and not controller_available:
    alert = 'startupNoControl'
  return alert


def load_interfaces(brand_names):
  ret = {}
  for brand_name in brand_names:
    path = ('selfdrive.car.%s' % brand_name)
    CarInterface = __import__(path + '.interface', fromlist=['CarInterface']).CarInterface
    if os.path.exists(BASEDIR + '/' + path.replace('.', '/') + '/carcontroller.py'):
      CarController = __import__(path + '.carcontroller', fromlist=['CarController']).CarController
    else:
      CarController = None
    for model_name in brand_names[brand_name]:
      ret[model_name] = (CarInterface, CarController)
  return ret


def _get_interface_names():
  # read all the folders in selfdrive/car and return a dict where:
  # - keys are all the car names that which we have an interface for
  # - values are lists of spefic car models for a given car
  brand_names = {}
  for car_folder in [x[0] for x in os.walk(BASEDIR + '/selfdrive/car')]:
    try:
      brand_name = car_folder.split('/')[-1]
      model_names = __import__('selfdrive.car.%s.values' % brand_name, fromlist=['CAR']).CAR
      model_names = [getattr(model_names, c) for c in model_names.__dict__.keys() if not c.startswith("__")]
      brand_names[brand_name] = model_names
    except (ImportError, IOError):
      pass

  return brand_names


# imports from directory selfdrive/car/<name>/
interfaces = load_interfaces(_get_interface_names())

def only_toyota_left(candidate_cars):
  return all(("TOYOTA" in c or "LEXUS" in c) for c in candidate_cars) and len(candidate_cars) > 0

# BOUNTY: every added fingerprint in selfdrive/car/*/values.py is a $100 coupon code on shop.comma.ai
# **** for use live only ****
def fingerprint(logcan, sendcan, is_panda_black):
  if os.getenv("SIMULATOR2") is not None:
    return ("simulator2", None, "")
  elif os.getenv("SIMULATOR") is not None:
    return ("simulator", None, "")

  params = Params()
  car_params = params.get("CarParams")

  if car_params is not None:
    # use already stored VIN: a new VIN query cannot be done, since panda isn't in ELM327 mode
    car_params = car.CarParams.from_bytes(car_params)
    vin = VIN_UNKNOWN if car_params.carVin == "" else car_params.carVin
  elif is_panda_black:
    # Vin query only reliably works thorugh OBDII
    vin = get_vin(logcan, sendcan, 1)
  else:
    vin = VIN_UNKNOWN

  cloudlog.warning("VIN %s", vin)
  Params().put("CarVin", vin)

  finger = {i: {} for i in range(0, 4)}  # collect on all buses
  candidate_cars = {i: all_known_cars() for i in [0, 1]}  # attempt fingerprint on both bus 0 and 1
  frame = 0
  frame_fingerprint = 10  # 0.1s
  car_fingerprint = None
  done = False

  while not done:
    a = messaging.recv_one(logcan)

    for can in a.can:
      # need to independently try to fingerprint both bus 0 and 1 to work
      # for the combo black_panda and honda_bosch. Ignore extended messages
      # and VIN query response.
      # Include bus 2 for toyotas to disambiguate cars using camera messages
      # (ideally should be done for all cars but we can't for Honda Bosch)
      for b in candidate_cars:
        if (can.src == b or (only_toyota_left(candidate_cars[b]) and can.src == 2)) and \
           can.address < 0x800 and can.address not in [0x7df, 0x7e0, 0x7e8]:
          finger[can.src][can.address] = len(can.dat)
          candidate_cars[b] = eliminate_incompatible_cars(can, candidate_cars[b])

    # if we only have one car choice and the time since we got our first
    # message has elapsed, exit
    for b in candidate_cars:
      # Toyota needs higher time to fingerprint, since DSU does not broadcast immediately
      if only_toyota_left(candidate_cars[b]):
        frame_fingerprint = 100  # 1s
      if len(candidate_cars[b]) == 1:
        if frame > frame_fingerprint:
          # fingerprint done
          car_fingerprint = candidate_cars[b][0]

    # bail if no cars left or we've been waiting for more than 2s
    failed = all(len(cc) == 0 for cc in candidate_cars.itervalues()) or frame > 200
    succeeded = car_fingerprint is not None
    done = failed or succeeded

    frame += 1

  cloudlog.warning("fingerprinted %s", car_fingerprint)
  return car_fingerprint, finger, vin


def get_car(logcan, sendcan, is_panda_black=False):

  candidate, fingerprints, vin = fingerprint(logcan, sendcan, is_panda_black)

  if candidate is None:
    cloudlog.warning("car doesn't match any fingerprints: %r", fingerprints)
    candidate = "mock"

  CarInterface, CarController = interfaces[candidate]
  car_params = CarInterface.get_params(candidate, fingerprints[0], vin, is_panda_black)

  return CarInterface(car_params, CarController), car_params
