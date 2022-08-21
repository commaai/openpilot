import os
from typing import Dict, List

from cereal import car
from common.params import Params
from common.basedir import BASEDIR
from system.version import is_comma_remote, is_tested_branch
from selfdrive.car.interfaces import get_interface_attr
from selfdrive.car.fingerprints import eliminate_incompatible_cars, all_legacy_fingerprint_cars
from selfdrive.car.vin import get_vin, is_valid_vin, VIN_UNKNOWN
from selfdrive.car.fw_versions import get_fw_versions_ordered, match_fw_to_car, get_present_ecus
from system.swaglog import cloudlog
import cereal.messaging as messaging
from selfdrive.car import gen_empty_fingerprint

EventName = car.CarEvent.EventName


def get_startup_event(car_recognized, controller_available, fw_seen):
  if is_comma_remote() and is_tested_branch():
    event = EventName.startup
  else:
    event = EventName.startupMaster

  if not car_recognized:
    if fw_seen:
      event = EventName.startupNoCar
    else:
      event = EventName.startupNoFw
  elif car_recognized and not controller_available:
    event = EventName.startupNoControl
  return event


def get_one_can(logcan):
  while True:
    can = messaging.recv_one_retry(logcan)
    if len(can.can) > 0:
      return can


def load_interfaces(brand_names):
  ret = {}
  for brand_name in brand_names:
    path = f'selfdrive.car.{brand_name}'
    CarInterface = __import__(path + '.interface', fromlist=['CarInterface']).CarInterface

    if os.path.exists(BASEDIR + '/' + path.replace('.', '/') + '/carstate.py'):
      CarState = __import__(path + '.carstate', fromlist=['CarState']).CarState
    else:
      CarState = None

    if os.path.exists(BASEDIR + '/' + path.replace('.', '/') + '/carcontroller.py'):
      CarController = __import__(path + '.carcontroller', fromlist=['CarController']).CarController
    else:
      CarController = None

    for model_name in brand_names[brand_name]:
      ret[model_name] = (CarInterface, CarController, CarState)
  return ret


def _get_interface_names() -> Dict[str, List[str]]:
  # returns a dict of brand name and its respective models
  brand_names = {}
  for brand_name, model_names in get_interface_attr("CAR").items():
    model_names = [getattr(model_names, c) for c in model_names.__dict__.keys() if not c.startswith("__")]
    brand_names[brand_name] = model_names

  return brand_names


# imports from directory selfdrive/car/<name>/
interface_names = _get_interface_names()
interfaces = load_interfaces(interface_names)


# **** for use live only ****
def fingerprint(logcan, sendcan):
  fixed_fingerprint = os.environ.get('FINGERPRINT', "")
  skip_fw_query = os.environ.get('SKIP_FW_QUERY', False)
  ecu_rx_addrs = set()

  if not fixed_fingerprint and not skip_fw_query:
    # Vin query only reliably works thorugh OBDII
    bus = 1

    cached_params = Params().get("CarParamsCache")
    if cached_params is not None:
      cached_params = car.CarParams.from_bytes(cached_params)
      if cached_params.carName == "mock":
        cached_params = None

    if cached_params is not None and len(cached_params.carFw) > 0 and cached_params.carVin is not VIN_UNKNOWN:
      cloudlog.warning("Using cached CarParams")
      vin, vin_rx_addr = cached_params.carVin, 0
      car_fw = list(cached_params.carFw)
    else:
      cloudlog.warning("Getting VIN & FW versions")
      _, vin_rx_addr, vin = get_vin(logcan, sendcan, bus)
      ecu_rx_addrs = get_present_ecus(logcan, sendcan)
      car_fw = get_fw_versions_ordered(logcan, sendcan, ecu_rx_addrs)

    exact_fw_match, fw_candidates = match_fw_to_car(car_fw)
  else:
    vin, vin_rx_addr = VIN_UNKNOWN, 0
    exact_fw_match, fw_candidates, car_fw = True, set(), []

  if not is_valid_vin(vin):
    cloudlog.event("Malformed VIN", vin=vin, error=True)
    vin = VIN_UNKNOWN
  cloudlog.warning("VIN %s", vin)
  Params().put("CarVin", vin)

  finger = gen_empty_fingerprint()
  candidate_cars = {i: all_legacy_fingerprint_cars() for i in [0, 1]}  # attempt fingerprint on both bus 0 and 1
  frame = 0
  frame_fingerprint = 100  # 1s
  car_fingerprint = None
  done = False

  # drain CAN socket so we always get the latest messages
  messaging.drain_sock_raw(logcan)

  while not done:
    a = get_one_can(logcan)

    for can in a.can:
      # The fingerprint dict is generated for all buses, this way the car interface
      # can use it to detect a (valid) multipanda setup and initialize accordingly
      if can.src < 128:
        if can.src not in finger:
          finger[can.src] = {}
        finger[can.src][can.address] = len(can.dat)

      for b in candidate_cars:
        # Ignore extended messages and VIN query response.
        if can.src == b and can.address < 0x800 and can.address not in (0x7df, 0x7e0, 0x7e8):
          candidate_cars[b] = eliminate_incompatible_cars(can, candidate_cars[b])

    # if we only have one car choice and the time since we got our first
    # message has elapsed, exit
    for b in candidate_cars:
      if len(candidate_cars[b]) == 1 and frame > frame_fingerprint:
        # fingerprint done
        car_fingerprint = candidate_cars[b][0]

    # bail if no cars left or we've been waiting for more than 2s
    failed = (all(len(cc) == 0 for cc in candidate_cars.values()) and frame > frame_fingerprint) or frame > 200
    succeeded = car_fingerprint is not None
    done = failed or succeeded

    frame += 1

  exact_match = True
  source = car.CarParams.FingerprintSource.can

  # If FW query returns exactly 1 candidate, use it
  if len(fw_candidates) == 1:
    car_fingerprint = list(fw_candidates)[0]
    source = car.CarParams.FingerprintSource.fw
    exact_match = exact_fw_match

  if fixed_fingerprint:
    car_fingerprint = fixed_fingerprint
    source = car.CarParams.FingerprintSource.fixed

  cloudlog.event("fingerprinted", car_fingerprint=car_fingerprint, source=source, fuzzy=not exact_match,
                 fw_count=len(car_fw), ecu_responses=list(ecu_rx_addrs), vin_rx_addr=vin_rx_addr, error=True)
  return car_fingerprint, finger, vin, car_fw, source, exact_match


def get_car(logcan, sendcan):
  candidate, fingerprints, vin, car_fw, source, exact_match = fingerprint(logcan, sendcan)

  if candidate is None:
    cloudlog.warning("car doesn't match any fingerprints: %r", fingerprints)
    candidate = "mock"

  disable_radar = Params().get_bool("DisableRadar")

  CarInterface, CarController, CarState = interfaces[candidate]
  CP = CarInterface.get_params(candidate, fingerprints, car_fw, disable_radar)
  CP.carVin = vin
  CP.carFw = car_fw
  CP.fingerprintSource = source
  CP.fuzzyFingerprint = not exact_match

  return CarInterface(CP, CarController, CarState), CP
