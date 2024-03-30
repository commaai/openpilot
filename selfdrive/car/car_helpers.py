import os
import time
from collections.abc import Callable

from cereal import car
from openpilot.common.params import Params
from openpilot.selfdrive.car.interfaces import get_interface_attr
from openpilot.selfdrive.car.fingerprints import eliminate_incompatible_cars, all_legacy_fingerprint_cars
from openpilot.selfdrive.car.vin import get_vin, is_valid_vin, VIN_UNKNOWN
from openpilot.selfdrive.car.fw_versions import get_fw_versions_ordered, get_present_ecus, match_fw_to_car, set_obd_multiplexing
from openpilot.selfdrive.car.mock.values import CAR as MOCK
from openpilot.common.swaglog import cloudlog
import cereal.messaging as messaging
from openpilot.selfdrive.car import gen_empty_fingerprint
from openpilot.system.version import get_build_metadata

FRAME_FINGERPRINT = 100  # 1s

EventName = car.CarEvent.EventName


def get_startup_event(car_recognized, controller_available, fw_seen):
  build_metadata = get_build_metadata()
  if build_metadata.openpilot.comma_remote and build_metadata.tested_channel:
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
    path = f'openpilot.selfdrive.car.{brand_name}'
    CarInterface = __import__(path + '.interface', fromlist=['CarInterface']).CarInterface
    CarState = __import__(path + '.carstate', fromlist=['CarState']).CarState
    CarController = __import__(path + '.carcontroller', fromlist=['CarController']).CarController
    for model_name in brand_names[brand_name]:
      ret[model_name] = (CarInterface, CarController, CarState)
  return ret


def _get_interface_names() -> dict[str, list[str]]:
  # returns a dict of brand name and its respective models
  brand_names = {}
  for brand_name, brand_models in get_interface_attr("CAR").items():
    brand_names[brand_name] = [model.value for model in brand_models]

  return brand_names


# imports from directory selfdrive/car/<name>/
interface_names = _get_interface_names()
interfaces = load_interfaces(interface_names)


def can_fingerprint(next_can: Callable) -> tuple[str | None, dict[int, dict]]:
  finger = gen_empty_fingerprint()
  candidate_cars = {i: all_legacy_fingerprint_cars() for i in [0, 1]}  # attempt fingerprint on both bus 0 and 1
  frame = 0
  car_fingerprint = None
  done = False

  while not done:
    a = next_can()

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
      if len(candidate_cars[b]) == 1 and frame > FRAME_FINGERPRINT:
        # fingerprint done
        car_fingerprint = candidate_cars[b][0]

    # bail if no cars left or we've been waiting for more than 2s
    failed = (all(len(cc) == 0 for cc in candidate_cars.values()) and frame > FRAME_FINGERPRINT) or frame > 200
    succeeded = car_fingerprint is not None
    done = failed or succeeded

    frame += 1

  return car_fingerprint, finger


# **** for use live only ****
def fingerprint(logcan, sendcan, num_pandas):
  fixed_fingerprint = os.environ.get('FINGERPRINT', "")
  skip_fw_query = os.environ.get('SKIP_FW_QUERY', False)
  disable_fw_cache = os.environ.get('DISABLE_FW_CACHE', False)
  ecu_rx_addrs = set()
  params = Params()

  start_time = time.monotonic()
  if not skip_fw_query:
    cached_params = params.get("CarParamsCache")
    if cached_params is not None:
      with car.CarParams.from_bytes(cached_params) as cached_params:
        if cached_params.carName == "mock":
          cached_params = None

    if cached_params is not None and len(cached_params.carFw) > 0 and \
       cached_params.carVin is not VIN_UNKNOWN and not disable_fw_cache:
      cloudlog.warning("Using cached CarParams")
      vin_rx_addr, vin_rx_bus, vin = -1, -1, cached_params.carVin
      car_fw = list(cached_params.carFw)
      cached = True
    else:
      cloudlog.warning("Getting VIN & FW versions")
      # enable OBD multiplexing for VIN query
      # NOTE: this takes ~0.1s and is relied on to allow sendcan subscriber to connect in time
      set_obd_multiplexing(params, True)
      # VIN query only reliably works through OBDII
      vin_rx_addr, vin_rx_bus, vin = get_vin(logcan, sendcan, (0, 1))
      ecu_rx_addrs = get_present_ecus(logcan, sendcan, num_pandas=num_pandas)
      car_fw = get_fw_versions_ordered(logcan, sendcan, ecu_rx_addrs, num_pandas=num_pandas)
      cached = False

    exact_fw_match, fw_candidates = match_fw_to_car(car_fw)
  else:
    vin_rx_addr, vin_rx_bus, vin = -1, -1, VIN_UNKNOWN
    exact_fw_match, fw_candidates, car_fw = True, set(), []
    cached = False

  if not is_valid_vin(vin):
    cloudlog.event("Malformed VIN", vin=vin, error=True)
    vin = VIN_UNKNOWN
  cloudlog.warning("VIN %s", vin)
  params.put("CarVin", vin)

  # disable OBD multiplexing for CAN fingerprinting and potential ECU knockouts
  set_obd_multiplexing(params, False)
  params.put_bool("FirmwareQueryDone", True)

  fw_query_time = time.monotonic() - start_time

  # CAN fingerprint
  # drain CAN socket so we get the latest messages
  messaging.drain_sock_raw(logcan)
  car_fingerprint, finger = can_fingerprint(lambda: get_one_can(logcan))

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

  cloudlog.event("fingerprinted", car_fingerprint=car_fingerprint, source=source, fuzzy=not exact_match, cached=cached,
                 fw_count=len(car_fw), ecu_responses=list(ecu_rx_addrs), vin_rx_addr=vin_rx_addr, vin_rx_bus=vin_rx_bus,
                 fingerprints=repr(finger), fw_query_time=fw_query_time, error=True)

  return car_fingerprint, finger, vin, car_fw, source, exact_match


def get_car_interface(CP):
  CarInterface, CarController, CarState = interfaces[CP.carFingerprint]
  return CarInterface(CP, CarController, CarState)


def get_car(logcan, sendcan, experimental_long_allowed, num_pandas=1):
  candidate, fingerprints, vin, car_fw, source, exact_match = fingerprint(logcan, sendcan, num_pandas)

  if candidate is None:
    cloudlog.event("car doesn't match any fingerprints", fingerprints=repr(fingerprints), error=True)
    candidate = "MOCK"

  CarInterface, _, _ = interfaces[candidate]
  CP = CarInterface.get_params(candidate, fingerprints, car_fw, experimental_long_allowed, docs=False)
  CP.carVin = vin
  CP.carFw = car_fw
  CP.fingerprintSource = source
  CP.fuzzyFingerprint = not exact_match

  return get_car_interface(CP), CP

def write_car_param(platform=MOCK.MOCK):
  params = Params()
  CarInterface, _, _ = interfaces[platform]
  CP = CarInterface.get_non_essential_params(platform)
  params.put("CarParams", CP.to_bytes())

def get_demo_car_params():
  platform = MOCK.MOCK
  CarInterface, _, _ = interfaces[platform]
  CP = CarInterface.get_non_essential_params(platform)
  return CP
