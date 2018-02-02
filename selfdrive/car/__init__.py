import os
from cereal import car

from common.realtime import sec_since_boot
from common.fingerprints import eliminate_incompatible_cars, all_known_cars

from selfdrive.swaglog import cloudlog
import selfdrive.messaging as messaging
from selfdrive.car.honda.interface import CarInterface as HondaInterface
from selfdrive.car.toyota.interface import CarInterface as ToyotaInterface
from selfdrive.car.mock.interface import CarInterface as MockInterface
from common.fingerprints import HONDA, TOYOTA

try:
  from .simulator.interface import CarInterface as SimInterface
except ImportError:
  SimInterface = None

try:
  from .simulator2.interface import CarInterface as Sim2Interface
except ImportError:
  Sim2Interface = None


interfaces = {
  HONDA.CIVIC: HondaInterface,
  HONDA.ACURA_ILX: HondaInterface,
  HONDA.CRV: HondaInterface,
  HONDA.ODYSSEY: HondaInterface,

  TOYOTA.PRIUS: ToyotaInterface,
  TOYOTA.RAV4: ToyotaInterface,
  TOYOTA.RAV4H: ToyotaInterface,
  TOYOTA.COROLLA: ToyotaInterface,

  "simulator2": Sim2Interface,
  "mock": MockInterface
}

# **** for use live only ****
def fingerprint(logcan, timeout):
  if os.getenv("SIMULATOR2") is not None:
    return ("simulator2", None)

  finger_st = sec_since_boot()

  cloudlog.warning("waiting for fingerprint...")
  candidate_cars = all_known_cars()
  finger = {}
  st = None
  while 1:
    for a in messaging.drain_sock(logcan, wait_for_one=True):
      if st is None:
        st = sec_since_boot()
      for can in a.can:
        if can.src == 0:
          finger[can.address] = len(can.dat)
        candidate_cars = eliminate_incompatible_cars(can, candidate_cars)

    ts = sec_since_boot()
    # if we only have one car choice and the time_fingerprint since we got our first
    # message has elapsed, exit. Toyota needs higher time_fingerprint, since DSU does not
    # broadcast immediately
    if len(candidate_cars) == 1 and st is not None:
      time_fingerprint = 1.0 if "TOYOTA" in candidate_cars[0] else 0.1
      if (ts-st) > time_fingerprint:
        break

    # bail if no cars left or we've been waiting too long
    elif len(candidate_cars) == 0 or (timeout and ts-finger_st > timeout):
      return None, finger

  cloudlog.warning("fingerprinted %s", candidate_cars[0])
  return (candidate_cars[0], finger)


def get_car(logcan, sendcan=None, passive=True):
  # TODO: timeout only useful for replays so controlsd can start before unlogger
  timeout = 1. if passive else None
  candidate, fingerprints = fingerprint(logcan, timeout)

  if candidate is None:
    cloudlog.warning("car doesn't match any fingerprints: %r", fingerprints)
    if passive:
      candidate = "mock"
    else:
      return None, None

  interface_cls = interfaces[candidate]
  params = interface_cls.get_params(candidate, fingerprints)

  return interface_cls(params, sendcan), params
