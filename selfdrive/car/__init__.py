from common.fingerprints import fingerprint

from .honda.interface import CarInterface as HondaInterface

try:
  from .toyota.interface import CarInterface as ToyotaInterface
except ImportError:
  ToyotaInterface = None

try:
  from .simulator.interface import CarInterface as SimInterface
except ImportError:
  SimInterface = None

try:
  from .simulator2.interface import CarInterface as Sim2Interface
except ImportError:
  Sim2Interface = None

interfaces = {
  "HONDA CIVIC 2016 TOURING": HondaInterface,
  "ACURA ILX 2016 ACURAWATCH PLUS": HondaInterface,
  "HONDA ACCORD 2016 TOURING": HondaInterface,
  "HONDA CR-V 2016 TOURING": HondaInterface,
  "TOYOTA PRIUS 2017": ToyotaInterface,

  "simulator": SimInterface,
  "simulator2": Sim2Interface
}

def get_car(logcan, sendcan=None):
  candidate, fingerprints = fingerprint(logcan)
  interface_cls = interfaces[candidate]
  params = interface_cls.get_params(candidate, fingerprints)

  return interface_cls(params, logcan, sendcan), params
