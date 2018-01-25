from selfdrive.car.honda.interface import CarInterface as HondaInterface
from selfdrive.car.toyota.interface import CarInterface as ToyotaInterface
from selfdrive.car.mock.interface import CarInterface as MockInterface
from selfdrive.car.toyota.values import CAR as TOYOTA

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
  "HONDA ODYSSEY 2018 EX-L": HondaInterface,
  TOYOTA.PRIUS: ToyotaInterface,
  TOYOTA.PRIUSP: ToyotaInterface,
  TOYOTA.RAV4: ToyotaInterface,
  TOYOTA.RAV4H: ToyotaInterface,

  "simulator": SimInterface,
  "simulator2": Sim2Interface,

  "mock": MockInterface
}

