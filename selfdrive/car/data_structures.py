from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum as _StrEnum, auto
# from typing import Type, TypeVar
from typing import Type, TypeVar, TYPE_CHECKING, Any, get_type_hints, get_origin

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
#
# DataclassT = TypeVar("DataclassT", bound="DataclassInstance")
#
# T = TypeVar('T', bound='Struct')


class StrEnum(_StrEnum):
  @staticmethod
  def _generate_next_value_(name, *args):
    # auto() defaults to name.lower()
    return name


# class Struct:
#   @classmethod
#   def new_message(cls, **kwargs):
#     init_values = {}
#     for f in fields(cls):
#       init_values[f.name] = kwargs.get(f.name, f.type())
#
#     return cls(**init_values)

T = TypeVar('T', bound='DataclassInstance')


class Struct:
  @classmethod
  def new_message(cls: Type[T], **kwargs: Any) -> T:
    if not is_dataclass(cls):
      raise TypeError(f"{cls.__name__} is not a dataclass")

    init_values = {}
    type_hints = get_type_hints(cls)
    print(type_hints)
    for f in fields(cls):
      field_type = type_hints[f.name]
      print(f.name, f.type, field_type)
      print(issubclass(field_type, Enum))
      if issubclass(field_type, Enum):
        init_values[f.name] = kwargs.get(f.name, list(field_type)[0])
        # TODO: fix this
        # assert issubclass(init_values[f.name], type(field_type)), f"Expected {field_type} for {f.name}, got {type(init_values[f.name])}"
      else:
        # FIXME: typing check hack since mypy doesn't catch anything
        init_values[f.name] = kwargs.get(f.name, field_type())
        print('field_type', field_type, f.type)
        # TODO: this is so bad
        assert isinstance(init_values[f.name], get_origin(f.type) or f.type), f"Expected {field_type} for {f.name}, got {type(init_values[f.name])}"

    return cls(**init_values)


@dataclass
class RadarData(Struct):
  errors: list['Error']
  points: list['RadarPoint']

  class Error(StrEnum):
    canError = auto()
    fault = auto()
    wrongConfig = auto()

  @dataclass
  class RadarPoint(Struct):
    trackId: int  # no trackId reuse

    # these 3 are the minimum required
    dRel: float  # m from the front bumper of the car
    yRel: float  # m
    vRel: float  # m/s

    # these are optional and valid if they are not NaN
    aRel: float  # m/s^2
    yvRel: float  # m/s

    # some radars flag measurements VS estimates
    measured: bool


@dataclass
class CarParams(Struct):
  carName: str
  carFingerprint: str
  fuzzyFingerprint: bool

  notCar: bool  # flag for non-car robotics platforms

  carFw: list['CarFw']

  class SteerControlType(StrEnum):
    torque = auto()
    angle = auto()

  @dataclass
  class CarFw(Struct):
    ecu: 'CarParams.Ecu'
    fwVersion: bytes
    address: int
    subAddress: int
    responseAddress: int
    request: list[bytes]
    brand: str
    bus: int
    logging: bool
    obdMultiplexing: bool

  class Ecu(StrEnum):
    eps = auto()
    abs = auto()
    fwdRadar = auto()
    fwdCamera = auto()
    engine = auto()
    unknown = auto()
    transmission = auto()  # Transmission Control Module
    hybrid = auto()  # hybrid control unit, e.g. Chrysler's HCP, Honda's IMA Control Unit, Toyota's hybrid control computer
    srs = auto()  # airbag
    gateway = auto()  # can gateway
    hud = auto()  # heads up display
    combinationMeter = auto()  # instrument cluster
    electricBrakeBooster = auto()
    shiftByWire = auto()
    adas = auto()
    cornerRadar = auto()
    hvac = auto()
    parkingAdas = auto()  # parking assist system ECU, e.g. Toyota's IPAS, Hyundai's RSPA, etc.
    epb = auto()  # electronic parking brake
    telematics = auto()
    body = auto()  # body control module

    # Toyota only
    dsu = auto()

    # Honda only
    vsa = auto()  # Vehicle Stability Assist
    programmedFuelInjection = auto()

    debug = auto()


# CP: CarParams = CarParams.new_message(carName='toyota', fuzzyFingerprint=123)
# CP: CarParams = CarParams(carName='toyota', fuzzyFingerprint=123)

import ast


# test = ast.literal_eval('CarParams.CarFw')
