from dataclasses import dataclass, fields, field, is_dataclass
from enum import Enum, StrEnum as _StrEnum, auto
# from typing import Type, TypeVar
from typing import TypeVar, TYPE_CHECKING, Any, get_type_hints, get_origin
from selfdrive.car.data_test_kinda_works_chatgpt import auto_field, apply_auto_fields

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
#
# DataclassT = TypeVar("DataclassT", bound="DataclassInstance")
#
# T = TypeVar('T', bound='Struct')

_FIELDS = '__dataclass_fields__'


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


# class Struct:
#   @classmethod
#   def new_message(cls: type[T], **kwargs: Any) -> T:
#     if not is_dataclass(cls):
#       raise TypeError(f"{cls.__name__} is not a dataclass")
#
#     init_values = {}
#     type_hints = get_type_hints(cls)
#     print(type_hints)
#     for f in fields(cls):
#       field_type = type_hints[f.name]
#       print(f.name, f.type, field_type)
#       print(issubclass(field_type, Enum))
#       if issubclass(field_type, Enum):
#         init_values[f.name] = kwargs.get(f.name, list(field_type)[0])
#         # TODO: fix this
#         # assert issubclass(init_values[f.name], type(field_type)), f"Expected {field_type} for {f.name}, got {type(init_values[f.name])}"
#       else:
#         # FIXME: typing check hack since mypy doesn't catch anything
#         init_values[f.name] = kwargs.get(f.name, field_type())
#         print('field_type', field_type, f.type)
#         # TODO: this is so bad
#         assert isinstance(init_values[f.name], get_origin(f.type) or f.type), f"Expected {field_type} for {f.name}, got {type(init_values[f.name])}"
#
#     return cls(**init_values)


@dataclass
class RadarData:
  errors: list['Error']
  points: list['RadarPoint']

  class Error(StrEnum):
    canError = auto()
    fault = auto()
    wrongConfig = auto()

  @dataclass
  class RadarPoint:
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
@apply_auto_fields
class CarParams:
  carName: str = auto_field()
  carFingerprint: str = auto_field()
  fuzzyFingerprint: bool = auto_field()

  notCar: bool = auto_field()  # flag for non-car robotics platforms

  carFw: list['CarParams.CarFw'] = auto_field()

  class SteerControlType(StrEnum):
    torque = auto()
    angle = auto()

  @dataclass
  @apply_auto_fields
  class CarFw:
    ecu: 'CarParams.Ecu' = field(default_factory=lambda: CarParams.Ecu.unknown)
    fwVersion: bytes = auto_field()
    address: int = auto_field()
    subAddress: int = auto_field()
    responseAddress: int = auto_field()
    request: list[bytes] = auto_field()
    brand: str = auto_field()
    bus: int = auto_field()
    logging: bool = auto_field()
    obdMultiplexing: bool = auto_field()

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


# # CP: CarParams = CarParams.new_message(carName='toyota', fuzzyFingerprint=123)
# # CP: CarParams = CarParams(carName='toyota', fuzzyFingerprint=123)
#
# # import ast
#
#
# # test = ast.literal_eval('CarParams.CarFw')
#
# def mywrapper(cls):
#
#   cls_annotations = cls.__dict__.get('__annotations__', {})
#   fields = {}
#   for name, _type in cls_annotations.items():
#     f = field(default_factory=_type)
#     setattr(cls, name, f)
#     fields[name] = f
#
#   setattr(cls, _FIELDS, fields)
#
#   print('cls_annotations', cls_annotations)
#   # cls.hi = 123
#
#   return cls
#
#
# # def mywrapper2(cls):
# #   class Test:
# #     pass
# #   return Test
#
#
# @dataclass
# class CarControl1:
#   enabled: bool
#
# @dataclass
# class CarControl2:
#   enabled: bool = field(default_factory=bool)
#
#
# # @mywrapper2
# @dataclass()
# @mywrapper
# class CarControl:
#   # enabled: bool = field(default_factory=bool)
#   enabled: bool = None
#   pts: list[int] = None
#   logMonoTime: int = None
#
#
# CC = CarControl()


@dataclass
@apply_auto_fields
class CarControl:
  enabled: bool = auto_field()
  pts: list[int] = auto_field()
  logMonoTime: int = auto_field()
  test: None = auto_field()


# testing: if origin_typ in (int, float, str, bytes, list, tuple, set, dict, bool):
@dataclass
@apply_auto_fields
class Test997:
  a: int = auto_field()
  b: float = auto_field()
  c: str = auto_field()
  d: bytes = auto_field()
  e: list[int] = auto_field()
  f: tuple[int] = auto_field()
  g: set[int] = auto_field()
  h: dict[str, int] = auto_field()
  i: bool = auto_field()
  ecu: CarParams.Ecu = auto_field()
  carFw: CarParams.CarFw = auto_field()

# Out[4]: Test997(a=0, b=0.0, c='', d=b'', e=[], f=(), g=set(), h={}, i=False)

CarControl()

CP = CarParams()
CP.carFw = [CarParams.CarFw()]
CP.carFw = [CarParams.Ecu.eps]
