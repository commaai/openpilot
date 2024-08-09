from dataclasses import dataclass as _dataclass, field, is_dataclass
from enum import Enum, StrEnum as _StrEnum, auto
from typing import get_origin


auto_obj = object()


def auto_field():
  return auto_obj


def auto_dataclass(cls=None, /, **kwargs):
  cls_annotations = cls.__dict__.get('__annotations__', {})
  for name, typ in cls_annotations.items():
    current_value = getattr(cls, name, None)
    if current_value is auto_obj:
      origin_typ = get_origin(typ) or typ
      if isinstance(origin_typ, str):
        raise TypeError(f"Forward references are not supported for auto_field: '{origin_typ}'. Use a default_factory with lambda instead.")
      elif origin_typ in (int, float, str, bytes, list, tuple, set, dict, bool) or is_dataclass(origin_typ):
        setattr(cls, name, field(default_factory=origin_typ))
      elif origin_typ is None:
        setattr(cls, name, field(default=origin_typ))
      elif issubclass(origin_typ, Enum):  # first enum is the default
        setattr(cls, name, field(default=next(iter(origin_typ))))
      else:
        raise TypeError(f"Unsupported type for auto_field: {origin_typ}")

  return _dataclass(cls, **kwargs)


class StrEnum(_StrEnum):
  @staticmethod
  def _generate_next_value_(name, *args):
    # auto() defaults to name.lower()
    return name


@auto_dataclass
class RadarData:
  errors: list['Error'] = auto_field()
  points: list['RadarPoint'] = auto_field()

  class Error(StrEnum):
    canError = auto()
    fault = auto()
    wrongConfig = auto()

  @auto_dataclass
  class RadarPoint:
    trackId: int = auto_field()  # no trackId reuse

    # these 3 are the minimum required
    dRel: float = auto_field()  # m from the front bumper of the car
    yRel: float = auto_field()  # m
    vRel: float = auto_field()  # m/s

    # these are optional and valid if they are not NaN
    aRel: float = auto_field()  # m/s^2
    yvRel: float = auto_field()  # m/s

    # some radars flag measurements VS estimates
    measured: bool = auto_field()


@auto_dataclass
class CarParams:
  carName: str = auto_field()
  carFingerprint: str = auto_field()
  fuzzyFingerprint: bool = auto_field()

  notCar: bool = auto_field()  # flag for non-car robotics platforms

  carFw: list['CarParams.CarFw'] = auto_field()

  class SteerControlType(StrEnum):
    torque = auto()
    angle = auto()

  @auto_dataclass
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


@auto_dataclass
class CarControl:
  enabled: bool = auto_field()
  pts: list[int] = auto_field()
  logMonoTime: int = auto_field()
  test: None = auto_field()


# testing: if origin_typ in (int, float, str, bytes, list, tuple, set, dict, bool):
@auto_dataclass
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
# CP.carFw = [CarParams.Ecu.eps]
