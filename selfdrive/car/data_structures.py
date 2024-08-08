from dataclasses import dataclass
from enum import StrEnum as _StrEnum, auto


class StrEnum(_StrEnum):
  @staticmethod
  def _generate_next_value_(name, *args):
    # auto() defaults to name.lower()
    return name


@dataclass
class CarParams:
  carName: str
  carFingerprint: str
  fuzzyFingerprint: bool

  notCar: bool  # flag for non-car robotics platforms

  class SteerControlType(StrEnum):
    torque = auto()
    angle = auto()

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
