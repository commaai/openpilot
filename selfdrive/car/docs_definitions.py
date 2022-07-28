import re

from cereal import car
from common.conversions import Conversions as CV
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, no_type_check

GOOD_TORQUE_THRESHOLD = 1.0  # m/s^2
MODEL_YEARS_RE = r"(?<= )((\d{4}-\d{2})|(\d{4}))(,|$)"


class Tier(Enum):
  GOLD = 0
  SILVER = 1
  BRONZE = 2


class Column(Enum):
  MAKE = "Make"
  MODEL = "Model"
  PACKAGE = "Supported Package"
  LONGITUDINAL = "openpilot ACC"
  FSR_LONGITUDINAL = "Stop and Go"
  FSR_STEERING = "Steer to 0"
  STEERING_TORQUE = "Steering Torque"


class Star(Enum):
  FULL = "full"
  HALF = "half"
  EMPTY = "empty"


StarColumns = list(Column)[3:]
TierColumns = (Column.FSR_LONGITUDINAL, Column.FSR_STEERING, Column.STEERING_TORQUE)
CarFootnote = namedtuple("CarFootnote", ["text", "column", "star"], defaults=[None])
CarPackage = namedtuple("CarPackage", ["short", "full"])


def get_footnotes(footnotes: List[Enum], column: Column) -> List[Enum]:
  # Returns applicable footnotes given current column
  return [fn for fn in footnotes if fn.value.column == column]


# TODO: clean this up or just make cars store list of years
def get_year_list(years):
  years_list = []
  if len(years) == 0:
    return years_list

  for year in years.split(','):
    year = year.strip()
    if len(year) == 4:
      years_list.append(str(year))
    elif "-" in year:
      start, end = year.split("-")
      years_list.extend(map(str, range(int(start), int(f"20{end}"))))
    else:
      raise Exception(f"Malformed year string: {years}")
  return years_list


def split_name(name: str) -> Tuple[str, str, str]:
  make, model = name.split(" ", 1)
  years = ""
  match = re.search(MODEL_YEARS_RE, model)
  if match is not None:
    years = model[match.start():]
    model = model[:match.start() - 1]
  return make, model, years


@dataclass
class CarInfo:
  name: str
  package: Enum
  video_link: Optional[str] = None
  footnotes: List[Enum] = field(default_factory=list)
  min_steer_speed: Optional[float] = None
  min_enable_speed: Optional[float] = None
  harness: Optional[Enum] = None

  def init(self, CP: car.CarParams, all_footnotes: Dict[Enum, int]):
    # TODO: set all the min steer speeds in carParams and remove this
    min_steer_speed = CP.minSteerSpeed
    if self.min_steer_speed is not None:
      min_steer_speed = self.min_steer_speed
      assert CP.minSteerSpeed == 0, f"{CP.carFingerprint}: Minimum steer speed set in both CarInfo and CarParams"

    assert self.harness is not None, f"{CP.carFingerprint}: Need to specify car harness"

    # TODO: set all the min enable speeds in carParams correctly and remove this
    min_enable_speed = CP.minEnableSpeed
    if self.min_enable_speed is not None:
      min_enable_speed = self.min_enable_speed

    self.car_name = CP.carName
    self.car_fingerprint = CP.carFingerprint
    self.make, self.model, self.years = split_name(self.name)
    self.row = {
      Column.MAKE: self.make,
      Column.MODEL: self.model,
      Column.PACKAGE: self.package,
      # StarColumns
      Column.LONGITUDINAL: Star.FULL if CP.openpilotLongitudinalControl and not CP.radarOffCan else Star.EMPTY,
      Column.FSR_LONGITUDINAL: Star.FULL if min_enable_speed <= 0. else Star.EMPTY,
      Column.FSR_STEERING: Star.FULL if min_steer_speed <= 0. else Star.EMPTY,
      Column.STEERING_TORQUE: Star.EMPTY,
    }

    # Set steering torque star from max lateral acceleration
    assert CP.maxLateralAccel > 0.1
    if CP.maxLateralAccel >= GOOD_TORQUE_THRESHOLD:
      self.row[Column.STEERING_TORQUE] = Star.FULL

    if CP.notCar:
      for col in StarColumns:
        self.row[col] = Star.FULL

    self.all_footnotes = all_footnotes
    for column in StarColumns:
      # Demote if footnote specifies a star
      for fn in get_footnotes(self.footnotes, column):
        if fn.value.star is not None:
          self.row[column] = fn.value.star

    # openpilot ACC star doesn't count for tiers
    full_stars = [s for col, s in self.row.items() if col in TierColumns].count(Star.FULL)
    if full_stars == len(TierColumns):
      self.tier = Tier.GOLD
    elif full_stars == len(TierColumns) - 1:
      self.tier = Tier.SILVER
    else:
      self.tier = Tier.BRONZE

    self.year_list = get_year_list(self.years)
    self.detail_sentence = self.get_detail_sentence(CP)

    return self

  def get_detail_sentence(self, CP):
    sentence_builder = "openpilot upgrades your {car_model} with hands-free lane centering {steer_speed}, and adaptive cruise control {acc_speed}."
    if CP.minSteerSpeed == 0:
      steer_speed = "at all speeds"
    else:
      steer_speed = f"above {CP.minSteerSpeed * CV.MS_TO_MPH:.0f} mph"

    if CP.minEnableSpeed == -1:
      acc_speed = "that automatically resumes from a stop"
    else:
      acc_speed = f"while driving above {CP.minEnableSpeed * CV.MS_TO_MPH:.0f} mph"

    if self.row[Column.STEERING_TORQUE] != Star.FULL:
      sentence_builder += " This car may not be able to take tight turns on its own."

    return sentence_builder.format(car_model=self.name, steer_speed=steer_speed, acc_speed=acc_speed)

  @no_type_check
  def get_column(self, column: Column, star_icon: str, footnote_tag: str, add_years: bool = True) -> str:
    item: Union[str, Star] = self.row[column]
    if column in StarColumns:
      item = star_icon.format(item.value)
    elif column == Column.MODEL and len(self.years) and add_years:
      item += f" {self.years}"
    elif column == Column.PACKAGE:
      item = item.value.short

    footnotes = get_footnotes(self.footnotes, column)
    if len(footnotes):
      sups = sorted([self.all_footnotes[fn] for fn in footnotes])
      item += footnote_tag.format(f'{",".join(map(str, sups))}')

    return item


class Harness(Enum):
  nidec = "Honda Nidec"
  bosch_a = "Honda Bosch A"
  bosch_b = "Honda Bosch B"
  toyota = "Toyota"
  subaru = "Subaru"
  fca = "FCA"
  vw = "VW"
  j533 = "J533"
  hyundai_a = "Hyundai A"
  hyundai_b = "Hyundai B"
  hyundai_c = "Hyundai C"
  hyundai_d = "Hyundai D"
  hyundai_e = "Hyundai E"
  hyundai_f = "Hyundai F"
  hyundai_g = "Hyundai G"
  hyundai_h = "Hyundai H"
  hyundai_i = "Hyundai I"
  hyundai_j = "Hyundai J"
  hyundai_k = "Hyundai K"
  hyundai_l = "Hyundai L"
  hyundai_m = "Hyundai M"
  hyundai_n = "Hyundai N"
  hyundai_o = "Hyundai O"
  hyundai_p = "Hyundai P"
  custom = "Developer"
  obd_ii = "OBD-II"
  nissan_a = "Nissan A"
  nissan_b = "Nissan B"
  mazda = "Mazda"
  none = "None"


class Package(Enum):
  all = CarPackage("All", "All")
  na = CarPackage("NA", "NA")

  toyota_tssp = CarPackage("TSS-P", "Toyota Safety Sense P")
  toyota_lss = CarPackage("LSS", "Lexus Safety System+")

  hyundai_scc_lkas = CarPackage("SCC + LKAS", "Smart Cruise Control (SCC) & LKAS")
  hyundai_scc_lfa = CarPackage("SCC + LFA", "Smart Cruise Control (SCC) & LFA")
  hyundai_scc = CarPackage("SCC", "Smart Cruise Control (SCC)")

  chrysler_acc = CarPackage("Adaptive Cruise", "Adaptive Cruise Control with Stop and Go")

  gm_acc = CarPackage("Adaptive Cruise", "Adaptive Cruise Control")
  gm_acc_lkas = CarPackage("ACC + LKAS", "Adaptive Cruise Control (ACC) & LKAS")

  vw_da = CarPackage("Driver Assistance", "Driver Assistance")
  vw_acc_la = CarPackage("ACC + Lane Assist", "ACC + Lane Assist")

  subaru_es = CarPackage("EyeSight", "EyeSight Driver Assist Technology")

  nissan_propilot = CarPackage("ProPILOT", "ProPILOT Assist")

  honda_sensing = CarPackage("Honda Sensing", "Honda Sensing")
  honda_acurawatch = CarPackage("AcuraWatch Plus", "AcuraWatch Plus")
  honda_touring = CarPackage("Touring", "Touring")


STAR_DESCRIPTIONS = {
  "Gas & Brakes": {  # icon and row name
    Column.FSR_LONGITUDINAL: {
      Star.FULL: "openpilot operates down to 0 mph.",
      Star.EMPTY: "openpilot operates only above a minimum speed. See your car's manual for the minimum speed.",
    },
  },
  "Steering": {
    Column.FSR_STEERING: {
      Star.FULL: "openpilot can control the steering wheel down to 0 mph.",
      Star.EMPTY: "No steering control below certain speeds. See your car's manual for the minimum speed.",
    },
    Column.STEERING_TORQUE: {
      Star.FULL: "Car has enough steering torque to comfortably take most highway turns.",
      Star.EMPTY: "Limited ability to make tighter turns.",
    },
  },
}
