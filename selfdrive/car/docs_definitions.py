import re
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from cereal import car
from common.conversions import Conversions as CV

GOOD_TORQUE_THRESHOLD = 1.0  # m/s^2
MODEL_YEARS_RE = r"(?<= )((\d{4}-\d{2})|(\d{4}))(,|$)"


class Column(Enum):
  MAKE = "Make"
  MODEL = "Model"
  PACKAGE = "Supported Package"
  LONGITUDINAL = "ACC"
  FSR_LONGITUDINAL = "No ACC accel below"
  FSR_STEERING = "No ALC below"
  STEERING_TORQUE = "Steering Torque"
  AUTO_RESUME = "Resume from stop"
  HARNESS = "Harness"


class Star(Enum):
  FULL = "full"
  HALF = "half"
  EMPTY = "empty"


class Harness(Enum):
  nidec = "Honda Nidec"
  bosch_a = "Honda Bosch A"
  bosch_b = "Honda Bosch B"
  toyota = "Toyota"
  subaru_a = "Subaru A"
  subaru_b = "Subaru B"
  fca = "FCA"
  ram = "Ram"
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
  hyundai_q = "Hyundai Q"
  custom = "Developer"
  obd_ii = "OBD-II"
  gm = "GM"
  nissan_a = "Nissan A"
  nissan_b = "Nissan B"
  mazda = "Mazda"
  ford_q3 = "Ford Q3"
  ford_q4 = "Ford Q4"
  none = "None"


CarFootnote = namedtuple("CarFootnote", ["text", "column", "docs_only", "shop_footnote"], defaults=(False, False))


class CommonFootnote(Enum):
  EXP_LONG_AVAIL = CarFootnote(
    "Experimental openpilot longitudinal control is available behind a toggle; the toggle is only available in non-release branches such as `devel` or `master-ci`. ",
    Column.LONGITUDINAL, docs_only=True)
  EXP_LONG_DSU = CarFootnote(
    "By default, this car will use the stock Adaptive Cruise Control (ACC) for longitudinal control. If the Driver Support Unit (DSU) is disconnected, openpilot ACC will replace " +
    "stock ACC. <b><i>NOTE: disconnecting the DSU disables Automatic Emergency Braking (AEB).</i></b>",
    Column.LONGITUDINAL)


def get_footnotes(footnotes: List[Enum], column: Column) -> List[Enum]:
  # Returns applicable footnotes given current column
  return [fn for fn in footnotes if fn.value.column == column]


# TODO: store years as a list
def get_year_list(years):
  years_list = []
  if len(years) == 0:
    return years_list

  for year in years.split(','):
    year = year.strip()
    if len(year) == 4:
      years_list.append(str(year))
    elif "-" in year and len(year) == 7:
      start, end = year.split("-")
      years_list.extend(map(str, range(int(start), int(f"20{end}") + 1)))
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
  package: str
  video_link: Optional[str] = None
  footnotes: List[Enum] = field(default_factory=list)
  min_steer_speed: Optional[float] = None
  min_enable_speed: Optional[float] = None
  harness: Enum = Harness.none

  def init(self, CP: car.CarParams, all_footnotes: Dict[Enum, int]):
    # TODO: set all the min steer speeds in carParams and remove this
    if self.min_steer_speed is not None:
      assert CP.minSteerSpeed == 0, f"{CP.carFingerprint}: Minimum steer speed set in both CarInfo and CarParams"
    else:
      self.min_steer_speed = CP.minSteerSpeed

    # TODO: set all the min enable speeds in carParams correctly and remove this
    if self.min_enable_speed is None:
      self.min_enable_speed = CP.minEnableSpeed

    self.car_name = CP.carName
    self.car_fingerprint = CP.carFingerprint
    self.make, self.model, self.years = split_name(self.name)

    op_long = "Stock"
    if CP.openpilotLongitudinalControl and not CP.enableDsu:
      op_long = "openpilot"
    elif CP.experimentalLongitudinalAvailable or CP.enableDsu:
      op_long = "openpilot available"
      if CP.enableDsu:
        self.footnotes.append(CommonFootnote.EXP_LONG_DSU)
      else:
        self.footnotes.append(CommonFootnote.EXP_LONG_AVAIL)

    self.row = {
      Column.MAKE: self.make,
      Column.MODEL: self.model,
      Column.PACKAGE: self.package,
      Column.LONGITUDINAL: op_long,
      Column.FSR_LONGITUDINAL: f"{max(self.min_enable_speed * CV.MS_TO_MPH, 0):.0f} mph",
      Column.FSR_STEERING: f"{max(self.min_steer_speed * CV.MS_TO_MPH, 0):.0f} mph",
      Column.STEERING_TORQUE: Star.EMPTY,
      Column.AUTO_RESUME: Star.FULL if CP.autoResumeSng else Star.EMPTY,
      Column.HARNESS: self.harness.value,
    }

    # Set steering torque star from max lateral acceleration
    assert CP.maxLateralAccel > 0.1
    if CP.maxLateralAccel >= GOOD_TORQUE_THRESHOLD:
      self.row[Column.STEERING_TORQUE] = Star.FULL

    self.all_footnotes = all_footnotes
    self.year_list = get_year_list(self.years)
    self.detail_sentence = self.get_detail_sentence(CP)

    return self

  def init_make(self, CP: car.CarParams):
    """CarInfo subclasses can add make-specific logic for harness selection, footnotes, etc."""

  def get_detail_sentence(self, CP):
    if not CP.notCar:
      sentence_builder = "openpilot upgrades your <strong>{car_model}</strong> with automated lane centering{alc} and adaptive cruise control{acc}."

      if self.min_steer_speed > self.min_enable_speed:
        alc = f" <strong>above {self.min_steer_speed * CV.MS_TO_MPH:.0f} mph</strong>," if self.min_steer_speed > 0 else " <strong>at all speeds</strong>,"
      else:
        alc = ""

      # Exception for cars which do not auto-resume yet
      acc = ""
      if self.min_enable_speed > 0:
        acc = f" <strong>while driving above {self.min_enable_speed * CV.MS_TO_MPH:.0f} mph</strong>"
      elif CP.autoResumeSng:
        acc = " <strong>that automatically resumes from a stop</strong>"

      if self.row[Column.STEERING_TORQUE] != Star.FULL:
        sentence_builder += " This car may not be able to take tight turns on its own."

      return sentence_builder.format(car_model=f"{self.make} {self.model}", alc=alc, acc=acc)

    else:
      if CP.carFingerprint == "COMMA BODY":
        return "The body is a robotics dev kit that can run openpilot. <a href='https://www.commabody.com'>Learn more.</a>"
      else:
        raise Exception(f"This notCar does not have a detail sentence: {CP.carFingerprint}")

  def get_column(self, column: Column, star_icon: str, footnote_tag: str) -> str:
    item: Union[str, Star] = self.row[column]
    if isinstance(item, Star):
      item = star_icon.format(item.value)
    elif column == Column.MODEL and len(self.years):
      item += f" {self.years}"

    footnotes = get_footnotes(self.footnotes, column)
    if len(footnotes):
      sups = sorted([self.all_footnotes[fn] for fn in footnotes])
      item += footnote_tag.format(f'{",".join(map(str, sups))}')

    return item
