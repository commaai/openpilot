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
  HARDWARE = "Hardware Needed"
  VIDEO = "Video"


class Star(Enum):
  FULL = "full"
  HALF = "half"
  EMPTY = "empty"


class PartType(Enum):
  harness_connector = "Harness Connector"
  generic_part = "Generic Part"


Part = namedtuple("Part", ["value","type"], defaults=("",PartType.generic_part))


class CarPart(Enum):
  #harness connectors
  nidec_connector = Part("Honda Nidec connector", type=PartType.harness_connector)
  bosch_a_connector = Part("Honda Bosch A connector", type=PartType.harness_connector)
  bosch_b_connector = Part("Honda Bosch B connector", type=PartType.harness_connector)
  toyota_connector = Part("Toyota connector", type=PartType.harness_connector)
  subaru_a_connector = Part("Subaru A connector", type=PartType.harness_connector)
  subaru_b_connector = Part("Subaru B connector", type=PartType.harness_connector)
  fca_connector = Part("FCA connector", type=PartType.harness_connector)
  ram_connector = Part("Ram connector", type=PartType.harness_connector)
  vw_connector = Part("VW connector", type=PartType.harness_connector)
  j533_connector = Part("J533 connector", type=PartType.harness_connector)
  hyundai_a_connector = Part("Hyundai A connector", type=PartType.harness_connector)
  hyundai_b_connector = Part("Hyundai B connector", type=PartType.harness_connector)
  hyundai_c_connector = Part("Hyundai C connector", type=PartType.harness_connector)
  hyundai_d_connector = Part("Hyundai D connector", type=PartType.harness_connector)
  hyundai_e_connector = Part("Hyundai E connector", type=PartType.harness_connector)
  hyundai_f_connector = Part("Hyundai F connector", type=PartType.harness_connector)
  hyundai_g_connector = Part("Hyundai G connector", type=PartType.harness_connector)
  hyundai_h_connector = Part("Hyundai H connector", type=PartType.harness_connector)
  hyundai_i_connector = Part("Hyundai I connector", type=PartType.harness_connector)
  hyundai_j_connector = Part("Hyundai J connector", type=PartType.harness_connector)
  hyundai_k_connector = Part("Hyundai K connector", type=PartType.harness_connector)
  hyundai_l_connector = Part("Hyundai L connector", type=PartType.harness_connector)
  hyundai_m_connector = Part("Hyundai M connector", type=PartType.harness_connector)
  hyundai_n_connector = Part("Hyundai N connector", type=PartType.harness_connector)
  hyundai_o_connector = Part("Hyundai O connector", type=PartType.harness_connector)
  hyundai_p_connector = Part("Hyundai P connector", type=PartType.harness_connector)
  hyundai_q_connector = Part("Hyundai Q connector", type=PartType.harness_connector)
  custom_connector = Part("Developer connector", type=PartType.harness_connector)
  obd_ii_connector = Part("OBD-II connector", type=PartType.harness_connector)
  gm_connector = Part("GM connector", type=PartType.harness_connector)
  nissan_a_connector = Part("Nissan A connector", type=PartType.harness_connector)
  nissan_b_connector = Part("Nissan B connector", type=PartType.harness_connector)
  mazda_connector = Part("Mazda connector", type=PartType.harness_connector)
  ford_q3_connector = Part("Ford Q3 connector", type=PartType.harness_connector)
  ford_q4_connector = Part("Ford Q4 connector", type=PartType.harness_connector)
  none_connector = Part("None connector", type=PartType.harness_connector)

  #harness accessories
  harness_box = Part("harness box")
  comma_power_v2 = Part("comma power v2")

  #mounts
  mount = Part("mount")
  angled_mount = Part("angled mount")

  #cables
  rj45_cable_7ft = Part("RJ45 cable (7 ft)")
  long_obdc_cable = Part("long OBD-C cable")
  usb_a_2_a_cable = Part("USB A-A cable")
  usbc_otg_cable = Part("USB C OTG cable")
  usbc_coupler = Part("USB-C coupler")
  obd_c_cable_1point5ft = Part("OBD-C cable (1.5 ft)")

  #devices
  comma_3 = Part("comma 3")
  red_panda = Part("red panda")


DEFAULT_CAR_PARTS: List[CarPart] = [CarPart.harness_box, CarPart.comma_power_v2, CarPart.rj45_cable_7ft, CarPart.mount]


@dataclass
class CarParts:
  parts: List[CarPart] = field(default_factory=list)

  @classmethod
  def default(cls, add: List[CarPart] = None, default: List[CarPart] = None, remove: List[CarPart] = None):
    p = [part for part in (add or []) + (default or DEFAULT_CAR_PARTS) if part not in (remove or [])]
    return cls(p)


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
  # make + model + model years
  name: str

  # Example for Toyota Corolla MY20
  # requirements: Lane Tracing Assist (LTA) and Dynamic Radar Cruise Control (DRCC)
  # US Market reference: "All", since all Corolla in the US come standard with LTA and DRCC

  # the simplest description of the requirements for the US market
  package: str

  # the minimum compatibility requirements for this model, regardless
  # of market. can be a package, trim, or list of features
  requirements: Optional[str] = None

  video_link: Optional[str] = None
  footnotes: List[Enum] = field(default_factory=list)
  min_steer_speed: Optional[float] = None
  min_enable_speed: Optional[float] = None

  # all the parts needed for the supported car
  car_parts: CarParts = CarParts()

  def init(self, CP: car.CarParams, all_footnotes: Dict[Enum, int]):
    self.car_name = CP.carName
    self.car_fingerprint = CP.carFingerprint
    self.make, self.model, self.years = split_name(self.name)

    # longitudinal column
    op_long = "Stock"
    if CP.openpilotLongitudinalControl and not CP.enableDsu:
      op_long = "openpilot"
    elif CP.experimentalLongitudinalAvailable or CP.enableDsu:
      op_long = "openpilot available"
      if CP.enableDsu:
        self.footnotes.append(CommonFootnote.EXP_LONG_DSU)
      else:
        self.footnotes.append(CommonFootnote.EXP_LONG_AVAIL)

    # min steer & enable speed columns
    # TODO: set all the min steer speeds in carParams and remove this
    if self.min_steer_speed is not None:
      assert CP.minSteerSpeed == 0, f"{CP.carFingerprint}: Minimum steer speed set in both CarInfo and CarParams"
    else:
      self.min_steer_speed = CP.minSteerSpeed

    # TODO: set all the min enable speeds in carParams correctly and remove this
    if self.min_enable_speed is None:
      self.min_enable_speed = CP.minEnableSpeed

    # hardware column
    hardware_col = "None"
    if self.car_parts.parts:
      model_years = self.model + (' ' + self.years if self.years else '')
      buy_link = f'<a href="https://comma.ai/shop/comma-three.html?make={self.make}&model={model_years}">Buy Here</a>'
      parts = '<br>'.join([f"- {self.car_parts.parts.count(part)} {part.value.value}" for part in sorted(set(self.car_parts.parts), key=lambda part: part.value)])
      hardware_col = f'<details><summary>View</summary><sub>{parts}<br>{buy_link}</sub></details>'

    self.row: Dict[Enum, Union[str, Star]] = {
      Column.MAKE: self.make,
      Column.MODEL: self.model,
      Column.PACKAGE: self.package,
      Column.LONGITUDINAL: op_long,
      Column.FSR_LONGITUDINAL: f"{max(self.min_enable_speed * CV.MS_TO_MPH, 0):.0f} mph",
      Column.FSR_STEERING: f"{max(self.min_steer_speed * CV.MS_TO_MPH, 0):.0f} mph",
      Column.STEERING_TORQUE: Star.EMPTY,
      Column.AUTO_RESUME: Star.FULL if CP.autoResumeSng else Star.EMPTY,
      Column.HARDWARE: hardware_col,
      Column.VIDEO: self.video_link if self.video_link is not None else "",  # replaced with an image and link from template in get_column
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

      # experimental mode
      exp_link = "<a href='https://blog.comma.ai/090release/#experimental-mode' target='_blank' class='link-light-new-regular-text'>Experimental mode</a>"
      if CP.openpilotLongitudinalControl or CP.experimentalLongitudinalAvailable:
        sentence_builder += f" Traffic light and stop sign handling is also available in {exp_link}."
      else:
        sentence_builder += f" {exp_link}, with traffic light and stop sign handling, is not currently available for this car, but may be added in a future software update."

      return sentence_builder.format(car_model=f"{self.make} {self.model}", alc=alc, acc=acc)

    else:
      if CP.carFingerprint == "COMMA BODY":
        return "The body is a robotics dev kit that can run openpilot. <a href='https://www.commabody.com'>Learn more.</a>"
      else:
        raise Exception(f"This notCar does not have a detail sentence: {CP.carFingerprint}")

  def get_column(self, column: Column, star_icon: str, video_icon: str, footnote_tag: str) -> str:
    item: Union[str, Star] = self.row[column]
    if isinstance(item, Star):
      item = star_icon.format(item.value)
    elif column == Column.MODEL and len(self.years):
      item += f" {self.years}"
    elif column == Column.VIDEO and len(item) > 0:
      item = video_icon.format(item)

    footnotes = get_footnotes(self.footnotes, column)
    if len(footnotes):
      sups = sorted([self.all_footnotes[fn] for fn in footnotes])
      item += footnote_tag.format(f'{",".join(map(str, sups))}')

    return item
