import re
from collections import namedtuple
import copy
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


# A part + its comprised parts
@dataclass
class BasePart:
  name: str
  parts: List[Enum] = field(default_factory=list)

  def all_parts(self):
    # Recursively get all parts
    _parts = 'parts'
    parts = []
    parts.extend(getattr(self, _parts))
    for part in getattr(self, _parts):
      parts.extend(part.value.all_parts())

    return parts


class EnumBase(Enum):
  @property
  def type(self):
    return PartType(self.__class__)


class Mount(EnumBase):
  mount = BasePart("mount")
  angled_mount_8_degrees = BasePart("angled mount (8 degrees)")


class Cable(EnumBase):
  rj45_cable_7ft = BasePart("RJ45 cable (7 ft)")
  long_obdc_cable = BasePart("long OBD-C cable")
  usb_a_2_a_cable = BasePart("USB A-A cable")
  usbc_otg_cable = BasePart("USB C OTG cable")
  usbc_coupler = BasePart("USB-C coupler")
  obd_c_cable_1_5ft = BasePart("OBD-C cable (1.5 ft)")
  right_angle_obd_c_cable_1_5ft = BasePart("right angle OBD-C cable (1.5 ft)")


class Accessory(EnumBase):
  harness_box = BasePart("harness box")
  comma_power_v2 = BasePart("comma power v2")


@dataclass
class BaseCarHarness(BasePart):
  parts: List[Enum] = field(default_factory=lambda: [Accessory.harness_box, Accessory.comma_power_v2, Cable.rj45_cable_7ft])
  has_connector: bool = True  # without are hidden on the harness connector page


class CarHarness(EnumBase):
  nidec = BaseCarHarness("Honda Nidec connector")
  bosch_a = BaseCarHarness("Honda Bosch A connector")
  bosch_b = BaseCarHarness("Honda Bosch B connector")
  toyota_a = BaseCarHarness("Toyota A connector")
  subaru_a = BaseCarHarness("Subaru A connector")
  subaru_b = BaseCarHarness("Subaru B connector")
  subaru_c = BaseCarHarness("Subaru C connector")
  subaru_d = BaseCarHarness("Subaru D connector")
  fca = BaseCarHarness("FCA connector")
  ram = BaseCarHarness("Ram connector")
  vw = BaseCarHarness("VW connector")
  j533 = BaseCarHarness("J533 connector", parts=[Accessory.harness_box, Cable.long_obdc_cable, Cable.usbc_coupler])
  hyundai_a = BaseCarHarness("Hyundai A connector")
  hyundai_b = BaseCarHarness("Hyundai B connector")
  hyundai_c = BaseCarHarness("Hyundai C connector")
  hyundai_d = BaseCarHarness("Hyundai D connector")
  hyundai_e = BaseCarHarness("Hyundai E connector")
  hyundai_f = BaseCarHarness("Hyundai F connector")
  hyundai_g = BaseCarHarness("Hyundai G connector")
  hyundai_h = BaseCarHarness("Hyundai H connector")
  hyundai_i = BaseCarHarness("Hyundai I connector")
  hyundai_j = BaseCarHarness("Hyundai J connector")
  hyundai_k = BaseCarHarness("Hyundai K connector")
  hyundai_l = BaseCarHarness("Hyundai L connector")
  hyundai_m = BaseCarHarness("Hyundai M connector")
  hyundai_n = BaseCarHarness("Hyundai N connector")
  hyundai_o = BaseCarHarness("Hyundai O connector")
  hyundai_p = BaseCarHarness("Hyundai P connector")
  hyundai_q = BaseCarHarness("Hyundai Q connector")
  custom = BaseCarHarness("Developer connector")
  obd_ii = BaseCarHarness("OBD-II connector", parts=[Cable.long_obdc_cable, Cable.long_obdc_cable], has_connector=False)
  gm = BaseCarHarness("GM connector")
  nissan_a = BaseCarHarness("Nissan A connector", parts=[Accessory.harness_box, Cable.rj45_cable_7ft, Cable.long_obdc_cable, Cable.usbc_coupler])
  nissan_b = BaseCarHarness("Nissan B connector", parts=[Accessory.harness_box, Cable.rj45_cable_7ft, Cable.long_obdc_cable, Cable.usbc_coupler])
  mazda = BaseCarHarness("Mazda connector")
  ford_q3 = BaseCarHarness("Ford Q3 connector")
  ford_q4 = BaseCarHarness("Ford Q4 connector")


class Device(EnumBase):
  three = BasePart("comma three", parts=[Mount.mount, Cable.right_angle_obd_c_cable_1_5ft])
  # variant of comma three with angled mounts
  three_angled_mount = BasePart("comma three", parts=[Mount.angled_mount_8_degrees, Cable.right_angle_obd_c_cable_1_5ft])
  red_panda = BasePart("red panda")


class Kit(EnumBase):
  red_panda_kit = BasePart("CAN FD panda kit", parts=[Device.red_panda, Accessory.harness_box,
                                                      Cable.usb_a_2_a_cable, Cable.usbc_otg_cable, Cable.obd_c_cable_1_5ft])


class Tool(EnumBase):
  socket_8mm_deep = BasePart("Socket Wrench 8mm or 5/16\" (deep)")
  pry_tool = BasePart("Pry Tool")


class PartType(Enum):
  accessory = Accessory
  cable = Cable
  connector = CarHarness
  device = Device
  kit = Kit
  mount = Mount
  tool = Tool


DEFAULT_CAR_PARTS: List[EnumBase] = [Device.three]


@dataclass
class CarParts:
  parts: List[EnumBase] = field(default_factory=list)

  def __call__(self):
    return copy.deepcopy(self)

  @classmethod
  def common(cls, add: Optional[List[EnumBase]] = None, remove: Optional[List[EnumBase]] = None):
    p = [part for part in (add or []) + DEFAULT_CAR_PARTS if part not in (remove or [])]
    return cls(p)

  def all_parts(self):
    parts = []
    for part in self.parts:
      parts.extend(part.value.all_parts())
    return self.parts + parts


CarFootnote = namedtuple("CarFootnote", ["text", "column", "docs_only", "shop_footnote"], defaults=(False, False))


class CommonFootnote(Enum):
  EXP_LONG_AVAIL = CarFootnote(
    "Experimental openpilot longitudinal control is available behind a toggle; " +
    "the toggle is only available in non-release branches such as `devel` or `master-ci`. ",
    Column.LONGITUDINAL, docs_only=True)
  EXP_LONG_DSU = CarFootnote(
    "By default, this car will use the stock Adaptive Cruise Control (ACC) for longitudinal control. " + 
    "If the Driver Support Unit (DSU) is disconnected, openpilot ACC will replace " +
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
  auto_resume: Optional[bool] = None

  # all the parts needed for the supported car
  car_parts: CarParts = field(default_factory=CarParts)

  def init(self, CP: car.CarParams, all_footnotes: Dict[Enum, int]):
    self.car_name = CP.carName
    self.car_fingerprint = CP.carFingerprint
    self.make, self.model, self.years = split_name(self.name)

    # longitudinal column
    op_long = "Stock"
    if CP.experimentalLongitudinalAvailable or CP.enableDsu:
      op_long = "openpilot available"
      if CP.enableDsu:
        self.footnotes.append(CommonFootnote.EXP_LONG_DSU)
      else:
        self.footnotes.append(CommonFootnote.EXP_LONG_AVAIL)
    elif CP.openpilotLongitudinalControl and not CP.enableDsu:
      op_long = "openpilot"

    # min steer & enable speed columns
    # TODO: set all the min steer speeds in carParams and remove this
    if self.min_steer_speed is not None:
      assert CP.minSteerSpeed == 0, f"{CP.carFingerprint}: Minimum steer speed set in both CarInfo and CarParams"
    else:
      self.min_steer_speed = CP.minSteerSpeed

    # TODO: set all the min enable speeds in carParams correctly and remove this
    if self.min_enable_speed is None:
      self.min_enable_speed = CP.minEnableSpeed

    if self.auto_resume is None:
      self.auto_resume = CP.autoResumeSng

    # hardware column
    hardware_col = "None"
    if self.car_parts.parts:
      model_years = self.model + (' ' + self.years if self.years else '')
      buy_link = f'<a href="https://comma.ai/shop/comma-3x.html?make={self.make}&model={model_years}">Buy Here</a>'

      tools_docs = [part for part in self.car_parts.all_parts() if isinstance(part, Tool)]
      parts_docs = [part for part in self.car_parts.all_parts() if not isinstance(part, Tool)]

      def display_func(parts):
        return '<br>'.join([f"- {parts.count(part)} {part.value.name}" for part in sorted(set(parts), key=lambda part: str(part.value.name))])

      hardware_col = f'<details><summary>Parts</summary><sub>{display_func(parts_docs)}<br>{buy_link}</sub></details>'
      if len(tools_docs):
        hardware_col += f'<details><summary>Tools</summary><sub>{display_func(tools_docs)}</sub></details>'

    self.row: Dict[Enum, Union[str, Star]] = {
      Column.MAKE: self.make,
      Column.MODEL: self.model,
      Column.PACKAGE: self.package,
      Column.LONGITUDINAL: op_long,
      Column.FSR_LONGITUDINAL: f"{max(self.min_enable_speed * CV.MS_TO_MPH, 0):.0f} mph",
      Column.FSR_STEERING: f"{max(self.min_steer_speed * CV.MS_TO_MPH, 0):.0f} mph",
      Column.STEERING_TORQUE: Star.EMPTY,
      Column.AUTO_RESUME: Star.FULL if self.auto_resume else Star.EMPTY,
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
      elif self.auto_resume:
        acc = " <strong>that automatically resumes from a stop</strong>"

      if self.row[Column.STEERING_TORQUE] != Star.FULL:
        sentence_builder += " This car may not be able to take tight turns on its own."

      # experimental mode
      exp_link = "<a href='https://blog.comma.ai/090release/#experimental-mode' target='_blank' class='link-light-new-regular-text'>Experimental mode</a>"
      if CP.openpilotLongitudinalControl or CP.experimentalLongitudinalAvailable:
        sentence_builder += f" Traffic light and stop sign handling is also available in {exp_link}."

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
