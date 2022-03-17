from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


@dataclass
class CarInfo:
  name: str
  package: str
  video_link: Optional[str] = None
  footnotes: Optional[List[Enum]] = None
  min_steer_speed: Optional[float] = None
  min_enable_speed: Optional[float] = None
  good_torque: bool = False

  def get_stars(self, CP, non_tested_cars):
    # TODO: set all the min steer speeds in carParams and remove this
    min_steer_speed = CP.minSteerSpeed
    if self.min_steer_speed is not None:
      min_steer_speed = self.min_steer_speed
      assert CP.minSteerSpeed == 0, f"Minimum steer speed set in both CarInfo and CarParams for {CP.carFingerprint}"

    # TODO: set all the min enable speeds in carParams correctly and remove this
    min_enable_speed = CP.minEnableSpeed
    if self.min_enable_speed is not None:
      min_enable_speed = self.min_enable_speed

    stars = {
      Column.LONGITUDINAL: CP.openpilotLongitudinalControl and not CP.radarOffCan,
      Column.FSR_LONGITUDINAL: min_enable_speed <= 0.,
      Column.FSR_STEERING: min_steer_speed <= 0.,
      Column.STEERING_TORQUE: self.good_torque,
      Column.MAINTAINED: CP.carFingerprint not in non_tested_cars,
    }

    for column in StarColumns:
      stars[column] = Star.FULL if stars[column] else Star.EMPTY

      # Demote if footnote specifies a star
      footnote = get_footnote(self.footnotes, column)
      if footnote is not None and footnote.value.star is not None:
        stars[column] = footnote.value.star

    return [stars[column] for column in StarColumns]

  def get_row(self, all_footnotes, stars):
    # TODO: add YouTube vidos
    make, model = self.name.split(' ', 1)
    row = [make, model, self.package, *stars]

    # Check for car footnotes and get star icons
    for row_idx, column in enumerate(Column):
      if column in StarColumns:
        row[row_idx] = row[row_idx].icon

      footnote = get_footnote(self.footnotes, column)
      if footnote is not None:
        row[row_idx] += f"[<sup>{all_footnotes[footnote]}</sup>](#Footnotes)"

    return row


class Tier(Enum):
  GOLD = "Gold"
  SILVER = "Silver"
  BRONZE = "Bronze"


class Column(Enum):
  MAKE = "Make"
  MODEL = "Model"
  PACKAGE = "Supported Package"
  LONGITUDINAL = "openpilot ACC"
  FSR_LONGITUDINAL = "Stop and Go"
  FSR_STEERING = "Steer to 0"
  STEERING_TORQUE = "Steering Torque"
  MAINTAINED = "Actively Maintained"


class Star(Enum):
  FULL = "full"
  HALF = "half"
  EMPTY = "empty"

  @property
  def icon(self):
    return f'<a href="#"><img valign="top" src="assets/icon-star-{self.value}.svg" width="22" /></a>'


StarColumns = list(Column)[3:]
CarFootnote = namedtuple("CarFootnote", ["text", "column", "star"], defaults=[None])


def get_footnote(footnotes: Optional[List[Enum]], column: Column) -> Optional[Enum]:
  # Returns applicable footnote given current column
  if footnotes is not None:
    for fn in footnotes:
      if fn.value.column == column:
        return fn
  return None
