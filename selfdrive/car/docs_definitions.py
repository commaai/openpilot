from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, NamedTuple


@dataclass
class CarInfo:
  name: str
  package: str
  video_link: Optional[str] = None
  footnotes: Optional[List[NamedTuple]] = None
  min_steer_speed: Optional[float] = None
  min_enable_speed: Optional[float] = None
  good_torque: bool = False

  def get_stars(self, CP, non_tested_cars):
    # TODO: set all the min steer speeds in carParams and remove this
    min_steer_speed = CP.minSteerSpeed
    if self.min_steer_speed is not None:
      min_steer_speed = self.min_steer_speed
      assert CP.minSteerSpeed == 0, f"Minimum steer speed set in both CarInfo and CarParams for {CP.carFingerprint}"

    # TODO: Prius V has stop and go, but only with smartDSU,
    # set all the min enable speeds in carParams correctly and remove this
    min_enable_speed = CP.minEnableSpeed
    if self.min_enable_speed is not None:
      min_enable_speed = self.min_enable_speed

    stars = [CP.openpilotLongitudinalControl and not CP.radarOffCan, min_enable_speed <= 0.,
             min_steer_speed <= 0., self.good_torque, CP.carFingerprint not in non_tested_cars]

    for idx, column in enumerate(StarColumns):
      stars[idx] = Star.FULL if stars[idx] else Star.EMPTY

      # Demote if footnote specifies a star
      footnote = get_footnote(self.footnotes, column)
      if footnote is not None and footnote.star is not None:
        stars[idx] = footnote.star

    return stars

  def get_row(self, all_footnotes, stars):
    # TODO: add YouTube videos
    row = [*self.name.split(' ', 1), self.package, *stars]

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
  SUPPORTED = "Actively Maintained"


class Star(Enum):
  FULL = "full"
  HALF = "half"
  EMPTY = "empty"

  @property
  def icon(self):
    return f'<a href="#"><img valign="top" src="assets/icon-star-{self.value}.svg" width="22" /></a>'


StarColumns = list(Column)[3:]
CarFootnote = namedtuple("CarFootnote", ["text", "column", "star"], defaults=[None])


def get_footnote(footnotes: Optional[List[CarFootnote]], column: Column) -> Optional[CarFootnote]:
  # Returns applicable footnote given current column
  if footnotes is not None:
    for fn in footnotes:
      if fn.column == column:
        return fn
  return None
