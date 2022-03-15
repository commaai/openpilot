#!/usr/bin/env python3
from collections import namedtuple
from enum import Enum


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


StarColumns = list(Column)[3:]
CarFootnote = namedtuple("CarFootnote", ["text", "column", "star"], defaults=[None])


def get_star_icon(variant):
  return '<a href="#"><img src="assets/icon-star-{}.svg" width="22" /></a>'.format(variant)


def get_footnote(car_info, column):
  # Returns applicable footnote given current column
  if car_info.footnotes is not None:
    for fn in car_info.footnotes:
      if fn.value.column == column:
        return fn
  return None
