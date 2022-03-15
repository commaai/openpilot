#!/usr/bin/env python3
from collections import namedtuple
from enum import Enum
from typing import Optional


class Tier(Enum):
  GOLD = "Gold"
  SILVER = "Silver"
  BRONZE = "Bronze"


class Column(Enum):
  MAKE = "Make"
  MODEL = "Model"
  PACKAGE = "Supported Package"
  LONGITUDINAL = "openpilot Longitudinal"
  FSR_LONGITUDINAL = "FSR Longitudinal"
  FSR_STEERING = "FSR Steering"
  STEERING_TORQUE = "Steering Torque"
  SUPPORTED = "Actively Maintained"


StarColumns = list(Column)[3:]
CarFootnote = namedtuple("CarFootnote", ["text", "column", "star"], defaults=[None])

# TODO: which other makes?
MAKES_GOOD_STEERING_TORQUE = ["toyota", "hyundai", "volkswagen"]


class Footnote(Enum):
  DSU = CarFootnote(
    "When disconnecting the Driver Support Unit (DSU), openpilot Adaptive Cruise Control (ACC) will replace " +
    "stock Adaptive Cruise Control (ACC). NOTE: disconnecting the DSU disables Automatic Emergency Braking (AEB).",
    Column.LONGITUDINAL, star="half")
  CAMRY = CarFootnote(
    "28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control.",
    Column.FSR_LONGITUDINAL)
  CIVIC_DIESEL = CarFootnote(
    "2019 Honda Civic 1.6L Diesel Sedan does not have ALC below 12mph.",
    Column.FSR_STEERING)
  OBD_II = CarFootnote(
    "Requires an [OBD-II](https://comma.ai/shop/products/comma-car-harness) car harness and [community built ASCM harness]" +
    "(https://github.com/commaai/openpilot/wiki/GM#hardware). NOTE: disconnecting the ASCM disables Automatic Emergency Braking (AEB).",
    Column.MODEL)
  KAMIQ = CarFootnote(
    "Not including the China market Kamiq, which is based on the (currently) unsupported PQ34 platform.",
    Column.MODEL)
  PASSAT = CarFootnote(
    "Not including the USA/China market Passat, which is based on the (currently) unsupported PQ35/NMS platform.",
    Column.MODEL)
  VW_HARNESS = CarFootnote(
    "Model-years 2021 and beyond may have a new camera harness design, which isn't yet available from the comma " +
    "store. Before ordering, remove the Lane Assist camera cover and check to see if the connector is black " +
    "(older design) or light brown (newer design). For the newer design, in the interim, choose \"VW J533 Development\" " +
    "from the vehicle drop-down for a harness that integrates at the CAN gateway inside the dashboard.",
    Column.MODEL)
  ANGLE_SENSOR = CarFootnote(
    "An inaccurate steering wheel angle sensor makes precise control difficult.",
    Column.STEERING_TORQUE, star="half")


def get_star_icon(variant):
  return '<img src="assets/icon-star-{}.svg" width="22" />'.format(variant)


def get_footnote(car_info, column) -> Optional[Footnote]:
  # Returns applicable footnote given current column
  if car_info.footnotes is not None:
    for fn in car_info.footnotes:
      if fn.value.column == column:
        return fn
  return None
