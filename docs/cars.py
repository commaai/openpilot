#!/usr/bin/env python3
from collections import defaultdict, namedtuple
from enum import Enum
import os
from typing import Dict

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.gm.values import CAR as GM
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.car.volkswagen.values import CAR as VOLKSWAGEN
from selfdrive.test.test_routes import non_tested_cars


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
CarException = namedtuple("CarException", ["cars", "text", "column", "star"], defaults=[None])


def make_row(columns):
  return "|{}|".format("|".join(columns))


def get_star_icon(variant):
  return '<img src="assets/icon-star-{}.png" width="22" />'.format(variant)


def get_exceptions(CP) -> Dict[Column, CarException]:
  exceptions = {}
  for car_exception in CAR_EXCEPTIONS:
    if CP.carFingerprint in car_exception.cars:
      exceptions[car_exception.column] = car_exception
  return exceptions


CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS_generated.md")
CAR_TABLE_HEADER = make_row(["---"] * 3 + [":---:"] * 5)  # first three aren't centered

# TODO: which other makes?
MAKES_GOOD_STEERING_TORQUE = ["toyota", "hyundai", "volkswagen"]
CAR_EXCEPTIONS = [
  CarException([TOYOTA.LEXUS_CTH, TOYOTA.LEXUS_ESH, TOYOTA.LEXUS_NX, TOYOTA.LEXUS_NXH, TOYOTA.LEXUS_RX,
                TOYOTA.LEXUS_RXH, TOYOTA.AVALON, TOYOTA.AVALONH_2019, TOYOTA.COROLLA, TOYOTA.HIGHLANDER,
                TOYOTA.HIGHLANDERH, TOYOTA.PRIUS, TOYOTA.PRIUS_V, TOYOTA.RAV4, TOYOTA.RAV4H, TOYOTA.SIENNA],
               "When disconnecting the Driver Support Unit (DSU), openpilot Adaptive Cruise Control (ACC) will replace "
               "stock Adaptive Cruise Control (ACC). NOTE: disconnecting the DSU disables Automatic Emergency Braking (AEB).",
               Column.LONGITUDINAL, star="half"),
  CarException([TOYOTA.CAMRY, TOYOTA.CAMRY_TSS2, TOYOTA.CAMRYH],
               "28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control.",
               Column.FSR_LONGITUDINAL),
  CarException([GM.ESCALADE_ESV, GM.VOLT, GM.ACADIA],
               "Requires an [OBD-II](https://comma.ai/shop/products/comma-car-harness) car harness and [community built ASCM harness]"
               "(https://github.com/commaai/openpilot/wiki/GM#hardware). NOTE: disconnecting the ASCM disables Automatic Emergency Braking (AEB).",
               Column.MODEL),
  CarException([VOLKSWAGEN.SKODA_KAMIQ_MK1],
               "Not including the China market Kamiq, which is based on the (currently) unsupported PQ34 platform.",
               Column.MODEL),
  CarException([VOLKSWAGEN.PASSAT_MK8],
               "Not including the USA/China market Passat, which is based on the (currently) unsupported PQ35/NMS platform.",
               Column.MODEL),
  CarException([VOLKSWAGEN.ARTEON_MK1, VOLKSWAGEN.ATLAS_MK1, VOLKSWAGEN.TRANSPORTER_T61, VOLKSWAGEN.TCROSS_MK1,
                VOLKSWAGEN.TROC_MK1, VOLKSWAGEN.TAOS_MK1, VOLKSWAGEN.TIGUAN_MK2],
               'Model-years 2021 and beyond may have a new camera harness design, which isn\'t yet available from the comma '
               'store. Before ordering, remove the Lane Assist camera cover and check to see if the connector is black '
               '(older design) or light brown (newer design). For the newer design, in the interim, choose "VW J533 Development" '
               'from the vehicle drop-down for a harness that integrates at the CAN gateway inside the dashboard.',
               Column.MODEL),
  CarException([TOYOTA.PRIUS, TOYOTA.PRIUS_V],
               "An inaccurate steering wheel angle sensor makes precise control difficult.",
               Column.STEERING_TORQUE, star="half"),
]


class Car:
  def __init__(self, car_info, CP):
    self.make, self.model = car_info.name.split(' ', 1)
    self.package = car_info.package
    self.exceptions = get_exceptions(CP)
    self.stars = self._calculate_stars(CP, car_info)

  @property
  def row(self):
    # TODO: add YouTube videos
    row = [self.make, self.model, self.package, *map(get_star_icon, self.stars)]

    # Check for car exceptions
    for row_idx, column in enumerate(Column):
      exception = self.exceptions.get(column, None)
      if exception is not None:
        superscript_number = CAR_EXCEPTIONS.index(exception) + 1
        row[row_idx] += "<sup>{}</sup>".format(superscript_number)

    return make_row(row)

  @property
  def tier(self):
    return {5: Tier.GOLD, 4: Tier.SILVER}.get(self.stars.count("full"), Tier.BRONZE)

  def _calculate_stars(self, CP, car_info):
    # Some minimum steering speeds are not yet in CarParams
    min_steer_speed = CP.minSteerSpeed
    if car_info.min_steer_speed is not None:
      min_steer_speed = car_info.min_steer_speed
      assert CP.minSteerSpeed == 0, "Minimum steer speed set in both CarInfo and CarParams for {}".format(CP.carFingerprint)

    min_enable_speed = CP.minEnableSpeed
    if car_info.min_enable_speed is not None:
      min_enable_speed = car_info.min_enable_speed

    # TODO: make sure well supported check is complete
    stars = [CP.openpilotLongitudinalControl and not CP.radarOffCan, min_enable_speed <= 1e-3, min_steer_speed <= 1e-3,
             CP.carName in MAKES_GOOD_STEERING_TORQUE, CP.carFingerprint not in non_tested_cars]

    # Check for star demotions from exceptions
    for idx, (star, column) in enumerate(zip(stars, StarColumns)):
      star = "full" if star else "empty"
      exception = self.exceptions.get(column, None)
      if exception is not None and exception.star is not None:
        star = exception.star.lower()
      stars[idx] = star
    return stars


def get_tiered_cars():
  tiered_cars = defaultdict(list)
  for _, models in get_interface_attr("CAR_INFO").items():
    for model, car_info in models.items():
      # Hyundai exception: all have openpilot longitudinal
      fingerprint = defaultdict(dict)
      fingerprint[1] = {HKG_RADAR_START_ADDR: 8}
      CP = interfaces[model][0].get_params(model, fingerprint=fingerprint)
      # Skip community supported
      if CP.dashcamOnly:
        continue

      # Some candidates have multiple variants
      if not isinstance(car_info, list):
        car_info = [car_info]

      for _car_info in car_info:
        car = Car(_car_info, CP)
        tiered_cars[car.tier].append(car)

  return tiered_cars


def generate_cars_md(tiered_cars):
  cars_md_doc = []
  for tier in Tier:
    # Sort by make, model name, and year
    cars = sorted(tiered_cars[tier], key=lambda car: car.make + car.model)

    cars_md_doc.append("## {} Cars\n".format(tier.name.title()))

    cars_md_doc.append(make_row([column.value for column in Column]))
    cars_md_doc.append(CAR_TABLE_HEADER)
    cars_md_doc.extend(map(lambda car: car.row, cars))
    cars_md_doc.append("")  # newline

  return '\n'.join(cars_md_doc)


if __name__ == "__main__":
  # TODO: add argparse for generating json or html (undecided)
  # Cars that can disable radar have openpilot longitudinal
  Params().put_bool("DisableRadar", True)

  tiered_cars = get_tiered_cars()
  with open(CARS_MD_OUT, 'w') as f:
    f.write(generate_cars_md(tiered_cars))

  print('Generated and written to {}'.format(CARS_MD_OUT))
