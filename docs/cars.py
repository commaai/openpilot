#!/usr/bin/env python3
from collections import defaultdict, namedtuple
from enum import Enum
import os

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.test.test_routes import non_tested_cars


class Tier(Enum):
  GOLD = "Gold"
  SILVER = "Silver"
  BRONZE = "Bronze"


class Car:
  def __init__(self, car_info, CP):
    self.make, self.model = car_info.name.split(' ', 1)

    assert len(car_info.years), 'Model {} has no years listed'.format(CP.carFingerprint)

    # TODO: properly format model years
    years = ' ' + str(max(car_info.years))
    self.model_string = "{}{}".format(self.model, years)
    self.package = car_info.package

    self.stars = Stars(
      CP.openpilotLongitudinalControl,
      CP.minEnableSpeed <= 1e-3,  # TODO: 0 is probably okay
      CP.minSteerSpeed <= 1e-3,
      CP.carName in MAKES_GOOD_STEERING_TORQUE,
      # TODO: make sure this check is complete
      CP.carFingerprint not in non_tested_cars,
    )

  def format_stars(self):
    # TODO: exceptions and half stars
    return [STAR_ICON_FULL if cat else STAR_ICON_EMPTY for cat in self.stars]

  @property
  def tier(self):
    return {5: Tier.GOLD, 4: Tier.SILVER}.get(sum(self.stars), Tier.BRONZE)


def make_row(columns):
  return "|{}|".format("|".join(columns))


# TODO: unify with column names below?
Stars = namedtuple("Stars", ["op_long", "fsr_long", "fsr_lat", "steering_torque", "well_supported"])

STAR_ICON_FULL = '<img src="assets/icon-star-full.png" width="22" />'
STAR_ICON_HALF = '<img src="assets/icon-star-half.png" width="22" />'
STAR_ICON_EMPTY = '<img src="assets/icon-star-empty.png" width="22" />'

CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS_generated.md")
CAR_TABLE_COLUMNS = make_row(['Make', 'Model (US Market Reference)', 'Supported Package', 'openpilot Longitudinal',
                              'FSR Longitudinal', 'FSR Steering', 'Steering Torque', 'Actively Maintained'])
CAR_TABLE_HEADER = make_row(["---"] * 3 + [":---:"] * 5)  # first three aren't centered
CAR_ROW_TEMPLATE = make_row(["{}"] * 8)

# TODO: which other makes?
MAKES_GOOD_STEERING_TORQUE = ["toyota", "hyundai", "volkswagen"]


def generate_cars_md():
  tiered_cars = defaultdict(list)

  for _, models in get_interface_attr("CAR_INFO").items():
    for model, car_info in models.items():
      CP = interfaces[model][0].get_params(model)
      # Skip community supported
      if CP.dashcamOnly:
        continue

      # Some candidates have multiple variants
      if not isinstance(car_info, list):
        car_info = [car_info]

      for _car_info in car_info:
        car = Car(_car_info, CP)
        tiered_cars[car.tier].append(car)

  # Build CARS.md
  cars_md_doc = []
  for tier in Tier:
    # Sort by make, model name, and year
    cars = sorted(tiered_cars[tier], key=lambda car: car.make + car.model_string)

    cars_md_doc.append("## {} Cars\n".format(tier.name.title()))
    cars_md_doc.append(CAR_TABLE_COLUMNS)
    cars_md_doc.append(CAR_TABLE_HEADER)
    for car in cars:
      line = CAR_ROW_TEMPLATE.format(car.make,
                                     car.model_string,
                                     car.package,
                                     *car.format_stars())
      cars_md_doc.append(line)
    cars_md_doc.append("")  # newline

  return '\n'.join(cars_md_doc)


if __name__ == "__main__":
  # TODO: add argparse for generating json or html (undecided)
  Params().put_bool("DisableRadar", True)

  with open(CARS_MD_OUT, 'w') as f:
    f.write(generate_cars_md())

  print('Generated and written to {}'.format(CARS_MD_OUT))
