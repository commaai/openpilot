#!/usr/bin/env python3
import jinja2
import os
from collections import defaultdict
from sortedcontainers import SortedList
from typing import Dict

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR
from selfdrive.test.test_routes import non_tested_cars
from selfdrive.car.docs import Tier, Column, StarColumns, MAKES_GOOD_STEERING_TORQUE, Footnote, get_footnote, get_star_icon


CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS_generated.md")


class Car:
  def __init__(self, car_info, CP):
    self.make, self.model = car_info.name.split(' ', 1)
    self.row, star_count = self.get_row(car_info, CP)
    self.tier = {5: Tier.GOLD, 4: Tier.SILVER}.get(star_count, Tier.BRONZE)

  def get_row(self, car_info, CP):
    # TODO: add YouTube videos
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
    row = [self.make, self.model, car_info.package, *map(lambda star: "full" if star else "empty", stars)]

    # Check for car footnotes and star demotions
    star_count = 0
    for row_idx, column in enumerate(Column):
      footnote = get_footnote(car_info, column)
      if column in StarColumns:
        # Demote if footnote specifies a star
        if footnote is not None and footnote.value.star is not None:
          row[row_idx] = footnote.value.star
        star_count += row[row_idx] == "full"
        row[row_idx] = get_star_icon(row[row_idx])

      if footnote is not None:
        superscript_number = list(Footnote).index(footnote) + 1
        row[row_idx] += "<sup>{}</sup>".format(superscript_number)

    return row, star_count


def get_tiered_cars():
  # Keep track of cars while sorting by make, model name, and year
  tiered_cars = {tier: SortedList(key=lambda car: car.make + car.model) for tier in Tier}

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
        tiered_cars[car.tier].add(car)

  # Return tier name and car rows for each tier
  for tier, cars in tiered_cars.items():
    yield [tier.name.title(), map(lambda car: car.row, cars)]


def generate_cars_md(tiered_cars):
  template_fn = os.path.join(BASEDIR, "docs", "CARS_template.md")
  with open(template_fn, "r") as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)  # TODO: remove lstrip_blocks if not needed

  footnotes = map(lambda fn: fn.value.text, Footnote)
  return template.render(tiers=tiered_cars, columns=[column.value for column in Column], footnotes=footnotes)


if __name__ == "__main__":
  # TODO: add argparse for generating json or html (undecided)
  # Cars that can disable radar have openpilot longitudinal
  Params().put_bool("DisableRadar", True)

  tiered_cars = get_tiered_cars()
  with open(CARS_MD_OUT, 'w') as f:
    f.write(generate_cars_md(tiered_cars))

  print('Generated and written to {}'.format(CARS_MD_OUT))
