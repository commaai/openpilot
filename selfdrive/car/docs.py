#!/usr/bin/env python3
import jinja2
import os

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.car.docs_definitions import Column, Tier
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR
from selfdrive.car.tests.routes import non_tested_cars


def get_all_footnotes():
  all_footnotes = {}
  i = 1
  for _, footnotes in get_interface_attr("FOOTNOTES").items():
    if footnotes is not None:
      for footnote in footnotes.values():
        all_footnotes[footnote] = i
        i += 1
  return all_footnotes


ALL_FOOTNOTES = get_all_footnotes()


def get_tier_car_rows():
  tier_car_rows = {tier: [] for tier in Tier}

  for models in get_interface_attr("CAR_INFO").values():
    for model, car_info in models.items():
      # Hyundai exception: all have openpilot longitudinal
      fingerprint = {0: {}, 1: {HKG_RADAR_START_ADDR: 8}, 2: {}, 3: {}}
      CP = interfaces[model][0].get_params(model, fingerprint=fingerprint)

      # Skip community supported
      if CP.dashcamOnly:
        continue

      # Some candidates have multiple variants
      if not isinstance(car_info, list):
        car_info = (car_info, )

      for _car_info in car_info:
        tier_car_rows[_car_info.tier].append(_car_info.get_row(CP, non_tested_cars, ALL_FOOTNOTES))

  # Return tier title and car rows for each tier
  for tier, car_rows in tier_car_rows.items():
    yield [tier.name.title(), sorted(car_rows)]


def generate_cars_md(tier_car_rows):
  template_fn = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")
  with open(template_fn, "r") as f:
    template = jinja2.Template(f.read(), trim_blocks=True)

  footnotes = map(lambda fn: fn.text, ALL_FOOTNOTES)
  return template.render(tiers=tier_car_rows, columns=[column.value for column in Column], footnotes=footnotes)


if __name__ == "__main__":
  # Auto generates supported cars documentation

  # TODO: Remove these Hyundai exceptions once full long support is added
  # Cars that can disable radar have openpilot longitudinal
  Params().put_bool("DisableRadar", True)
  tier_car_rows = get_tier_car_rows()

  cars_md_fn = os.path.join(BASEDIR, "docs", "CARS.md")
  with open(cars_md_fn, 'w') as f:
    f.write(generate_cars_md(tier_car_rows))

  print(f"Generated and written to {cars_md_fn}")
