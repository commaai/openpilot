#!/usr/bin/env python3
import argparse
import jinja2
import os
from enum import Enum
from typing import Dict, List

from common.basedir import BASEDIR
from selfdrive.car.docs_definitions import CarInfo, Column, Star, Tier
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR
from selfdrive.car.tests.routes import non_tested_cars


def get_all_footnotes() -> Dict[Enum, int]:
  all_footnotes = []
  for _, footnotes in get_interface_attr("Footnote").items():
    if footnotes is not None:
      all_footnotes += footnotes
  return {fn: idx + 1 for idx, fn in enumerate(all_footnotes)}


ALL_FOOTNOTES: Dict[Enum, int] = get_all_footnotes()
CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")


def get_tier_car_info() -> Dict[Tier, List[CarInfo]]:
  tier_car_info: Dict[Tier, List[CarInfo]] = {tier: [] for tier in Tier}

  for models in get_interface_attr("CAR_INFO").values():
    for model, car_info in models.items():
      # Hyundai exception: those with radar have openpilot longitudinal
      fingerprint = {0: {}, 1: {HKG_RADAR_START_ADDR: 8}, 2: {}, 3: {}}
      CP = interfaces[model][0].get_params(model, fingerprint=fingerprint, disable_radar=True)

      if CP.dashcamOnly:
        continue

      # A platform can include multiple car models
      if not isinstance(car_info, list):
        car_info = (car_info,)

      for _car_info in car_info:
        _car_info.init(CP, non_tested_cars, ALL_FOOTNOTES)
        tier_car_info[_car_info.tier].append(_car_info)

  # Sort cars by make and model + year
  for tier, cars in tier_car_info.items():
    tier_car_info[tier] = sorted(cars, key=lambda x: x.make + x.model)

  return tier_car_info


def generate_cars_md(tier_car_info: Dict[Tier, List[CarInfo]], template_fn: str) -> str:
  with open(template_fn, "r") as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)

  footnotes = [fn.value.text for fn in ALL_FOOTNOTES]
  return template.render(tiers=tier_car_info, footnotes=footnotes, Star=Star, Column=Column)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supported cars documentation",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--template", default=CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_tier_car_info(), args.template))
  print(f"Generated and written to {args.out}")
