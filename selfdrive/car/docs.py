#!/usr/bin/env python3
import argparse
import jinja2
import os
from natsort import natsorted
from typing import List

from common.basedir import BASEDIR
from selfdrive.car.docs_definitions import STAR_DESCRIPTIONS, StarColumns, TierColumns, CarInfo, Column, Star, get_all_footnotes
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR


CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")


def get_all_car_info(hide_cols=None) -> List[CarInfo]:
  if hide_cols is None:
    hide_cols = []
  all_footnotes = get_all_footnotes(hide_cols)

  all_car_info: List[CarInfo] = []
  for model, car_info in get_interface_attr("CAR_INFO", combine_brands=True).items():
    # Hyundai exception: those with radar have openpilot longitudinal
    fingerprint = {0: {}, 1: {HKG_RADAR_START_ADDR: 8}, 2: {}, 3: {}}
    CP = interfaces[model][0].get_params(model, fingerprint=fingerprint, disable_radar=True)

    if CP.dashcamOnly or car_info is None:
      continue

    # A platform can include multiple car models
    if not isinstance(car_info, list):
      car_info = (car_info,)

    for _car_info in car_info:
      all_car_info.append(_car_info.init(CP, hide_cols, all_footnotes))

  # Sort cars by make and model + year
  sorted_cars: List[CarInfo] = natsorted(all_car_info, key=lambda car: car.name.lower())
  return sorted_cars


def generate_cars_md(all_car_info: List[CarInfo], template_fn: str, hide_cols: bool) -> str:
  with open(template_fn, "r") as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)

  cols = [c for c in Column if c not in hide_cols]
  footnotes = [fn.value.text for fn in get_all_footnotes(hide_cols)]
  cars_md: str = template.render(all_car_info=all_car_info, footnotes=footnotes,
                                 Star=Star, Column=cols, star_descriptions=STAR_DESCRIPTIONS)
  return cars_md


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supported cars documentation",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--tier-columns", action="store_true", help="Include only columns that count in the tier")
  parser.add_argument("--template", default=CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  hide_cols = set(StarColumns) - set(TierColumns) if args.tier_columns else []
  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_all_car_info(hide_cols), args.template, hide_cols))
  print(f"Generated and written to {args.out}")
