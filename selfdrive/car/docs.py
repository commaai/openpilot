#!/usr/bin/env python3
import argparse
from collections import defaultdict
import jinja2
import os
from enum import Enum
from natsort import natsorted
from typing import Dict, List

from common.basedir import BASEDIR
from selfdrive.car.docs_definitions import STAR_DESCRIPTIONS, StarColumns, TierColumns, CarInfo, Column, Star
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR


def get_all_footnotes(only_tier_cols: bool = False) -> Dict[Enum, int]:
  all_footnotes = []
  hide_cols = set(StarColumns) - set(TierColumns) if only_tier_cols else []
  for footnotes in get_interface_attr("Footnote", ignore_none=True).values():
    all_footnotes.extend([fn for fn in footnotes if fn.value.column not in hide_cols])
  return {fn: idx + 1 for idx, fn in enumerate(all_footnotes)}


CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")


def get_all_car_info(only_tier_cols: bool = False) -> List[CarInfo]:
  all_car_info: List[CarInfo] = []
  footnotes = get_all_footnotes(only_tier_cols)
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
      if not hasattr(_car_info, "row"):
        _car_info.init(CP, footnotes)
      all_car_info.append(_car_info)

  # Sort cars by make and model + year
  sorted_cars: List[CarInfo] = natsorted(all_car_info, key=lambda car: car.name.lower())
  return sorted_cars


def group_by_make(all_car_info: List[CarInfo]) -> Dict[str, List[CarInfo]]:
  sorted_car_info = defaultdict(list)
  for car_info in all_car_info:
    sorted_car_info[car_info.make].append(car_info)
  return dict(sorted_car_info)


def generate_cars_md(all_car_info: List[CarInfo], template_fn: str, only_tier_cols: bool) -> str:
  with open(template_fn, "r") as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)

  cols = list(Column)
  if only_tier_cols:
    hide_cols = set(StarColumns) - set(TierColumns)
    cols = [c for c in cols if c not in hide_cols]
    for car in all_car_info:
      for c in hide_cols:
        del car.row[c]

  footnotes = [fn.value.text for fn in get_all_footnotes(only_tier_cols)]
  cars_md: str = template.render(all_car_info=all_car_info, group_by_make=group_by_make,
                                 footnotes=footnotes, Star=Star, Column=cols, STAR_DESCRIPTIONS=STAR_DESCRIPTIONS)
  return cars_md


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supported cars documentation",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--tier-columns", action="store_true", help="Include only columns that count in the tier")
  parser.add_argument("--template", default=CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_all_car_info(args.tier_columns), args.template, args.tier_columns))
  print(f"Generated and written to {args.out}")
