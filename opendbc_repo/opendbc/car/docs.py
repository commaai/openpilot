#!/usr/bin/env python3
import argparse
import os
from typing import get_args

from collections import defaultdict
import jinja2
from enum import Enum
from natsort import natsorted

from opendbc.car.common.basedir import BASEDIR
from opendbc.car import gen_empty_fingerprint
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import BaseCarHarness, CarDocs, Device, ExtraCarDocs, Column, ExtraCarsColumn, CommonFootnote, PartType, SupportType
from opendbc.car.car_helpers import interfaces
from opendbc.car.interfaces import get_interface_attr
from opendbc.car.values import Platform
from opendbc.car.mock.values import CAR as MOCK
from opendbc.car.extra_cars import CAR as EXTRA


EXTRA_CARS_MD_OUT = os.path.join(BASEDIR, "../", "../", "docs", "CARS.md")
EXTRA_CARS_MD_TEMPLATE = os.path.join(BASEDIR, "CARS_template.md")

# TODO: merge these platforms into normal car ports with SupportType flag
ExtraPlatform = Platform | EXTRA
EXTRA_BRANDS = get_args(ExtraPlatform)
EXTRA_PLATFORMS: dict[str, ExtraPlatform] = {str(platform): platform for brand in EXTRA_BRANDS for platform in brand}


def get_params_for_docs(platform) -> CarParams:
  cp_platform = platform if platform in interfaces else MOCK.MOCK
  CP: CarParams = interfaces[cp_platform].get_params(cp_platform, fingerprint=gen_empty_fingerprint(),
                                                     car_fw=[CarParams.CarFw(ecu=CarParams.Ecu.unknown)],
                                                     alpha_long=True, is_release=False, docs=True)
  return CP


def get_all_footnotes() -> dict[Enum, int]:
  all_footnotes = list(CommonFootnote)
  for footnotes in get_interface_attr("Footnote", ignore_none=True).values():
    all_footnotes.extend(footnotes)
  return {fn: idx + 1 for idx, fn in enumerate(all_footnotes)}


def build_sorted_car_docs_list(platforms, footnotes=None):
  collected_car_docs: list[CarDocs | ExtraCarDocs] = []
  for platform in platforms.values():
    car_docs = platform.config.car_docs
    CP = get_params_for_docs(platform)

    if not len(car_docs):
      continue

    # A platform can include multiple car models
    for _car_docs in car_docs:
      if not hasattr(_car_docs, "row"):
        _car_docs.init_make(CP)
        _car_docs.init(CP, footnotes)
      collected_car_docs.append(_car_docs)

  # Sort cars by make and model + year
  sorted_cars = natsorted(collected_car_docs, key=lambda car: car.name.lower())
  return sorted_cars


# CAUTION: This function is imported by shop.comma.ai and comma.ai/vehicles, test changes carefully
def get_all_car_docs() -> list[CarDocs]:
  collected_footnotes = get_all_footnotes()
  sorted_list: list[CarDocs] = build_sorted_car_docs_list(EXTRA_PLATFORMS, footnotes=collected_footnotes)
  return sorted_list


def group_by_make(all_car_docs: list[CarDocs]) -> dict[str, list[CarDocs]]:
  sorted_car_docs = defaultdict(list)
  for car_docs in all_car_docs:
    sorted_car_docs[car_docs.make].append(car_docs)
  return dict(sorted_car_docs)


# CAUTION: This function is imported by shop.comma.ai and comma.ai/vehicles, test changes carefully
def generate_cars_md(all_car_docs: list[CarDocs], template_fn: str, **kwargs) -> str:
  with open(template_fn) as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)

  footnotes = [fn.value.text for fn in get_all_footnotes()]
  cars_md: str = template.render(all_car_docs=all_car_docs, PartType=PartType,
                                 group_by_make=group_by_make, footnotes=footnotes,
                                 Device=Device, Column=Column, ExtraCarsColumn=ExtraCarsColumn,
                                 BaseCarHarness=BaseCarHarness, SupportType=SupportType,
                                 **kwargs)
  return cars_md


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supportability info docs for all known cars",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--template", default=EXTRA_CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=EXTRA_CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_all_car_docs(), args.template))
  print(f"Generated and written to {args.out}")
