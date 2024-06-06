#!/usr/bin/env python3
import argparse
from collections import defaultdict
import jinja2
import os
from enum import Enum
from natsort import natsorted

from cereal import car
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car import gen_empty_fingerprint
from openpilot.selfdrive.car.docs_definitions import CarDocs, Column, CommonFootnote, PartType
from openpilot.selfdrive.car.car_helpers import interfaces, get_interface_attr
from openpilot.selfdrive.car.values import PLATFORMS


def get_all_footnotes() -> dict[Enum, int]:
  all_footnotes = list(CommonFootnote)
  for footnotes in get_interface_attr("Footnote", ignore_none=True).values():
    all_footnotes.extend(footnotes)
  return {fn: idx + 1 for idx, fn in enumerate(all_footnotes)}


CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")


def get_all_car_docs() -> list[CarDocs]:
  all_car_docs: list[CarDocs] = []
  footnotes = get_all_footnotes()
  for model, platform in PLATFORMS.items():
    car_docs = platform.config.car_docs
    # If available, uses experimental longitudinal limits for the docs
    CP = interfaces[model][0].get_params(platform, fingerprint=gen_empty_fingerprint(),
                                         car_fw=[car.CarParams.CarFw(ecu="unknown")], experimental_long=True, docs=True)

    if CP.dashcamOnly or not len(car_docs):
      continue

    # A platform can include multiple car models
    for _car_docs in car_docs:
      if not hasattr(_car_docs, "row"):
        _car_docs.init_make(CP)
        _car_docs.init(CP, footnotes)
      all_car_docs.append(_car_docs)

  # Sort cars by make and model + year
  sorted_cars: list[CarDocs] = natsorted(all_car_docs, key=lambda car: car.name.lower())
  return sorted_cars


def group_by_make(all_car_docs: list[CarDocs]) -> dict[str, list[CarDocs]]:
  sorted_car_docs = defaultdict(list)
  for car_docs in all_car_docs:
    sorted_car_docs[car_docs.make].append(car_docs)
  return dict(sorted_car_docs)


def generate_cars_md(all_car_docs: list[CarDocs], template_fn: str) -> str:
  with open(template_fn) as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)

  footnotes = [fn.value.text for fn in get_all_footnotes()]
  cars_md: str = template.render(all_car_docs=all_car_docs, PartType=PartType,
                                 group_by_make=group_by_make, footnotes=footnotes,
                                 Column=Column)
  return cars_md


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supported cars documentation",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--template", default=CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_all_car_docs(), args.template))
  print(f"Generated and written to {args.out}")
