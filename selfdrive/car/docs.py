#!/usr/bin/env python3
import argparse
from collections import defaultdict
import jinja2
import os
from enum import Enum
from natsort import natsorted
from typing import cast
from tqdm import tqdm

from cereal import car
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car import gen_empty_fingerprint
from openpilot.selfdrive.car.tests.routes import routes
from openpilot.selfdrive.car.tests.test_models import TestCarModel, TestCarModelBase
from openpilot.selfdrive.car.docs_definitions import CarInfo, Column, CommonFootnote, PartType
from openpilot.selfdrive.car.car_helpers import interfaces, get_interface_attr
from openpilot.tools.lib.logreader import LogReader
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def get_all_footnotes() -> dict[Enum, int]:
  all_footnotes = list(CommonFootnote)
  for footnotes in get_interface_attr("Footnote", ignore_none=True).values():
    all_footnotes.extend(footnotes)
  return {fn: idx + 1 for idx, fn in enumerate(all_footnotes)}


CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")


def init_car_info_for_model(model: str, car_info: CarInfo | list[CarInfo]) -> list[CarInfo]:
  footnotes = get_all_footnotes()
  # If available, uses experimental longitudinal limits for the docs
  fingerprint = gen_empty_fingerprint()
  car_fw = [car.CarParams.CarFw(ecu="unknown")]

  # Use test route to consider live detected features if available
  test_route = next((rt for rt in routes if rt.car_model == model), None)
  if test_route is not None:
    test_case_args = {"car_model": test_route.car_model, "test_route": test_route}
    tcm = cast(TestCarModel, type("CarModelTestCase", (TestCarModel,), test_case_args))
    car_fw, _, _ = tcm.get_testing_data()
    # tcm.get_testing_data()
    fingerprint = tcm.fingerprint
    # remove DSU from FW versions so enableDsu=True
    # remove gas interceptor from the fingerprints so Honda doesn't show 0 mph enable speed
    fingerprint = {b: {a: l for a, l in f.items() if a != 0x201} for b, f in fingerprint.items()}
    car_fw = [fw for fw in car_fw if fw.ecu != 'dsu']

  CP = interfaces[model][0].get_params(model, fingerprint=fingerprint,
                                       car_fw=car_fw, experimental_long=True, docs=True)

  if CP.dashcamOnly or car_info is None:
    return []

  # A platform can include multiple car models
  if not isinstance(car_info, list):
    car_info = (car_info,)

  for _car_info in car_info:
    if not hasattr(_car_info, "row"):
      _car_info.init_make(CP)
      _car_info.init(CP, footnotes)

  return car_info


def get_all_car_info() -> list[CarInfo]:
  """
  This function uses the CAN fingerprints and FW from each make
  to generate accurate car docs considering live detected features
  """

  all_car_info: list[CarInfo] = []
  with ProcessPoolExecutor(max_workers=24) as executor:
    futures = []
    for model, car_info in get_interface_attr("CAR_INFO", combine_brands=True).items():
      futures.append(executor.submit(init_car_info_for_model, model, car_info))

    for future in tqdm(as_completed(futures), total=len(futures)):
      all_car_info.extend(future.result())

  # Sort cars by make and model + year
  sorted_cars: list[CarInfo] = natsorted(all_car_info, key=lambda car: car.name.lower())
  return sorted_cars


def group_by_make(all_car_info: list[CarInfo]) -> dict[str, list[CarInfo]]:
  sorted_car_info = defaultdict(list)
  for car_info in all_car_info:
    sorted_car_info[car_info.make].append(car_info)
  return dict(sorted_car_info)


def generate_cars_md(all_car_info: list[CarInfo], template_fn: str) -> str:
  with open(template_fn) as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)

  footnotes = [fn.value.text for fn in get_all_footnotes()]
  cars_md: str = template.render(all_car_info=all_car_info, PartType=PartType,
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
    f.write(generate_cars_md(get_all_car_info(), args.template))
  print(f"Generated and written to {args.out}")
