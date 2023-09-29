#!/usr/bin/env python3
import os

from cereal import car
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car import gen_empty_fingerprint
from openpilot.selfdrive.car.car_helpers import interfaces, get_interface_attr
from openpilot.selfdrive.car.docs import get_all_car_info, generate_cars_md
from openpilot.selfdrive.car.docs_definitions import CarInfo, Column, CommonFootnote, PartType

CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")

PORTS_MD_OUT = os.path.join(BASEDIR, "docs", "PORTS.md")
PORTS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "PORTS_template.md")


if __name__ == "__main__":
  with open(CARS_MD_OUT, 'w') as f:
    f.write(generate_cars_md(get_all_car_info(dashcam_only=False), CARS_MD_TEMPLATE))

  with open(PORTS_MD_OUT, 'w') as f:
    f.write(generate_cars_md(get_all_car_info(dashcam_only=True), PORTS_MD_TEMPLATE))
