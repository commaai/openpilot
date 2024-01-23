#!/usr/bin/env python3
from typing import NamedTuple, Optional
from openpilot.common.basedir import BASEDIR

from openpilot.selfdrive.car.gm.values import CAR as GM
from openpilot.selfdrive.car.ford.values import CAR as FORD
from openpilot.selfdrive.car.honda.values import CAR as HONDA
from openpilot.selfdrive.car.hyundai.values import CAR as HYUNDAI
from openpilot.selfdrive.car.subaru.values import CAR as SUBARU
from openpilot.selfdrive.car.volkswagen.values import CAR as VOLKSWAGEN
from openpilot.selfdrive.test.helpers import read_segment_list
from openpilot.tools.lib.route import SegmentRange

# TODO: add routes for these cars
non_tested_cars = [
  FORD.F_150_MK14,
  GM.CADILLAC_ATS,
  GM.HOLDEN_ASTRA,
  GM.MALIBU,
  GM.EQUINOX,
  HYUNDAI.GENESIS_G90,
  HONDA.ODYSSEY_CHN,
  VOLKSWAGEN.CRAFTER_MK2,  # need a route from an ACC-equipped Crafter
  SUBARU.FORESTER_HYBRID,
]


class CarTestRoute(NamedTuple):
  segment_range: str
  car_model: Optional[str]


routes = [CarTestRoute(str(SegmentRange(segment)), car_model) for car_model, segment in read_segment_list(f"{BASEDIR}/selfdrive/car/tests/routes.txt")] + \
         [CarTestRoute(str(SegmentRange(segment)), car_model) for car_model, segment in read_segment_list(f"{BASEDIR}/selfdrive/car/tests/cases.txt")]
