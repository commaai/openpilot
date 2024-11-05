#!/usr/bin/env python3
import argparse
import sys
import unittest # noqa: TID251

from opendbc.car.tests.routes import CarTestRoute
from openpilot.selfdrive.car.tests.test_models import TestCarModel
from openpilot.tools.lib.route import SegmentRange


def create_test_models_suite(routes: list[CarTestRoute]) -> unittest.TestSuite:
  test_suite = unittest.TestSuite()
  for test_route in routes:
    # create new test case and discover tests
    test_case_args = {"platform": test_route.car_model, "test_route": test_route}
    CarModelTestCase = type("CarModelTestCase", (TestCarModel,), test_case_args)
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(CarModelTestCase))
  return test_suite


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Test any route against common issues with a new car port. " +
                                               "Uses selfdrive/car/tests/test_models.py")
  parser.add_argument("route_or_segment_name", help="Specify route to run tests on")
  parser.add_argument("--car", help="Specify car model for test route")
  args = parser.parse_args()
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

  sr = SegmentRange(args.route_or_segment_name)

  test_routes = [CarTestRoute(sr.route_name, args.car, segment=seg_idx) for seg_idx in sr.seg_idxs]
  test_suite = create_test_models_suite(test_routes)

  unittest.TextTestRunner().run(test_suite)
