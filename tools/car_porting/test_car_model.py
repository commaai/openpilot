#!/usr/bin/env python3
import argparse
import sys
import unittest

from openpilot.selfdrive.car.tests.routes import CarTestRoute
from openpilot.selfdrive.car.tests.test_models import TestCarModel
from openpilot.tools.lib.route import SegmentName


def create_test_models_suite(routes: list[CarTestRoute], ci=False) -> unittest.TestSuite:
  test_suite = unittest.TestSuite()
  for test_route in routes:
    # create new test case and discover tests
    test_case_args = {"car_model": test_route.car_model, "test_route": test_route, "ci": ci}
    CarModelTestCase = type("CarModelTestCase", (TestCarModel,), test_case_args)
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(CarModelTestCase))
  return test_suite


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Test any route against common issues with a new car port. " +
                                               "Uses selfdrive/car/tests/test_models.py")
  parser.add_argument("route_or_segment_name", help="Specify route to run tests on")
  parser.add_argument("--car", help="Specify car model for test route")
  parser.add_argument("--ci", action="store_true", help="Attempt to get logs using openpilotci, need to specify car")
  args = parser.parse_args()
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

  route_or_segment_name = SegmentName(args.route_or_segment_name.strip(), allow_route_name=True)
  segment_num = route_or_segment_name.segment_num if route_or_segment_name.segment_num != -1 else None

  test_route = CarTestRoute(route_or_segment_name.route_name.canonical_name, args.car, segment=segment_num)
  test_suite = create_test_models_suite([test_route], ci=args.ci)

  unittest.TextTestRunner().run(test_suite)
