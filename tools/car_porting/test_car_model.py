#!/usr/bin/env python3
import argparse
import sys

from opendbc.car.tests.routes import CarTestRoute
from openpilot.selfdrive.car.tests.test_models import TestCarModel
from openpilot.tools.lib.route import SegmentRange


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

  # Create dynamic test classes in the global scope so pytest can find them
  for seg_idx in sr.seg_idxs:
    test_route = CarTestRoute(sr.route_name, args.car, segment=seg_idx)
    test_case_args = {"platform": test_route.car_model, "test_route": test_route}
    class_name = f"TestCarModel_{test_route.car_model}_{seg_idx}".replace("|", "_").replace("-", "_").replace("/", "_")
    globals()[class_name] = type(class_name, (TestCarModel,), test_case_args)

  import subprocess
  subprocess.run([sys.executable, "-m", "pytest", __file__, *sys.argv[2:]], check=True)
