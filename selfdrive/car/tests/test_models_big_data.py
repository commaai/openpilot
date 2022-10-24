#!/usr/bin/env python
import unittest
from parameterized import parameterized_class

from selfdrive.car.tests.routes import CarTestRoute
from selfdrive.car.tests.test_models import TestCarModelBase
from xx.pipeline.route_lists.make_test_models_seg_list import SEG_LIST_PATH

with open(SEG_LIST_PATH, "r") as f:
  seg_list = f.read().splitlines()
platforms, segs = seg_list[0::2], seg_list[1::2]

car_test_routes = []
for platform, seg in list(zip(platforms, segs)):
  car_test_routes.append(CarTestRoute(seg[:37], platform[2:], segment=int(seg[39:])))


@parameterized_class(('car_model', 'test_route'), [(test_route.car_model, test_route) for test_route in car_test_routes])
class TestCarModel(TestCarModelBase):
  pass


if __name__ == "__main__":
  unittest.main()
