#!/usr/bin/env python3
import os
import pytest
from parameterized import parameterized_class
from typing import Any

from openpilot.selfdrive.car.tests.routes import CarTestRoute
from openpilot.selfdrive.car.tests.test_models import TestCarModelBase
from openpilot.tools.lib.route import SegmentName

route = os.getenv('ROUTE')
car: Any = os.getenv('CAR')

assert route is not None, "Must specify route or segment name"

route_or_segment_name = SegmentName(route, allow_route_name=True)
segment_num = route_or_segment_name.segment_num if route_or_segment_name.segment_num != -1 else None
test_route = CarTestRoute(route_or_segment_name.route_name.canonical_name, car, segment=segment_num)

@parameterized_class(('platform', 'test_route'), [(test_route.car_model, test_route)])
@pytest.mark.xdist_group_class_property('test_route')
class TestCarModel(TestCarModelBase):
  pass
