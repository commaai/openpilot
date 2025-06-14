import pytest

from opendbc.car.values import PLATFORMS
from opendbc.car.tests.routes import non_tested_cars, routes


@pytest.mark.parametrize("platform", PLATFORMS.keys())
def test_test_route_present(platform):
  tested_platforms = [r.car_model for r in routes]
  assert platform in set(tested_platforms) | set(non_tested_cars), \
    f"Missing test route for {platform}. Add a route to opendbc/car/tests/routes.py"
