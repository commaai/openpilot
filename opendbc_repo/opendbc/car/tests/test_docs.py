from collections import defaultdict
import pytest

from opendbc.car.car_helpers import interfaces
from opendbc.car.docs import get_all_car_docs
from opendbc.car.docs_definitions import Cable, Column, PartType, Star, SupportType
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.values import PLATFORMS


class TestCarDocs:
  @classmethod
  def setup_class(cls):
    cls.all_cars = get_all_car_docs()

  def test_duplicate_years(self, subtests):
    make_model_years = defaultdict(list)
    for car in self.all_cars:
      with subtests.test(car_docs_name=car.name):
        if car.support_type != SupportType.UPSTREAM:
          pytest.skip()

        make_model = (car.make, car.model)
        for year in car.year_list:
          assert year not in make_model_years[make_model], f"{car.name}: Duplicate model year"
          make_model_years[make_model].append(year)

  def test_missing_car_docs(self, subtests):
    all_car_docs_platforms = [name for name, config in PLATFORMS.items()]
    for platform in sorted(interfaces.keys()):
      with subtests.test(platform=platform):
        assert platform in all_car_docs_platforms, f"Platform: {platform} doesn't have a CarDocs entry"

  def test_naming_conventions(self, subtests):
    # Asserts market-standard car naming conventions by brand
    for car in self.all_cars:
      with subtests.test(car=car.name):
        tokens = car.model.lower().split(" ")
        if car.brand == "hyundai":
          assert "phev" not in tokens, "Use `Plug-in Hybrid`"
          assert "hev" not in tokens, "Use `Hybrid`"
          if "plug-in hybrid" in car.model.lower():
            assert "Plug-in Hybrid" in car.model, "Use correct capitalization"
          if car.make != "Kia":
            assert "ev" not in tokens, "Use `Electric`"
        elif car.brand == "toyota":
          if "rav4" in tokens:
            assert "RAV4" in car.model, "Use correct capitalization"

  def test_torque_star(self, subtests):
    # Asserts brand-specific assumptions around steering torque star
    for car in self.all_cars:
      with subtests.test(car=car.name):
        # honda sanity check, it's the definition of a no torque star
        if car.car_fingerprint in (HONDA.HONDA_ACCORD, HONDA.HONDA_CIVIC, HONDA.HONDA_CRV, HONDA.HONDA_ODYSSEY, HONDA.HONDA_PILOT):
          assert car.row[Column.STEERING_TORQUE] == Star.EMPTY, f"{car.name} has full torque star"
        elif car.brand in ("toyota", "hyundai"):
          assert car.row[Column.STEERING_TORQUE] != Star.EMPTY, f"{car.name} has no torque star"

  def test_year_format(self, subtests):
    for car in self.all_cars:
      with subtests.test(car=car.name):
        if car.name == "comma body":
          pytest.skip()

        assert car.years and car.year_list, f"Format years correctly: {car.name}"

  def test_harnesses(self, subtests):
    for car in self.all_cars:
      with subtests.test(car=car.name):
        if car.name == "comma body" or car.support_type != SupportType.UPSTREAM:
          pytest.skip()

        car_part_type = [p.part_type for p in car.car_parts.all_parts()]
        car_parts = list(car.car_parts.all_parts())
        assert len(car_parts) > 0, f"Need to specify car parts: {car.name}"
        assert car_part_type.count(PartType.connector) == 1, f"Need to specify one harness connector: {car.name}"
        assert car_part_type.count(PartType.mount) == 1, f"Need to specify one mount: {car.name}"
        assert Cable.right_angle_obd_c_cable_1_5ft in car_parts, f"Need to specify a right angle OBD-C cable (1.5ft): {car.name}"
