#!/usr/bin/env python3
from collections import defaultdict
import re
import unittest

from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.docs import CARS_MD_OUT, CARS_MD_TEMPLATE, generate_cars_md, get_all_car_info
from selfdrive.car.docs_definitions import Column, Harness, Star
from selfdrive.car.honda.values import CAR as HONDA


class TestCarDocs(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.all_cars = get_all_car_info()

  def test_generator(self):
    generated_cars_md = generate_cars_md(self.all_cars, CARS_MD_TEMPLATE)
    with open(CARS_MD_OUT, "r") as f:
      current_cars_md = f.read()

    self.assertEqual(generated_cars_md, current_cars_md,
                     "Run selfdrive/car/docs.py to update the compatibility documentation")

  def test_duplicate_years(self):
    make_model_years = defaultdict(list)
    for car in self.all_cars:
      with self.subTest(car_info_name=car.name):
        make_model = (car.make, car.model)
        for year in car.year_list:
          self.assertNotIn(year, make_model_years[make_model], f"{car.name}: Duplicate model year")
          make_model_years[make_model].append(year)

  def test_missing_car_info(self):
    all_car_info_platforms = get_interface_attr("CAR_INFO", combine_brands=True).keys()
    for platform in sorted(interfaces.keys()):
      with self.subTest(platform=platform):
        self.assertTrue(platform in all_car_info_platforms, "Platform: {} doesn't exist in CarInfo".format(platform))

  def test_naming_conventions(self):
    # Asserts market-standard car naming conventions by brand
    for car in self.all_cars:
      with self.subTest(car=car):
        tokens = car.model.lower().split(" ")
        if car.car_name == "hyundai":
          self.assertNotIn("phev", tokens, "Use `Plug-in Hybrid`")
          self.assertNotIn("hev", tokens, "Use `Hybrid`")
          if "plug-in hybrid" in car.model.lower():
            self.assertIn("Plug-in Hybrid", car.model, "Use correct capitalization")
          if car.make != "Kia":
            self.assertNotIn("ev", tokens, "Use `Electric`")
        elif car.car_name == "toyota":
          if "rav4" in tokens:
            self.assertIn("RAV4", car.model, "Use correct capitalization")

  def test_torque_star(self):
    # Asserts brand-specific assumptions around steering torque star
    for car in self.all_cars:
      with self.subTest(car=car):
        # honda sanity check, it's the definition of a no torque star
        if car.car_fingerprint in (HONDA.ACCORD, HONDA.CIVIC, HONDA.CRV, HONDA.ODYSSEY, HONDA.PILOT):
          self.assertEqual(car.row[Column.STEERING_TORQUE], Star.EMPTY, f"{car.name} has full torque star")
        elif car.car_name in ("toyota", "hyundai"):
          self.assertNotEqual(car.row[Column.STEERING_TORQUE], Star.EMPTY, f"{car.name} has no torque star")

  def test_year_format(self):
    for car in self.all_cars:
      with self.subTest(car=car):
        self.assertIsNone(re.search(r"\d{4}-\d{4}", car.name), f"Format years correctly: {car.name}")

  def test_harnesses(self):
    for car in self.all_cars:
      with self.subTest(car=car):
        if car.name == "comma body":
          raise unittest.SkipTest

        self.assertNotIn(car.harness, [None, Harness.none], f"Need to specify car harness: {car.name}")


if __name__ == "__main__":
  unittest.main()
