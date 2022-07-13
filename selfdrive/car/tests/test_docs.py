#!/usr/bin/env python3
import unittest

from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.docs import CARS_MD_OUT, CARS_MD_TEMPLATE, generate_cars_md, get_all_car_info
from selfdrive.car.docs_definitions import Column, Star


class TestCarDocs(unittest.TestCase):
  def setUp(self):
    self.all_cars = get_all_car_info()

  def test_generator(self):
    generated_cars_md = generate_cars_md(self.all_cars, CARS_MD_TEMPLATE)
    with open(CARS_MD_OUT, "r") as f:
      current_cars_md = f.read()

    self.assertEqual(generated_cars_md, current_cars_md,
                     "Run selfdrive/car/docs.py to update the compatibility documentation")

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
          self.assertNotIn("ev", tokens, "Use `Electric`")
          if "plug-in hybrid" in car.model.lower():
            self.assertIn("Plug-in Hybrid", car.model, "Use correct capitalization")
        elif car.car_name == "toyota":
          if "rav4" in tokens:
            self.assertIn("RAV4", car.model, "Use correct capitalization")

  def test_torque_star(self):
    # Asserts brand-specific assumptions around steering torque star
    for car in self.all_cars:
      with self.subTest(car=car):
        if car.car_name == "honda":
          self.assertIn(car.row[Column.STEERING_TORQUE], (Star.EMPTY, Star.HALF), f"{car.name} has full torque star")
        elif car.car_name in ("toyota", "hyundai"):
          self.assertNotEqual(car.row[Column.STEERING_TORQUE], Star.EMPTY, f"{car.name} has no torque star")


if __name__ == "__main__":
  unittest.main()
