#!/usr/bin/env python3
import unittest

from selfdrive.car.docs import CARS_MD_OUT, CARS_MD_TEMPLATE, generate_cars_md, get_tier_car_info


class TestCarDocs(unittest.TestCase):
  def setUp(self):
    self.tier_cars = get_tier_car_info()

  def test_generator(self):
    generated_cars_md = generate_cars_md(self.tier_cars, CARS_MD_TEMPLATE)
    with open(CARS_MD_OUT, "r") as f:
      current_cars_md = f.read()

    self.assertEqual(generated_cars_md, current_cars_md,
                     "Run selfdrive/car/docs.py to generate new supported cars documentation")

  def test_naming_conventions(self):
    # Asserts market-standard car naming conventions by make
    for cars in self.tier_cars.values():
      for car in cars:
        if car.car_name == "hyundai":
          tokens = car.model.lower().split(" ")
          self.assertNotIn("phev", tokens, "Use `Plug-in Hybrid`")
          self.assertNotIn("hev", tokens, "Use `Hybrid`")
          self.assertNotIn("ev", tokens, "Use `Electric`")
          if "plug-in hybrid" in car.model.lower():
            self.assertIn("Plug-in Hybrid", car.model, "Use correct capitalization")


if __name__ == "__main__":
  unittest.main()
