#!/usr/bin/env python3
import unittest

from selfdrive.car.docs import CARS_MD_OUT, CARS_MD_TEMPLATE, generate_cars_md, get_tier_car_info


class TestCarDocs(unittest.TestCase):
  def test_car_docs(self):
    generated_cars_md = generate_cars_md(get_tier_car_info(), CARS_MD_TEMPLATE)
    with open(CARS_MD_OUT, "r") as f:
      current_cars_md = f.read()

    self.assertEqual(generated_cars_md, current_cars_md,
                     "Run selfdrive/car/docs.py to generate new supported cars documentation")


if __name__ == "__main__":
  unittest.main()
