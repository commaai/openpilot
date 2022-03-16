#!/usr/bin/env python3
from selfdrive.car.docs import CARS_MD_OUT, CARS_MD_TEMPLATE, generate_cars_md, get_tier_car_rows


def test_cars_docs():
  generated_cars_md = generate_cars_md(get_tier_car_rows(), CARS_MD_TEMPLATE)
  with open(CARS_MD_OUT, "r") as f:
    current_cars_md = f.read()

  assert generated_cars_md == current_cars_md, "Run selfdrive/car/docs.py to generate new supported car documentation"


if __name__ == "__main__":
  test_cars_docs()
