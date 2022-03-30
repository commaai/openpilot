#!/usr/bin/env python3
from collections import Counter
from pprint import pprint

from selfdrive.car.docs import get_tier_car_info

if __name__ == "__main__":
  tiers = get_tier_car_info()
  cars = [car for tier_cars in tiers.values() for car in tier_cars]

  make_count = Counter(l.make for l in cars)
  print("\n", "*" * 20, len(cars), "total", "*" * 20, "\n")
  pprint(make_count)
