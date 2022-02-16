#!/usr/bin/env python3
from enum import Enum

from selfdrive.car import CarInfo
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import get_attr_from_cars, all_known_cars
from selfdrive.config import Conversions as CV


class Tier(Enum):
  GOLD = "Gold"
  SILVER = "Silver"
  BRONZE = "Bronze"


class Car:
  def __init__(self, CP, car_info):
    self.CP = CP
    self.info = car_info
    self.make, self.model = self.info.name.split(' ', 1)

  @property
  def tier(self):
    if self.CP.openpilotLongitudinalControl:
      return Tier.GOLD
    return Tier.BRONZE

  @property
  def years(self):
    years =
    for year in self.info.years:


  def __str__(self):
    min_alc = int(max(0, self.CP.minSteerSpeed) * CV.MS_TO_MPH)
    min_acc = int(max(0, self.CP.minEnableSpeed) * CV.MS_TO_MPH)
    acc = "openpilot" if self.CP.openpilotLongitudinalControl else "Stock"
    years = {}
    return f"| {self.make} | {self.model} | {self.info.package} | {acc} | {min_acc}mph | {min_alc}mph |"


if __name__ == "__main__":
  print("# Supported Cars")

  print("Cars are sorted into three tiers:\n")
  print("  Gold - a full openpilot experience\n")
  print("  Silver - a pretty good, albeit limited experience\n")
  print("  Bronze - significantly limited\n")

  for t in Tier:
    print(f"## {t.value} Cars")
    print("| Make      | Model (US Market Reference)   | Supported Package | ACC              | No ACC accel below | No ALC below      |")
    print("| ----------| ------------------------------| ------------------| -----------------| -------------------| ------------------|")
    for fingerprint, car_info in sorted(get_attr_from_cars('CAR_INFO').items(), key=lambda x: x[0]):
      # print(car_info)
      CI, _, _ = interfaces[fingerprint]
      CP = CI.get_params(fingerprint)

      # Skip community supported
      if CP.dashcamOnly:
        continue

      if isinstance(car_info, CarInfo):
        car_info = (car_info,)

      for info in car_info:
        car = Car(CP, info)
        if car.tier == t:
          print(car)
