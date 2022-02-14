#!/usr/bin/env python3
from enum import Enum

from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.config import Conversions as CV
import locale
locale.setlocale(locale.LC_ALL, '')

class Tier(Enum):
  GOLD = "Gold"
  SILVER = "Silver"
  BRONZE = "Bronze"


class Car:
  def __init__(self, CP):
    self.CP = CP

    make, model = self.CP.carFingerprint.split(' ', 1)
    self.make = make.capitalize()
    self.model = model.title()

    self.package = None

  @property
  def tier(self):
    if self.CP.openpilotLongitudinalControl:
      return Tier.GOLD
    return Tier.BRONZE

  def __str__(self):
    min_alc = int(max(0, self.CP.minSteerSpeed) * CV.MS_TO_MPH)
    min_acc = int(max(0, self.CP.minEnableSpeed) * CV.MS_TO_MPH)
    acc = "openpilot" if self.CP.openpilotLongitudinalControl else "Stock"
    return f"| {self.make} | {self.model} | {self.package} | {acc} | {min_acc}mph | {min_alc}mph |"


if __name__ == "__main__":
  print("# Supported Cars")

  print("Cars are sorted into three tiers:\n")
  print("  Gold - a full openpilot experience\n")
  print("  Silver - a pretty good, albeit limited experience\n")
  print("  Bronze - signifincatly limited\n")

  for t in Tier:
    print(f"## {t.value} Cars")
    print("| Make      | Model (US Market Reference)   | Supported Package | ACC              | No ACC accel below | No ALC below      |")
    print("| ----------| ------------------------------| ------------------| -----------------| -------------------| ------------------|")
    for c in sorted(all_known_cars(), key=locale.strxfrm):
      CI, _, _ = interfaces[c]
      CP = CI.get_params(c, {0: [], 1: [], 2: []}, [])
      car = Car(CP)
      if car.tier == t:
        print(car)
