#!/usr/bin/env python3
import json
import os
from collections import defaultdict

from openpilot.selfdrive.car.docs import get_all_car_docs

# Import helper functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from json_helper import deserialize_from_json

BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")


def print_car_docs_diff():
  prev_car_docs_path = os.path.join(BASEDIR, "prev_car_docs.json")
  if not os.path.exists(prev_car_docs_path):
    print("No previous car docs found, skipping comparison")
    return

  with open(prev_car_docs_path, "r") as f:
    prev_car_docs_raw = json.load(f)
    prev_car_docs = deserialize_from_json(prev_car_docs_raw)

  cur_car_docs = get_all_car_docs()

  changes = defaultdict(list)
  for car_model, car_info in cur_car_docs.items():
    if car_model not in prev_car_docs:
      changes["added"].append(car_model)
    elif car_info != prev_car_docs[car_model]:
      changes["modified"].append(car_model)

  for car_model in prev_car_docs:
    if car_model not in cur_car_docs:
      changes["removed"].append(car_model)

  if any(changes.values()):
    print("Car docs changed:")
    for change_type, car_models in changes.items():
      if car_models:
        print(f"  {change_type.capitalize()}:")
        for car_model in sorted(car_models):
          print(f"    - {car_model}")
  else:
    print("No changes to car docs")


if __name__ == "__main__":
  print_car_docs_diff()
