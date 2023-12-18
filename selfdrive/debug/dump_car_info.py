#!/usr/bin/env python3
import argparse
import pickle

from openpilot.selfdrive.car.docs import get_all_car_info


def dump_car_info(path):
  with open(path, 'wb') as f:
    pickle.dump(get_all_car_info(), f)
  print(f'Dumping car info to {path}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  args = parser.parse_args()
  dump_car_info(args.path)
