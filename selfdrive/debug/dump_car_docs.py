#!/usr/bin/env python3
import argparse
import os
import requests

LINK_TO_MAIN_CARS = "https://raw.githubusercontent.com/commaai/openpilot/master/docs/CARS.md"

def dump_car_docs(path, url=LINK_TO_MAIN_CARS):
  response = requests.get(url)
  MASTER_CARS_MD = os.path.join(path, "CARS.md")
  with open(MASTER_CARS_MD, 'wb') as file:
    file.write(response.content)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  parser.add_argument("--url", required=False, default=LINK_TO_MAIN_CARS)
  args = parser.parse_args()
  dump_car_docs(args.path, args.url)
