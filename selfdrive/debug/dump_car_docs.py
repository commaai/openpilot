#!/usr/bin/env python3
import argparse

import os
import requests

def dump_car_docs(path):
  car_docs_url = "https://raw.githubusercontent.com/commaai/openpilot/master/docs/CARS.md"
  output_path = os.path.join(path, "CARS.md")
  response = requests.get(car_docs_url)
  with open(output_path, "wb") as file:
    file.write(response.content)
  print(f'Dumping car info to {path}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  args = parser.parse_args()
  dump_car_docs(args.path)
