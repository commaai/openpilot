#!/usr/bin/env python3
import argparse
import json

from openpilot.selfdrive.car.car_helpers import get_interface_attr


def generate_dbc_json() -> str:
  all_cars_by_brand = get_interface_attr("CAR_INFO")
  all_dbcs_by_brand = get_interface_attr("DBC")
  dbc_map = {car: all_dbcs_by_brand[brand][car]['pt'] for brand, cars in all_cars_by_brand.items() for car in cars if car != 'mock'}
  return json.dumps(dict(sorted(dbc_map.items())), indent=2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate mapping for all car fingerprints to DBC names and outputs json file",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--out", required=True, help="Generated json filepath")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_dbc_json())
  print(f"Generated and written to {args.out}")
