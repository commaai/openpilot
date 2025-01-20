#!/usr/bin/env python3
import argparse
import json

from opendbc.car import Bus
from opendbc.car.fingerprints import MIGRATION
from opendbc.car.values import PLATFORMS


def generate_dbc_json() -> str:
  dbc_map = {}
  for platform in PLATFORMS.values():
    if platform != "MOCK":
      if Bus.pt in platform.config.dbc_dict:
        dbc_map[platform.name] = platform.config.dbc_dict[Bus.pt]
      elif Bus.main in platform.config.dbc_dict:
        dbc_map[platform.name] = platform.config.dbc_dict[Bus.main]
      elif Bus.party in platform.config.dbc_dict:
        dbc_map[platform.name] = platform.config.dbc_dict[Bus.party]
      else:
        raise ValueError("Unknown main type")

  for m in MIGRATION:
    if MIGRATION[m] in dbc_map:
      dbc_map[m] = dbc_map[MIGRATION[m]]

  return json.dumps(dict(sorted(dbc_map.items())), indent=2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate mapping for all car fingerprints to DBC names and outputs json file",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--out", required=True, help="Generated json filepath")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_dbc_json())
  print(f"Generated and written to {args.out}")
