#!/usr/bin/env python3
import argparse
import json

from openpilot.selfdrive.car.values import create_platform_map


def generate_dbc_json() -> str:
  dbc_map = create_platform_map(lambda platform: platform.config.dbc_dict["pt"] if platform != "mock" else None)
  return json.dumps(dict(sorted(dbc_map.items())), indent=2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate mapping for all car fingerprints to DBC names and outputs json file",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--out", required=True, help="Generated json filepath")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_dbc_json())
  print(f"Generated and written to {args.out}")
