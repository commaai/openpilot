#!/usr/bin/env python3

import argparse
from cereal import car

from openpilot.tools.lib.logreader import MultiLogIterator
from openpilot.tools.lib.route import Route
from selfdrive.car.fw_versions import match_fw_to_car
from selfdrive.car.interfaces import get_interface_attr


ALL_FW_VERSIONS = get_interface_attr("FW_VERSIONS")
ALL_CARS = get_interface_attr("CAR")

PYTHON_CAR_NAME_TO_PLATFORM = {brand: {m: v for v, m in vars(ALL_CARS[brand]).items() if not (v.startswith('_')  or callable(m))} for brand in ALL_CARS}
ECU_LOOKUP = {m: v for v, m in vars(car.CarParams.Ecu).items() if not (v.startswith('_')  or callable(m) or v.startswith("schema"))}
REVERSE_ECU_LOOKUP = {v: k for k, v in ECU_LOOKUP.items()}

def format_fw_versions(brand, fw_versions):
  ret = ""

  ret += "FW_VERSIONS = {\n"
  for platform in fw_versions:
    ret += f"  CAR.{PYTHON_CAR_NAME_TO_PLATFORM[brand][platform]}: {{\n"
    for key in fw_versions[platform]:
        (ecu, addr, subAddr) = key
        ret += f"    (Ecu.{ECU_LOOKUP[ecu]}, {hex(addr)}, {None if subAddr == 0 else subAddr}): [\n"
        for version in fw_versions[platform][key]:
          ret += f"      {version},\n"
        ret += "    ],\n"
    ret += "  },\n"
  ret += "}"

  return ret

def write_fw_versions(brand, fw_versions):
  filename = f"selfdrive/car/{brand}/values.py"
  with open(filename, "r") as f:
    values_py = f.read()

  start = values_py.index("FW_VERSIONS = {")

  end_str = "},\n}\n"
  try:
    end = values_py.index(end_str)
  except:
    end_str = "}\n}\n"
    end = values_py.index(end_str)

  new_values_py = values_py[0:start] + format_fw_versions(brand, fw_versions) + "\n" + values_py[end+len(end_str):]
  
  with open(filename, "w") as f:
    f.write(new_values_py)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto fingerprint from a route")
  parser.add_argument("route", help="The route name to use")
  parser.add_argument("platform", help="The platform, or leave empty to auto-determine using fuzzy", default=None, nargs='?')
  args = parser.parse_args()

  route = Route(args.route)
  lr = MultiLogIterator(route.qlog_paths())

  carFw = None
  carVin = None

  for msg in lr:
    if msg.which() == "carParams":
      carFw = msg.carParams.carFw
      carVin = msg.carParams.carVin
      break
  
  if args.platform is None: # attempt to auto-determine platform with other fuzzy fingerprints
    _, possible_platforms = match_fw_to_car(carFw)

    if len(possible_platforms) != 1:
      raise Exception(f"Unable to auto-determine platform, possible platforms: {possible_platforms}")
    
    platform = list(possible_platforms)[0]
  else:
    platform = args.platform

  print("Adding fingerprint for: ", platform)

  all_cars_by_brand = get_interface_attr("CAR_INFO")
  brand = None
  for b in all_cars_by_brand:
    if platform in all_cars_by_brand[b]:
      brand = b
  
  for fw in carFw:
    if fw.brand == brand:
      ecu = REVERSE_ECU_LOOKUP[fw.ecu]
      addr = fw.address
      subAddr = None if fw.subAddress == 0 else fw.subAddress
      key = (ecu, addr, subAddr)

      if key in ALL_FW_VERSIONS[brand][platform]:
        fw_versions = set(ALL_FW_VERSIONS[brand][platform][key])
        fw_versions.add(fw.fwVersion)
        ALL_FW_VERSIONS[brand][platform][key] = list(fw_versions)

  write_fw_versions(brand, ALL_FW_VERSIONS[brand])