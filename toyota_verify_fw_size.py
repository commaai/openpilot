#!/usr/bin/env python3
from selfdrive.car.toyota.values import FW_VERSIONS

for car, versions in FW_VERSIONS.items():
  for ecu_type, fws in versions.items():
    for fw in fws:
      expected_byte_length = (fw[0] if fw[0] <= 3 else 1) * 16
      new_fw = fw[1:] if fw[0] <= 3 else fw
      if len(new_fw) != expected_byte_length:
        print(fw, len(fw), expected_byte_length)
