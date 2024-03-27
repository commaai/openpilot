#!/usr/bin/env python3
"""Some Hyundai radars can be reconfigured to output (debug) radar points on bus 1.
Reconfiguration is done over UDS by reading/writing to 0x0142 using the Read/Write Data By Identifier
endpoints (0x22 & 0x2E). This script checks your radar firmware version against a list of known
firmware versions. If you want to try on a new radar make sure to note the default config value
in case it's different from the other radars and you need to revert the changes.

After changing the config the car should not show any faults when openpilot is not running.
These config changes are persistent across car reboots. You need to run this script again
to go back to the default values.

USE AT YOUR OWN RISK! Safety features, like AEB and FCW, might be affected by these changes."""

import sys
import argparse
from typing import NamedTuple
from subprocess import check_output, CalledProcessError

from panda.python import Panda
from panda.python.uds import UdsClient, SESSION_TYPE, DATA_IDENTIFIER_TYPE

class ConfigValues(NamedTuple):
  default_config: bytes
  tracks_enabled: bytes

# If your radar supports changing data identifier 0x0142 as well make a PR to
# this file to add your firmware version. Make sure to post a drive as proof!
# NOTE: these firmware versions do not match what openpilot uses
#       because this script uses a different diagnostic session type
SUPPORTED_FW_VERSIONS = {
  # 2020 SONATA
  b"DN8_ SCC FHCUP      1.00 1.00 99110-L0000\x19\x08)\x15T    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  b"DN8_ SCC F-CUP      1.00 1.00 99110-L0000\x19\x08)\x15T    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  # 2021 SONATA HYBRID
  b"DNhe SCC FHCUP      1.00 1.00 99110-L5000\x19\x04&\x13'    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  b"DNhe SCC FHCUP      1.00 1.02 99110-L5000 \x01#\x15#    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  # 2020 PALISADE
  b"LX2_ SCC FHCUP      1.00 1.04 99110-S8100\x19\x05\x02\x16V    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  # 2022 PALISADE
  b"LX2_ SCC FHCUP      1.00 1.00 99110-S8110!\x04\x05\x17\x01    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  # 2020 SANTA FE
  b"TM__ SCC F-CUP      1.00 1.03 99110-S2000\x19\x050\x13'    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  # 2020 GENESIS G70
  b'IK__ SCC F-CUP      1.00 1.02 96400-G9100\x18\x07\x06\x17\x12    ': ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  # 2019 SANTA FE
  b"TM__ SCC F-CUP      1.00 1.00 99110-S1210\x19\x01%\x168    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  b"TM__ SCC F-CUP      1.00 1.02 99110-S2000\x18\x07\x08\x18W    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
  # 2021 K5 HEV
  b"DLhe SCC FHCUP      1.00 1.02 99110-L7000 \x01 \x102    ": ConfigValues(
    default_config=b"\x00\x00\x00\x01\x00\x00",
    tracks_enabled=b"\x00\x00\x00\x01\x00\x01"),
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='configure radar to output points (or reset to default)')
  parser.add_argument('--default', action="store_true", default=False, help='reset to default configuration (default: false)')
  parser.add_argument('--debug', action="store_true", default=False, help='enable debug output (default: false)')
  parser.add_argument('--bus', type=int, default=0, help='can bus to use (default: 0)')
  args = parser.parse_args()

  try:
    check_output(["pidof", "boardd"])
    print("boardd is running, please kill openpilot before running this script! (aborted)")
    sys.exit(1)
  except CalledProcessError as e:
    if e.returncode != 1: # 1 == no process found (boardd not running)
      raise e

  confirm = input("power on the vehicle keeping the engine off (press start button twice) then type OK to continue: ").upper().strip()
  if confirm != "OK":
    print("\nyou didn't type 'OK! (aborted)")
    sys.exit(0)

  panda = Panda()
  panda.set_safety_mode(Panda.SAFETY_ELM327)
  uds_client = UdsClient(panda, 0x7D0, bus=args.bus, debug=args.debug)

  print("\n[START DIAGNOSTIC SESSION]")
  session_type : SESSION_TYPE = 0x07 # type: ignore
  uds_client.diagnostic_session_control(session_type)

  print("[HARDWARE/SOFTWARE VERSION]")
  fw_version_data_id : DATA_IDENTIFIER_TYPE = 0xf100 # type: ignore
  fw_version = uds_client.read_data_by_identifier(fw_version_data_id)
  print(fw_version)
  if fw_version not in SUPPORTED_FW_VERSIONS.keys():
    print("radar not supported! (aborted)")
    sys.exit(1)

  print("[GET CONFIGURATION]")
  config_data_id : DATA_IDENTIFIER_TYPE = 0x0142 # type: ignore
  current_config = uds_client.read_data_by_identifier(config_data_id)
  config_values = SUPPORTED_FW_VERSIONS[fw_version]
  new_config = config_values.default_config if args.default else config_values.tracks_enabled
  print(f"current config: 0x{current_config.hex()}")
  if current_config != new_config:
    print("[CHANGE CONFIGURATION]")
    print(f"new config:     0x{new_config.hex()}")
    uds_client.write_data_by_identifier(config_data_id, new_config)
    if not args.default and current_config != SUPPORTED_FW_VERSIONS[fw_version].default_config:
      print("\ncurrent config does not match expected default! (aborted)")
      sys.exit(1)

    print("[DONE]")
    print("\nrestart your vehicle and ensure there are no faults")
    if not args.default:
      print("you can run this script again with --default to go back to the original (factory) settings")
  else:
    print("[DONE]")
    print("\ncurrent config is already the desired configuration")
    sys.exit(0)
