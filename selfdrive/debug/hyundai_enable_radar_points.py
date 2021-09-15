#!/usr/bin/env python3
import sys
import argparse
from subprocess import check_output, CalledProcessError

from panda.python import Panda
from panda.python.uds import UdsClient, SESSION_TYPE, DATA_IDENTIFIER_TYPE

SUPPORTED_FW_VERSIONS = {
  # 2020 SONATA
  b"DN8_ SCC FHCUP      1.00 1.00 99110-L0000         ": {
    "default_config": b"\x00\x00\x00\x01\x00\x00",
    "tracks_enabled": b"\x00\x00\x00\x01\x00\x01",
  },
  # 2020 PALISADE
  b"LX2_ SCC FHCUP      1.00 1.04 99110-S8100         ": {
    "default_config": b"\x00\x00\x00\x01\x00\x00",
    "tracks_enabled": b"\x00\x00\x00\x01\x00\x01",
  },
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

  confirm = input("put your vehicle in accessory mode now and type OK to continue: ").upper().strip()
  if confirm != "OK":
    print("\nyou didn't type 'OK! (aborted)")
    sys.exit(0)

  panda = Panda() # type: ignore
  panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  uds_client = UdsClient(panda, 0x7D0, bus=args.bus, debug=args.debug)

  print("\n[START DIAGNOSTIC SESSION]")
  session_type : SESSION_TYPE = 0x07 # type: ignore
  uds_client.diagnostic_session_control(session_type)

  print("[HARDWARE/SOFTWARE VERSION]")
  fw_version_data_id : DATA_IDENTIFIER_TYPE = 0xf100 # type: ignore
  fw_version = uds_client.read_data_by_identifier(fw_version_data_id)
  print(fw_version)
  if fw_version != SUPPORTED_FW_VERSIONS:
    print("radar not supported! (aborted)")
    sys.exit(1)

  print("[GET CONFIGURATION]")
  config_data_id : DATA_IDENTIFIER_TYPE = 0x0142 # type: ignore
  current_config = uds_client.read_data_by_identifier(config_data_id)
  new_config = SUPPORTED_FW_VERSIONS[fw_version]["default_config" if args.default else "tracks_enabled"]
  print(f"current config: 0x{current_config.hex()}")
  if current_config != new_config:
    print("[CHANGE CONFIGURATION]")
    print(f"new config:     0x{new_config.hex()}")
    uds_client.write_data_by_identifier(config_data_id, new_config)
    if not args.default and current_config != SUPPORTED_FW_VERSIONS[fw_version]["default_config"]:
      print("\ncurrent config does not match expected default! (aborted)")
      sys.exit(1)

    print("[DONE]")
    print("\nrestart your vehicle and ensure there are no faults")
    print("you can run this script again with --default to go back to the original (factory) settings")
  else:
    print("[DONE]")
    print("\ncurrent config is already the desired configuration")
    sys.exit(0)
