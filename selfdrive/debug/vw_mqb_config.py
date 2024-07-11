#!/usr/bin/env python3

import argparse
import struct
from enum import IntEnum
from panda import Panda
from panda.python.uds import UdsClient, MessageTimeoutError, NegativeResponseError, SESSION_TYPE,\
  DATA_IDENTIFIER_TYPE, ACCESS_TYPE
from datetime import date

# TODO: extend UDS library to allow custom/vendor-defined data identifiers without ignoring type checks
class VOLKSWAGEN_DATA_IDENTIFIER_TYPE(IntEnum):
  CODING = 0x0600

# TODO: extend UDS library security_access() to take an access level offset per ISO 14229-1:2020 10.4 and remove this
class ACCESS_TYPE_LEVEL_1(IntEnum):
  REQUEST_SEED = ACCESS_TYPE.REQUEST_SEED + 2
  SEND_KEY = ACCESS_TYPE.SEND_KEY + 2

MQB_EPS_CAN_ADDR = 0x712
RX_OFFSET = 0x6a

if __name__ == "__main__":
  desc_text =   "Shows Volkswagen EPS software and coding info, and enables or disables Heading Control Assist " + \
                "(Lane Assist). Useful for enabling HCA on cars without factory Lane Assist that want to use " + \
                "openpilot integrated at the CAN gateway (J533)."
  epilog_text = "This tool is meant to run directly on a vehicle-installed comma three, with the " + \
                "openpilot/tmux processes stopped. It should also work on a separate PC with a USB-attached comma " + \
                "panda. Vehicle ignition must be on. Recommend engine not be running when making changes. Must " + \
                "turn ignition off and on again for any changes to take effect."
  parser = argparse.ArgumentParser(description=desc_text, epilog=epilog_text)
  parser.add_argument("--debug", action="store_true", help="enable ISO-TP/UDS stack debugging output")
  parser.add_argument("action", choices={"show", "enable", "disable"}, help="show or modify current EPS HCA config")
  args = parser.parse_args()

  panda = Panda()
  panda.set_safety_mode(Panda.SAFETY_ELM327)
  bus = 1 if panda.has_obd() else 0
  uds_client = UdsClient(panda, MQB_EPS_CAN_ADDR, MQB_EPS_CAN_ADDR + RX_OFFSET, bus, timeout=0.2, debug=args.debug)

  try:
    uds_client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
  except MessageTimeoutError:
    print("Timeout opening session with EPS")
    quit()

  odx_file, current_coding = None, None
  try:
    hw_pn = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_HARDWARE_NUMBER).decode("utf-8")
    sw_pn = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER).decode("utf-8")
    sw_ver = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_VERSION_NUMBER).decode("utf-8")
    component = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.SYSTEM_NAME_OR_ENGINE_TYPE).decode("utf-8")
    odx_file = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.ODX_FILE).decode("utf-8").rstrip('\x00')
    current_coding = uds_client.read_data_by_identifier(VOLKSWAGEN_DATA_IDENTIFIER_TYPE.CODING)  # type: ignore
    coding_text = current_coding.hex()

    print("\nEPS diagnostic data\n")
    print(f"   Part No HW:   {hw_pn}")
    print(f"   Part No SW:   {sw_pn}")
    print(f"   SW Version:   {sw_ver}")
    print(f"   Component:    {component}")
    print(f"   Coding:       {coding_text}")
    print(f"   ASAM Dataset: {odx_file}")
  except NegativeResponseError:
    print("Error fetching data from EPS")
    quit()
  except MessageTimeoutError:
    print("Timeout fetching data from EPS")
    quit()

  coding_variant, current_coding_array, coding_byte, coding_bit = None, None, 0, 0
  coding_length = len(current_coding)

  # EPS_MQB_ZFLS
  if odx_file in ("EV_SteerAssisMQB", "EV_SteerAssisMNB"):
    coding_variant = "ZFLS"
    coding_byte = 0
    coding_bit = 4

  # MQB_PP_APA, MQB_VWBS_GEN2
  elif odx_file in ("EV_SteerAssisVWBSMQBA", "EV_SteerAssisVWBSMQBGen2"):
    coding_variant = "APA"
    coding_byte = 3
    coding_bit = 0

  else:
    print("Configuration changes not yet supported on this EPS!")
    quit()

  current_coding_array = struct.unpack(f"!{coding_length}B", current_coding)
  hca_enabled = (current_coding_array[coding_byte] & (1 << coding_bit) != 0)
  hca_text = ("DISABLED", "ENABLED")[hca_enabled]
  print(f"   Lane Assist:  {hca_text}")

  try:
    params = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION).decode("utf-8")
    param_version_system_params = params[1:3]
    param_vehicle_type = params[3:5]
    param_index_char_curve = params[5:7]
    param_version_char_values = params[7:9]
    param_version_memory_map = params[9:11]
    print("\nEPS parameterization (per-vehicle calibration) data\n")
    print(f"   Version of system parameters:     {param_version_system_params}")
    print(f"   Vehicle type:                     {param_vehicle_type}")
    print(f"   Index of characteristic curve:    {param_index_char_curve}")
    print(f"   Version of characteristic values: {param_version_char_values}")
    print(f"   Version of memory map:            {param_version_memory_map}")
  except (NegativeResponseError, MessageTimeoutError):
    print("Error fetching parameterization data from EPS!")
    quit()

  if args.action in ["enable", "disable"]:
    print("\nAttempting configuration update")

    assert(coding_variant in ("ZFLS", "APA"))
    # ZFLS EPS config coding length can be anywhere from 1 to 4 bytes, but the
    # bit we care about is always in the same place in the first byte
    if args.action == "enable":
      new_byte = current_coding_array[coding_byte] | (1 << coding_bit)
    else:
      new_byte = current_coding_array[coding_byte] & ~(1 << coding_bit)
    new_coding = current_coding[0:coding_byte] + new_byte.to_bytes(1, "little") + current_coding[coding_byte+1:]

    try:
      seed = uds_client.security_access(ACCESS_TYPE_LEVEL_1.REQUEST_SEED)  # type: ignore
      key = struct.unpack("!I", seed)[0] + 28183  # yeah, it's like that
      uds_client.security_access(ACCESS_TYPE_LEVEL_1.SEND_KEY, struct.pack("!I", key))  # type: ignore
    except (NegativeResponseError, MessageTimeoutError):
      print("Security access failed!")
      print("Open the hood and retry (disables the \"diagnostic firewall\" on newer vehicles)")
      quit()

    try:
      # Programming date and tester number must be written before making
      # a change, or write to CODING will fail with request sequence error
      # Encoding on tester is unclear, it contains the workshop code in the
      # last two bytes, but not the VZ/importer or tester serial number
      # Can't seem to read it back, but we can read the calibration tester,
      # so fib a little and say that same tester did the programming
      current_date = date.today()
      formatted_date = current_date.strftime('%y-%m-%d')
      year, month, day = (int(part) for part in formatted_date.split('-'))
      prog_date = bytes([year, month, day])
      uds_client.write_data_by_identifier(DATA_IDENTIFIER_TYPE.PROGRAMMING_DATE, prog_date)
      tester_num = uds_client.read_data_by_identifier(DATA_IDENTIFIER_TYPE.CALIBRATION_REPAIR_SHOP_CODE_OR_CALIBRATION_EQUIPMENT_SERIAL_NUMBER)
      uds_client.write_data_by_identifier(DATA_IDENTIFIER_TYPE.REPAIR_SHOP_CODE_OR_TESTER_SERIAL_NUMBER, tester_num)
      uds_client.write_data_by_identifier(VOLKSWAGEN_DATA_IDENTIFIER_TYPE.CODING, new_coding)  # type: ignore
    except (NegativeResponseError, MessageTimeoutError):
      print("Writing new configuration failed!")
      print("Make sure the comma processes are stopped: tmux kill-session -t comma")
      quit()

    try:
      # Read back result just to make 100% sure everything worked
      current_coding_text = uds_client.read_data_by_identifier(VOLKSWAGEN_DATA_IDENTIFIER_TYPE.CODING).hex()  # type: ignore
      print(f"   New coding:   {current_coding_text}")
    except (NegativeResponseError, MessageTimeoutError):
      print("Reading back updated coding failed!")
      quit()
    print("EPS configuration successfully updated")
