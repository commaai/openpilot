#!/usr/bin/env python3
import argparse
from tqdm import tqdm
from panda import Panda
from panda.python.uds import UdsClient, MessageTimeoutError, NegativeResponseError, SESSION_TYPE, DATA_IDENTIFIER_TYPE

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--rxoffset', default="")
  parser.add_argument('--nonstandard', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--addr')
  args = parser.parse_args()

  if args.addr:
    addrs = [int(args.addr, base=16)]
  else:
    addrs = [0x700 + i for i in range(256)]
    addrs += [0x18da0000 + (i << 8) + 0xf1 for i in range(256)]
  results = {}

  uds_data_ids = {}
  for std_id in DATA_IDENTIFIER_TYPE:
    uds_data_ids[std_id.value] = std_id.name
  if args.nonstandard:
    for uds_id in range(0xf100,0xf180):
      uds_data_ids[uds_id] = "IDENTIFICATION_OPTION_VEHICLE_MANUFACTURER_SPECIFIC_DATA_IDENTIFIER"
    for uds_id in range(0xf1a0,0xf1f0):
      uds_data_ids[uds_id] = "IDENTIFICATION_OPTION_VEHICLE_MANUFACTURER_SPECIFIC"
    for uds_id in range(0xf1f0,0xf200):
      uds_data_ids[uds_id] = "IDENTIFICATION_OPTION_SYSTEM_SUPPLIER_SPECIFIC"

  panda = Panda()
  panda.set_safety_mode(Panda.SAFETY_ELM327)
  print("querying addresses ...")
  with tqdm(addrs) as t:
    for addr in t:
      # skip functional broadcast addrs
      if addr == 0x7df or addr == 0x18db33f1:
        continue
      t.set_description(hex(addr))
      panda.send_heartbeat()

      bus = 1 if panda.has_obd() else 0
      rx_addr = addr + int(args.rxoffset, base=16) if args.rxoffset else None
      uds_client = UdsClient(panda, addr, rx_addr, bus, timeout=0.2, debug=args.debug)
      # Check for anything alive at this address, and switch to the highest
      # available diagnostic session without security access
      try:
        uds_client.tester_present()
        uds_client.diagnostic_session_control(SESSION_TYPE.DEFAULT)
        uds_client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
      except NegativeResponseError:
        pass
      except MessageTimeoutError:
        continue

      # Run queries against all standard UDS data identifiers, plus selected
      # non-standardized identifier ranges if requested
      resp = {}
      for uds_data_id in sorted(uds_data_ids):
        try:
          data = uds_client.read_data_by_identifier(uds_data_id)  # type: ignore
          if data:
            resp[uds_data_id] = data
        except (NegativeResponseError, MessageTimeoutError):
          pass

      if resp.keys():
        results[addr] = resp

    if len(results.items()):
      for addr, resp in results.items():
        print(f"\n\n*** Results for address 0x{addr:X} ***\n\n")
        for rid, dat in resp.items():
          print(f"0x{rid:02X} {uds_data_ids[rid]}: {dat}")
    else:
      print("no fw versions found!")
