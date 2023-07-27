#!/usr/bin/env python3
import argparse
from typing import List, Optional
from tqdm import tqdm
from panda import Panda
from panda.python.uds import UdsClient, MessageTimeoutError, NegativeResponseError, InvalidSubAddressError, \
                             SESSION_TYPE, DATA_IDENTIFIER_TYPE

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--rxoffset", default="")
  parser.add_argument("--nonstandard", action="store_true")
  parser.add_argument("--no-obd", action="store_true", help="Bus 1 will not be multiplexed to the OBD-II port")
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--addr")
  parser.add_argument("--sub_addr", "--subaddr", help="A hex sub-address or `scan` to scan the full sub-address range")
  parser.add_argument("--bus")
  parser.add_argument('-s', '--serial', help="Serial number of panda to use")
  args = parser.parse_args()

  if args.addr:
    addrs = [int(args.addr, base=16)]
  else:
    addrs = [0x700 + i for i in range(256)]
    addrs += [0x18da0000 + (i << 8) + 0xf1 for i in range(256)]
  results = {}

  sub_addrs: List[Optional[int]] = [None]
  if args.sub_addr:
    if args.sub_addr == "scan":
      sub_addrs = list(range(0xff + 1))
    else:
      sub_addrs = [int(args.sub_addr, base=16)]
      if sub_addrs[0] > 0xff:  # type: ignore
        print(f"Invalid sub-address: 0x{sub_addrs[0]:X}, needs to be in range 0x0 to 0xff")
        parser.print_help()
        exit()

  uds_data_ids = {}
  for std_id in DATA_IDENTIFIER_TYPE:
    uds_data_ids[std_id.value] = std_id.name
  if args.nonstandard:
    for uds_id in range(0xf100, 0xf180):
      uds_data_ids[uds_id] = "IDENTIFICATION_OPTION_VEHICLE_MANUFACTURER_SPECIFIC_DATA_IDENTIFIER"
    for uds_id in range(0xf1a0, 0xf1f0):
      uds_data_ids[uds_id] = "IDENTIFICATION_OPTION_VEHICLE_MANUFACTURER_SPECIFIC"
    for uds_id in range(0xf1f0, 0xf200):
      uds_data_ids[uds_id] = "IDENTIFICATION_OPTION_SYSTEM_SUPPLIER_SPECIFIC"

  panda_serials = Panda.list()
  if args.serial is None and len(panda_serials) > 1:
    print("\nMultiple pandas found, choose one:")
    for serial in panda_serials:
      with Panda(serial) as panda:
        print(f"  {serial}: internal={panda.is_internal()}")
    print()
    parser.print_help()
    exit()

  panda = Panda(serial=args.serial)
  panda.set_safety_mode(Panda.SAFETY_ELM327, 1 if args.no_obd else 0)
  print("querying addresses ...")
  with tqdm(addrs) as t:
    for addr in t:
      # skip functional broadcast addrs
      if addr == 0x7df or addr == 0x18db33f1:
        continue

      if args.bus:
        bus = int(args.bus)
      else:
        bus = 1 if panda.has_obd() else 0
      rx_addr = addr + int(args.rxoffset, base=16) if args.rxoffset else None

      # Try all sub-addresses for addr. By default, this is None
      for sub_addr in sub_addrs:
        sub_addr_str = hex(sub_addr) if sub_addr is not None else None
        t.set_description(f"{hex(addr)}, {sub_addr_str}")
        uds_client = UdsClient(panda, addr, rx_addr, bus, sub_addr=sub_addr, timeout=0.2, debug=args.debug)
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
        except InvalidSubAddressError as e:
          print(f'*** Skipping address {hex(addr)}: {e}')
          break

        # Run queries against all standard UDS data identifiers, plus selected
        # non-standardized identifier ranges if requested
        resp = {}
        for uds_data_id in sorted(uds_data_ids):
          try:
            data = uds_client.read_data_by_identifier(uds_data_id)  # type: ignore
            if data:
              resp[uds_data_id] = data
          except (NegativeResponseError, MessageTimeoutError, InvalidSubAddressError):
            pass

        if resp.keys():
          results[(addr, sub_addr)] = resp

    if len(results.items()):
      for (addr, sub_addr), resp in results.items():
        sub_addr_str = f", sub-address 0x{sub_addr:X}" if sub_addr is not None else ""
        print(f"\n\n*** Results for address 0x{addr:X}{sub_addr_str} ***\n\n")
        for rid, dat in resp.items():
          print(f"0x{rid:02X} {uds_data_ids[rid]}: {dat}")
    else:
      print("no fw versions found!")
