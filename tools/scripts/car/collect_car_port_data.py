#!/usr/bin/env python3
"""Collect diagnostics, or use --wizard for timestamped stationary CAN capture."""
import json
import time
import argparse
import select
import sys
import zstandard as zstd
from pathlib import Path
from collections import defaultdict
import openpilot.cereal.messaging as messaging
from opendbc.car.carlog import carlog
from opendbc.car.ecu_addrs import get_all_ecu_addrs
from opendbc.car.fw_versions import get_fw_versions, match_fw_to_car
from opendbc.car.isotp_parallel_query import IsoTpParallelQuery
from opendbc.car.uds import FUNCTIONAL_ADDRS, get_dtc_num_as_str, get_dtc_status_names
from opendbc.car.vin import get_vin
from openpilot.common.params import Params
from openpilot.selfdrive.car.card import can_comm_callbacks, obd_callback

EXT_DIAG_REQUEST = b'\x10\x03'
EXT_DIAG_RESPONSE = b'\x50\x03'
READ_DTC_REQUEST = b'\x19\x02\xff'
READ_DTC_RESPONSE = b'\x59\x02'
CAN_CLOCK = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)  # pandad's logMonoTime clock

CAN_STEPS = [
  ("baseline", "Leave the controls at rest for five seconds."),
  ("left_on", "Turn the left indicator on; wait for five flashes."),
  ("left_off", "Turn the left indicator off."),
  ("right_on", "Turn the right indicator on; wait for five flashes."),
  ("right_off", "Turn the right indicator off."),
  ("hazards_on", "Turn hazards on; wait for five flashes."),
  ("hazards_left", "With hazards on, move the indicator stalk left; wait for five flashes."),
  ("hazards_right", "With hazards on, move the indicator stalk right; wait for five flashes."),
  ("hazards_neutral", "With hazards on, return the indicator stalk to neutral."),
  ("hazards_off", "Turn hazards off."),
] + [
  (f"{door}_{action}", f"{action.capitalize()} only the {door.replace('_', ' ')} door.")
  for door in ("front_left", "front_right", "rear_left", "rear_right")
  for action in ("open", "close")
] + [
  ("seatbelt_off", "Unbuckle the driver's seatbelt."),
  ("seatbelt_on", "Buckle the driver's seatbelt."),
  ("brake_light", "Press the brake pedal lightly and hold."),
  ("brake_release_light", "Release the brake pedal."),
  ("brake_firm", "Press the brake pedal firmly and hold."),
  ("brake_release_firm", "Release the brake pedal."),
  ("accelerator_press", "In Park, gently press and hold the accelerator briefly."),
  ("accelerator_release", "Release the accelerator."),
  ("parking_brake_off", "Hold the foot brake and release the parking brake."),
  ("parking_brake_on", "Hold the foot brake and apply the parking brake."),
  ("steer_left", "Turn the steering wheel left without forcing the end stop."),
  ("steer_right", "Turn the steering wheel right without forcing the end stop."),
  ("steer_center", "Return the steering wheel to center."),
] + [
  (f"{button}_{press}", f"{instruction} the {button.replace('_', ' ')} button, then release it.")
  for button in ("cruise_main", "set", "resume", "cancel", "gap", "lta")
  for press, instruction in (("short", "Briefly press"), ("held", "Hold for two seconds"))
]


def collect_can_steps(output):
  can_path = Path(output).with_suffix(".can.zst")
  markers_path = Path(output).with_suffix(".events.jsonl")
  print("Keep the vehicle stationary in Park, parking brake applied, openpilot disengaged.")
  print("Start with doors closed, driver's belt buckled, indicators off. Leave pandad running.")
  print("Enter: start an action; do it; Enter again: finish. s: skip; q: quit and save.")
  print(f"CAN: {can_path}\nEvents: {markers_path}")
  logcan = messaging.sub_sock('can')
  step = 0
  active = False
  count = 0
  receiving = False
  status = "error"
  last_can = time.monotonic()
  with open(can_path, "xb") as raw, zstd.ZstdCompressor().stream_writer(raw) as capture, open(markers_path, "x") as markers:
    def mark(event, **fields):
      markers.write(json.dumps({"event": event, "logMonoTime": time.clock_gettime_ns(CAN_CLOCK), "wallTimeNs": time.time_ns(), **fields}) + "\n")
      markers.flush()
      capture.flush()

    mark("session_start")
    print("Waiting for CAN...", flush=True)
    try:
      while step < len(CAN_STEPS):
        for _ in range(256):
          data = logcan.receive(non_blocking=True)
          if data is None:
            break
          capture.write(data)
          count += 1
          if messaging.log_from_bytes(data).can:
            last_can = time.monotonic()
            if not receiving:
              print(f"\n[1/{len(CAN_STEPS)}] {CAN_STEPS[0][1]}\nPress Enter to start.", flush=True)
            receiving = True
        if time.monotonic() - last_can > 3:
          raise RuntimeError("No CAN received for three seconds. Check ignition and pandad.")
        if not receiving or not select.select([sys.stdin], [], [], 0.02)[0]:
          if not receiving:
            time.sleep(0.02)
          continue
        line = sys.stdin.readline()
        if not line:
          status = "input_closed"
          break
        command = line.strip().lower()
        action, instruction = CAN_STEPS[step]
        if command == "q":
          status = "quit"
          break
        if command == "s":
          mark("skip", action=action, instruction=instruction)
        elif command == "":
          mark("end" if active else "start", action=action, instruction=instruction)
          if not active:
            active = True
            print("Recording action. Press Enter when finished.", flush=True)
            continue
        else:
          print("Use Enter, s, or q.", flush=True)
          continue
        active = False
        step += 1
        if step < len(CAN_STEPS):
          print(f"\n[{step + 1}/{len(CAN_STEPS)}] {CAN_STEPS[step][1]}\nPress Enter to start.", flush=True)
      else:
        status = "completed"
    except KeyboardInterrupt:
      status = "interrupted"
      print("\nStopped.")
    finally:
      if active:
        mark("abort", action=CAN_STEPS[step][0])
      mark("session_end", status=status, canEvents=count)
  print(f"Saved {count} CAN events and action markers.")


def get_dtcs(can_recv, can_send, set_obd_multiplexing, queries, timeout):
  dtcs = {}
  for (bus, obd_multiplexing, response_offset), addrs in queries.items():
    set_obd_multiplexing(obd_multiplexing)
    query = IsoTpParallelQuery(can_send, can_recv, bus, list(addrs), [EXT_DIAG_REQUEST, READ_DTC_REQUEST],
                               [EXT_DIAG_RESPONSE, READ_DTC_RESPONSE], response_offset=response_offset)
    results = query.get_data(timeout)
    for addr, sub_addr in addrs:
      data = results.get((addr, sub_addr))
      dtcs[(addr, sub_addr, bus)] = None if data is None else [(get_dtc_num_as_str(data[i:i+3]), " ".join(get_dtc_status_names(data[i+3])))
                                                               for i in range(1, len(data) - 3, 4)]
  return dtcs


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Collect car diagnostics or guided stationary CAN recordings')
  parser.add_argument('--wizard', action='store_true', help='record CAN with prompted action start/end markers; no diagnostic queries')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--brand', help='Only query addresses/with requests for this brand')
  parser.add_argument('--timeout', type=float, default=1.0, help='ECU address and DTC query timeout')
  parser.add_argument('--out', default=f'/data/car_port_data_{time.strftime("%Y%m%d-%H%M%S")}.json',
                      help='diagnostic JSON path; wizard writes sibling .can.zst and .events.jsonl files')
  args = parser.parse_args()

  if args.debug:
    carlog.setLevel('DEBUG')

  if args.wizard:
    collect_can_steps(args.out)
    sys.exit(0)

  logcan = messaging.sub_sock('can')
  sendcan = messaging.pub_sock('sendcan')
  can_callbacks = can_comm_callbacks(logcan, sendcan)

  # Set up params for pandad
  params = Params()
  params.remove("FirmwareQueryDone")
  params.put_bool("IsOffroad", True, block=True)
  time.sleep(0.2)  # thread is 10 Hz
  params.put_bool("IsOffroad", False, block=True)
  set_obd_multiplexing = obd_callback(params)

  print("Getting vin...")
  set_obd_multiplexing(True)
  _, _, vin = get_vin(*can_callbacks, (0, 1))
  print(f"VIN: {vin}")

  print("Getting fw versions...")
  fw_vers = get_fw_versions(*can_callbacks, set_obd_multiplexing, query_brand=args.brand, progress=True)
  exact_match, candidates = match_fw_to_car(fw_vers, vin)
  for version in fw_vers:
    subaddr = None if version.subAddress == 0 else hex(version.subAddress)
    print(f"  Brand: {version.brand}, bus: {version.bus}, OBD: {version.obdMultiplexing} - " +
          f"(Ecu.{version.ecu}, {hex(version.address)}, {subaddr}): [{version.fwVersion!r}]")
  print(f"Possible matches: {candidates} (exact: {exact_match})")

  # DTCs from every fw query ECU and every ECU responding to tester present
  dtc_queries = defaultdict(set)
  for fw in fw_vers:
    sub_addr = None if fw.subAddress == 0 else fw.subAddress
    dtc_queries[(fw.bus, fw.obdMultiplexing, fw.responseAddress - fw.address)].add((fw.address, sub_addr))

  print("Getting ECU addresses...")
  ecu_addrs = []
  for bus, obd_multiplexing in ((0, False), (1, True)):
    set_obd_multiplexing(obd_multiplexing)
    for rx_addr, _, _ in get_all_ecu_addrs(*can_callbacks, bus, args.timeout):
      print(f"  bus {bus}: {hex(rx_addr)}")
      ecu_addrs.append({"rxAddress": hex(rx_addr), "bus": bus})
      if rx_addr < 0x800 and rx_addr - 8 not in FUNCTIONAL_ADDRS:
        dtc_queries[(bus, obd_multiplexing, 8)].add((rx_addr - 8, None))

  print("Getting DTCs...")
  dtcs = get_dtcs(*can_callbacks, set_obd_multiplexing, dtc_queries, args.timeout)
  for (addr, sub_addr, bus), ecu_dtcs in sorted(dtcs.items(), key=lambda d: (d[0][2], d[0][0], d[0][1] or 0)):
    print(f"  {hex(addr)}, {sub_addr}, bus {bus}: {'no response' if ecu_dtcs is None else ecu_dtcs or 'no DTCs'}")

  with open(args.out, "w") as f:
    json.dump({
      "vin": vin,
      "exactMatch": exact_match,
      "candidates": sorted(candidates),
      "fwVersions": [fw.to_dict() | {"fwVersion": repr(fw.fwVersion), "request": [r.hex() for r in fw.request]} for fw in fw_vers],
      "ecuAddrs": ecu_addrs,
      "dtcs": [{"address": hex(addr), "subAddress": sub_addr, "bus": bus, "dtcs": ecu_dtcs} for (addr, sub_addr, bus), ecu_dtcs in dtcs.items()],
    }, f, indent=2)
  print(f"Wrote {args.out}")
