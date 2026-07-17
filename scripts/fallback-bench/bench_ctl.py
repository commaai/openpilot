#!/usr/bin/env python3
# bench CAN replay with controllable cruise engagement and ignition
# commands: echo engage|disengage > /tmp/bench_cruise ; echo 1|0 > /tmp/bench_ign
import os
import sys
import time
import threading

import usb1

sys.path.insert(0, "/home/batman/openpilot")
from openpilot.common.realtime import Ratekeeper, DT_CTRL
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.tools.lib.logreader import LogReader
from opendbc.can.packer import CANPacker
from panda import PandaJungle

CRUISE_CMD = "/tmp/bench_cruise"
IGN_CMD = "/tmp/bench_ign"
PWR_CMD = "/tmp/bench_pwr"
ROUTE = "5f5afb36036506e4/2019-05-14--02-09-54/0:2"
# The route's first 14.11 seconds contain its parked startup sequence. Replay
# that sequence once, then loop only the continuous driving portion so a long
# bench drive does not jump from Drive back through P/R/N at every route seam.
DRIVING_LOOP_START = 1411
STEER_TOUCH_TORQUE = 150

packer = CANPacker('toyota_nodsu_pt_generated')
PCM1_ON = packer.make_can_msg('PCM_CRUISE', 0, {'CRUISE_ACTIVE': 1, 'GAS_RELEASED': 1, 'CRUISE_STATE': 8})
PCM1_OFF = packer.make_can_msg('PCM_CRUISE', 0, {'CRUISE_ACTIVE': 0, 'GAS_RELEASED': 1, 'CRUISE_STATE': 0})
PCM2_ON = packer.make_can_msg('PCM_CRUISE_2', 0, {'MAIN_ON': 1, 'SET_SPEED': 60})
# These are actuator frames echoed into the route log. Replaying them onto the
# physical bench makes the target Panda correctly diagnose a stock-ECU/relay
# malfunction, so they are not vehicle-input traffic for this fixture.
RELAY_PROTECTED_ADDRS = {0x2E4, 0x191, 0x412, 0x343}
CAM_FAKE_BUS = {0x343: 2, 0x412: 2}


def with_steering_touch(dat):
  d = bytearray(dat)
  d[1:3] = STEER_TOUCH_TORQUE.to_bytes(2, 'big', signed=True)
  d[-1] = (len(d) + 0x60 + 0x02 + sum(d[:-1])) & 0xFF
  return bytes(d)


def read_cmd(path, default):
  try:
    with open(path) as f:
      return f.read().strip() or default
  except FileNotFoundError:
    return default


def load_route():
  print("loading route...")
  lr = LogReader(ROUTE)
  mbytes = [m.as_builder().to_bytes() for m in lr if m.which() == 'can']
  msgs = [m[1] for m in can_capnp_to_list(mbytes)]
  print(f"{len(msgs)} can frames")
  return msgs


def main():
  can_msgs = load_route()
  if len(can_msgs) != 21017 or not 0 < DRIVING_LOOP_START < len(can_msgs):
    raise RuntimeError(f"unexpected route shape: {len(can_msgs)} frames")
  j = PandaJungle("09001e000d51333038363231")
  j.reset()
  for i in [0, 1, 2, 3, 0xFFFF]:
    j.can_clear(i)
    j.set_can_speed_kbps(i, 500)
  j.set_can_loopback(False)
  desired_pwr = read_cmd(PWR_CMD, "1")
  desired_ign = read_cmd(IGN_CMD, "1")
  j.set_panda_power(desired_pwr == "1")
  j.set_ignition(desired_ign == "1")
  print(f"replaying, cruise=disengage ign={desired_ign} power={desired_pwr}")

  engage_since = None
  rk = Ratekeeper(1 / DT_CTRL, print_delay_threshold=None)
  last_ign = desired_ign
  last_pwr = desired_pwr
  while True:
    if rk.frame % 50 == 0:
      ign = read_cmd(IGN_CMD, "1")
      if ign != last_ign:
        j.set_ignition(ign == "1")
        print(f"ignition -> {ign}", flush=True)
        last_ign = ign
      pwr = read_cmd(PWR_CMD, "1")
      if pwr != last_pwr:
        j.set_panda_power(pwr == "1")
        print(f"power -> {pwr}", flush=True)
        last_pwr = pwr
    cruise = read_cmd(CRUISE_CMD, "disengage")
    if cruise == "engage":
      if engage_since is None:
        engage_since = rk.frame
      # brief off period generates the rising edge openpilot needs
      active = (rk.frame - engage_since) > 50
    else:
      engage_since = None
      active = False

    # batch two 10ms route frames per USB transfer: at full-speed (12M) the
    # bottleneck is per-transfer latency, and unbatched replay runs at ~55%
    if rk.frame % 2:
      rk.keep_time()
      continue
    send = []
    for fofs in (0, 1):
      vframe = rk.frame + fofs
      if vframe < len(can_msgs):
        route_frame = vframe
      else:
        route_frame = DRIVING_LOOP_START + ((vframe - len(can_msgs)) % (len(can_msgs) - DRIVING_LOOP_START))
      send += [m for m in can_msgs[route_frame] if m[-1] <= 2]
      # master's toyota parser requires these on the cam/radar buses; the 2019 route
      # has them only as bus-0 TX echoes (src 128). re-emit there, never on bus 0.
      send += [(a, d, CAM_FAKE_BUS[a]) for a, d, b in can_msgs[route_frame] if b == 128 and a in CAM_FAKE_BUS]
    out = []
    for addr, dat, bus in send:
      if addr in RELAY_PROTECTED_ADDRS and bus == 0:
        continue
      if bus == 0 and addr == 466:
        out.append(PCM1_ON if active else PCM1_OFF)
      elif bus == 0 and addr == 467:
        out.append(PCM2_ON)
      elif bus == 0 and addr == 550:
        # clear BRAKE_PRESSED (bit 37) so route brake taps cannot disengage
        d = bytearray(dat)
        d[4] &= ~0x20
        out.append((addr, bytes(d), bus))
      elif bus == 0 and addr == 608:
        # Simulate a light wheel touch so bench DM does not escalate on an empty seat.
        out.append((addr, with_steering_touch(dat), bus))
      else:
        out.append((addr, dat, bus))
    try:
      j.can_send_many(out)
    except usb1.USBErrorTimeout:
      pass
    j.can_recv()
    if rk.frame % 6000 == 0:
      try:
        h = j.health()
        print(f"health {h}", flush=True)
      except Exception as e:
        print(f"health err {e}", flush=True)
    rk.keep_time()


if __name__ == "__main__":
  main()
