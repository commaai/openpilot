#!/usr/bin/env python3
import argparse
import os

from panda.tests.safety import libpandasafety_py
from panda.tests.safety_replay.helpers import package_can_msg, init_segment

# replay a drive to check for safety violations
def replay_drive(lr, safety_mode, param, alternative_experience, segment=False):
  safety = libpandasafety_py.libpandasafety

  err = safety.set_safety_hooks(safety_mode, param)
  assert err == 0, "invalid safety mode: %d" % safety_mode
  safety.set_alternative_experience(alternative_experience)

  if segment:
    init_segment(safety, lr, safety_mode)
    lr.reset()

  rx_tot, rx_invalid, tx_tot, tx_blocked, tx_controls, tx_controls_blocked = 0, 0, 0, 0, 0, 0
  blocked_addrs = set()
  invalid_addrs = set()
  start_t = None

  for msg in lr:
    if start_t is None:
      start_t = msg.logMonoTime
    safety.set_timer(((msg.logMonoTime // 1000)) % 0xFFFFFFFF)

    if msg.which() == 'sendcan':
     for canmsg in msg.sendcan:
        to_send = package_can_msg(canmsg)
        sent = safety.safety_tx_hook(to_send)
        if not sent:
          tx_blocked += 1
          tx_controls_blocked += safety.get_controls_allowed()
          blocked_addrs.add(canmsg.address)

          if "DEBUG" in os.environ:
            print("blocked bus %d msg %d at %f" % (canmsg.src, canmsg.address, (msg.logMonoTime - start_t) / (1e9)))
        tx_controls += safety.get_controls_allowed()
        tx_tot += 1
    elif msg.which() == 'can':
      for canmsg in msg.can:
        # ignore msgs we sent
        if canmsg.src >= 128:
          continue
        to_push = package_can_msg(canmsg)
        recv = safety.safety_rx_hook(to_push)
        if not recv:
          rx_invalid += 1
          invalid_addrs.add(canmsg.address)
        rx_tot += 1

  print("\nRX")
  print("total rx msgs:", rx_tot)
  print("invalid rx msgs:", rx_invalid)
  print("invalid addrs:", invalid_addrs)
  print("\nTX")
  print("total openpilot msgs:", tx_tot)
  print("total msgs with controls allowed:", tx_controls)
  print("blocked msgs:", tx_blocked)
  print("blocked with controls allowed:", tx_controls_blocked)
  print("blocked addrs:", blocked_addrs)

  return tx_controls_blocked == 0 and rx_invalid == 0

if __name__ == "__main__":
  from tools.lib.route import Route, SegmentName
  from tools.lib.logreader import MultiLogIterator  # pylint: disable=import-error

  parser = argparse.ArgumentParser(description="Replay CAN messages from a route or segment through a safety mode",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("route_or_segment_name", nargs='+')
  parser.add_argument("--mode", type=int, help="Override the safety mode from the log")
  parser.add_argument("--param", type=int, help="Override the safety param from the log")
  parser.add_argument("--alternative-experience", type=int, help="Override the alternative experience from the log")
  args = parser.parse_args()

  s = SegmentName(args.route_or_segment_name[0], allow_route_name=True)

  r = Route(s.route_name.canonical_name)
  logs = r.log_paths()[s.segment_num:s.segment_num+1] if s.segment_num >= 0 else r.log_paths()
  lr = MultiLogIterator(logs)

  if None in (args.mode, args.param):
    for msg in lr:
      if msg.which() == 'carParams':
        if args.mode is None:
          args.mode = msg.carParams.safetyConfigs[0].safetyModel.raw
        if args.param is None:
          args.param = msg.carParams.safetyConfigs[0].safetyParam
        if args.alternative_experience is None:
          args.alternative_experience = msg.carParams.alternativeExperience
        break
    else:
      raise Exception("carParams not found in log. Set safety mode and param manually.")

    lr.reset()

  print(f"replaying {args.route_or_segment_name[0]} with safety mode {args.mode}, param {args.param}, alternative experience {args.alternative_experience}")
  replay_drive(lr, args.mode, args.param, args.alternative_experience, segment=(s.segment_num >= 0))
