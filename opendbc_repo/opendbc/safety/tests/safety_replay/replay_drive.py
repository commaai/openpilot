#!/usr/bin/env python3
import argparse
from collections import Counter
from tqdm import tqdm

from opendbc.car.carlog import carlog
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.safety_replay.helpers import package_can_msg, init_segment


# replay a drive to check for safety violations
def replay_drive(msgs, safety_mode, param, alternative_experience):
  safety = libsafety_py.libsafety
  msgs.sort(key=lambda m: m.logMonoTime)

  err = safety.set_safety_hooks(safety_mode, param)
  assert err == 0, "invalid safety mode: %d" % safety_mode
  safety.set_alternative_experience(alternative_experience)

  init_segment(safety, msgs, safety_mode, param)

  rx_tot, rx_invalid, tx_tot, tx_blocked, tx_controls, tx_controls_blocked = 0, 0, 0, 0, 0, 0
  safety_tick_rx_invalid = False
  blocked_addrs = Counter()
  invalid_addrs = set()

  can_msgs = [m for m in msgs if m.which() in ('can', 'sendcan')]
  start_t = can_msgs[0].logMonoTime
  end_t = can_msgs[-1].logMonoTime
  for msg in tqdm(can_msgs):
    safety.set_timer((msg.logMonoTime // 1000) % 0xFFFFFFFF)

    # skip start and end of route, warm up/down period
    if msg.logMonoTime - start_t > 1e9 and end_t - msg.logMonoTime > 1e9:
      safety.safety_tick_current_safety_config()
      safety_tick_rx_invalid |= not safety.safety_config_valid() or safety_tick_rx_invalid

    if msg.which() == 'sendcan':
      for canmsg in msg.sendcan:
        _msg = package_can_msg(canmsg)
        sent = safety.safety_tx_hook(_msg)
        if not sent:
          tx_blocked += 1
          tx_controls_blocked += safety.get_controls_allowed()
          blocked_addrs[canmsg.address] += 1

          carlog.debug("blocked bus %d msg %d at %f" % (canmsg.src, canmsg.address, (msg.logMonoTime - start_t) / 1e9))
        tx_controls += safety.get_controls_allowed()
        tx_tot += 1
    elif msg.which() == 'can':
      # ignore msgs we sent
      for canmsg in filter(lambda m: m.src < 128, msg.can):
        safety.safety_fwd_hook(canmsg.src, canmsg.address)
        _msg = package_can_msg(canmsg)
        recv = safety.safety_rx_hook(_msg)
        if not recv:
          rx_invalid += 1
          invalid_addrs.add(canmsg.address)
        rx_tot += 1

  print("\nRX")
  print("total rx msgs:", rx_tot)
  print("invalid rx msgs:", rx_invalid)
  print("safety tick rx invalid:", safety_tick_rx_invalid)
  print("invalid addrs:", invalid_addrs)
  print("\nTX")
  print("total openpilot msgs:", tx_tot)
  print("total msgs with controls allowed:", tx_controls)
  print("blocked msgs:", tx_blocked)
  print("blocked with controls allowed:", tx_controls_blocked)
  print("blocked addrs:", blocked_addrs)

  return tx_controls_blocked == 0 and rx_invalid == 0 and not safety_tick_rx_invalid


if __name__ == "__main__":
  from openpilot.tools.lib.logreader import LogReader

  parser = argparse.ArgumentParser(description="Replay CAN messages from a route or segment through a safety mode",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("route_or_segment_name", nargs='+')
  parser.add_argument("--mode", type=int, help="Override the safety mode from the log")
  parser.add_argument("--param", type=int, help="Override the safety param from the log")
  parser.add_argument("--alternative-experience", type=int, help="Override the alternative experience from the log")
  args = parser.parse_args()

  lr = LogReader(args.route_or_segment_name[0])

  if None in (args.mode, args.param, args.alternative_experience):
    CP = lr.first('carParams')
    if args.mode is None:
      args.mode = CP.safetyConfigs[-1].safetyModel.raw
    if args.param is None:
      args.param = CP.safetyConfigs[-1].safetyParam
    if args.alternative_experience is None:
      args.alternative_experience = CP.alternativeExperience

  print(f"replaying {args.route_or_segment_name[0]} with safety mode {args.mode}, param {args.param}, alternative experience {args.alternative_experience}")
  replay_drive(list(lr), args.mode, args.param, args.alternative_experience)
