#!/usr/bin/env python3
import argparse

from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.tools.lib.logreader import LogReader


def setup(j):
  for i in [0, 1, 2, 3, 0xFFFF]:
    j.can_clear(i)
    j.set_can_speed_kbps(i, 500)
  j.set_ignition(True)
  j.set_panda_power(True)


def load_route(route_or_segment_name):
  print("Loading log...")
  lr = LogReader(route_or_segment_name)
  CP = lr.first("carParams")
  print(f"carFingerprint: '{CP.carFingerprint}'")
  mbytes = [m.as_builder().to_bytes() for m in lr if m.which() == 'can']
  return [m[1] for m in can_capnp_to_list(mbytes)]


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="On a DUT, have its UP replay CAN messages from a route until the `up run` or `up ssh` that started this ends.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to replay. If not specified, a default public route will be used.")
  args = parser.parse_args()

  if args.route_or_segment_name is None:
    args.route_or_segment_name = "77611a1fac303767/2020-03-24--09-50-38/2:4"

  CAN_MSGS = load_route(args.route_or_segment_name)

  from ultimate_provisioning import UP
  up = UP.remote()
  up.set_harness_orientation(UP.HARNESS_ORIENTATION_1)
  setup(up)
  up.can_replay([[m for m in send if m[-1] <= 2] for send in CAN_MSGS])
