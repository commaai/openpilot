#!/usr/bin/env python3
import argparse
import importlib

from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.tools.lib.logreader import LogReader


def load_route(route_or_segment_name):
  print("Loading log...")
  lr = LogReader(route_or_segment_name)
  CP = lr.first("carParams")
  print(f"carFingerprint: '{CP.carFingerprint}'")
  mbytes = [m.as_builder().to_bytes() for m in lr if m.which() == 'can']
  return [m[1] for m in can_capnp_to_list(mbytes)]


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="On a DUT, have its UP replay a route's CAN messages in a loop until this `up run` or `up ssh` ends.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("route_or_segment_name", nargs='?', default="77611a1fac303767/2020-03-24--09-50-38/2:4", help="The route or segment name to replay.")
  args = parser.parse_args()

  can_msgs = load_route(args.route_or_segment_name)

  # ultimate_provisioning isn't a dependency, `up run` and `up ssh` put it on the DUT's PYTHONPATH
  up = importlib.import_module("ultimate_provisioning").UP.remote()
  up.reset()
  up.set_ignition(True)
  up.can_replay([[m for m in send if m[-1] <= 2] for send in can_msgs])
