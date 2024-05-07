#!/usr/bin/env python3
import unittest

import cereal.messaging as messaging

from openpilot.selfdrive.test.process_replay import replay_process_with_name
from openpilot.selfdrive.car.toyota.values import CAR as TOYOTA


class TestLeads(unittest.TestCase):
  def test_radar_fault(self):
    # if there's no radar-related can traffic, radard should either not respond or respond with an error
    # this is tightly coupled with underlying car radar_interface implementation, but it's a good sanity check
    def single_iter_pkg():
      # single iter package, with meaningless cans and empty carState/modelV2
      msgs = []
      for _ in range(5):
        can = messaging.new_message("can", 1)
        cs = messaging.new_message("carState")
        msgs.append(can.as_reader())
        msgs.append(cs.as_reader())
      model = messaging.new_message("modelV2")
      msgs.append(model.as_reader())

      return msgs

    msgs = [m for _ in range(3) for m in single_iter_pkg()]
    out = replay_process_with_name("radard", msgs, fingerprint=TOYOTA.TOYOTA_COROLLA_TSS2)
    states = [m for m in out if m.which() == "radarState"]
    failures = [not state.valid and len(state.radarState.radarErrors) for state in states]

    self.assertTrue(len(states) == 0 or all(failures))


if __name__ == "__main__":
  unittest.main()
