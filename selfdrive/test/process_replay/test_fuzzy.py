#!/usr/bin/env python3
import copy
import unittest

from hypothesis import given, HealthCheck, Phase, settings
import hypothesis.strategies as st
from parameterized import parameterized

from cereal import log
from openpilot.selfdrive.car.toyota.values import CAR as TOYOTA
from openpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator
from openpilot.selfdrive.test.process_replay import replay_process, CONFIGS
from openpilot.selfdrive.test.process_replay.support_utils import skip_if_process_replay_unsupported

# These processes currently fail because of unrealistic data breaking assumptions
# that openpilot makes causing error with NaN, inf, int size, array indexing ...
# TODO: Make each one testable
NOT_TESTED = ['controlsd', 'plannerd', 'calibrationd', 'dmonitoringd', 'paramsd', 'laikad', 'dmonitoringmodeld', 'modeld']
TEST_CASES = [(cfg.proc_name, copy.deepcopy(cfg)) for cfg in CONFIGS if cfg.proc_name not in NOT_TESTED]

@skip_if_process_replay_unsupported
class TestFuzzProcesses(unittest.TestCase):

  @parameterized.expand(TEST_CASES)
  @given(st.data())
  @settings(phases=[Phase.generate, Phase.target], max_examples=50, deadline=1000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
  def test_fuzz_process(self, proc_name, cfg, data):
    msgs = FuzzyGenerator.get_random_event_msg(data.draw, events=cfg.pubs, real_floats=True)
    lr = [log.Event.new_message(**m).as_reader() for m in msgs]
    cfg.timeout = 5
    replay_process(cfg, lr, fingerprint=TOYOTA.COROLLA_TSS2, disable_progress=True)

if __name__ == "__main__":
  unittest.main()
