import copy
import os
from openpilot.common.parameterized import parameterized

from cereal import log
from opendbc.car.toyota.values import CAR as TOYOTA
from openpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator
import openpilot.selfdrive.test.process_replay.process_replay as pr

# These processes currently fail because of unrealistic data breaking assumptions
# that openpilot makes causing error with NaN, inf, int size, array indexing ...
# TODO: Make each one testable
NOT_TESTED = ['selfdrived', 'controlsd', 'card', 'plannerd', 'calibrationd', 'dmonitoringd', 'paramsd', 'dmonitoringmodeld', 'modeld']

TEST_CASES = [(cfg.proc_name, copy.deepcopy(cfg)) for cfg in pr.CONFIGS if cfg.proc_name not in NOT_TESTED]
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "10"))


class TestFuzzProcesses:

  @parameterized.expand(TEST_CASES)
  def test_fuzz_process(self, proc_name, cfg):
    cfg.timeout = 5
    for _ in range(MAX_EXAMPLES):
      msgs = FuzzyGenerator.get_random_event_msg(events=cfg.pubs, real_floats=True)
      lr = [log.Event.new_message(**m).as_reader() for m in msgs]
      pr.replay_process(cfg, lr, fingerprint=TOYOTA.TOYOTA_COROLLA_TSS2, disable_progress=True)
