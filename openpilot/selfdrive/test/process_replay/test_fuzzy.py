import copy
import os
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized
from openpilot.common.fuzzy import capnp_random_dict, fuzzy_test

from openpilot.cereal import log
from opendbc.car.toyota.values import CAR as TOYOTA
import openpilot.selfdrive.test.process_replay.process_replay as pr

# These processes currently fail because of unrealistic data breaking assumptions
# that openpilot makes causing error with NaN, inf, int size, array indexing ...
# TODO: Make each one testable
NOT_TESTED = ['selfdrived', 'controlsd', 'card', 'plannerd', 'calibrationd', 'dmonitoringd', 'paramsd', 'dmonitoringmodeld', 'modeld']

TEST_CASES = [(cfg.proc_name, copy.deepcopy(cfg)) for cfg in pr.CONFIGS if cfg.proc_name not in NOT_TESTED]
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "10"))

class TestFuzzProcesses(OpenpilotTestCase):

  # TODO: make this faster and increase examples
  @parameterized.expand(TEST_CASES)
  @fuzzy_test(max_examples=MAX_EXAMPLES)
  def test_fuzz_process(self, proc_name, cfg, fuzzy):
    msgs = [capnp_random_dict(fuzzy, log.Event.schema, event, real_floats=True) for event in sorted(cfg.pubs)]
    for i, msg in enumerate(msgs):
      msg["logMonoTime"] = i * int(1e9)
    lr = [log.Event.new_message(**m).as_reader() for m in msgs]
    cfg.timeout = 5
    pr.replay_process(cfg, lr, fingerprint=TOYOTA.TOYOTA_COROLLA_TSS2, disable_progress=True, disable_migrations=True)
