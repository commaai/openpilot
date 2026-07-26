import copy
import numpy as np
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized
from openpilot.common.fuzzy import capnp_random_dict, fuzzy_test

from openpilot.cereal import log
from opendbc.car.toyota.values import CAR as TOYOTA
import openpilot.selfdrive.test.process_replay.process_replay as pr

# All processes are fuzzed. Vision processes (modeld, dmonitoringmodeld) need
# synthetic camera frames since the fuzzer only generates message payloads.
VISION_FRAME_SIZES = {
  "driverCameraState": (1344, 760),
  "roadCameraState": (1928, 1208),
  "wideRoadCameraState": (1928, 1208),
}


class FakeFrameReader:
  """Minimal stand-in for tools.lib.framereader.FrameReader.

  process_replay only needs ``pix_fmt``, ``w``, ``h`` and ``get(frameId)`` (which
  must return an NV12 byte buffer). The fuzzer produces arbitrary ``frameId``
  values, so we ignore the id and return a zeroed frame of the right size.
  """

  def __init__(self, w: int, h: int):
    self.w = w
    self.h = h
    self.pix_fmt = "nv12"

  def get(self, frame_id, pix_fmt=None):
    return np.zeros(self.h * self.w * 3 // 2, dtype=np.uint8)


TEST_CASES = [(cfg.proc_name, copy.deepcopy(cfg)) for cfg in pr.CONFIGS]


class TestFuzzProcesses(OpenpilotTestCase):

  # TODO: make this faster and increase examples
  @parameterized.expand(TEST_CASES)
  @fuzzy_test(max_examples=10)
  def test_fuzz_process(self, proc_name, cfg, fuzzy):
    msgs = [capnp_random_dict(fuzzy, log.Event.schema, event, real_floats=True) for event in sorted(cfg.pubs)]
    for i, msg in enumerate(msgs):
      msg["logMonoTime"] = (i + 1) * 1_000_000_000
    lr = [log.Event.new_message(**m).as_reader() for m in msgs]

    # Vision processes need frame data and more time to load/run the model
    frs = None
    if len(cfg.vision_pubs) != 0:
      frs = {cam: FakeFrameReader(*VISION_FRAME_SIZES[cam]) for cam in cfg.vision_pubs}
      cfg.timeout = 30
    else:
      cfg.timeout = 5

    pr.replay_process(cfg, lr, frs=frs, fingerprint=TOYOTA.TOYOTA_COROLLA_TSS2, disable_progress=True)
