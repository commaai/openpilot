"""
``SEND_RAW_PRED`` branch: ``modelV2.rawPredictions`` bytes passthrough.

``fill_model_msg`` reads ``SEND_RAW_PRED`` at import time; tests patch the module flag.

LOW-LEVEL §7.1 Phase B.

Maps: R1.
"""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.modeld import fill_model_msg as fill_model_msg_mod
from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg
from openpilot.selfdrive.modeld.tests.modeld_test_fixtures import DummyBuilder, minimal_net_output_data


def test_raw_predictions_copied_when_flag_enabled(monkeypatch):
  monkeypatch.setattr(fill_model_msg_mod, "SEND_RAW_PRED", True, raising=False)

  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  raw = np.arange(64, dtype=np.float32).reshape(4, 16)
  outs["raw_pred"] = raw

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=5.0,
    delay=0.0,
    publish_state=PublishState(),
    vipc_frame_id=1,
    vipc_frame_id_extra=1,
    frame_id=1,
    frame_drop=0.0,
    timestamp_eof=1,
    model_execution_time=0.01,
    valid=True,
  )

  assert ext.modelV2.rawPredictions == raw.tobytes()
